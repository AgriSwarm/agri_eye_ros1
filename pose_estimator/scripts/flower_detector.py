#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
import torch
import random
import os

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3
from agri_eye_msgs.msg import EstimatedPose2DArray, EstimatedPose2D

from ultralytics import YOLO
from torchvision import transforms
# from module import SixDRepNetModule
from scripts.module import SixDRepNetModule
from scripts.module_trt import TensorRTInference
import scripts.utils as utils
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation

class FlowerPoseEstimator:
    def __init__(self):
        rospy.init_node('flower_detector', anonymous=True)

        self.last_inference_time = rospy.Time.now()

        # パラメータ取得
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, '..', '..', 'models')
        
        self.drone_id = rospy.get_param('~drone_id', 1)
        self.conf_threshold = rospy.get_param('~conf_threshold', 0.3)
        self.estimate_pos = rospy.get_param('~estimate_pos', True)
        self.estimate_att = rospy.get_param('~estimate_att', True)
        self.pub_image = rospy.get_param('~pub_image', True)
        self.inference_freq = rospy.get_param('~inference_freq', 10)
        
        self.verbose = rospy.get_param('~verbose', False)
        self.estimate_whole_att = rospy.get_param('~estimate_whole_att', False)
        self.use_tensorrt = rospy.get_param('~use_tensorrt', False)
        self.use_fp16 = rospy.get_param('~use_fp16', False)
        self.use_fp16_dla = rospy.get_param('~use_fp16_dla', False)
        self.use_int8 = rospy.get_param('~use_int8', False)
        if self.use_tensorrt:
            if self.use_fp16:
                print("Using FP16 TensorRT engine.")
                default_yolo = os.path.join(models_dir, 'YOLOv8_fp16.engine')
                default_sixd = os.path.join(models_dir, 'sixdrepnet_fp16.engine')
            elif self.use_fp16_dla:
                print("Using FP16 DLA TensorRT engine.")
                default_yolo = os.path.join(models_dir, 'YOLOv8_fp16_dla.engine')
                # default_sixd = os.path.join(models_dir, 'sixdrepnet_fp16_dla.engine')  
                default_sixd = os.path.join(models_dir, 'sixdrepnet_fp16.engine')                                              
            elif self.use_int8:
                print("Using INT8 TensorRT engine.")
                default_yolo = os.path.join(models_dir, 'YOLOv8_int8.engine')
                default_sixd = os.path.join(models_dir, 'sixdrepnet_int8.engine')
            else:
                print("Using FP32 TensorRT engine.")
                default_yolo = os.path.join(models_dir, 'YOLOv8_fp32.engine')
                default_sixd = os.path.join(models_dir, 'sixdrepnet_fp32.engine')
        else:
            default_yolo = os.path.join(models_dir, 'YOLOv8.pt')
            default_sixd = os.path.join(models_dir, 'sixdrepnet.ckpt')

        self.yolo_checkpoint = rospy.get_param('~yolo_checkpoint', default_yolo)
        self.sixd_checkpoint = rospy.get_param('~sixd_checkpoint', default_sixd)

        # YOLOロード
        rospy.loginfo("Loading YOLO model: %s", self.yolo_checkpoint)
        self.yolo_model = YOLO(self.yolo_checkpoint)
        # self.yolo_model.export(format="engine")
        # self.yolo_model.export(format="engine", device="dla:0", half=True)

        # SixDRepNetロード
        if self.use_tensorrt:
            rospy.loginfo("Loading SixDRepNet TensorRT engine: %s", self.sixd_checkpoint)
            self.trt_inference = TensorRTInference(self.sixd_checkpoint)
        else:
            rospy.loginfo("Loading SixDRepNet: %s", self.sixd_checkpoint)
            self.sixd_model = SixDRepNetModule.load_from_checkpoint(self.sixd_checkpoint)
            self.sixd_model.eval()

        # GPU対応
        if self.use_tensorrt:
            # self.sixd_model.cuda()
            rospy.loginfo("Using TensorRT for inference.")
        elif torch.cuda.is_available():
            self.yolo_model.to('cuda')
            self.sixd_model.cuda()
            rospy.loginfo("Using CUDA for inference.")
        else:
            rospy.logwarn("CUDA not available. Using CPU.")

        # Publisher
        self.pose_pub = rospy.Publisher('/flower_poses', EstimatedPose2DArray, queue_size=10)
        self.image_pub = rospy.Publisher('/annotated_image', Image, queue_size=10)
        # Subscriber
        self.image_sub = rospy.Subscriber('camera/color/image_raw', Image, self.callback)

        self.bridge = CvBridge()

        # コード3の描画用色
        self.rect_color = (0, 204, 255)    # オレンジ色
        self.arrow_color = (255, 50, 50)   # 明るい青
        self.pos_text_color = (0, 255, 255)
        self.ori_text_color = (255, 150, 150)

        # SixDRepNet 入力前処理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        rospy.loginfo("FlowerPoseEstimator initialized.")

    def callback(self, image_msg):
        current_time = rospy.Time.now()
        elapsed_time = (current_time - self.last_inference_time).to_sec()
        if elapsed_time < 1.0 / self.inference_freq:
            return
        self.last_inference_time = current_time

        tick_time = rospy.Time.now()
        start_time = rospy.Time.now()
        # 画像をOpenCVに変換
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        # 180度回転
        cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)

        # YOLOでBBox検出
        if self.estimate_pos:
            results = self.yolo_model.predict(cv_image, conf=self.conf_threshold, verbose=False)
        else:
            results = []

        if len(results) == 0:
            rospy.logwarn("No YOLO inference results.")
            return

        if self.verbose:
            print(f"YOLO Inference Time(ms): {(rospy.Time.now() - tick_time).to_sec() * 1000:.2f}")

        result = results[0]
        boxes = result.boxes

        if len(boxes) == 0:
            rospy.logwarn("No boxes detected.")
            if self.pub_image:
                rotate_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
                rotate_msg.header = image_msg.header
                self.image_pub.publish(rotate_msg)
            return

        height, width = cv_image.shape[:2]

        # 最大信頼度のボックスのインデックスを特定
        max_conf_index = None
        if not self.estimate_whole_att and self.estimate_att:
            max_conf = -1.0
            for idx, box in enumerate(boxes):
                conf_val = float(box.conf[0].cpu().numpy())
                if conf_val > max_conf:
                    max_conf = conf_val
                    max_conf_index = idx

        pose_array = EstimatedPose2DArray()
        pose_array_rotated = EstimatedPose2DArray()
        pose_array.header = Header()
        pose_array.header.stamp = image_msg.header.stamp
        pose_array.header.frame_id = "camera_frame"
        
        for idx, box in enumerate(boxes):
            tick_time = rospy.Time.now()
            xyxy = box.xyxy[0].cpu().numpy()  # (x1, y1, x2, y2)
            conf_ = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = map(int, xyxy)

            # 切り出し
            x1c = max(0, min(width-1, x1))
            y1c = max(0, min(height-1, y1))
            x2c = max(0, min(width-1, x2))
            y2c = max(0, min(height-1, y2))
            cropped = cv_image[y1c:y2c, x1c:x2c]
            if cropped.size == 0:
                continue

            # pose_msg の生成
            pose_msg = EstimatedPose2D()
            pose_msg.drone_id = self.drone_id
            pose_msg.x_1 = width - x1c
            pose_msg.y_1 = height - y1c
            pose_msg.x_2 = width - x2c
            pose_msg.y_2 = height - y2c
            pose_msg.pos_prob = conf_

            pose_msg_rotated = EstimatedPose2D()
            pose_msg_rotated.x_1 = x1c
            pose_msg_rotated.y_1 = y1c
            pose_msg_rotated.x_2 = x2c
            pose_msg_rotated.y_2 = y2c
            pose_msg_rotated.pos_prob = conf_

            # 最大信頼度のボックスに対してのみ姿勢推定を実行
            if self.estimate_whole_att or (not self.estimate_whole_att and idx == max_conf_index):
                z_axis_3d, euler = self.get_attitude(cropped)
                # print(f"Z-axis: {z_axis_3d}, Euler: {euler}")
                pose_msg.normal = Vector3(z_axis_3d[0], z_axis_3d[1], z_axis_3d[2])
                pose_msg.euler = Vector3(euler[0], euler[1], euler[2])
                pose_msg.ori_prob = 1.0
                pose_msg_rotated.normal = Vector3(z_axis_3d[0], z_axis_3d[1], z_axis_3d[2])
                pose_msg_rotated.euler = Vector3(euler[0], euler[1], euler[2])
                pose_msg_rotated.ori_prob = 1.0
            else:
                # pose_msg.normal = Vector3(0.0, 0.0, 0.0)
                # pose_msg.euler = Vector3(0.0, 0.0, 0.0)
                pose_msg.normal = Vector3(0.0, 0.0, 1.0)
                pose_msg.euler = Vector3(3.141592653589793, 0.0, 0.0)
                pose_msg.ori_prob = 0.0
                pose_msg_rotated.normal = Vector3(0.0, 0.0, 0.0)
                pose_msg_rotated.euler = Vector3(0.0, 0.0, 0.0)
                pose_msg_rotated.ori_prob = 0.0

            pose_array.poses.append(pose_msg)
            pose_array_rotated.poses.append(pose_msg_rotated)

            if self.verbose:
                print(f"Pose Estimation Time(ms): {(rospy.Time.now() - tick_time).to_sec() * 1000:.2f}")

        self.pose_pub.publish(pose_array)

        # 描画処理
        if self.pub_image:
            for pose in pose_array_rotated.poses:
                self.draw_pose(cv_image, pose, height, width)
            annotated_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            annotated_msg.header = image_msg.header
            self.image_pub.publish(annotated_msg)

        if self.verbose:
            print(f"Total Time(ms): {(rospy.Time.now() - start_time).to_sec() * 1000:.2f}")
            print(f"Image Delay(ms): {(rospy.Time.now() - image_msg.header.stamp).to_sec() * 1000:.2f}")

    def get_attitude(self, bgr_image):
        """
        SixDRepNetモデルで推定した回転行列 R (3x3) の第3列 (z軸) を取り出して返す。
        """
        pil_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(pil_img)

        input_tensor = self.transform(pil_img).unsqueeze(0)
        
        if self.use_tensorrt:
            input_data = input_tensor.cpu().numpy().astype(np.float32).ravel()
            outputs = self.trt_inference.infer(input_data)
            R_flat = outputs[0]
            R_np = np.array(R_flat).reshape(3, 3)
        else:
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            with torch.no_grad():
                R = self.sixd_model(input_tensor)  # shape: [B, 3, 3]
            R_np = R[0].cpu().numpy()  # (3,3)

        R_np = np.identity(3)
        R_np = np.dot(R_np, utils.get_R(3.141592653589793, 0, 0))

        r = Rotation.from_matrix(R_np)
        euler = r.as_euler("xyz", degrees=True)
        z_axis = r.apply([0, 0, 1])

        return z_axis, euler

    def draw_pose(self, cv_image, pose, height, width):
        """
        コード3と同じ処理 + z軸ベクトルの「2D射影」をここで行う。
        """

        # BBoxを描画
        cv2.rectangle(
            cv_image,
            (pose.x_2, pose.y_2),
            (pose.x_1, pose.y_1),
            self.rect_color, 3
        )

        # (2) center (やはり180度回転後の座標系で)
        center_x = int((pose.x_1 + pose.x_2) / 2)
        center_y = int((pose.y_1 + pose.y_2) / 2)

        # (3) 矢印の向き: 3Dベクトルを2D平面に射影
        normal_3d = np.array([pose.normal.x, pose.normal.y, pose.normal.z], dtype=float)
        nx, ny = normal_3d[0], normal_3d[1]

        arrow_length = 60
        end_x = int(center_x - arrow_length * nx)
        end_y = int(center_y + arrow_length * ny)
        cv2.arrowedLine(cv_image, (center_x, center_y), (end_x, end_y),
                        self.arrow_color, 3, tipLength=0.3)
        
        # print(f"Euler: {pose.euler.x:.2f}, {pose.euler.y:.2f}, {pose.euler.z:.2f}")
        if pose.ori_prob != 0.0:
            cv_image = utils.draw_axis(cv_image, pose.euler.x, pose.euler.y, pose.euler.z, center_x, center_y, 30)

        # (4) Probabilityテキスト
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        pos_text = f'Pos: {pose.pos_prob:.2f}'
        cv2.putText(cv_image, pos_text, (pose.x_1, pose.y_1 - 15),
                    font, font_scale, (0,0,0), thickness+2)
        cv2.putText(cv_image, pos_text, (pose.x_1, pose.y_1 - 15),
                    font, font_scale, self.pos_text_color, thickness)

        ori_text = f'Ori: {pose.ori_prob:.2f}'
        cv2.putText(cv_image, ori_text, (pose.x_1, pose.y_1 - 40),
                    font, font_scale, (0,0,0), thickness+2)
        cv2.putText(cv_image, ori_text, (pose.x_1, pose.y_1 - 40),
                    font, font_scale, self.ori_text_color, thickness)

def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)
    np.random.seed(42)

    try:
        node = FlowerPoseEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()