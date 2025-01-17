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
import scripts.utils as utils
from PIL import Image as PILImage

class FlowerPoseEstimator:
    def __init__(self):
        rospy.init_node('flower_detector', anonymous=True)

        # パラメータ取得
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, '..', '..', 'models')
        default_yolo = os.path.join(models_dir, 'YOLOv8.pt')
        default_sixd = os.path.join(models_dir, 'HPE.ckpt')
        self.drone_id = rospy.get_param('~drone_id', 1)
        self.conf_threshold = rospy.get_param('~conf_threshold', 0.3)
        self.yolo_checkpoint = rospy.get_param('~yolo_checkpoint', default_yolo)
        self.sixd_checkpoint = rospy.get_param('~sixd_checkpoint', default_sixd)
        self.verbose = rospy.get_param('~verbose', False)
        self.estimate_whole_pose = rospy.get_param('~estimate_whole_pose', False)


        # YOLOロード
        rospy.loginfo("Loading YOLO model: %s", self.yolo_checkpoint)
        self.yolo_model = YOLO(self.yolo_checkpoint)

        # SixDRepNetロード
        rospy.loginfo("Loading SixDRepNet: %s", self.sixd_checkpoint)
        self.sixd_model = SixDRepNetModule.load_from_checkpoint(self.sixd_checkpoint)
        self.sixd_model.eval()

        # GPU対応
        if torch.cuda.is_available():
            self.yolo_model.to('cuda')
            self.sixd_model.cuda()
            rospy.loginfo("Using CUDA for inference.")
        else:
            rospy.logwarn("CUDA not available. Using CPU.")

        # Publisher
        self.pose_pub = rospy.Publisher('~flower_poses', EstimatedPose2DArray, queue_size=10)
        self.image_pub = rospy.Publisher('~annotated_image', Image, queue_size=10)
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
        start_time = rospy.Time.now()
        # 画像をOpenCVに変換
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        # 180度回転
        cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)

        # YOLOでBBox検出
        results = self.yolo_model.predict(cv_image, conf=self.conf_threshold, verbose=False)
        if len(results) == 0:
            rospy.logwarn("No YOLO inference results.")
            return

        if self.verbose:
            print(f"YOLO Inference Time(ms): {(rospy.Time.now() - start_time).to_sec() * 1000:.2f}")

        result = results[0]
        boxes = result.boxes

        if len(boxes) == 0:
            rospy.logwarn("No boxes detected.")
            return

        height, width = cv_image.shape[:2]

        # 最大信頼度のボックスのインデックスを特定
        max_conf_index = None
        if not self.estimate_whole_pose:
            max_conf = -1.0
            for idx, box in enumerate(boxes):
                conf_val = float(box.conf[0].cpu().numpy())
                if conf_val > max_conf:
                    max_conf = conf_val
                    max_conf_index = idx

        pose_array = EstimatedPose2DArray()
        pose_array.header = Header()
        pose_array.header.stamp = image_msg.header.stamp
        pose_array.header.frame_id = "camera_frame"

        for idx, box in enumerate(boxes):
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

            # 最大信頼度のボックスに対してのみ姿勢推定を実行
            if self.estimate_whole_pose or (not self.estimate_whole_pose and idx == max_conf_index):
                z_axis_3d, euler = self.get_attitude(cropped)
                pose_msg.normal = Vector3(z_axis_3d[0], z_axis_3d[1], z_axis_3d[2])
                pose_msg.euler = Vector3(euler[0], euler[1], euler[2])
                pose_msg.ori_prob = 1.0
            else:
                pose_msg.normal = Vector3(0.0, 0.0, 0.0)
                pose_msg.euler = Vector3(0.0, 0.0, 0.0)
                pose_msg.ori_prob = 0.0

            pose_array.poses.append(pose_msg)

        if self.verbose:
            print(f"Pose Estimation Time(ms): {(rospy.Time.now() - start_time).to_sec() * 1000:.2f}")

        # 描画処理
        for pose in pose_array.poses:
            self.draw_pose(cv_image, pose, height, width)

        # Publish
        self.pose_pub.publish(pose_array)
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
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        with torch.no_grad():
            R = self.sixd_model(input_tensor)  # shape: [B, 3, 3]
        R_np = R[0].cpu().numpy()  # (3,3)

        # R_np = np.identity(3)
        # R_np = np.dot(R_np, utils.get_R(0.0, 0.3, 0.0))

        z_axis = R_np[:, 2]  # (x, y, z)
        euler = cv2.Rodrigues(R_np)[0].flatten()  # (roll, pitch, yaw)
        return z_axis, euler

    def draw_pose(self, cv_image, pose, height, width):
        """
        コード3と同じ処理 + z軸ベクトルの「2D射影」をここで行う。
        """
        # (1) rectangle: 180度回転を加味して反転
        rotated_x1 = width - pose.x_1
        rotated_y1 = height - pose.y_1
        rotated_x2 = width - pose.x_2
        rotated_y2 = height - pose.y_2

        # BBoxを描画
        cv2.rectangle(
            cv_image,
            (rotated_x2, rotated_y2),
            (rotated_x1, rotated_y1),
            self.rect_color, 3
        )

        # (2) center (やはり180度回転後の座標系で)
        center_x = int((rotated_x1 + rotated_x2) / 2)
        center_y = int((rotated_y1 + rotated_y2) / 2)

        # (3) 矢印の向き: 3Dベクトルを2D平面に射影
        normal_3d = np.array([pose.normal.x, pose.normal.y, pose.normal.z], dtype=float)
        nx, ny = normal_3d[0], normal_3d[1]

        # 正規化して矢印の長さを一定に
        norm_xy = math.sqrt(nx*nx + ny*ny)
        if norm_xy > 1e-6:
            nx /= norm_xy
            ny /= norm_xy
        else:
            nx, ny = 0.0, 0.0

        # arrow_length = 30
        # end_x = int(center_x - arrow_length * nx)
        # end_y = int(center_y - arrow_length * ny)
        # cv2.arrowedLine(cv_image, (center_x, center_y), (end_x, end_y),
        #                 self.arrow_color, 3, tipLength=0.3)
        
        # print(f"Euler: {pose.euler.x:.2f}, {pose.euler.y:.2f}, {pose.euler.z:.2f}")
        if pose.ori_prob != 0.0:
            cv_image = utils.draw_axis(cv_image, pose.euler.x, pose.euler.y, pose.euler.z, center_x, center_y)

        # (4) Probabilityテキスト
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        pos_text = f'Pos: {pose.pos_prob:.2f}'
        cv2.putText(cv_image, pos_text, (rotated_x1, rotated_y1 - 15),
                    font, font_scale, (0,0,0), thickness+2)
        cv2.putText(cv_image, pos_text, (rotated_x1, rotated_y1 - 15),
                    font, font_scale, self.pos_text_color, thickness)

        ori_text = f'Ori: {pose.ori_prob:.2f}'
        cv2.putText(cv_image, ori_text, (rotated_x1, rotated_y1 - 40),
                    font, font_scale, (0,0,0), thickness+2)
        cv2.putText(cv_image, ori_text, (rotated_x1, rotated_y1 - 40),
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
