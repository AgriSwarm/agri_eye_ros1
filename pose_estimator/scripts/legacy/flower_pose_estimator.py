#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from agri_eye_msgs.msg import EstimatedPose2DArray, EstimatedPose2D
import random
import string
import sys
import shutil
from pathlib import Path
import os
import json
import subprocess
import cv2
import requests
from PIL import Image as PILImage
import io

class FlowerPoseEstimator:
    def __init__(self):
        self.node_name = "flower_pose_estimator"

        container_name = "mito_detector"  
        self.container_id = self.find_docker_container(container_name)
        if not self.container_id:
            print("Docker container not found.")
            sys.exit(1)

        rospy.init_node(self.node_name)

        self.bridge = CvBridge()
        self.pose_publisher = rospy.Publisher("/flower_poses", EstimatedPose2DArray, queue_size=1)
        self.boxed_image_publisher = rospy.Publisher("/flower_boxed_image", Image, queue_size=1)

        self.image_subscriber = rospy.Subscriber("/camera/image_raw", Image, self.callback)

    def callback(self, data):
        # 画像読み込む
        # image_path = "/home/torobo/Downloads/IMG_6698.jpg"
        # cv_image = cv2.imread(image_path)
        start_time = rospy.Time.now()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(cv_image)
        binary_stream = io.BytesIO()
        pil_image.save(binary_stream, format='JPEG') 
        image = binary_stream.getvalue()
        results = self.inference(image)
        results = json.loads(results)
        
        if len(results) > 0:
            self.addbbox(cv_image, results)
            self.publish_poses(results)
        
        self.publish_image(cv_image)
        # print(f"processing time: {(rospy.Time.now() - start_time).to_sec()} sec")

    def addbbox(self, image, results):
        for res in results:
            x_min = int(res["bbox_xyxy"][0])
            y_min = int(res["bbox_xyxy"][1])
            x_max = int(res["bbox_xyxy"][2])
            y_max = int(res["bbox_xyxy"][3])
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 15)
            cv2.putText(image, f"{res['conf']:.2f}", (int(x_center), int(y_center)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return image

    def inference(self, image):
        url = "http://localhost:3100/predict/"
        response = requests.post(url, files={"file": ("tmp.jpg", image, "image/jpeg")})
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to get a response, status code: {response.status_code}")
            return None

    def find_docker_container(self, container_name, call_count=0):
        try:
            output = subprocess.check_output(["docker", "ps", "-q", "-f", f"name={container_name}"])
            results = output.decode('utf-8').strip()
            if results:
                return results
        except subprocess.CalledProcessError:
            pass

        print("Cannot find Docker container. Starting a new one.")
        subprocess.check_call("sh run_docker.sh".split())
        if call_count == 0:
            return self.find_docker_container(container_name, call_count=1)
        else:
            return None
        
    # def run_inference_in_docker(self, container_id, image, result_path):
    #     try:
    #         subprocess.check_call(["docker", "exec", container_id, "sh", "workspace/inference.sh", image, result_path])
    #     except subprocess.CalledProcessError:
    #         print("Error running inference script in Docker")

    # def read_results(self, file_path):
    #     if not os.path.exists(file_path):
    #         print(f"Results file not found: {file_path}")
    #         return None

    #     with open(file_path, 'r') as file:
    #         data = json.load(file)
    #     os.remove(file_path)
    #     return data

    def publish_image(self, image):
        try:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.boxed_image_publisher.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
        except CvBridgeError as e:
            print(e)

    def publish_poses(self, results):
        pose_array = EstimatedPose2DArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = "world"  # update as per your TF frames

        for res in results:  # Assuming 'poses' is the key containing the results
            pose = EstimatedPose2D()
            x_min = res["bbox_xyxy"][0]
            y_min = res["bbox_xyxy"][1]
            x_max = res["bbox_xyxy"][2]
            y_max = res["bbox_xyxy"][3]
            pose.x = (x_min + x_max) / 2
            pose.y = (y_min + y_max) / 2
            pose.normal.x = 0
            pose.normal.y = 0
            pose.normal.z = -1
            pose.probability = res["conf"]

            pose_array.poses.append(pose)

        self.pose_publisher.publish(pose_array)

if __name__ == '__main__':
    try:
        flower_pose_estimator = FlowerPoseEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass