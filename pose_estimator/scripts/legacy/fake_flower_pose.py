#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from agri_eye_msgs.msg import EstimatedPose2DArray, EstimatedPose2D
from geometry_msgs.msg import Vector3
from std_msgs.msg import Header

class FakeFlowerPose:
    def __init__(self):
        rospy.init_node('fake_flower_pose', anonymous=True)
        self.drone_id = rospy.get_param('~drone_id', 1)
        self.pose_pub = rospy.Publisher('~flower_poses', EstimatedPose2DArray, queue_size=10)
        self.image_pub = rospy.Publisher('~annotated_image', Image, queue_size=10)
        self.image_sub = rospy.Subscriber('camera/color/image_raw', Image, self.callback)
        self.bridge = CvBridge()

        # Define colors (BGR format)
        self.rect_color = (0, 204, 255)  # オレンジ色
        self.arrow_color = (255, 50, 50)  # 明るい青
        self.pos_text_color = (0, 255, 255)  # 黄色
        self.ori_text_color = (255, 150, 150)  # 薄い青

    def draw_pose(self, cv_image, pose, height, width):
        """
        Draw single pose annotation on the image
        """
        # Adjust coordinates for rotated image
        rotated_x1 = width - pose.x_1
        rotated_y1 = height - pose.y_1
        rotated_x2 = width - pose.x_2
        rotated_y2 = height - pose.y_2

        # Draw rectangle with thicker line
        cv2.rectangle(cv_image, (rotated_x2, rotated_y2), (rotated_x1, rotated_y1), 
                     self.rect_color, 3)
        
        # Calculate center point
        center_x = int((rotated_x1 + rotated_x2) / 2)
        center_y = int((rotated_y1 + rotated_y2) / 2)
        
        # Draw arrow with thicker line
        arrow_length = 30
        end_x = int(center_x - arrow_length * pose.normal.x)
        end_y = int(center_y - arrow_length * pose.normal.y)
        cv2.arrowedLine(cv_image, (center_x, center_y), (end_x, end_y), 
                       self.arrow_color, 3, tipLength=0.3)

        # Add probability text with thicker font
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Position probability text
        pos_text = f'Pos: {pose.pos_prob:.2f}'
        cv2.putText(cv_image, pos_text, (rotated_x1, rotated_y1-15), 
                    font, font_scale, (0, 0, 0), thickness + 2)  # outline
        cv2.putText(cv_image, pos_text, (rotated_x1, rotated_y1-15), 
                    font, font_scale, self.pos_text_color, thickness)

        # Orientation probability text
        ori_text = f'Ori: {pose.ori_prob:.2f}'
        cv2.putText(cv_image, ori_text, (rotated_x1, rotated_y1-40), 
                    font, font_scale, (0, 0, 0), thickness + 2)  # outline
        cv2.putText(cv_image, ori_text, (rotated_x1, rotated_y1-40), 
                    font, font_scale, self.ori_text_color, thickness)

    def create_pose(self, x1, y1, x2, y2, normal_x=1.0, normal_y=0.0):
        """
        Create a pose object with given parameters
        """
        pose = EstimatedPose2D()
        pose.drone_id = self.drone_id
        pose.x_1 = x1
        pose.y_1 = y1
        pose.x_2 = x2
        pose.y_2 = y2
        pose.normal = Vector3(normal_x, normal_y, 0.0)
        pose.pos_prob = 0.95
        pose.ori_prob = 0.95
        return pose

    def callback(self, image_msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        # Rotate image 180 degrees
        cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)

        pose_array = EstimatedPose2DArray()
        pose_array.header = Header()
        pose_array.header.stamp = image_msg.header.stamp
        pose_array.header.frame_id = "camera_frame"

        # Create multiple poses
        pose1 = self.create_pose(120, 120, 200, 200)
        pose2 = self.create_pose(300, 300, 380, 380, 0.0, 1.0)
        pose3 = self.create_pose(150, 100, 230, 180, -1.0, 0.0)
        pose4 = self.create_pose(100, 300, 180, 380, 0.0, -1.0)
        pose5 = self.create_pose(200, 120, 280, 200, 1.0, 0.0)
        pose6 = self.create_pose(20, 300, 100, 380, 0.0, 1.0)

        height = cv_image.shape[0]
        width = cv_image.shape[1]

        # Draw all poses
        for pose in [pose1, pose2, pose3, pose4, pose5, pose6]:
            self.draw_pose(cv_image, pose, height, width)

        pose_array.poses = [pose1, pose2, pose3, pose4, pose5, pose6]
        self.pose_pub.publish(pose_array)

        # Publish annotated image
        annotated_image_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        annotated_image_msg.header = image_msg.header
        self.image_pub.publish(annotated_image_msg)

def main():
    try:
        fake_flower_pose = FakeFlowerPose()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()