#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

def main():
    # ROSノードの初期化
    rospy.init_node('image_talker', anonymous=True)

    # パブリッシャの定義
    pub = rospy.Publisher('/camera/color/image_raw', Image, queue_size=10)

    # cv_bridgeの初期化
    bridge = CvBridge()

    # PNG画像の読み込み (画像のパスを適宜変更してください)
    image_path = '/home/tomoking/catkin_ws/src/agri_eye_ros1/data/sample_2.png'
    image_cv = cv2.imread(image_path)

    # 画像が正しく読み込めたか確認
    if image_cv is None:
        rospy.logerr("画像ファイルを読み込めませんでした: %s", image_path)
        return

    # パブリッシュレートの設定 (例: 10Hz)
    rate = rospy.Rate(10)

    # メインループ: 画像を継続的にパブリッシュ
    while not rospy.is_shutdown():
        # OpenCVの画像をROSのImageメッセージに変換
        try:
            image_msg = bridge.cv2_to_imgmsg(image_cv, encoding='bgr8')
        except Exception as e:
            rospy.logerr("画像の変換に失敗しました: %s", str(e))
            continue

        # メッセージをパブリッシュ
        pub.publish(image_msg)

        # 指定したレートでスリープ
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
