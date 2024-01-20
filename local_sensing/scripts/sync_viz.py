#!/usr/bin/env python
import rospy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sensor_msgs.msg import PointCloud, Imu
from collections import deque
from threading import Lock

# データアクセス用ロック
data_lock = Lock()

# データ記録用キュー
feature_velocities_x = deque(maxlen=50) # 5秒間分のデータ
feature_velocities_y = deque(maxlen=50)
imu_angular_velocities_x = deque(maxlen=500)
imu_angular_velocities_y = deque(maxlen=500)
imu_angular_velocities_z = deque(maxlen=500)
imu_linear_accelerations_x = deque(maxlen=500)
imu_linear_accelerations_y = deque(maxlen=500)
imu_linear_accelerations_z = deque(maxlen=500)

feature_timestamps = deque(maxlen=50)
imu_timestamps = deque(maxlen=500)

time_offset = 0.05

# コールバック関数
def feature_callback(msg):
    with data_lock:
        feature_timestamps.append(msg.header.stamp.to_sec())
        velocities_x = [msg.channels[3].values[i] for i in range(len(msg.points))]
        velocities_y = [msg.channels[4].values[i] for i in range(len(msg.points))]
        
        feature_velocities_x.append(sum(velocities_x) / len(velocities_x))
        feature_velocities_y.append(sum(velocities_y) / len(velocities_y))

def imu_callback(msg):
    with data_lock:
        imu_timestamps.append(msg.header.stamp.to_sec()+time_offset)
        imu_angular_velocities_x.append(msg.angular_velocity.x)
        imu_angular_velocities_y.append(-msg.angular_velocity.y)
        imu_angular_velocities_z.append(msg.angular_velocity.z)
        imu_linear_accelerations_x.append(msg.linear_acceleration.x)
        imu_linear_accelerations_y.append(msg.linear_acceleration.y)
        imu_linear_accelerations_z.append(msg.linear_acceleration.z)

# プロット更新関数
def update_plot(
        num, 
        feature_timestamps, 
        feature_velocities_x, 
        feature_velocities_y, 
        imu_timestamps, 
        imu_angular_velocities_x,
        imu_angular_velocities_y,
        imu_angular_velocities_z,
        imu_linear_accelerations_x,
        imu_linear_accelerations_y,
        imu_linear_accelerations_z
    ):
    with data_lock:
        ax1.clear()
        ax2.clear()

        # # IMUデータをプロット
        # if imu_timestamps:
        #     ax1.plot(imu_timestamps, imu_angular_velocities_x, label='Angular Velocity X')
        #     ax1.plot(imu_timestamps, imu_angular_velocities_y, label='Angular Velocity Y')
        #     ax1.plot(imu_timestamps, imu_angular_velocities_z, label='Angular Velocity Z')
        #     ax1.plot(imu_timestamps, imu_linear_accelerations_x, label='Linear Acceleration X')
        #     ax1.plot(imu_timestamps, imu_linear_accelerations_y, label='Linear Acceleration Y')
        #     ax1.plot(imu_timestamps, imu_linear_accelerations_z, label='Linear Acceleration Z')
        #     ax1.legend(loc='upper left')
        #     ax1.set_title('IMU Data')

        # # Feature Trackerデータをプロット
        # if feature_timestamps:
        #     ax2.plot(feature_timestamps, feature_velocities_x, label='Feature Velocity X')
        #     ax2.plot(feature_timestamps, feature_velocities_y, label='Feature Velocity Y')
        #     ax2.legend(loc='upper left')
        #     ax2.set_title('Feature Tracker Velocity Data')

        if imu_timestamps and feature_timestamps:
            ax1.plot(imu_timestamps, imu_angular_velocities_z, label='Angular Velocity Z')
            ax1.plot(feature_timestamps, feature_velocities_x, label='Feature Velocity X')
            ax1.legend(loc='upper left')
            ax1.set_title('X Axis Data')

            ax2.plot(imu_timestamps, imu_angular_velocities_y, label='Angular Velocity Y')
            ax2.plot(feature_timestamps, feature_velocities_y, label='Feature Velocity Y')
            ax2.legend(loc='upper left')
            ax2.set_title('Y Axis Data')

        # 再度レジェンドを描く
        ax1.legend()
        ax2.legend()

        # X軸（時間）のラベル設定
        ax1.set_xlabel('Time (sec)')
        ax1.set_ylabel('IMU Values')
        ax1.set_ylim([-5, 5])

        ax2.set_xlabel('Time (sec)')
        ax2.set_ylabel('Feature Tracker Velocities')
        ax2.set_ylim([-5, 5])

        plt.tight_layout()

# 初期化
rospy.init_node('data_comparison_plotter')
rospy.Subscriber("/feature_tracker/feature", PointCloud, feature_callback)
rospy.Subscriber("/cf231/imu", Imu, imu_callback)

# Matplotlib設定
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
ani = animation.FuncAnimation(
    fig, 
    update_plot, 
    fargs=(
        feature_timestamps, 
        feature_velocities_x, 
        feature_velocities_y, 
        imu_timestamps, 
        imu_angular_velocities_x,
        imu_angular_velocities_y,
        imu_angular_velocities_z,
        imu_linear_accelerations_x,
        imu_linear_accelerations_y,
        imu_linear_accelerations_z
    ), 
    interval=100)

plt.show(block=True)
rospy.spin()