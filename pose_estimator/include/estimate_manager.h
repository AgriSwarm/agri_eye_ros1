#ifndef _ESTIMATE_MANAGER_H_
#define _ESTIMATE_MANAGER_H_

#include <ros/ros.h>
#include <motion_capture_tracking_msgs/EstimatedPose2D.h>
#include <motion_capture_tracking_msgs/EstimatedPose2DArray.h>
#include <motion_capture_tracking_msgs/EstimatedPose3D.h>
#include <motion_capture_tracking_msgs/EstimatedPose3DArray.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include <cv_bridge/cv_bridge.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <quadrotor_msgs/TrackingPose.h>
#include <std_srvs/Empty.h>
#include <std_srvs/SetBool.h>

namespace pose_estimator
{
  struct FlowerPose
  {
    Eigen::Vector3d position;
    Eigen::Vector3d normal;
    float probability;
  };
  typedef std::vector<FlowerPose> FlowerPoseArray;
  typedef std::shared_ptr<FlowerPose> FlowerPosePtr;

  // モードを定義
  enum class Mode
  {
    ON_SEARCH,
    ON_TRACK
  };
  enum class Color
  {
    RED,
    BLUE,
    YELLOW,
    GREEN
  };

  class EstimateManager
  {
  public:
    EstimateManager(ros::NodeHandle &nh);
    ~EstimateManager(){};

  private:
    ros::NodeHandle nh_;
    ros::Subscriber extrinsic_sub_, odom_sub_;
    ros::Publisher flower_poses_pub_, flower_poses_marker_pub_, target_flower_pose_pub_, target_pose_marker_pub_;
    ros::ServiceServer start_tracking_service_;
    ros::Timer timer_;
    ros::Time last_refind_time_;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry, motion_capture_tracking_msgs::EstimatedPose2DArray>
      SyncPolicyImageOdomPose;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, motion_capture_tracking_msgs::EstimatedPose2DArray>
      SyncPolicyImagePose;
    typedef std::shared_ptr<message_filters::Synchronizer<SyncPolicyImageOdomPose>> 
      SynchronizerImageOdomPose;
    typedef std::shared_ptr<message_filters::Synchronizer<SyncPolicyImagePose>> 
      SynchronizerImagePose;
    SynchronizerImageOdomPose sync_image_odom_pose_;
    SynchronizerImagePose sync_image_pose_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> depth_sub_;
    std::shared_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_sync_sub_;
    std::shared_ptr<message_filters::Subscriber<motion_capture_tracking_msgs::EstimatedPose2DArray>> flower_poses_sub_;

    camodocal::PinholeCameraPtr m_camera;
    Eigen::Matrix4d body2cam_;
    double depth_scaling_factor_, identity_threshold_, tracking_distance_, lost_time_threshold_, est_detect_pose_rate_;
    double capture_area_margin_, capture_area_radius_;
    bool camera_coordinate_, inverse_depth_;
    FlowerPosePtr target_flower_pose_;
    FlowerPoseArray flower_poses_;
    Eigen::Vector3d drone_position_;
    Eigen::Quaterniond drone_orientation_;
    Mode navigation_mode_;
    std::string world_frame_id_;

    bool startTrackingCallback(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res);
    void extrinsicCallback(const nav_msgs::Odometry::ConstPtr &msg);

    void depthOdomPoseCallback(const sensor_msgs::Image::ConstPtr &depth_msg,
                              const nav_msgs::Odometry::ConstPtr &odom_msg,
                              const boost::shared_ptr<motion_capture_tracking_msgs::EstimatedPose2DArray const> flower_poses_msg);

    void depthPoseCallback(const sensor_msgs::Image::ConstPtr &depth_msg,
                          const boost::shared_ptr<motion_capture_tracking_msgs::EstimatedPose2DArray const> flower_poses_msg);
    void odomCallback(const nav_msgs::Odometry::ConstPtr &msg);
    void targetCallback(const ros::TimerEvent &event);
    void visualizeFlowerPoses(
      const FlowerPoseArray &flower_poses, 
      ros::Publisher &pub,
      Color color);

    void selectTargetFlowerPose(FlowerPoseArray &flower_poses);
    FlowerPosePtr searchNearbyFlowerPose(const FlowerPoseArray &flower_poses, double threshold_distance);
    FlowerPosePtr correctFlowerPose(const FlowerPosePtr &target_flower_pose, const FlowerPoseArray &flower_poses);
  };
} // namespace pose_estimator

#endif // _ESTIMATE_MANAGER_H_