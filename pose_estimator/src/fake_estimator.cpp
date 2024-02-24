// fake_estimator.cpp

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/MarkerArray.h>
#include <quadrotor_msgs/TrackingPose.h>
#include "estimate_manager.h"

using namespace pose_estimator;

FlowerPoseArray fake_flower_poses_;
FlowerPosePtr target_flower_pose_;
Eigen::Vector3d drone_position_;
Eigen::Quaterniond drone_orientation_;
double sensing_distance_;
double tracking_distance_, capture_area_radius_, capture_area_margin_;
ros::Publisher flower_poses_marker_pub_, target_pose_marker_pub_, target_flower_pose_pub_, dummy_flower_poses_marker_pub_;
std::string world_frame_id_;
double est_detect_pose_rate_;
bool odom_received_ = false;

// ダミーの花の位置を設定
void setFakeFlowerPoses()
{
    FlowerPose pose;
    pose.position = Eigen::Vector3d(0.0, 0.0, 0.0);
    pose.normal = Eigen::Vector3d(0.5, 1.0, 0.2);
    pose.probability = 1.0;
    fake_flower_poses_.push_back(pose);
    pose.position = Eigen::Vector3d(3.0, 0.0, 0.0);
    pose.normal = Eigen::Vector3d(0.2, 0.9, 0.2);
    pose.probability = 1.0;
    fake_flower_poses_.push_back(pose);
    pose.position = Eigen::Vector3d(0.0, 3.0, 0.0);
    pose.normal = Eigen::Vector3d(0.3, 1.3, 0.2);
    pose.probability = 1.0;
    fake_flower_poses_.push_back(pose);
}

FlowerPoseArray getFakeFlowerPosesWithNoise()
{
    double translation_noise = 0.2;
    double rotation_noise = 0.8;

    FlowerPoseArray flower_poses;
    for (auto &pose : fake_flower_poses_)
    {
        FlowerPose noisy_pose;
        noisy_pose.position = pose.position + Eigen::Vector3d::Random() * translation_noise;
        noisy_pose.normal = pose.normal + Eigen::Vector3d::Random() * rotation_noise;
        noisy_pose.probability = pose.probability;
        flower_poses.push_back(noisy_pose);
    }
    return flower_poses;
}

void odomCallback(const nav_msgs::Odometry::ConstPtr &msg)
{
    odom_received_ = true;
    drone_position_ = Eigen::Vector3d(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    drone_orientation_ = Eigen::Quaterniond(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
}

FlowerPosePtr searchNearbyFlowerPose(const FlowerPoseArray &flower_poses, double threshold_distance)
{
    // ROS_INFO("searchNearbyFlowerPose");
    std::vector<FlowerPosePtr> nearby_flower_poses;
    for(auto &flower_pose : flower_poses)
    {
        if((flower_pose.position - drone_position_).norm() < threshold_distance)
        {
        nearby_flower_poses.push_back(std::make_shared<FlowerPose>(flower_pose));
        }
    }
    // nearby_flower_posesの中で最も近いものを選択
    if(nearby_flower_poses.size() > 0)
    {
        std::sort(nearby_flower_poses.begin(), nearby_flower_poses.end(), 
                [&](const FlowerPosePtr a, const FlowerPosePtr b)
                {
                    return (a->position - drone_position_).norm() < (b->position - drone_position_).norm();
                });     
        return nearby_flower_poses[0];
    }
    else
    {
        return nullptr;
    }
}

void visualizeFlowerPoses(
    const FlowerPoseArray &flower_poses, 
    ros::Publisher &pub,
    Color color)
{
// rviz visualization
visualization_msgs::MarkerArray marker_array;
for (size_t i = 0; i < flower_poses.size(); i++)
{
    visualization_msgs::Marker marker;
    marker.header.frame_id = world_frame_id_;
    marker.header.stamp = ros::Time::now();
    marker.ns = "flower_poses";
    marker.id = i;
    marker.lifetime = ros::Duration(1/est_detect_pose_rate_);
    marker.type = visualization_msgs::Marker::ARROW;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = flower_poses[i].position(0);
    marker.pose.position.y = flower_poses[i].position(1);
    marker.pose.position.z = flower_poses[i].position(2);
    Eigen::Vector3d normal = flower_poses[i].normal;

    Eigen::Vector3d u1, u2, u3;
    u1 = normal.normalized();
    if ((std::fabs(u1(0)) > 0.001) || (std::fabs(u1(1)) > 0.001)) {
        u2 = Eigen::Vector3d(-u1(1), u1(0), 0);
    } else {
        u2 = Eigen::Vector3d(0, u1(2), -u1(1));
    }
    u2.normalize();
    u3 = u1.cross(u2);

    Eigen::Matrix3d R;
    R.col(0) = u1;
    R.col(1) = u2;
    R.col(2) = u3;
    Eigen::Quaterniond q(R);

    marker.pose.orientation.x = q.x();
    marker.pose.orientation.y = q.y();
    marker.pose.orientation.z = q.z();
    marker.pose.orientation.w = q.w();

    marker.scale.x = 0.5;
    marker.scale.y = 0.05;
    marker.scale.z = 0.05;
    marker.lifetime = ros::Duration(0.1);
    // marker.color.a = flower_poses[i].probability;
    marker.color.a = 1.0;
    if (color == Color::RED)
    {
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    }
    else if (color == Color::BLUE)
    {
    marker.color.r = 0.0;
    marker.color.g = 0.0;
    marker.color.b = 1.0;
    }
    else if (color == Color::YELLOW)
    {
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    }
    else if (color == Color::GREEN)
    {
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    }
    else
    {
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    }
    marker_array.markers.push_back(marker);
}
pub.publish(marker_array);
}

void timerCallback(const ros::TimerEvent &event)
{
    if(!odom_received_)
    {
        return;
    }
    fake_flower_poses_ = getFakeFlowerPosesWithNoise();
    FlowerPoseArray flower_poses;
    for (auto &pose : fake_flower_poses_)
    {
        if ((pose.position - drone_position_).norm() < sensing_distance_)
        {
            flower_poses.push_back(pose);
        }
    }
    visualizeFlowerPoses(fake_flower_poses_, dummy_flower_poses_marker_pub_, Color::BLUE);
    visualizeFlowerPoses(flower_poses, flower_poses_marker_pub_, Color::RED);
    target_flower_pose_ = searchNearbyFlowerPose(fake_flower_poses_, capture_area_radius_);
    if (target_flower_pose_ != nullptr)
    {
        FlowerPoseArray target_flower_poses;
        target_flower_poses.push_back(*target_flower_pose_);
        visualizeFlowerPoses(target_flower_poses, target_pose_marker_pub_, Color::GREEN);

        quadrotor_msgs::TrackingPose tracking_pose;
        tracking_pose.center.x = target_flower_pose_->position(0);
        tracking_pose.center.y = target_flower_pose_->position(1);  
        tracking_pose.center.z = target_flower_pose_->position(2);
        tracking_pose.normal.x = target_flower_pose_->normal(0);
        tracking_pose.normal.y = target_flower_pose_->normal(1);
        tracking_pose.normal.z = target_flower_pose_->normal(2);
        tracking_pose.distance = tracking_distance_;
        tracking_pose.target_status = quadrotor_msgs::TrackingPose::TARGET_STATUS_CAPTURED;
        target_flower_pose_pub_.publish(tracking_pose);
    }else{
        FlowerPosePtr approx_flower_pose = searchNearbyFlowerPose(fake_flower_poses_, sensing_distance_);
        if(approx_flower_pose != nullptr)
        {
            quadrotor_msgs::TrackingPose tracking_pose;
            tracking_pose.center.x = approx_flower_pose->position(0);
            tracking_pose.center.y = approx_flower_pose->position(1);  
            tracking_pose.center.z = approx_flower_pose->position(2);
            tracking_pose.normal.x = approx_flower_pose->normal(0);
            tracking_pose.normal.y = approx_flower_pose->normal(1);
            tracking_pose.normal.z = approx_flower_pose->normal(2);
            tracking_pose.target_status = quadrotor_msgs::TrackingPose::TARGET_STATUS_APPROXIMATE;
            target_flower_pose_pub_.publish(tracking_pose);
        }
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pose_estimator");
    ros::NodeHandle nh("~");
    nh.param("world_frame_id", world_frame_id_, std::string("world"));
    nh.param("est_detect_pose_rate", est_detect_pose_rate_, 10.0);
    nh.param("sensing_distance", sensing_distance_, 5.0);
    nh.param("tracking_distance", tracking_distance_, 1.0);
    nh.param("capture_area_margin", capture_area_margin_, 0.5);
    capture_area_radius_ = tracking_distance_ + capture_area_margin_;

    dummy_flower_poses_marker_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/dummy_flower_poses_marker", 1);
    flower_poses_marker_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/flower_poses_marker", 1);
    target_pose_marker_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/target_pose_marker", 1);
    target_flower_pose_pub_ = nh.advertise<quadrotor_msgs::TrackingPose>("/target_flower_pose", 1);

    ros::Subscriber odom_sub = nh.subscribe("/odom", 1, odomCallback);
    ros::Timer timer_ = nh.createTimer(ros::Duration(1/est_detect_pose_rate_), timerCallback);

    setFakeFlowerPoses();

    ros::spin();
    return 0;
}