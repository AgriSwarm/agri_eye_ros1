#include "estimate_manager.h"

namespace pose_estimator
{
  EstimateManager::EstimateManager(ros::NodeHandle &nh) : nh_(nh)
  {
    // flower_poses_sub_ = nh_.subscribe("/flower_poses", 1, &EstimateManager::flowerPosesCallback, this);
    // depth_sub_ = nh_.subscribe("/depth", 1, &EstimateManager::depthCallback, this);
    // odom_sub_ = nh_.subscribe("/odom", 1, &EstimateManager::odomCallback, this);
    // flower_poses_pub_ = nh_.advertise<agri_eye_msgs::EstimatedPose3DArray>("/flower_poses_3d", 1);

    std::string config_file;
    nh_.param<std::string>("config_file", config_file, "");
    nh_.param("depth_scaling_factor", depth_scaling_factor_, 1.0);
    nh_.param("inverse_depth", inverse_depth_, false);
    nh_.param("camera_coordinate", camera_coordinate_, false);
    nh_.param("identity_threshold", identity_threshold_, 0.5);
    nh_.param("tracking_distance", tracking_distance_, 0.5);
    nh_.param("capture_area_margin", capture_area_margin_, 1.0);
    nh_.param("lost_time_threshold", lost_time_threshold_, 1.0);
    nh_.param("world_frame_id", world_frame_id_, std::string("world"));
    nh_.param("est_detect_pose_rate", est_detect_pose_rate_, 10.0);
    // nh_.param("approximated_target_pose", approximated_target_pose_, Eigen::Vector3d(0.0, 0.0, 0.0));
    std::vector<double> temp_vector;
    nh_.param("approximated_target_pose", temp_vector, std::vector<double>{0.0, 0.0, 0.0});
    if (temp_vector.size() == 3) {
        approximated_target_pose_ = Eigen::Vector3d(temp_vector[0], temp_vector[1], temp_vector[2]);
    } else {
        approximated_target_pose_ = Eigen::Vector3d(0.0, 0.0, 0.0);
    }
    nh_.param("sensing_distance", sensing_radius_, 1.0);

    cv::FileStorage general_fs;

    ROS_INFO("Loading config file: %s", config_file.c_str());
    general_fs.open(config_file.c_str(), cv::FileStorage::READ);
    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);

    std::string camera_calib_path = (std::string) general_fs["color_cam_calib"];
    camera_calib_path = configPath + "/" + camera_calib_path;

    ROS_INFO("Loading camera model: %s", camera_calib_path.c_str());
    auto camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(camera_calib_path.c_str());
    m_camera = boost::dynamic_pointer_cast<camodocal::PinholeCamera>(camera);
    if(m_camera == nullptr)
    {
      ROS_ERROR("Failed to load camera model");
      return;
    }

    cv::Mat T_imu_cam, T_base_imu;
    Eigen::Matrix4d imu2cam_, base2imu;
    general_fs["imu_T_depth_cam"] >> T_imu_cam;
    general_fs["base_T_imu"] >> T_base_imu;
    // if(general_fs["approximated_target_pose"] != NULL)
    // {
    //   cv::Mat approximated_target_pose;
    //   general_fs["approximated_target_pose"] >> approximated_target_pose;
    //   approximated_target_pose_ << approximated_target_pose.at<double>(0), approximated_target_pose.at<double>(1), approximated_target_pose.at<double>(2);
    // }
    cv::cv2eigen(T_imu_cam, imu2cam_);
    cv::cv2eigen(T_base_imu, base2imu);
    body2cam_ = base2imu * imu2cam_;

    navigation_mode_ = Mode::ON_SEARCH;
    capture_area_radius_ = tracking_distance_ + capture_area_margin_;

    flower_poses_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/flower_poses_marker", 1);
    target_pose_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/target_pose_marker", 1);
    target_flower_pose_pub_ = nh_.advertise<quadrotor_msgs::TrackingPose>("/target_flower_pose", 1);
    pose_estimate_delay_pub = nh.advertise<std_msgs::Float32>("pose_estimate_delay", 1000);

    depth_sub_.reset(new message_filters::Subscriber<sensor_msgs::Image>(nh, "/depth", 50));
    flower_poses_sub_.reset(new message_filters::Subscriber<agri_eye_msgs::EstimatedPose2DArray>(nh, "/flower_poses", 100));
    approximated_target_pose_sub_ = nh.subscribe<quadrotor_msgs::Ball>("/approximated_target_pose", 1, &EstimateManager::approximatedTargetPoseCallback, this);

    if (camera_coordinate_)
    {
      sync_image_pose_.reset(new message_filters::Synchronizer<SyncPolicyImagePose>(
          SyncPolicyImagePose(100), *depth_sub_, *flower_poses_sub_));
      sync_image_pose_->registerCallback(boost::bind(&EstimateManager::depthPoseCallback, this, _1, _2));
    }
    else{
      odom_sync_sub_.reset(new message_filters::Subscriber<nav_msgs::Odometry>(nh, "/odom", 100, ros::TransportHints().tcpNoDelay()));
      extrinsic_sub_ = nh.subscribe<nav_msgs::Odometry>(
        "/vins_estimator/extrinsic", 10, &EstimateManager::extrinsicCallback, this); //sub

      sync_image_odom_pose_.reset(new message_filters::Synchronizer<SyncPolicyImageOdomPose>(
          SyncPolicyImageOdomPose(100), *depth_sub_, *odom_sync_sub_, *flower_poses_sub_));
      sync_image_odom_pose_->registerCallback(boost::bind(&EstimateManager::depthOdomPoseCallback, this, _1, _2, _3)); //sub
    }
    odom_sub_ = nh.subscribe("/odom", 1, &EstimateManager::odomCallback, this);
    // start_tracking_service_ = nh.advertiseService("/custom", &EstimateManager::startTrackingCallback, this);
    timer_ = nh_.createTimer(ros::Duration(0.1), &EstimateManager::targetCallback, this);
  }

  void EstimateManager::approximatedTargetPoseCallback(const quadrotor_msgs::Ball::ConstPtr &msg)
  {
    approximated_target_pose_ << msg->centroid.x, msg->centroid.y, msg->centroid.z;
    sensing_radius_ = msg->radius;
  }

  bool EstimateManager::startTrackingCallback(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res)
  {
    timer_ = nh_.createTimer(ros::Duration(0.1), &EstimateManager::targetCallback, this);
    ROS_INFO("Start tracking service called");
    return true;
  }

  void EstimateManager::targetCallback(const ros::TimerEvent &event)
  {
    if (navigation_mode_ == Mode::ON_TRACK){
      if (ros::Time::now() - last_refind_time_ > ros::Duration(lost_time_threshold_))
      {
        ROS_INFO("Lost target flower, Mode changed from ON_TRACK to ON_SEARCH");
        navigation_mode_ = Mode::ON_SEARCH;
        quadrotor_msgs::TrackingPose tracking_pose;
        tracking_pose.target_status = quadrotor_msgs::TrackingPose::TARGET_STATUS_LOST;
        target_flower_pose_pub_.publish(tracking_pose);
        return;
      }
      if (target_flower_pose_ != nullptr)
      {
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
        FlowerPoseArray flower_poses;
        flower_poses.push_back(*target_flower_pose_);
        visualizeFlowerPoses(flower_poses, target_pose_marker_pub_, Color::GREEN);
      }
    }else{
      // TODO: p \in B
      FlowerPosePtr pose = searchNearbyFlowerPose(flower_poses_, 100);
      if (pose != nullptr && checkInBall(pose, approximated_target_pose_, sensing_radius_))
      {
        quadrotor_msgs::TrackingPose tracking_pose;
        tracking_pose.center.x = pose->position(0);
        tracking_pose.center.y = pose->position(1);  
        tracking_pose.center.z = pose->position(2);
        tracking_pose.normal.x = pose->normal(0);
        tracking_pose.normal.y = pose->normal(1);
        tracking_pose.normal.z = pose->normal(2);
        tracking_pose.target_status = quadrotor_msgs::TrackingPose::TARGET_STATUS_APPROXIMATE;
        target_flower_pose_pub_.publish(tracking_pose);
      }
    }
  }

  void EstimateManager::extrinsicCallback(const nav_msgs::Odometry::ConstPtr &msg)
  {
    ROS_INFO("extrinsicCallback");
    Eigen::Quaterniond q;
    q.x() = msg->pose.pose.orientation.x;
    q.y() = msg->pose.pose.orientation.y;
    q.z() = msg->pose.pose.orientation.z;
    q.w() = msg->pose.pose.orientation.w;
    Eigen::Vector3d t;
    t(0) = msg->pose.pose.position.x;
    t(1) = msg->pose.pose.position.y;
    t(2) = msg->pose.pose.position.z;
    body2cam_.block<3, 3>(0, 0) = q.toRotationMatrix();
    body2cam_.block<3, 1>(0, 3) = t;
  }

  void EstimateManager::depthOdomPoseCallback(const sensor_msgs::Image::ConstPtr &depth_msg,
                                              const nav_msgs::Odometry::ConstPtr &odom_msg,
                                              const boost::shared_ptr<agri_eye_msgs::EstimatedPose2DArray const> flower_poses_msg)
  {
    // ROS_INFO("depthOdomPoseCallback");
    if (flower_poses_msg->poses.size() == 0)
    {
      return;
    }

    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(depth_msg, depth_msg->encoding);
    if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
    {
      (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1);
    }

    Eigen::Matrix4d world2body;
    world2body = Eigen::Matrix4d::Identity();
    world2body.block<3, 3>(0, 0) = Eigen::Quaterniond(odom_msg->pose.pose.orientation.w,
                                                      odom_msg->pose.pose.orientation.x,
                                                      odom_msg->pose.pose.orientation.y,
                                                      odom_msg->pose.pose.orientation.z)
                                       .toRotationMatrix();
    world2body.block<3, 1>(0, 3) = Eigen::Vector3d(odom_msg->pose.pose.position.x,
                                                  odom_msg->pose.pose.position.y,
                                                  odom_msg->pose.pose.position.z);

    Eigen::Matrix4d cam_T = world2body * body2cam_;
    Eigen::Matrix3d cam_R = cam_T.block<3, 3>(0, 0);
    Eigen::Vector3d cam_t = cam_T.block<3, 1>(0, 3);

    flower_poses_.clear();

    for (auto &pose : flower_poses_msg->poses)
    {
      FlowerPose flower_pose;
      Eigen::Vector3d point, normal;
      point = LiftProjective(Eigen::Vector4d(pose.x_1, pose.y_1, pose.x_2, pose.y_2), cv_ptr);
      if (point.norm() < 0.01)
      {
        continue;
      }
      normal << pose.normal.x, pose.normal.y, pose.normal.z;
      
      flower_pose.position = cam_R * point + cam_t;
      flower_pose.normal = cam_R * normal;

      // flower_pose.probability = pose.probability;
      flower_pose.pos_prob = pose.pos_prob;
      flower_pose.ori_prob = pose.ori_prob;
      flower_poses_.push_back(flower_pose);
    }

    if (flower_poses_.size() == 0)
    {
      return;
    }

    double delay = (ros::Time::now().toSec() - flower_poses_msg->header.stamp.toSec()) * 1000;
    std_msgs::Float32 delay_msg;
    delay_msg.data = delay;
    pose_estimate_delay_pub.publish(delay_msg);

    selectTargetFlowerPose(flower_poses_);
    visualizeFlowerPoses(flower_poses_, flower_poses_marker_pub_, Color::RED);
  }

  Eigen::Vector3d EstimateManager::LiftProjective(Eigen::Vector4d rect, cv_bridge::CvImagePtr cv_ptr)
  {
    // calculate center of the rectangle
    Eigen::Vector2d center;
    center << (rect(0) + rect(2)) / 2, (rect(1) + rect(3)) / 2;
    // calculate depth
    uint16_t val = cv_ptr->image.at<uint16_t>(center(1), center(0));
    float depth;
    if (inverse_depth_)
    {
      depth = m_camera->getParameters().fx() / (val * depth_scaling_factor_);
    }
    else
    {
      depth = val * depth_scaling_factor_;
    }
    // calculate point
    Eigen::Vector3d point;
    m_camera->liftProjective(center, point);
    return depth * point;
  }

  void EstimateManager::odomCallback(const nav_msgs::Odometry::ConstPtr &msg)
  {
    // ROS_INFO("odomCallback");
    drone_position_ = Eigen::Vector3d(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    drone_orientation_ = Eigen::Quaterniond(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
  }

  void EstimateManager::depthPoseCallback(const sensor_msgs::Image::ConstPtr &depth_msg,
                                          const boost::shared_ptr<agri_eye_msgs::EstimatedPose2DArray const> flower_poses_msg)
  {
    // ROS_INFO("depthPoseCallback");
    if (flower_poses_msg->poses.size() == 0)
    {
      return;
    }

    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(depth_msg, depth_msg->encoding);
    if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
    {
      (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1);
    }

    Eigen::Matrix3d cam_R = body2cam_.block<3, 3>(0, 0);
    Eigen::Vector3d cam_t = body2cam_.block<3, 1>(0, 3);

    FlowerPoseArray flower_poses;

    for (auto &pose : flower_poses_msg->poses)
    {
      FlowerPose flower_pose;
      Eigen::Vector3d point, normal;
      point = LiftProjective(Eigen::Vector4d(pose.x_1, pose.y_1, pose.x_2, pose.y_2), cv_ptr);
      normal << pose.normal.x, pose.normal.y, pose.normal.z;

      flower_pose.position = cam_R * point + cam_t;
      flower_pose.normal = cam_R * normal;
      // flower_pose.probability = pose.probability;
      flower_pose.pos_prob = pose.pos_prob;
      flower_pose.ori_prob = pose.ori_prob;
      flower_poses.push_back(flower_pose);
    }

    if (flower_poses.size() == 0)
    {
      return;
    }

    double delay = (ros::Time::now().toSec() - flower_poses_msg->header.stamp.toSec()) * 1000;
    std_msgs::Float32 delay_msg;
    delay_msg.data = delay;
    pose_estimate_delay_pub.publish(delay_msg);

    selectTargetFlowerPose(flower_poses);
    visualizeFlowerPoses(flower_poses, flower_poses_marker_pub_, Color::RED);
  }

  // Mockup function
  void EstimateManager::selectTargetFlowerPose(FlowerPoseArray &flower_poses)
  {
    if(navigation_mode_ == Mode::ON_SEARCH)
    {
      target_flower_pose_ = searchNearbyFlowerPose(flower_poses, capture_area_radius_);
      if (target_flower_pose_ != nullptr && checkInBall(target_flower_pose_, approximated_target_pose_, sensing_radius_))
      {
        ROS_INFO("Mode changed from ON_SEARCH to ON_TRACK");
        navigation_mode_ = Mode::ON_TRACK;
      }
    }
    else if(navigation_mode_ == Mode::ON_TRACK)
    {
      target_flower_pose_ = correctFlowerPose(target_flower_pose_, flower_poses);
      if(target_flower_pose_ == nullptr){
        ROS_INFO("Lost target flower, Mode changed from ON_TRACK to ON_SEARCH");
        navigation_mode_ = Mode::ON_SEARCH;
      }
    }
  }

  FlowerPosePtr EstimateManager::searchNearbyFlowerPose(const FlowerPoseArray &flower_poses, double threshold_distance)
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
                [this](const FlowerPosePtr &a, const FlowerPosePtr &b)
                {
                  return (a->position - drone_position_).norm() < (b->position - drone_position_).norm();
                });   
      last_refind_time_ = ros::Time::now();        
      return nearby_flower_poses[0];
    }
    else
    {
      return nullptr;
    }
  }

  bool EstimateManager::checkInBall(const FlowerPosePtr &target_flower_pose, Eigen::Vector3d centroid, double radius)
  {
    // ROS_INFO("checkInBall");
    if(target_flower_pose == nullptr)
    {
      return false;
    }
    if((target_flower_pose->position - centroid).norm() < radius)
    {
      return true;
    }
    else
    {
      return false;
    }
  }

  FlowerPosePtr EstimateManager::correctFlowerPose(const FlowerPosePtr &target_flower_pose, const FlowerPoseArray &flower_poses)
  {
    // ROS_INFO("correctFlowerPose");
    if(target_flower_pose == nullptr)
    {
      return nullptr;
    }
    FlowerPoseArray poses = flower_poses;

    // ROS_INFO("poses.size() = %d", poses.size());
    std::sort(poses.begin(), poses.end(), 
          [this, &target_flower_pose](const FlowerPose &a, const FlowerPose &b) // ここのキャプチャリストを修正
          {
            return (a.position - target_flower_pose->position).norm() < (b.position - target_flower_pose->position).norm();
          });
    // ROS_INFO("poses[0].position = %f, %f, %f", poses[0].position(0), poses[0].position(1), poses[0].position(2));

    // // flower_poses[0]とtarget_flower_poseの距離がidentity_threshold_以下ならtarget_flower_poseを更新
    if((poses[0].position - target_flower_pose->position).norm() < identity_threshold_)
    {
      // ROS_INFO("Target flower pose updated");
      last_refind_time_ = ros::Time::now();
      return std::make_shared<FlowerPose>(poses[0]);
    }
    else
    {
      // ROS_INFO("Target flower pose not updated");
      return target_flower_pose;
    }
  }

  void EstimateManager::visualizeFlowerPoses(
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

      marker.scale.x = 0.2;
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

} // namespace pose_estimator