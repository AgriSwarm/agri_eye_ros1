#include <ros/ros.h>
#include "estimate_manager.h"

using namespace pose_estimator;

int main(int argc, char **argv)
{
  ros::init(argc, argv, "pose_estimator");
//   ros::NodeHandle nh;
  ros::NodeHandle nh("~");

  EstimateManager manager(nh);

  ros::spin();
  return 0;
}