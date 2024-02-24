cmake_minimum_required(VERSION 2.8.3)
project(motion_capture_tracking_msgs)

find_package(catkin REQUIRED COMPONENTS
  #armadillo
  roscpp
  nav_msgs
  geometry_msgs
  message_generation
)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include_directories(
    ${catkin_INCLUDE_DIRS}
    include
    )

add_message_files(
    FILES
    NamedPose.msg
    NamedPoseArray.msg
    ExtendedNamedPose.msg
    ExtendedNamedPoseArray.msg
    EstimatedPose2D.msg
    EstimatedPose2DArray.msg
    EstimatedPose3D.msg
    EstimatedPose3DArray.msg)

generate_messages(
    DEPENDENCIES
    std_msgs
    geometry_msgs)

catkin_package(
  #INCLUDE_DIRS include
  #CATKIN_DEPENDS geometry_msgs nav_msgs
  #DEPENDS system_lib
  CATKIN_DEPENDS message_runtime
)



find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})


# target_link_libraries(test_exe 
#     decode_msgs 
#     encode_msgs
# )

