cmake_minimum_required(VERSION 2.8.3)
project(pose_estimator)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    geometry_msgs
    quadrotor_msgs
    agri_eye_msgs
    sensor_msgs
    camera_models
    cv_bridge
    roslib
    message_generation
    tf
    image_transport
    )

SET("OpenCV_DIR"  "/usr/local/share/OpenCV/")
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found.")
endif()

find_package(Ceres REQUIRED)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)

include_directories(
    include
    SYSTEM
    ${catkin_INCLUDE_DIRS} 
    ${PROJECT_SOURCE_DIR}/include
    ${CERES_INCLUDE_DIRS}  
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

catkin_package(
    INCLUDE_DIRS include
)

add_executable(pose_estimator
    src/pose_estimator_node.cpp
    # src/flower_pose_estimator.cpp
    src/estimate_manager.cpp
    )

add_executable(fake_estimator
    src/fake_estimator.cpp
    )

add_executable(webcam src/webcam.cpp)

target_link_libraries(pose_estimator 
    ${catkin_LIBRARIES}  
    ${OpenCV_LIBRARIES}
) 

target_link_libraries(fake_estimator 
    ${catkin_LIBRARIES}  
    ${OpenCV_LIBRARIES}
)

target_link_libraries(webcam
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
# message("catkin_lib  ${catkin_LIBRARIES}")

catkin_install_python(PROGRAMS 
#   scripts/flower_pose_estimator.py
#   scripts/fake_flower_pose.py
  scripts/flower_detector.py
#   scripts/flower_detector_dev.py
  scripts/image_talker.py
  DESTINATION 
  ${CATKIN_PACKAGE_BIN_DESTINATION})