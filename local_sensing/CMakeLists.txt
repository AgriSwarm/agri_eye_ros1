cmake_minimum_required(VERSION 3.0.2)
project(local_sensing)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  image_transport
  cv_bridge
  sensor_msgs
)

find_package(OpenCV REQUIRED)

# catkin_python_setup()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES local_sensing
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(local_sensing src/local_sensing.cpp)

target_link_libraries(local_sensing
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

catkin_install_python(PROGRAMS 
  scripts/sync_viz.py
  DESTINATION 
  ${CATKIN_PACKAGE_BIN_DESTINATION})

add_dependencies(local_sensing ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})