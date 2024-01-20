# Agri Eye ROS1

> [!IMPORTANT]
> You need [agri_resources](https://github.com/AgriSwarm/agri_resources) and [VINS-Mono](https://github.com/libing64/VINS-Mono/tree/noetic-devel)

## VTX test
```bash
roslaunch local_sensing vtx_demo.launch
```

## VIO test
```bash
roslaunch local_sensing vins_estimate.launch
```

## with Crazyswarm2
```bash
# terminal 1
source /opt/ros/noetic/setup.bash
source /opt/ros/galactic/setup.bash
source ros1_bridge_ws/install/setup.bash
export ROS_MASTER_URI=http://localhost:11311
ros2 run ros1_bridge dynamic_bridge
# terminal 2
ros2 launch crazyflie launch.py backend:=cflib
```

