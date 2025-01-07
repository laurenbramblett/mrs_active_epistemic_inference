cmake_minimum_required(VERSION 3.0.2)
project(multi_jackal_aif)
add_definitions(-std=c++17)

find_package(catkin REQUIRED COMPONENTS 
  roslaunch  
  roscpp
  rospy
  sensor_msgs
  std_msgs  
  nav_msgs
  tf2_ros
  tf2_geometry_msgs
  std_srvs
)

catkin_package()



roslaunch_add_file_check(launch/amcl_aif_launch.launch)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${catkin_INCLUDE_DIRS}
)
catkin_install_python(PROGRAMS src/active_inference_server.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
	DIRECTORY launch rviz worlds
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
