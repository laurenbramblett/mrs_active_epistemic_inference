### This repository is a playground for collaboration in Active Inference and Epistemic Planning
The most updated notebook is `notebook_aif_iterative_goals.ipynb` which allows the robots to iteratively select and complete goals. The most updated notebook for single goals is `notebook_aif_single_goals.ipynb`. The output of these files is a mp4 or interactive plot of robots completing the mission objectives. Toggle parameters such as:
1. `use_ep`: Set to `True` or `False` to toggle our approach
2. `greedy`: Set to `True` if you want to use a greedy approach (zero-order reasoning). If true, does not allow first or higher order reasoning from (1)
3. `convergence_type`: Set to `convergent` if you want robots to have a rendezvous mission and `exclusive` if running task allocation

Note: `convergence_type` should always be exclusive in the `notebook_aif_iterative_goals.ipynb`

##### Installation
The following installation instructions assume ROS Noetic, Gazebo, and Ubuntu 20.04 are installed
```
sudo apt-get install python3-rosdep ros-noetic-jackal-simulator ros-noetic-jackal-desktop ros-noetic-jackal-navigation ros-noetic-pointgrey-camera-driver
mkdir -p ~/mrs_aif_ws/src # Make a catkin workspace
git clone git@github.com:laurenbramblett/active_inference_playground.git
cd active_inference_playground/
pip install -r requirements.txt
```
To run the multi jackal gazebo simulation, the multi jackal simulator must be installed using the following instructions:
```
cd ~/mrs_aif_ws/src
git clone https://github.com/UVA-BezzoRobotics-AMRLab/multi_jackal_amcl.git
```
To run the heterogeneous gazebo simulation, we have added the capability to use the rotors simulator (https://github.com/ethz-asl/rotors_simulator.git). However, installation is accomplished via slightly modified instructions:
```
mkdir -p ~/rotors_ws/src
sudo apt-get install ros-noetic-joy ros-noetic-octomap-ros ros-noetic-mavlink python3-wstool python3-catkin-tools protobuf-compiler libgoogle-glog-dev ros-noetic-control-toolbox ros-noetic-mavros ros-noetic-libmavconn
git clone https://github.com/ethz-asl/rotors_simulator.git
git clone https://github.com/ethz-asl/mav_comm.git

cd ~/rotors_ws
source /opt/ros/noetic/setup.bash
catkin_make # This is pretty important for ROS Noetic, otherwise the drivers are not found for the simulator
```
The active inference package needs to be rebuilt by extending the catkin configuration to include the rotors simulator package so that we can use the quadrotor drivers.
```
cd ~/mrs_aif_ws
catkin init
catkin config --extend ~/rotors_ws/devel
catkin build
source ~/mrs_aif_ws/devel/setup.bash
```

##### Running Gazebo Experiments

```
# Make sure that ROS is setup properly -- only tested with ROS Noetic 
sudo apt install tmux
#Tmux is required for the multi-vehicle setup
```
In two separate terminals pass these lines. You can manipulate the config file that exists or make your own (this one only launches jackals):
```
roslaunch multi_jackal_aif gazebo_ground_truth_aif_launch.launch
rosservice call /aif_start_planning "data: true" #This starts the planning service node
```
If instead, you want to run localization, make sure all robots are jackals because it uses lidar data and run the following:
```
roslaunch multi_jackal_aif amcl_aif_launch.launch
```

You can view the rviz instance by going to:
```
rosrun rviz rviz -d <rviz_path> # Example included in: ~/jackal_ws/src/aif_package/aif_multi_robot/aif_catkin_ws/aif_gazebo/rviz/two_jackal_rviz.rviz
```
If you aren't familiar with tmux and you want to stop the entire tmux server type below in a new terminal window:
```
tmux kill-server
```

##### Running lab experiment
This lab experiment assumes that you are running vicon as the motion capture system and runs the drivers for you. If any other motion capture system is used, you will need to launch this driver separately and comment out the vicon driver.

The yaml file in the configs folder (~/mrs_aif_ws/aif_multi_robot/aif_catkin_ws/aif_gazebo/configs/) includes an example `aif_lab_params.yaml`. Update the pose topic names and namespaces as needed. You may also manipulate the transformed poses.


###### For AMR Lab Experiments
set goals within the aif_lab_params.yaml file
ssh into robots using ssh husarion@192.168.8.xxx
run drivers using roslaunch husarion_ros rosbot_drivers_nolidar.launch (only on 4 and 8. for the rest us rosbot_drivers.launch)

changes to code will occur in the AIF_GAZEBO>scripts>experiment_methods>aif_functions_isobeliefs_convergent_lab.py

##### Here is some of our papers on epistemic planning that are useful and relevant
1. [Robust Online Epistemic Replanning for Multi Robot Systems](https://arxiv.org/pdf/2403.00641)
2. [Epistemic Prediction and Planning with Implicit Coordination for Multi-Robot Teams in Communication Restricted Environments](https://arxiv.org/pdf/2302.10393)

##### Here are some papers that are relevant for our work and where we are heading
1. [Interactive Inference: A Multi-Agent Model of Cooperative Joint Actions](https://arxiv.org/pdf/2210.13113)
2. [Active Inference and Behavior Trees for Reactive Action Planning and Execution in Robotics](https://arxiv.org/pdf/2011.09756)
3. Baseline paper for multi-robot task allocation (CBBA): [Consensus-Based Decentralized Auctions for Robust Task Allocation](https://dspace.mit.edu/bitstream/handle/1721.1/52330/Choi_Consensus-Based-Decentralized.pdf?sequence=2)
4. Simple decentralized rendezvous paper that I really like: [Bayesian Rendezvous for Distributed Robotic Systems](https://web.archive.org/web/20170818191431id_/https://infoscience.epfl.ch/record/168217/files/paper.pdf)

