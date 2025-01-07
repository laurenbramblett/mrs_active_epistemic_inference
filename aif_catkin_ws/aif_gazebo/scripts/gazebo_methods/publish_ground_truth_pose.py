#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Path
from gazebo_msgs.srv import GetModelState
import yaml

# Load the YAML file
def get_yaml_params(file_path):
    with open(file_path, 'r') as file:
        yaml_params = yaml.safe_load(file)
    return yaml_params

# Get the true model pose in gazebo (if in simulation)
def get_model_pose(model_name):
    # Wait for the service to become available
    rospy.wait_for_service('/gazebo/get_model_state')
    try:
        # Create a service proxy
        get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        # Request the model state
        resp = get_model_state(model_name, "")

        if resp.success:
            # print("Got model state for model: %s", model_name)
            return resp.pose
        else:
            # rospy.logerr("Failed to get model state for model: %s", model_name)
            return None

    except rospy.ServiceException as e:
        # rospy.logerr("Service call failed: %s", e)
        return None

def publish_world_odom(pose, pub, frame_id="map"):
    msg = PoseWithCovarianceStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    if pose:
        # print("Publishing pose: ", pose)
        msg.pose.pose.position.x = pose.position.x
        msg.pose.pose.position.y = pose.position.y
        msg.pose.pose.position.z = pose.position.z
        msg.pose.pose.orientation.x = pose.orientation.x
        msg.pose.pose.orientation.y = pose.orientation.y
        msg.pose.pose.orientation.z = pose.orientation.z
        msg.pose.pose.orientation.w = pose.orientation.w
        pub.publish(msg)

def publish_path(pose, path, pub_path, frame_id="map"):
    path.header.frame_id = frame_id
    path.header.stamp = rospy.Time.now()
    pose_msg = PoseStamped()
    pose_msg.header.frame_id = frame_id
    pose_msg.header.stamp = rospy.Time.now()
    pose_msg.pose.position.x = pose.position.x
    pose_msg.pose.position.y = pose.position.y
    pose_msg.pose.position.z = pose.position.z
    pose_msg.pose.orientation.x = pose.orientation.x
    pose_msg.pose.orientation.y = pose.orientation.y
    pose_msg.pose.orientation.z = pose.orientation.z
    pose_msg.pose.orientation.w = pose.orientation.w
    path.poses.append(pose_msg)
    pub_path.publish(path)
    return path

    

if __name__ == '__main__':
    rospy.init_node('world_odom_publisher')
    params_file = rospy.get_param('params_file')
    params = get_yaml_params(params_file)
    paths = [Path() for i in range(len(params['agents']))]
    pub_names = [params['agents'][i]['namespace'] + "/" + params['agents'][i]['pose_topic'] for i in range(len(params['agents']))]
    pub_path_names = [params['agents'][i]['namespace'] + "/" + params['agents'][i]['pose_topic'] + '_path' for i in range(len(params['agents']))]
    pubs = [rospy.Publisher(pub_names[i], PoseWithCovarianceStamped, queue_size=1) for i in range(len(pub_names))]
    pub_paths = [rospy.Publisher(pub_path_names[i], Path, queue_size=1) for i in range(len(pub_path_names))]
    sub_names = [params['agents'][i]['namespace'] for i in range(len(params['agents']))]
    rate = rospy.Rate(30)
    while not rospy.is_shutdown(): 
        pose = [get_model_pose(sub_names[i]) for i in range(len(sub_names))]  
        if None in pose:
            rate.sleep()
            continue    
        for idx, pub in enumerate(pubs):
            publish_world_odom(pose[idx], pub, params['world_frame_id'])
            paths[idx] = publish_path(pose[idx], paths[idx], pub_paths[idx], params['world_frame_id'])
        rate.sleep()