#!/usr/bin/env python

import rospy
import yaml
from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped, PoseWithCovariance

def get_yaml_params(file_path):
    with open(file_path, 'r') as file:
        yaml_params = yaml.safe_load(file)
    return yaml_params

def transform_to_pose_with_covariance(transform_stamped):
    pose_with_covariance = PoseWithCovarianceStamped()

    pose_with_covariance.header = transform_stamped.header
    pose_with_covariance.header.frame_id = "vicon/world"

    pose_with_covariance.pose.pose.position.x = transform_stamped.transform.translation.x
    pose_with_covariance.pose.pose.position.y = transform_stamped.transform.translation.y
    pose_with_covariance.pose.pose.position.z = transform_stamped.transform.translation.z

    pose_with_covariance.pose.pose.orientation = transform_stamped.transform.rotation

    # Initialize covariance with some default values (e.g., identity matrix)
    # Modify this as necessary for your application
    pose_with_covariance.pose.covariance = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]

    return pose_with_covariance

def callback(msg, args):
    pose_with_covariance_stamped = transform_to_pose_with_covariance(msg)
    publishers[args].publish(pose_with_covariance_stamped)

if __name__ == '__main__':
    rospy.init_node('transform_to_pose_node')

    # Get the params file path from the parameter server
    params_file = rospy.get_param('params_file')
    
    # Load parameters from the YAML file
    params = get_yaml_params(params_file)
    
    # Create a dictionary to hold publishers for each agent
    publishers = {}
    
    for agent in range(len(params['agents'])):
        transformed_topic = params['agents'][agent]['transformed_pose_topic']
        pose_topic = params['agents'][agent]['pose_topic']

        rospy.Subscriber(pose_topic, TransformStamped, callback, (agent))
        publishers[agent] = rospy.Publisher(transformed_topic, PoseWithCovarianceStamped, queue_size=1)

    rospy.spin()
