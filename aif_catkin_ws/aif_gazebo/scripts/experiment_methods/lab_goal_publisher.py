#!/usr/bin/env python
import rospy
import yaml
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, TransformStamped

global agent_positions, pose_received
pose_received = []
agent_positions = {}

def callback(msg, args):
    id = args['id']
    msg.header.frame_id = str(args['namespace']) + '/' + str(args['scan_frame_id'])
    transformed_scan_msg = args['namespace'] + '/' + args['transformed_scan_topic']
    pub = rospy.Publisher(transformed_scan_msg, LaserScan, queue_size=1, latch=True)
    pub.publish(msg)

def pose_callback(msg, args):
    global agent_positions, pose_received
    id = args['id']
    if id not in agent_positions:
        agent_positions[id] = np.zeros((1, 2))
    agent_positions[id] = np.array([msg.transform.translation.x, msg.transform.translation.y])
    pose_received[id] = True

def publish_goal_markers(args):
    goals = np.array(args['goals'], dtype=float)
    markers = MarkerArray()
    for i in range(len(goals)):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = rospy.Time.now()
        marker.ns = 'goal_markers_' + str(i)
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.pose.position.x = goals[i][0]
        marker.pose.position.y = goals[i][1]
        marker.pose.position.z = 0.01
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        markers.markers.append(marker)
    pub = rospy.Publisher('goal_markers', MarkerArray, queue_size=1, latch=True)
    pub.publish(markers)

def create_path(id, pub, params, path):
    global agent_positions
    path.header.frame_id = params['world_frame_id']
    path.header.stamp = rospy.Time.now()
    pose = PoseStamped()
    pose.header.frame_id = params['world_frame_id']
    pose.header.stamp = rospy.Time.now()
    pose.pose.position.x = agent_positions[id][0]
    pose.pose.position.y = agent_positions[id][1]
    pose.pose.position.z = 0.0
    pose.pose.orientation.x = 0.0
    pose.pose.orientation.y = 0.0
    pose.pose.orientation.z = 0.0
    pose.pose.orientation.w = 1.0
    path.poses.append(pose)
    pub.publish(path)

def get_yaml_params(file_path):
    with open(file_path, 'r') as file:
        yaml_params = yaml.safe_load(file)
    return yaml_params

if __name__ == '__main__':
    rospy.init_node('publish_paths_goals', anonymous=True)
    params = get_yaml_params(rospy.get_param('params_file'))
    num_agents = len(params['agents'])
    pose_received = [False for i in range(num_agents)]
    pose_topic_lists = [params['agents'][i]['pose_topic'] for i in range(num_agents)]
    pose_sub = [rospy.Subscriber(pose_topic_lists[i], TransformStamped, pose_callback, {'id': i}) for i in range(num_agents)]
    print("Subscribed to pose topics: ", pose_topic_lists)
    path_pub = [rospy.Publisher(agent['namespace'] + '/path', Path, queue_size=10, latch=True) for agent in params['agents']]
    rate = rospy.Rate(10)
    paths = [Path() for i in range(num_agents)]

    while not rospy.is_shutdown():
        if not all(pose_received):
            continue
        publish_goal_markers(params)
        for i in range(num_agents):
            create_path(i, path_pub[i], params, paths[i])
        rate.sleep()
