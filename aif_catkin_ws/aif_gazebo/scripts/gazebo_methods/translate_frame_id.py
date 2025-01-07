#!/usr/bin/env python
import rospy, yaml
from sensor_msgs.msg import LaserScan
import subprocess
from visualization_msgs.msg import Marker, MarkerArray

def callback(msg, args):
    id = args['id']
    # print("Received Laser message from agent " + str(id))
    msg.header.frame_id = str(args['namespace']) + '/' + str(args['scan_frame_id'])
    transformed_scan_msg = args['namespace'] + '/' + args['transformed_scan_topic']
    pub = rospy.Publisher(transformed_scan_msg, LaserScan, queue_size=1,latch=True)
    pub.publish(msg)

def publish_goal_markers(args):
    goals = args['goals']
    markers = MarkerArray()
    for i in range(len(goals)):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = rospy.Time.now()
        marker.ns = 'goal_markers_' + str(i)
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.pose.position.x = goals[i][0]
        marker.pose.position.y = goals[i][1]
        marker.pose.position.z = 0.01
        markers.markers.append(marker)
    pub = rospy.Publisher('goal_markers', MarkerArray, queue_size=1, latch=True)
    pub.publish(markers)

def get_yaml_params(file_path):
    with open(file_path, 'r') as file:
        yaml_params = yaml.safe_load(file)
    return yaml_params

def transform_scan_topics(params):
    """Transform for the front_laser_base and front_laser topics"""
    for agent in params['agents']:
        ns0 = agent['namespace']
        command = [
            'rosrun', 'tf', 'static_transform_publisher',
            '0', '0', '0', '0', '0', '0',
            f'/{ns0}/front_laser_base', f'/{ns0}/front_laser', '100'
        ]
        subprocess.run(command)

if __name__ == '__main__':
    rospy.init_node('transform_frame', anonymous=True)
    params = get_yaml_params(rospy.get_param('params_file'))
    print(params['agents'][0])
    publish_goal_markers(params)
    subs = [rospy.Subscriber(agent['namespace'] + '/' + agent['scan_topic'], LaserScan, callback, (agent)) for agent in params['agents']]
    rate = rospy.Rate(100)
    rospy.spin()