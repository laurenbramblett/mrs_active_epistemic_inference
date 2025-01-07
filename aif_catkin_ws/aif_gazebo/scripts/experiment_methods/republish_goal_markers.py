#!/usr/bin/env python
from visualization_msgs.msg import Marker, MarkerArray
import rospy
def goal_marker_callback(msg):
    new_markers = MarkerArray()
    for i in range(len(msg.markers)):
        marker = Marker()
        marker.header.frame_id = 'vicon/world'
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
        marker.pose.position.x = msg.markers[i].pose.position.x
        marker.pose.position.y = msg.markers[i].pose.position.y
        marker.pose.position.z = 0.01
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        new_markers.markers.append(marker)
    pub = rospy.Publisher('goal_markers_repub', MarkerArray, queue_size=1, latch=True)
    pub.publish(new_markers)

if __name__=='__main__':
    rospy.init_node('lab_goal_publisher')
    rospy.Subscriber('/goal_markers', MarkerArray, goal_marker_callback)
    rospy.spin()