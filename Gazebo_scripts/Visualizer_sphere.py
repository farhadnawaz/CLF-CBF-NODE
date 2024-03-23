#! /usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker

rospy.init_node('Visualizer_sphere')

marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size = 10)

marker = Marker()

marker.header.frame_id = "panda_link0"
marker.header.stamp = rospy.Time.now()

# set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
marker.type = 2
marker.id = 0

r = 0.2
# Set the scale of the marker
marker.scale.x = r/1.732
marker.scale.y = r/1.732
marker.scale.z = r/1.732

# Set the color
marker.color.r = 1.0
marker.color.g = 0.0
marker.color.b = 0.0
marker.color.a = 1.0

# Set the pose of the marker
marker.pose.position.x = 0.55
marker.pose.position.y = 0.07
marker.pose.position.z = 0.04
marker.pose.orientation.x = 0.0
marker.pose.orientation.y = 0.0
marker.pose.orientation.z = 0.0
marker.pose.orientation.w = 1.0

while not rospy.is_shutdown():
  marker_pub.publish(marker)
  rospy.rostime.wallsleep(1.0)