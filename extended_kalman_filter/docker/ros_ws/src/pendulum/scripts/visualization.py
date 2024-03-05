#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Float64,Float32MultiArray
import numpy as np

class SphereAndLineVisualizer:
    def __init__(self):
        rospy.init_node('visualization_node', anonymous=True)

        self.true_state_sub = rospy.Subscriber('true_state', Float32MultiArray, self.true_callback)
        self.estimated_state_sub = rospy.Subscriber('estimated_state', Float32MultiArray, self.estimated_callback)
        self.marker_pub = rospy.Publisher('pendulum_marker', Marker, queue_size=10)


    def true_callback(self, msg):
        self.true_state = msg

        height = 1.2
        l = 1
        m = 1
        angle = msg.data[0]
        y = l*np.sin(angle)
        z = height - l*np.cos(angle)

        sphere = self.create_sphere_marker({'x': 0, 'y': y, 'z': z}, m, "sphere", a=0.8)
        link = self.create_link_marker({'x': 0, 'y': 0, 'z': height}, {'x': 0, 'y': y, 'z': z}, "link", a=0.8)

        self.marker_pub.publish(sphere)
        self.marker_pub.publish(link)
    
    def estimated_callback(self, msg):
        height = 1.2
        l = 1
        m = 1
        angle = msg.data[0]
        y = l*np.sin(angle)
        z = height - l*np.cos(angle)

        rgb_est = (39/255, 174/255, 96/255 )

        sphere_est = self.create_sphere_marker({'x': 0, 'y': y, 'z': z}, m, "sphere_est", rgb=rgb_est, a=0.4)
        link_est = self.create_link_marker({'x': 0, 'y': 0, 'z': height}, {'x': 0, 'y': y, 'z': z}, "link_est", a=0.4)

        self.marker_pub.publish(sphere_est)
        self.marker_pub.publish(link_est)

    def create_sphere_marker(self, p, m, ns, rgb=(1., 0., 0.), a=1.0):
        sphere = Marker()
        sphere.header.frame_id = "pendulum_frame"
        sphere.header.stamp = rospy.Time.now()
        sphere.ns = ns
        sphere.id = 0
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD

        # Sphere
        sphere.pose.position.x = p['x']
        sphere.pose.position.y = p['y']
        sphere.pose.position.z = p['z']
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = m/5
        sphere.scale.y = m/5
        sphere.scale.z = m/5
        sphere.color.a = a
        sphere.color.r = rgb[0]
        sphere.color.g = rgb[1]
        sphere.color.b = rgb[2]

        return sphere

    def create_link_marker(self, p1, p2, ns, rgb=(0., 0., 1.), a=1.0):
        link = Marker()
        link.header.frame_id = "pendulum_frame"
        link.header.stamp = rospy.Time.now()
        link.ns = ns
        link.id = 1
        link.type = Marker.LINE_STRIP
        link.action = Marker.ADD
        link.pose.orientation.w = 1.0
        link.scale.x = 0.05
        link.color.a = a
        link.color.r = rgb[0]
        link.color.g = rgb[1]
        link.color.b = rgb[2]

        # Points for the line
        point1 = Point()
        point1.x = p1["x"]
        point1.y = p1["y"]
        point1.z = p1["z"]

        point2 = Point()
        point2.x = p2["x"]
        point2.y = p2["y"]
        point2.z = p2["z"]

        link.points = [point1, point2]

        return link

if __name__ == '__main__':
    try:
        visualizer = SphereAndLineVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
