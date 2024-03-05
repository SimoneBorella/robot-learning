#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float64,Float32MultiArray


class SensorNode:
    def __init__(self):
        rospy.init_node('sensor_node')
        self.pub = rospy.Publisher('noisy_measurement', Float64, queue_size=10)
        self.true_state_sub = rospy.Subscriber('true_state', Float32MultiArray, self.callback)

    def callback(self, msg):
        noisy_measurement = msg.data[0] + np.sqrt(0.05)*np.random.randn()
        self.pub.publish(Float64(noisy_measurement))
        rospy.loginfo('Sensor measurement : %f', noisy_measurement)

if __name__ == '__main__':
    try:
        sensor_node = SensorNode()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
