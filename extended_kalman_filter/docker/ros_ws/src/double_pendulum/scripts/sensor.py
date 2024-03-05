#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
from pprint import pprint


class SensorNode:
    def __init__(self):
        rospy.init_node('sensor_node')
        self.pub = rospy.Publisher('noisy_measurement', Float32MultiArray, queue_size=10)
        self.true_state_sub = rospy.Subscriber('true_state', Float32MultiArray, self.callback)

        self.noisy_measurement = Float32MultiArray()

    def callback(self, msg):
        self.noisy_measurement.data = [
            msg.data[0] + np.sqrt(0.05)*np.random.randn(),
            msg.data[2] + np.sqrt(0.05)*np.random.randn()
        ]
        self.pub.publish(self.noisy_measurement)
        # rospy.loginfo(f'Sensor measurement : {self.noisy_measurement.data}')

if __name__ == '__main__':
    try:
        sensor_node = SensorNode()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
