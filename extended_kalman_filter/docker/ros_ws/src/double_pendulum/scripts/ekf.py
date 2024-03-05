#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float32, Float32MultiArray
from ExtendedKalmanFilter import ExtendedKalmanFilter



class EKFNode:
    def __init__(self):
        rospy.init_node('ekf_node')
        self.ekf = self.get_ekf()
        self.noisy_measurement_sub = rospy.Subscriber('noisy_measurement', Float32MultiArray, self.callback)
        self.pub = rospy.Publisher('estimated_state', Float32MultiArray, queue_size=10)
        self.pub_x1_est = rospy.Publisher('x1_est', Float32, queue_size=10)
        self.pub_x2_est = rospy.Publisher('x2_est', Float32, queue_size=10)
        self.pub_x3_est = rospy.Publisher('x3_est', Float32, queue_size=10)
        self.pub_x4_est = rospy.Publisher('x4_est', Float32, queue_size=10)

        self.estimated_state = Float32MultiArray()


    def get_ekf(self):
        deltaTime = 0.01
        x0 = [np.pi / 3, 0.5, 0, 0]
        # x0 = [2*np.pi / 3, 0.5, np.pi / 2, 0]

        x_0_mean = np.zeros(shape=(4, 1))
        x_0_mean[0] = x0[0] + 3 * np.random.randn()
        x_0_mean[1] = x0[1] + 3 * np.random.randn()
        x_0_mean[2] = x0[2] + 3 * np.random.randn()
        x_0_mean[3] = x0[3] + 3 * np.random.randn()
        x_0_cov = 10 * np.eye(4, 4)

        Q = 0.00001 * np.eye(4, 4)
        R = 0.05 * np.eye(2, 2)
        
        ekf = ExtendedKalmanFilter(x_0_mean, x_0_cov, Q, R, deltaTime)
        return ekf

    def estimation(self, z_t):
        self.ekf.forwardDynamics()
        self.ekf.updateEstimate(z_t)
        x = np.array(self.ekf.posteriorMeans[-1]).T
        return x[0]

    def callback(self, msg):
        self.estimated_state.data = self.estimation(np.array(msg.data).reshape((2,1)))

        rospy.loginfo(f"Estimated state: {self.estimated_state.data}")

        self.pub_x1_est.publish(self.estimated_state.data[0])
        self.pub_x2_est.publish(self.estimated_state.data[1])
        self.pub_x3_est.publish(self.estimated_state.data[2])
        self.pub_x4_est.publish(self.estimated_state.data[3])
        self.pub.publish(self.estimated_state)


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        ekf_node = EKFNode()
        ekf_node.run()
    except rospy.ROSInterruptException:
        pass
