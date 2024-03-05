#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float32, Float64, Float32MultiArray
from ExtendedKalmanFilter import ExtendedKalmanFilter



class EKFNode:
    def __init__(self):
        rospy.init_node('ekf_node')
        self.ekf = self.get_ekf()
        self.noisy_measurement_sub = rospy.Subscriber('noisy_measurement', Float64, self.callback)
        self.true_state_sub = rospy.Subscriber('true_state', Float32MultiArray, self.true_state_callback)
        self.pub = rospy.Publisher('estimated_state', Float32MultiArray, queue_size=10)
        self.pub_x1_est = rospy.Publisher('x1_est', Float32, queue_size=10)
        self.pub_x2_est = rospy.Publisher('x2_est', Float32, queue_size=10)

        self.x1_err_pub = rospy.Publisher('x1_err', Float32, queue_size=10)
        self.x2_err_pub = rospy.Publisher('x2_err', Float32, queue_size=10)

        self.estimated_state = Float32MultiArray()


    def get_ekf(self):
        deltaTime = 0.01
        x0 = np.array([np.pi/3, 0.5])

        x_0_mean = np.zeros(shape=(2, 1))
        x_0_mean[0] = x0[0] + 3 * np.random.randn()
        x_0_mean[1] = x0[1] + 3 * np.random.randn()
        x_0_cov = 10 * np.eye(2, 2)

        Q = 0.00001 * np.eye(2, 2)
        R = np.array([[0.05]])

        ekf = ExtendedKalmanFilter(x_0_mean, x_0_cov, Q, R, deltaTime)
        return ekf

    def estimation(self, z_t):
        self.ekf.forwardDynamics()
        self.ekf.updateEstimate(z_t)
        x = np.array(self.ekf.posteriorMeans[-1]).T
        return x[0]
    
    def true_state_callback(self, msg):
        self.true_state = msg

    def callback(self, msg):
        self.estimated_state.data = self.estimation(msg.data)

        rospy.loginfo(f"Estimated state: {self.estimated_state.data}")

        self.pub_x1_est.publish(self.estimated_state.data[0])
        self.pub_x2_est.publish(self.estimated_state.data[1])
        self.pub.publish(self.estimated_state)


        x1_err = Float32(self.true_state.data[0] - self.estimated_state.data[0])
        x2_err = Float32(self.true_state.data[1] - self.estimated_state.data[1])
        self.x1_err_pub.publish(x1_err)
        self.x2_err_pub.publish(x2_err)



    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        ekf_node = EKFNode()
        ekf_node.run()
    except rospy.ROSInterruptException:
        pass
