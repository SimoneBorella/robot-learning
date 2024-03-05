#!/usr/bin/env python3

import rospy
import numpy as np
from scipy.integrate import odeint
from std_msgs.msg import Float32MultiArray, Float32


class PendulumNode:
    def __init__(self):
        rospy.init_node('pendulum_node')

        self.pub = rospy.Publisher('true_state', Float32MultiArray, queue_size=10)
        self.pub_x1 = rospy.Publisher('x1', Float32, queue_size=10)
        self.pub_x2 = rospy.Publisher('x2', Float32, queue_size=10)
        self.pub_x3 = rospy.Publisher('x3', Float32, queue_size=10)
        self.pub_x4 = rospy.Publisher('x4', Float32, queue_size=10)

        self.rate = rospy.Rate(100)

        self.true_state = Float32MultiArray()
    
    def run(self):
        deltaTime = 0.01
        x0 = [np.pi / 3, 0.5, 0, 0]
        # x0 = [2*np.pi / 3, 0.5, np.pi / 2, 0]


        def state_space_model(x, t):
            g = 9.81
            l1 = 1
            l2 = 0.5
            m1 = 1
            m2 = 1

            dxdt = np.array(
                [
                    x[1],
                    (-g*(2*m1+m2)*np.sin(x[0]) - m2*g*np.sin(x[0]-2*x[2]) - 2*np.sin(x[0]-x[2])*m2*((x[3]**2)*l2 + (x[1]**2)*l1*np.cos(x[0]-x[2])))/(l1*(2*m1+m2-m2*np.cos(2*x[0]-2*x[2]))),
                    x[3],
                    (2*np.sin(x[0]-x[2])*((x[1]**2)*l1*(m1+m2) + g*(m1+m2)*np.cos(x[0]) + (x[3]**2)*l2*m2*np.cos(x[0]-x[2])))/(l2*(2*m1+m2-m2*np.cos(2*x[0]-2*x[2])))
                ]
            )
            return dxdt
        
        t = 0
        while not rospy.is_shutdown():
            self.true_state.data = odeint(state_space_model, x0, [t, t+deltaTime])[-1]
            x0 = self.true_state.data
            self.pub.publish(self.true_state)
            self.pub_x1.publish(self.true_state.data[0])
            self.pub_x2.publish(self.true_state.data[1])
            self.pub_x3.publish(self.true_state.data[2])
            self.pub_x4.publish(self.true_state.data[3])
            rospy.loginfo(f'Pendulum state: {self.true_state.data}')
            t += deltaTime
            self.rate.sleep()

if __name__ == '__main__':
    try:
        pendulum_node = PendulumNode()
        pendulum_node.run()
    except rospy.ROSInterruptException as e:
        pass