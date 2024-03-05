#!/usr/bin/env python3

import rospy
import numpy as np
from scipy.integrate import odeint
from std_msgs.msg import Float32MultiArray, Int32, Float32


class PendulumNode:
    def __init__(self):
        rospy.init_node('pendulum_node')

        self.pub = rospy.Publisher('true_state', Float32MultiArray, queue_size=10)
        self.pub_x1 = rospy.Publisher('x1', Float32, queue_size=10)
        self.pub_x2 = rospy.Publisher('x2', Float32, queue_size=10)

        self.rate = rospy.Rate(100)

        self.true_state = Float32MultiArray()
    
    def run(self):
        deltaTime = 0.01
        x0 = [np.pi / 3, 0.5]

        def state_space_model(x, t):
            g = 9.81
            l = 1
            dxdt = np.array([x[1], -(g / l) * np.sin(x[0])])
            return dxdt
        
        t = 0
        while not rospy.is_shutdown():
            self.true_state.data = odeint(state_space_model, x0, [t, t+deltaTime])[-1]
            x0 = self.true_state.data
            self.pub.publish(self.true_state)
            self.pub_x1.publish(self.true_state.data[0])
            self.pub_x2.publish(self.true_state.data[1])
            rospy.loginfo(f'Pendulum state: {self.true_state.data}')
            t += deltaTime
            self.rate.sleep()

if __name__ == '__main__':
    try:
        pendulum_node = PendulumNode()
        pendulum_node.run()
    except rospy.ROSInterruptException as e:
        pass