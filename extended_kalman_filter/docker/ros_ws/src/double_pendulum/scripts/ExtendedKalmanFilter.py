"""
    Implementation of the Extended Kalman Filter
    for an unactuated pendulum system
"""
import numpy as np
import sympy as sp
from scipy.linalg import expm

class ExtendedKalmanFilter(object):
    def __init__(self, x0, P0, Q, R, dT):
        """
           Initialize EKF
            
            Parameters
            x0 - mean of initial state prior
            P0 - covariance of initial state prior
            Q  - covariance matrix of the process noise 
            R  - covariance matrix of the measurement noise
            dT - discretization step for forward dynamics
        """

        self.x0=x0
        self.P0=P0
        self.Q=Q
        self.R=R
        self.dT=dT        
        
        self.g = 9.81
        self.l1 = 1
        self.l2 = 0.5 
        self.m1 = 1
        self.m2 = 1
        
        self.currentTimeStep = 0

        self.priorMeans = []
        self.priorMeans.append(None)
        self.posteriorMeans = []
        self.posteriorMeans.append(x0)
        
        self.priorCovariances=[]
        self.priorCovariances.append(None)
        self.posteriorCovariances=[]
        self.posteriorCovariances.append(P0)
    

    def stateSpaceModel(self, x, t):
        """
            Dynamics may be described as a system of first-order
            differential equations: 
            dx(t)/dt = f(t, x(t))

            Dynamics are time-invariant in our case, so t is not used.
            
            Parameters:
                x : state variables (column-vector)
                t : time

            Returns:
                f : dx(t)/dt, describes the system of ODEs
        """

        g = self.g
        l1 = self.l1
        l2 = self.l2
        m1 = self.m1
        m2 = self.m2
        
        dxdt = np.array(
            [
                x[1],
                (-g*(2*m1+m2)*np.sin(x[0]) - m2*g*np.sin(x[0]-2*x[2]) - 2*np.sin(x[0]-x[2])*m2*((x[3]**2)*l2 + (x[1]**2)*l1*np.cos(x[0]-x[2])))/(l1*(2*m1+m2-m2*np.cos(2*x[0]-2*x[2]))),
                x[3],
                (2*np.sin(x[0]-x[2])*((x[1]**2)*l1*(m1+m2) + g*(m1+m2)*np.cos(x[0]) + (x[3]**2)*l2*m2*np.cos(x[0]-x[2])))/(l2*(2*m1+m2-m2*np.cos(2*x[0]-2*x[2])))
            ]
        )
        return dxdt
    

    def discreteTimeDynamics(self, x_t):
        """
            Forward Euler integration.
            
            returns next state as x_t+1 = x_t + dT * (dx/dt)|_{x_t}
        """

        x_tp1 = x_t + self.dT*self.stateSpaceModel(x_t, None)
        return x_tp1
    

    def jacobianStateEquation(self, x_t):
        """
            Jacobian of discrete dynamics w.r.t. the state variables,
            evaluated at x_t

            Parameters:
                x_t : state variables (column-vector)
        """

        x1 = x_t[0][-1]
        x2 = x_t[1][-1]
        x3 = x_t[2][-1]
        x4 = x_t[3][-1]
        m1 = self.m1
        m2 = self.m2
        g = self.g
        l1 = self.l1
        l2 = self.l2
        
        A = np.zeros(shape=(4,4))

        A[0][0] = 1
        A[0][1] = self.dT
        A[0][2] = 0
        A[0][3] = 0
        A[1][0] = self.dT * ((2*m2*(g*m2*np.sin(x1 - 2*x3) + g*(2*m1 + m2)*np.sin(x1) + 2*m2*(l1*x2**2*np.cos(x1 - x3) + l2*x4**2)*np.sin(x1 - x3))*np.sin(2*x1 - 2*x3) + (2*m1 - m2*np.cos(2*x1 - 2*x3) + m2)*(-g*m2*np.cos(x1 - 2*x3) - g*(2*m1 + m2)*np.cos(x1) + 2*l1*m2*x2**2*np.sin(x1 - x3)**2 - 2*m2*(l1*x2**2*np.cos(x1 - x3) + l2*x4**2)*np.cos(x1 - x3)))/(l1*(2*m1 - m2*np.cos(2*x1 - 2*x3) + m2)**2))
        A[1][1] = 1 + self.dT * (-2*m2*x2*np.sin(2*x1 - 2*x3)/(2*m1 - m2*np.cos(2*x1 - 2*x3) + m2))
        A[1][2] = self.dT * (2*m2*((2*m1 - m2*np.cos(2*x1 - 2*x3) + m2)*(g*np.cos(x1 - 2*x3) - l1*x2**2*np.sin(x1 - x3)**2 + (l1*x2**2*np.cos(x1 - x3) + l2*x4**2)*np.cos(x1 - x3)) - (g*m2*np.sin(x1 - 2*x3) + g*(2*m1 + m2)*np.sin(x1) + 2*m2*(l1*x2**2*np.cos(x1 - x3) + l2*x4**2)*np.sin(x1 - x3))*np.sin(2*x1 - 2*x3))/(l1*(2*m1 - m2*np.cos(2*x1 - 2*x3) + m2)**2))
        A[1][3] = self.dT * (-4*l2*m2*x4*np.sin(x1 - x3)/(l1*(2*m1 - m2*np.cos(2*x1 - 2*x3) + m2)))
        A[2][0] = 0
        A[2][1] = 0
        A[2][2] = 1
        A[2][3] = self.dT
        A[3][0] = self.dT * ((-4*m2*(g*(m1 + m2)*np.cos(x1) + l1*x2**2*(m1 + m2) + l2*m2*x4**2*np.cos(x1 - x3))*np.sin(x1 - x3)*np.sin(2*x1 - 2*x3) + 2*(-(g*(m1 + m2)*np.sin(x1) + l2*m2*x4**2*np.sin(x1 - x3))*np.sin(x1 - x3) + (g*(m1 + m2)*np.cos(x1) + l1*x2**2*(m1 + m2) + l2*m2*x4**2*np.cos(x1 - x3))*np.cos(x1 - x3))*(2*m1 - m2*np.cos(2*x1 - 2*x3) + m2))/(l2*(2*m1 - m2*np.cos(2*x1 - 2*x3) + m2)**2))
        A[3][1] = self.dT * (4*l1*x2*(m1 + m2)*np.sin(x1 - x3)/(l2*(2*m1 - m2*np.cos(2*x1 - 2*x3) + m2)))
        A[3][2] = self.dT * ((2*l2*m2*x4**2*(2*m1 - m2*np.cos(2*x1 - 2*x3) + m2)*np.sin(x1 - x3)**2 + 4*m2*(g*(m1 + m2)*np.cos(x1) + l1*x2**2*(m1 + m2) + l2*m2*x4**2*np.cos(x1 - x3))*np.sin(x1 - x3)*np.sin(2*x1 - 2*x3) - 2*(2*m1 - m2*np.cos(2*x1 - 2*x3) + m2)*(g*(m1 + m2)*np.cos(x1) + l1*x2**2*(m1 + m2) + l2*m2*x4**2*np.cos(x1 - x3))*np.cos(x1 - x3))/(l2*(2*m1 - m2*np.cos(2*x1 - 2*x3) + m2)**2))
        A[3][3] = 1 + self.dT * (2*m2*x4*np.sin(2*x1 - 2*x3)/(2*m1 - m2*np.cos(2*x1 - 2*x3) + m2))

        return A
    
    
    def jacobianMeasurementEquation(self, x_t):
        """
            Jacobian of measurement model.

            Measurement model is linear, hence its Jacobian
            does not actually depend on x_t
        """

        C = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])

        return C
    
     
    def forwardDynamics(self):
        self.currentTimeStep = self.currentTimeStep+1

        """
            Predict the new prior mean for timestep t
        """
        # x_t_prior_mean = self.discreteTimeDynamics(self.posteriorMeans[self.currentTimeStep-1])
        x_t_prior_mean = self.discreteTimeDynamics(self.posteriorMeans[self.currentTimeStep-1])
        
        """
            Predict the new prior covariance for timestep t
        """
        # Linearization: jacobian of the dynamics at the current a posteriori estimate
        A_t_minus = self.jacobianStateEquation(self.posteriorMeans[self.currentTimeStep-1])

        x_t_prior_cov = A_t_minus @ self.posteriorCovariances[self.currentTimeStep-1] @ A_t_minus.T + self.Q
        
        # Save values
        self.priorMeans.append(x_t_prior_mean)
        self.priorCovariances.append(x_t_prior_cov)
    

    def updateEstimate(self, z_t):
        """
            Compute Posterior Gaussian distribution,
            given the new measurement z_t
        """

        # Jacobian of measurement model at x_t
        Ct = self.jacobianMeasurementEquation(self.priorMeans[self.currentTimeStep]) 
        
        sigma_t = self.priorCovariances[self.currentTimeStep]
        K_t = sigma_t@Ct.T @ np.linalg.inv(Ct@sigma_t@Ct.T+self.R)
        
        x_pt = self.priorMeans[self.currentTimeStep]
        x_t_mean = x_pt + K_t @ (z_t - Ct @ x_pt)
        
        x_t_cov = (np.eye(sigma_t.shape[0]) - K_t @ Ct) @ sigma_t
        
        # Save values

        self.posteriorMeans.append(x_t_mean)
        self.posteriorCovariances.append(x_t_cov)
