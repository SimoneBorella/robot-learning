"""
    Robot Learning
    Exercise 2

    Linear Quadratic Regulator

    Polito A-Y 2023-2024
"""
import gym
import numpy as np
from scipy import linalg
import argparse
import matplotlib.pyplot as plt
import sys
from utils import get_space_dim, set_seed
import pdb 
import time


def plot_obs(obs_hist):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle("Cartpole state")

    axs[0][0].set_title("Position")
    axs[0][1].set_title("Velocity")
    axs[1][0].set_title("Angle")
    axs[1][1].set_title("Angular velocity")

    axs[0][0].set_xlabel("Timestep")
    axs[0][1].set_xlabel("Timestep")
    axs[1][0].set_xlabel("Timestep")
    axs[1][1].set_xlabel("Timestep")

    axs[0][0].set_ylabel("m")
    axs[0][1].set_ylabel("m/s")
    axs[1][0].set_ylabel("rad")
    axs[1][1].set_ylabel("rad/s")

    axs[0][0].plot([obs[0] for obs in obs_hist])
    axs[0][1].plot([obs[1] for obs in obs_hist])
    axs[1][0].plot([obs[2] for obs in obs_hist])
    axs[1][1].plot([obs[3] for obs in obs_hist])

    axs[0][0].plot([0.005 for _ in obs_hist], '--', color='red', alpha=0.5)
    axs[0][0].plot([-0.005 for _ in obs_hist], '--', color='red', alpha=0.5)
    axs[0][1].plot([0.005 for _ in obs_hist], '--', color='red', alpha=0.5)
    axs[0][1].plot([-0.005 for _ in obs_hist], '--', color='red', alpha=0.5)
    axs[1][0].plot([0.005 for _ in obs_hist], '--', color='red', alpha=0.5)
    axs[1][0].plot([-0.005 for _ in obs_hist], '--', color='red', alpha=0.5)
    axs[1][1].plot([0.005 for _ in obs_hist], '--', color='red', alpha=0.5)
    axs[1][1].plot([-0.005 for _ in obs_hist], '--', color='red', alpha=0.5)

    plt.show()


def plot_obs_R(obs_hist):
    Rs = list(obs_hist.keys())
    Rs.sort()

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("Cartpole state")

    axs[0][0].set_title("Position")
    axs[0][1].set_title("Velocity")
    axs[1][0].set_title("Angle")
    axs[1][1].set_title("Angular velocity")

    axs[0][0].set_xlabel("Timestep")
    axs[0][1].set_xlabel("Timestep")
    axs[1][0].set_xlabel("Timestep")
    axs[1][1].set_xlabel("Timestep")

    axs[0][0].set_ylabel("m")
    axs[0][1].set_ylabel("m/s")
    axs[1][0].set_ylabel("rad")
    axs[1][1].set_ylabel("rad/s")

    for R in Rs:
        axs[0][0].plot([obs[0] for obs in obs_hist[R]], label=f"{R=}")
        axs[0][1].plot([obs[1] for obs in obs_hist[R]], label=f"{R=}")
        axs[1][0].plot([obs[2] for obs in obs_hist[R]], label=f"{R=}")
        axs[1][1].plot([obs[3] for obs in obs_hist[R]], label=f"{R=}")

    axs[0][0].legend()
    axs[0][1].legend()
    axs[1][0].legend()
    axs[1][1].legend()

    plt.show()


def plot_forces_R(forces_hist):
    Rs = list(forces_hist.keys())
    Rs.sort()

    fig, axs = plt.subplots(len(Rs))
    fig.suptitle("Cartpole applied force")

    for i, R in enumerate(Rs):
        forces = forces_hist[R]
        if forces[0] > 0:
            forces = [-f for f in forces]

        axs[i].set_title(f"R = {R}")
        axs[i].set_ylabel("N")
        axs[i].plot(forces)
        
    plt.xlabel("Timestep")
    plt.show()

# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0",
                        help="Environment to use")
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument("--time_sleep", action='store_true',
                        help="Add timer for visualizing rendering with a slower frame rate")
    parser.add_argument("--mode", type=str, default="control",
                        help="Type of test ['control', 'multiple_R']")
    return parser.parse_args(args)

def linerized_cartpole_system(mp, mk, lp, g=9.81):
    mt=mp+mk
    a = -g/(lp*(mp/(mp+mk) - 4.0/3))
    # state matrix
    A = np.array([[0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, a, 0]])

    # input matrix
    b = 1/(lp*mt*(mp/mt - 4.0/3))
    B = np.array([[0], [1/mt], [0], [b]])
    return A, B


def optimal_controller(A, B, R_value=1):
    R = R_value*np.eye(1, dtype=int)  # choose R (weight for input)
    Q = 5*np.eye(4, dtype=int)        # choose Q (weight for state)
   # solve ricatti equation
    P = linalg.solve_continuous_are(A, B, Q, R)

    # calculate optimal controller gain
    K = np.dot(np.linalg.inv(R),
            np.dot(B.T, P))
    return K

def apply_state_controller(K, x):
    # feedback controller
    u = -np.dot(K, x)   # u = -Kx
    if u > 0:
        return 1, u     # if force_dem > 0 -> move cart right
    else:
        return 0, u     # if force_dem <= 0 -> move cart left

def multiple_R(env, mp, mk, l, g, time_sleep=False, terminate=True):

    R_values = [0.01, 0.1, 10, 100]

    forces_hist = {}
    obs_hist = {}

    for R in R_values:
        forces_hist[R] = []
        obs_hist[R] = []
        obs = env.reset()    # Reset the environment for a new episode
        
        A, B = linerized_cartpole_system(mp, mk, l, g)
        K = optimal_controller(A, B, R)    # Re-compute the optimal controller for the current R value

        time_steps = 400
        i = 0
        while True:
            # env.render()
            # if time_sleep:
            #     time.sleep(.1)
            
            # get force direction (action) and force value (force)
            action, force = apply_state_controller(K, obs)

            forces_hist[R].append(force)
            
            # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
            abs_force = abs(float(np.clip(force, -10, 10)))
            
            # change magnitute of the applied force in CartPole
            env.env.force_mag = abs_force

            # apply action
            obs, reward, done, _ = env.step(action)

            obs_hist[R].append(obs)
            
            # if terminate and done:
            if (time_steps is not None and i >= time_steps-1) or (time_steps is None and terminate and done):
                print(f'Terminated after {i+1} iterations.')
                break
            
            i+=1

    plot_obs_R(obs_hist)
    plot_forces_R(forces_hist)

def control(env, mp, mk, l, g, time_sleep=False, terminate=True):
    """
    Control using LQR
    """
    
    obs_hist = []

    obs = env.reset()    # Reset the environment for a new episode
    
    A, B = linerized_cartpole_system(mp, mk, l, g)
    K = optimal_controller(A, B)    # Re-compute the optimal controller for the current R value


    time_steps = 400
    i = 0
    while True:
        # env.render()
        # if time_sleep:
        #     time.sleep(.1)
        
        # get force direction (action) and force value (force)
        action, force = apply_state_controller(K, obs)
        
        # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
        abs_force = abs(float(np.clip(force, -10, 10)))
        
        # change magnitute of the applied force in CartPole
        env.env.force_mag = abs_force

        # apply action
        obs, reward, done, _ = env.step(action)

        # Plot
        obs_hist.append(obs)
        
        
        # if terminate and done:
        if (time_steps is not None and i >= time_steps-1) or (time_steps is None and terminate and done):
            print(f'Terminated after {i+1} iterations.')
            break
        
        i+=1

    plot_obs(obs_hist)

# The main function
def main(args):
    # Create a Gym environment with the argument CartPole-v0 (already embedded in)
    env = gym.make(args.env)

    # Get dimensionalities of actions and observations
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    # Print some stuff
    print("Environment:", args.env)
    print("Observation space dimensions:", observation_space_dim)
    print("Action space dimensions:", action_space_dim)

    set_seed(args.seed)    # seed for reproducibility
    env.env.seed(args.seed)
    
    mp, mk, l, g = env.masspole, env.masscart, env.length, env.gravity

    if args.mode == "control":
        control(env, mp, mk, l, g, args.time_sleep, terminate=True)
    elif args.mode == "multiple_R":
        multiple_R(env, mp, mk, l, g, args.time_sleep, terminate=True)

    env.close()

# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)

