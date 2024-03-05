"""
    Robot Learning
    Exercise 2

    Reinforcement Learning 

    Polito A-Y 2023-2024
"""
import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from agent import Agent, Policy
from utils import get_space_dim
from time import sleep

import sys


# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None,
                        help="Model to be tested")
    parser.add_argument("--env", type=str, default="CartPole-v0",
                        help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=500,
                        help="Number of episodes to train for")
    parser.add_argument("--episode_len", type=int, default=500,
                        help="Episodes length")
    parser.add_argument("--render_training", action='store_true',
                        help="Render each frame during training. Will be slower.")
    parser.add_argument("--test_episodes", type=int, default=500,
                        help="Number of episodes to train for")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    parser.add_argument("--central_point", type=float, default=0.0,
                        help="Point x0 to fluctuate around")
    parser.add_argument("--random_policy", action='store_true', help="Applying a random policy training")
    parser.add_argument("--plot_test", action='store_true', help="Plot state history for each test")
    return parser.parse_args(args)

def plot_obs(obs_hist):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle("Cart pole state")

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

    plt.show()



# Policy training function
def train(agent, env, train_episodes, early_stop=True, render=False,
          silent=False, train_run_id=0, x0=0, random_policy=False):
    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []

    max_reward_sum = 0

    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state (it's a random initial state with small values)
        observation = env.reset()
        side = 1

        # Loop until the episode is over
        while not done:
            # Get action from the agent
            action, action_probabilities = agent.get_action(observation)

            if random_policy:
                action = np.random.choice([0, 1])

            previous_observation = observation

            # Perform the action on the environment, get new state and reward
            # note that after env._max_episode_steps the episode is over, if we stay alive that long
            observation, reward, done, info = env.step(action)


            # Reward funtion to x0
            # reward = reward_to_x0(observation, x0)


            # Reward function right side - left side - loop
            if side == 1 and observation[0] > 1.9:
                side = -1
            elif side == -1 and observation[0] < -1.9:
                side = 1 
            reward = reward_max_vel(observation, side)


            # Store action's outcome (so that the agent can improve its policy)
            agent.store_outcome(previous_observation, action_probabilities, action, reward)

            # Draw the frame, if desired
            if render:
                env.render()

            # Store total episode reward
            reward_sum += reward
            timesteps += 1

        if not silent:
            print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  .format(episode_number, reward_sum, timesteps))

        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        # If we managed to stay alive for 15 full episodes, assume it's learned
        # (in the default setting)
        if early_stop and np.mean(timestep_history[-15:]) == env._max_episode_steps:
            if not silent:
                print("Looks like it's learned. Finishing up early")
            break

        # Let the agent do its magic (update the policy)
        agent.episode_finished(episode_number)

        # Save best model
        if reward_sum > max_reward_sum:
            torch.save(agent.policy.state_dict(), "model_params.ai")
            print("BEST")
            print("Model saved to model_params.ai")
            max_reward_sum = reward_sum


    # Store the data in a Pandas dataframe for easy visualization
    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id]*len(reward_history),
                         "reward": reward_history,
                         "mean_reward": average_reward_history})
    return data


# Function to test a trained policy
def test(agent, env, episodes, render=False, x0=0, plot=False):
    test_reward, test_len = 0, 0

    print('Num testing episodes:', episodes)

    for ep in range(episodes):
        done = False
        observation = env.reset()
        side = 1

        obs_hist = []

        while not done:
            action, _ = agent.get_action(observation, evaluation=True)  # Similar to the training loop above -
                                                                        # get the action, act on the environment, save total reward
                                                                        # (evaluation=True makes the agent always return what it thinks to be
                                                                        # the best action - there is no exploration at this point)
            observation, reward, done, info = env.step(action)

            # Plot
            obs_hist.append(observation)


            # Reward funtion to x0
            # reward = reward_to_x0(observation, x0)


            # Reward function right side - left side - loop
            if side == 1 and observation[0] > 1.9:
                side = -1
            elif side == -1 and observation[0] < -1.9:
                side = 1 
            reward = reward_max_vel(observation, side)

            if render:
                env.render()
                # sleep(0.001)
            test_reward += reward
            test_len += 1
        
        if plot:
            plot_obs(obs_hist)

    print("Average test reward:", test_reward/episodes, "episode length:", test_len/episodes)

    

def reward_to_x0(obs, x0):
    pos = obs[0]
    reward = 1/(abs(pos-x0) + 1)
    return reward



def reward_max_vel(obs, side):
    vel = obs[1]
    angle = obs[2]
    angular_vel = obs[3]
    
    pos_factor = 0
    vel_factor = abs(vel)
    angle_factor = 1/(abs(angle) + 1)
    angular_vel_factor = 1/(abs(angular_vel) + 1)

    if (side == -1 and vel < 0) or (side == 1 and vel > 0):
        pos_factor = 1

    reward = 8 * pos_factor + 4* vel_factor + 2 * angular_vel_factor + 0.5 * angle_factor   
    
    return reward




# The main function
def main(args):
    # Create a Gym environment with the argument CartPole-v0 (already embedded in)
    env = gym.make(args.env)

    env._max_episode_steps = args.episode_len

    # Get dimensionalities of actions and observations
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy)

    # Print some stuff
    print("Environment:", args.env)
    print("Training device:", agent.train_device)
    print("Observation space dimensions:", observation_space_dim)
    print("Action space dimensions:", action_space_dim)

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.
    if args.test is None:
        # Train
        training_history = train(agent, env, args.train_episodes, False, args.render_training, x0=args.central_point, random_policy=args.random_policy)

        # Save the model
        model_file = "%s_params.ai" % args.env
        torch.save(policy.state_dict(), model_file)
        print("Model saved to", model_file)

        # Plot rewards
        sns.lineplot(x="episode", y="reward", data=training_history, color='blue', label='Reward')
        sns.lineplot(x="episode", y="mean_reward", data=training_history, color='orange', label='100-episode average')
        plt.legend()
        plt.title("Reward history (%s)" % args.env)
        plt.show()
        print("Training finished.")
    else:
        # Test
        print("Loading model from", args.test, "...")
        state_dict = torch.load(args.test)
        policy.load_state_dict(state_dict)
        print("Testing...")
        test(agent, env, args.test_episodes, args.render_test, x0=args.central_point, plot=args.plot_test)


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)

