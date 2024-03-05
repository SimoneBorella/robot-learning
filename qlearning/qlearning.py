import gym
import numpy as np
from matplotlib import pyplot as plt
from time import sleep
import random
import seaborn as sns
import pandas as pd

MODE = 'TRAINING' # TRAINING, TESTING

def plot_reward_history(reward_history, average_reward_history):
    plt.title(f"Reward history")
    plt.plot(reward_history)
    plt.plot(average_reward_history)
    plt.show()

def plot_epsilon_history(epsilon_history):
    plt.title(f"Epsilon history")
    plt.plot(epsilon_history)
    plt.show()


def plot_heatmap(value_function):
    plt.title("Value function in therms of x and th")
    data = np.mean(value_function, axis=(1, 3))
    sns.heatmap(data, annot=True, cbar=True)
    plt.show()


def find_nearest(array, value):
    return np.argmin(np.abs(array - value))

def get_cell_index(state):
    """Returns discrete state from continuous state"""
    x = find_nearest(x_grid, state[0])
    v = find_nearest(v_grid, state[1])
    th = find_nearest(th_grid, state[2])
    av = find_nearest(av_grid, state[3])
    return x, v, th, av


# def get_action(state, q_values, epsilon, greedy=False):
#     x, v, th, av = get_cell_index(state)

#     if greedy: # TEST -> greedy policy
#         best_action_estimated = np.argmax(q_values[x, v, th, av])  # TODO: greedy w.r.t. q_grid

#         return best_action_estimated

#     else: # TRAINING -> epsilon-greedy policy
#         if np.random.rand() < epsilon:
#             # Random action
#             action_chosen = np.random.choice([0, 1])  # TODO: choose random action with equal probability among all actions
#             return action_chosen
#         else:
#             # Greedy action
#             best_action_estimated = np.argmax(q_values[x, v, th, av])  # TODO: greedy w.r.t. q_grid

#             return best_action_estimated

def get_action(state, q_values, epsilon):
    x, v, th, av = get_cell_index(state)

    if np.random.rand() < epsilon:
        # Random action
        action_chosen = np.random.choice([0, 1])  # TODO: choose random action with equal probability among all actions
        return action_chosen
    else:
        # Greedy action
        best_action_estimated = np.argmax(q_values[x, v, th, av])  # TODO: greedy w.r.t. q_grid

        return best_action_estimated


def update_q_value(old_state, action, new_state, reward, done, q_array):
    old_cell_index = get_cell_index(old_state)
    new_cell_index = get_cell_index(new_state)

    # Target value used for updating our current Q-function estimate at Q(old_state, action)
    if done is True:
        target_value = reward  # HINT: if the episode is finished, there is not next_state. Hence, the target value is simply the current reward.
    else:
        target_value = reward + gamma*max(q_array[new_cell_index])  # TODO

    # Update Q value
    q_value_old = q_array[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action]
    q_array[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action] = q_value_old + alpha*(target_value - q_value_old)  # TODO

    return









def train(q_grid , env, train_episodes, render=False):
    best_reward = 0

    reward_history = []
    average_reward_history = []
    epsilon_history = []

    for ep in range(train_episodes):

        state, done, steps = env.reset(), False, 0
        reward_sum = 0

        # TODO: change to GLIE schedule (task 3.1) or 0 (task 3.3)
        # epsilon = constant_eps

        # b = 2222.222222
        # epsilon = b/(b+ep)

        epsilon = 0.

        epsilon_history.append(epsilon)

        while not done:
            action = get_action(state, q_grid, epsilon)
            new_state, reward, done, _ = env.step(action)
            reward_sum += reward

            update_q_value(state, action, new_state, reward, done, q_grid)

            state = new_state
            steps += 1

        if reward_sum >= best_reward:
            best_reward = reward_sum
            np.save("q_values_best.npy", q_grid)

        reward_history.append(reward_sum)
        if ep > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        print("Episode {} finished. Total reward: {:.3g} ({} timesteps)".format(ep, reward_sum, steps))

    np.save("q_values.npy", q_grid)

    return reward_history, average_reward_history, epsilon_history




def test(q_grid , env, test_episodes, render=False):
    reward_history = []
    average_reward_history = []

    for ep in range(test_episodes):

        state, done, steps = env.reset(), False, 0
        reward_sum = 0

        # TODO: change to GLIE schedule (task 3.1) or 0 (task 3.3)
        # epsilon = constant_eps
        epsilon = 0


        while not done:
            # action = get_action(state, q_grid, epsilon, greedy=True)
            action = get_action(state, q_grid, epsilon)
            new_state, reward, done, _ = env.step(action)
            reward_sum += reward

            if render:
                env.render()

            state = new_state
            steps += 1

        reward_history.append(reward_sum)
        if ep > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        print("Episode {} finished. Total reward: {:.3g} ({} timesteps)".format(ep, reward_sum, steps))

    print(f"Average reward {np.mean(reward_history)}")

    return reward_history, average_reward_history





np.random.seed(123)

env = gym.make('CartPole-v0')
env.seed(321)

train_episodes = 20000
test_episodes = 100
num_of_actions = 2  # 2 discrete actions for Cartpole

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

# Parameters
gamma = 0.98
alpha = 0.1
constant_eps = 0.2


# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)

# Initialize Q values
# q_grid = np.zeros((discr, discr, discr, discr, num_of_actions))
q_grid = 50 * np.ones((discr, discr, discr, discr, num_of_actions))

if MODE == "TRAINING":
    env._max_episode_steps = 500
    reward_history, average_reward_history, epsilon_hist = train(q_grid, env, train_episodes)
    plot_epsilon_history(epsilon_hist)
elif MODE == "TESTING":
    env._max_episode_steps = 200
    q_grid = np.load('q_values_best.npy')
    reward_history, average_reward_history = test(q_grid, env, test_episodes, render=False)

plot_reward_history(reward_history, average_reward_history)
value_function = np.max(q_grid, axis=(4))
plot_heatmap(value_function)
