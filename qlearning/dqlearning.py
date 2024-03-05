import gym
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split


class DeepQNetwork(nn.Module):
    def __init__(self, lr, in_features, h1, h2, out_features):
        super().__init__()
        self.in_features = in_features
        self.h1 = h1
        self.h2 = h2
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.out(x)
        return actions


class Agent():
    def __init__(self, gamma, epsilon, lr, state_dim, batch_size, actions_dim, mem_size=100000, eps_end=0.01, eps_dec=8e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(actions_dim)]
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(self.lr, state_dim, 256, 256, actions_dim)

        self.state_memory = np.zeros((mem_size, state_dim), dtype=np.float32)
        self.new_state_memory = np.zeros((mem_size, state_dim), dtype=np.float32)

        self.action_memory = np.zeros(mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1
    



    def choose_action(self, observation, test=False):
        if test:
            if np.random.random() > self.epsilon:
                state = T.tensor([observation]).to(self.Q_eval.device)
                actions = self.Q_eval.forward(state)
                action = T.argmax(actions).item()
            else:
                action = np.random.choice(self.action_space)
        else:
            with torch.no_grad():
                state = T.tensor([observation]).to(self.Q_eval.device)
                actions = self.Q_eval(state)
                action = T.argmax(actions).item()
        
        return action
    


    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        if self.epsilon > self.eps_end:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.epsilon = self.eps_end

    def save_model(self, filename="model.pt"):
        torch.save(self.Q_eval.state_dict(), filename)
    
    def load_model(self, filename="model.pt"):
        self.Q_eval.load_state_dict(torch.load(filename))




def plot_reward_history(reward_hist, avg_reward_hist):
    plt.title(f"Reward history (CartPole-v0)")
    plt.plot(reward_hist)
    plt.plot(avg_reward_hist)
    plt.show()



def train(render=False):
    env = gym.make('CartPole-v0')

    agent = Agent(gamma=0.98, epsilon=1.0, lr=0.003, state_dim=4, batch_size=64, actions_dim=2)

    episodes = 300
    env._max_episode_steps = 400

    reward_history = []
    avg_reward_hist = []
    epsilon_history = []
    avg_window = 100

    best_reward = 0

    for ep in range(episodes):
        state = env.reset()
        reward_sum = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, reward, new_state, done)
            agent.learn()

            if render:
                env.render()

            state = new_state
            reward_sum += reward

        reward_history.append(reward_sum)
        if ep > avg_window:
            avg_reward = np.mean(reward_history[-100:])
        else:
            avg_reward = np.mean(reward_history)
        avg_reward_hist.append(avg_reward)
        epsilon_history.append(agent.epsilon)

        if reward_sum >= best_reward:
            best_reward = reward_sum
            agent.save_model("best_reward_model.pt")


        print(f"Episode {ep} \tReward {reward_sum:.2f} \tAvg_reward {avg_reward:.2f} \tEpsilon {agent.epsilon:.2f}")

    plot_reward_history(reward_history, avg_reward_hist)





def test(render = False):
    env = gym.make('CartPole-v0')

    agent = Agent(gamma=0.98, epsilon=0.0, lr=0.003, state_dim=4, batch_size=64, actions_dim=2)

    agent.load_model("best_reward_model.pt")

    episodes = 200
    env._max_episode_steps = 200

    reward_history = []
    avg_reward_hist = []
    avg_window = 100


    for ep in range(episodes):
        state = env.reset()
        reward_sum = 0
        done = False

        while not done:
            action = agent.choose_action(state, test=True)
            new_state, reward, done, info = env.step(action)

            if render:
                env.render()

            state = new_state
            reward_sum += reward

        reward_history.append(reward_sum)
        if ep > avg_window:
            avg_reward = np.mean(reward_history[-100:])
        else:
            avg_reward = np.mean(reward_history)
        avg_reward_hist.append(avg_reward)

        print(f"Episode {ep} \tReward {reward_sum:.2f} \tAvg_reward {avg_reward:.2f} \tEpsilon {agent.epsilon:.2f}")

    plot_reward_history(reward_history, avg_reward_hist)



if __name__ == "__main__":
    train()
    # test(render=True)
