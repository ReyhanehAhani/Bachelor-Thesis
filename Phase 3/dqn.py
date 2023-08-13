import torch
import torch.nn as nn
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=128, fc2_units=128):
        super(DQN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_size, fc1_units, bias=True),
            nn.ReLU(),
            
            nn.Linear(fc1_units, fc2_units, bias=True),
            nn.ReLU(),
            
            nn.Linear(fc2_units, action_size, bias=True),
        )

    def forward(self, state):
        return self.model(state)


import dataclasses
import random
from collections import deque

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 128
GAMMA = 0.5
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.0005
LR = 1e-3
MEMORY_SIZE = 100000

class Agent:
    def __init__(self, actions, states):
        self.max_reward = -torch.inf
        self.min_loss = torch.inf

        self.actions = actions
        self.action_ids = [i for i, _ in enumerate(self.actions)]
        self.states = states 
        
        self.policy_net = DQN(len(self.states), len(self.actions)).to(device)
        self.target_net = DQN(len(self.states), len(self.actions)).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0
    
    def push(self, state, action, reward, next_state, finished):
        self.memory.push(state, action, reward, next_state, finished)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            self.policy_net.eval()
            with torch.no_grad():
                return self.policy_net(torch.tensor(state, dtype=torch.float32)).max(1)[1].view(1, 1)
        else:
            return torch.tensor([random.sample(self.action_ids, 1)], device=device, dtype=torch.long)

    def optimize_model(self, gamma=GAMMA):
        if len(self.memory) < BATCH_SIZE:
            return
        
        self.policy_net.train()
        self.target_net.train()

        transitions = self.memory.sample(BATCH_SIZE)
        batch = tuple(zip(*transitions))

        state_batch = torch.tensor(batch[0], device=device).float().squeeze(1)
        action_batch = torch.tensor(batch[1], device=device).long().squeeze(-1)
        reward_batch = torch.tensor(batch[2], device=device).float().squeeze(-1)
        next_states_batch = torch.tensor(batch[3], device=device).float().squeeze(1)
        finished_batch = torch.tensor(batch[4], device=device).float()

        q_targets_next = self.target_net(next_states_batch).detach().max(1)[0]
        q_targets = reward_batch + (gamma * q_targets_next * (1 - finished_batch))

        q_expected = self.policy_net(state_batch).gather(1, action_batch)

        criterion = nn.SmoothL1Loss()

        loss = criterion(q_expected, q_targets.unsqueeze(1))
        if torch.max(reward_batch) > self.max_reward:
            self.max_reward = torch.max(reward_batch)
        if self.min_loss > loss:
            self.min_loss = loss

        if self.steps_done % 100 == 0:
            print(f'{self.steps_done}\tLowest loss: {self.min_loss}\tMax Reward: {self.max_reward}')

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        # Soft update
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)

        return loss.item(), torch.mean(reward_batch).item()

from road_env import Road
import torch
from itertools import count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = Road(use_tkinter=True)
agent = Agent(tuple(env.actions.keys()), env.state_features)

num_episodes = 1000

losses = []
rewards = []

for i_episode in range(num_episodes):

    #print(f'Episode {i_episode} started\n\n')

    # Initialize the environment and get it's state
    episode_losses = []
    episode_rewards = []

    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    env_actions = tuple(env.actions.keys())
    for t in count():
        action = agent.select_action(state)
        observation, reward, finished = env.step(env_actions[action.item()])
        reward = torch.tensor([reward], device=device)

        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        agent.push(state.tolist(), action.tolist(), reward.tolist(), next_state.tolist(), finished)

        state = next_state

        r = agent.optimize_model()
        if r:
            episode_losses.append(r[0])
            episode_rewards.append(r[1])

        if finished:
            break

    if episode_rewards:
        losses.append(sum(episode_losses) / len(episode_losses))
        rewards.append(sum(episode_rewards) / len(episode_rewards))

import matplotlib.pyplot as plt

plt.figure()
plt.title('Loss')
plt.plot(losses)

plt.figure()
plt.title('Reward')
plt.plot(rewards)

plt.show()

