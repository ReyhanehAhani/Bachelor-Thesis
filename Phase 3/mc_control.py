import numpy as np
import pandas as pd
import random
from collections import defaultdict

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000
LR = 1e-3
GAMMA = 0.5

class Agent:
    def __init__(self, actions_names, state_features):
        self.actions_list = actions_names
        self.actinos_n = len(actions_names)
        self.state_features_list = state_features 
        self.columns_q_table = actions_names + state_features

        self.q_table = None
        self.reset_q_table()
        self.steps_done = 0

    def reset_q_table(self):
        # ToDo: dtype=np.float32 not necessary. Try lower precision
        self.q_table = defaultdict(lambda: np.zeros(self.actinos_n))

    def select_action(self, observation):
        observation = tuple(observation)

        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        possible_actions = self.actions_list

        if np.random.uniform() > eps_threshold:
            state_action = (self.q_table[observation]).copy()
            
            for action in self.actions_list:
                if action not in possible_actions:
                    action_id = self.actions_list.index(action)
                    state_action[action_id] = -np.inf 

            if np.all(np.isneginf([state_action])):
                action_id = random.choice(possible_actions)
            else:
                action_id = np.argmax(state_action)
            action_to_do = self.actions_list[action_id]
        else:
            action_to_do = np.random.choice(possible_actions)

        return action_to_do

    def learn(self, episode, gamma=GAMMA, learning_rate=LR):
        states, actions, rewards = zip(*episode)
        discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])
        for i, state in enumerate(states):
            state = tuple(state)
            action_id = self.actions_list.index(actions[i])
            old_q = self.q_table[state][action_id]
            self.q_table[state][action_id] = old_q + learning_rate * (sum(rewards[i:] * discounts[:-(1 + i)]) - old_q)


from road_env import Road
from itertools import count

env = Road()
agent = Agent(list(env.actions.keys()), env.state_features)

num_episodes = 1000

rewards = []

for i_episode in range(num_episodes):
    episode_rewards = []
    episode = []
    state = env.reset()
    
    for t in count():
        action = agent.select_action(state)
        observation, reward, finished = env.step(action)
        
        episode.append((state, action, reward))
        episode_rewards.append(reward)
        
        state = observation

        if finished:
            break
    
    agent.learn(episode)

    if i_episode % 10 == 0 and episode_rewards:
        print(f'Episode: {i_episode}, Reward: {sum(episode_rewards) / len(episode_rewards)}')
        rewards.append(sum(episode_rewards) / len(episode_rewards))

import matplotlib.pyplot as plt

plt.figure()
plt.title('Reward')
plt.plot(rewards)

plt.show()


