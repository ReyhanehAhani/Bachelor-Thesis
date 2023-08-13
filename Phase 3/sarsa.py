import numpy as np
import pandas as pd
import random

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
LR = 1e-3
GAMMA = 0.5

class Agent:
    def __init__(self, actions_names, state_features):
        self.actions_list = actions_names
        self.state_features_list = state_features 
        self.columns_q_table = actions_names + state_features

        self.q_table = None
        self.reset_q_table()
        self.steps_done = 0

    def reset_q_table(self):
        self.q_table = pd.DataFrame(columns=self.columns_q_table, dtype=np.float32)

    def select_action(self, observation):
        self.check_state_exist(observation)

        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        possible_actions = self.actions_list

        if np.random.uniform() > eps_threshold:
            state_action = self.q_table.loc[
                (self.q_table[self.state_features_list[0]] == observation[0])
                & (self.q_table[self.state_features_list[1]] == observation[1])
                # & (self.q_table[self.state_features_list[2]] == observation[2])
                ]

            state_action = state_action.filter(self.actions_list, axis=1)
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            state_action = state_action.filter(items=possible_actions)

            if state_action.empty:
                action = random.choice(possible_actions)
            else:
                action = state_action.idxmax(axis=1)

            action_to_do = action.iloc[0]
        else:
            action_to_do = np.random.choice(possible_actions)
        return action_to_do

    def learn(self, s, a, r, s_, a_, termination_flag, gamma=GAMMA, learning_rate=LR):
        self.check_state_exist(s_)

        id_row_previous_state = self.get_id_row_state(s)
        id_row_next_state = self.get_id_row_state(s_)
        q_predict = self.q_table.loc[id_row_previous_state, a]

        if termination_flag:
            q_target = r
        else:
            q_expected = self.q_table.loc[id_row_next_state, a_]
            q_target = r + gamma * q_expected

        self.q_table.loc[id_row_previous_state, a] += learning_rate * (q_target - q_predict)


    def check_state_exist(self, state):
        state_id_list_previous_state = self.q_table.index[(self.q_table[self.state_features_list[0]] == state[0]) &
                                                          (self.q_table[self.state_features_list[1]] ==
                                                           state[1])].tolist()

        if not state_id_list_previous_state:
            new_data = np.concatenate((np.array(len(self.actions_list)*[0]), np.array(state)), axis=0)
            new_row = pd.Series(new_data, index=self.q_table.columns)
            self.q_table = self.q_table.append(new_row, ignore_index=True)

    def get_id_row_state(self, s):
        id_list_state = self.q_table.index[(self.q_table[self.state_features_list[0]] == s[0]) &
                                           (self.q_table[self.state_features_list[1]] == s[1])].tolist()
        id_row_state = id_list_state[0]
        return id_row_state


from road_env import Road
from itertools import count

env = Road()
agent = Agent(list(env.actions.keys()), env.state_features)

num_episodes = 1000

rewards = []

for i_episode in range(num_episodes):
    state = env.reset()

    episode_rewards = []
    
    for t in count():
        action = agent.select_action(state)
        observation, reward, finished = env.step(action)

        next_state = observation

        next_action = agent.select_action(observation)

        agent.learn(state, action, reward, next_state, next_action, finished)
        
        episode_rewards.append(reward)
        state = next_state
        action = next_action

        if finished:
            break

    if i_episode % 10 == 0 and episode_rewards:
        print(f'Episode: {i_episode}, Reward: {sum(episode_rewards) / len(episode_rewards)}')
        rewards.append(sum(episode_rewards) / len(episode_rewards))

import matplotlib.pyplot as plt

plt.figure()
plt.title('Reward')
plt.plot(rewards)

plt.show()


