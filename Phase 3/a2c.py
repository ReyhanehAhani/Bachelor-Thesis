from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim

class RolloutStorage(object):
    def __init__(self, rollout_size, num_envs, state_size, is_cuda=True, value_coeff=0.5, entropy_coeff=0.02):
        super().__init__()

        self.rollout_size = rollout_size
        self.num_envs = num_envs
        self.is_cuda = is_cuda
        self.state_size = state_size

        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff

        # initialize the buffers with zeros
        self.reset_buffers()

    def _generate_buffer(self, size):
        """
        Generates a `torch.zeros` tensor with the specified size.

        :param size: size of the tensor (tuple)
        :return:  tensor filled with zeros of 'size'
                    on the device specified by self.is_cuda
        """
        if self.is_cuda:
            return torch.zeros(size).cuda()
        else:
            return torch.zeros(size)

    def reset_buffers(self):
        self.rewards = self._generate_buffer((self.rollout_size, self.num_envs))
        self.states = self._generate_buffer((self.rollout_size + 1, self.num_envs, self.state_size))
        self.actions = self._generate_buffer((self.rollout_size, self.num_envs))
        self.log_probs = self._generate_buffer((self.rollout_size, self.num_envs))
        self.values = self._generate_buffer((self.rollout_size, self.num_envs))
        self.dones = self._generate_buffer((self.rollout_size, self.num_envs))

    def after_update(self):
        self.states = self._generate_buffer((self.rollout_size + 1, self.num_envs, self.state_size))
        self.actions = self._generate_buffer((self.rollout_size, self.num_envs))
        self.log_probs = self._generate_buffer((self.rollout_size, self.num_envs))
        self.values = self._generate_buffer((self.rollout_size, self.num_envs))

    def get_state(self, step):
        return self.states[step].clone()

    def insert(self, step, reward, obs, action, log_prob, value, dones):
        self.rewards[step].copy_(torch.tensor(reward).clone())
        self.states[step + 1].copy_(torch.tensor(obs).clone())
        self.actions[step].copy_(torch.tensor(action).clone())
        self.log_probs[step].copy_(torch.tensor(log_prob).clone())
        self.values[step].copy_(torch.tensor(value).clone())
        self.dones[step].copy_(torch.tensor(dones).clone())

    def _discount_rewards(self, final_value, discount=0.99):
        r_discounted = self._generate_buffer((self.rollout_size, self.num_envs))
        R = self._generate_buffer(self.num_envs).masked_scatter((1 - self.dones[-1]), final_value)

        for i in reversed(range(self.rollout_size)):
            R = self._generate_buffer(self.num_envs).masked_scatter((1 - self.dones[-1]),
                                                                    self.rewards[i] + discount * R)
            r_discounted[i] = R
        return r_discounted

    def a2c_loss(self, final_value, entropy):
        rewards = self._discount_rewards(final_value)
        advantage = rewards - self.values
        policy_loss = (-self.log_probs * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy
        return loss

class A2CNet(nn.Module):
    def __init__(self, num_actions, in_size):
        super().__init__()

        self.in_size = in_size
        self.num_actions = num_actions

        h = 64

        self.actor = nn.Sequential(
            nn.Linear(in_size, h, bias=True),
            nn.ReLU(),
            
            nn.Linear(h, h, bias=True),
            nn.ReLU(),
            
            nn.Linear(h, num_actions, bias=True),
        )

        self.critic = nn.Sequential(
            nn.Linear(in_size, h, bias=True),
            nn.ReLU(),
            
            nn.Linear(h, h, bias=True),
            nn.ReLU(),
            
            nn.Linear(h, 1, bias=True),
        )


    def forward(self, state):
        policy = self.actor(state)
        value = self.critic(state)

        return policy, torch.squeeze(value)

    def get_action(self, state):
        policy, value = self(state)  # use A3C to get policy and value

        action_prob = torch.softmax(policy, dim=-1)
        cat = Categorical(action_prob)
        action = cat.sample()

        return (action, cat.log_prob(action), cat.entropy().mean(), value)

class ICMAgent(nn.Module):
    def __init__(self, num_envs, num_actions, in_size, lr=1e-4):
        super().__init__()

        self.num_envs = num_envs
        self.num_actions = num_actions
        self.in_size = in_size
        self.is_cuda = torch.cuda.is_available()

        self.a2c = A2CNet(self.num_actions, self.in_size)

        if self.is_cuda:
            self.a2c.cuda()

        self.lr = lr
        self.optimizer = optim.Adam(self.a2c.parameters(), self.lr)


class Runner(object):

    def __init__(self, net, env, num_envs, state_size, rollout_size=50, num_updates=250000, max_grad_norm=0.5,
                 value_coeff=0.5, entropy_coeff=0.02, is_cuda=True, seed=42):
        super().__init__()

        self.num_envs = num_envs
        self.rollout_size = rollout_size
        self.num_updates = num_updates
        self.seed = seed

        self.max_grad_norm = max_grad_norm

        self.is_cuda = torch.cuda.is_available() and is_cuda
        self.env = env

        self.storage = RolloutStorage(self.rollout_size, self.num_envs, state_size=state_size, is_cuda=self.is_cuda, value_coeff=value_coeff,
                                      entropy_coeff=entropy_coeff)

        self.net = net

        if self.is_cuda:
            self.net = self.net.cuda()

    def train(self):
        obs = self.env.reset()
        self.storage.states[0].copy_(torch.tensor(obs).clone())
        best_loss = np.inf
        best_reward = -np.inf

        for num_update in range(self.num_updates):
            final_value, entropy = self.episode_rollout()

            self.net.optimizer.zero_grad()

            loss = self.storage.a2c_loss(final_value, entropy)
            loss.backward(retain_graph=False)

            nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)

            self.net.optimizer.step()

            if loss < best_loss:
                best_loss = loss.item()
                print("model with best loss: ", best_loss, " at update #", num_update)
            
            if self.storage.rewards.max().item() > best_reward:
                print("model with best rewards: ", self.storage.rewards.max().item(), " at update #", num_update)
                best_reward = self.storage.rewards.max().item()

            elif num_update % 10 == 0:
                print("current loss: ", loss.item(), " at update #", num_update)
            
            self.storage.after_update()
        self.env.close()

    def episode_rollout(self):
        episode_entropy = 0
        m = tuple(env.actions.keys())

        for step in range(self.rollout_size):
            a_t, log_p_a_t, entropy, value = self.net.a2c.get_action(self.storage.get_state(step))
            episode_entropy += entropy

            obs = []
            rewards = []
            dones = []
            for act in a_t.cpu().numpy():
                ob, reward, done = self.env.step(m[act])
                obs.append(ob)
                rewards.append(reward)
                dones.append(done)
            self.storage.insert(step, rewards, obs, a_t, log_p_a_t, value, dones)

        with torch.no_grad():
            _, _, _, final_value = self.net.a2c.get_action(self.storage.get_state(step + 1))

        return final_value, episode_entropy

from road_env import Road

env = Road()
agent = ICMAgent(1, len(env.actions.keys()), len(env.state_features))
runner = Runner(agent, env, 1, len(env.state_features))
runner.train()
