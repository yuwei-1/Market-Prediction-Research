import gymnasium as gym
import gym_trading_env
import random
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import namedtuple, deque
from itertools import count
import sys
sys.path.append(r"C:\Users\YuweiZhu\OneDrive - Alloyed\Documents\Market-Prediction-Research")
sys.path.append(r"C:\Users\YuweiZhu\OneDrive - Alloyed\Documents\Market-Prediction-Research\terminal")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.feed_forward_network import FeedForward
from models.reinforcement_learning_agent import Agent
from terminal.environments.custom_stock_env import StockEnvironment
from utils.guards import shape_guard

from IPython import display
is_ipython = 'inline' in matplotlib.get_backend()

class RLDojo:

    def __init__(self, 
                agent : Agent,
                environment=gym.make('CartPole-v1', render_mode="human")):
    
        self.environment = environment
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_obs, self.n_actions = self.get_env_info()
        self.replay_memory = self.init_agent_memory()
        #self.net = self.init_agent(self.n_obs, self.n_actions, **net_kwargs)

        # if self.double_dqn:
        #     self.target_net = self.init_agent(self.n_obs, self.n_actions, **net_kwargs)
        #     self.target_net.load_state_dict(self.net.state_dict())
        # self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        # self.stats = "training"
        
    def get_env_info(self):
        n_actions = self.environment.action_space.n
        n_obs = self.environment.observation_space.shape[0]
        return n_obs, n_actions

    def init_agent(self, input_size, output_size, q_func_approximator=FeedForward, **net_kwargs):
        return q_func_approximator(input_size, output_size, non_linearity=self.activation, **net_kwargs).to(self.device)

    def train(self, 
              episodes=2000, 
              discount=0.99,
              title = "",
              batch_size=128, 
              train_threshold=500, 
              learning_rate=0.0001,
              optimizer=lambda param, lr: optim.AdamW(param, lr=lr, amsgrad=True), 
              loss=nn.MSELoss(),
              continuous=False,
              plot=False):
        '''
        Optimizer should be passed in as a lambda function only accepting the model parameters and learning rate
        '''
        self.episodes = episodes
        self.title = title if title else ""
        self.discount = discount
        self.batch_size = batch_size
        self.train_threshold = train_threshold
        self.loss_fn = loss
        self.optimizer = optimizer(self.net.parameters(), learning_rate)
        self.steps_done = 0
        self.episode_durations = []
        self.plot = plot

        # if continuous:
        #     self.episodes = 1
        
        self.net.train()
        if self.double_dqn:
            self.target_net.train()
        
        self.train_loop(continuous=continuous)

    def train_loop(self, continuous=False):
        if continuous:
            self.test()
        for ep in range(self.episodes):
            state, _ = self.environment.reset()
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            cum_return = 0
            for t in count():
                # agent chooses an action
                action = self.get_action(state)
                # observe new obs from environment
                next_state, reward, terminated, truncated, _ = self.environment.step(action.item())
                cum_return += reward
                # logic for different scenarios
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
                # save experience to replay memory
                reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
                self.retain_experience(state, action, next_state, reward)
                # change current state
                state = next_state
                # learn from some past experiences
                self.step_action_value_function()
                # terminate the episode under stopping conditions
                if continuous or terminated or truncated:
                    self.episode_durations.append(cum_return)
                    if self.plot:
                        self.plot_durations()
                    if terminated or truncated:
                        break
                if self.double_dqn:
                    if self.target_net_update == "soft":
                        self.perform_soft_target_update()
                    elif self.target_net_update == "hard":
                        self.perform_hard_target_update()
                self.steps_done += 1
            if continuous:
                self.test()

        self.environment.close()
        if self.plot:
            self.plot_final(title=self.title)

    def plot_durations(self, show_result=False):
        # TODO: change up code, so not entirely identical
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title(self.stats)
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def test(self):
        # code for testing stock predictive performance of DQN
        self.net.eval()
        state, _ = self.environment.reset(dataset='test')
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        actions = []
        for t in count():
            action = self.get_action(state, explore=False).item()
            actions += [action]
            next_state, reward, terminated, truncated, _ = self.environment.step(action)
            if next_state is not None:
                next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
            if terminated:
                break
            state = next_state
        optimal = self.environment.y_test[1:]
        actions = np.array(actions[:-1])
        mask = ~(actions == 2)
        acc = (optimal[mask] == actions[mask]).sum()/optimal[mask].shape[0]

        print(f"test score: {acc}")
        self.stats = f"Training .. test score = {acc}"
        return actions, optimal

    def save(self, file_path):
        net_state_dict = self.net.state_dict()
        torch.save(net_state_dict, file_path)
        if self.double_dqn:
            target_state_dict = self.target_net.state_dict()
            torch.save(target_state_dict, file_path)

    def load(self, model_path, target_path=None):
        self.net.load_state_dict(torch.load(model_path))
        if self.double_dqn:
            assert target_path is None, "There is no target in use"
            self.target_net.load_state_dict(torch.load(target_path))

    def plot_final(self, title="title", moving_episode_means=100):
        ed = pd.Series(self.episode_durations)
        ra = ed.rolling(moving_episode_means).mean()

        plt.figure(figsize=(15,10))
        plt.plot(range(ed.shape[0]), ed, label="episode duration")
        plt.plot(range(ed.shape[0]), ra, label=f"{moving_episode_means} episode moving average")
        plt.title(title)
        plt.xlabel("episode")
        plt.ylabel("episode duration")
        plt.grid('on')
        plt.legend(loc="best")
        plt.show()

    def perform_soft_target_update(self):
        net_state_dict = self.net.state_dict()
        target_net_state_dict = self.target_net.state_dict()
        for key in net_state_dict.keys():
            target_net_state_dict[key] = self.tau*net_state_dict[key] + (1-self.tau)*target_net_state_dict[key]
        self.target_net.load_state_dict(target_net_state_dict)

    def perform_hard_target_update(self):
        if self.steps_done % self.hard_update_interval == 0 and self.steps_done > 0:
            net_state_dict = self.net.state_dict()
            target_net_state_dict = self.target_net.state_dict()
            for key in net_state_dict.keys():
                target_net_state_dict[key] = net_state_dict[key]
            self.target_net.load_state_dict(target_net_state_dict)

    def step_action_value_function(self):
        if self.get_memory_len() < self.train_threshold:
            return
        loss = self.get_network_loss()
        self.backprop_network(loss, self.gradient_clipping > 0 or self.gradient_norm_clipping > 0)

    def get_network_loss(self):

        non_terminal_states = torch.tensor([])

        while not non_terminal_states.shape[0]:
            states, actions, next_states, rewards = self.retrieve_batch_info()
            non_terminal_mask = torch.tensor(tuple(map(lambda ns:ns is not None, next_states)), device=self.device, dtype=torch.bool)
            non_terminal_states = torch.stack([ns for ns in next_states if ns is not None]).to(self.device)

        current_estimate = torch.gather(self.net(states), 1, actions)
        q_estimate = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            if self.double_dqn:
                q_estimate[non_terminal_mask] = (self.discount * torch.max(self.target_net(non_terminal_states), dim=1)[0])
            else:
                q_estimate[non_terminal_mask] = (self.discount * torch.max(self.net(non_terminal_states), dim=1)[0])

        target = q_estimate.unsqueeze(1) + rewards
        shape_guard(current_estimate, target)
        loss = self.loss_fn(current_estimate, target)
        return loss

    def backprop_network(self, loss, clip=False):
        self.optimizer.zero_grad()
        loss.backward()
        if clip:
            if self.gradient_norm_clipping > 0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.gradient_norm_clipping)
            if self.gradient_clipping > 0:
                torch.nn.utils.clip_grad_value_(self.net.parameters(), self.gradient_clipping)
        self.optimizer.step()

    def retrieve_batch_info(self):
        batch = self.transition(*zip(*self.recall_experience(self.batch_size)))
        states = torch.stack(batch.state)
        rewards = torch.stack(batch.reward).to(torch.float32)
        actions = torch.stack(batch.action)
        next_states = batch.next_state
        return states, actions, next_states, rewards
    
    def get_action(self, state, explore=True):
        # TODO: how to refactor this so can use other behaviour policies?

        ep_threshold = self.epsilon_greedy(self.steps_done)
        rand = random.uniform(0,1)
        if rand <= ep_threshold and explore:
            # choose random action
            return torch.tensor([self.environment.action_space.sample()], device=self.device, dtype=torch.long)
        else:
            # greedy: choose action with largest value
            with torch.no_grad():
                return torch.argmax(self.net(state)).unsqueeze(0)

    def init_agent_memory(self):
        return deque([], maxlen=self.mem_length)

    def retain_experience(self, *args):
        self.replay_memory.append(self.transition(*args))

    def recall_experience(self, sample_size):
        return random.sample(self.replay_memory, sample_size)
    
    def get_memory_len(self):
        return len(self.replay_memory)
    
    def epsilon_greedy(self, it):
        return max(self.eps_start - (self.eps_start - self.eps_end) * (it / self.eps_decay_length), self.eps_end)
    
    @staticmethod
    def update_guard(update_type):
        assert isinstance(update_type, str), "update type must be string"
        update_type = update_type.lower()
        allowed = {'soft', 'hard'}
        assert update_type in allowed, "target updates must be either hard or soft"
        return update_type