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
        self.n_obs, self.n_actions = self.get_env_info()
        self.agent = agent(self.n_obs, self.n_actions)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        # self.stats = "training"
        
    def get_env_info(self):
        n_actions = self.environment.action_space.n
        n_obs = self.environment.observation_space.shape[0]
        return n_obs, n_actions

    def train(self, 
              episodes=2000, 
              title = "",
              continuous=False,
              plot=False):
        '''
        Optimizer should be passed in as a lambda function only accepting the model parameters and learning rate
        '''
        self.episodes = episodes
        self.title = title if title else ""
        self.steps_done = 0
        self.episode_durations = []
        self.plot = plot

        # if continuous:
        #     self.episodes = 1
        
        self.agent.train()
        
        self.train_loop(continuous=continuous)

    def train_loop(self, continuous=False):
        #if continuous:
        #    self.test()
        for ep in range(self.episodes):
            state, _ = self.environment.reset()
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            cum_return = 0
            for t in count():
                # agent chooses an action
                action = self.agent.get_action(state, self.steps_done, self.environment)
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
                self.agent.retain_experience(state, action, next_state, reward)
                # change current state
                state = next_state
                # learn from some past experiences
                self.agent.step_action_value_function(self.steps_done)
                # terminate the episode under stopping conditions
                if continuous or terminated or truncated:
                    self.episode_durations.append(cum_return)
                    if self.plot:
                        self.plot_durations()
                    if terminated or truncated:
                        break
                self.steps_done += 1
            #if continuous:
            #    self.test()

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
            plt.title(self.agent.stats)
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
            
    # def test(self):
    #     # code for testing stock predictive performance of DQN
    #     self.agent.eval()
    #     state, _ = self.environment.reset(dataset='test')
    #     state = torch.tensor(state, dtype=torch.float32).to(self.device)
    #     actions = []
    #     for t in count():
    #         action = self.agent.get_action(state, explore=False).item()
    #         actions += [action]
    #         next_state, reward, terminated, truncated, _ = self.environment.step(action)
    #         if next_state is not None:
    #             next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
    #         if terminated:
    #             break
    #         state = next_state
    #     optimal = self.environment.y_test[1:]
    #     actions = np.array(actions[:-1])
    #     mask = ~(actions == 2)
    #     acc = (optimal[mask] == actions[mask]).sum()/optimal[mask].shape[0]

    #     print(f"test score: {acc}")
    #     self.stats = f"Training .. test score = {acc}"
    #     return actions, optimal