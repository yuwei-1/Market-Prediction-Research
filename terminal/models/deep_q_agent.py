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
from models.generic_feed_forward_network_creator import GenericFeedForwardNetwork
from models.reinforcement_learning_agent import Agent
from terminal.environments.custom_stock_env import StockEnvironment
from utils.guards import shape_guard

from IPython import display
is_ipython = 'inline' in matplotlib.get_backend()

class DQNAgent(Agent):

    def __init__(self,
                n_observations,
                n_actions,
                hidden_size=128,
                layers=4,
                double_dqn=False, 
                target_net_update='hard',
                tau=0.005,
                hard_update_interval=500,
                mem_length=10000,
                eps_start=1.0,
                eps_end=0.05,
                eps_decay_length=10000,
                gradient_clipping=-1,
                gradient_norm_clipping=1,
                activation='relu',
                discount=0.99,
                batch_size=128, 
                train_threshold=500,
                learning_rate=0.0001,
                optimizer=lambda param, lr: optim.AdamW(param, lr=lr, amsgrad=True), 
                loss=nn.SmoothL1Loss()):
        
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.layers = layers
        self.n_layer_guard(layers)
        self.hidden_size = hidden_size
        self.mem_length = mem_length
        self.double_dqn = double_dqn
        self.tau = tau
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_length = eps_decay_length
        self.hard_update_interval = hard_update_interval
        self.activation = activation
        self.target_net_update = self.update_guard(target_net_update)
        self.gradient_clipping = gradient_clipping
        self.gradient_norm_clipping = gradient_norm_clipping
        self.discount = discount
        self.batch_size = batch_size
        self.train_threshold = train_threshold
        self.learning_rate = learning_rate
        self.loss_fn = loss
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_memory = self.init_agent_memory()
        self.net = self.init_agent(n_observations, n_actions)
        self.optimizer = optimizer(self.net.parameters(), learning_rate)
        if self.double_dqn:
            self.target_net = self.init_agent(n_observations, n_actions)
            self.target_net.load_state_dict(self.net.state_dict())
        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        self.stats = "training"
        
    def init_agent(self, input_size, output_size, q_func_approximator=GenericFeedForwardNetwork):
        n_hidden_layers = self.layers - 2
        nodes_per_layer = [input_size] + [self.hidden_size]*n_hidden_layers + [output_size]
        activations = [self.activation]*n_hidden_layers + ['none']
        return q_func_approximator(self.layers, activations, nodes_per_layer).to(self.device)
    
    def train(self):
        self.net.train()
        if self.double_dqn:
            self.target_net.train()
    
    def eval(self):
        self.net.eval()
        if self.double_dqn:
            self.target_net.eval()

    def save(self, file_path):
        net_state_dict = self.net.state_dict()
        torch.save(net_state_dict, file_path)

    def load(self, model_path):
        self.net.load_state_dict(torch.load(model_path))
        if self.double_dqn:
            self.target_net.load_state_dict(torch.load(model_path))

    def perform_soft_target_update(self):
        net_state_dict = self.net.state_dict()
        target_net_state_dict = self.target_net.state_dict()
        for key in net_state_dict.keys():
            target_net_state_dict[key] = self.tau*net_state_dict[key] + (1-self.tau)*target_net_state_dict[key]
        self.target_net.load_state_dict(target_net_state_dict)

    def perform_hard_target_update(self, steps_done):
        if steps_done % self.hard_update_interval == 0 and steps_done > 0:
            net_state_dict = self.net.state_dict()
            target_net_state_dict = self.target_net.state_dict()
            for key in net_state_dict.keys():
                target_net_state_dict[key] = net_state_dict[key]
            self.target_net.load_state_dict(target_net_state_dict)

    def step_action_value_function(self, steps_done):
        if self.get_memory_len() < self.train_threshold:
            return False
        self.loss = self.get_network_loss()
        self.backprop_network(self.gradient_clipping > 0 or self.gradient_norm_clipping > 0)

        if self.double_dqn:
            if self.target_net_update == "soft":
                self.perform_soft_target_update()
            elif self.target_net_update == "hard":
                self.perform_hard_target_update(steps_done)
        return True
    
    def get_network_loss(self):
        #non_terminal_states = torch.tensor([])

        #while not non_terminal_states.shape[0]:
        states, actions, next_states, rewards = self.retrieve_batch_info()
        non_terminal_mask = torch.tensor(tuple(map(lambda ns:ns is not None, next_states)), device=self.device, dtype=torch.bool)
        non_terminal_states = torch.stack([ns for ns in next_states if ns is not None]).to(self.device)

        current_estimate = torch.gather(self.net(states), 1, actions)
        q_estimate = torch.zeros(self.batch_size, device=self.device)
        shape_guard(non_terminal_mask, q_estimate)
        with torch.no_grad():
            if self.double_dqn:
                q_estimate[non_terminal_mask] = (self.discount * torch.max(self.target_net(non_terminal_states), dim=1)[0])
            else:
                q_estimate[non_terminal_mask] = (self.discount * torch.max(self.net(non_terminal_states), dim=1)[0])

        shape_guard(q_estimate.unsqueeze(1), rewards)
        target = q_estimate.unsqueeze(1) + rewards
        shape_guard(current_estimate, target)
        loss = self.loss_fn(current_estimate, target)
        return loss

    def backprop_network(self, clip=False):
        self.optimizer.zero_grad()
        self.loss.backward()
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
    
    def get_action(self, state, steps_done, environment, explore=True):
        ep_threshold = self.epsilon_greedy(steps_done)
        rand = random.uniform(0,1)
        if rand <= ep_threshold and explore:
            # choose random action
            return torch.tensor([environment.action_space.sample()], device=self.device, dtype=torch.long)
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
    def n_layer_guard(layers):
        assert layers > 2, "must have at least input and output layer"

    @staticmethod
    def update_guard(update_type):
        assert isinstance(update_type, str), "update type must be string"
        update_type = update_type.lower()
        allowed = {'soft', 'hard'}
        assert update_type in allowed, "target updates must be either hard or soft"
        return update_type