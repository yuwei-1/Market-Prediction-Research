import gymnasium as gym
import random
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.feed_forward_network import FeedForward
from models.deep_q_agent import DQNAgent
from models.nn_world_model import NNWorldModel

from IPython import display
is_ipython = 'inline' in matplotlib.get_backend()

class DynaDQNAgent(DQNAgent):

    def __init__(self, *dqn_args, planning_steps=50, world_hidden_size=8, non_linearity='gelu', **dqn_kwargs):
        super().__init__(*dqn_args, **dqn_kwargs)
        self.planning_steps = planning_steps
        self.dyna = False
        self.state_action_mem = set()
        self.wm = NNWorldModel(self.n_actions, self.n_obs, hidden_size=world_hidden_size, non_linearity=non_linearity, device=self.device)

    def step_action_value_function(self):

        if self.get_memory_len() < self.train_threshold:
            return

        loss = self.get_network_loss()
        self.backprop_network(loss, self.gradient_clipping > 0 or self.gradient_norm_clipping > 0)
            
        self.construct_world_model()
        self.dyna_planning_steps()

        if self.steps_done % 500 == 0:
            self.stats = self.wm.test(*self.retrieve_all())

    def construct_world_model(self):
        self.wm.observe(*self.retrieve_batch_info())

    def dyna_planning_steps(self):
        self.dyna = True
        for s in range(self.planning_steps):
            loss = self.get_network_loss()
            for g in self.optimizer.param_groups:
                g['lr'] *= 0.1
            self.backprop_network(loss)
            if self.double_dqn:
                if self.target_net_update == "soft":
                    self.perform_soft_target_update()
                elif self.target_net_update == "hard":
                    self.perform_hard_target_update()
        for g in self.optimizer.param_groups:
            g['lr'] *= 10
        self.dyna = False

    def retrieve_batch_info(self):
        if not self.dyna:
            return super().retrieve_batch_info()
        else:
            #states, actions, _, _ = super().retrieve_batch_info()
            states, actions = zip(*random.sample(list(self.state_action_mem), self.batch_size))
            state_batches = torch.stack(states, dim=0)
            action_batches = torch.stack(actions, dim=0)

            next_state_pred, rewards = self.wm.predict(state_batches, action_batches)
            return state_batches, action_batches, next_state_pred, rewards
        
    def retrieve_all(self):
        batch = self.transition(*zip(*self.replay_memory))
        states = torch.stack(batch.state)
        rewards = torch.stack(batch.reward).squeeze()
        actions = torch.stack(batch.action)
        next_states = batch.next_state
        return states, actions, next_states, rewards
    
    def retain_experience(self, *args):
        self.replay_memory.append(self.transition(*args))
        self.state_action_mem.add((args[:2]))