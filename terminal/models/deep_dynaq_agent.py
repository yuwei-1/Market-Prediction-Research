import gymnasium as gym
import random
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple, deque
from itertools import count

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.deep_q_agent import DQNAgent
from models.linear_world_model import LinearWorldModel
from models.nn_world_model import NNWorldModel


class DynaDQNAgent(DQNAgent):

    def __init__(self, n_observations, n_actions, world_model, dyna_train_threshold=1000, tabular=False, planning_steps=50, **dqn_kwargs):
        super().__init__(n_observations, n_actions, **dqn_kwargs)
        self.tabular = tabular
        self.planning_steps = planning_steps
        self.dyna = False
        self.min_state, self.max_state, self.total_state, self.k = None, None, None, 0
        self.wm = world_model
        self.dyna_train_threshold = dyna_train_threshold

    def step_action_value_function(self, steps_done):
        self.stats = steps_done
        if super().step_action_value_function(steps_done):
            if steps_done >= self.dyna_train_threshold:
                self.construct_world_model(steps_done)
                self.dyna_planning_steps(steps_done)
                #if steps_done % 500 == 0:
                    #self.stats = 
                    #self.wm.test(*self.retrieve_all())

    def construct_world_model(self, steps_done):
        if isinstance(self.wm, NNWorldModel):
            self.wm.observe(*self.retrieve_batch_info())
        elif isinstance(self.wm, LinearWorldModel) and steps_done % self.dyna_train_threshold == 0:
            self.wm.observe(*self.retrieve_all())

    def dyna_planning_steps(self, steps_done):
        self.dyna = True
        for s in range(self.planning_steps):
            self.loss = self.get_network_loss()
            self.backprop_network()
            if self.double_dqn:
                if self.target_net_update == "soft":
                    self.perform_soft_target_update()
                elif self.target_net_update == "hard":
                    self.perform_hard_target_update(steps_done)
        self.dyna = False

    def retrieve_batch_info(self):
        if (not self.dyna) or self.tabular:
            return super().retrieve_batch_info()
        else:
            state_batches, action_batches, _, _ = super().retrieve_batch_info()
            next_state_pred, rewards = self.wm.predict(state_batches, action_batches)
            return state_batches, action_batches, next_state_pred, rewards.unsqueeze(1)
        
    def retrieve_all(self):
        batch = self.transition(*zip(*self.replay_memory))
        states = torch.stack(batch.state)
        rewards = torch.stack(batch.reward)
        actions = torch.stack(batch.action)
        next_states = batch.next_state
        return states, actions, next_states, rewards

    def maintain_state_bounds(self, state):
        if self.min_state is None:
            self.min_state = torch.ones_like(state, device=self.device)*math.inf
            self.max_state = torch.ones_like(state, device=self.device)*-math.inf
            self.total_state = torch.zeros_like(state, device=self.device)
        self.min_state = torch.where(state < self.min_state, state, self.min_state)
        self.max_state = torch.where(state > self.max_state, state, self.max_state)
        self.total_state += state
        self.k += 1

    def sample_hypothetical_state_action_uniform(self):
        lower_bounds = self.min_state.tolist()
        upper_bounds = self.max_state.tolist()
        sample_observations = []
        self.stats = f"lower: {lower_bounds}, upper : {upper_bounds}"
        for i in range(self.n_observations):
            l = lower_bounds[i]
            u = upper_bounds[i]
            sample_observations += [(l - u)*torch.rand(self.batch_size, 1) + u]
        states = torch.cat(sample_observations, dim=1).to(self.device)
        actions = torch.randint(0, self.n_actions, (self.batch_size, 1)).to(self.device)
        return states, actions
    
    def sample_hypothetical_state_action_gaussian(self, std_adjust=0.8):
        mean = self.total_state/self.k
        std = std_adjust*(self.max_state - self.min_state)/6
        self.stats = f"mean: {mean}, std : {std}"

        mean = mean.repeat(self.batch_size, 1)
        std = std.repeat(self.batch_size, 1)
        states = torch.normal(mean, std).to(self.device)
        actions = torch.randint(0, self.n_actions, (self.batch_size, 1)).to(self.device)
        return states, actions