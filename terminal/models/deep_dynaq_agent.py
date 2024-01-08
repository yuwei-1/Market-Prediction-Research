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
from models.nn_world_model import NNWorldModel


class DynaDQNAgent(DQNAgent):

    def __init__(self, n_observations, n_actions, planning_steps=50, world_hidden_size=8, **dqn_kwargs):
        super().__init__(n_observations, n_actions, **dqn_kwargs)
        self.planning_steps = planning_steps
        self.dyna = False
        self.state_action_mem = set()
        self.min_state, self.max_state, self.total_state, self.k = None, None, None, 0
        self.wm = NNWorldModel(n_actions, n_observations, hidden_size=world_hidden_size, device=self.device)

    def step_action_value_function(self, steps_done):
        if super().step_action_value_function(steps_done):
            self.construct_world_model()
            self.dyna_planning_steps(steps_done)
            if steps_done % 500 == 0:
                #self.stats = 
                self.wm.test(*self.retrieve_all())

    def construct_world_model(self):
        self.wm.observe(*self.retrieve_batch_info())

    def dyna_planning_steps(self, steps_done):
        self.dyna = True
        # for g in self.optimizer.param_groups:
        #     g['lr'] *= 0.1
        for s in range(self.planning_steps):
            loss = self.get_network_loss()
            self.backprop_network(loss)
            if self.double_dqn:
                if self.target_net_update == "soft":
                    self.perform_soft_target_update()
                elif self.target_net_update == "hard":
                    self.perform_hard_target_update(steps_done)
        # for g in self.optimizer.param_groups:
        #     g['lr'] *= 10
        self.dyna = False

    def retrieve_batch_info(self):
        if not self.dyna:
            return super().retrieve_batch_info()
        else:
            # states, actions = zip(*random.sample(list(self.state_action_mem), self.batch_size))
            # state_batches = torch.stack(states, dim=0)
            # action_batches = torch.stack(actions, dim=0)
            state_batches, action_batches = self.sample_hypothetical_state_action_gaussian()

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
        #self.state_action_mem.add((args[:2]))
        state = args[0]
        if state is not None:
            self.maintain_state_bounds(state)

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
    
    def sample_hypothetical_state_action_gaussian(self, std_adjust=1):
        lower_bounds = self.min_state.tolist()
        upper_bounds = self.max_state.tolist()
        #mean = (self.min_state + self.max_state)/2
        mean = self.total_state/self.k
        std = std_adjust*(self.max_state - self.min_state)/6
        self.stats = f"mean: {mean}, std : {std}"

        mean = mean.repeat(self.batch_size, 1)
        std = std.repeat(self.batch_size, 1)
        states = torch.normal(mean, std).to(self.device)
        actions = torch.randint(0, self.n_actions, (self.batch_size, 1)).to(self.device)
        return states, actions