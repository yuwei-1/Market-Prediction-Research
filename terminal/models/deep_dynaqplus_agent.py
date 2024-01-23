from models.deep_dynaq_agent import DynaDQNAgent
from collections import namedtuple
import torch
import random
import math


class DynaPlusDQNAgent(DynaDQNAgent):
    
    def __init__(self, *args, reward_increase_rate=0.01, **kwargs, ):
        super().__init__(*args, **kwargs)
        self.reward_increase_rate = reward_increase_rate
        self.reward_plus = self.init_agent_memory()
        self.t = 0
    
    def retain_experience(self, *args):
        self.replay_memory.append(self.transition(*args))
        self.reward_plus.append(self.t)
        self.t += 1
    
    def retrieve_batch_info(self):
        if (not self.dyna) or self.tabular:
            return super().retrieve_batch_info()
        else:
            state_batches, action_batches, _, _ = self.retrieve_batch_info()
            next_state_pred, rewards = self.wm.predict(state_batches, action_batches)
            return state_batches, action_batches, next_state_pred, rewards

    def retrieve_batch_info(self):
        batch, rewards_plus = self.recall_experience(self.batch_size)
        batch = self.transition(*zip(*batch))
        states = torch.stack(batch.state)
        rewards = torch.stack(batch.reward)
        rewards_plus = torch.tensor(rewards_plus, device=self.device).unsqueeze(1)
        actions = torch.stack(batch.action)
        next_states = batch.next_state
        self.stats = f"max reward: {max(rewards_plus)}"
        return states, actions, next_states, rewards+rewards_plus
    
    def recall_experience(self, sample_size):
        idcs = random.sample(range(self.get_memory_len()), sample_size)
        batch, rewards_plus = [], []
        for i in idcs:
            batch += [self.replay_memory[i]]
            rewards_plus += [self.reward_increase_rate*math.sqrt(self.t - self.reward_plus[i])]
            self.reward_plus[i] = self.t
        return batch, rewards_plus