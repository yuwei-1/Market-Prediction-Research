from abc import abstractmethod, ABC
from collections import namedtuple
import random
import numpy as np
from environments.reinforcement_learning_env import ReinforcementLearningEnvironment


class UnittestEnvironment(ReinforcementLearningEnvironment):

    def __init__(self) -> None:
        super().__init__()
        self.actions = {0, 1}
        self.obs_dim = 2
        self.state = np.array([1,1])
        self.action_space = self.Aspace(len(self.actions), self.sample)
        self.observation_space = self.Obspace((self.obs_dim,))
        self.count = 0

    def make(self, *args, **kwargs):
        pass

    def step(self, action):
        terminated = False
        if action == 0:
            reward = 0
        else:
            reward = 1
        self.count += 1
        if self.count == 10:
            terminated = True
        return self.state, reward, terminated, False, None
    
    def reset(self, *args, **kwargs):
        return self.state, None

    def close(self):
        pass

    def sample(self):
        return random.sample(range(len(self.actions)), 1)[0]