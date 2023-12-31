# import gymnasium as gym
# import math
# import random
# import matplotlib
# import matplotlib.pyplot as plt
# from collections import namedtuple, deque
# from itertools import count

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
from abc import abstractmethod, ABC
import random

class Agent:

    def __init__(self):
        pass

    @abstractmethod
    def init_agent(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self, path):
        pass