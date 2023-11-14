import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict

class FeedForward(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_size_1=128, hidden_size_2=128):
        super().__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions

        # network
        self.net = nn.Sequential(
                OrderedDict([
                ('projection_1', nn.Linear(n_observations, hidden_size_1)),
                ('non_linearity_1', nn.ReLU()),
                ('projection_2', nn.Linear(hidden_size_1, hidden_size_2)),
                ('non_linearity_2', nn.ReLU()),
                ('output_projection', nn.Linear(hidden_size_2, n_actions))
                ])
        )

    def forward(self, x):
        return self.net(x)