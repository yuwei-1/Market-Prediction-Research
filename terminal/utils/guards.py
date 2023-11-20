import torch
import torch.nn as nn


def non_linearity_guard(activation):
    activation = activation.lower()
    allowed = {'tanh', 'relu', 'gelu', 'none'}
    assert activation in allowed, "The non-linearity must be in 'tanh', 'relu', 'gelu', 'none'"

    if activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'none':
        return nn.Identity()