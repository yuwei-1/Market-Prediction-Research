import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from utils.guards import non_linearity_guard
from collections import OrderedDict

class GenericFeedForwardNetwork(nn.Module):

    def __init__(self, n_layers : int, activations : List[str], nodes_per_layer : List[int]) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.activations = [""] + activations
        self.nodes_per_layer = nodes_per_layer
        self.minimum_layer_guard(n_layers)
        self.valid_information_guard(n_layers, activations, nodes_per_layer)
        self.net = self.create_network()
    
    def foward(self, x):
        return self.net(x)
    
    def create_network(self):
        module_list = OrderedDict()
        input_dim = self.nodes_per_layer[0]
        for layer_num in range(1, self.n_layers):
            output_dim = self.nodes_per_layer[layer_num]
            module_list.update({f'layer_{layer_num}' : nn.Linear(input_dim, output_dim)})
            module_list.update({f'activation_{layer_num}' : non_linearity_guard(self.activations[layer_num])})
            input_dim = output_dim
        module_list.update({f'activation_{layer_num+1}' : non_linearity_guard(self.activations[layer_num+1])})
        return nn.Sequential(module_list)
    
    @staticmethod
    def minimum_layer_guard(layers):
        assert layers >= 2, "The feed forward network must have at least an input and output layer"
    
    @staticmethod
    def valid_information_guard(layers, activation_list, nodes_list):
        activation_length = len(activation_list)
        nodes_length = len(nodes_list)
        assert layers == activation_length, "activation information doesn't match number of layers"
        assert layers == nodes_length, "node information doesn't match number of layers"