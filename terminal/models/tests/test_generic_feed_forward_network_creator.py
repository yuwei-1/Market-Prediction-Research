import unittest
import sys
import torch
import torch.nn as nn
sys.path.append(r"C:\Users\YuweiZhu\OneDrive - Alloyed\Documents\Market-Prediction-Research")
sys.path.append(r"C:\Users\YuweiZhu\OneDrive - Alloyed\Documents\Market-Prediction-Research\terminal")
from models.generic_feed_forward_network_creator import GenericFeedForwardNetwork


class TestGenericFFNetworkCreator(unittest.TestCase):

    def test_create_feed_forward_network_example_1(self):
        n_layers = 2
        activations = ['tanh', 'tanh']
        nodes = [5, 5]
        ffn = GenericFeedForwardNetwork(n_layers, activations, nodes)
        self.assertEqual(ffn.net.layer_1.in_features, 5)
        self.assertEqual(ffn.net.layer_1.out_features, 5)
        self.assertIsInstance(ffn.net.activation_1, nn.Tanh)
        self.assertIsInstance(ffn.net.activation_2, nn.Tanh)

    def test_create_feed_forward_network_example_2(self):
        n_layers = 3
        activations = ['sigmoid', 'tanh', 'none']
        nodes = [1, 2, 3]
        ffn = GenericFeedForwardNetwork(n_layers, activations, nodes)
        self.assertEqual(ffn.net.layer_1.in_features, 1)
        self.assertEqual(ffn.net.layer_1.out_features, 2)
        self.assertEqual(ffn.net.layer_2.in_features, 2)
        self.assertEqual(ffn.net.layer_2.out_features, 3)
        self.assertIsInstance(ffn.net.activation_1, nn.Sigmoid)
        self.assertIsInstance(ffn.net.activation_2, nn.Tanh)
        self.assertIsInstance(ffn.net.activation_3, nn.Identity)