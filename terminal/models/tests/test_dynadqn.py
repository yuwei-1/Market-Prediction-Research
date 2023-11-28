import unittest
import sys
import gymnasium
sys.path.append("C:/Users/YuweiZhu/OneDrive - Alloyed/Documents/Market-Prediction-Research/")
sys.path.append("C:/Users/YuweiZhu/OneDrive - Alloyed/Documents/Market-Prediction-Research/terminal")
from models.deep_dynaq_agent import DynaDQNAgent
from models.feed_forward_network import FeedForward
from terminal.models.tests.test_base_q_learning import BaseDQNTests
import torch


class TestDynaDQN(BaseDQNTests.TestDQN):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.dqn = DynaDQNAgent(double_dqn=True)
        self.batch_size = 2
        self.train_threshold = 10

if __name__ == "__main__":
    unittest.main()