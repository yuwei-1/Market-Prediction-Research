import unittest
import sys
import torch
sys.path.append(r"C:\Users\YuweiZhu\OneDrive - Alloyed\Documents\Market-Prediction-Research")
sys.path.append(r"C:\Users\YuweiZhu\OneDrive - Alloyed\Documents\Market-Prediction-Research\terminal")
from models.deep_q_agent import DQNAgent
from models.feed_forward_network import FeedForward
from models.tests.test_base_q_learning import BaseDQNTests


class TestDQN(BaseDQNTests.TestDQN):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        torch.manual_seed(0)
        self.dqn = DQNAgent(1, 1, double_dqn=True)
        self.batch_size = 1
        self.train_threshold = 1

if __name__ == "__main__":
    unittest.main()