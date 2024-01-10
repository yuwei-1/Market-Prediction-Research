import unittest
import sys
import torch
sys.path.append(r"C:\Users\YuweiZhu\OneDrive - Alloyed\Documents\Market-Prediction-Research")
sys.path.append(r"C:\Users\YuweiZhu\OneDrive - Alloyed\Documents\Market-Prediction-Research\terminal")
from models.deep_q_agent import DQNAgent
from environments.unittest_environment import UnittestEnvironment
from reinforcement_learning_tools.reinforcement_learning_dojo import RLDojo
from models.tests.test_base_q_learning import BaseDQNTests


class TestDQN(BaseDQNTests.TestDQN):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        torch.manual_seed(0)
        self.dqn = DQNAgent(1, 1, double_dqn=True)
        self.batch_size = 1
        self.train_threshold = 1

    def test_agent_training(self):
        test_env = UnittestEnvironment()
        dqn = lambda obs, actions : DQNAgent(obs, 
                                             actions,
                                             hidden_size=3, 
                                             batch_size=2, 
                                             train_threshold=2, 
                                             learning_rate=0.01)
        dojo = RLDojo(dqn, test_env)
        dojo.train(episodes=2)

if __name__ == "__main__":
    unittest.main()