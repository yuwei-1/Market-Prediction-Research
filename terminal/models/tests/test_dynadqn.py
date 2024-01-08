import unittest
import sys
import torch
sys.path.append(r"C:\Users\YuweiZhu\OneDrive - Alloyed\Documents\Market-Prediction-Research")
sys.path.append(r"C:\Users\YuweiZhu\OneDrive - Alloyed\Documents\Market-Prediction-Research\terminal")
from environments.unittest_environment import UnittestEnvironment
from reinforcement_learning_tools.reinforcement_learning_dojo import RLDojo
from models.deep_dynaq_agent import DynaDQNAgent
from models.tests.test_base_q_learning import BaseDQNTests


class TestDynaDQN(BaseDQNTests.TestDQN):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        torch.manual_seed(0)
        self.dqn = DynaDQNAgent(1, 1, double_dqn=True)
        self.batch_size = 2
        self.train_threshold = 10

    def test_agent_training(self):
        test_env = UnittestEnvironment()
        dqn = lambda obs, actions : DynaDQNAgent(obs, 
                                             actions,
                                             double_dqn=True,
                                             hidden_size=3, 
                                             batch_size=2, 
                                             train_threshold=2, 
                                             planning_steps=2,
                                             learning_rate=0.01)
        dojo = RLDojo(dqn, test_env)
        dojo.train(episodes=2)

if __name__ == "__main__":
    unittest.main()