import unittest
import sys
import torch
sys.path.append(r"C:\Users\YuweiZhu\OneDrive - Alloyed\Documents\Market-Prediction-Research")
sys.path.append(r"C:\Users\YuweiZhu\OneDrive - Alloyed\Documents\Market-Prediction-Research\terminal")
from environments.unittest_environment import UnittestEnvironment
from reinforcement_learning_tools.reinforcement_learning_dojo import RLDojo
from models.deep_dynaq_agent import DynaDQNAgent
from models.linear_world_model import LinearWorldModel
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from models.tests.test_base_q_learning import BaseDQNTests


class TestDynaDQN(BaseDQNTests.TestDQN):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        torch.manual_seed(0)
        self.dqn = DynaDQNAgent(1, 1, None, double_dqn=True)
        self.batch_size = 2
        self.train_threshold = 10

    def test_agent_training(self):
        test_env = UnittestEnvironment()
        dqn = lambda obs, actions : DynaDQNAgent(obs, 
                                             actions,
                                             None,
                                             double_dqn=True,
                                             hidden_size=3, 
                                             batch_size=2, 
                                             train_threshold=2, 
                                             planning_steps=2,
                                             learning_rate=0.01)
        dojo = RLDojo(dqn, test_env)
        dojo.train(episodes=2)

    def test_cartpole_environment(self):
        agent = lambda obs, actions : DynaDQNAgent(obs, 
                                            actions,
                                            LinearWorldModel(DecisionTreeRegressor, DecisionTreeClassifier),
                                            double_dqn=True,
                                            target_net_update="soft",
                                            activation='relu',
                                            tau=0.001,
                                            eps_decay_length=1000,
                                            planning_steps=5,
                                            gradient_clipping=100,
                                            gradient_norm_clipping=-1)

        # agent = lambda obs, actions : DQNAgent(obs, 
        #                                        actions,
        #                                        double_dqn=True,
        #                                        target_net_update="soft",
        #                                        activation='relu',
        #                                        eps_decay_length=1000,
        #                                        gradient_clipping=100,
        #                                        gradient_norm_clipping=-1)


        dojo = RLDojo(agent)
        dojo.train()

if __name__ == "__main__":
    unittest.main()