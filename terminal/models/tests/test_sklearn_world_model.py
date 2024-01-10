import unittest
import torch
import numpy as np
from models.linear_world_model import LinearWorldModel
from sklearn.linear_model import LinearRegression, LogisticRegression

class TestLinearWorldModel(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        reg = LinearRegression
        log = LogisticRegression
        self.lwm = LinearWorldModel(reg, log)

    def test_linear_world_model_observe(self):
        states = torch.ones((2, 4))
        actions = torch.tensor([[0],[1]], dtype=torch.long)
        next_states = [torch.ones(4), None]
        rewards = torch.ones((2, 1))

        self.lwm.observe(states, actions, next_states, rewards)
        self.assertTrue((self.lwm.state_transition.predict(np.ones((1,6))) == np.ones((1,4))).all())
        self.assertTrue((self.lwm.reward_transition.predict(np.ones((1,6))) == np.ones((1,1))).all())
        self.assertTrue((self.lwm.terminal_transition.predict(np.ones((1,6))) == np.array([0], dtype=bool)).all())

    def test_linear_world_model_predict(self):
        states = torch.ones((2, 4))
        actions = torch.tensor([[0],[1]], dtype=torch.long)
        next_states = [torch.ones(4), None]
        rewards = torch.ones((2, 1))

        self.lwm.observe(states, actions, next_states, rewards)
        pred_next_state, rewards = self.lwm.predict(states, actions)

        self.assertTrue((pred_next_state[0].to("cpu") == torch.ones(4)).all())
        self.assertTrue(pred_next_state[1] == next_states[1])
        self.assertTrue((rewards.to("cpu") == torch.ones((2,1))).all())