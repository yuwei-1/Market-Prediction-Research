import unittest
import sys
import gymnasium
import numpy as np
sys.path.append(r"C:\Users\YuweiZhu\OneDrive - Alloyed\Documents\Market-Prediction-Research")
sys.path.append(r"C:\Users\YuweiZhu\OneDrive - Alloyed\Documents\Market-Prediction-Research\terminal")
from terminal.models.deep_q_agent import DQNAgent
from terminal.models.feed_forward_network import FeedForward
from terminal.environments.stock_env import StockEnvironment


class TestStockEnv(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.folder_path = "raw_data/"
        self.file_name = "AAPL"
        self.env = StockEnvironment()
        self.env.make(self.file_name)
        
    def test_make(self):
        with self.assertRaises(AssertionError):
            self.env.make("FAKE", self.folder_path)
        
        self.assertEqual(self.env.X_train.shape[0], self.env.y_train.shape[0])
        self.assertEqual(self.env.X_test.shape[0], self.env.y_test.shape[0])
        self.assertEqual(self.env.X_train.shape[1], self.env.X_test.shape[1])

        self.assertEqual(self.env.action_space.n, len(self.env.actions))
        self.assertIsInstance(self.env.action_space.sample(), int)

    def test_stock_generator(self):
        generator = self.env.stock_generator("train")
        x, y = next(generator)
        test_generator = self.env.stock_generator("test")
        x1, y1 = next(test_generator)

        self.assertEqual(x.shape, x1.shape)
        self.assertEqual(y.shape, y1.shape)

    def test_reset(self):
        res = self.env.reset()
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], np.ndarray)

    def test_step(self):
        self.env.reset()
        res = self.env.step(0)

        self.assertEqual(len(res), 5)
        self.assertIsInstance(res[0], np.ndarray)
        self.assertIsInstance(res[1], int)
        self.assertIsInstance(res[2], bool)
        self.assertIsInstance(res[3], bool)
        self.assertEqual(res[-1], None)

if __name__ == "__main__":
    unittest.main()