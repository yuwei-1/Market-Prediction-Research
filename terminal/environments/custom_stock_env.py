import sys

sys.path.append(r"C:\Users\YuweiZhu\OneDrive - Alloyed\Documents\Market-Prediction-Research")
sys.path.append(r"C:\Users\YuweiZhu\OneDrive - Alloyed\Documents\Market-Prediction-Research\terminal")

from environments.reinforcement_learning_env import ReinforcementLearningEnvironment
from data_processing.feature_generation import FeatureGenerator
from data_processing.csv_reader import CSVReader
from data_processing.y_finance_reader import YahooFinanceReader
from utils.guards import stock_ticker_guard
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import random
import pandas as pd
from copy import deepcopy


class StockEnvironment(ReinforcementLearningEnvironment):
    '''
    Given the name of a stock, this creates a learning environment
    for a reinforcement learning agent
    '''

    def __init__(self) -> None:
        super().__init__()
        self.actions = {"buy":1, "sell":0, "hold":2}
        self.action_history = []
        self.action_mapping = {0:-1, 1:1, 2:0}
    
    def make(self, ticker, prediction_period=1):
        # Check if the ticker symbol is valid
        stock_ticker_guard(ticker)

        yahoo_adjust_close = "Close"
        reader = YahooFinanceReader()
        stock = reader.read_stock_price_data(ticker)
        fg = FeatureGenerator(stock, task="custom", target_variable=yahoo_adjust_close)
    
        fg._create_classes(period=prediction_period)
        fg.apply_IG_top_ten_indicators()
        self.X_train, self.X_test, self.y_train, self.y_test = fg.create_modelling_data(features=["Volume", yahoo_adjust_close], target="Profit")
        self.train_close_prices = fg.info["train_close_prices"]
        self.test_close_prices = fg.info["test_close_prices"]

        self.action_space = self.Aspace(len(self.actions), self.sample)
        self.observation_space = self.Obspace((self.X_train.shape[1],))
        #print("The following derived features are in use: ", fg.training_features)

    def stock_generator(self, dataset):
        if dataset == "train":
            for i in range(self.X_train.shape[0]):
                yield self.X_train[i, :], self.y_train[i]
        elif dataset == "test":
            for i in range(self.X_test.shape[0]):
                yield self.X_test[i, :], self.y_test[i]

    def step(self, action):

        terminated = False
        truncated = False
        # action = self.action_mapping[action]
        if action == self.expected_action:
            reward = 1
        elif action == 2:
            reward = 0
        else:
            reward = -1

        # prev_price, prev_action = self.history["price"], self.history["action"]
        # if prev_price is not None and action == -prev_action:
        #     reward = prev_action*(self.curr_price - prev_price)/prev_price
        #     self.history["action"] = 0
        #     self.history["price"] = None
        #     self.action_history += [0]
        # elif action == prev_action:
        #     reward = 0
        #     self.action_history += [0]
        # else:
        #     self.history["price"] = self.curr_price
        #     self.history["action"] = action
        #     reward = 0
        #     self.action_history += [action]

        try:
            next_state, self.expected_action = next(self.gen)
        except:
            next_state = None
            terminated = True

        return next_state, reward, terminated, truncated, None
        
    def make_portfolio(self):
        self.history = {"price":None, "action":0}

    def reset(self, dataset='train'):
        self.make_portfolio()
        self.gen = self.stock_generator(dataset)
        state, self.expected_action = next(self.gen)
        return state, None

    def close(self):
        pass
    
    def sample(self):
        return random.sample(range(len(self.actions)), 1)[0]
    

if __name__ == "__main__":
    stock_env = StockEnvironment()
    stock_env.make("AAPL")