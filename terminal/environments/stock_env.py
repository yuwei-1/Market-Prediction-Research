from environments.reinforcement_learning_env import ReinforcementLearningEnvironment
from data_processing.feature_generation import FeatureGenerator
from data_processing.csv_reader import CSVReader
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
    
    def make(self, ticker, folder_path, prediction_period=1):
        # Check if the ticker symbol is valid
        stock_ticker_guard(ticker)

        path_to_file = folder_path + f"{ticker.upper()}.csv"
        csvr = CSVReader(path=path_to_file)
        fg = FeatureGenerator(csvr.get_df(), task="custom")
    
        fg._create_classes(period=prediction_period)
        fg.apply_IG_top_ten_indicators()
        self.X_train, self.X_test, self.y_train, self.y_test = fg.create_modelling_data(features=["Volume", "Adj Close"], target="Profit")
        
        self.action_space = self.Aspace(len(self.actions), self.sample)
        self.observation_space = self.Obspace((self.X_train.shape[1],))
        print("The following derived features are in use: ", fg.training_features)

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

        if action == self.expected_action:
            reward = 1
        else:
            reward = -1

        try:
            next_state, self.expected_action = next(self.gen)
        except:
            next_state = None
            terminated = True

        return next_state, reward, terminated, truncated, None
        
    def make_portfolio(self):
        pass

    def reset(self, dataset='train'):
        self.make_portfolio()
        self.gen = self.stock_generator(dataset)
        state, self.expected_action = next(self.gen)
        return state, None

    def close(self):
        pass
    
    def sample(self):
        return random.sample(range(len(self.actions)), 1)[0]