from environments.reinforcement_learning_env import ReinforcementLearningEnvironment
from data_processing.feature_generation import FeatureGenerator
from data_processing.csv_reader import CSVReader
from utils.guards import stock_ticker_guard
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import random
import pandas as pd

class StockEnvironment(ReinforcementLearningEnvironment):
    '''
    Given the name of a stock, this creates a learning environment
    for a reinforcement learning agent
    '''

    def __init__(self) -> None:
        super().__init__()
    
    def make(self, ticker, folder_path):
        # Check if the ticker symbol is valid
        stock_ticker_guard(ticker)

        # TODO: clean up this
        # Get file from raw data folder
        file_name = f"{ticker.upper()}.csv"
        csvr = CSVReader(path=folder_path + file_name)
        df = csvr.get_df()

        fg = FeatureGenerator(df, task="agent_env")
        fg.apply_IG_top_ten_indicators()

        data = fg.df
        data.dropna(inplace=True)
        self.close_idx = data.columns.get_loc("Adj Close")
        print("The following features are in use: ", data.columns)
        
        data = data.to_numpy()
        scaler = StandardScaler()

        X_train, X_test = train_test_split(data, random_state=123, test_size=0.2)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        self.X_train = X_train
        self.X_test = X_test

        self.action_space = self.Aspace(len(self.actions), self.sample)
        self.observation_space = self.Obspace((X_train.shape[1],))

    def stock_generator(self):
        for i in range(self.X_train.shape[0]):
            yield self.X_train[i, :]
        return None

    def step(self, action):
        next_state = next(self.gen)
        curr_price = next_state[self.close_idx]

        if action == 0:
            # sell
            self.stock_information["qty"] = -1
        elif action == 1:
            # buy
            self.stock_information["qty"] = 1
        else:
            # hold
            self.stock_information["qty"] = 0
        
        pred_result = self.stock_information["qty"]*(curr_price - self.prev_price)

        if pred_result > 0:
            reward = 1
        elif pred_result == 0:
            reward = 0
        else:
            reward = -1
        
        if next_state is None:
            terminated = True
        else:
            terminated = False

        self.prev_price = curr_price
        return next_state.astype(np.float32), reward, terminated
        
    def make_portfolio(self):
        self.stock_information = {"qty":0}

    def reset(self):
        self.make_portfolio()
        self.gen = self.stock_generator()
        state = next(self.gen)
        self.prev_price = state[self.close_idx]
        return state.astype(np.float32)
    
    def sample(self):
        return random.sample(range(len(self.actions)), 1)[0]