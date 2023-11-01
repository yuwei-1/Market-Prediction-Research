import pandas as pd
import numpy as np
from data_processing.reader import DataReader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class FeatureGenerator:

    def __init__(self, data:pd.DataFrame, task="classification") -> None:
        self.task_guard(task)
        self.df = data
        if task == "classification":
            self._create_classes()
        elif task == "regression":
            self._create_regression_target()

    def _create_regression_target(self, reg_tgt="log pct returns"):
        if reg_tgt == "log pct returns":
            self.df["returns"] = self.df["Close"].pct_change()
            self.df["log_returns"] = np.log(1 + self.df["returns"])
        elif reg_tgt == "returns":
            self.df["returns"] = self.df["Close"].pct_change()
    
    def _create_classes(self):
        self.df["Next Close"] = self.df["Close"].shift(1)
        self.df["Difference"] = self.df["Close"] - self.df["Next Close"]
        self.df["Profit"] = (self.df["Difference"] > 0).astype(int)

    def apply_moving_averages(self, periods:list or int, target="Close"):        
        periods = list(periods) if isinstance(periods, int) else periods

        for p in periods:
            column_name = f"{p} day moving average"
            self.df[column_name] = self.df[target].rolling(p).mean()
    
    def apply_moving_cumulation(self, periods:list or int, target="Close"):
        periods = list(periods) if isinstance(periods, int) else periods

        for p in periods:
            column_name = f"{p} day cumulation"
            self.df[column_name] = self.df[target].rolling(p).sum()

    def create_modelling_data(self, features:list, target:str, random_state=123, val_split=0.1, test_split=0.2):
        X = self.df[features]
        y = self.df[target]

        scaler = StandardScaler()
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_split)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        self.scaler = scaler

        return X_train, X_test, y_train, y_test

    @staticmethod
    def task_guard(task):
        assert task in ["classification", "regression"], \
        "the task specification must be either classification or regression"