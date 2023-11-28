import pandas as pd
import numpy as np
from data_processing.reader import DataReader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class FeatureGenerator:

    def __init__(self, data:pd.DataFrame, task="classification") -> None:
        task = task.lower()
        self.task_guard(task)
        self.df = data
        if task == "classification":
            self._create_classes()
        elif task == "regression":
            self._create_regression_target()
        elif task == "agent_env":
            self._create_env_history()
        self.training_features = []
    
    def _create_env_history(self):
        self.df.drop(columns=["Date", "Close"], inplace=True)

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
            self._apply_single_moving_average(period=p, target=target)

    def _apply_single_moving_average(self, period:int, target="Close"):
        column_name = f"{period} day moving average"
        self.training_features += [column_name]
        self.df[column_name] = col = self.df[target].rolling(period).mean()
        return col

    def _apply_single_moving_deviation(self, period:int, target="Close"):
        column_name = f"{period} day moving deviation"
        #self.training_features += [column_name]
        self.df[column_name] = col = self.df[target].rolling(period).std()
        return col

    def apply_bollinger_bands(self, period:int, target="Close"):
        ma = self._apply_single_moving_average(period=period, target=target)
        std = self._apply_single_moving_deviation(period=period, target=target)
        col_names = [f"{period} day lower bband", f"{period} day upper bband"]
        self.training_features += col_names
        self.df[col_names[0]] = ma - 2*std
        self.df[col_names[1]] = ma + 2*std

    def apply_moving_cumulation(self, periods:list or int, target="Close"):
        periods = list(periods) if isinstance(periods, int) else periods
        for p in periods:
            self._apply_single_moving_cumulation(period=p, target=target)
    
    def _apply_single_moving_cumulation(self, period, target="Close"):
        column_name = f"{period} day cumulation"
        self.training_features += [column_name]
        self.df[column_name] = self.df[target].rolling(period).sum()

    def create_modelling_data(self, features:list, target:str, random_state=123, val_split=0.1, test_split=0.2):
        self.df = self.df.dropna()
        features += self.training_features

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
        assert task.lower() in ["classification", "regression", "agent_env"], \
        "the task specification must be either classification, agent_env or regression"