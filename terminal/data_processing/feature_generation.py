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
        self._remove_unused_columns()
        if task == "classification":
            self._create_classes()
        elif task == "regression":
            self._create_regression_target()
        self.training_features = []
    
    def _remove_unused_columns(self, unused={"Date", "Close", "Open", "High", "Low"}):
        for col in self.df.columns:
            if col in unused:
                self.df.drop(columns=[col], inplace=True)

    def _create_regression_target(self, reg_tgt="log pct returns"):
        if reg_tgt == "log pct returns":
            self.df["returns"] = self.df["Adj Close"].pct_change()
            self.df["log_returns"] = np.log(1 + self.df["returns"])
        elif reg_tgt == "returns":
            self.df["returns"] = self.df["Adj Close"].pct_change()
    
    def _create_classes(self, period=1):
        self.df["Next Close"] = self.df["Adj Close"].shift(period)
        self.df["Difference"] = self.df["Next Close"] - self.df["Adj Close"]
        self.df["Profit"] = (self.df["Difference"] > 0).astype(int)
        self.df["Profit"] = self.df["Profit"].shift(-period)
        self.df.drop(self.df.tail(period).index,inplace=True)

    def apply_IG_top_ten_indicators(self):
        # bbands
        self.apply_bollinger_bands(period=20)

        # moving averages
        periods = [50,200]
        self.apply_moving_averages(periods=periods)

        # emas
        periods = [12, 26, 50, 200]
        self.apply_exponential_moving_average(periods=periods)

    def apply_moving_averages(self, periods:list or int, target="Adj Close"):        
        periods = list(periods) if isinstance(periods, int) else periods
        for p in periods:
            self._apply_single_moving_average(period=p, target=target)

    def _apply_single_moving_average(self, period:int, target="Adj Close"):
        column_name = f"{period} day moving average"
        self.training_features += [column_name]
        self.df[column_name] = col = self.df[target].rolling(period).mean()
        return col

    def _apply_single_moving_deviation(self, period:int, target="Adj Close"):
        column_name = f"{period} day moving deviation"
        #self.training_features += [column_name]
        self.df[column_name] = col = self.df[target].rolling(period).std()
        return col

    def apply_bollinger_bands(self, period:int, target="Adj Close"):
        ma = self._apply_single_moving_average(period=period, target=target)
        std = self._apply_single_moving_deviation(period=period, target=target)
        col_names = [f"{period} day lower bband", f"{period} day upper bband"]
        self.training_features += col_names
        self.df[col_names[0]] = ma - 2*std
        self.df[col_names[1]] = ma + 2*std
    
    def apply_exponential_moving_average(self, periods:list or int, target="Adj Close"):        
        periods = list(periods) if isinstance(periods, int) else periods
        for p in periods:
            self.apply_single_exponential_moving_average(period=p, target=target)

    def apply_single_exponential_moving_average(self, period:int, target="Adj Close"):
        column_name = f"{period} day ema"
        self.training_features += [column_name]
        self.df[column_name] = self.df[target].ewm(span=period, adjust=False).mean()

    def apply_momentum(self, period:int, target="Adj Close"):
        periods = list(periods) if isinstance(periods, int) else periods
        for p in periods:
            self.apply_single_momentum(period=p, target=target)

    def apply_single_momentum(self, period:int, target="Adj Close"):
        column_name = f"{period} day momentum"
        self.training_features += [column_name]
        self.df[column_name] = self.df[target].pct_change(period)

    def apply_moving_cumulation(self, periods:list or int, target="Adj Close"):
        periods = list(periods) if isinstance(periods, int) else periods
        for p in periods:
            self._apply_single_moving_cumulation(period=p, target=target)
    
    def _apply_single_moving_cumulation(self, period, target="Adj Close"):
        column_name = f"{period} day cumulation"
        self.training_features += [column_name]
        self.df[column_name] = self.df[target].rolling(period).sum()

    def create_modelling_data(self, features:list, target:str, random_state=123, val_split=0.1, test_split=0.2):
        self.df = self.df.dropna()
        features += self.training_features

        X = self.df[features].to_numpy()
        y = self.df[target].to_numpy()

        scaler = StandardScaler()
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_split)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        self.scaler = scaler

        return X_train, X_test, y_train, y_test

    @staticmethod
    def task_guard(task):
        assert task.lower() in ["classification", "regression", "custom"], \
        "the task specification must be either classification, regression or custom"