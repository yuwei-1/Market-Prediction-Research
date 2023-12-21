import pandas as pd
import numpy as np
import sys
sys.path.append(r"C:\Users\YuweiZhu\OneDrive - Alloyed\Documents\Market-Prediction-Research\terminal")
from data_processing.reader import DataReader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class FeatureGenerator:

    def __init__(self, data:pd.DataFrame, task="classification", target_variable="Adj Close") -> None:
        task = task.lower()
        self.task_guard(task)
        self.df = data
        self.target_variable = target_variable
        self._remove_unused_columns()
        if task == "classification":
            self._create_classes()
        elif task == "regression":
            self._create_regression_target()
        self.training_features = []
    
    def _remove_unused_columns(self, unused={"Date", "Open", "High", "Low", "Dividends", "Stock Splits"}):
        for col in self.df.columns:
            if col in unused:
                self.df.drop(columns=[col], inplace=True)

    def _create_regression_target(self, reg_tgt="log pct returns"):
        if reg_tgt == "log pct returns":
            self.df["returns"] = self.df[self.target_variable].pct_change()
            self.df["log_returns"] = np.log(1 + self.df["returns"])
        elif reg_tgt == "returns":
            self.df["returns"] = self.df[self.target_variable].pct_change()
    
    def _create_classes(self, period=1):
        self.df["Next Close"] = self.df[self.target_variable].shift(-period)
        self.df["Difference"] = self.df["Next Close"] - self.df[self.target_variable]
        self.df["Profit"] = (self.df["Difference"] > 0).astype(int)
        #self.df["Profit"] = self.df["Profit"].shift(period)
        self.df.drop(self.df.tail(period).index,inplace=True)

    def apply_IG_top_ten_indicators(self):
        # bbands
        self.apply_bollinger_bands(period=7)

        # moving averages
        periods = [7, 14, 30, 100]
        self.apply_moving_averages(periods=periods)

        # emas
        periods = [7, 14, 30, 100]
        self.apply_exponential_moving_average(periods=periods)

        # RSI
        self.apply_relative_strength_index(period=7)

        # MACD
        self.apply_MACD_indicator()

    def apply_relative_strength_index(self, period=14):
        column_name = f"{period} day RSI"
        if "returns" not in self.df.columns:
            self.df["returns"] = self.df[self.target_variable].pct_change()
        self.df["gains"] = np.where(self.df["returns"] > 0, self.df["returns"], 0)
        self.df["loss"] = np.where(self.df["returns"] < 0, np.abs(self.df["returns"]), 0)
        self.df[f"{period} day average gains"] = self.df["gains"].rolling(period).mean()
        self.df[f"{period} day average loss"] = self.df["loss"].rolling(period).mean()
        relative_strength = self.df[f"{period} day average gains"] / self.df[f"{period} day average loss"]
        self.df[column_name] = 100 - (100/(1 + relative_strength))

    def apply_MACD_indicator(self, long_term=26, short_term=12, signal=9):
        long_ema = self.apply_single_exponential_moving_average(period=long_term)
        short_ema = self.apply_single_exponential_moving_average(period=short_term)
        self.df[f"{long_term}/{short_term} macd"] = macd = short_ema - long_ema
        self.df[f"{long_term}/{short_term} macd {signal} day signal"] = macd.ewm(span=signal, adjust=False).mean()
        self.training_features += [f"{long_term}/{short_term} macd", f"{long_term}/{short_term} macd {signal} day signal"]

    def apply_moving_averages(self, periods:list or int):        
        periods = list(periods) if isinstance(periods, int) else periods
        for p in periods:
            self._apply_single_moving_average(period=p)

    def _apply_single_moving_average(self, period:int):
        column_name = f"{period} day simple moving average"
        self.training_features += [column_name]
        self.df[column_name] = col = self.df[self.target_variable].rolling(period).mean()
        return col

    def _apply_single_moving_deviation(self, period:int):
        column_name = f"{period} day moving deviation"
        self.training_features += [column_name]
        self.df[column_name] = col = self.df[self.target_variable].rolling(period).std()
        return col

    def apply_bollinger_bands(self, period:int):
        ma = self._apply_single_moving_average(period=period)
        std = self._apply_single_moving_deviation(period=period)
        col_names = [f"{period} day lower bband", f"{period} day upper bband"]
        self.training_features += col_names
        self.df[col_names[0]] = ma - 2*std
        self.df[col_names[1]] = ma + 2*std
    
    def apply_exponential_moving_average(self, periods:list or int):        
        periods = list(periods) if isinstance(periods, int) else periods
        for p in periods:
            self.apply_single_exponential_moving_average(period=p)

    def apply_single_exponential_moving_average(self, period:int):
        column_name = f"{period} day ema"
        self.training_features += [column_name]
        self.df[column_name] = col = self.df[self.target_variable].ewm(span=period, adjust=False).mean()
        return col

    def apply_momentum(self, periods:list):
        periods = list(periods) if isinstance(periods, int) else periods
        for p in periods:
            self.apply_single_momentum(period=p)

    def apply_single_momentum(self, period:int):
        column_name = f"{period} day momentum"
        self.training_features += [column_name]
        self.df[column_name] = self.df[self.target_variable].pct_change(period)

    def apply_moving_cumulation(self, periods:list or int):
        periods = list(periods) if isinstance(periods, int) else periods
        for p in periods:
            self._apply_single_moving_cumulation(period=p)
    
    def _apply_single_moving_cumulation(self, period):
        column_name = f"{period} day cumulation"
        self.training_features += [column_name]
        self.df[column_name] = self.df[self.target_variable].rolling(period).sum()

    def create_modelling_data(self, features:list, target:str, last_n_years=4, random_state=123, test_split=0.2):
        self.df = self.df.dropna()
        latest = self.df.index[-1]
        earliest = latest - pd.DateOffset(years=last_n_years)
        self.df = self.df[earliest : latest]

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