import unittest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(r"C:\Users\YuweiZhu\OneDrive - Alloyed\Documents\Market-Prediction-Research")
sys.path.append(r"C:\Users\YuweiZhu\OneDrive - Alloyed\Documents\Market-Prediction-Research\terminal")
from data_processing.feature_generation import FeatureGenerator


class TestFeatureGenerator(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.data = pd.DataFrame({"Adj Close":[1,2,3,4,5,1,2,3,4,5]})

    def test_classification_feature_generator(self):
        feature_gen = FeatureGenerator(self.data, task="custom")
        feature_gen._create_classes(period=1)
        self.assertTrue((feature_gen.df["Profit"] == pd.Series([1,1,1,1,0,1,1,1,1])).all())
        self.assertTrue((feature_gen.df["Next Close"] == pd.Series([2,3,4,5,1,2,3,4,5])).all())

    def test_apply_sma(self):
        feature_gen = FeatureGenerator(self.data, task="custom")
        res = feature_gen._apply_single_moving_average(period=2)
        self.assertTrue((feature_gen.df["2 day simple moving average"].dropna() == res.dropna()).all())
        self.assertTrue((res.dropna(ignore_index=True) == pd.Series([1.5,2.5,3.5,4.5,3,1.5,2.5,3.5,4.5])).all())

    def test_apply_moving_deviation(self):
        feature_gen = FeatureGenerator(self.data, task="custom")
        res = feature_gen._apply_single_moving_deviation(period=2)
        self.assertTrue((feature_gen.df["2 day moving deviation"].dropna() == res.dropna()).all())
        self.assertTrue(np.allclose(res.dropna(ignore_index=True), pd.Series([0.7071067,0.7071067,0.7071067,0.7071067,2.828427,0.7071067,0.7071067,0.7071067,0.7071067])))

    def test_apply_exponential_moving_average(self):
        feature_gen = FeatureGenerator(self.data, task="custom")
        res = feature_gen._apply_single_exponential_moving_average(period=2)
        self.assertTrue((feature_gen.df["2 day ema"].dropna() == res.dropna()).all())
        self.assertTrue(np.allclose(res.dropna(ignore_index=True), pd.Series([1.5,2.5,3.5,4.5,13/6,37/18,2.685185,3.56172,4.520576])))

    def test_apply_single_momentum(self):
        feature_gen = FeatureGenerator(self.data, task="custom")
        feature_gen._apply_single_momentum(period=1)
        self.assertTrue(np.allclose(feature_gen.df["1 day momentum"].dropna(ignore_index=True), pd.Series([1,0.5,1/3,0.25,-0.8,1,0.5,1/3,0.25])))

    def test_apply_single__moving_cumulation(self):
        feature_gen = FeatureGenerator(self.data, task="custom")
        feature_gen._apply_single_moving_cumulation(period=2)
        self.assertTrue(np.allclose(feature_gen.df["2 day cumulation"].dropna(ignore_index=True), pd.Series([3,5,7,9,6,3,5,7,9])))

if __name__ == "__main__":
    unittest.main()