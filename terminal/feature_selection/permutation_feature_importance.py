from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import *

class PermutationFeatureImportance:

    def __init__(self,
                 data : pd.DataFrame,
                 training_features : List[str],
                 target : str,
                 model) -> None:
        self.data = data
        self.training_features = training_features
        self.target = target
        self.model = model

    def calculate(self):
        X = self.data[self.training_features].to_numpy()
        y = self.data[self.target].to_numpy()

        scaler = StandardScaler()
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, shuffle=False, test_size=0.2)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = self.model().fit(X_train, y_train)
        self.res = res = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=123)
        return res
    
    def plot(self):
        plt.figure()
        plt.barh(self.training_features, self.res.importances_mean, xerr=self.res.importances_std, linewidth=1, capsize=5)
        plt.show()