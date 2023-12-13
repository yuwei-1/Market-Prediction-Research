import pandas as pd
from typing import *
from sklearn.gaussian_process import GaussianProcessClassifier
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from feature_selection.ifeature_selector import IFeatureSelector
from tqdm import tqdm

class RecursiveFeatureSelection(IFeatureSelector):
    def __init__(self, 
                 data : pd.DataFrame, 
                 training_features : List[str], 
                 target_feature : str, 
                 min_features : int,
                 model = GaussianProcessClassifier) -> None:
        self.data = data
        self.training_features = training_features
        self.target_feature = target_feature
        self.min_features = min_features
        self.model = model

    def plot(self):
        pass

    def run(self):
        scores = []
        removed = []
        features_used = []
        for i in tqdm(range(len(self.training_features) - self.min_features)):
            max_score = 0
            for j, feature in enumerate(self.training_features):
                curr_features = deepcopy(self.training_features)
                removed_feature = curr_features.pop(j)
                score = self._train_model(curr_features)
                if score > max_score:
                    max_score = score
                    best_feat = feature
            self.training_features.remove(best_feat)
            scores += [max_score]
            removed += [best_feat]
            features_used += [deepcopy(self.training_features)]
        return pd.DataFrame({"features_used":features_used, "removed":removed, "scores":scores})

    def _train_model(self, features_in_use):
        X = self.data[features_in_use].to_numpy()
        y = self.data[self.target_feature].to_numpy()

        scaler = StandardScaler()
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, shuffle=False, test_size=0.2)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        mdl = self.model()
        mdl.fit(X_train, y_train)
        r2_score = mdl.score(X_test, y_test)
        return r2_score