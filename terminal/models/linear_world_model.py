from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from models.world_models import WorldModel
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torcheval.metrics import R2Score
from utils.guards import non_linearity_guard, shape_guard


class LinearWorldModel(WorldModel):

    def __init__(self, max_mem, device="cpu") -> None:
        super().__init__()

        self.state_transition = DecisionTreeRegressor()
        #self.reward_transition = DecisionTreeRegressor()
        self.terminal_transition = DecisionTreeClassifier()
        self.device = device
        self.max_mem = max_mem

        self.enc = OneHotEncoder(sparse_output=False)
        self.to_np = lambda x : x.to("cpu").detach().numpy()
        self.update_counter = 0

    
    def observe(self, states, actions, next_states, rewards):
        
        if self.update_counter == 0:
            non_terminal_mask = np.array(list(map(lambda ns:ns is not None, next_states)), dtype=bool)
            next_states = np.array([self.to_np(x) for x in next_states if x is not None])
        
            states = self.to_np(states)
            actions = self.enc.fit_transform(self.to_np(actions))
            rewards = self.to_np(rewards)[non_terminal_mask]
            feats = np.concatenate([states, actions], axis=1)

            ntf = feats[non_terminal_mask, :]

            self.state_transition.fit(ntf, next_states)
            #self.reward_transition.fit(ntf, rewards)
            self.terminal_transition.fit(feats, non_terminal_mask)

            if len(next_states) == self.max_mem:
                self.update_counter = 2000
            else:
                self.update_counter = 200
        else:
            self.update_counter -= 1


    def predict(self, states, actions):
        states = self.to_np(states)
        actions = self.to_np(actions)
        actions = self.enc.transform(actions)

        feats = np.concatenate([states, actions], axis=1)

        next_state_pred = self.state_transition.predict(feats).tolist()
        #next_reward_pred = self.reward_transition.predict(feats)
        non_terminal_states = self.terminal_transition.predict(feats)

        #reward     = np.array([np.multivariate_normal(r, self.reward_covariance) for r in self.lr_reward.predict(X)])

        next_state_pred = [torch.tensor(next_state_pred[i]).to(self.device).float() if non_terminal_states[i] else None for i in range(len(non_terminal_states))]
        return next_state_pred, None