from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC

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
from utils.guards import non_linearity_guard

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=4, non_linearity='none') -> None:
        super().__init__()
        self.activation = non_linearity_guard(non_linearity)
        self.simple_net = nn.Sequential(OrderedDict([
                ('projection_1', nn.Linear(input_size, hidden_size)),
                ('non_linearity_1', self.activation),
                ('projection_2', nn.Linear(hidden_size, output_size))
                ]))
    def forward(self, x):
        return self.simple_net(x)

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

            print(feats[0])

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
    

class NNWorldModel(WorldModel):

    def __init__(self, device="cpu") -> None:
        super().__init__()


        self.device = device
        self.state_transition = SimpleNN(6, 4, hidden_size=8, non_linearity='gelu').to(self.device)
        self.reward_transition = SimpleNN(6, 1, hidden_size=8, non_linearity='gelu').to(self.device)
        self.terminal_transition = SimpleNN(6, 1, hidden_size=4, non_linearity='gelu').to(self.device)

        self.enc = OneHotEncoder(sparse_output=False)
        self.to_np = lambda x : x.to("cpu").detach().numpy()
        self.update_counter = 0
        self.loss_fn = nn.MSELoss()
        self.state_opt = torch.optim.Adam(self.state_transition.parameters(), lr=0.001)
        self.reward_opt = torch.optim.Adam(self.reward_transition.parameters(), lr=0.001)
        self.term_opt = torch.optim.Adam(self.terminal_transition.parameters(), lr=0.001)

    def observe(self, states, actions, next_states, rewards):
        self.state_transition.train()
        self.terminal_transition.train()
        self.reward_transition.train()

        non_terminal_mask = torch.tensor(tuple(map(lambda ns:ns is not None, next_states)), dtype=torch.bool, device=self.device)

        if self.update_counter == 0:
            actions = torch.tensor(self.enc.fit_transform(self.to_np(actions)), device=self.device)
            self.update_counter = -1
        else:
            actions = torch.tensor(self.enc.transform(self.to_np(actions)), device=self.device)

        nts = states[non_terminal_mask]
        ntns = torch.stack([x for x in next_states if x is not None], dim=0).to(self.device)
        nta = actions[non_terminal_mask]

        ntf = torch.cat([nts, nta], dim=-1)
        feats = torch.cat([states, actions], dim=-1)

        self.state_opt.zero_grad()
        self.term_opt.zero_grad()
        self.reward_opt.zero_grad()
        
        state_loss = self.loss_fn(self.state_transition(ntf.float()), ntns.float())
        terminal_loss = self.loss_fn(self.terminal_transition(feats.float()).squeeze(), non_terminal_mask.float())
        reward_loss = self.loss_fn(self.reward_transition(feats.float()).squeeze(), rewards.squeeze())

        state_loss.backward()
        terminal_loss.backward()
        reward_loss.backward()

        self.state_opt.step()
        self.term_opt.step()
        self.reward_opt.step()

    def predict(self, states, actions):
        self.state_transition.eval()
        self.terminal_transition.eval()
        self.reward_transition.eval()

        actions = torch.tensor(self.enc.transform(self.to_np(actions)), device=self.device)
        feats = torch.cat([states, actions], dim=1)

        next_state_pred = self.state_transition(feats.float())
        rewards_pred = F.sigmoid(self.reward_transition(feats.float()))
        non_terminal_states = F.sigmoid(self.terminal_transition(feats.float())).squeeze() >= 0.5

        # TODO: catch exception where all are predicted as terminal
        if non_terminal_states.any():
            next_state_pred = [next_state_pred[i] if non_terminal_states[i] else None for i in range(len(non_terminal_states))]
        else:
            next_state_pred = [next_state_pred[i] for i in range(len(non_terminal_states))]

        return next_state_pred, rewards_pred.squeeze()
    
    def test(self, states, actions, next_states, rewards):
        self.state_transition.eval()
        self.terminal_transition.eval()
        self.reward_transition.eval()

        actions = torch.tensor(self.enc.transform(self.to_np(actions)), device=self.device)
        feats = torch.cat([states, actions], dim=1)
        non_terminal_mask = torch.tensor(tuple(map(lambda ns:ns is not None, next_states)), dtype=torch.bool, device=self.device)
        ntns = torch.stack([x for x in next_states if x is not None], dim=0).to(self.device)

        next_state_pred = self.state_transition(feats.float())[non_terminal_mask]
        non_terminal_states = F.sigmoid(self.terminal_transition(feats.float())).squeeze() >= 0.5
        reward_pred = self.reward_transition(feats.float()).squeeze()

        score = R2Score()
        score.update(next_state_pred, ntns)
        r2_state = score.compute().item()

        # score = R2Score()
        # score.update(reward_pred, rewards.squeeze())
        # r2_reward = score.compute().item()

        acc = (non_terminal_states == non_terminal_mask).sum()/len(non_terminal_mask)

        #print(f"The r2 score for state prediction is {r2}, and the accuracy of terminal episode is {acc}")

        return f"r2 for state:{r2_state}, r2 for reward:{0}, accuracy:{acc}"