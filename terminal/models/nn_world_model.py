from models.world_models import WorldModel
from models.generic_feed_forward_network_creator import GenericFeedForwardNetwork
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torcheval.metrics import R2Score
from utils.guards import non_linearity_guard, shape_guard


class NNWorldModel(WorldModel):

    def __init__(self, n_actions, state_dim, hidden_size=8, loss=nn.MSELoss(), device="cpu") -> None:
        super().__init__()
        self.reward_size = 1
        self.terminal_size = 1
        self.device = device
        self.n_actions = n_actions

        self.state_transition = GenericFeedForwardNetwork(3, activations=['gelu', 'none'], nodes_per_layer=[n_actions + state_dim, hidden_size, state_dim]).to(self.device)
        self.reward_transition = GenericFeedForwardNetwork(3, activations=['gelu', 'none'], nodes_per_layer=[n_actions + state_dim, hidden_size, self.reward_size]).to(self.device)
        self.terminal_transition = GenericFeedForwardNetwork(3, activations=['gelu', 'sigmoid'], nodes_per_layer=[n_actions + state_dim, hidden_size, self.terminal_size]).to(self.device)

        self.loss_fn = loss
        self.state_opt = torch.optim.Adam(self.state_transition.parameters(), lr=0.01)
        self.reward_opt = torch.optim.Adam(self.reward_transition.parameters(), lr=0.01)
        self.term_opt = torch.optim.Adam(self.terminal_transition.parameters(), lr=0.01)

    def _create_features(self, states, actions):
        batch_size = actions.shape[0]
        actions = F.one_hot(actions, num_classes=self.n_actions).view(batch_size, -1)
        features = torch.cat([states, actions], dim=-1)
        return features

    def observe(self, states, actions, next_states, rewards):
        self.state_transition.train()
        self.terminal_transition.train()
        self.reward_transition.train()

        features = self._create_features(states, actions)
        non_terminal_mask = torch.tensor(tuple(map(lambda ns:ns is not None, next_states)), dtype=torch.bool, device=self.device)
        non_terminal_features = features[non_terminal_mask]
        non_terminal_next_states = torch.stack([x for x in next_states if x is not None], dim=0).to(self.device)
        
        pred_next_state = self.state_transition(non_terminal_features)
        pred_terminal = self.terminal_transition(features).squeeze(-1)
        pred_reward = self.reward_transition(features)

        shape_guard(pred_next_state, non_terminal_next_states)
        shape_guard(pred_terminal, non_terminal_mask)
        shape_guard(pred_reward, rewards)
        self.state_loss = self.loss_fn(pred_next_state, non_terminal_next_states)
        self.terminal_loss = self.loss_fn(pred_terminal, non_terminal_mask.float())
        self.reward_loss = self.loss_fn(pred_reward, rewards)

        self.backprop_network()

    def backprop_network(self):
        self.state_opt.zero_grad()
        self.term_opt.zero_grad()
        self.reward_opt.zero_grad()

        self.state_loss.backward()
        self.terminal_loss.backward()
        self.reward_loss.backward()

        self.state_opt.step()
        self.term_opt.step()
        self.reward_opt.step()

    def predict(self, states, actions):
        self.state_transition.eval()
        self.terminal_transition.eval()
        self.reward_transition.eval()

        features = self._create_features(states, actions)
        next_state_pred = self.state_transition(features)
        rewards_pred = self.reward_transition(features)
        non_terminal_states = F.sigmoid(self.terminal_transition(features)) >= 0.5

        # TODO: catch exception where all are predicted as terminal
        if non_terminal_states.any():
            next_state_pred = [next_state_pred[i] if non_terminal_states[i] else None for i in range(non_terminal_states.shape[0])]
        else:
            next_state_pred = [next_state_pred[i] for i in range(non_terminal_states.shape[0])]
        return next_state_pred, rewards_pred
    
    def test(self, states, actions, next_states, rewards):
        self.state_transition.eval()
        self.terminal_transition.eval()
        self.reward_transition.eval()

        features = self._create_features(states, actions)
        non_terminal_mask = torch.tensor(tuple(map(lambda ns:ns is not None, next_states)), dtype=torch.bool, device=self.device)
        non_terminal_next_state = torch.stack([x for x in next_states if x is not None], dim=0).to(self.device)

        next_state_pred = self.state_transition(features)[non_terminal_mask].squeeze()
        non_terminal_states = F.sigmoid(self.terminal_transition(features).squeeze()) >= 0.5
        reward_pred = self.reward_transition(features).squeeze() >= 0.5

        shape_guard(next_state_pred, non_terminal_next_state)
        shape_guard(reward_pred, rewards)
        shape_guard(non_terminal_states, non_terminal_mask)
        r2_state = self.compute_r2_score(next_state_pred, non_terminal_next_state)
        #r2_reward = self.compute_r2_score(reward_pred, rewards)
        r2_acc = (reward_pred == rewards).sum()/len(rewards)
        acc = (non_terminal_states == non_terminal_mask).sum()/len(non_terminal_mask)

        return f"r2 state:{r2_state}, acc reward:{r2_acc}, acc ep. termination:{acc}"
    
    @staticmethod
    def compute_r2_score(predicted, target):
        score = R2Score()
        score.update(predicted, target)
        r2_score = score.compute().item()
        return r2_score