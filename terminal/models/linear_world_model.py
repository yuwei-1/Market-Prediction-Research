from models.world_models import WorldModel
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torcheval.metrics import R2Score
from utils.guards import non_linearity_guard, shape_guard


class LinearWorldModel(WorldModel):

    def __init__(self, sklearn_regression_model, sklearn_classification_model) -> None:
        super().__init__()
        self.state_transition = sklearn_regression_model()
        self.reward_transition = sklearn_regression_model()
        self.terminal_transition = sklearn_classification_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to_np = lambda x : x.to("cpu").detach().numpy()

    def _create_features(self, states, actions):
        batch_size = actions.shape[0]
        actions = F.one_hot(actions, num_classes=-1).view(batch_size, -1)
        features = torch.cat([states, actions], dim=-1)
        return features
    
    def observe(self, states, actions, next_states, rewards):
        
        features = self.to_np(self._create_features(states, actions))
        non_terminal_mask = np.array(tuple(map(lambda ns:ns is not None, next_states)), dtype=bool)
        non_terminal_features = features[non_terminal_mask, :]
        non_terminal_next_states = self.to_np(torch.stack([x for x in next_states if x is not None], dim=0))    

        self.state_transition.fit(non_terminal_features, non_terminal_next_states)
        self.reward_transition.fit(features, self.to_np(rewards))
        try:
            self.terminal_transition.fit(features, non_terminal_mask)
        except:
            pass

    def predict(self, states, actions):

        features = self.to_np(self._create_features(states, actions))

        next_state = self.state_transition.predict(features)
        rewards = torch.tensor(self.reward_transition.predict(features), requires_grad=True, device=self.device).float()
        non_terminal_next_state = self.terminal_transition.predict(features)

        next_state = [torch.tensor(next_state[i], requires_grad=True, device=self.device).float() if non_terminal_next_state[i] \
                      else None for i in range(non_terminal_next_state.shape[0])]

        return next_state, rewards
    
    # def test(self):
    #     pass
    #     next_state = self.state_transition.predict(features)
    #     rewards = torch.tensor(self.reward_transition.predict(features), requires_grad=True, device=self.device).float()
    #     if len(rewards.shape) == 1:
    #         rewards = rewards.unsqueeze(1)
    #     non_terminal_next_state = self.terminal_transition.predict(features)

    #     next_state = [torch.tensor(next_state[i], requires_grad=True, device=self.device).float() if non_terminal_next_state[i] \
    #                   else None for i in range(non_terminal_next_state.shape[0])]

    #     return next_state, rewards
    
    def test(self):
        pass