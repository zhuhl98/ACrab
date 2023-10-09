import torch
import torch.nn as nn
from .util import mlp

# Below are modified from gwthomas/IQL-PyTorch

class TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2, ignore_actions=False):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims, squeeze_output=True)
        self.q2 = mlp(dims, squeeze_output=True)
        self.ignore_actions = ignore_actions

    def both(self, state, action):
        if self.ignore_actions:
            action = action * 0
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state, action):
        return torch.min(*self.both(state, action))

class TwinW(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2, w_l_2_bound = 50, w_l_infty_bound = 200, ignore_actions=False):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]

        # add ReLU at the last layer to make sure w is non-negative
        self.w1 = mlp(dims, squeeze_output=True, output_activation=nn.ReLU)
        self.w2 = mlp(dims, squeeze_output=True, output_activation=nn.ReLU)

        # # allow w to be negative
        # self.w1 = mlp(dims, squeeze_output=True)
        # self.w2 = mlp(dims, squeeze_output=True)

        self.w_l_2_bound = w_l_2_bound
        self.w_l_infty_bound = w_l_infty_bound
        self.ignore_actions = ignore_actions

    def norm_constraint(self, w_pred_raw):
        # norm = torch.norm(w_pred_raw)
        # if norm > self.w_l_2_bound:
        #     ratio = torch.div(norm, self.w_l_2_bound)
        #     w_pred_raw = torch.div(w_pred_raw, ratio)
        w_pred = torch.clip(w_pred_raw, max = self.w_l_infty_bound, min = 0)
        return w_pred

    def forward(self, state, action):
        if self.ignore_actions:
            action = action * 0
        sa = torch.cat([state, action], 1)
        # return self.norm_constraint(self.w1(sa)), self.norm_constraint(self.w2(sa))
        return self.w1(sa), self.w2(sa)

class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, squeeze_output=True)
    
    def forward(self, state):
        return self.v(state)