from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0
EPS = 1e-6

def mlp(sizes, activation=nn.ReLU, output_activation=None):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1])]
        if act is not None:
            layers += [act()]
    return nn.Sequential(*layers)

class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256,256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden], activation=nn.ReLU, output_activation=nn.ReLU)
        self.mu_layer = nn.Linear(hidden[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden[-1], act_dim)

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std, log_std

    def sample(self, obs):
        mu, std, log_std = self.forward(obs)
        noise = torch.randn_like(mu)
        u = mu + std * noise
        a = torch.tanh(u)
        log_prob = (-0.5 * ((u - mu) / (std + EPS))**2 - log_std - 0.5 * math.log(2*math.pi)).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - a.pow(2) + EPS).sum(dim=-1, keepdim=True)
        return a, log_prob, torch.tanh(mu)

class QFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256,256)):
        super().__init__()
        self.q = mlp([obs_dim + act_dim, *hidden, 1], activation=nn.ReLU, output_activation=None)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        q = self.q(x).squeeze(-1)
        return q

class TwinQ(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256,256)):
        super().__init__()
        self.q1 = QFunction(obs_dim, act_dim, hidden)
        self.q2 = QFunction(obs_dim, act_dim, hidden)

    def forward(self, obs, act):
        return self.q1(obs, act), self.q2(obs, act)
