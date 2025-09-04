from __future__ import annotations
import torch.nn as nn

class MLPPolicyValue(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(64,64)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last, act_dim)
        self.value_head = nn.Linear(last, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="tanh")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.trunk(x)
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        return logits, value
