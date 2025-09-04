from __future__ import annotations
import torch
import torch.nn.functional as F
from rl.common.nets import MLPPolicyValue

class PPOAgent(torch.nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(64,64)):
        super().__init__()
        self.model = MLPPolicyValue(obs_dim, act_dim, hidden_sizes=hidden)

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        logits, value = self.model(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value

    def forward(self, x):
        return self.model(x)
