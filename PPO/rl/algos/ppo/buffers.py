from __future__ import annotations
import torch

class RolloutBuffer:
    def __init__(self, obs_dim, buf_size, device):
        self.obs = torch.zeros((buf_size, obs_dim), dtype=torch.float32, device=device)
        self.acts = torch.zeros((buf_size,), dtype=torch.long, device=device)
        self.rews = torch.zeros((buf_size,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buf_size,), dtype=torch.bool, device=device)
        self.vals = torch.zeros((buf_size,), dtype=torch.float32, device=device)
        self.logps = torch.zeros((buf_size,), dtype=torch.float32, device=device)
        self.ptr = 0
        self.max = buf_size
        self.device = device
        self.advs = None
        self.returns = None

    def add(self, obs, act, rew, done, val, logp):
        assert self.ptr < self.max, "Buffer overflow"
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.dones[self.ptr] = done
        self.vals[self.ptr] = val
        self.logps[self.ptr] = logp
        self.ptr += 1

    def compute_gae(self, last_value, gamma, lam):
        adv = torch.zeros_like(self.rews)
        lastgaelam = 0.0
        for t in reversed(range(self.max)):
            nonterminal = 1.0 - self.dones[t].float()
            next_value = last_value if t == self.max - 1 else self.vals[t+1]
            delta = self.rews[t] + gamma * next_value * nonterminal - self.vals[t]
            lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
            adv[t] = lastgaelam
        self.advs = adv
        self.returns = adv + self.vals

    def get_minibatches(self, batch_size, shuffle=True):
        idxs = torch.arange(self.max, device=self.device)
        if shuffle:
            idxs = idxs[torch.randperm(self.max)]
        for start in range(0, self.max, batch_size):
            end = start + batch_size
            mb_idx = idxs[start:end]
            yield (self.obs[mb_idx],
                   self.acts[mb_idx],
                   self.logps[mb_idx],
                   self.vals[mb_idx],
                   self.advs[mb_idx],
                   self.returns[mb_idx])
