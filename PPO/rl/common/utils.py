from __future__ import annotations
import random, os
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def explained_variance(y_pred, y_true):
    var_y = torch.var(y_true)
    return torch.tensor(0.0) if var_y == 0 else 1 - torch.var(y_true - y_pred) / var_y

def compute_gae(rewards, values, dones, last_value, gamma: float, lam: float):
    T = len(rewards)
    adv = torch.zeros(T, dtype=torch.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])
        next_value = last_value if t == T-1 else values[t+1]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values
    return adv, returns

def normalize_advantages(adv, eps: float = 1e-8):
    mean = adv.mean()
    std = adv.std()
    return (adv - mean) / (std + eps), mean.item(), std.item()
