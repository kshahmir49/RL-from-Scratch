from __future__ import annotations
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int, device: str = "cpu"):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((size,), dtype=np.float32)
        self.done_buf = np.zeros((size,), dtype=np.float32)
        self.max_size = size
        self.ptr = 0
        self.size = 0
        self.device = device

    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs = torch.as_tensor(self.obs_buf[idxs], device=self.device),
            acts = torch.as_tensor(self.acts_buf[idxs], device=self.device),
            rews = torch.as_tensor(self.rews_buf[idxs], device=self.device),
            next_obs = torch.as_tensor(self.next_obs_buf[idxs], device=self.device),
            done = torch.as_tensor(self.done_buf[idxs], device=self.device),
        )
        return batch
