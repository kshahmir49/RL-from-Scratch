from __future__ import annotations
import gymnasium as gym

def make_env(env_id: str, seed: int = 0):
    env = gym.make(env_id)
    env.reset(seed=seed)
    return env
