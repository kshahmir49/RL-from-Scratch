import numpy as np
from rl.algos.sac.replay_buffer import ReplayBuffer

def test_replay_buffer_shapes():
    rb = ReplayBuffer(obs_dim=4, act_dim=2, size=100)
    for i in range(50):
        rb.add(np.ones(4)*i, np.ones(2), 1.0, np.ones(4)*(i+1), 0.0)
    batch = rb.sample_batch(32)
    assert batch["obs"].shape == (32, 4)
    assert batch["acts"].shape == (32, 2)
    assert batch["rews"].shape == (32,)
    assert batch["done"].min().item() >= 0.0 and batch["done"].max().item() <= 1.0
