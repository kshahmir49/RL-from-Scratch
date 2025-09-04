import torch
from rl.common.utils import compute_gae

def test_gae_toy():
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values  = torch.tensor([0.5, 0.5, 0.5])
    dones   = torch.tensor([False, False, True])
    last_v  = 0.0
    gamma, lam = 0.99, 0.95

    adv, ret = compute_gae(rewards, values, dones, last_v, gamma, lam)
    assert adv.shape == (3,)
    assert ret.shape == (3,)
    assert torch.allclose(adv[-1], rewards[-1] - values[-1], atol=1e-5)
    assert torch.allclose(ret, adv + values, atol=1e-5)
