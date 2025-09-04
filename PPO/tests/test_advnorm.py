import torch
from rl.common.utils import normalize_advantages

def test_advantage_normalization():
    adv = torch.tensor([1.0, 2.0, 3.0, 4.0])
    normed, mean, std = normalize_advantages(adv)
    assert abs(normed.mean().item()) < 1e-6
    assert abs(normed.std().item() - 1.0) < 1e-6
    assert abs(mean - adv.mean().item()) < 1e-6
    assert abs(std - adv.std().item()) < 1e-6
