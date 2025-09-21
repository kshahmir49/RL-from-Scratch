import torch
from rl.algos.sac.nets import SquashedGaussianActor

def test_shapes_and_finiteness():
    obs_dim, act_dim = 3, 2
    actor = SquashedGaussianActor(obs_dim, act_dim)
    obs = torch.zeros((8, obs_dim))
    a, logp, mu_tanh = actor.sample(obs)
    assert a.shape == (8, act_dim)
    assert logp.shape == (8, 1)
    assert mu_tanh.shape == (8, act_dim)
    assert torch.isfinite(logp).all()
