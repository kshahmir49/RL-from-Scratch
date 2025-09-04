from __future__ import annotations
import torch
import torch.nn.functional as F

def ppo_loss(model, obs, acts, old_logps, old_vals, advs, returns, clip_range, vf_coef, ent_coef):
    logits, values = model(obs)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    act_logps = log_probs.gather(1, acts.unsqueeze(1)).squeeze(1)
    entropy = -(probs * log_probs).sum(-1).mean()

    ratio = (act_logps - old_logps).exp()
    unclipped = ratio * advs
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advs
    policy_loss = -torch.min(unclipped, clipped).mean()

    value_loss = F.mse_loss(values, returns)

    with torch.no_grad():
        approx_kl = 0.5 * ((act_logps - old_logps) ** 2).mean()
        clipfrac = (torch.abs(ratio - 1.0) > clip_range).float().mean()

    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    info = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "approx_kl": approx_kl.item(),
        "clipfrac": clipfrac.item(),
    }
    return loss, info
