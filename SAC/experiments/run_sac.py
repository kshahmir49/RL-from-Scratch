from __future__ import annotations
import argparse, os, time, csv
import yaml
import numpy as np
import torch
import torch.optim as optim
from collections import deque
import gymnasium as gym
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from rl.algos.sac.nets import SquashedGaussianActor, TwinQ
from rl.algos.sac.replay_buffer import ReplayBuffer

def set_seed_everywhere(seed: int):
    import random, os
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

@torch.no_grad()
def eval_policy(env_id, actor, device, episodes=5, seed=123):
    env = gym.make(env_id)
    env.reset(seed=seed)
    total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mu, std, log_std = actor.forward(obs_t)
            action = torch.tanh(mu)
            obs, rew, term, trunc, _ = env.step(action.squeeze(0).cpu().numpy())
            done = term or trunc
            ep_ret += rew
        total += ep_ret
    env.close()
    return total / episodes

def sac_update(actor, qnets, target_qnets, rb, batch_size, gamma, tau,
               actor_opt, q_opt, log_alpha, alpha_opt, target_entropy, device):
    batch = rb.sample_batch(batch_size)
    obs = batch["obs"]; acts = batch["acts"]; rews = batch["rews"].unsqueeze(-1)
    next_obs = batch["next_obs"]; done = batch["done"].unsqueeze(-1)

    with torch.no_grad():
        next_a, next_logp, _ = actor.sample(next_obs)
        tq1, tq2 = target_qnets(next_obs, next_a)
        tmin = torch.min(tq1, tq2).unsqueeze(-1)
        alpha = log_alpha.exp()
        target_q = rews + gamma * (1 - done) * (tmin - alpha * next_logp)

    q1, q2 = qnets(obs, acts)
    q1 = q1.unsqueeze(-1); q2 = q2.unsqueeze(-1)
    q_loss = ((q1 - target_q)**2).mean() + ((q2 - target_q)**2).mean()
    q_opt.zero_grad(set_to_none=True); q_loss.backward(); q_opt.step()

    new_a, logp, _ = actor.sample(obs)
    q1_pi, q2_pi = qnets(obs, new_a)
    q_pi = torch.min(q1_pi, q2_pi).unsqueeze(-1)
    alpha = log_alpha.exp()
    actor_loss = (alpha * logp - q_pi).mean()
    actor_opt.zero_grad(set_to_none=True); actor_loss.backward(); actor_opt.step()

    alpha_loss = torch.tensor(0.0, device=device)
    if alpha_opt is not None:
        alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
        alpha_opt.zero_grad(set_to_none=True); alpha_loss.backward(); alpha_opt.step()

    with torch.no_grad():
        for p, tp in zip(qnets.parameters(), target_qnets.parameters()):
            tp.data.mul_(1 - tau); tp.data.add_(tau * p.data)

    return {
        "q_loss": q_loss.item(),
        "actor_loss": actor_loss.item(),
        "alpha": log_alpha.exp().item(),
        "alpha_loss": alpha_loss.item() if hasattr(alpha_loss, "item") else float(alpha_loss),
    }

def train_one_seed(cfg, seed, device="cpu"):
    set_seed_everywhere(seed)
    env = gym.make(cfg["env_id"]); env.reset(seed=seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = SquashedGaussianActor(obs_dim, act_dim).to(device)
    qnets = TwinQ(obs_dim, act_dim).to(device)
    target_qnets = TwinQ(obs_dim, act_dim).to(device); target_qnets.load_state_dict(qnets.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=cfg["actor_lr"])
    q_opt = optim.Adam(qnets.parameters(), lr=cfg["critic_lr"])

    if cfg.get("auto_alpha", True):
        log_alpha = torch.tensor(np.log(cfg.get("init_alpha", 0.2)), requires_grad=True, device=device)
        alpha_opt = optim.Adam([log_alpha], lr=cfg["actor_lr"])
        target_entropy = - cfg.get("target_entropy_scale", 1.0) * act_dim
    else:
        log_alpha = torch.tensor(np.log(cfg.get("init_alpha", 0.2)), requires_grad=False, device=device)
        alpha_opt = None
        target_entropy = -act_dim

    rb = ReplayBuffer(obs_dim, act_dim, size=cfg["replay_size"], device=device)

    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = f"outputs/{ts}-seed{seed}"; os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step","ep_return","ep_len","q_loss","actor_loss","alpha","alpha_loss"])

    obs, _ = env.reset(seed=seed)
    ep_return, ep_len = 0.0, 0
    returns_window = deque(maxlen=100)

    total_steps = cfg["total_env_steps"]
    start_steps = cfg["start_steps"]
    update_after = cfg["update_after"]
    update_every = cfg["update_every"]
    updates_per_round = cfg["updates_per_round"]
    batch_size = cfg["batch_size"]
    gamma = cfg["gamma"]
    tau = cfg["tau"]

    for step in range(1, total_steps + 1):
        if step <= start_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                a, _, _ = actor.sample(obs_t)
                action = a.squeeze(0).cpu().numpy()

        next_obs, rew, term, trunc, _ = env.step(action)
        done = term or trunc
        rb.add(obs, action, rew, next_obs, float(done))

        ep_return += rew; ep_len += 1
        if done:
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step, ep_return, ep_len, "", "", "", ""])
            ep_return, ep_len = 0.0, 0
            next_obs, _ = env.reset()
        obs = next_obs

        if step >= update_after and step % update_every == 0 and rb.size >= batch_size:
            for _ in range(updates_per_round):
                info = sac_update(actor, qnets, target_qnets, rb, batch_size, gamma, tau,
                                  actor_opt, q_opt, log_alpha, alpha_opt, target_entropy, device)
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step, "", "", info["q_loss"], info["actor_loss"], info["alpha"], info["alpha_loss"]])

        if step % cfg["log_interval"] == 0:
            avg = float(np.mean(returns_window)) if returns_window else float("nan")
            print(f"[Seed {seed}] step={step} avg100={avg:.1f} alpha={log_alpha.exp().item():.3f}")

    torch.save({
        "actor": actor.state_dict(),
        "qnets": qnets.state_dict(),
        "target_qnets": target_qnets.state_dict(),
        "log_alpha": log_alpha.detach().cpu().numpy(),
        "config": cfg, "seed": seed
    }, os.path.join(outdir, "checkpoint.pt"))
    env.close()
    return csv_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    all_csv = []
    for seed in cfg["seed_list"]:
        csv_path = train_one_seed(cfg, seed, device=args.device)
        all_csv.append(csv_path)
    print("[done] CSVs:", all_csv)

if __name__ == "__main__":
    main()
