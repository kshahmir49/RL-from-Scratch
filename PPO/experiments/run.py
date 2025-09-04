from __future__ import annotations
import argparse, os, time, csv
import yaml
import numpy as np
import torch
import torch.optim as optim
from collections import deque

from rl.common.envs import make_env
from rl.common.utils import set_seed, normalize_advantages, explained_variance
from rl.algos.ppo.agent import PPOAgent
from rl.algos.ppo.buffers import RolloutBuffer
from rl.algos.ppo.losses import ppo_loss

def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("num_updates") is None:
        cfg["num_updates"] = cfg["total_timesteps"] // cfg["update_steps"]
    return cfg

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def train_one_seed(cfg, seed, device="cpu"):
    set_seed(seed)
    env = make_env(cfg["env_id"], seed=seed)
    obs_dim = env.observation_space.shape[0]
    assert hasattr(env.action_space, "n"), "This starter only supports discrete action spaces."
    act_dim = env.action_space.n

    agent = PPOAgent(obs_dim, act_dim).to(device)
    opt = optim.Adam(agent.parameters(), lr=cfg["learning_rate"])

    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = f"outputs/{ts}-seed{seed}"
    ensure_dir(outdir)
    csv_path = os.path.join(outdir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step","ep_return","ep_len","value_loss","policy_loss","entropy","approx_kl","clipfrac"])

    obs, _ = env.reset(seed=seed)
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    ep_return = 0.0
    ep_len = 0
    returns_window = deque(maxlen=100)
    global_step = 0

    for update in range(cfg["num_updates"]):
        buffer = RolloutBuffer(obs_dim, cfg["update_steps"], device=device)

        for t in range(cfg["update_steps"]):
            with torch.no_grad():
                action, logp, value = agent.act(obs.unsqueeze(0))
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            buffer.add(obs, action.item(), float(reward), bool(done), value.item(), logp.item())

            ep_return += reward
            ep_len += 1
            global_step += 1

            if done:
                returns_window.append(ep_return)
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([global_step, ep_return, ep_len, "", "", "", "", ""])
                ep_return, ep_len = 0.0, 0
                next_obs, _ = env.reset()

            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            _, last_value = agent(obs.unsqueeze(0))
            last_value = last_value.item()

        buffer.compute_gae(last_value, cfg["gamma"], cfg["gae_lambda"])

        advs, mean_adv, std_adv = normalize_advantages(buffer.advs)

        for epoch in range(cfg["update_epochs"]):
            for mb in buffer.get_minibatches(cfg["batch_size"], shuffle=True):
                mb_obs, mb_acts, mb_oldlogp, mb_oldvals, mb_advs, mb_rets = mb
                loss, info = ppo_loss(
                    agent, mb_obs, mb_acts, mb_oldlogp, mb_oldvals, mb_advs, mb_rets,
                    clip_range=cfg["clip_range"], vf_coef=cfg["vf_coef"], ent_coef=cfg["ent_coef"]
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg["max_grad_norm"])
                opt.step()

        with torch.no_grad():
            _, vals = agent(buffer.obs)
            ev = explained_variance(vals, buffer.returns).item()

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([global_step, "", "", info["value_loss"], info["policy_loss"], info["entropy"], info["approx_kl"], info["clipfrac"]])

        if len(returns_window) == returns_window.maxlen:
            avg100 = np.mean(returns_window)
            if avg100 >= cfg["early_stop_avg"]:
                print(f"[Seed {seed}] Early stop: running avg100={avg100:.1f} at step={global_step}")
                break

        if (update + 1) % cfg["log_interval"] == 0:
            avg = np.mean(returns_window) if returns_window else float("nan")
            print(f"[Seed {seed}] update {update+1}/{cfg['num_updates']} step={global_step} avg100={avg:.1f}")

    ckpt_path = os.path.join(outdir, "checkpoint.pt")
    torch.save({"model": agent.state_dict(), "config": cfg, "seed": seed}, ckpt_path)
    env.close()
    return csv_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    all_csv = []
    for seed in cfg["seed_list"]:
        csv_path = train_one_seed(cfg, seed, device=args.device)
        all_csv.append(csv_path)

    # Plotting helper: optional, run your own notebook/plots as needed.

if __name__ == "__main__":
    main()
