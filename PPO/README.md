# RL Assignment 1 — PPO from scratch (PyTorch)

A clean, reproducible implementation of PPO (discrete actions) with GAE, advantage normalization, gradient clipping, and basic experiment hygiene for **CartPole-v1** (and optional LunarLander-v2).

## Quickstart

### 1) Create an environment
```bash
# conda (recommended)
conda create -n rlppo python=3.10 -y
conda activate rlppo

# or with venv
python -m venv .venv && source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
# For LunarLander later: pip install "gymnasium[box2d]"
```

### 3) Reproduce a CartPole run
```bash
python experiments/run.py --config configs/ppo_cartpole.yaml
```
