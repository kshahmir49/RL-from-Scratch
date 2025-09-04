# PPO from scratch (PyTorch)

A clean, reproducible implementation of PPO (discrete actions) with GAE, advantage normalization, gradient clipping, and basic experiment hygiene for **CartPole-v1**.

## Quickstart

### 1) Create an environment
```bash
conda create -n rlppo python=3.10 -y
conda activate rlppo
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Reproduce a CartPole run
```bash
python experiments/run.py --config configs/ppo_cartpole.yaml
```
