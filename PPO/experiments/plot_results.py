import argparse
import glob
import os
import re
from typing import List, Tuple
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def discover_csvs(glob_pat: str) -> List[str]:
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        # Fallback common pattern
        paths = sorted(glob.glob("outputs/*-seed*/metrics.csv"))
    return paths


def parse_seed_from_path(path: str) -> str:
    # Expect parent dir like ".../<timestamp>-seed3/metrics.csv"
    parent = os.path.basename(os.path.dirname(path))
    m = re.search(r"-seed(\d+)", parent)
    return m.group(1) if m else parent


def split_episode_update_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Episode rows have 'ep_return' filled.
    Update rows have losses/diagnostics filled.
    """
    # Normalize column presence
    cols = ["step", "ep_return", "ep_len", "value_loss", "policy_loss", "entropy", "approx_kl", "clipfrac"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    # Identify empties robustly
    ep_mask = df["ep_return"].astype(str).str.strip() != ""
    ep = df.loc[ep_mask, ["step", "ep_return", "ep_len"]].copy()
    up = df.loc[~ep_mask, ["step", "value_loss", "policy_loss", "entropy", "approx_kl", "clipfrac"]].copy()

    # Coerce numeric
    for c in ep.columns:
        ep[c] = pd.to_numeric(ep[c], errors="coerce")
    for c in up.columns:
        up[c] = pd.to_numeric(up[c], errors="coerce")
    return ep.dropna(subset=["step", "ep_return"]), up.dropna(subset=["step"])


def load_all_episode_returns(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[warn] failed to read {p}: {e}")
            continue
        ep, _ = split_episode_update_rows(df)
        if ep.empty:
            continue
        ep["seed"] = parse_seed_from_path(p)
        frames.append(ep[["step", "ep_return", "seed"]])
    if not frames:
        return pd.DataFrame(columns=["step", "ep_return", "seed"])
    return pd.concat(frames, ignore_index=True)


def load_all_updates(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[warn] failed to read {p}: {e}")
            continue
        _, up = split_episode_update_rows(df)
        if up.empty:
            continue
        up = up.copy()
        up["seed"] = parse_seed_from_path(p)
        # Give each row an "update_index" within its seed (monotonic)
        up = up.reset_index(drop=True)
        up["update_idx"] = np.arange(len(up))
        frames.append(up)
    if not frames:
        return pd.DataFrame(
            columns=["step", "value_loss", "policy_loss", "entropy", "approx_kl", "clipfrac", "seed", "update_idx"]
        )
    return pd.concat(frames, ignore_index=True)


def aggregate_by_bins(ep: pd.DataFrame, bins: int = 80) -> pd.DataFrame:
    """
    Bin by 'step' so seeds with different episode timings align.
    Returns columns: step(mean per bin), mean_return, ci95
    """
    if ep.empty:
        return ep
    ep = ep.sort_values("step")
    ep["bin"] = pd.cut(ep["step"], bins=bins, labels=False)
    agg = (
        ep.groupby("bin")
        .agg(
            step=("step", "mean"),
            mean_return=("ep_return", "mean"),
            std=("ep_return", "std"),
            n=("ep_return", "count"),
        )
        .reset_index(drop=True)
    )
    agg["sem"] = agg["std"] / np.sqrt(agg["n"].clip(lower=1))
    agg["ci95"] = 1.96 * agg["sem"]
    return agg


def plot_episodic_returns(agg: pd.DataFrame, out_path: str, title: str):
    if agg.empty:
        print("[warn] no episodic data to plot.")
        return
    plt.figure(figsize=(9, 5.5))
    plt.plot(agg["step"], agg["mean_return"], label="Mean episodic return")
    plt.fill_between(agg["step"], agg["mean_return"] - agg["ci95"], agg["mean_return"] + agg["ci95"], alpha=0.3)
    plt.xlabel("Environment steps")
    plt.ylabel("Episodic return")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[ok] wrote {out_path}")


def plot_per_seed_curves(ep: pd.DataFrame, out_path: str, title: str):
    if ep.empty:
        print("[warn] no episodic data to plot (per-seed).")
        return
    plt.figure(figsize=(9, 5.5))
    for seed, df_s in ep.sort_values("step").groupby("seed"):
        plt.plot(df_s["step"], df_s["ep_return"], alpha=0.6, label=f"seed {seed}")
    plt.xlabel("Environment steps")
    plt.ylabel("Episodic return")
    plt.title(title + " — per-seed")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[ok] wrote {out_path}")


def aggregate_updates(up: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate update-level diagnostics across seeds by update_idx.
    Returns mean and 95% CI for each metric.
    """
    if up.empty:
        return up
    def _agg_col(col):
        g = up.groupby("update_idx")[col]
        mean = g.mean()
        std = g.std()
        n = g.count().clip(lower=1)
        sem = std / np.sqrt(n)
        ci95 = 1.96 * sem
        out = pd.DataFrame({col + "_mean": mean, col + "_ci95": ci95})
        return out

    cols = ["value_loss", "policy_loss", "entropy", "approx_kl", "clipfrac"]
    parts = [_agg_col(c) for c in cols]
    agg = pd.concat(parts, axis=1).reset_index()
    return agg


def plot_updates(agg_up: pd.DataFrame, out_dir: str, prefix: str):
    if agg_up.empty:
        print("[warn] no update diagnostics to plot.")
        return
    os.makedirs(out_dir, exist_ok=True)

    def _one(metric, ylabel):
        plt.figure(figsize=(8.5, 5))
        m = agg_up[f"{metric}_mean"]
        c = agg_up[f"{metric}_ci95"]
        x = agg_up["update_idx"]
        plt.plot(x, m, label=f"mean {metric}")
        plt.fill_between(x, m - c, m + c, alpha=0.3)
        plt.xlabel("Update index")
        plt.ylabel(ylabel)
        plt.title(f"{prefix}: {metric} vs update")
        plt.legend()
        plt.tight_layout()
        p = os.path.join(out_dir, f"{prefix}_{metric}_vs_update.png")
        plt.savefig(p, dpi=160)
        plt.close()
        print(f"[ok] wrote {p}")

    _one("value_loss", "Value loss (MSE)")
    _one("policy_loss", "Policy loss")
    _one("entropy", "Entropy")
    _one("approx_kl", "Approx KL")
    _one("clipfrac", "Clip fraction")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", type=str, default="outputs/*-seed*/metrics.csv", help="Glob for per-seed metrics CSVs")
    ap.add_argument("--bins", type=int, default=80, help="Bins for step-wise aggregation")
    ap.add_argument("--out_dir", type=str, default="reports/figures", help="Where to save figures")
    ap.add_argument("--prefix", type=str, default="ppo_results", help="Filename prefix for figures")
    args = ap.parse_args()

    csvs = discover_csvs(args.glob)
    if not csvs:
        print(f"[error] no CSVs found for pattern: {args.glob}")
        return
    print(f"[info] found {len(csvs)} CSV(s)")

    ep = load_all_episode_returns(csvs)        # step, ep_return, seed
    up = load_all_updates(csvs)                # step, losses, seed, update_idx

    # Aggregate episodic returns by step bins (CI across all seed-episodes per bin)
    agg_ep = aggregate_by_bins(ep, bins=args.bins)
    plot_episodic_returns(agg_ep, os.path.join(args.out_dir, f"{args.prefix}_episodic_return_ci.png"),
                          title=f"{args.prefix}: episodic return (mean ± 95% CI)")

    # Optional: per-seed curves on one chart
    plot_per_seed_curves(ep, os.path.join(args.out_dir, f"{args.prefix}_per_seed_returns.png"),
                         title=f"{args.prefix}")

    # Aggregate and plot update diagnostics
    agg_up = aggregate_updates(up)
    plot_updates(agg_up, args.out_dir, args.prefix)


if __name__ == "__main__":
    main()