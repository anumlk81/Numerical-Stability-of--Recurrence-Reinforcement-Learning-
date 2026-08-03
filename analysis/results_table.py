"""
analysis/results_table.py

Consolidates per-seed statistics from every analysis script run so far
(inspect_runs.py, primary_analysis.py, lagged_correlation.py,
condition_number.py, grad_norm_correlation.py) into a single CSV table,
one row per (agent, seed). Pure aggregation of existing .npy output --
recomputes each statistic directly from the saved arrays rather than
parsing prior scripts' printed output, so this stays correct if any
upstream run directory changes.

FF rows leave the recurrence-only columns (spectral_radius,
hidden_state_drift, condition_number, and everything derived from them)
blank -- that data doesn't exist for the feedforward agent yet (see
extra docs/progress_summary.txt, "FF history-window deviation").

Output: output/summary/results_table.csv
"""

import os
import csv
import glob
import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, "output")
SUMMARY = os.path.join(OUTPUT, "summary")
os.makedirs(SUMMARY, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4, 5]
SPIKE_THRESHOLD = 0.05
MAX_LAG = 10


def latest_run(agent, seed):
    pattern = os.path.join(OUTPUT, agent, "partial_obs", f"seed_{seed}", "*")
    runs = sorted(glob.glob(pattern))
    return runs[-1] if runs else None


def load(run_dir, metric):
    if run_dir is None:
        return None
    path = os.path.join(run_dir, f"{metric}.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


def fmt(x, nd=4):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def best_lag_corr(a, b, max_lag=MAX_LAG):
    """Returns (r at lag0, best_lag, best_r) for a[t] vs b[t+lag]."""
    if a is None or b is None:
        return None, None, None
    rs = []
    for lag in range(max_lag + 1):
        n = min(len(a), len(b)) - lag
        if n < 3:
            rs.append(np.nan)
            continue
        r, _ = stats.pearsonr(a[:n], b[lag:lag + n])
        rs.append(r)
    rs = np.array(rs)
    if np.all(np.isnan(rs)):
        return None, None, None
    best_lag = int(np.nanargmax(rs))
    return rs[0], best_lag, rs[best_lag]


rows = []

for agent in ["lstm", "feedforward"]:
    for seed in SEEDS:
        run = latest_run(agent, seed)
        if run is None:
            continue

        reward = load(run, "episode_reward")
        grad_norm = load(run, "grad_norm")
        spectral_radius = load(run, "spectral_radius")
        hidden_drift = load(run, "hidden_state_drift")
        cond_full = load(run, "condition_number_full")

        row = {
            "agent": agent,
            "seed": seed,
            "run_dir": os.path.relpath(run, BASE),
            "n_checkpoints": len(spectral_radius) if spectral_radius is not None else (
                len(grad_norm) if grad_norm is not None else ""),

            "reward_mean": fmt(reward.mean()) if reward is not None else "",
            "reward_final": fmt(reward[-1]) if reward is not None else "",
            "reward_std": fmt(reward.std()) if reward is not None else "",

            "grad_norm_mean": fmt(grad_norm.mean()) if grad_norm is not None else "",
            "grad_norm_std": fmt(grad_norm.std()) if grad_norm is not None else "",

            "spectral_radius_mean": fmt(spectral_radius.mean()) if spectral_radius is not None else "",
            "spectral_radius_std": fmt(spectral_radius.std()) if spectral_radius is not None else "",
            "spectral_radius_max": fmt(spectral_radius.max()) if spectral_radius is not None else "",
            "spectral_radius_skew": fmt(stats.skew(spectral_radius)) if spectral_radius is not None else "",
            "spectral_radius_kurtosis": fmt(stats.kurtosis(spectral_radius)) if spectral_radius is not None else "",
            "n_spikes_rho_gt_0.05": (int((spectral_radius > SPIKE_THRESHOLD).sum())
                                      if spectral_radius is not None else ""),

            "hidden_state_drift_mean": fmt(hidden_drift.mean()) if hidden_drift is not None else "",

            "condition_number_full_mean": fmt(cond_full.mean()) if cond_full is not None else "",
            "condition_number_full_max": fmt(cond_full.max()) if cond_full is not None else "",
        }

        # spectral_radius[t] vs reward[t] correlation (lag 0 only, matches
        # primary_analysis.py's per-seed correlation)
        if spectral_radius is not None and reward is not None:
            n = min(len(spectral_radius), len(reward))
            r, p = stats.pearsonr(spectral_radius[:n], reward[:n])
            row["sr_reward_corr_r"] = fmt(r)
            row["sr_reward_corr_p"] = fmt(p, 4)
        else:
            row["sr_reward_corr_r"] = ""
            row["sr_reward_corr_p"] = ""

        # spectral_radius[t] vs grad_norm[t+lag]: lag0 r + best lag/r
        r0, best_lag, best_r = best_lag_corr(spectral_radius, grad_norm)
        row["sr_gradnorm_corr_lag0_r"] = fmt(r0)
        row["sr_gradnorm_best_lag"] = "" if best_lag is None else best_lag
        row["sr_gradnorm_best_r"] = fmt(best_r)

        # condition_number_full[t] vs spectral_radius[t+lag]: lag0 r + best lag/r
        r0c, best_lag_c, best_r_c = best_lag_corr(cond_full, spectral_radius)
        row["condnum_sr_corr_lag0_r"] = fmt(r0c)
        row["condnum_sr_best_lag"] = "" if best_lag_c is None else best_lag_c
        row["condnum_sr_best_r"] = fmt(best_r_c)

        rows.append(row)

fieldnames = [
    "agent", "seed", "run_dir", "n_checkpoints",
    "reward_mean", "reward_final", "reward_std",
    "grad_norm_mean", "grad_norm_std",
    "spectral_radius_mean", "spectral_radius_std", "spectral_radius_max",
    "spectral_radius_skew", "spectral_radius_kurtosis", "n_spikes_rho_gt_0.05",
    "hidden_state_drift_mean",
    "condition_number_full_mean", "condition_number_full_max",
    "sr_reward_corr_r", "sr_reward_corr_p",
    "sr_gradnorm_corr_lag0_r", "sr_gradnorm_best_lag", "sr_gradnorm_best_r",
    "condnum_sr_corr_lag0_r", "condnum_sr_best_lag", "condnum_sr_best_r",
]

out_path = os.path.join(SUMMARY, "results_table.csv")
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"saved {out_path}  ({len(rows)} rows)")

# quick pooled sanity summary printed to console
print("\nPer-agent pooled reward_mean:")
for agent in ["lstm", "feedforward"]:
    vals = [r["reward_mean"] for r in rows if r["agent"] == agent and r["reward_mean"] != ""]
    if vals:
        print(f"  {agent:>12}: mean of per-seed means = {np.mean(vals):.4f}  (n={len(vals)} seeds)")

print("\nLSTM spectral_radius_mean per seed:")
for r in rows:
    if r["agent"] == "lstm":
        print(f"  seed {r['seed']}: {r['spectral_radius_mean']}")
