"""
analysis/condition_number.py

Analyzes the condition_number_{cell,forget,full,input,output}.npy arrays
collected per LSTM seed (agents/LSTM/agent.py::GradientDynamicsRecorder
.compute_condition_number) alongside spectral_radius, at the same
checkpoint cadence.

Methodology's claim under test: "A rising condition number should precede
spectral radius spikes, since high condition number is what makes the
eigenvalues sensitive to tiny perturbations." This script checks that
claim directly via a lagged cross-correlation between condition_number[t]
and spectral_radius[t+lag] (mirrors analysis/lagged_correlation.py's
rho[t]-vs-reward[t+lag] approach), plus a spike-event window analysis
(mirrors reward_around_spikes.png but for condition number instead of
reward).

If condition number is a genuine leading indicator, the lagged
correlation should peak at a small POSITIVE lag (condition number now
predicts a spike shortly after) rather than at lag 0 or negative lags.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, "output")
FIGURES = os.path.join(OUTPUT, "figures")
os.makedirs(FIGURES, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4, 5]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
GATES = ["full", "input", "forget", "cell", "output"]
MAX_LAG = 10
SPIKE_THRESHOLD = 0.05
WINDOW_BEFORE = 5


def latest_run(agent, seed):
    pattern = os.path.join(OUTPUT, agent, "partial_obs", f"seed_{seed}", "*")
    runs = sorted(glob.glob(pattern))
    return runs[-1] if runs else None


def load(run_dir, metric):
    path = os.path.join(run_dir, f"{metric}.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


lstm_sr = {}
lstm_cond = {}  # seed -> {gate: array}

for seed in SEEDS:
    run = latest_run("lstm", seed)
    if run:
        lstm_sr[seed] = load(run, "spectral_radius")
        lstm_cond[seed] = {gate: load(run, f"condition_number_{gate}") for gate in GATES}


# --------------------------------------------------------------------------
# figure 1: condition number trajectories over training, one subplot per gate
# --------------------------------------------------------------------------

fig, axes = plt.subplots(3, 2, figsize=(14, 14))
fig.suptitle("LSTM Condition Number Trajectories (W_hh gate blocks)", fontsize=13)

for gi, gate in enumerate(GATES):
    ax = axes[gi // 2, gi % 2]
    for i, seed in enumerate(SEEDS):
        cn = lstm_cond[seed][gate]
        if cn is not None:
            ax.plot(cn, color=COLORS[i], alpha=0.7, linewidth=1.1, label=f"seed {seed}")
    ax.set_title(f"gate: {gate}")
    ax.set_xlabel("measurement step")
    ax.set_ylabel("condition number")
    ax.legend(fontsize=7)

axes[2, 1].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3.5, w_pad=3.0)
plt.savefig(os.path.join(FIGURES, "condition_number_trajectories.png"), dpi=150)
print("saved condition_number_trajectories.png")
plt.show()


# --------------------------------------------------------------------------
# figure 2: lagged cross-correlation, condition_number_full[t] vs
# spectral_radius[t+lag], per seed -- tests the "leading indicator" claim
# --------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Lagged Cross-Correlation: Condition Number (full) vs Spectral Radius",
             fontsize=13)

all_lag_r = np.zeros((len(SEEDS), MAX_LAG + 1))

for i, seed in enumerate(SEEDS):
    cn = lstm_cond[seed]["full"]
    sr = lstm_sr[seed]
    ax = axes[i // 3, i % 3]

    if cn is None or sr is None:
        continue

    lags = range(0, MAX_LAG + 1)
    rs, ps = [], []
    for lag in lags:
        n = min(len(cn), len(sr)) - lag
        r, p = stats.pearsonr(cn[:n], sr[lag:lag + n])
        rs.append(r)
        ps.append(p)
    rs = np.array(rs)
    ps = np.array(ps)
    all_lag_r[i] = rs

    bars = ax.bar(list(lags), rs, color=[
        "#d62728" if p < 0.05 else "#aec7e8" for p in ps
    ], edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"seed {seed}")
    ax.set_xlabel("lag (steps)")
    ax.set_ylabel("r")
    ax.set_xticks(list(lags))

fig.text(0.5, 0.01, "red bars = p < 0.05  |  condition_number[t] vs spectral_radius[t+lag]",
          ha="center", fontsize=9)
plt.tight_layout(rect=[0, 0.03, 1, 0.96], h_pad=3.0, w_pad=2.0)
plt.savefig(os.path.join(FIGURES, "condition_number_lagged_correlation.png"), dpi=150)
print("saved condition_number_lagged_correlation.png")
plt.show()


mean_lag_r = all_lag_r.mean(axis=0)
std_lag_r = all_lag_r.std(axis=0)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(MAX_LAG + 1), mean_lag_r, yerr=std_lag_r,
       color="#1f77b4", edgecolor="white", linewidth=0.4,
       error_kw={"elinewidth": 1.2, "capsize": 3})
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Mean Lagged Cross-Correlation across Seeds "
              "(condition_number_full[t] vs spectral_radius[t+lag])")
ax.set_xlabel("lag (steps)")
ax.set_ylabel("mean r")
ax.set_xticks(range(MAX_LAG + 1))
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "condition_number_lagged_correlation_mean.png"), dpi=150)
print("saved condition_number_lagged_correlation_mean.png")
plt.show()


# --------------------------------------------------------------------------
# figure 3: condition number in the window BEFORE spectral radius spikes,
# vs its own running baseline -- does condition number visibly rise before
# a spike, rather than merely coincide with or follow it?
# --------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))
window = range(-WINDOW_BEFORE, 4)
all_windows = []

for seed in SEEDS:
    cn = lstm_cond[seed]["full"]
    sr = lstm_sr[seed]
    if cn is None or sr is None:
        continue
    n = min(len(cn), len(sr))
    spike_steps = [t for t in range(n) if sr[t] > SPIKE_THRESHOLD]
    for t in spike_steps:
        w = [t + offset for offset in window]
        if all(0 <= idx < n for idx in w):
            all_windows.append([cn[idx] for idx in w])

if all_windows:
    all_windows = np.array(all_windows)
    mean_window = all_windows.mean(axis=0)
    std_window = all_windows.std(axis=0)
    x = list(window)
    ax.plot(x, mean_window, color="black", linewidth=1.5, label="mean condition number (full)")
    ax.fill_between(x, mean_window - std_window, mean_window + std_window,
                     color="black", alpha=0.15, label="±1 std")
    ax.axvline(0, color="#d62728", linewidth=1.2, linestyle="--", label="spike step")
    ax.set_title(f"Condition Number Around Spectral Radius Spikes "
                 f"(rho > {SPIKE_THRESHOLD}), n={len(all_windows)} events")
    ax.set_xlabel("steps relative to spike")
    ax.set_ylabel("condition number (full W_hh)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "condition_number_around_spikes.png"), dpi=150)
    print("saved condition_number_around_spikes.png")
    plt.show()

    pre_spike = all_windows[:, :WINDOW_BEFORE]
    at_and_after = all_windows[:, WINDOW_BEFORE:]
    t_stat, t_p = stats.ttest_rel(pre_spike.mean(axis=1), at_and_after.mean(axis=1))
    print(f"\nPaired t-test, mean condition number in the {WINDOW_BEFORE} steps "
          f"before vs at/after each spike:")
    print(f"  pre-spike mean:    {pre_spike.mean():.3f}")
    print(f"  at/after-spike mean: {at_and_after.mean():.3f}")
    print(f"  t={t_stat:.4f}  p={t_p:.4e}")
    if t_p < 0.05:
        print("  difference is statistically significant (p < 0.05)")
    else:
        print("  difference is not statistically significant (p >= 0.05)")
else:
    print("\nno spike events found above threshold for window plot")


# --------------------------------------------------------------------------
# summary stats + best-lag check per seed
# --------------------------------------------------------------------------

print("\nCondition number (full W_hh) summary per seed:")
print(f"  {'seed':>5}  {'mean':>10}  {'std':>10}  {'max':>10}  "
      f"{'corr@lag0':>10}  {'best_lag':>9}  {'best_r':>8}")
for i, seed in enumerate(SEEDS):
    cn = lstm_cond[seed]["full"]
    if cn is None:
        continue
    r0 = all_lag_r[i, 0]
    best_lag = int(np.argmax(all_lag_r[i]))
    best_r = all_lag_r[i, best_lag]
    print(f"  {seed:>5}  {cn.mean():>10.3f}  {cn.std():>10.3f}  {cn.max():>10.3f}  "
          f"{r0:>10.4f}  {best_lag:>9d}  {best_r:>8.4f}")

print(f"\nMean best lag across seeds: {np.argmax(mean_lag_r)} "
      f"(peak mean r = {mean_lag_r.max():+.4f})")
if np.argmax(mean_lag_r) > 0:
    print("  Peak correlation occurs at a positive lag -- consistent with condition "
          "number leading spectral radius spikes.")
else:
    print("  Peak correlation occurs at lag 0 -- condition number and spectral radius "
          "move together rather than one clearly leading the other.")

print("\nCondition number per gate, pooled mean across all seeds:")
for gate in GATES:
    pooled = np.concatenate([lstm_cond[s][gate] for s in SEEDS if lstm_cond[s][gate] is not None])
    print(f"  {gate:>8}: mean={pooled.mean():>10.3f}  std={pooled.std():>10.3f}  max={pooled.max():>10.3f}")
