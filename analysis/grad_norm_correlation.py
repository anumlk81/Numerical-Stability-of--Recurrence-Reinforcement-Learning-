"""
analysis/grad_norm_correlation.py

Analyzes grad_norm alongside spectral_radius for the LSTM agent, at the
same checkpoint cadence both are logged at
(agents/LSTM/agent.py::GradientDynamicsRecorder). Mirrors
analysis/lagged_correlation.py's reward analysis, but for gradient norm.

Methodology's claim under test: "Gradient norms per update -- the overall
magnitude of the gradient used for each weight update. Spikes here should
correlate with spectral radius spikes if the theoretical predictions
hold" -- because the temporal Jacobian product is exactly what gradients
get multiplied through during BPTT, a high spectral radius at a checkpoint
should co-occur with (or shortly precede) an elevated gradient norm.
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
MAX_LAG = 10
SPIKE_THRESHOLD = 0.05
WINDOW_AFTER = 5


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
lstm_gn = {}

for seed in SEEDS:
    run = latest_run("lstm", seed)
    if run:
        lstm_sr[seed] = load(run, "spectral_radius")
        lstm_gn[seed] = load(run, "grad_norm")


# --------------------------------------------------------------------------
# figure 1: lagged cross-correlation, spectral_radius[t] vs grad_norm[t+lag]
# --------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Lagged Cross-Correlation: Spectral Radius vs Gradient Norm", fontsize=13)

all_lag_r = np.zeros((len(SEEDS), MAX_LAG + 1))

for i, seed in enumerate(SEEDS):
    sr = lstm_sr[seed]
    gn = lstm_gn[seed]
    ax = axes[i // 3, i % 3]

    if sr is None or gn is None:
        continue

    lags = range(0, MAX_LAG + 1)
    rs, ps = [], []
    for lag in lags:
        n = min(len(sr), len(gn)) - lag
        r, p = stats.pearsonr(sr[:n], gn[lag:lag + n])
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

fig.text(0.5, 0.01, "red bars = p < 0.05  |  spectral_radius[t] vs grad_norm[t+lag]",
          ha="center", fontsize=9)
plt.tight_layout(rect=[0, 0.03, 1, 0.96], h_pad=3.0, w_pad=2.0)
plt.savefig(os.path.join(FIGURES, "grad_norm_lagged_correlation.png"), dpi=150)
print("saved grad_norm_lagged_correlation.png")
plt.show()


mean_lag_r = all_lag_r.mean(axis=0)
std_lag_r = all_lag_r.std(axis=0)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(MAX_LAG + 1), mean_lag_r, yerr=std_lag_r,
       color="#1f77b4", edgecolor="white", linewidth=0.4,
       error_kw={"elinewidth": 1.2, "capsize": 3})
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Mean Lagged Cross-Correlation across Seeds (spectral_radius[t] vs grad_norm[t+lag])")
ax.set_xlabel("lag (steps)")
ax.set_ylabel("mean r")
ax.set_xticks(range(MAX_LAG + 1))
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "grad_norm_lagged_correlation_mean.png"), dpi=150)
print("saved grad_norm_lagged_correlation_mean.png")
plt.show()


# --------------------------------------------------------------------------
# threshold analysis: after spike steps (rho > threshold), what happens to
# gradient norm?
# --------------------------------------------------------------------------

print(f"\nThreshold analysis (spike = rho > {SPIKE_THRESHOLD})")
print(f"Tracking grad_norm change over {WINDOW_AFTER} steps after each spike\n")

all_pre = []
all_post = []

for seed in SEEDS:
    sr = lstm_sr[seed]
    gn = lstm_gn[seed]
    if sr is None or gn is None:
        continue

    n = min(len(sr), len(gn))
    spike_steps = [t for t in range(n) if sr[t] > SPIKE_THRESHOLD]

    if not spike_steps:
        print(f"  seed {seed}: no spikes above threshold")
        continue

    pre_gn = []
    post_gn = []

    for t in spike_steps:
        if t > 0 and t + WINDOW_AFTER < n:
            pre_gn.append(gn[t - 1])
            post_gn.append(gn[t + WINDOW_AFTER])

    if not pre_gn:
        continue

    pre = np.mean(pre_gn)
    post = np.mean(post_gn)
    all_pre.extend(pre_gn)
    all_post.extend(post_gn)

    print(f"  seed {seed}: {len(spike_steps)} spike(s) at steps {spike_steps}")
    print(f"    grad_norm before spike:  {pre:.4f}")
    print(f"    grad_norm {WINDOW_AFTER} steps after: {post:.4f}")
    print(f"    change: {post - pre:+.4f}")

if all_pre:
    t_stat, t_p = stats.ttest_rel(all_pre, all_post)
    print(f"\nPooled across all seeds:")
    print(f"  mean grad_norm before spike:        {np.mean(all_pre):.4f}")
    print(f"  mean grad_norm {WINDOW_AFTER} steps after spike: {np.mean(all_post):.4f}")
    print(f"  mean change:                        {np.mean(all_post) - np.mean(all_pre):+.4f}")
    print(f"  paired t-test: t={t_stat:.4f}  p={t_p:.4e}")
    if t_p < 0.05:
        print("  grad_norm change after spike is statistically significant (p < 0.05)")
    else:
        print("  grad_norm change after spike is not statistically significant (p >= 0.05)")


# --------------------------------------------------------------------------
# plot: grad_norm trajectory around spike events (average), same-step
# coincidence view (does grad_norm elevate AT the spike, not just after?)
# --------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))
window = range(-3, WINDOW_AFTER + 1)
all_windows = []

for seed in SEEDS:
    sr = lstm_sr[seed]
    gn = lstm_gn[seed]
    if sr is None or gn is None:
        continue
    n = min(len(sr), len(gn))
    spike_steps = [t for t in range(n) if sr[t] > SPIKE_THRESHOLD]
    for t in spike_steps:
        w = [t + offset for offset in window]
        if all(0 <= idx < n for idx in w):
            all_windows.append([gn[idx] for idx in w])

if all_windows:
    all_windows = np.array(all_windows)
    mean_window = all_windows.mean(axis=0)
    std_window = all_windows.std(axis=0)
    x = list(window)
    ax.plot(x, mean_window, color="black", linewidth=1.5, label="mean grad_norm")
    ax.fill_between(x, mean_window - std_window, mean_window + std_window,
                    color="black", alpha=0.15, label="±1 std")
    ax.axvline(0, color="#d62728", linewidth=1.2, linestyle="--", label="spike step")
    ax.set_title(f"Average Gradient Norm Around Spike Events (rho > {SPIKE_THRESHOLD}), n={len(all_windows)} events")
    ax.set_xlabel("steps relative to spike")
    ax.set_ylabel("grad_norm")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "grad_norm_around_spikes.png"), dpi=150)
    print("\nsaved grad_norm_around_spikes.png")
    plt.show()

    at_spike = all_windows[:, list(window).index(0)]
    baseline = all_windows[:, :list(window).index(0)]  # steps -3..-1
    t_stat2, p2 = stats.ttest_rel(at_spike, baseline.mean(axis=1))
    print(f"\nPaired t-test, grad_norm AT the spike step vs the 3 preceding steps:")
    print(f"  at-spike mean:  {at_spike.mean():.4f}")
    print(f"  baseline mean:  {baseline.mean():.4f}")
    print(f"  t={t_stat2:.4f}  p={p2:.4e}")
    if p2 < 0.05:
        print("  grad_norm is significantly different AT the spike step (p < 0.05) --"
              " consistent with spectral radius and grad_norm spikes co-occurring.")
    else:
        print("  no significant same-step coincidence (p >= 0.05).")
else:
    print("\nno spike events found above threshold for window plot")


# --------------------------------------------------------------------------
# print lagged r table + summary
# --------------------------------------------------------------------------

print("\nLagged cross-correlation r values per seed (spectral_radius[t] vs grad_norm[t+lag]):")
header = "seed  |  " + "  ".join([f"lag{l:02d}" for l in range(MAX_LAG + 1)])
print(header)
print("-" * len(header))
for i, seed in enumerate(SEEDS):
    row = f"  {seed}   |  " + "  ".join([f"{all_lag_r[i, l]:+.3f}" for l in range(MAX_LAG + 1)])
    print(row)
mean_row = " mean |  " + "  ".join([f"{mean_lag_r[l]:+.3f}" for l in range(MAX_LAG + 1)])
print("-" * len(header))
print(mean_row)
