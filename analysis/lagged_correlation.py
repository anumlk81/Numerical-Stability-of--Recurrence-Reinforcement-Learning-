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
lstm_reward = {}

for seed in SEEDS:
    run = latest_run("lstm", seed)
    if run:
        lstm_sr[seed] = load(run, "spectral_radius")
        lstm_reward[seed] = load(run, "episode_reward")


# lagged cross-correlation: r between rho[t] and reward[t+lag] for each lag
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Lagged Cross-Correlation: Spectral Radius vs Reward", fontsize=13)

all_lag_r = np.zeros((len(SEEDS), MAX_LAG + 1))

for i, seed in enumerate(SEEDS):
    sr = lstm_sr[seed]
    er = lstm_reward[seed]
    ax = axes[i // 3, i % 3]

    if sr is None or er is None:
        continue

    lags = range(0, MAX_LAG + 1)
    rs = []
    ps = []
    for lag in lags:
        n = min(len(sr), len(er)) - lag
        r, p = stats.pearsonr(sr[:n], er[lag:lag + n])
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
    ax.set_ylim(-0.6, 0.6)

fig.text(0.5, 0.01, "red bars = p < 0.05", ha="center", fontsize=9)
plt.tight_layout(rect=[0, 0.03, 1, 0.96], h_pad=3.0, w_pad=2.0)
plt.savefig(os.path.join(FIGURES, "lagged_correlation.png"), dpi=150)
print("saved lagged_correlation.png")
plt.show()


# mean lagged cross-correlation across all seeds
mean_lag_r = all_lag_r.mean(axis=0)
std_lag_r = all_lag_r.std(axis=0)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(MAX_LAG + 1), mean_lag_r, yerr=std_lag_r,
       color="#1f77b4", edgecolor="white", linewidth=0.4,
       error_kw={"elinewidth": 1.2, "capsize": 3})
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Mean Lagged Cross-Correlation across Seeds (rho[t] vs reward[t+lag])")
ax.set_xlabel("lag (steps)")
ax.set_ylabel("mean r")
ax.set_xticks(range(MAX_LAG + 1))
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "lagged_correlation_mean.png"), dpi=150)
print("saved lagged_correlation_mean.png")
plt.show()


# threshold analysis: after spike steps (rho > threshold), what happens to reward?
print(f"\nThreshold analysis (spike = rho > {SPIKE_THRESHOLD})")
print(f"Tracking reward change over {WINDOW_AFTER} steps after each spike\n")

all_pre = []
all_post = []

for seed in SEEDS:
    sr = lstm_sr[seed]
    er = lstm_reward[seed]
    if sr is None or er is None:
        continue

    n = min(len(sr), len(er))
    spike_steps = [t for t in range(n) if sr[t] > SPIKE_THRESHOLD]

    if not spike_steps:
        print(f"  seed {seed}: no spikes above threshold")
        continue

    pre_rewards = []
    post_rewards = []

    for t in spike_steps:
        if t > 0 and t + WINDOW_AFTER < n:
            pre_rewards.append(er[t - 1])
            post_rewards.append(er[t + WINDOW_AFTER])

    if not pre_rewards:
        continue

    pre = np.mean(pre_rewards)
    post = np.mean(post_rewards)
    all_pre.extend(pre_rewards)
    all_post.extend(post_rewards)

    print(f"  seed {seed}: {len(spike_steps)} spike(s) at steps {spike_steps}")
    print(f"    reward before spike:  {pre:.4f}")
    print(f"    reward {WINDOW_AFTER} steps after: {post:.4f}")
    print(f"    change: {post - pre:+.4f}")

if all_pre:
    t_stat, t_p = stats.ttest_rel(all_pre, all_post)
    print(f"\nPooled across all seeds:")
    print(f"  mean reward before spike:        {np.mean(all_pre):.4f}")
    print(f"  mean reward {WINDOW_AFTER} steps after spike: {np.mean(all_post):.4f}")
    print(f"  mean change:                     {np.mean(all_post) - np.mean(all_pre):+.4f}")
    print(f"  paired t-test: t={t_stat:.4f}  p={t_p:.4e}")
    if t_p < 0.05:
        print("  reward change after spike is statistically significant (p < 0.05)")
    else:
        print("  reward change after spike is not statistically significant (p >= 0.05)")


# plot: reward trajectory around spike events (average)
fig, ax = plt.subplots(figsize=(10, 5))
window = range(-3, WINDOW_AFTER + 1)
all_windows = []

for seed in SEEDS:
    sr = lstm_sr[seed]
    er = lstm_reward[seed]
    if sr is None or er is None:
        continue
    n = min(len(sr), len(er))
    spike_steps = [t for t in range(n) if sr[t] > SPIKE_THRESHOLD]
    for t in spike_steps:
        w = [t + offset for offset in window]
        if all(0 <= idx < n for idx in w):
            all_windows.append([er[idx] for idx in w])

if all_windows:
    all_windows = np.array(all_windows)
    mean_window = all_windows.mean(axis=0)
    std_window = all_windows.std(axis=0)
    x = list(window)
    ax.plot(x, mean_window, color="black", linewidth=1.5, label="mean reward")
    ax.fill_between(x, mean_window - std_window, mean_window + std_window,
                    color="black", alpha=0.15, label="±1 std")
    ax.axvline(0, color="#d62728", linewidth=1.2, linestyle="--", label="spike step")
    ax.set_title(f"Average Reward Around Spike Events (rho > {SPIKE_THRESHOLD}), n={len(all_windows)} events")
    ax.set_xlabel("steps relative to spike")
    ax.set_ylabel("episode reward")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "reward_around_spikes.png"), dpi=150)
    print("\nsaved reward_around_spikes.png")
    plt.show()
else:
    print("\nno spike events found above threshold for window plot")


# print lagged r table
print("\nLagged cross-correlation r values per seed:")
header = "seed  |  " + "  ".join([f"lag{l:02d}" for l in range(MAX_LAG + 1)])
print(header)
print("-" * len(header))
for i, seed in enumerate(SEEDS):
    row = f"  {seed}   |  " + "  ".join([f"{all_lag_r[i, l]:+.3f}" for l in range(MAX_LAG + 1)])
    print(row)
mean_row = " mean |  " + "  ".join([f"{mean_lag_r[l]:+.3f}" for l in range(MAX_LAG + 1)])
print("-" * len(header))
print(mean_row)
