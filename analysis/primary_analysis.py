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
ff_reward = {}

for seed in SEEDS:
    run = latest_run("lstm", seed)
    if run:
        lstm_sr[seed] = load(run, "spectral_radius")
        lstm_reward[seed] = load(run, "episode_reward")
    run = latest_run("feedforward", seed)
    if run:
        ff_reward[seed] = load(run, "episode_reward")


# align series to the shortest length across seeds
min_len = min(len(v) for v in lstm_sr.values() if v is not None)
sr_matrix = np.array([lstm_sr[s][:min_len] for s in SEEDS if lstm_sr[s] is not None])

sr_mean = sr_matrix.mean(axis=0)
sr_std  = sr_matrix.std(axis=0)
sr_var  = sr_matrix.var(axis=0)
steps   = np.arange(min_len)


# figure 1: spectral radius mean +/- std across seeds with individual seeds
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle("LSTM Spectral Radius - Primary Analysis", fontsize=13)

ax = axes[0]
for i, seed in enumerate(SEEDS):
    if lstm_sr[seed] is not None:
        ax.plot(lstm_sr[seed][:min_len], color=COLORS[i], alpha=0.4,
                linewidth=0.9, label=f"seed {seed}")
ax.plot(sr_mean, color="black", linewidth=1.8, label="mean")
ax.fill_between(steps, sr_mean - sr_std, sr_mean + sr_std,
                color="black", alpha=0.15, label="±1 std")
ax.set_title("Spectral Radius over Training")
ax.set_xlabel("measurement step")
ax.set_ylabel("rho")
ax.legend(fontsize=7)

ax = axes[1]
ax.plot(sr_var, color="black", linewidth=1.4)
ax.set_title("Spectral Radius Variance over Training")
ax.set_xlabel("measurement step")
ax.set_ylabel("variance")

plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0)
plt.savefig(os.path.join(FIGURES, "spectral_radius_variance.png"), dpi=150)
print("saved spectral_radius_variance.png")
plt.show()


# figure 2: distribution of spectral radius values (all seeds pooled)
sr_pooled = sr_matrix.flatten()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("LSTM Spectral Radius Distribution", fontsize=13)

ax = axes[0]
ax.hist(sr_pooled, bins=40, color="#1f77b4", edgecolor="white", linewidth=0.4)
ax.set_title("Histogram (all seeds pooled)")
ax.set_xlabel("rho")
ax.set_ylabel("count")

ax = axes[1]
stats.probplot(sr_pooled, dist="norm", plot=ax)
ax.set_title("Q-Q Plot vs Normal")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(FIGURES, "distribution_tails.png"), dpi=150)
print("saved distribution_tails.png")
plt.show()


# figure 3: spectral radius vs reward (per seed, aligned by step)
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle("Spectral Radius vs Episode Reward per Seed", fontsize=13)

for i, seed in enumerate(SEEDS):
    ax = axes[i // 2, i % 2]
    sr = lstm_sr[seed]
    er = lstm_reward[seed]
    if sr is not None and er is not None:
        n = min(len(sr), len(er))
        x = np.arange(n)
        ax2 = ax.twinx()
        ax.plot(x, sr[:n], color="#1f77b4", linewidth=1.2, label="spectral radius")
        ax2.plot(x, er[:n], color="#d62728", linewidth=1.2, alpha=0.7, label="reward")
        ax.set_title(f"seed {seed}")
        ax.set_xlabel("step")
        ax.set_ylabel("rho", color="#1f77b4")
        ax2.set_ylabel("reward", color="#d62728")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=4.0, w_pad=3.0)
plt.savefig(os.path.join(FIGURES, "reward_vs_instability.png"), dpi=150)
print("saved reward_vs_instability.png")
plt.show()


# stats summary
print("\nSpectral radius per seed:")
print(f"  {'seed':>5}  {'mean':>8}  {'std':>8}  {'var':>8}  {'max':>8}  {'skew':>8}  {'kurtosis':>10}")
for seed in SEEDS:
    sr = lstm_sr[seed]
    if sr is not None:
        print(f"  {seed:>5}  {sr.mean():>8.4f}  {sr.std():>8.4f}  "
              f"{sr.var():>8.4f}  {sr.max():>8.4f}  "
              f"{stats.skew(sr):>8.4f}  {stats.kurtosis(sr):>10.4f}")

print(f"\nPooled across all seeds:")
print(f"  mean     = {sr_pooled.mean():.4f}")
print(f"  std      = {sr_pooled.std():.4f}")
print(f"  variance = {sr_pooled.var():.4f}")
print(f"  max      = {sr_pooled.max():.4f}")
print(f"  skewness = {stats.skew(sr_pooled):.4f}")
print(f"  kurtosis = {stats.kurtosis(sr_pooled):.4f}  (excess, normal=0)")

shapiro_stat, shapiro_p = stats.shapiro(sr_pooled[:5000])
print(f"\nShapiro-Wilk normality test (n={min(len(sr_pooled), 5000)}):")
print(f"  W={shapiro_stat:.4f}  p={shapiro_p:.4e}")
if shapiro_p < 0.05:
    print("  distribution is non-normal (p < 0.05)")
else:
    print("  cannot reject normality (p >= 0.05)")

print("\nCorrelation between spectral radius and reward per seed:")
for seed in SEEDS:
    sr = lstm_sr[seed]
    er = lstm_reward[seed]
    if sr is not None and er is not None:
        n = min(len(sr), len(er))
        r, p = stats.pearsonr(sr[:n], er[:n])
        print(f"  seed {seed}: r={r:.4f}  p={p:.4e}")
