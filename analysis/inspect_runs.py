import os
import glob
import numpy as np
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, "output")

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


lstm_data = {}
ff_data = {}

for seed in SEEDS:
    run = latest_run("lstm", seed)
    if run:
        lstm_data[seed] = {
            "spectral_radius":    load(run, "spectral_radius"),
            "episode_reward":     load(run, "episode_reward"),
            "grad_norm":          load(run, "grad_norm"),
            "hidden_state_drift": load(run, "hidden_state_drift"),
        }

    run = latest_run("feedforward", seed)
    if run:
        ff_data[seed] = {
            "episode_reward": load(run, "episode_reward"),
            "grad_norm":      load(run, "grad_norm"),
        }


fig, axes = plt.subplots(3, 2, figsize=(16, 15))
fig.suptitle("Raw Training Curves (all seeds)", fontsize=13, y=0.98)

ax = axes[0, 0]
for seed, d in lstm_data.items():
    sr = d["spectral_radius"]
    if sr is not None:
        ax.plot(sr, color=COLORS[seed], alpha=0.7, linewidth=1.2, label=f"seed {seed}")
ax.set_title("LSTM - Spectral Radius")
ax.set_xlabel("measurement step")
ax.set_ylabel("rho")
ax.legend(fontsize=7)

ax = axes[0, 1]
for seed, d in lstm_data.items():
    hd = d["hidden_state_drift"]
    if hd is not None:
        ax.plot(hd, color=COLORS[seed], alpha=0.7, linewidth=1.2, label=f"seed {seed}")
ax.set_title("LSTM - Hidden State Drift")
ax.set_xlabel("measurement step")
ax.set_ylabel("L2 drift")
ax.legend(fontsize=7)

ax = axes[1, 0]
for seed, d in lstm_data.items():
    er = d["episode_reward"]
    if er is not None:
        ax.plot(er, color=COLORS[seed], alpha=0.7, linewidth=1.2, label=f"seed {seed}")
ax.set_title("LSTM - Episode Reward")
ax.set_xlabel("measurement step")
ax.set_ylabel("reward")
ax.legend(fontsize=7)

ax = axes[1, 1]
for seed, d in ff_data.items():
    er = d["episode_reward"]
    if er is not None:
        ax.plot(er, color=COLORS[seed], alpha=0.7, linewidth=1.2, label=f"seed {seed}")
ax.set_title("Feedforward - Episode Reward")
ax.set_xlabel("measurement step")
ax.set_ylabel("reward")
ax.legend(fontsize=7)

ax = axes[2, 0]
for seed, d in lstm_data.items():
    gn = d["grad_norm"]
    if gn is not None:
        ax.plot(gn, color=COLORS[seed], alpha=0.7, linewidth=1.2, label=f"seed {seed}")
ax.set_title("LSTM - Gradient Norm")
ax.set_xlabel("update step")
ax.set_ylabel("grad norm")
ax.legend(fontsize=7)

ax = axes[2, 1]
for seed, d in ff_data.items():
    gn = d["grad_norm"]
    if gn is not None:
        ax.plot(gn, color=COLORS[seed], alpha=0.7, linewidth=1.2, label=f"seed {seed}")
ax.set_title("Feedforward - Gradient Norm")
ax.set_xlabel("update step")
ax.set_ylabel("grad norm")
ax.legend(fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=4.0, w_pad=3.0)

out_path = os.path.join(BASE, "output", "inspect_runs.png")
plt.savefig(out_path, dpi=150)
print(f"saved to {out_path}")
plt.show()

print("\nLSTM spectral radius:")
for seed, d in lstm_data.items():
    sr = d["spectral_radius"]
    if sr is not None:
        print(f"  seed {seed}: n={len(sr)}  mean={sr.mean():.4f}  std={sr.std():.4f}  max={sr.max():.4f}")

print("\nLSTM episode reward:")
for seed, d in lstm_data.items():
    er = d["episode_reward"]
    if er is not None:
        print(f"  seed {seed}: n={len(er)}  mean={er.mean():.3f}  final={er[-1]:.3f}")

print("\nFeedforward episode reward:")
for seed, d in ff_data.items():
    er = d["episode_reward"]
    if er is not None:
        print(f"  seed {seed}: n={len(er)}  mean={er.mean():.3f}  final={er[-1]:.3f}")
