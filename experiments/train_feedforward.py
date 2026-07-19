"""
experiments/train_feedforward.py

Trains FeedforwardActorCritic (agents/feedforward_ppo/agent.py) on
GridWorldEnv (environment/environment.py, unmodified) using PPO with GAE
advantages -- the control condition for the recurrence comparison.

At the same fixed checkpoint interval train_lstm.py uses, logs the
secondary measurements that remain meaningful without recurrence (gradient
norm, episode reward) via FeedforwardDiagnosticsRecorder. There is no
spectral radius, hidden state drift, or condition number here: no h_t
depends on h_{t-1}, so there is no per-step Jacobian and no recurrent
weight matrix to track.

Structurally this mirrors experiments/train_lstm.py step for step (same
GAE via agents/common.py, same clipped-surrogate PPO objective, same
checkpoint cadence, same output layout) so the two runs are as directly
comparable as the shared config fields allow -- kept as a separate script,
not a shared train.py, but importing the same GAE/gradient-norm utilities
so that math can't quietly diverge between the two.

Usage:
    python -m experiments.train_feedforward --seed 0 --num_updates 500
    python -m experiments.train_feedforward --seed 0 --full_obs
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from environment.environment import GridWorldEnv
from agents.common import compute_gae, collect_episode, make_run_dir, save_arrays
from agents.Feedforward.agent import FeedforwardActorCritic, FeedforwardDiagnosticsRecorder
from agents.Feedforward.config import FeedforwardConfig


# --------------------------------------------------------------------------
# PPO update over a batch of episodes
# --------------------------------------------------------------------------

def ppo_update(agent, optimizer, episodes, recorder, update_idx, cfg: FeedforwardConfig, device="cpu"):
    """
    Same PPO objective and cadence as experiments/train_lstm.py's
    ppo_update, minus anything Jacobian-related. Each episode still goes
    through evaluate_actions once per epoch for consistency with the LSTM
    script, even though the feedforward agent doesn't need per-episode
    replay for correctness (its evaluate_actions has no sequential
    dependency) -- keeping the loop shape identical makes the two scripts
    easier to diff against each other.

    Returns the checkpoint row (dict) if this update lands on
    cfg.checkpoint_interval, else None.
    """
    for ep in episodes:
        adv, ret = compute_gae(ep["rewards"], ep["values"], ep["dones"], gamma=cfg.gamma, lam=cfg.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ep["advantages"] = torch.as_tensor(adv, dtype=torch.float32, device=device)
        ep["returns"] = torch.as_tensor(ret, dtype=torch.float32, device=device)
        ep["old_log_probs"] = torch.as_tensor(ep["log_probs"], dtype=torch.float32, device=device)

    last_grad_norm = 0.0
    for _epoch in range(cfg.epochs):
        optimizer.zero_grad()
        total_loss = 0.0

        for ep in episodes:
            obs_seq = torch.as_tensor(np.stack(ep["obs"]), dtype=torch.float32, device=device).unsqueeze(1)
            actions_seq = torch.as_tensor(ep["actions"], dtype=torch.long, device=device).unsqueeze(1)
            h0, c0 = agent.init_hidden(batch_size=1, device=device)  # (None, None), ignored

            new_log_probs, new_values, entropies = agent.evaluate_actions(obs_seq, actions_seq, h0, c0)
            new_log_probs = new_log_probs.squeeze(-1)
            new_values = new_values.squeeze(-1)
            entropies = entropies.squeeze(-1)

            ratio = torch.exp(new_log_probs - ep["old_log_probs"])
            surr1 = ratio * ep["advantages"]
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * ep["advantages"]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(new_values, ep["returns"])
            entropy_bonus = entropies.mean()

            total_loss = total_loss + policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_bonus

        total_loss = total_loss / len(episodes)
        total_loss.backward()
        last_grad_norm = recorder.compute_gradient_norm(agent)  # measured pre-clip, pre-step
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=cfg.max_grad_norm)
        optimizer.step()

    if update_idx % cfg.checkpoint_interval != 0:
        return None

    mean_reward = float(np.mean([np.sum(ep["rewards"]) for ep in episodes]))
    row = recorder.record_checkpoint(
        update_idx=update_idx,
        grad_norm=last_grad_norm,
        episode_reward=mean_reward,
    )
    return row


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------

def train(seed=0, num_updates=500, size=10, obs_radius=2, full_obs=False,
          output_dir="output", device="cpu", cfg: FeedforwardConfig = None):
    cfg = cfg or FeedforwardConfig()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = GridWorldEnv(size=size, obs_radius=obs_radius, full_obs=full_obs, seed=seed)
    agent = FeedforwardActorCritic(obs_dim=env.obs_dim, action_dim=env.action_space_size,
                                    hidden_dim=cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr)
    recorder = FeedforwardDiagnosticsRecorder(agent)

    run_dir = make_run_dir(output_dir, "feedforward", seed, full_obs)
    print(f"Writing to {run_dir}")

    reward_history = []
    t_start = time.time()

    for update_idx in range(1, num_updates + 1):
        episodes = [collect_episode(env, agent, device=device) for _ in range(cfg.episodes_per_update)]

        row = ppo_update(agent, optimizer, episodes, recorder, update_idx, cfg, device=device)

        mean_reward = float(np.mean([np.sum(ep["rewards"]) for ep in episodes]))
        reward_history.append(mean_reward)

        if row is not None:
            elapsed = time.time() - t_start
            print(f"[feedforward update {update_idx:5d} | {elapsed:6.1f}s] "
                  f"reward={row['episode_reward']:+.3f}  "
                  f"grad_norm={row['grad_norm']:.4f}")
            save_arrays(run_dir, recorder, reward_history)

    save_arrays(run_dir, recorder, reward_history)  # final save, covers any tail after the last checkpoint
    print(f"Finished. Saved run to {run_dir}")
    return agent, recorder, reward_history, run_dir


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train the feedforward-PPO agent on GridWorldEnv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_updates", type=int, default=500)
    parser.add_argument("--episodes_per_update", type=int, default=4)
    parser.add_argument("--size", type=int, default=10, choices=[10, 15])
    parser.add_argument("--obs_radius", type=int, default=2)
    parser.add_argument("--full_obs", action="store_true",
                         help="Sanity-check baseline: full observability instead of partial")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                         help="Log secondary measurements every N PPO updates")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = FeedforwardConfig(
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        episodes_per_update=args.episodes_per_update,
        checkpoint_interval=args.checkpoint_interval,
    )
    train(
        seed=args.seed,
        num_updates=args.num_updates,
        size=args.size,
        obs_radius=args.obs_radius,
        full_obs=args.full_obs,
        output_dir=args.output_dir,
        device=args.device,
        cfg=cfg,
    )