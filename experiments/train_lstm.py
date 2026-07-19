"""
experiments/train_lstm.py

Trains LSTMActorCritic (agents/lstm_ppo/agent.py) on GridWorldEnv
(environment/environment.py, unmodified) using PPO with GAE advantages.

At a fixed checkpoint interval, logs the primary measurement (spectral
radius of the temporal Jacobian product, taken over a window of consecutive
real steps from a probe episode) alongside the four secondary measurements
(gradient norm, episode reward, hidden state drift, condition number of
W_h), via GradientDynamicsRecorder.

This script is deliberately separate from experiments/train_feedforward.py
(see that file for the feedforward agent's training loop) -- but both
import GAE and gradient-norm computation from agents/common.py, so the
core PPO math can't silently diverge between the two even though the
scripts themselves are independent.

Usage:
    python -m experiments.train_lstm --seed 0 --num_updates 500
    python -m experiments.train_lstm --seed 0 --full_obs   # sanity-check baseline
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from environment.environment import GridWorldEnv
from agents.common import compute_gae, collect_episode, make_run_dir, save_arrays
from agents.LSTM.agent import LSTMActorCritic, GradientDynamicsRecorder
from agents.LSTM.config import LSTMConfig


# --------------------------------------------------------------------------
# PPO update over a batch of episodes
# --------------------------------------------------------------------------

def ppo_update(agent, optimizer, episodes, recorder, update_idx, cfg: LSTMConfig, device="cpu"):
    """
    episodes: list of rollout dicts from collect_episode (obs, actions,
        log_probs, values, rewards, dones, hidden_states already populated).

    Runs cfg.epochs passes of the clipped PPO objective over the same batch
    of episodes. Each episode is replayed start-to-finish through
    evaluate_actions every epoch (BPTT needs the whole sequence in one
    graph, so episodes aren't split into minibatches the way i.i.d.
    transitions would be).

    Returns the checkpoint row (dict) if this update lands on
    cfg.checkpoint_interval, else None.
    """
    # Advantages/returns use the *old* value function from rollout time,
    # computed once, not recomputed as the policy updates across epochs.
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
            h0, c0 = agent.init_hidden(batch_size=1, device=device)

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

    # Probe episode for the Jacobian window + hidden state drift: reuse the
    # first episode in this update's batch, first jacobian_window steps.
    probe_ep = episodes[0]
    window_len = min(cfg.jacobian_window, len(probe_ep["obs"]))
    obs_window = [
        torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
        for o in probe_ep["obs"][:window_len]
    ]
    h_start, c_start = agent.init_hidden(batch_size=1, device=device)
    mean_reward = float(np.mean([np.sum(ep["rewards"]) for ep in episodes]))

    row = recorder.record_checkpoint(
        update_idx=update_idx,
        obs_window=obs_window,
        h_start=h_start,
        c_start=c_start,
        grad_norm=last_grad_norm,
        episode_reward=mean_reward,
        hidden_states=probe_ep["hidden_states"],
    )
    return row


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------

def train(seed=0, num_updates=500, size=10, obs_radius=2, full_obs=False,
          output_dir="output", device="cpu", cfg: LSTMConfig = None):
    cfg = cfg or LSTMConfig()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = GridWorldEnv(size=size, obs_radius=obs_radius, full_obs=full_obs, seed=seed)
    agent = LSTMActorCritic(obs_dim=env.obs_dim, action_dim=env.action_space_size,
                             hidden_dim=cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr)
    recorder = GradientDynamicsRecorder(agent, jacobian_window=cfg.jacobian_window)

    run_dir = make_run_dir(output_dir, "lstm", seed, full_obs)
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
            print(f"[lstm update {update_idx:5d} | {elapsed:6.1f}s] "
                  f"reward={row['episode_reward']:+.3f}  "
                  f"spectral_radius={row['spectral_radius']:.3e}  "
                  f"grad_norm={row['grad_norm']:.4f}  "
                  f"hidden_drift={row['hidden_state_drift']:.4f}  "
                  f"cond_full={row['condition_number_full']:.2f}")
            save_arrays(run_dir, recorder, reward_history)

    save_arrays(run_dir, recorder, reward_history)  # final save, covers any tail after the last checkpoint
    print(f"Finished. Saved run to {run_dir}")
    return agent, recorder, reward_history, run_dir


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train the LSTM-PPO agent on GridWorldEnv")
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
                         help="Log spectral radius + secondary measurements every N PPO updates")
    parser.add_argument("--jacobian_window", type=int, default=20,
                         help="Number of consecutive steps the temporal Jacobian product is taken over")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = LSTMConfig(
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        episodes_per_update=args.episodes_per_update,
        checkpoint_interval=args.checkpoint_interval,
        jacobian_window=args.jacobian_window,
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