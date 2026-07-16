"""
agents/common.py

Utilities shared by every agent type. Kept here (not duplicated inside
each agent.py) so that anything meant to be directly comparable across
architectures -- gradient norm, episode rollout collection -- is computed
by literally the same code for both, rather than two implementations that
could quietly drift apart.
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def compute_gradient_norm(model: nn.Module) -> float:
    """
    Global L2 norm of gradients across all parameters. Call this after
    loss.backward() and before optimizer.step(), so it reflects the
    gradient that actually produced the update (not a post-clipping value).
    """
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += p.grad.detach().pow(2).sum().item()
    return total_sq ** 0.5


def compute_gae(rewards, values, dones, gamma: float = 0.99, lam: float = 0.95):
    """
    Generalized Advantage Estimation for one episode.

    rewards, values, dones: length-T sequences from one episode.
    Bootstrap value after the final step is always 0: every GridWorldEnv
    episode ends either on the goal (true terminal, future value is 0) or
    on a timeout (dones[-1] is still True), and next_non_terminal below
    zeroes the bootstrap term in both cases -- so a flat 0 bootstrap is
    correct either way.

    Shared here rather than duplicated in each training script, so both
    agents' PPO updates use identically-computed advantages -- a difference
    in GAE math between the two would confound the recurrence comparison
    just as much as a difference in the PPO loss itself.

    Returns (advantages, returns) as float32 np arrays of length T.
    """
    T = len(rewards)
    values = np.asarray(values, dtype=np.float32)
    values_ext = np.append(values, 0.0)
    advantages = np.zeros(T, dtype=np.float32)

    last_gae = 0.0
    for t in reversed(range(T)):
        next_non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values_ext[t + 1] * next_non_terminal - values_ext[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def collect_episode(env, agent, device: str = "cpu") -> dict:
    """
    Runs one episode with any agent that implements init_hidden / get_action
    with the (h, c) hidden-tuple convention used here. Works unchanged for
    both LSTMActorCritic (real hidden state) and FeedforwardActorCritic
    (hidden is always (None, None) and simply passed through) -- this is
    what lets a single train.py drive either agent without special-casing
    rollout collection.

    Note: assumes batch=1 / single environment, which both agents' Jacobian
    and step interfaces are built around.
    """
    obs = env.reset()
    h, c = agent.init_hidden(batch_size=1, device=device)

    rollout = {"obs": [], "actions": [], "log_probs": [], "values": [], "rewards": [], "dones": []}
    hidden_states = []
    if h is not None:
        hidden_states.append(h.squeeze(0).detach().clone())

    done = False
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action, log_prob, value, (h_next, c_next) = agent.get_action(obs_t, (h, c))
        next_obs, reward, done, _info = env.step(action.item())

        rollout["obs"].append(obs)
        rollout["actions"].append(action.item())
        rollout["log_probs"].append(log_prob.item())
        rollout["values"].append(value.item())
        rollout["rewards"].append(reward)
        rollout["dones"].append(done)

        obs = next_obs
        h, c = h_next, c_next
        if h is not None:
            hidden_states.append(h.squeeze(0).detach().clone())

    if hidden_states:
        rollout["hidden_states"] = torch.stack(hidden_states, dim=0)  # (T+1, hidden_dim)
    return rollout


def make_run_dir(output_dir, agent_name: str, seed: int, full_obs: bool) -> Path:
    """
    Per-agent, per-seed output layout with a run timestamp, so repeated
    runs never silently overwrite each other:

        output/{agent_name}/{partial_obs,full_obs}/seed_{seed}/{timestamp}/

    Called once at the START of training (not the end), so the directory
    -- and the files written into it via save_arrays -- exist for the
    whole run, not only after it finishes.
    """
    obs_mode = "full_obs" if full_obs else "partial_obs"
    run_dir = Path(output_dir) / agent_name / obs_mode / f"seed_{seed}" / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_arrays(run_dir: Path, recorder, reward_history) -> None:
    """
    Writes (overwrites) every .npy file in run_dir with whatever the
    recorder and reward_history hold so far. Meant to be called at every
    checkpoint during training, not just once at the end -- these arrays
    are small, so re-saving them repeatedly is cheap, and it means a run
    that gets killed partway through still leaves real data on disk
    instead of losing the whole run.

    Works with either GradientDynamicsRecorder (LSTM) or
    FeedforwardDiagnosticsRecorder -- both expose the same to_arrays()
    interface, just with a different set of measurement keys.
    """
    for name, array in recorder.to_arrays().items():
        np.save(run_dir / f"{name}.npy", array)
    np.save(run_dir / "reward_per_update.npy", np.array(reward_history, dtype=np.float32))