"""
agents/feedforward_ppo/agent.py

Two pieces, one file (mirrors agents/lstm/agent.py's structure):
  1. FeedforwardActorCritic     -- rollout / PPO update.
  2. FeedforwardDiagnosticsRecorder -- logs the secondary measurements that
     remain meaningful without recurrence (grad_norm, episode_reward) at
     the same checkpoint cadence agents/lstm_ppo/agent.py's
     GradientDynamicsRecorder uses. spectral_radius, hidden_state_drift,
     and condition_number of a *recurrent* weight matrix are all
     undefined here -- there is no W_h and no h_t to differentiate --
     so this recorder only carries the two measurements that transfer.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical #handles the numerical stabilization and normalization so that
#they form a valid probability distribution across the action space

from agents.common import compute_gradient_norm


class FeedforwardActorCritic(nn.Module):
    """
    Two-hidden-layer MLP actor-critic. No state carries across timesteps --
    each action is a pure function of the current observation.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        # Same orthogonal-init philosophy as the LSTM agent: start from a
        # well-conditioned network so anything we measure later is
        # attributable to training, not to init.
        gain = nn.init.calculate_gain("tanh")
        for layer in self.body:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def init_hidden(self, batch_size: int = 1, device="cpu"):
        """No recurrent state. Returns (None, None) purely so callers that
        thread a hidden tuple uniformly across agent types don't need a
        special case for this agent."""
        return None, None

    def step(self, obs: torch.Tensor, hidden=None):
        """
        One forward pass. `hidden` is accepted for interface parity with
        LSTMActorCritic.step but is ignored -- the output depends only on
        obs. Returned hidden is always (None, None).
        """
        features = self.body(obs)
        action_logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return None, None, action_logits, value

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor, hidden=None, deterministic: bool = False):
        _, _, action_logits, value = self.step(obs, hidden)
        dist = Categorical(logits=action_logits)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, (None, None)

    def evaluate_actions(self, obs_seq: torch.Tensor, actions_seq: torch.Tensor,
                          h0=None, c0=None):
        """
        obs_seq: (T, batch, obs_dim), actions_seq: (T, batch)
        h0, c0: accepted for interface parity with the LSTM agent's PPO
            update loop, ignored here.

        All T steps are collapsed into a single batched forward pass. This is the concrete
        computational cost recurrence imposes: the feedforward agent's
        evaluate_actions is O(1) kernel launches in T; the LSTM agent's is
        O(T) Python-loop iterations.
        """
        T, B, _ = obs_seq.shape
        flat_obs = obs_seq.reshape(T * B, self.obs_dim)

        features = self.body(flat_obs)
        action_logits = self.actor_head(features).reshape(T, B, self.action_dim)
        values = self.critic_head(features).reshape(T, B)

        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions_seq)
        entropies = dist.entropy()
        return log_probs, values, entropies


# --------------------------------------------------------------------------
# Secondary measurements
# --------------------------------------------------------------------------

class FeedforwardDiagnosticsRecorder:
    """
    Logs the secondary measurements at the same checkpoint interval
    agents/lstm_ppo/agent.py's GradientDynamicsRecorder uses:

        update | grad_norm | episode_reward
    """

    def __init__(self, agent: FeedforwardActorCritic):
        self.agent = agent
        self.history = {"update": [], "grad_norm": [], "episode_reward": []}

    compute_gradient_norm = staticmethod(compute_gradient_norm)

    def record_checkpoint(self, update_idx: int, grad_norm: float, episode_reward: float) -> dict:
        """
        Records one row. Call this at the same fixed interval (e.g. every
        K PPO updates) as the LSTM agent's checkpoint calls, so the two
        agents' diagnostic time series line up update-for-update.
        """
        row = {"update": update_idx, "grad_norm": grad_norm, "episode_reward": episode_reward}
        for key, value in row.items():
            self.history[key].append(value)
        return row

    def to_arrays(self) -> dict:
        """History as a dict of float32 arrays, one per measurement --
        ready to np.save under the run's per-agent/per-seed output
        directory, same convention as the LSTM agent's recorder."""
        return {key: np.array(values, dtype=np.float32) for key, values in self.history.items()}