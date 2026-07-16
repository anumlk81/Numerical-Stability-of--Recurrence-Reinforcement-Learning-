"""
agents/lstm_ppo/agent.py

LSTM-based recurrent actor-critic agent, trained with PPO, with an explicit
step-by-step forward pass that exposes h_t so we can differentiate it
w.r.t. h_{t-1}. That per-step Jacobian is the object the gradient-dynamics
hypothesis is about: its product over a temporal window, and the spectral
radius of that product, is what we compare against the feedforward agent
(agents/feedforward_ppo/agent.py), whose recurrent Jacobian is trivially
zero / undefined.

Why a manual loop instead of nn.LSTM
-------------------------------------
nn.LSTM fuses the whole sequence into a single cuDNN/ATen kernel and never
materializes h_t as a distinct autograd node -- backward only exposes the
gradient of the final hidden state, not every intermediate one. So we use
nn.LSTMCell in an explicit Python loop: at every t, h_t is a normal tensor
produced by one cell call, and we can freely call
`torch.autograd.grad(h_t, h_prev, ...)` on it.



Three pieces, one file
------------------------
  1. LSTMActorCritic     -- rollout / PPO update, no Jacobian bookkeeping.
  2. HiddenJacobianTracker -- re-runs single steps with h_prev detached and
     re-attached as a leaf, extracting the full (hidden_dim x hidden_dim)
     Jacobian d h_t / d h_{t-1}. Run at a checkpoint interval, not every
     step, since it costs hidden_dim backward passes per recorded step.
  3. GradientDynamicsRecorder -- bundles the Jacobian tracker's primary
     measurement with the secondary measurements (grad norm, reward,
     hidden state drift, condition number of W_h) at the same cadence.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from agents.common import compute_gradient_norm


# --------------------------------------------------------------------------
# Actor-critic network
# --------------------------------------------------------------------------

class LSTMActorCritic(nn.Module):
    """
    Single-layer LSTM recurrent actor-critic.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.lstm_cell = nn.LSTMCell(obs_dim, hidden_dim) #process exactly one timestep at a time
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        # Orthogonal init for the recurrent weights keeps the initial
        # per-step Jacobian well-conditioned, so any spectral-radius growth
        # we later measure is attributable to training, not to a bad init.
        for name, param in self.lstm_cell.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)#intital policy action distribution is nearly uniform
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def init_hidden(self, batch_size: int = 1, device="cpu"):
        h0 = torch.zeros(batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h0, c0

    def step(self, obs: torch.Tensor, hidden):
        """
        One environment step.

        obs:    (batch, obs_dim)
        hidden: (h_prev, c_prev), each (batch, hidden_dim)

        Returns h_t, c_t, action_logits, value.
        """
        h_prev, c_prev = hidden
        h_t, c_t = self.lstm_cell(obs, (h_prev, c_prev))
        action_logits = self.actor_head(h_t)
        value = self.critic_head(h_t).squeeze(-1)
        return h_t, c_t, action_logits, value

    @torch.no_grad() #disables gradient tracking during environment execution loops.
    def get_action(self, obs: torch.Tensor, hidden, deterministic: bool = False):
        """
        Sample an action during rollout collection. Hidden state is carried
        by the caller and reset to zero at episode boundaries.
        """
        h_t, c_t, action_logits, value = self.step(obs, hidden)
        dist = Categorical(logits=action_logits)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, (h_t, c_t)

    def evaluate_actions(self, obs_seq: torch.Tensor, actions_seq: torch.Tensor,
                          h0: torch.Tensor, c0: torch.Tensor):
        """
        Re-run a full episode segment sequentially for the PPO update phase,
        returning log_probs, values, entropy at every step for the clipped
        surrogate loss. This is the pass BPTT gradients flow back through --
        unlike the feedforward agent, this cannot be collapsed into a single
        batched forward pass, because h_t depends on h_{t-1}.

        obs_seq:     (T, batch, obs_dim)
        actions_seq: (T, batch)
        """
        h, c = h0, c0
        log_probs, values, entropies = [], [], []
        for t in range(obs_seq.shape[0]):
            h, c, action_logits, value = self.step(obs_seq[t], (h, c))
            dist = Categorical(logits=action_logits)
            log_probs.append(dist.log_prob(actions_seq[t]))
            values.append(value)
            entropies.append(dist.entropy())
        return torch.stack(log_probs), torch.stack(values), torch.stack(entropies)


# --------------------------------------------------------------------------
# Per-step Jacobian + spectral radius tracking
# --------------------------------------------------------------------------

class HiddenJacobianTracker:
    """
    Computes the per-step hidden-to-hidden Jacobian

        J_t = d h_t / d h_{t-1}      (hidden_dim x hidden_dim)

    along a real trajectory, and the spectral radius (max |eigenvalue|) of
    the product of J_t over a window of `window` CONSECUTIVE timesteps.
    This is the empirical quantity the bifurcation hypothesis (Pascanu et
    al.) is about: a spectral radius that drifts above 1 and stays there,
    or that spikes / develops a heavy tail across training, is the
    signature we're looking for.

    One call to `probe_window` = one reading of the primary measurement,
    taken at a single point in training from a single window of real,
    consecutive steps in that update's trajectory. Calling this at a fixed
    checkpoint interval across training (e.g. every K PPO updates) builds
    the time series whose variance we compare between agent types --
    `radii_array()` returns that series.

    Cost: `window` sequential steps, each costing hidden_dim backward
    passes (one per output unit). This is why it's called at a checkpoint
    interval and not every env step.
    """

    def __init__(self, agent: LSTMActorCritic, window: int = 20):
        self.agent = agent
        self.hidden_dim = agent.hidden_dim
        self.window = window
        self._radii: list[float] = []

    def step_jacobian(self, obs_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        """
        Computes J_t = d h_t / d h_prev by looping over each output unit of
        h_t and pulling its gradient w.r.t. h_prev via torch.autograd.grad.

        obs_t: (1, obs_dim)
        h_prev, c_prev: (1, hidden_dim), detached from any prior graph --
            the caller should pass h_prev.detach().requires_grad_(True) so
            this Jacobian probe doesn't interfere with the PPO graph.

        Returns J_t (hidden_dim, hidden_dim), plus detached h_t, c_t so the
        caller can keep stepping the trajectory forward.
        """
        assert h_prev.requires_grad, (
            "h_prev must require grad -- pass h_prev.detach().requires_grad_(True)"
        )
        h_t, c_t = self.agent.lstm_cell(obs_t, (h_prev, c_prev))

        rows = []
        for unit in range(self.hidden_dim):
            grad_outputs = torch.zeros_like(h_t)
            grad_outputs[0, unit] = 1.0 #computes vector jacobian products
            (grad_h_prev,) = torch.autograd.grad(
                outputs=h_t,
                inputs=h_prev,
                grad_outputs=grad_outputs,
                retain_graph=True,   # h_t's graph is reused for every unit
                create_graph=False,  # we only need the value, not higher-order grads
            )
            rows.append(grad_h_prev.squeeze(0).detach())

        J_t = torch.stack(rows, dim=0)  # row i = d h_t[i] / d h_prev
        return J_t, h_t.detach(), c_t.detach()

    def probe_window(self, obs_window: list, h_start: torch.Tensor, c_start: torch.Tensor) -> float:
        """
        Walks `len(obs_window)` (normally == self.window) CONSECUTIVE real
        observations from a trajectory, computing each step's Jacobian and
        multiplying them together in temporal order, then returns the
        spectral radius of the resulting product. Also appends the reading
        to this tracker's history for `radii_array()`.

        obs_window: list of (1, obs_dim) tensors, consecutive timesteps.
        h_start, c_start: (1, hidden_dim) hidden/cell state immediately
            preceding obs_window[0] (i.e. h_{t0 - 1}), detached.

        Product convention: most recent step applied last, i.e.
        product = J_{t0+W-1} @ ... @ J_{t0+1} @ J_{t0}.
        """
        h_prev = h_start.detach().requires_grad_(True)
        c_prev = c_start.detach()

        product = None
        for obs_t in obs_window:
            J_t, h_t, c_t = self.step_jacobian(obs_t, h_prev, c_prev)
            product = J_t if product is None else J_t @ product
            h_prev = h_t.detach().requires_grad_(True)
            c_prev = c_t

        radius = self._spectral_radius(product)
        self._radii.append(radius)
        return radius

    @staticmethod
    def _spectral_radius(matrix: torch.Tensor) -> float:
        eigenvalues = torch.linalg.eigvals(matrix)
        return eigenvalues.abs().max().item()

    def radii_array(self) -> np.ndarray:
        """History of recorded spectral radii across checkpoints, for
        saving to .npy and comparing variance / tail behavior against the
        feedforward agent."""
        return np.array(self._radii, dtype=np.float32)


# --------------------------------------------------------------------------
# Secondary measurements
# --------------------------------------------------------------------------

class GradientDynamicsRecorder:
    """
    Bundles the primary measurement (spectral radius of the temporal
    Jacobian product) with four secondary measurements, all logged at the
    same checkpoint interval so every row of the resulting table is
    directly comparable:

        update | spectral_radius | grad_norm | episode_reward
              | hidden_state_drift | condition_number (per gate + full)

    Rationale for each secondary measurement:
      - grad_norm: if the theory holds, spikes here should co-occur with
        spectral radius spikes -- the Jacobian product is exactly what
        gradients get multiplied through during BPTT.
      - episode_reward: the behavioral read-out. If spectral radius
        variance spikes line up with reward drops, that's the empirical
        link from numerical instability to behavioral collapse (Jin &
        Lavaei).
      - hidden_state_drift: ||h_t - h_{t-1}||, averaged over an episode --
        a behavioral (not just mathematical) correlate of the same
        instability the spectral radius captures in the Jacobian.
      - condition_number of W_h: tracked at the same checkpoints as a
        leading indicator -- a rising condition number should, per the
        hypothesis, precede spectral radius spikes rather than merely
        coincide with them.

    grad_norm uses agents.common.compute_gradient_norm, the same function
    the feedforward agent's training loop uses, so the two are computed
    identically.
    """

    def __init__(self, agent: LSTMActorCritic, jacobian_window: int = 20):
        self.agent = agent
        self.jacobian_tracker = HiddenJacobianTracker(agent, window=jacobian_window)
        self.history = {
            "update": [],
            "spectral_radius": [],
            "grad_norm": [],
            "episode_reward": [],
            "hidden_state_drift": [],
            "condition_number_full": [],
            "condition_number_input": [],
            "condition_number_forget": [],
            "condition_number_cell": [],
            "condition_number_output": [],
        }

    # -- individual measurements, usable standalone --------------------

    compute_gradient_norm = staticmethod(compute_gradient_norm)

    @staticmethod
    def compute_hidden_state_drift(hidden_states: torch.Tensor) -> float:
        """
        hidden_states: (T, hidden_dim) sequence of h_t collected over one
        episode. Returns mean_t ||h_t - h_{t-1}||_2, i.e. the average
        per-step displacement of the agent's memory.
        """
        assert hidden_states.dim() == 2 and hidden_states.shape[0] > 1, (
            "need a (T, hidden_dim) sequence with T > 1 to compute drift"
        )
        step_deltas = hidden_states[1:] - hidden_states[:-1]
        step_norms = step_deltas.norm(dim=-1)
        return step_norms.mean().item()

    def compute_condition_number(self) -> dict:
        """
        Condition number (ratio of largest to smallest singular value) of
        the recurrent weight matrix W_h inside the LSTM cell.
        nn.LSTMCell stores this as weight_hh with shape
        (4 * hidden_dim, hidden_dim) -- the input, forget, cell, and output
        gates stacked. We report the condition number of each gate's
        (hidden_dim, hidden_dim) block individually as well as of the full
        stacked matrix, since any single gate's block can be the one
        driving eigenvalue sensitivity even if the others are well
        conditioned.
        """
        W_hh = self.agent.lstm_cell.weight_hh.detach()  # (4H, H)
        H = self.agent.hidden_dim
        gate_names = ["input", "forget", "cell", "output"]

        conditions = {}
        for i, name in enumerate(gate_names):
            block = W_hh[i * H:(i + 1) * H, :]
            svals = torch.linalg.svdvals(block)
            conditions[name] = (svals.max() / svals.min().clamp_min(1e-12)).item()

        full_svals = torch.linalg.svdvals(W_hh)
        conditions["full"] = (full_svals.max() / full_svals.min().clamp_min(1e-12)).item()
        return conditions

    # -- combined checkpoint logging ------------------------------------

    def record_checkpoint(self, update_idx: int, obs_window: list,
                           h_start: torch.Tensor, c_start: torch.Tensor,
                           grad_norm: float, episode_reward: float,
                           hidden_states: torch.Tensor) -> dict:
        """
        Records one row of all five measurements at once. Call this at a
        fixed interval (e.g. every K PPO updates), not every step -- both
        the Jacobian probe and the SVD-based condition number are too
        costly to run every env step.

        obs_window: list of (1, obs_dim) tensors, `jacobian_window`
            CONSECUTIVE real observations from one trajectory, used to
            build the temporal Jacobian product for this checkpoint.
        h_start, c_start: hidden/cell state immediately preceding
            obs_window[0].
        grad_norm: from compute_gradient_norm(agent), called right after
            this update's loss.backward().
        episode_reward: total (or mean) reward for the episode this
            checkpoint falls in.
        hidden_states: (T, hidden_dim) h_t sequence from that same episode,
            used for the drift measurement.
        """
        spectral_radius = self.jacobian_tracker.probe_window(obs_window, h_start, c_start)

        drift = self.compute_hidden_state_drift(hidden_states)
        conditions = self.compute_condition_number()

        row = {
            "update": update_idx,
            "spectral_radius": spectral_radius,
            "grad_norm": grad_norm,
            "episode_reward": episode_reward,
            "hidden_state_drift": drift,
            "condition_number_full": conditions["full"],
            "condition_number_input": conditions["input"],
            "condition_number_forget": conditions["forget"],
            "condition_number_cell": conditions["cell"],
            "condition_number_output": conditions["output"],
        }
        for key, value in row.items():
            self.history[key].append(value)
        return row

    def to_arrays(self) -> dict:
        """History as a dict of float32 arrays, one per measurement --
        ready to np.save individually or stack into a structured array
        under the run's per-agent/per-seed output directory."""
        return {key: np.array(values, dtype=np.float32) for key, values in self.history.items()}