"""
agents/feedforward_ppo/config.py

Feedforward-specific hyperparameters. Deliberately mirrors
agents/lstm_ppo/config.py field-for-field wherever a field is shared
(hidden_dim, lr, gamma, lam, clip_eps, value_coef, entropy_coef,
max_grad_norm, epochs, episodes_per_update, checkpoint_interval) --
those values should be kept equal between the two configs for any run
used in the agent comparison, so that recurrence remains the only
architectural difference. Fields absent here (e.g. jacobian_window) are
recurrence-only and have no feedforward analogue.
"""

from dataclasses import dataclass


@dataclass
class FeedforwardConfig:
    # Architecture
    hidden_dim: int = 64

    # Optimizer
    lr: float = 3e-4

    # PPO / GAE
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    epochs: int = 4

    # Rollout / logging cadence
    episodes_per_update: int = 4
    checkpoint_interval: int = 10