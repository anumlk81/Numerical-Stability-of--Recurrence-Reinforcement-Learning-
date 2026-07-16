"""
agents/lstm_ppo/config.py

LSTM-specific hyperparameters. Every field shared with
agents/feedforward_ppo/config.py (hidden_dim, lr, gamma, lam, clip_eps,
value_coef, entropy_coef, max_grad_norm, epochs, episodes_per_update,
checkpoint_interval) must be kept equal to the feedforward config's value
in any run used for the agent comparison -- recurrence is supposed to be
the only architectural difference. jacobian_window has no feedforward
analogue: it controls how many consecutive steps the temporal Jacobian
product is taken over, which only exists for a recurrent agent.
"""

from dataclasses import dataclass


@dataclass
class LSTMConfig:
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

    # Recurrence-only: number of consecutive real timesteps the temporal
    # Jacobian product is taken over at each checkpoint
    jacobian_window: int = 20