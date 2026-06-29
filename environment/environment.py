"""
Partially observable gridworld environment for gradient dynamics research.

Supports both feedforward and LSTM-PPO agents through a unified interface.
Observation is a flattened (2*radius+1)^2 local window around the agent.
Designed to surface long-horizon gradient signal: sparse rewards, long corridors,
and forced sequential dependencies that stress recurrent credit assignment.

Grid layout (randomly regenerated per reset):
  - Walls: border + random interior obstacles
  - Agent: single start position (random, non-wall)
  - Goal:  single target position (random, far from agent)
  - Empty: traversable cells

Action space: 0=up, 1=right, 2=down, 3=left
"""

import numpy as np


# ---------------------------------------------------------------------------
# Cell type constants
# ---------------------------------------------------------------------------
EMPTY  = 0
WALL   = 1
AGENT  = 2
GOAL   = 3


class GridWorldEnv:
    """
    Partially observable gridworld with configurable size.

    Parameters
    ----------
    size : int
        Side length of the square grid. Must be 10 or 15.
    obs_radius : int
        Chebyshev radius of the local observation window (default 2).
    max_steps : int
        Episode time limit. Defaults to 4 * size^2.
    wall_density : float
        Fraction of interior cells converted to walls (default 0.15).
    seed : int | None
        RNG seed for reproducibility.
    """

    VALID_SIZES = (10, 15)
    N_ACTIONS   = 4
    _DELTAS     = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up right down left

    def __init__(
        self,
        size: int = 10,
        obs_radius: int = 2,
        max_steps: int | None = None,
        wall_density: float = 0.15,
        seed: int | None = None,
    ) -> None:
        if size not in self.VALID_SIZES:
            raise ValueError(f"size must be one of {self.VALID_SIZES}, got {size}")
        if not 0 < obs_radius < size:
            raise ValueError(f"obs_radius must be in (0, {size}), got {obs_radius}")
        if not 0.0 <= wall_density < 0.5:
            raise ValueError(f"wall_density must be in [0, 0.5), got {wall_density}")

        self.size         = size
        self.obs_radius   = obs_radius
        self.max_steps    = max_steps if max_steps is not None else 4 * size * size
        self.wall_density = wall_density

        self.rng          = np.random.default_rng(seed)
        self.obs_dim      = (2 * obs_radius + 1) ** 2  # flat observation length

        # mutable state (initialised properly on reset)
        self._grid        = np.zeros((size, size), dtype=np.int32)
        self._agent_pos   = (0, 0)
        self._goal_pos    = (0, 0)
        self._step_count  = 0
        self._done        = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> np.ndarray:
        """
        Reset the environment and return the initial observation.

        Parameters
        ----------
        seed : int | None
            If provided, re-seeds the RNG before resetting.

        Returns
        -------
        obs : np.ndarray, shape (obs_dim,), dtype float32
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._build_grid()
        self._step_count = 0
        self._done       = False
        return self._observe()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Apply action and return (obs, reward, done, info).

        Parameters
        ----------
        action : int
            Integer in [0, N_ACTIONS).

        Returns
        -------
        obs    : np.ndarray, shape (obs_dim,), dtype float32
        reward : float
        done   : bool
        info   : dict  -- 'step', 'agent_pos', 'goal_pos', 'success'
        """
        if self._done:
            raise RuntimeError("Cannot call step() on a finished episode. Call reset() first.")
        if action not in range(self.N_ACTIONS):
            raise ValueError(f"action must be in [0, {self.N_ACTIONS}), got {action}")

        self._move_agent(action)
        self._step_count += 1

        reward, success = self._compute_reward()
        timeout = self._step_count >= self.max_steps
        self._done = success or timeout

        info = {
            "step":      self._step_count,
            "agent_pos": self._agent_pos,
            "goal_pos":  self._goal_pos,
            "success":   success,
            "timeout":   timeout,
        }
        return self._observe(), reward, self._done, info

    def render(self) -> str:
        """
        Return a plain-text ASCII rendering of the full grid.

        Returns
        -------
        str
        """
        symbols = {EMPTY: ".", WALL: "#", AGENT: "A", GOAL: "G"}
        rows = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                cell = self._grid[r, c]
                if (r, c) == self._agent_pos:
                    row.append("A")
                elif (r, c) == self._goal_pos:
                    row.append("G")
                else:
                    row.append(symbols.get(cell, "?"))
            rows.append(" ".join(row))
        return "\n".join(rows)

    @property
    def observation_space_shape(self) -> tuple[int]:
        """Shape of the flat observation vector."""
        return (self.obs_dim,)

    @property
    def action_space_size(self) -> int:
        """Number of discrete actions."""
        return self.N_ACTIONS

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_grid(self) -> None:
        """Construct a new random grid with border walls, interior obstacles, agent, goal."""
        g = np.zeros((self.size, self.size), dtype=np.int32)

        # border walls
        g[0, :]  = WALL
        g[-1, :] = WALL
        g[:, 0]  = WALL
        g[:, -1] = WALL

        # interior walls
        interior = [(r, c)
                    for r in range(1, self.size - 1)
                    for c in range(1, self.size - 1)]
        n_walls = int(len(interior) * self.wall_density)
        wall_indices = self.rng.choice(len(interior), size=n_walls, replace=False)
        for idx in wall_indices:
            r, c = interior[idx]
            g[r, c] = WALL

        self._grid = g

        # place agent and goal on distinct empty interior cells
        empty_cells = [(r, c) for r, c in interior if g[r, c] == EMPTY]
        if len(empty_cells) < 2:
            raise RuntimeError("Grid too dense: fewer than 2 empty cells after wall placement.")

        chosen = self.rng.choice(len(empty_cells), size=2, replace=False)
        self._agent_pos = empty_cells[chosen[0]]
        self._goal_pos  = empty_cells[chosen[1]]

    def _move_agent(self, action: int) -> None:
        """Attempt to move agent; stay in place if target is a wall."""
        dr, dc = self._DELTAS[action]
        r, c   = self._agent_pos
        nr, nc = r + dr, c + dc

        if 0 <= nr < self.size and 0 <= nc < self.size and self._grid[nr, nc] != WALL:
            self._agent_pos = (nr, nc)

    def _compute_reward(self) -> tuple[float, bool]:
        """
        Sparse reward scheme:
          +1.0  on reaching the goal
          -0.01 per step (time penalty to encourage efficiency)

        Returns
        -------
        (reward, success)
        """
        if self._agent_pos == self._goal_pos:
            return 1.0, True
        return -0.01, False

    def _observe(self) -> np.ndarray:
        """
        Extract a (2*radius+1)^2 local window centred on the agent.

        Out-of-bound cells are treated as walls. Values are normalised to [0, 1]
        by dividing by the number of distinct cell types (4).

        Returns
        -------
        np.ndarray, shape (obs_dim,), dtype float32
        """
        r, c   = self._agent_pos
        radius = self.obs_radius
        window = np.full((2 * radius + 1, 2 * radius + 1), WALL, dtype=np.float32)

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                wr, wc = dr + radius, dc + radius
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if (nr, nc) == self._agent_pos:
                        window[wr, wc] = AGENT
                    elif (nr, nc) == self._goal_pos:
                        window[wr, wc] = GOAL
                    else:
                        window[wr, wc] = self._grid[nr, nc]

        return (window.flatten() / 3.0).astype(np.float32)  # normalise to [0,1]
    