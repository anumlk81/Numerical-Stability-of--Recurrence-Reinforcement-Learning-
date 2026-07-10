"""
Validation script for environment/environment.py.

Runs from the project root:
    python experiments/validate_environment.py

Each section validates one contract of the environment.
A final summary reports total pass/fail counts.
No testing framework is used -- plain assertions with descriptive messages.
"""

import sys
import collections
import numpy as np

sys.path.insert(0, ".")
from environment import GridWorldEnv, WALL, AGENT, GOAL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    suffix = f"  -- {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    results.append((name, status))


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def navigate_to(env: GridWorldEnv, target: tuple[int, int]) -> list[int]:
    """
    BFS from current agent position to target on the current grid.
    Returns list of actions or empty list if unreachable.
    Treats all non-wall cells as traversable.
    """
    start = env._agent_pos
    if start == target:
        return []
    deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    queue  = collections.deque([(start, [])])
    seen   = {start}
    while queue:
        pos, path = queue.popleft()
        for action, (dr, dc) in enumerate(deltas):
            nr, nc = pos[0] + dr, pos[1] + dc
            npos   = (nr, nc)
            if npos in seen:
                continue
            if not (0 <= nr < env.size and 0 <= nc < env.size):
                continue
            if env._grid[nr, nc] == WALL:
                continue
            new_path = path + [action]
            if npos == target:
                return new_path
            seen.add(npos)
            queue.append((npos, new_path))
    return []


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

section("1. Construction")

env10 = GridWorldEnv(size=10, obs_radius=2, seed=0)
check("10x10 instantiates", True)

env15 = GridWorldEnv(size=15, obs_radius=2, seed=0)
check("15x15 instantiates", True)

check(
    "observation_space_shape is (25,)",
    env10.observation_space_shape == (25,),
    f"got {env10.observation_space_shape}",
)
check(
    "action_space_size is 4",
    env10.action_space_size == 4,
    f"got {env10.action_space_size}",
)
check(
    "obs_dim matches (2*radius+1)^2",
    env10.obs_dim == 25,
    f"got {env10.obs_dim}",
)
check(
    "max_steps defaults to 4*size^2",
    env10.max_steps == 4 * 10 * 10,
    f"got {env10.max_steps}",
)

# ---------------------------------------------------------------------------
# 2. Observation contract
# ---------------------------------------------------------------------------

section("2. Observation contract")

obs = env10.reset()
check("reset returns ndarray",     isinstance(obs, np.ndarray))
check("dtype is float32",          obs.dtype == np.float32, f"got {obs.dtype}")
check("shape is (25,)",            obs.shape == (25,),       f"got {obs.shape}")
check("values in [0, 1]",          float(obs.min()) >= 0.0 and float(obs.max()) <= 1.0,
      f"range [{obs.min():.4f}, {obs.max():.4f}]")
check("agent encoded at centre",   obs[12] == AGENT / 3.0,  f"centre={obs[12]:.4f}")

obs2, _, _, _ = env10.step(0)
check("step returns ndarray",      isinstance(obs2, np.ndarray))
check("step obs dtype is float32", obs2.dtype == np.float32)
check("step obs shape is (25,)",   obs2.shape == (25,))
check("step obs values in [0, 1]", float(obs2.min()) >= 0.0 and float(obs2.max()) <= 1.0)

obs15 = env15.reset()
check("15x15 obs shape is (25,)", obs15.shape == (25,), f"got {obs15.shape}")

full_env = GridWorldEnv(size=10, obs_radius=2, full_obs=True, seed=0)
full_obs = full_env.reset()
check(
    "full_obs observation_space_shape is (100,)",
    full_env.observation_space_shape == (100,),
    f"got {full_env.observation_space_shape}",
)
check(
    "full_obs obs_dim is 100",
    full_env.obs_dim == 100,
    f"got {full_env.obs_dim}",
)
check(
    "full_obs reset shape is (100,)",
    full_obs.shape == (100,),
    f"got {full_obs.shape}",
)
check(
    "full_obs reset dtype is float32",
    full_obs.dtype == np.float32,
    f"got {full_obs.dtype}",
)

local_env = GridWorldEnv(size=10, obs_radius=2, full_obs=False, seed=0)
local_obs = local_env.reset()
check(
    "full_obs and local envs build identical grids with the same seed",
    np.array_equal(full_env._grid, local_env._grid)
    and full_env._agent_pos == local_env._agent_pos
    and full_env._goal_pos == local_env._goal_pos,
)

expected_full = full_env._grid.astype(np.float32).copy()
ar, ac = full_env._agent_pos
gr, gc = full_env._goal_pos
expected_full[ar, ac] = AGENT
expected_full[gr, gc] = GOAL
check(
    "full_obs exposes the complete flattened grid",
    np.allclose(full_obs * 3.0, expected_full.flatten()),
)
check(
    "full_obs differs from local window observation",
    full_obs.shape != local_obs.shape,
    f"local={local_obs.shape}, full={full_obs.shape}",
)

# ---------------------------------------------------------------------------
# 3. Grid structure
# ---------------------------------------------------------------------------

section("3. Grid structure")

env10.reset()
g = env10._grid

check("border row 0 is all wall",    np.all(g[0, :]  == WALL))
check("border row -1 is all wall",   np.all(g[-1, :] == WALL))
check("border col 0 is all wall",    np.all(g[:, 0]  == WALL))
check("border col -1 is all wall",   np.all(g[:, -1] == WALL))

interior_vals = g[1:-1, 1:-1].flatten()
check(
    "interior cells are only EMPTY or WALL",
    set(interior_vals.tolist()).issubset({0, 1}),
    f"found values {set(interior_vals.tolist())}",
)

agent_r, agent_c = env10._agent_pos
goal_r,  goal_c  = env10._goal_pos
check("agent not on wall", g[agent_r, agent_c] != WALL)
check("goal not on wall",  g[goal_r,  goal_c]  != WALL)
check("agent != goal pos", env10._agent_pos != env10._goal_pos)
check("agent inside border", 0 < agent_r < env10.size - 1 and 0 < agent_c < env10.size - 1)
check("goal inside border",  0 < goal_r  < env10.size - 1 and 0 < goal_c  < env10.size - 1)

# ---------------------------------------------------------------------------
# 4. Wall collision
# ---------------------------------------------------------------------------

section("4. Wall collision")

env = GridWorldEnv(size=10, obs_radius=2, wall_density=0.0, seed=1)
env.reset()

failed_bounds = False
for trial in range(200):
    action = int(env.rng.integers(0, 4))
    _, _, done, _ = env.step(action)
    after = env._agent_pos
    if done:
        env.reset()
        continue
    if not (0 < after[0] < env.size - 1 and 0 < after[1] < env.size - 1):
        check(f"agent stays in bounds (trial {trial})", False, f"pos={after}")
        failed_bounds = True
        break

if not failed_bounds:
    check("agent stays in bounds (200 random steps)", True)

env2 = GridWorldEnv(size=10, obs_radius=2, wall_density=0.0, seed=2)
env2.reset()
r, c = env2._agent_pos
if r > 1:
    env2._grid[r - 1, c] = WALL
    pos_before = env2._agent_pos
    env2.step(0)
    check(
        "agent blocked by injected wall",
        env2._agent_pos == pos_before,
        f"before={pos_before}, after={env2._agent_pos}",
    )

# ---------------------------------------------------------------------------
# 5. Reward and termination
# ---------------------------------------------------------------------------

section("5. Reward and termination")

env = GridWorldEnv(size=10, obs_radius=2, wall_density=0.0, seed=3)
env.reset()
path = navigate_to(env, env._goal_pos)

if not path:
    check("BFS path to goal found", False, "goal unreachable -- cannot validate reward")
else:
    check("BFS path to goal found", True, f"path length={len(path)}")

    for i, action in enumerate(path[:-1]):
        obs, reward, done, info = env.step(action)
        check(
            f"step {i+1}: reward=-0.01",
            abs(reward - (-0.01)) < 1e-6,
            f"reward={reward}",
        )
        check(
            f"step {i+1}: done=False",
            done is False,
            f"done={done}",
        )

    obs, reward, done, info = env.step(path[-1])
    check("goal step: reward=+1.0",   abs(reward - 1.0) < 1e-6,  f"reward={reward}")
    check("goal step: done=True",     done is True,               f"done={done}")
    check("goal step: success=True",  info["success"] is True)
    check("goal step: timeout=False", info["timeout"] is False)

# timeout path -- force an unreachable goal position
env_t = GridWorldEnv(size=10, obs_radius=2, max_steps=5, seed=4)
env_t.reset()
env_t._goal_pos = (-1, -1)
for step_i in range(4):
    _, reward, done, info = env_t.step(0)
    check(f"pre-timeout step {step_i+1}/5: done=False", done is False, f"done={done}")
    check(f"pre-timeout step {step_i+1}/5: reward=-0.01", abs(reward - (-0.01)) < 1e-6)
_, _, done, info = env_t.step(0)
check("timeout at max_steps: done=True",     done is True,            f"done={done}")
check("timeout at max_steps: timeout=True",  info["timeout"] is True)
check("timeout at max_steps: success=False", info["success"] is False)

# ---------------------------------------------------------------------------
# 6. Reproducibility
# ---------------------------------------------------------------------------

section("6. Reproducibility")


def collect_episode(size: int, seed_env: int, seed_reset: int, n_steps: int = 20) -> list:
    e   = GridWorldEnv(size=size, obs_radius=2, seed=seed_env)
    obs = e.reset(seed=seed_reset)
    rng = np.random.default_rng(0)
    trajectory = [obs.copy()]
    for _ in range(n_steps):
        action = int(rng.integers(0, 4))
        obs, _, done, _ = e.step(action)
        trajectory.append(obs.copy())
        if done:
            break
    return trajectory


traj_a = collect_episode(10, seed_env=7, seed_reset=99)
traj_b = collect_episode(10, seed_env=7, seed_reset=99)
check(
    "same seed yields identical trajectory",
    all(np.array_equal(a, b) for a, b in zip(traj_a, traj_b)),
)

traj_c = collect_episode(10, seed_env=7, seed_reset=100)
check(
    "different seed yields different trajectory",
    not all(np.array_equal(a, b) for a, b in zip(traj_a, traj_c)),
)

env_r = GridWorldEnv(size=10, obs_radius=2, seed=5)
env_r.reset()
for _ in range(10):
    env_r.step(int(env_r.rng.integers(0, 4)))
env_r.reset()
check("reset clears step count", env_r._step_count == 0, f"got {env_r._step_count}")
check("reset clears done flag",  env_r._done is False,    f"got {env_r._done}")

# ---------------------------------------------------------------------------
# 7. Exception handling
# ---------------------------------------------------------------------------

section("7. Exception handling")


def assert_raises(exc_type: type, fn, label: str) -> None:
    try:
        fn()
        check(label, False, f"expected {exc_type.__name__}, no exception raised")
    except exc_type as e:
        check(label, True, str(e))
    except Exception as e:
        check(label, False, f"wrong exception: {type(e).__name__}: {e}")


assert_raises(ValueError,   lambda: GridWorldEnv(size=12),                    "invalid size")
assert_raises(ValueError,   lambda: GridWorldEnv(size=10, obs_radius=0),      "obs_radius=0")
assert_raises(ValueError,   lambda: GridWorldEnv(size=10, obs_radius=10),     "obs_radius>=size")
assert_raises(ValueError,   lambda: GridWorldEnv(size=10, wall_density=-0.1), "negative wall_density")
assert_raises(ValueError,   lambda: GridWorldEnv(size=10, wall_density=0.5),  "wall_density>=0.5")

env_ex = GridWorldEnv(size=10, obs_radius=2, seed=6)
env_ex.reset()
env_ex._done = True
assert_raises(RuntimeError, lambda: env_ex.step(0),  "step on done episode")

env_ex2 = GridWorldEnv(size=10, obs_radius=2, seed=6)
env_ex2.reset()
assert_raises(ValueError, lambda: env_ex2.step(-1), "action=-1")
assert_raises(ValueError, lambda: env_ex2.step(4),  "action=4")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("=" * 40)
total  = len(results)
passed = sum(1 for _, s in results if s == PASS)
failed = total - passed
print(f"Results: {passed}/{total} passed")
if failed:
    print(f"Failed ({failed}):")
    for name, status in results:
        if status == FAIL:
            print(f"  - {name}")
print("=" * 40)

sys.exit(0 if failed == 0 else 1)