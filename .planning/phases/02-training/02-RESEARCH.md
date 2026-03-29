# Phase 2: Training - Research

**Researched:** 2026-03-29
**Domain:** SB3 PPO self-play training, SubprocVecEnv, VecNormalize, TensorBoard callbacks
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Per-env opponent assignment — each SubprocVecEnv subprocess is randomly assigned an opponent from the pool (50% latest checkpoint, 50% historical). Not all-envs-same-snapshot.
- **D-02:** A SB3 callback fires every ~500k steps, snapshots the current policy to the checkpoint pool, then re-assigns opponents to each env.
- **D-03:** Pool size target: ~20 checkpoints (older snapshots evicted when pool exceeds max size).
- **D-04:** Frozen opponent policies are loaded from disk by path — each subprocess env receives a checkpoint path string and loads the SB3 model on demand. No pickling of model weights across subprocess boundary.
- **D-05:** Phase 1 reward function is accepted as-is — no coefficient changes.
- **D-06:** PROJECT.md note "Full reward shaping (Phase 2)" is acknowledging Phase 1 work, not a new Phase 2 requirement.
- **D-07:** Single `train.py` script with argparse. Flags: `--total-steps`, `--n-envs`, `--resume` (path to checkpoint .zip to resume from).
- **D-08:** Set-and-forget run — no mid-run hyperparameter changes.
- **D-09:** Checkpoint directory hardcoded to `/workspace/checkpoints/` (RunPod persistent volume path). Not configurable via flag.
- **D-10:** Checkpoint naming: `step_{N}.zip` + `step_{N}_vecnorm.pkl`.
- **D-11:** VecNormalize wraps SubprocVecEnv from step 1. Stats accumulate over the entire training run.
- **D-12:** VecNormalize stats saved as a separate `.pkl` file alongside each SB3 `.zip` checkpoint.
- **D-13:** Custom TensorBoard callback logs `goal_rate` and `puck_possession` averaged over episode infos per rollout. `score` key and `r_possession` key already in `info` dict from `HockeyEnv.step()`.

### Claude's Discretion

- PPO hyperparameters (learning rate, clip range, n_steps, batch_size, n_epochs, gamma, gae_lambda) — use SB3 defaults as starting point, tune for RTX 4090 memory
- Number of parallel envs between 8–16 (TRAIN-02 range) — choose based on env step throughput benchmarking
- TensorBoard callback implementation details (averaging window, logging frequency)

### Deferred Ideas (OUT OF SCOPE)

- Curriculum reward shaping (decay shaped rewards over training)
- Config YAML for hyperparameters
- Per-agent reward component logging in TensorBoard (goal_rate + puck_possession are sufficient per TRAIN-04)
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TRAIN-01 | Self-play uses a historical opponent pool of ~20 checkpoints; pool updated every ~500k training steps with 50% latest / 50% historical mix | Self-play pool callback pattern; `set_attr` via VecEnv for per-env opponent reassignment |
| TRAIN-02 | SubprocVecEnv configured for 8–16 parallel environments on RunPod RTX 4090 | SB3 SubprocVecEnv API; n_envs=16 with n_steps=512 gives 8,192-step rollout buffer; RTX 4090 comfortably handles this for obs_dim=22 |
| TRAIN-03 | Checkpoints saved every 30 minutes of wall-time, labelled by step count; target 50–100M steps in 8–10 hour run | Wall-time checkpoint callback using `time.time()`; `model.save()` + `VecNormalize.save()` pattern; `/workspace/checkpoints/` volume |
| TRAIN-04 | TensorBoard logging records episode reward, goal rate, and puck possession stats per checkpoint | `self.logger.record()` in `_on_rollout_end`; `self.locals["infos"]` to collect per-step info dicts; `score` and `r_possession` already emitted by `HockeyEnv.step()` |
</phase_requirements>

---

## Summary

Phase 2 produces a single `train.py` entry-point for a human to run on a RunPod RTX 4090 for an 8–10 hour session. The output is a set of SB3 `.zip` + VecNormalize `.pkl` checkpoint pairs on `/workspace/checkpoints/`, plus TensorBoard logs on `/workspace/tb_logs/`. No server or CI automation is involved — this is a manual run-and-monitor workflow.

The two architectural challenges unique to this phase are (1) the self-play pool mechanic and (2) the wall-time checkpoint trigger. Both are implemented as SB3 `BaseCallback` subclasses. The self-play pool callback must cross the `SubprocVecEnv` subprocess boundary without pickling model weights — it does this by passing checkpoint path strings to each subprocess via `VecEnv.set_attr()` and having each subprocess load the SB3 model inside its own process. The wall-time callback uses `time.time()` to fire independently of step count.

All environment integration points (the `frozen_opponent_fn` constructor param, the `info` dict keys, the 22-float obs contract) are already established by Phase 1. Phase 2 consumes them without modification.

**Primary recommendation:** Implement two callbacks — `SelfPlayPoolCallback` and `WallTimeCheckpointCallback` — composed with `CallbackList`. Use SB3 defaults for most PPO hyperparameters; the main tuning lever is `n_envs` (16 recommended) and `n_steps` (512, giving 8,192-step rollout buffer for fast iteration).

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| stable-baselines3 | 2.7.1 | PPO implementation, VecEnv, VecNormalize | Pinned in CLAUDE.md; proven SubprocVecEnv support; SB3 2.7.1 is latest stable |
| PyTorch | 2.11 | Neural network backend | Pinned in CLAUDE.md; required for SB3; dynamo=True ONNX export downstream |
| tensorboard | 2.x | Training metrics dashboard | Built into SB3 logger; `logger.record()` API |
| Python | 3.12 | Runtime | 3.11 unavailable on this machine (confirmed Phase 1); 3.12 works with SB3 2.7.1 |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | 1.x/2.x | Array manipulation in callbacks | Already installed; used in obs/reward code |
| torch (CPU eval) | 2.11 | Loading saved SB3 checkpoints inside subprocesses | Opponent policy loads via `PPO.load()` in each subprocess |

### Installation (on RunPod)

```bash
pip install stable-baselines3==2.7.1 tensorboard torch==2.11 --extra-index-url https://download.pytorch.org/whl/cu126
```

Note: `requirements.txt` currently only has `dm-control==1.0.38 gymnasium==0.29.1 numpy pytest`. Phase 2 must extend this or provide a separate `requirements-train.txt` for the RunPod environment.

---

## Architecture Patterns

### Recommended Project Structure

```
train.py                      # Entry point — argparse, env factory, PPO setup, .learn()
training/
├── __init__.py
├── self_play_callback.py     # SelfPlayPoolCallback — pool management + per-env reassignment
├── checkpoint_callback.py    # WallTimeCheckpointCallback — wall-time trigger
└── tb_callback.py            # TensorBoardCustomCallback — goal_rate + puck_possession
/workspace/
├── checkpoints/
│   ├── step_5000000.zip
│   ├── step_5000000_vecnorm.pkl
│   └── pool/                 # Opponent pool snapshots (managed by SelfPlayPoolCallback)
│       ├── pool_step_500000.zip
│       └── ...
└── tb_logs/
    └── hockey_ppo_1/
```

### Pattern 1: SubprocVecEnv + VecNormalize Setup

**What:** Each subprocess gets a `HockeyEnv` instance. VecNormalize wraps the whole vec env. Opponent function is injected via a path-based loader inside the subprocess.

**When to use:** Always — this is the core training stack.

```python
# Source: SB3 docs + env interface from Phase 1
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

CHECKPOINT_DIR = "/workspace/checkpoints"
POOL_DIR = os.path.join(CHECKPOINT_DIR, "pool")

def make_env(agent_idx: int, opponent_path_holder: list):
    """Factory that captures a mutable list[str] as opponent path holder."""
    def _init():
        from stable_baselines3 import PPO as _PPO
        import numpy as np

        def frozen_fn(obs):
            path = opponent_path_holder[0]
            if path is None:
                return np.random.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)
            # Load lazily — cache inside closure or reload each call
            model = _PPO.load(path, device="cpu")
            action, _ = model.predict(obs, deterministic=True)
            return action

        from env.hockey_env import HockeyEnv
        return HockeyEnv(agent_idx=0, frozen_opponent_fn=frozen_fn)
    return _init
```

**Critical note on D-04:** The `frozen_fn` closure MUST load the model inside the subprocess. Do not pass a loaded `PPO` object across the subprocess boundary — SB3 models are not picklable by default. Pass a path string (or use `set_attr` to update a mutable holder). The simplest pattern is a `list[str]` holder that the callback updates via `training_env.set_attr("opponent_path", new_path, indices=[i])`, where the env exposes `opponent_path` as a settable attribute.

### Pattern 2: SelfPlayPoolCallback

**What:** A `BaseCallback` subclass that fires every `pool_update_freq` steps. It snapshots the current policy, manages the pool queue, and reassigns opponents across all envs.

**When to use:** Satisfies TRAIN-01 (D-01, D-02, D-03, D-04).

```python
# Source: SB3 BaseCallback docs + decisions D-01 through D-04
import os, time, random
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback

class SelfPlayPoolCallback(BaseCallback):
    def __init__(self, pool_dir: str, pool_update_freq: int = 500_000,
                 max_pool_size: int = 20, verbose: int = 0):
        super().__init__(verbose)
        self.pool_dir = pool_dir
        self.pool_update_freq = pool_update_freq
        self.max_pool_size = max_pool_size
        self._pool: deque = deque()  # deque of checkpoint paths
        self._latest_path: str | None = None
        os.makedirs(pool_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.pool_update_freq < self.training_env.num_envs:
            self._snapshot_and_reassign()
        return True

    def _snapshot_and_reassign(self):
        step = self.num_timesteps
        path = os.path.join(self.pool_dir, f"pool_step_{step}.zip")
        self.model.save(path)
        self._pool.append(path)
        self._latest_path = path
        # Evict oldest if pool exceeds max
        while len(self._pool) > self.max_pool_size:
            old = self._pool.popleft()
            if os.path.exists(old):
                os.remove(old)
        # Reassign each env: 50% latest, 50% historical
        pool_list = list(self._pool)
        n_envs = self.training_env.num_envs
        for i in range(n_envs):
            if random.random() < 0.5 or len(pool_list) <= 1:
                chosen = self._latest_path
            else:
                chosen = random.choice(pool_list[:-1])  # historical only
            self.training_env.set_attr("opponent_path", chosen, indices=[i])
```

**Key constraint:** `HockeyEnv` must expose `opponent_path` as a writable attribute or setter that the subprocess can receive via `set_attr`. The subprocess's `frozen_fn` closure reads this value each call.

### Pattern 3: WallTimeCheckpointCallback

**What:** A `BaseCallback` that saves both `.zip` and `_vecnorm.pkl` every 30 minutes of elapsed wall time.

**When to use:** Satisfies TRAIN-03 (D-09, D-10, D-11, D-12).

```python
# Source: SB3 BaseCallback docs + decisions D-09 through D-12
import time, os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

CHECKPOINT_DIR = "/workspace/checkpoints"

class WallTimeCheckpointCallback(BaseCallback):
    def __init__(self, checkpoint_dir: str = CHECKPOINT_DIR,
                 interval_minutes: float = 30.0, verbose: int = 1):
        super().__init__(verbose)
        self.checkpoint_dir = checkpoint_dir
        self.interval_seconds = interval_minutes * 60
        self._last_save_time: float = 0.0
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _on_training_start(self) -> None:
        self._last_save_time = time.time()

    def _on_step(self) -> bool:
        now = time.time()
        if now - self._last_save_time >= self.interval_seconds:
            self._save()
            self._last_save_time = now
        return True

    def _save(self):
        step = self.num_timesteps
        model_path = os.path.join(self.checkpoint_dir, f"step_{step}.zip")
        vecnorm_path = os.path.join(self.checkpoint_dir, f"step_{step}_vecnorm.pkl")
        self.model.save(model_path)
        # Retrieve VecNormalize wrapper from training env
        vec_norm = self.model.get_vec_normalize_env()
        if vec_norm is not None:
            vec_norm.save(vecnorm_path)
        if self.verbose:
            print(f"[Checkpoint] step={step} saved to {model_path}")
```

### Pattern 4: TensorBoard Custom Callback

**What:** A `BaseCallback` subclass that collects `info` dict values per rollout and logs averages to TensorBoard.

**When to use:** Satisfies TRAIN-04 (D-13).

```python
# Source: SB3 callbacks.py source + TensorBoard integration
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class TensorBoardCustomCallback(BaseCallback):
    """Log goal_rate and puck_possession per rollout."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_scores: list = []
        self._episode_possessions: list = []

    def _on_step(self) -> bool:
        # self.locals["infos"] is a list[dict] of length n_envs
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if done:
                # Episode ended: collect final episode stats
                score = info.get("score", [0, 0])
                goals = sum(score)
                self._episode_scores.append(float(goals > 0))
                self._episode_possessions.append(
                    float(info.get("r_possession", 0.0) > 0))
        return True

    def _on_rollout_end(self) -> None:
        if self._episode_scores:
            self.logger.record(
                "hockey/goal_rate",
                float(np.mean(self._episode_scores)))
        if self._episode_possessions:
            self.logger.record(
                "hockey/puck_possession_rate",
                float(np.mean(self._episode_possessions)))
        self._episode_scores.clear()
        self._episode_possessions.clear()
```

**Note on `info` key at episode end:** SB3's `SubprocVecEnv` stores the final-step `info` dict in `info["terminal_observation"]` for truncated episodes, but the raw `info` dict (including `score`) is still passed through `self.locals["infos"]` at every step. Collect on `done=True` steps to get episode-end values, not mid-episode accumulations.

### Pattern 5: Resume from Checkpoint

**What:** Load both `.zip` and `_vecnorm.pkl`, re-wrap the env, continue `.learn()`.

**When to use:** Satisfies D-07 (`--resume` flag).

```python
# Source: SB3 VecNormalize docs
if args.resume:
    vec_norm_path = args.resume.replace(".zip", "_vecnorm.pkl")
    env = SubprocVecEnv([make_env(0, holder) for holder in holders])
    env = VecNormalize.load(vec_norm_path, env)
    env.training = True  # continue updating stats
    model = PPO.load(args.resume, env=env, device="cuda")
else:
    env = SubprocVecEnv([...])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    model = PPO("MlpPolicy", env, ...)
```

### Pattern 6: Compose Callbacks with CallbackList

```python
from stable_baselines3.common.callbacks import CallbackList

callback = CallbackList([
    SelfPlayPoolCallback(pool_dir=POOL_DIR, pool_update_freq=500_000),
    WallTimeCheckpointCallback(checkpoint_dir=CHECKPOINT_DIR, interval_minutes=30),
    TensorBoardCustomCallback(),
])
model.learn(total_timesteps=args.total_steps, callback=callback)
```

### Anti-Patterns to Avoid

- **Pickling SB3 model objects across subprocess boundary:** SB3 `PPO` objects contain PyTorch models and are not safely picklable. Always pass path strings and load inside the target process.
- **Using `DummyVecEnv` for training:** The dm_control environment is computationally non-trivial. `SubprocVecEnv` distributes across processes; `DummyVecEnv` runs serially in one thread — it is only suitable for fast toy envs.
- **Adding VecNormalize after any steps have been taken:** VecNormalize must wrap the vec env before any step is taken. Calling `model.set_env()` to add it later voids the accumulated statistics.
- **Saving VecNormalize stats only at the end:** The run may be interrupted. Save `_vecnorm.pkl` alongside every checkpoint so any checkpoint is self-contained.
- **Checkpoint freq in `n_calls` (number of `.step()` calls):** With `n_envs=16`, each `._on_step()` call counts 16 environment steps. Use `self.num_timesteps` (total env steps), not `self.n_calls`, for step-count logic.
- **`if __name__ == "__main__"` guard:** Required on Windows (spawn/forkserver start method). On Linux (RunPod), fork is the default and this guard is optional — but is still good practice to include.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Rollout buffer management | Custom ring buffer | SB3's built-in `RolloutBuffer` (used by PPO internally) | GAE advantage computation, mini-batch slicing — non-trivial to get right |
| Observation normalization | Inline mean/variance tracking | `VecNormalize` | Handles running stats correctly across resets and episode boundaries; Phase 3 ONNX export reads stats from the `.pkl` file |
| Subprocess env management | Custom multiprocessing pool | `SubprocVecEnv` | Handles pipe communication, exception propagation, reset synchronization across processes |
| Checkpoint numbering / rotation | Custom cleanup script | Pool eviction in `SelfPlayPoolCallback` | Inline with training loop; deque makes eviction trivial |
| TensorBoard file format | Raw file writes | `self.logger.record()` + SB3 TensorBoard backend | SB3 initializes the TensorBoard `SummaryWriter` automatically when `tensorboard_log` arg is set |

**Key insight:** SB3 handles all the RL plumbing. Phase 2 code is ~300 lines total — three callback classes and a 50-line `train.py`. Anything beyond that is scope creep.

---

## Common Pitfalls

### Pitfall 1: batch_size must divide n_steps * n_envs

**What goes wrong:** SB3 raises `AssertionError: `batch_size` (X) is not a divisor of `n_steps * n_envs` (Y)` immediately at `.learn()`.

**Why it happens:** PPO splits the rollout buffer into mini-batches. The buffer size is `n_steps * n_envs`; `batch_size` must be an integer divisor.

**How to avoid:** With `n_envs=16, n_steps=512`, buffer = 8,192. Use `batch_size=256` (32 mini-batches) or `batch_size=512` (16 mini-batches). With `n_envs=8, n_steps=512`, buffer = 4,096. Use `batch_size=256` or `batch_size=128`.

**Warning signs:** Immediate crash on `model.learn()` first call.

### Pitfall 2: VecNormalize training mode must be True during training

**What goes wrong:** If `env.training = False` is set (e.g., copied from an eval snippet), the running stats freeze and observations diverge from the policy's training distribution over time.

**Why it happens:** Copy-paste from inference/eval code snippets which set `training=False` before predicting.

**How to avoid:** Only set `training=False` and `norm_reward=False` when doing evaluation outside the main training loop. The training env should always have `env.training=True`.

**Warning signs:** Reward curve that was rising suddenly flattens or collapses after a code edit.

### Pitfall 3: `num_timesteps` vs `n_calls` in callback step-count logic

**What goes wrong:** Pool update fires at wrong intervals (16x too frequently or too rarely).

**Why it happens:** `self.n_calls` counts the number of times `_on_step()` was invoked — once per `env.step()` call. With `n_envs=16`, each call advances 16 timesteps. `self.num_timesteps` is the actual total environment step count.

**How to avoid:** Always use `self.num_timesteps` for threshold comparisons like `self.num_timesteps % 500_000 < self.training_env.num_envs`.

**Warning signs:** Pool updates fire on every step or never fire during a run.

### Pitfall 4: SubprocVecEnv set_attr timing

**What goes wrong:** Opponent path updated in callback fires mid-rollout, causing inconsistent behavior within a single rollout batch.

**Why it happens:** `_on_step()` fires inside the rollout collection loop. `set_attr()` sends an IPC message to the subprocess immediately.

**How to avoid:** Call `set_attr` in `_on_rollout_end()` or at minimum check that you're OK with mid-rollout transitions (which are fine for self-play — no correctness requirement that a rollout uses a single opponent).

### Pitfall 5: HockeyEnv frozen_fn reloads model on every call

**What goes wrong:** Training throughput degrades catastrophically. Loading a SB3 model from disk takes ~100–500ms. With 16 envs at 50+ fps, calling `PPO.load()` inside `frozen_fn` on every `step()` call is a ~50x slowdown.

**Why it happens:** Naive implementation of `frozen_fn` calls `PPO.load()` inside the function body.

**How to avoid:** Cache the loaded model inside the subprocess. One pattern: the `frozen_fn` closure holds a `dict{"path": str, "model": PPO | None}`. When `path` changes (detected by `set_attr`), reload once. On subsequent calls with the same path, use the cached model.

**Warning signs:** GPU utilization on RunPod drops from 90%+ to <10% during training; `htop` shows 16 subprocesses consuming 100% CPU.

### Pitfall 6: Episode info collection on wrong step

**What goes wrong:** `goal_rate` in TensorBoard is always 0 or counts double.

**Why it happens:** Collecting `info["score"]` on every step rather than only on `done=True` steps. The `score` key in the info dict persists mid-episode at the same value.

**How to avoid:** Filter: `if dones[i]: collect(infos[i])`.

### Pitfall 7: Pool snapshot path collision

**What goes wrong:** Pool files from a resumed run overwrite earlier pool files at the same step number.

**Why it happens:** Resume restores `num_timesteps` to the checkpoint step count, so the first pool snapshot after resume may write to `pool_step_{N}.zip` which already exists.

**How to avoid:** Use `os.makedirs(exist_ok=True)` and consider appending a timestamp to pool file names, or skip snapshot if the file already exists.

---

## Code Examples

### Complete train.py skeleton

```python
# Source: SB3 docs + CONTEXT.md decisions D-07, D-09
import argparse, os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

CHECKPOINT_DIR = "/workspace/checkpoints"
TB_LOG_DIR = "/workspace/tb_logs"
POOL_DIR = os.path.join(CHECKPOINT_DIR, "pool")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=100_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint .zip to resume from")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(POOL_DIR, exist_ok=True)

    # Each env gets its own mutable opponent-path holder (a list[str | None])
    holders = [["random"] for _ in range(args.n_envs)]

    def make_env_fn(idx):
        return make_env(agent_idx=0, opponent_path_holder=holders[idx])

    if args.resume:
        vec_norm_path = args.resume.replace(".zip", "_vecnorm.pkl")
        env = SubprocVecEnv([make_env_fn(i) for i in range(args.n_envs)])
        env = VecNormalize.load(vec_norm_path, env)
        env.training = True
        model = PPO.load(args.resume, env=env, device="cuda",
                         tensorboard_log=TB_LOG_DIR)
    else:
        env = SubprocVecEnv([make_env_fn(i) for i in range(args.n_envs)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        model = PPO(
            "MlpPolicy", env,
            n_steps=512,
            batch_size=256,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,   # small entropy bonus helps exploration
            verbose=1,
            tensorboard_log=TB_LOG_DIR,
            device="cuda",
        )

    from training.self_play_callback import SelfPlayPoolCallback
    from training.checkpoint_callback import WallTimeCheckpointCallback
    from training.tb_callback import TensorBoardCustomCallback
    from stable_baselines3.common.callbacks import CallbackList

    callback = CallbackList([
        SelfPlayPoolCallback(pool_dir=POOL_DIR, pool_update_freq=500_000),
        WallTimeCheckpointCallback(checkpoint_dir=CHECKPOINT_DIR),
        TensorBoardCustomCallback(),
    ])

    model.learn(total_timesteps=args.total_steps, callback=callback,
                reset_num_timesteps=not bool(args.resume))


if __name__ == "__main__":
    main()
```

### PPO hyperparameters for RTX 4090 (n_envs=16)

```python
# Rollout buffer = n_steps * n_envs = 512 * 16 = 8,192 steps
# batch_size=256 → 32 mini-batches per update
# With obs_dim=22, act_dim=4: GPU memory usage is negligible; n_steps is the
# primary throughput lever. Larger n_steps = more on-policy data per update.
PPO(
    "MlpPolicy", env,
    n_steps=512,          # 512 * 16 = 8,192 step rollout buffer
    batch_size=256,       # divides 8,192 evenly (32 mini-batches)
    n_epochs=10,          # SB3 default; fine for continuous control
    learning_rate=3e-4,   # SB3 default
    gamma=0.99,           # SB3 default
    gae_lambda=0.95,      # SB3 default
    clip_range=0.2,       # SB3 default
    ent_coef=0.01,        # slight increase from 0.0 default; aids exploration
    vf_coef=0.5,          # SB3 default
    max_grad_norm=0.5,    # SB3 default
    device="cuda",
)
```

**Rationale for `n_steps=512` (discretion area):** With 16 envs, the default `n_steps=2048` gives a 32,768-step buffer. An update then takes ~32k samples before any gradient step — very slow for early exploration of a sparse reward task. `n_steps=512` gives 8,192-step updates, which is still sufficient for PPO while allowing the policy to improve ~4x more frequently at the start of training.

### Model-caching opponent loader (subprocess-safe)

```python
# Source: addresses Pitfall 5
def make_frozen_fn(holder: list):
    """Returns a frozen_opponent_fn that caches the loaded model."""
    _cache = {"path": None, "model": None}

    def frozen_fn(obs):
        import numpy as np
        path = holder[0]
        if path == "random" or path is None:
            return np.random.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)
        if path != _cache["path"]:
            from stable_baselines3 import PPO as _PPO
            _cache["model"] = _PPO.load(path, device="cpu")
            _cache["path"] = path
        action, _ = _cache["model"].predict(obs, deterministic=True)
        return action

    return frozen_fn
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `CheckpointCallback` for periodic saves | Custom wall-time callback | Phase 2 decision | Step-based saves don't map cleanly to "30-min wall-time" sessions |
| Single-env training | SubprocVecEnv 8–16 | SB3 2.x | 8–16x throughput for non-trivial envs |
| Passing model weights to subprocess | Path strings + in-process load | D-04 | Avoids pickle errors with SB3 models |
| Torch legacy ONNX exporter | `dynamo=True` (Phase 3) | PyTorch 2.5 | Legacy exporter removed in 2.11 |

---

## Environment Availability

This phase executes on a RunPod RTX 4090 VM, not the local machine. The local machine (WSL2) is only used to write and commit code.

| Dependency | Required By | Available (local) | Available (RunPod) | Fallback |
|------------|------------|-------------------|--------------------|----------|
| Python 3.12 | Runtime | Yes (3.12.3) | Yes (RunPod PyTorch image) | — |
| stable-baselines3==2.7.1 | Training | No (not installed) | Install via pip | — |
| PyTorch 2.11 + CUDA 12.x | Training | No | RunPod PyTorch image | — |
| tensorboard | TensorBoard logs | No | Install via pip | — |
| /workspace/checkpoints/ | Checkpoint storage | N/A | RunPod persistent volume | Create dir at run start |
| dm-control==1.0.38 | Env import | Yes | Install via requirements.txt | — |
| MuJoCo 3.6.0 | dm-control | Yes (implicit) | Install via requirements.txt | — |

**Missing dependencies on RunPod with no fallback:**
- None — all are installable via `pip install`

**Note:** RunPod PyTorch images ship with PyTorch and CUDA pre-installed. SB3 and TensorBoard must be added via pip. Phase 2 should include a `setup.sh` or updated `requirements.txt` for the RunPod environment.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (already configured, pytest.ini exists) |
| Config file | `/home/napier19/hockeyViz/pytest.ini` |
| Quick run command | `pytest tests/ -m "not slow" -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TRAIN-01 | Pool callback snapshots policy and reassigns opponents | unit | `pytest tests/test_training.py::test_self_play_pool_snapshot -x` | ❌ Wave 0 |
| TRAIN-01 | Pool eviction when >20 checkpoints | unit | `pytest tests/test_training.py::test_pool_eviction -x` | ❌ Wave 0 |
| TRAIN-02 | SubprocVecEnv wrapping HockeyEnv runs N steps without crash | integration | `pytest tests/test_training.py::test_subproc_vec_env -x -m "not slow"` | ❌ Wave 0 |
| TRAIN-03 | WallTimeCheckpointCallback saves .zip + _vecnorm.pkl | unit | `pytest tests/test_training.py::test_wall_time_checkpoint -x` | ❌ Wave 0 |
| TRAIN-04 | TensorBoardCustomCallback records goal_rate and puck_possession | unit | `pytest tests/test_training.py::test_tb_callback_logging -x` | ❌ Wave 0 |

**Note:** Full training run (50–100M steps) is manual-only — it is a human-executed step on RunPod. Tests validate the callback logic and env wiring in isolation using short runs (100–1000 steps) in DummyVecEnv or a tiny SubprocVecEnv.

### Sampling Rate

- **Per task commit:** `pytest tests/test_training.py -x -m "not slow"`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_training.py` — covers TRAIN-01 through TRAIN-04 callback unit tests
- [ ] `training/__init__.py` — package marker
- [ ] `training/self_play_callback.py` — SelfPlayPoolCallback implementation
- [ ] `training/checkpoint_callback.py` — WallTimeCheckpointCallback implementation
- [ ] `training/tb_callback.py` — TensorBoardCustomCallback implementation
- [ ] Framework install on RunPod: `pip install stable-baselines3==2.7.1 tensorboard`

---

## Open Questions

1. **`set_attr` + mutable holder synchronization**
   - What we know: `SubprocVecEnv.set_attr(name, value, indices)` sends value to subprocess via IPC pipe; the subprocess sets it as an attribute on the env object.
   - What's unclear: The `frozen_fn` closure doesn't have a direct reference to an env attribute — it captures `holder` (a Python list) at factory time. `set_attr` sets an attribute on the `HockeyEnv` instance, not on the `holder` list. The env needs a `@opponent_path.setter` property that updates the holder in-place, not replaces it.
   - Recommendation: Add an `opponent_path` property to `HockeyEnv` that sets `self._opponent_path_holder[0] = value`, so `set_attr("opponent_path", path, indices=[i])` correctly reaches the closure.

2. **`requirements.txt` split: local vs RunPod**
   - What we know: Current `requirements.txt` works for local Phase 1 tests. SB3 + PyTorch are not needed locally.
   - What's unclear: Whether to add a separate `requirements-train.txt` or annotate sections.
   - Recommendation: Create `requirements-train.txt` containing `stable-baselines3==2.7.1 tensorboard` (torch is pre-installed on RunPod image). Document RunPod setup in a `RUNPOD_SETUP.md` or inline in `train.py` header comment.

3. **ent_coef=0.01 vs default 0.0**
   - What we know: The reward signal includes `r_goal=10.0` (sparse) and small shaped components. Pure exploitation early may collapse to suboptimal deterministc behaviors before goal-scoring is discovered.
   - What's unclear: Whether the shaped components (puck_toward_goal, possession) are dense enough to avoid entropy collapse without explicit ent_coef boost.
   - Recommendation: Start with `ent_coef=0.01`. If TensorBoard shows entropy collapsing while goal_rate stays at 0 past 10M steps, increase to 0.05 and restart.

---

## Project Constraints (from CLAUDE.md)

Directives that the planner must verify all tasks comply with:

- **Required versions:** SB3 2.7.1, PyTorch 2.11, Python 3.12 (3.11 unavailable; 3.12 confirmed working in Phase 1), CUDA 12.x, dm-control 1.0.38, gymnasium==0.29.1
- **No server runtime:** All training is a CLI script on RunPod — no Flask, FastAPI, or background service
- **Static deploy constraint:** Not directly applicable to Phase 2, but ONNX export downstream must be kept in mind — do not change obs_dim=22
- **OBS_SPEC contract:** MUST NOT change `obs_spec.py` or the 22-float layout during Phase 2
- **Checkpoint format:** SB3 `.zip` + separate `_vecnorm.pkl` — per D-12; Phase 3 ONNX export script reads both files independently
- **GSD workflow:** All file edits go through `/gsd:execute-phase`, not ad-hoc

---

## Sources

### Primary (HIGH confidence)

- [SB3 PPO source — default hyperparameters](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py) — verified defaults: lr=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0
- [SB3 callbacks.py source](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py) — BaseCallback interface, self.locals["infos"], CheckpointCallback pattern, self.logger.record()
- `env/hockey_env.py` (Phase 1 output) — `frozen_opponent_fn` signature, `info` dict keys, Gymnasium API
- `env/hockey_task.py` (Phase 1 output) — reward component keys: `r_goal, r_puck_toward_goal, r_possession, r_positioning, r_clustering, r_step_penalty` + `score`
- `env/obs_spec.py` (Phase 1 output) — OBS_DIM=22, canonical layout

### Secondary (MEDIUM confidence)

- [SB3 VecNormalize docs — save/load pattern](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html) — `VecNormalize.load(path, env)`, `model.get_vec_normalize_env()`
- [RunPod persistent volume docs](https://docs.runpod.io/tutorials/introduction/containers/persist-data) — `/workspace` volume confirmed persistent across pod restarts
- [SB3 VecEnv set_attr/get_attr API](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html) — `set_attr(name, value, indices)` confirmed for SubprocVecEnv

### Tertiary (LOW confidence)

- Community pattern for self-play with SB3 — no official SB3 self-play tutorial; pattern reconstructed from D-01 through D-04 decisions and SB3 VecEnv API. The specific `set_attr` + mutable holder wiring is novel and must be validated in tests (see Open Question 1).

---

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — CLAUDE.md specifies exact versions; SB3 2.7.1 API verified via source
- Architecture (callbacks): HIGH — SB3 callback interface verified from source; patterns align with decisions D-01 through D-13
- Self-play pool via set_attr: MEDIUM — SB3 set_attr API is documented; the specific HockeyEnv property bridge is designed but not yet validated
- Pitfalls: HIGH — batch_size divisibility is a hard assertion in SB3; num_timesteps vs n_calls is a well-known footgun; caching pattern for frozen_fn addresses the performance pitfall definitively

**Research date:** 2026-03-29
**Valid until:** 2026-05-01 (SB3 stable releases infrequently; 2.7.1 API is stable)
