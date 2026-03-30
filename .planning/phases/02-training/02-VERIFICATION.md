---
phase: 02-training
verified: 2026-03-30T00:00:00Z
status: passed
score: 9/9 must-haves verified
re_verification: false
---

# Phase 2: Training Verification Report

**Phase Goal:** A complete self-play PPO training run on RunPod RTX 4090 producing 50–100M step checkpoints — this is a human-executed step on a remote GPU VM, not an automated CI step
**Verified:** 2026-03-30
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Framing Note

Phase 2 has a split nature: the *goal* is the RunPod training run (un-automatable, human-executed), but the *verifiable deliverable* is the training pipeline code that enables that run. The success criteria in ROADMAP.md are all properties of a live training session (rising goal-rate curve, checkpoints past 50M steps, TensorBoard logs). These can only be confirmed by the human running the job on RunPod.

This verification therefore assesses: (a) all code infrastructure required to execute the run exists and is correct, (b) the pipeline is wired end-to-end, (c) all must-have behaviors are implemented and tested, and (d) the four training requirements are fully satisfied by the code. The actual execution outcome is routed to human verification.

---

## Goal Achievement

### Observable Truths (from PLAN frontmatter must_haves, both plans)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SelfPlayPoolCallback snapshots policy to pool directory and reassigns opponents via set_attr | VERIFIED | `_snapshot_and_reassign` in self_play_callback.py:55–85; `set_attr("opponent_path", chosen, indices=[i])` at line 82; `test_self_play_pool_snapshot` passes |
| 2 | Pool evicts oldest checkpoint when exceeding max_pool_size=20 | VERIFIED | `while len(self._pool) > self.max_pool_size: old = self._pool.popleft(); os.remove(old)` at lines 69–72; `test_pool_eviction` passes |
| 3 | WallTimeCheckpointCallback saves .zip + _vecnorm.pkl based on wall-clock elapsed time | VERIFIED | `_save()` at checkpoint_callback.py:52–65; `time.time()` delta check at line 47; `test_wall_time_checkpoint` passes confirming both file names |
| 4 | TensorBoardCustomCallback logs hockey/goal_rate and hockey/puck_possession_rate on rollout end | VERIFIED | `_on_rollout_end` at tb_callback.py:48–60; `self.logger.record("hockey/goal_rate", ...)` at line 52; `test_tb_callback_logging` asserts exact key names and value 0.5 |
| 5 | HockeyEnv exposes opponent_path property that updates the frozen_fn closure's cached model | VERIFIED | Property getter/setter at hockey_env.py:143–152; `_default_opponent_fn` with cache at lines 154–171; `test_opponent_path_property` and `test_opponent_path_caching` both pass |
| 6 | train.py is a runnable entry point with argparse flags --total-steps, --n-envs, --resume | VERIFIED | train.py lines 43–51; all three flags present with correct defaults (100M, 16, None); `test_train_parse_args_defaults` and `test_train_parse_args_custom` pass |
| 7 | SubprocVecEnv wraps N HockeyEnv instances, VecNormalize wraps SubprocVecEnv, all three callbacks composed via CallbackList | VERIFIED | train.py lines 63, 69/81, 105–118; `from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize` at lines 19–20; `test_train_make_env` confirms factory creates HockeyEnv with opponent_path |
| 8 | --resume flag loads both .zip and _vecnorm.pkl and continues training | VERIFIED | train.py lines 65–78; `vecnorm_path = args.resume.replace(".zip", "_vecnorm.pkl")`; `VecNormalize.load(vecnorm_path, env)`; `reset_num_timesteps=not bool(args.resume)` at line 128 |
| 9 | requirements-train.txt lists RunPod pip dependencies with pinned torch version | VERIFIED | requirements-train.txt has `torch==2.11 --extra-index-url https://download.pytorch.org/whl/cu126`, `stable-baselines3==2.7.1`, `dm-control==1.0.38`, `gymnasium==0.29.1`, `tensorboard>=2.0` |

**Score:** 9/9 truths verified

---

### Required Artifacts

| Artifact | Expected | Lines | Status | Details |
|----------|----------|-------|--------|---------|
| `training/__init__.py` | Package marker | 1 | VERIFIED | Non-empty docstring present |
| `training/self_play_callback.py` | SelfPlayPoolCallback class | 85 | VERIFIED | Full implementation; `SelfPlayPoolCallback`, `deque`, `pool_update_freq`, `max_pool_size`, `set_attr("opponent_path"` all present |
| `training/checkpoint_callback.py` | WallTimeCheckpointCallback class | 65 | VERIFIED | `WallTimeCheckpointCallback`, `time.time()`, `step_{step}.zip`, `step_{step}_vecnorm.pkl`, `get_vec_normalize_env` all present |
| `training/tb_callback.py` | TensorBoardCustomCallback class | 60 | VERIFIED | `TensorBoardCustomCallback`, `hockey/goal_rate`, `hockey/puck_possession_rate`, `logger.record` all present |
| `tests/test_training.py` | Unit tests for all callbacks | 303 | VERIFIED | All 7 required test functions present including `@pytest.mark.slow`; 10/10 non-slow tests pass |
| `env/hockey_env.py` | opponent_path property for set_attr bridge | — | VERIFIED | Property getter, setter, `_opponent_model_cache`, `_default_opponent_fn` all confirmed |
| `train.py` | Training entry point | 144 | VERIFIED | Valid Python (ast.parse OK); argparse, SubprocVecEnv, VecNormalize, CallbackList, n_steps=512, batch_size=256, ent_coef=0.01, device="cuda", `if __name__ == "__main__"` all present |
| `requirements-train.txt` | RunPod pip dependencies | 8 | VERIFIED | stable-baselines3==2.7.1, torch==2.11 (cu126), dm-control==1.0.38, tensorboard present |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `training/self_play_callback.py` | `env/hockey_env.py` | `set_attr("opponent_path", path, indices=[i])` | WIRED | Line 82 of self_play_callback.py matches pattern; HockeyEnv.opponent_path setter confirmed at hockey_env.py:148 |
| `training/checkpoint_callback.py` | `model.get_vec_normalize_env()` | `VecNormalize.save()` | WIRED | Lines 60–62 of checkpoint_callback.py; guarded with `if vec_norm is not None` |
| `training/tb_callback.py` | `self.logger.record` | SB3 TensorBoard logger | WIRED | Lines 51–57 of tb_callback.py; both `hockey/goal_rate` and `hockey/puck_possession_rate` keys confirmed |
| `train.py` | `training/self_play_callback.py` | `from training.self_play_callback import SelfPlayPoolCallback` | WIRED | train.py line 22; instantiated in CallbackList at line 106 |
| `train.py` | `training/checkpoint_callback.py` | `from training.checkpoint_callback import WallTimeCheckpointCallback` | WIRED | train.py line 23; instantiated in CallbackList at line 112 |
| `train.py` | `training/tb_callback.py` | `from training.tb_callback import TensorBoardCustomCallback` | WIRED | train.py line 24; instantiated in CallbackList at line 117 |
| `train.py` | `env/hockey_env.py` | `SubprocVecEnv([make_env(...)])` | WIRED | train.py line 63; make_env factory imports HockeyEnv inside subprocess callable (deferred import pattern for pickling safety) |
| `train.py` | `stable_baselines3` | `PPO, VecNormalize, CallbackList` | WIRED | train.py lines 18–20; all three used in main() |

---

### Data-Flow Trace (Level 4)

Not applicable. Phase 2 produces training infrastructure code (callbacks, entry point) — not UI components or data-rendering artifacts. No dynamic data flows to a browser or user-visible render surface in this phase.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| train.py is syntactically valid Python | `python3 -c "import ast; ast.parse(open('train.py').read())"` | Exited 0, "train.py syntax OK" | PASS |
| All non-slow training tests pass | `.venv/bin/pytest tests/test_training.py -x -m "not slow" -v` | 10/10 passed, 0 failed | PASS |
| Full non-slow test suite passes (regression) | `.venv/bin/pytest tests/ -x -m "not slow" -q` | 28/28 passed, 1 deselected | PASS |
| Callback modules use correct logic patterns | grep checks on pool eviction, wall-time save, logger.record | All patterns confirmed present | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| TRAIN-01 | 02-01 + 02-02 | Self-play uses historical opponent pool of ~20 checkpoints; pool updated every ~500k training steps with 50% latest / 50% historical mix | SATISFIED | `SelfPlayPoolCallback`: `pool_update_freq=500_000`, `max_pool_size=20`, 50/50 sampling logic at self_play_callback.py:78–81; wired into train.py CallbackList |
| TRAIN-02 | 02-02 | SubprocVecEnv configured for 8–16 parallel environments on RunPod RTX 4090 | SATISFIED | `SubprocVecEnv([make_env(agent_idx=0) for _ in range(args.n_envs)])` with `--n-envs` defaulting to 16 (range 8–16 documented in argparse help); `test_train_make_env` confirms factory creates correct env |
| TRAIN-03 | 02-01 + 02-02 | Checkpoints saved every 30 minutes of wall-time, labelled by step count; target 50–100M steps in 8–10 hour run | SATISFIED | `WallTimeCheckpointCallback(interval_minutes=30.0)` in train.py; saves `step_{N}.zip` + `step_{N}_vecnorm.pkl`; `--total-steps` defaults to 100M |
| TRAIN-04 | 02-01 + 02-02 | TensorBoard logging records episode reward, goal rate, and puck possession stats per checkpoint | SATISFIED | `TensorBoardCustomCallback` records `hockey/goal_rate` and `hockey/puck_possession_rate`; SB3 default PPO logging records episode reward automatically; `tensorboard_log=TB_LOG_DIR` passed to PPO constructor |

**Orphaned requirements check:** REQUIREMENTS.md maps TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04 to Phase 2. Both plans claim all four. No orphaned requirements.

---

### Anti-Patterns Found

No anti-patterns detected.

Scanned files: `training/self_play_callback.py`, `training/checkpoint_callback.py`, `training/tb_callback.py`, `train.py`, `tests/test_training.py`

Patterns checked: TODO/FIXME/HACK/PLACEHOLDER comments, empty return values, hardcoded empty data, console.log-only handlers. None found.

---

### Human Verification Required

Phase 2's ROADMAP success criteria are all properties of a live training run on RunPod, which cannot be verified statically. The code infrastructure is fully verified above. The following must be confirmed after executing the training run:

#### 1. TensorBoard goal-rate curve rises over training

**Test:** After starting `python train.py --total-steps 100000000 --n-envs 16` on RunPod, monitor TensorBoard at `/workspace/tb_logs/`. Check the `hockey/goal_rate` plot over time.
**Expected:** The `hockey/goal_rate` curve shows a statistically significant upward trend over the run — agents learn to score, not merely optimize shaped reward components.
**Why human:** Cannot verify a rising learning curve without executing the training run and observing logged data across millions of steps.

#### 2. At least one checkpoint exists past 50M steps

**Test:** After the run completes or reaches ~50M steps, check `/workspace/checkpoints/` on the RunPod volume.
**Expected:** At least one file named `step_50000000.zip` or higher (e.g. `step_60000000.zip`) exists alongside its paired `_vecnorm.pkl`.
**Why human:** Requires executing on RunPod with a real GPU; cannot be simulated locally.

#### 3. Checkpoints saved at 30-minute wall-time intervals

**Test:** During the training run, observe the RunPod terminal output. `WallTimeCheckpointCallback` prints `[Checkpoint] step=N saved to ...` on each save.
**Expected:** Checkpoint save messages appear approximately every 30 minutes of wall-clock time throughout the run.
**Why human:** Requires a live training session to validate wall-clock behavior.

#### 4. TensorBoard logs downloadable and contain all three metrics

**Test:** After any checkpoint interval, download the TensorBoard event file from `/workspace/tb_logs/` and open it with `tensorboard --logdir .` locally.
**Expected:** Scalars view shows `rollout/ep_rew_mean` (SB3 default), `hockey/goal_rate`, and `hockey/puck_possession_rate` — all three advancing per the TRAIN-04 requirement.
**Why human:** Requires a live training session and manual inspection of downloaded logs.

---

### Gaps Summary

No gaps. All code infrastructure is present, substantive, wired, and tested. The four unverifiable items above are inherent to the phase's definition as a human-executed RunPod step, not deficiencies in the deliverable.

---

## Commit Evidence

All documented commits exist in git history:

| Commit | Description |
|--------|-------------|
| `849ecb2` | test(02-01): RED tests for callbacks + opponent_path (254 lines added) |
| `86281d2` | feat(02-01): opponent_path property + model-caching closure in HockeyEnv |
| `96e7088` | feat(02-01): three SB3 callbacks implemented (211 lines added across 4 files) |
| `f2e5d4f` | feat(02-02): train.py entry point and requirements-train.txt (144 lines) |
| `6b5f48a` | test(02-02): integration tests for train.py wiring |
| `d9ef8fa` | docs(02-02): SUMMARY, STATE, ROADMAP updated |

---

_Verified: 2026-03-30_
_Verifier: Claude (gsd-verifier)_
