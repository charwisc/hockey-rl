---
phase: 02-training
plan: 01
subsystem: training
tags: [stable-baselines3, ppo, self-play, tensorboard, mujoco, callbacks, gymnasium]

# Dependency graph
requires:
  - phase: 01-environment
    provides: HockeyEnv Gymnasium wrapper, 22-float obs contract, set_attr-compatible env structure

provides:
  - SelfPlayPoolCallback: opponent pool management via set_attr bridge
  - WallTimeCheckpointCallback: wall-clock checkpointing with VecNormalize persistence
  - TensorBoardCustomCallback: hockey/goal_rate and hockey/puck_possession_rate logging
  - HockeyEnv.opponent_path property: SubprocVecEnv.set_attr bridge to frozen_fn closure

affects:
  - 02-02 (train.py wires all three callbacks)
  - 03-export (checkpoint .zip files are the input artifacts)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "sys.modules patching for SB3-free local callback testing"
    - "model-caching closure pattern: check cache before PPO.load to avoid redundant loads"
    - "set_attr bridge: SubprocVecEnv.set_attr('opponent_path', path, indices=[i]) reaches HockeyEnv.opponent_path setter in worker subprocess"
    - "TDD with autouse mock_sb3_modules fixture: enables importing callback classes without SB3 installed"

key-files:
  created:
    - training/__init__.py
    - training/self_play_callback.py
    - training/checkpoint_callback.py
    - training/tb_callback.py
    - tests/test_training.py
  modified:
    - env/hockey_env.py

key-decisions:
  - "SelfPlayPoolCallback fires every 500k timesteps (pool_update_freq), max pool size 20, evicts oldest with os.remove"
  - "WallTimeCheckpointCallback saves step_{N}.zip + step_{N}_vecnorm.pkl — step count in filename for post-hoc analysis"
  - "TensorBoardCustomCallback accumulates per-episode signals across rollout, reports mean on _on_rollout_end"
  - "verbose stored explicitly in __init__ since mock BaseCallback parent does not set it"
  - "opponent_path setter leaves model cache invalidation to _default_opponent_fn (lazy load on next call)"

patterns-established:
  - "Callback verbose attribute: always store self.verbose = verbose after super().__init__() since mocked BaseCallback parent does not"
  - "Pool snapshot guard: os.path.exists check before model.save handles resume collision (pool file already exists)"
  - "VecNormalize save: guarded by 'if vec_norm is not None' — SubprocVecEnv may not have VecNormalize wrapper"

requirements-completed: [TRAIN-01, TRAIN-03, TRAIN-04]

# Metrics
duration: 3min
completed: 2026-03-29
---

# Phase 2 Plan 1: Training Callbacks Summary

**SB3 self-play pool callback, wall-time checkpoint callback, and TensorBoard hockey metrics callback — plus HockeyEnv.opponent_path property bridge for SubprocVecEnv.set_attr**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-03-29T19:18:52Z
- **Completed:** 2026-03-29T19:22:00Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- HockeyEnv gains `opponent_path` property with lazy PPO.load model caching, enabling SubprocVecEnv.set_attr to update opponent policies across worker processes without restarting them
- SelfPlayPoolCallback manages a disk-backed deque of opponent checkpoints, evicts oldest, and randomly reassigns each worker to either the latest policy or a historical snapshot
- WallTimeCheckpointCallback saves `step_{N}.zip` + `step_{N}_vecnorm.pkl` every 30 min wall-clock time — robust to step-count variance across training runs
- TensorBoardCustomCallback accumulates per-episode `hockey/goal_rate` and `hockey/puck_possession_rate` and logs them per rollout
- Full test suite (24/24 non-slow tests) passes including all Phase 1 env tests

## Task Commits

1. **test(02-01):** RED tests for callbacks + opponent_path - `849ecb2`
2. **feat(02-01):** opponent_path property + model-caching closure in HockeyEnv - `86281d2`
3. **feat(02-01):** three SB3 callbacks implemented - `96e7088`

## Files Created/Modified

- `env/hockey_env.py` - Added opponent_path property, setter, _default_opponent_fn with model caching, _opponent_model_cache dict
- `training/__init__.py` - Package marker
- `training/self_play_callback.py` - SelfPlayPoolCallback with deque pool, eviction, set_attr reassignment
- `training/checkpoint_callback.py` - WallTimeCheckpointCallback with wall-clock interval, zip + pkl saves
- `training/tb_callback.py` - TensorBoardCustomCallback logging hockey/goal_rate and hockey/puck_possession_rate
- `tests/test_training.py` - 7 tests (6 non-slow) using sys.modules patching for SB3-free local testing

## Decisions Made

- `verbose` must be stored explicitly in each callback's `__init__` (after `super().__init__()`) because the mock `BaseCallback` parent used in tests does not set this attribute. Real SB3 `BaseCallback` does set it, so this is safe and correct in both environments.
- `_default_opponent_fn` uses a dict-based cache (`{"path": None, "model": None}`) rather than separate instance variables — makes the cache invalidation check a single dict key comparison.
- Pool eviction uses `os.remove` with an existence guard — handles edge cases where files may have been deleted externally (e.g., RunPod volume cleanup).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Mock BaseCallback does not set self.verbose**
- **Found during:** Task 2 (callback implementation, first test run)
- **Issue:** Mock `BaseCallback.__init__` is `lambda self, verbose=0: None` — it does not set `self.verbose`. Callbacks calling `if self.verbose:` raised `AttributeError`.
- **Fix:** Added `self.verbose = verbose` explicitly in `SelfPlayPoolCallback.__init__` and `WallTimeCheckpointCallback.__init__` after `super().__init__()`.
- **Files modified:** training/self_play_callback.py, training/checkpoint_callback.py
- **Verification:** All 6 non-slow tests pass.
- **Committed in:** 96e7088

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug fix)
**Impact on plan:** Minimal — single-line fix per callback, no scope change.

## Issues Encountered

None beyond the verbose attribute bug documented above.

## User Setup Required

None — no external service configuration required for this plan.

## Next Phase Readiness

- All three callbacks are ready to wire into `train.py` (Plan 02-02)
- `train.py` will use: `SelfPlayPoolCallback(pool_dir=...)`, `WallTimeCheckpointCallback(checkpoint_dir=...)`, `TensorBoardCustomCallback()` inside `CallbackList`
- The `opponent_path` setter bridge requires SubprocVecEnv (not DummyVecEnv) — SubprocVecEnv spawns real subprocess workers, so `set_attr` goes through IPC to reach the setter
- SB3 is not installed locally — these callbacks are designed for execution on RunPod where SB3 2.7.1 is available

## Self-Check: PASSED

All created files exist on disk. All task commits verified in git history.

---
*Phase: 02-training*
*Completed: 2026-03-29*
