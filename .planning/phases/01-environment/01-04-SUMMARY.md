---
phase: 01-environment
plan: 04
subsystem: environment
tags: [gymnasium, dm_control, hockey, rl-env, onnx-contract, obs-spec]

requires:
  - phase: 01-environment/01-03
    provides: HockeyTask with build_obs_for_agent, get_reward_components, HockeyPlayer, HockeyPuck, HockeyArena

provides:
  - HockeyEnv(gym.Env): single-agent Gymnasium wrapper over dm_control 2v2 hockey task
  - docs/obs_spec.md: human-readable cross-boundary contract for 22-float observation vector
  - Full test suite: 18 tests passing green, all ENV requirements validated

affects:
  - Phase 02 (training): SB3 PPO consumes HockeyEnv via SubprocVecEnv
  - Phase 03 (export): obs_spec.py + docs/obs_spec.md are the ONNX export contract
  - Phase 04 (browser): JS physics mirror must match obs_spec.md index layout exactly

tech-stack:
  added: [gymnasium]
  patterns:
    - Hand-written gym.Env wrapper (not Shimmy) for explicit obs vector contract control
    - 4-float action [move_x, move_y, speed, stick_angle] mapped to 3 actuators per player via speed-scaling
    - frozen_opponent_fn callback pattern for self-play opponent swapping
    - NaN guard in step() resets episode on physics explosion
    - _entities_attached guard prevents double-attach on repeated reset()

key-files:
  created:
    - env/hockey_env.py
    - docs/obs_spec.md
  modified:
    - env/hockey_task.py
    - tests/conftest.py
    - tests/test_physics.py
    - tests/test_gymnasium.py
    - tests/test_observations.py
    - tests/test_rewards.py

key-decisions:
  - "Hand-written gym.Env over Shimmy: explicit control over 22-float obs layout which is the ONNX/JS cross-boundary contract"
  - "frozen_opponent_fn callback: enables opponent pool swapping in Phase 2 self-play without env redesign"
  - "_entities_attached guard in HockeyTask.initialize_episode_mjcf: dm_control calls this hook on every reset, not just first compile"

patterns-established:
  - "Pattern: env.unwrapped._dm_env.physics for physics access in tests"
  - "Pattern: env.unwrapped._task for task access in tests (possession/reward inspection)"
  - "Pattern: reset before physics manipulation in test_rewards.py (initialize_episode sets face-off positions)"

requirements-completed: [ENV-01, ENV-02, ENV-03, ENV-04, ENV-05, ENV-06, ENV-07]

duration: 4min
completed: 2026-03-29
---

# Phase 01 Plan 04: Gymnasium Wrapper and Test Suite Summary

**HockeyEnv(gym.Env) wrapping dm_control 2v2 hockey task — 22-float obs, 4-float action, 18 tests green, check_env passes, obs_spec.md cross-boundary contract written**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-03-29T18:35:41Z
- **Completed:** 2026-03-29T18:39:21Z
- **Tasks:** 3
- **Files modified:** 8

## Accomplishments

- HockeyEnv presents clean Gymnasium API: reset returns (obs, info), step returns 5-tuple with 6 reward components in info
- gymnasium.utils.env_checker.check_env() passes with zero errors
- 18 tests pass green with no xfail remaining — all ENV-01 through ENV-07 requirements validated
- test_sparse_goal_reward uses mandatory physics manipulation and asserts r_goal == 10.0
- docs/obs_spec.md documents all 22 obs indices with obs[20] RESERVED/always-zero status clearly documented for ONNX/JS consumers

## Task Commits

Each task was committed atomically:

1. **Task 1: Create HockeyEnv Gymnasium wrapper** - `da72fca` (feat)
2. **Task 2: Create obs_spec.md and update conftest.py** - `c3ca9d4` (feat)
3. **Task 3: Convert all test stubs to real implementations** - `444e412` (feat + Rule 1 bug fix)

**Plan metadata:** (this commit)

## Files Created/Modified

- `env/hockey_env.py` - HockeyEnv(gym.Env) wrapping dm_control composer.Environment; 4-float action, 22-float obs, NaN guard, frozen_opponent_fn
- `docs/obs_spec.md` - Human-readable rendering of 22-float obs spec; index table, action table, obs[20] RESERVED note, modification protocol
- `env/hockey_task.py` - Added `_entities_attached` guard to `initialize_episode_mjcf` (bug fix)
- `tests/conftest.py` - Direct import of HockeyEnv (not deferred)
- `tests/test_physics.py` - Replaced 2 xfail stubs with real test_agents_load and test_1000_steps_no_nan
- `tests/test_gymnasium.py` - Replaced 2 xfail stubs with real check_env and action space tests
- `tests/test_observations.py` - Replaced 3 xfail stubs with real obs tracking tests
- `tests/test_rewards.py` - Replaced 5 xfail stubs with real reward isolation tests

## Decisions Made

- Hand-written gym.Env over Shimmy: gives explicit control over the 22-float observation vector layout, which is the cross-boundary contract for ONNX export and JS physics mirror. Shimmy's generic wrapper would hide this.
- frozen_opponent_fn callback pattern: allows Phase 2 self-play to swap opponent policies without changing the env interface. Defaults to random actions.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed double-attach crash in HockeyTask.initialize_episode_mjcf**
- **Found during:** Task 3 (test_check_env_passes)
- **Issue:** gymnasium check_env calls reset() multiple times. dm_control calls `initialize_episode_mjcf` on every reset, not just first compile. The second call tried to re-attach already-attached player/puck entities, raising `ValueError: The model specified is already attached elsewhere`.
- **Fix:** Added `_entities_attached` boolean flag to HockeyTask. Guard at start of `initialize_episode_mjcf` returns early if already attached.
- **Files modified:** env/hockey_task.py
- **Verification:** test_check_env_passes passes; all 18 tests pass
- **Committed in:** 444e412 (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Necessary correctness fix. dm_control's hook lifecycle was not documented in the plan's interface spec. No scope creep.

## Issues Encountered

- `test_agents_load` initially failed with nu=0 because the env fixture doesn't call reset() before the test accesses physics. Physics is recompiled on first reset. Fixed by adding `env.reset()` at the start of the test.

## Known Stubs

None — all observation indices are wired to real physics values. obs[20] (stick_angle) is always-zero intentionally per v1.0.0 design (no independent stick joint); this is documented in obs_spec.md and obs_spec.py.

## Next Phase Readiness

- HockeyEnv is ready for SB3 SubprocVecEnv wrapping in Phase 2
- frozen_opponent_fn is the hook for self-play opponent pool swapping
- Phase 2 needs: VecNormalize wrapper, opponent pool FIFO queue, checkpoint callbacks
- No blockers for Phase 2 start

---
*Phase: 01-environment*
*Completed: 2026-03-29*
