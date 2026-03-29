---
phase: 01-environment
plan: 03
subsystem: environment
tags: [dm_control, mujoco, mjcf, composer, rl-environment, reward-shaping, observation-space]

# Dependency graph
requires:
  - phase: 01-environment-plan-01
    provides: obs_spec.py — 22-float OBS_SPEC contract with field slices
  - phase: 01-environment-plan-02
    provides: HockeyArena (rink geometry, goal sites), HockeyPuck (cylinder with slide+hinge joints)
provides:
  - HockeyPlayer entity: capsule body + stick hitbox + 3 velocity actuators
  - HockeyTask: 22-float per-agent obs assembly using OBS_SPEC slices + 6-component reward function
  - Full composer.Environment: all 4 entities assemble into valid MuJoCo model with 12 actuators
affects: [02-training, 03-export, 04-browser]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "physics.bind(joint).qpos[0] — bind returns SynchronizingArrayWrapper shape (1,), must index [0] for scalar reads"
    - "Possession gating pattern: r_puck_toward_goal is zero when stick_tip distance > POSSESSION_DIST"
    - "Reward component dict: all float() casts for JSON serialization safety"

key-files:
  created:
    - env/hockey_player.py
    - env/hockey_task.py
  modified: []

key-decisions:
  - "physics.bind(joint).qpos returns shape-(1,) SynchronizingArrayWrapper — all scalar reads use [0] indexing"
  - "obs[20] stick_angle RESERVED as always-zero in v1.0.0 — no independent stick joint; stick_angle action input controls body vrot"
  - "Possession detection uses stick tip approximation (agent_pos + 0.4m in facing direction) — no contact-based detection"

patterns-established:
  - "Pattern 1: All per-agent reward components returned as Python float() (not np.float64) for JSON serialization"
  - "Pattern 2: Teammate/opponent lookup via _get_teammate_idx/_get_opponent_indices helpers (agents 0,1 = team 0; agents 2,3 = team 1)"
  - "Pattern 3: Obs assembly hardcodes indices to match OBS_SPEC exactly — cross-boundary contract"

requirements-completed: [ENV-02, ENV-04, ENV-05, ENV-07]

# Metrics
duration: 3min
completed: 2026-03-29
---

# Phase 01 Plan 03: Player Entities and HockeyTask Summary

**HockeyPlayer capsule agents with stick hitboxes and HockeyTask producing 22-float OBS_SPEC-aligned observations and 6-component possession-gated reward function**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-29T18:29:37Z
- **Completed:** 2026-03-29T18:32:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- HockeyPlayer entity: 75kg capsule (contype=2), stick hitbox (contype=1), slide+hinge joints with rink boundary limits, 3 velocity actuators (vx/vy/vrot)
- HockeyTask: per-agent obs assembly matching OBS_SPEC exactly (22 floats, float32), 6-component reward with possession gating on r_puck_toward_goal
- Full composition verified: arena + 4 players + puck compile to valid MuJoCo model with 12 actuators; composer.Environment resets and steps without error

## Task Commits

1. **Task 1: HockeyPlayer entity** - `5981b19` (feat)
2. **Task 2: HockeyTask** - `795f9f5` (feat)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified

- `env/hockey_player.py` - Capsule agent entity: team-colored body (contype=2), stick hitbox (contype=1), x/y/rot joints with limits, vx/vy/vrot velocity actuators
- `env/hockey_task.py` - Task class: obs assembly (22-float per agent), 6 reward components, face-off reset, goal detection, time limit

## Decisions Made

- `physics.bind(joint).qpos` returns a `SynchronizingArrayWrapper` of shape `(1,)` — all reads use `[0]` indexing. This is a dm_control API detail not in the research notes; discovered during Task 2 verification.
- `obs[20]` (stick_angle) is RESERVED/always-zero in v1.0.0. No independent stick joint exists — the 4-float action's `stick_angle` controls body `vrot`. Documented in both `hockey_task.py` and the obs comment block to prevent downstream consumers from relying on this field.
- Possession detection uses stick tip approximation (agent_pos + 0.4m in facing direction). This is geometrically approximate but sufficient for reward shaping — avoids contact-sensor complexity.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] physics.bind(joint).qpos returns shape-(1,) array, not scalar**
- **Found during:** Task 2 (HockeyTask — obs assembly verification)
- **Issue:** The plan's code samples used `physics.bind(joint).qpos` directly as a scalar. dm_control's `SynchronizingArrayWrapper` returns a shape `(1,)` array. Assigning directly to a float32 obs array element raised `ValueError: setting an array element with a sequence`.
- **Fix:** Added `[0]` indexing to all `qpos`/`qvel` reads in `build_obs_for_agent`, `get_reward_components`, and `after_step`. Assignment (e.g., `physics.bind(x).qpos = 0.0`) works without indexing.
- **Files modified:** env/hockey_task.py
- **Verification:** All 6 plan verification checks pass including shape/dtype checks on obs and type checks on reward values
- **Committed in:** `795f9f5` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Single API-behavior fix, no scope change. All plan acceptance criteria still met.

## Issues Encountered

None beyond the qpos array indexing fix documented above.

## Known Stubs

- `obs[20]` (stick_angle) is always 0.0 — placeholder per v1.0.0 spec. Downstream: ONNX export and JS mirror must treat this as zero; training PPO policy should learn to ignore this dimension. No future plan is currently scheduled to wire a real stick joint here; adding one would require a major obs version bump.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- HockeyPlayer and HockeyTask are ready; the Gymnasium wrapper (Plan 04) can import both directly
- Remaining Plan 04 work: wrap HockeyTask in a Gymnasium-compatible env for SB3 PPO training
- No blockers for Plan 04

---
*Phase: 01-environment*
*Completed: 2026-03-29*
