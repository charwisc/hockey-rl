---
phase: 1
slug: environment
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-28
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pytest.ini` — Wave 0 installs |
| **Quick run command** | `pytest tests/ -x -q` |
| **Full suite command** | `pytest tests/ -v --tb=short` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -x -q`
- **After every plan wave:** Run `pytest tests/ -v --tb=short`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| ENV-01 | TBD | 1 | ENV-01 | unit | `pytest tests/test_physics.py::test_arena_compiles -x` | ❌ W0 | ⬜ pending |
| ENV-02 | TBD | 1 | ENV-02 | unit | `pytest tests/test_physics.py::test_agents_load -x` | ❌ W0 | ⬜ pending |
| ENV-03a | TBD | 1 | ENV-03 | integration | `pytest tests/test_physics.py::test_puck_stability_1000steps -x` | ❌ W0 | ⬜ pending |
| ENV-03b | TBD | 1 | ENV-03 | unit | `pytest tests/test_physics.py::test_board_bounce_angle -x` | ❌ W0 | ⬜ pending |
| ENV-03c | TBD | 1 | ENV-03 | unit | `pytest tests/test_physics.py::test_puck_friction_decay -x` | ❌ W0 | ⬜ pending |
| ENV-04 | TBD | 2 | ENV-04 | unit | `pytest tests/test_gymnasium.py::test_action_space_spec -x` | ❌ W0 | ⬜ pending |
| ENV-05a | TBD | 2 | ENV-05 | unit | `pytest tests/test_observations.py::test_obs_shape_and_dtype -x` | ❌ W0 | ⬜ pending |
| ENV-05b | TBD | 2 | ENV-05 | integration | `pytest tests/test_observations.py::test_obs_agent_pos_tracks_physics -x` | ❌ W0 | ⬜ pending |
| ENV-05c | TBD | 2 | ENV-05 | integration | `pytest tests/test_observations.py::test_obs_puck_tracks_physics -x` | ❌ W0 | ⬜ pending |
| ENV-06 | TBD | 2 | ENV-06 | unit | `pytest tests/test_observations.py::test_obs_spec_integrity -x` | ❌ W0 | ⬜ pending |
| ENV-07a | TBD | 2 | ENV-07 | unit | `pytest tests/test_rewards.py::test_sparse_goal_reward -x` | ❌ W0 | ⬜ pending |
| ENV-07b | TBD | 2 | ENV-07 | unit | `pytest tests/test_rewards.py::test_puck_toward_goal_fires_with_possession -x` | ❌ W0 | ⬜ pending |
| ENV-07c | TBD | 2 | ENV-07 | unit | `pytest tests/test_rewards.py::test_puck_toward_goal_gated_on_possession -x` | ❌ W0 | ⬜ pending |
| ENV-07d | TBD | 2 | ENV-07 | unit | `pytest tests/test_rewards.py::test_reward_components_in_info -x` | ❌ W0 | ⬜ pending |
| SC-1 | TBD | 2 | ENV-03 | integration | `pytest tests/test_physics.py::test_1000_steps_no_nan -x` | ❌ W0 | ⬜ pending |
| SC-3 | TBD | 2 | ENV-07 | integration | `pytest tests/test_rewards.py::test_reward_independent_extraction -x` | ❌ W0 | ⬜ pending |
| SC-5 | TBD | 2 | ENV-06 | unit | `pytest tests/test_observations.py::test_obs_spec_version_required -x` | ❌ W0 | ⬜ pending |
| GYM | TBD | 2 | ENV-04 | integration | `pytest tests/test_gymnasium.py::test_check_env_passes -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_physics.py` — stubs for ENV-01, ENV-02, ENV-03 physics tests
- [ ] `tests/test_gymnasium.py` — stubs for ENV-04 action space and check_env tests
- [ ] `tests/test_observations.py` — stubs for ENV-05, ENV-06 obs spec tests
- [ ] `tests/test_rewards.py` — stubs for ENV-07 reward component tests
- [ ] `tests/conftest.py` — shared HockeyEnv fixture
- [ ] `pytest.ini` — config with testpaths=tests, minimum verbosity
- [ ] Python venv + `pip install dm-control==1.0.38 gymnasium pytest` — if no venv detected

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Puck rebounds at visually plausible angles during random rollout | ENV-03 | Visual physics plausibility hard to assert numerically | Run `python scripts/visualize_env.py` and observe 30s of random rollout |
| obs_spec.md document is human-readable and complete | ENV-06 | Document quality is subjective | Read `docs/obs_spec.md` — every index must have label, range, and units |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
