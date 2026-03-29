---
phase: 2
slug: training
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-29
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (already configured, pytest.ini exists) |
| **Config file** | `pytest.ini` |
| **Quick run command** | `pytest tests/test_training.py -x -m "not slow"` |
| **Full suite command** | `pytest tests/ -x` |
| **Estimated runtime** | ~15 seconds (unit/integration only; full training run is manual) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_training.py -x -m "not slow"`
- **After every plan wave:** Run `pytest tests/ -x`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 2-01-01 | 01 | 0 | TRAIN-01 | unit | `pytest tests/test_training.py::test_self_play_pool_snapshot -x` | ❌ Wave 0 | ⬜ pending |
| 2-01-02 | 01 | 0 | TRAIN-01 | unit | `pytest tests/test_training.py::test_pool_eviction -x` | ❌ Wave 0 | ⬜ pending |
| 2-02-01 | 02 | 1 | TRAIN-02 | integration | `pytest tests/test_training.py::test_subproc_vec_env -x -m "not slow"` | ❌ Wave 0 | ⬜ pending |
| 2-03-01 | 03 | 1 | TRAIN-03 | unit | `pytest tests/test_training.py::test_wall_time_checkpoint -x` | ❌ Wave 0 | ⬜ pending |
| 2-04-01 | 04 | 1 | TRAIN-04 | unit | `pytest tests/test_training.py::test_tb_callback_logging -x` | ❌ Wave 0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_training.py` — test stubs covering TRAIN-01 through TRAIN-04 callback unit tests
- [ ] `training/__init__.py` — package marker
- [ ] `training/self_play_callback.py` — SelfPlayPoolCallback implementation stub
- [ ] `training/checkpoint_callback.py` — WallTimeCheckpointCallback implementation stub
- [ ] `training/tb_callback.py` — TensorBoardCustomCallback implementation stub

*Framework is already installed locally (pytest). SB3 and TensorBoard are RunPod-only — tests must use DummyVecEnv or tiny SubprocVecEnv with mocked components for local test runs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| TensorBoard goal-rate curve rises over 50–100M step run | TRAIN-01 (success criterion 1) | Full training run executes on RunPod RTX 4090; 8–10 hour wall-time; cannot run locally | Launch `python train.py --total-steps 100000000 --n-envs 16` on RunPod, monitor TensorBoard at `localhost:6006` via tunnel |
| At least one checkpoint past 50M steps with .zip + _vecnorm.pkl | TRAIN-02 (success criterion 2) | Requires full training run on RunPod GPU VM | After run, verify `ls /workspace/checkpoints/step_50000000*.zip` and matching `*_vecnorm.pkl` |
| Checkpoints saved at 30-minute wall-time intervals | TRAIN-03 (success criterion 3) | Wall-time behaviour cannot be unit-tested without sleeping 30 minutes | Review checkpoint timestamps after training run; confirm interval ≤ 32 min |
| TensorBoard logs downloadable from RunPod volume | TRAIN-04 (success criterion 4) | Requires live RunPod environment | After training, `scp` or use RunPod file manager to download `tb_logs/` directory |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
