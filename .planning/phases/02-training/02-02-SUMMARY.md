---
phase: 02-training
plan: 02
subsystem: training
tags: [stable-baselines3, ppo, subprocvecenv, vecnormalize, callbacklist, self-play, checkpoint, tensorboard]

# Dependency graph
requires:
  - phase: 02-01
    provides: "SelfPlayPoolCallback, WallTimeCheckpointCallback, TensorBoardCustomCallback, HockeyEnv.opponent_path property"
provides:
  - "train.py: complete RunPod training entry point with argparse (--total-steps, --n-envs, --resume)"
  - "requirements-train.txt: pinned pip dependencies for RunPod RTX 4090 environment"
  - "Integration tests verifying train.py argument parsing, env factory, and callback imports"
affects: [03-export, phase-02-training-complete]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "make_env factory pattern: deferred HockeyEnv import inside subprocess factory callable to avoid pickling"
    - "Resume pattern: loads _vecnorm.pkl alongside .zip; warns and wraps fresh VecNormalize if pkl missing"
    - "Final checkpoint save: model.num_timesteps used for naming to encode actual step count"

key-files:
  created:
    - train.py
    - requirements-train.txt
  modified:
    - tests/test_training.py

key-decisions:
  - "PPO n_steps=512, batch_size=256 for n_envs=16: 8192-step rollout buffer, 32 mini-batches per update"
  - "ent_coef=0.01: slight entropy bonus for exploration in sparse hockey reward environment"
  - "CHECKPOINT_DIR hardcoded to /workspace/checkpoints per D-09 (RunPod persistent volume, not a CLI flag)"
  - "reset_num_timesteps=not bool(args.resume): continues step counter on resume, resets on fresh start"

patterns-established:
  - "Train entry point: single train.py script with argparse; set-and-forget for RunPod sessions"
  - "Test isolation: sys.modules.pop('train', None) before import ensures fresh SB3 mock applies"

requirements-completed: [TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04]

# Metrics
duration: 2min
completed: 2026-03-30
---

# Phase 2 Plan 02: Training Entry Point Summary

**SB3 PPO train.py wiring SubprocVecEnv + VecNormalize + CallbackList with argparse flags for RunPod RTX 4090 self-play training**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-30T11:46:15Z
- **Completed:** 2026-03-30T11:48:14Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Created complete `train.py` entry point: argparse (--total-steps, --n-envs, --resume), SubprocVecEnv factory, VecNormalize wrapping, PPO with hyperparameters tuned for RTX 4090, composed CallbackList, and final checkpoint save on exit
- Created `requirements-train.txt` with pinned versions (torch==2.11, stable-baselines3==2.7.1, dm-control==1.0.38) per CLAUDE.md Technology Stack spec
- Added 4 integration tests for train.py (parse_args defaults/custom, make_env factory, imports); full suite 28 passed

## Task Commits

Each task was committed atomically:

1. **Task 1: Create train.py entry point and requirements-train.txt** - `f2e5d4f` (feat)
2. **Task 2: Integration test for train.py wiring** - `6b5f48a` (test)

**Plan metadata:** _(final docs commit follows)_

## Files Created/Modified
- `/home/napier19/hockeyViz/train.py` - Complete training entry point; argparse; SubprocVecEnv + VecNormalize + CallbackList; PPO with n_steps=512, batch_size=256, ent_coef=0.01, device=cuda; resume support loading both .zip and _vecnorm.pkl
- `/home/napier19/hockeyViz/requirements-train.txt` - RunPod pip dependencies: torch==2.11 (cu126), stable-baselines3==2.7.1, tensorboard, dm-control==1.0.38, gymnasium==0.29.1, numpy
- `/home/napier19/hockeyViz/tests/test_training.py` - Added test_train_parse_args_defaults, test_train_parse_args_custom, test_train_make_env, test_train_imports

## Decisions Made
- PPO `n_steps=512`, `batch_size=256` for `n_envs=16`: produces 8192-step rollout buffer; 256 divides 8192 evenly (32 mini-batches per update) — fits RTX 4090 VRAM comfortably for obs_dim=22 MLP policy
- `ent_coef=0.01`: slight entropy bonus to encourage exploration in sparse hockey reward environment
- `reset_num_timesteps=not bool(args.resume)`: preserves TensorBoard step counter continuity on resume

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- `pytest` not on system PATH; resolved by using `/home/napier19/hockeyViz/.venv/bin/pytest` (venv discovered during verification)
- `python` not on PATH; used `python3` for syntax check (standard WSL behavior)

## Known Stubs

None - train.py does not render to UI and has no placeholder values in data paths.

## User Setup Required

None. train.py is designed to run on RunPod — no local configuration required. Copy files and run:
```bash
pip install -r requirements-train.txt
python train.py --total-steps 100000000 --n-envs 16
```

## Next Phase Readiness
- Phase 2 training code is complete. User runs `train.py` on RunPod to produce checkpoints.
- Phase 3 (ONNX export) can begin as soon as a checkpoint exists at `/workspace/checkpoints/step_*.zip` with its paired `_vecnorm.pkl`
- Note: Phase 3 requires a live training checkpoint — Phase 3 plan scaffolding can be written pre-emptively but cannot be fully validated until training produces output

---
*Phase: 02-training*
*Completed: 2026-03-30*

## Self-Check: PASSED
