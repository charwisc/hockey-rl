# Phase 2: Training - Discussion Log

**Date:** 2026-03-29
**Phase:** 02-training
**Areas discussed:** Self-play pool mechanic, Reward coefficient tuning, Training entry point, VecNormalize + checkpoint bundling

---

## Area: Self-play pool mechanic

**Q: How should the opponent pool be managed across parallel envs?**
- Options: Per-env opponent assignment (recommended) / All-envs rotate together / Self-mirror only (no pool)
- **Selected:** Per-env opponent assignment — each SubprocVecEnv subprocess randomly assigned opponent from pool (50% latest, 50% historical). Full TRAIN-01 compliance.

**Q: How are frozen opponent policies loaded in the subprocess envs?**
- Options: Load from disk by path (recommended) / Serialize policy weights via shared memory / You decide
- **Selected:** Load from disk by path — pool manager writes checkpoint paths; each subprocess env loads SB3 model from path on demand. Avoids subprocess pickling issues.

---

## Area: Reward coefficient tuning

**Q: Is the Phase 1 reward function training-ready, or do we adjust coefficients in Phase 2?**
- Options: Accept Phase 1 reward as-is (recommended) / Tune coefficients now / Add curriculum decay
- **Selected:** Accept Phase 1 reward as-is — all 6 components implemented; train first, tune if goal-rate stays flat after 20M steps.

**Q: The PROJECT.md lists 'Full reward shaping (Phase 2)' — new requirement or just acknowledging Phase 1?**
- Options: Just acknowledging Phase 1 (no new reward work) / There's more reward work planned
- **Selected:** Just acknowledging Phase 1 — all reward components were delivered in Phase 1.

---

## Area: Training entry point

**Q: What should the training script structure look like?**
- Options: Single train.py with argparse (recommended) / Config YAML + launcher
- **Selected:** Single train.py with argparse — flags: --total-steps, --n-envs, --resume. Self-contained; ssh in, clone repo, run one command.

**Q: Should the script support mid-run hyperparameter changes?**
- Options: No — set-and-forget run (recommended) / Yes — signal-based or file-watch LR schedule
- **Selected:** No — set-and-forget. If adjustment needed, kill run and restart from last checkpoint with new flags.

**Q: Where should checkpoints save on RunPod?**
- Options: Configurable via --checkpoint-dir (recommended) / Hardcoded to /workspace/checkpoints/
- **Selected:** Hardcoded to /workspace/checkpoints/ — simpler for a one-time RunPod run.

---

## Area: VecNormalize + checkpoint bundling

**Q: How should VecNormalize stats be saved with each checkpoint?**
- Options: Separate .pkl alongside each .zip (recommended) / Bundled inside the .zip
- **Selected:** Separate .pkl — step_50M.zip + step_50M_vecnorm.pkl. Phase 3 export script reads both independently.

**Q: Should VecNormalize be enabled from step 1, or added later?**
- Options: Enabled from step 1 (recommended) / Start raw, add after warmup
- **Selected:** Enabled from step 1 — mandatory for Phase 3 ONNX export; can't add retroactively.

**Q: What TensorBoard metrics should the custom callback log beyond SB3 defaults?**
- Options: goal_rate + puck_possession from info dict (recommended) / All 6 reward components individually
- **Selected:** goal_rate + puck_possession — satisfies TRAIN-04; already in info dict from HockeyEnv.step().

---

*Discussion log generated: 2026-03-29*
