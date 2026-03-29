# Phase 2: Training - Context

**Gathered:** 2026-03-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Self-play PPO training infrastructure — Python scripts that a human runs on a RunPod RTX 4090 VM, producing 50–100M step checkpoints (SB3 .zip + VecNormalize .pkl pairs) ready for ONNX export. This phase delivers runnable training code, not a deployed service. The output is artifacts on disk: checkpoints, TensorBoard logs, and a working train.py.

Reward engineering is NOT part of this phase — the Phase 1 reward function is accepted as-is.

</domain>

<decisions>
## Implementation Decisions

### Self-play pool mechanic
- **D-01:** Per-env opponent assignment — each SubprocVecEnv subprocess is randomly assigned an opponent from the pool (50% latest checkpoint, 50% historical). Not all-envs-same-snapshot.
- **D-02:** A SB3 callback fires every ~500k steps, snapshots the current policy to the checkpoint pool, then re-assigns opponents to each env.
- **D-03:** Pool size target: ~20 checkpoints (older snapshots evicted when pool exceeds max size).
- **D-04:** Frozen opponent policies are loaded from disk by path — each subprocess env receives a checkpoint path string and loads the SB3 model on demand. No pickling of model weights across subprocess boundary.

### Reward function
- **D-05:** Phase 1 reward function is accepted as-is — no coefficient changes. All 6 components (r_goal, r_puck_toward_goal, r_possession, r_positioning, r_clustering, r_step_penalty) are training-ready.
- **D-06:** PROJECT.md note "Full reward shaping (Phase 2)" is acknowledging Phase 1 work, not a new Phase 2 requirement.

### Training entry point
- **D-07:** Single `train.py` script with argparse. Flags: `--total-steps`, `--n-envs`, `--resume` (path to checkpoint .zip to resume from).
- **D-08:** Set-and-forget run — no mid-run hyperparameter changes. If adjustment is needed, kill and restart from last checkpoint with new flags.
- **D-09:** Checkpoint directory hardcoded to `/workspace/checkpoints/` (RunPod persistent volume path). Not configurable via flag.
- **D-10:** Checkpoint naming convention: `step_{N}.zip` + `step_{N}_vecnorm.pkl` (e.g., `step_50000000.zip`, `step_50000000_vecnorm.pkl`).

### VecNormalize + checkpoints
- **D-11:** VecNormalize wraps SubprocVecEnv from step 1. Stats accumulate over the entire training run. Cannot be added retroactively.
- **D-12:** VecNormalize stats saved as a separate `.pkl` file alongside each SB3 `.zip` checkpoint — not bundled inside the .zip. Phase 3 export script reads both files independently.
- **D-13:** Custom TensorBoard callback logs `goal_rate` and `puck_possession` averaged over episode infos per rollout. These are already in the `info` dict from `HockeyEnv.step()` (via `score` key and `r_possession` component). SB3 default episode reward logging is preserved.

### Claude's Discretion
- PPO hyperparameters (learning rate, clip range, n_steps, batch_size, n_epochs, gamma, gae_lambda) — use SB3 defaults as starting point, tune for RTX 4090 memory
- Number of parallel envs between 8–16 (TRAIN-02 range) — choose based on env step throughput benchmarking
- TensorBoard callback implementation details (averaging window, logging frequency)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Environment interface (what train.py wraps)
- `env/hockey_env.py` — HockeyEnv Gymnasium wrapper; `frozen_opponent_fn` constructor param is the self-play injection point
- `env/hockey_task.py` — Reward components; `info` dict keys (`score`, `r_possession`, etc.) used by TensorBoard callback
- `env/obs_spec.py` — OBS_DIM=22, OBS_SPEC layout; training must not change this

### Observation contract
- `docs/obs_spec.md` — Canonical obs vector spec; all 22 dimensions; modification protocol; MUST NOT change during Phase 2

### Project requirements for this phase
- `.planning/REQUIREMENTS.md` §Training — TRAIN-01 through TRAIN-04; every requirement must map to a plan task

### Stack versions
- `CLAUDE.md` §Technology Stack — SB3 2.7.1, PyTorch 2.11, Python 3.11, CUDA 12.x; use exact versions

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `HockeyEnv(agent_idx, frozen_opponent_fn, time_limit)` — already Gymnasium-compatible with SubprocVecEnv; the `frozen_opponent_fn` is the exact hook for self-play opponent injection
- `HockeyTask.get_reward_components()` — returns dict with all 6 reward keys; `info` dict from `step()` already exposes these + `score`

### Established Patterns
- Gymnasium API (obs, reward, terminated, truncated, info) — SB3 SubprocVecEnv expects this exact signature
- `frozen_opponent_fn(obs) -> action` callable interface — keep this signature for loaded checkpoint policies

### Integration Points
- `SubprocVecEnv([lambda: HockeyEnv(agent_idx=0, frozen_opponent_fn=pool_fn_i) for i in range(N)])` — each lambda captures a different pool function
- VecNormalize wraps SubprocVecEnv output; `VecNormalize.save(path)` saves stats .pkl
- SB3 `BaseCallback.on_rollout_end()` is the hook for custom TensorBoard logging from `self.training_env.env_method("get_episode_infos")` or similar

</code_context>

<specifics>
## Specific Ideas

- Checkpoint every 30 minutes of wall-time (not every N steps) — use `time.time()` in callback to track wall-time and trigger checkpoint when interval elapsed
- Target 50–100M total training steps in an 8–10 hour RunPod session
- TensorBoard logs must be downloadable from RunPod volume (standard SB3 TensorBoard output to `/workspace/tb_logs/`)

</specifics>

<deferred>
## Deferred Ideas

- Curriculum reward shaping (decay shaped rewards over training) — considered but out of scope; revisit if goal-rate stays flat
- Config YAML for hyperparameters — out of scope; argparse is sufficient for a single showcase run
- Per-agent reward component logging in TensorBoard — out of scope; goal_rate + puck_possession are sufficient per TRAIN-04

</deferred>

---

*Phase: 02-training*
*Context gathered: 2026-03-29*
