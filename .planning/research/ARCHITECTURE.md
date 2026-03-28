# Architecture Patterns

**Project:** 2v2 Hockey RL Agent — Portfolio Web Demo
**Researched:** 2026-03-28
**Dimension:** Full pipeline — training → export → browser

---

## System Overview

The system has three physically separate runtime environments that must stay
coordinated through a set of stable data contracts:

```
┌─────────────────────────────────────────────────────────────────┐
│  TRAINING RUNTIME (RunPod, Python)                              │
│                                                                 │
│  MuJoCo XML ──► dm_control Composer Env ──► SB3 VecEnv         │
│                                                  │              │
│                          Opponent Pool Callback ◄┤              │
│                                  │               │              │
│                         PPO Training Loop ◄──────┘              │
│                                  │                              │
│                         Checkpoints (.zip + obs_stats.pkl)      │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
                         ONNX Export Script
                         (offline, run after training)
                                   │
                    ┌──────────────┴──────────────┐
                    │  ARTIFACT STORE (R2 bucket)  │
                    │                              │
                    │  policy_5M.onnx              │
                    │  policy_50M.onnx             │
                    │  ...                         │
                    │  obs_stats.json  (per model) │
                    └──────────────┬───────────────┘
                                   │  fetch on demand
┌──────────────────────────────────▼──────────────────────────────┐
│  BROWSER RUNTIME (static site, Vercel/Netlify)                  │
│                                                                 │
│  Main thread: Three.js render loop (rAF)                        │
│       │   postMessage (obs Float32Array)                        │
│  Inference Worker: onnxruntime-web WASM session                 │
│       │   postMessage (action Float32Array)                     │
│  Main thread: JS physics step + scene update                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Boundaries

| Component | Responsibility | Inputs | Outputs | Talks To |
|-----------|---------------|--------|---------|----------|
| `HockeyArena` (MJCF XML) | Ice rink geometry, board/goal collision bodies, puck, 4 capsule agents with stick geoms | — | MuJoCo XML string | dm_control Physics |
| `HockeyEntity` (dm_control Composer Entity) | Per-agent capsule + stick hitbox; actuators; local observables | Physics | Observable dict | HockeyTask |
| `HockeyTask` (dm_control Composer Task) | Reward computation, termination, episode reset, global observables (puck, scoreboard) | Physics, entities | reward, discount, obs | dm_control Environment |
| `HockeyEnv` (Gymnasium wrapper) | Translates dm_env timestep → Gymnasium step/reset API; flattens dict obs | dm_control timestep | (obs, reward, done, info) | SB3 VecEnv |
| `SelfPlayVecEnv` (VecEnv adapter) | Manages 8–16 parallel `HockeyEnv` copies; applies `SubprocVecEnv` for true parallelism; drives opponent actions from frozen policy | Gymnasium envs | batched (obs, reward, done, info) | SB3 PPO |
| `OpponentPoolCallback` (SB3 BaseCallback) | At N-step intervals: snapshot current policy weights → push to pool; sample from pool → inject into frozen opponent slots | Training step count | Updated frozen policy weights | SelfPlayVecEnv |
| `CheckpointCallback` (SB3) | Save `.zip` + `VecNormalize` stats every 30 min wall-time | Training step count | `.zip`, `obs_stats.pkl` | filesystem |
| `OnnxExporter` (offline script) | Load SB3 checkpoint; wrap actor-only; export to ONNX opset 17; validate numerical parity; write `obs_stats.json` | `.zip`, `obs_stats.pkl` | `policy_NM.onnx`, `obs_stats.json` | Cloudflare R2 |
| `InferenceWorker` (Web Worker) | Load ONNX model from R2; create `InferenceSession`; receive obs, return actions | Fetched `.onnx` bytes | action Float32Array | Main thread |
| `PhysicsStep` (JS) | Lightweight hockey physics mirror: capsule movement, puck momentum, board bounce, goal detection | Current game state + actions | Next game state | Three.js render loop |
| `SceneRenderer` (Three.js) | Ice rink, boards, goal meshes, capsules, puck; broadcast-angle camera; rAF loop | Game state | WebGL draw calls | DOM canvas |
| `UILayer` (Chart.js + DOM) | Scoreboard, period timer, reward curve chart, checkpoint timeline slider | Game state, metadata | DOM updates | Main thread |

---

## Data Flow

### Training Direction (Python)

```
MuJoCo physics step
  → dm_control Physics.data (qpos, qvel, contact forces)
  → HockeyTask.get_observation() → OrderedDict of named arrays
  → HockeyEnv.__flatten_obs() → flat float32 numpy array  [shape: (obs_dim,)]
  → SubprocVecEnv batches N envs → (N, obs_dim) batch
  → VecNormalize.normalize_obs() → zero-mean, unit-variance
  → PPO policy forward → (N, action_dim) actions
  → VecNormalize.unnormalize_actions() (if action normalization enabled)
  → env.step(actions) → next obs, rewards, dones
  → OpponentPoolCallback every 500k steps:
      frozen_policy.set_parameters(current_policy.parameters)
```

**Critical invariant:** The observation vector layout (field ordering, dtype,
shape) is the contract between training and browser. Any change here breaks
the deployed ONNX model.

### Export Direction (offline)

```
SB3 checkpoint (.zip)
  → model.policy.actor   ← only this subgraph is exported
  → OnnxableSB3Policy wrapper (strips value head, handles preprocessing)
  → torch.onnx.export(opset=17, dynamic_axes={'obs': {0: 'batch'}})
  → onnx.checker.check_model()
  → onnxruntime.InferenceSession parity check:
      torch_out = policy(test_obs)
      ort_out   = session.run(None, {'obs': test_obs.numpy()})
      np.testing.assert_allclose(torch_out, ort_out, rtol=1e-4, atol=1e-5)
  → VecNormalize stats → obs_stats.json {mean: [...], var: [...]}
  → Upload policy_NM.onnx + obs_stats.json to R2
```

### Browser Direction

```
User loads page
  → fetch('r2/obs_stats.json')      ← synchronous before inference starts
  → user selects checkpoint
  → fetch('r2/policy_NM.onnx')      ← lazy, on demand
  → InferenceWorker.postMessage({type: 'load', buffer: ArrayBuffer})

Each frame (rAF, ~60 Hz):
  game_state = PhysicsStep.step(game_state, last_actions)
  obs = buildObsVector(game_state)          ← must match training obs layout
  obs_norm = (obs - mean) / sqrt(var + 1e-8)
  InferenceWorker.postMessage({type: 'infer', obs: obs_norm})
    → session.run(['actions'], {obs: Float32Array})
    → postMessage({actions: Float32Array})
  SceneRenderer.update(game_state)
  UILayer.update(game_state, score, timer)
```

**Threading model:** The inference worker runs onnxruntime-web with the WASM
backend (`wasm` execution provider). WebGPU is available but adds complexity
and has narrower browser support; WASM covers all modern browsers and is
sufficient for a small PPO policy network. The worker owns the ONNX session
for its lifetime; the main thread only sends/receives plain Float32Arrays via
`postMessage` with `Transferable` semantics to avoid copies.

---

## Component Detail

### 1. dm_control Environment Design

**Pattern:** Use `dm_control.composer`, not `dm_control.rl.control`. Composer
is the right layer for custom multi-entity environments.

**Class hierarchy:**

```
dm_control.composer.Entity
  └── HockeyPlayer(Entity)
        mjcf_model: capsule body + stick geom + 3 actuators (x, y, rot)
        observables: {body_pos, body_vel, stick_angle}

dm_control.composer.Arena
  └── HockeyArena(Arena)
        mjcf_model: flat ice plane, 4 board walls, 2 goals, friction params

dm_control.composer.Task
  └── HockeyTask(Task)
        entities: [arena, player0, player1, player2, player3, puck]
        initialize_episode(): reset positions, zero velocities
        get_observation(): per-agent view (own pos/vel, teammate, 2 opponents, puck)
        get_reward(): shaped reward per agent
        should_terminate_episode(): goal scored or time limit

dm_control.composer.Environment
  └── wraps HockeyTask, returns dm_env.TimeStep
```

**MJCF XML composition:** Define the arena as a standalone XML file. Agents
and puck are defined as separate XML fragments via `mjcf.from_path()` or
`mjcf.RootElement`. `HockeyTask.initialize_episode_mjcf()` handles attaching
entities to the arena at compile time. This is the recommended Composer
pattern for scenes that don't require runtime XML mutation.

**Observation space design decision:** Use a per-agent egocentric observation
(agent sees the world from its own reference frame: own pos/vel, teammate
pos/vel, both opponent pos/vel, puck pos/vel, stick angle). This means all 4
agents share an identical policy — critical for a single exported ONNX model
to drive all 4 agents in the browser.

### 2. SB3 VecEnv Adapter

**Pattern:** Thin Gymnasium wrapper → `SubprocVecEnv` → `VecNormalize`

```
make_env(agent_idx, frozen_pool) -> gym.Env
  # Returns a HockeyEnv that:
  # - steps all 4 agents each physics step
  # - queries frozen_pool for opponent actions (agents 2,3 if agent_idx in {0,1})
  # - returns only agent_idx's (obs, reward, done)

envs = SubprocVecEnv([make_env(0, pool), make_env(0, pool), ...])  # 8-16 copies
envs = VecNormalize(envs, norm_obs=True, norm_reward=True)
```

**SubprocVecEnv vs DummyVecEnv:** Use `SubprocVecEnv` for actual parallelism;
each subprocess runs its own MuJoCo instance, fully utilizing the RTX 4090's
parallel CPU feed. `DummyVecEnv` runs serially in the same process — correct
for debugging but defeats the purpose on an 8-16 env setup.

**On-policy data leakage constraint (HIGH confidence):** PPO is on-policy.
The wrapper must only expose the active training agent's observation to the
SB3 training loop. Opponent actions are computed inside `env.step()` using
the frozen pool policy, not by the SB3 optimizer. This pattern is confirmed
by both the SB3 docs and community implementations for multi-agent games.

### 3. Self-Play Opponent Pool

**Pattern:** SB3 `BaseCallback` managing a list of frozen policy snapshots.

```
OpponentPoolCallback(BaseCallback):
    pool: List[PolicySnapshot]   # ordered by step count
    snapshot_interval: int = 500_000

    on_step():
        if self.n_calls % snapshot_interval == 0:
            snapshot = copy_policy_weights(self.model.policy)
            self.pool.append(snapshot)
            # Inject into all parallel envs
            for env in self.training_env.envs:
                env.set_opponent(sample_from_pool(self.pool))
```

**Opponent sampling strategy:**
- 70% chance: sample uniformly from pool (diversity)
- 30% chance: use current policy (prevents catastrophic forgetting)
- Pool capped at ~20 snapshots (older ones discarded when full)

**Curriculum stages:**
1. Steps 0–5M: Opponent is random-action policy. Agent learns basic movement
   and puck interaction.
2. Steps 5–20M: Opponent pool seeded with step-5M snapshot. Self-play begins.
3. Steps 20M+: Normal self-play with pool as above.

**Why not league training:** Full league training (AlphaStar-style) requires
multiple independent policies. For a single-GPU 8–10 hour run, a single
policy with opponent pool is the standard and sufficient approach. League
training would require infrastructure the project doesn't have (multi-node or
very long wall-time).

### 4. PyTorch → ONNX Export Pipeline

**What to export:** Only the actor network — `policy.mlp_extractor` +
`policy.action_net`. Do not export the value head (critic). Do not export
VecNormalize inside the ONNX graph.

**Wrapper pattern (confirmed by SB3 docs):**

```python
class OnnxableSB3Policy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Returns mean action (deterministic); no sampling for inference
        return self.policy._predict(obs, deterministic=True)

# Load
model = PPO.load("checkpoint.zip", env=None)
onnx_policy = OnnxableSB3Policy(model.policy)
onnx_policy.eval()

dummy_obs = torch.zeros(1, OBS_DIM, dtype=torch.float32)
torch.onnx.export(
    onnx_policy,
    dummy_obs,
    "policy_NM.onnx",
    opset_version=17,
    input_names=["obs"],
    output_names=["actions"],
    dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
)
```

**VecNormalize handling:** Export the normalization statistics separately as
`obs_stats.json` containing `mean` and `var` arrays. Apply normalization in
the browser before calling the ONNX session. This is simpler and more
portable than baking normalization into the graph.

```python
vec_env = model.get_vec_normalize_env()
stats = {
    "mean": vec_env.obs_rms.mean.tolist(),
    "var": vec_env.obs_rms.var.tolist(),
    "clip_obs": float(vec_env.clip_obs),
}
```

**Numerical parity validation:**

```python
session = onnxruntime.InferenceSession("policy_NM.onnx")
with torch.no_grad():
    torch_out = onnx_policy(dummy_obs).numpy()
ort_out = session.run(["actions"], {"obs": dummy_obs.numpy()})[0]
np.testing.assert_allclose(torch_out, ort_out, rtol=1e-4, atol=1e-5)
# If this fails, check: opset version, unsupported ops, dynamic shapes
```

**Known export pitfalls:**
- **Dynamic axes omitted:** Without `dynamic_axes`, the batch dimension is
  baked as 1 and the ONNX model will refuse any other batch size. Always
  export with `dynamic_axes`.
- **CuDNN LSTM incompatibility:** If using RecurrentPPO (sb3-contrib), the
  CuDNN LSTM module traces poorly. Must swap in the pure-Python LSTM
  implementation before export. Standard PPO (MLP policy) has no such issue.
- **Model in eval mode:** `policy.eval()` is required before export.
  BatchNorm and Dropout behave differently in train mode and will produce
  wrong ONNX graphs.
- **Opset 17 vs 9:** Use opset 17. Opset 9 lacks support for several modern
  ops. ONNX Runtime Web supports opset up to 18 as of 2025.
- **Action clipping not in ONNX:** SB3's `_predict` in deterministic mode
  returns raw tanh-squashed actions. Ensure the browser applies the same
  clipping/scaling the training env applied (usually [-1, 1] → action range).

### 5. Browser-Side Architecture

**Render loop vs inference worker separation:**

```
Main Thread
  ├── Three.js scene (canvas, renderer, camera, meshes)
  ├── requestAnimationFrame loop:
  │     1. Receive inference result from worker (if ready)
  │     2. PhysicsStep.advance(state, actions)
  │     3. renderer.render(scene, camera)
  │     4. UILayer.update(state)
  │     5. Build next obs vector
  │     6. worker.postMessage({type:'infer', obs}, [obs.buffer])
  │         ↑ Transfer ownership to avoid copy
  └── Event handlers (timeline slider, resize)

Inference Worker (worker.js)
  ├── InferenceSession (onnxruntime-web WASM)
  ├── onmessage handler:
  │     type='load':  fetch .onnx from R2, session.create()
  │     type='infer': session.run() → postMessage({actions})
  └── Runs independently; main thread never blocks on inference
```

**Key design choices:**

- Use WASM execution provider, not WebGPU. WASM has near-universal browser
  support; WebGPU is still behind a flag or unavailable on iOS/Firefox as of
  2026. The hockey policy is a small MLP — WASM latency will be well under
  16 ms per frame.
- Transfer `obs.buffer` as a `Transferable` so the Float32Array's backing
  ArrayBuffer moves to the worker without a copy. The main thread must not
  access `obs` after the transfer.
- The worker is created once; the `InferenceSession` is recreated when the
  user switches checkpoints. Destroy the old session explicitly
  (`session.release()`) to free WASM heap.
- The render loop does not wait for inference. If the worker has not returned
  a result, the previous action is reused. This decouples inference latency
  from frame rate.

**JS physics mirror:**

The mirror need not be pixel-perfect with MuJoCo. It must be good enough that
the trained policy's observation inputs look plausible. Required fidelity:
- Correct agent capsule movement from 2D velocity actions
- Puck momentum with linear damping (ice friction)
- Axis-aligned board bounce (reflect velocity component)
- Goal detection (puck crosses goal line AABB)
- Agent-puck collision: simple circle-circle push response

Not required: MuJoCo's contact model, friction cones, soft-body dynamics.
The policy was trained with shaped rewards that make it robust to small
physics differences.

**Implementation:** Plain JavaScript class `HockeyPhysics` operating on a
plain-object game state. No external physics library needed — the geometry is
simple enough. Keep the state object flat (no nested objects) for easy
`structuredClone` / serialization.

### 6. Static Deploy and Model Hosting

**Cloudflare R2 bucket layout:**

```
hockey-models/
  manifest.json          ← index of all available checkpoints
  5M/
    policy.onnx
    obs_stats.json
  20M/
    policy.onnx
    obs_stats.json
  50M/
    policy.onnx
    obs_stats.json
  100M/
    policy.onnx
    obs_stats.json
```

**CORS configuration:** R2 bucket must have a CORS rule allowing `GET` from
the portfolio site's origin. Set bucket to public read; no presigned URLs
needed for public read-only access.

**Lazy loading strategy:**
1. On page load: fetch `manifest.json` only (~1 KB). Parse checkpoint labels
   and sizes. Do not fetch any `.onnx` file yet.
2. Auto-load the final (largest step) checkpoint after a short delay or on
   first user interaction.
3. On timeline slider change: fetch the selected checkpoint `.onnx` if not
   cached. Cache in a `Map<checkpoint_id, ArrayBuffer>` so scrubbing back
   does not re-fetch.
4. Show a loading indicator in the UI while fetching; the current checkpoint
   keeps running.

**Size estimates:** A small MLP PPO policy (2–3 hidden layers, 256 units)
exports to roughly 1–3 MB in ONNX format. Six to eight checkpoints = 8–24 MB
total. Well within R2's free tier (10 GB storage, zero egress fee).

---

## Build Order (Phase Dependencies)

```
Phase 1: Environment Foundation
  ├── MJCF XML (arena geometry) — no deps
  ├── HockeyEntity (per-agent model) — depends on XML spec
  ├── HockeyTask (reward, obs) — depends on entities
  └── HockeyEnv (Gymnasium wrapper) — depends on HockeyTask
        ↓ BLOCKING: Nothing downstream can be built without a working env

Phase 2: Training Pipeline
  ├── SelfPlayVecEnv + SubprocVecEnv — depends on HockeyEnv
  ├── VecNormalize integration — depends on VecEnv
  ├── OpponentPoolCallback — depends on VecEnv + SB3 PPO
  └── CheckpointCallback + TensorBoard — depends on SB3 PPO
        ↓ BLOCKING: Export pipeline requires a trained checkpoint

Phase 3: Export Pipeline
  ├── OnnxableSB3Policy wrapper — depends on a checkpoint
  ├── Parity test harness — depends on ONNX export
  └── obs_stats.json writer — depends on VecNormalize stats
        ↓ BLOCKING: Browser cannot run inference without .onnx + obs_stats.json

Phase 4: Browser Core
  ├── HockeyPhysics (JS) — no deps (can build in parallel with Phase 2-3)
  ├── InferenceWorker — depends on onnxruntime-web + .onnx artifact
  ├── Three.js SceneRenderer — depends on HockeyPhysics
  └── Main render loop integration — depends on all of the above

Phase 5: UI + Deploy
  ├── UILayer (Chart.js, scoreboard, timer) — depends on game state API
  ├── Timeline slider + R2 lazy loading — depends on R2 bucket + manifest
  └── Cloudflare R2 upload + CORS config — depends on exported checkpoints
```

**Parallelism opportunity:** Phase 4's `HockeyPhysics` JS module and the
Three.js scene scaffolding can be built concurrently with Phase 2 training,
using dummy/hardcoded game states. Full integration requires the ONNX
artifacts from Phase 3.

---

## Critical Interfaces (Must Stay Stable)

These three contracts cross the training → export → browser boundary. Changing
any of them after Phase 3 is complete requires re-exporting all checkpoints
and likely rebuilding the JS physics mirror.

### Interface 1: Observation Vector Layout

```
obs[0:2]   = agent_pos (x, y)           # egocentric, world units
obs[2:4]   = agent_vel (x, y)
obs[4:6]   = teammate_pos (x, y)
obs[6:8]   = teammate_vel (x, y)
obs[8:10]  = opponent0_pos (x, y)
obs[10:12] = opponent0_vel (x, y)
obs[12:14] = opponent1_pos (x, y)
obs[14:16] = opponent1_vel (x, y)
obs[16:18] = puck_pos (x, y)
obs[18:20] = puck_vel (x, y)
obs[20]    = stick_angle
obs[21]    = facing_angle
# Total: 22 floats (illustrative; finalized in Phase 1)
```

This layout is defined once in `HockeyTask.get_observation()` and must be
replicated exactly in the browser's `buildObsVector()` function.

### Interface 2: Action Vector Layout

```
action[0] = move_x  ([-1, 1])
action[1] = move_y  ([-1, 1])
action[2] = speed   ([0, 1])
action[3] = stick_swing_angle ([-1, 1], mapped to radians in env)
```

The ONNX model outputs raw (tanh-squashed) actions in this layout. The JS
physics mirror must apply the same de-squash / scaling the Python env used.

### Interface 3: Normalization Statistics Format

```json
{
  "mean": [float, ...],   // length = obs_dim
  "var":  [float, ...],   // length = obs_dim
  "clip_obs": 10.0        // clip value used during training
}
```

Applied in browser as: `obs_norm[i] = clip((obs[i] - mean[i]) / sqrt(var[i] + 1e-8), -clip_obs, clip_obs)`

---

## Scalability Considerations

This project does not need to scale beyond a single portfolio demo. The
relevant concern is inference latency at 60 Hz in the browser.

| Concern | This project | If it were production |
|---------|-------------|----------------------|
| Inference latency | ~5–15 ms WASM on modern laptop; well within 16 ms budget | Switch to WebGPU EP |
| Model size | 1–3 MB per checkpoint; lazy loaded | CDN cache + service worker |
| Concurrent users | Static site; no server; Cloudflare R2 scales automatically | Already handled |
| Training time | Single RTX 4090, 8–10 hrs | Multi-node, larger policy |

---

## Sources

- [dm_control GitHub — google-deepmind/dm_control](https://github.com/google-deepmind/dm_control)
- [Shimmy DM Control Multi-Agent docs](https://shimmy.farama.org/environments/dm_multi/)
- [SB3 Vectorized Environments](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)
- [SB3 Exporting Models](https://stable-baselines3.readthedocs.io/en/master/guide/export.html)
- [PyTorch ONNX export docs](https://docs.pytorch.org/docs/stable/onnx.html)
- [PyTorch torch.onnx.verification](https://docs.pytorch.org/docs/stable/onnx_verification.html)
- [ONNX Runtime Web overview](https://onnxruntime.ai/docs/tutorials/web/)
- [onnxruntime-web npm package](https://www.npmjs.com/package/onnxruntime-web)
- [ONNX Runtime Web Worker inference — issue #12589](https://github.com/microsoft/onnxruntime/issues/12589)
- [HuggingFace Deep RL — Self-Play](https://huggingface.co/learn/deep-rl-course/en/unit7/self-play)
- [Hockey PPO self-play implementation (Meier)](https://github.com/meier-johannes94/Reinforcement-Learning-PPO)
- [Multi-agent game training in single-agent libraries (Scott Vinay, 2025)](http://scottvinay.com/blog/2025/10/08/psai01-4-seat-training/)
- [Cloudflare R2 CORS configuration](https://developers.cloudflare.com/r2/buckets/cors/)
- [Three.js OffscreenCanvas Web Worker pattern](https://evilmartians.com/chronicles/faster-webgl-three-js-3d-graphics-with-offscreencanvas-and-web-workers)
- [37 PPO implementation details (ICLR)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
