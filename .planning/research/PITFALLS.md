# Domain Pitfalls: RL Hockey Portfolio Web Demo

**Domain:** MuJoCo/dm_control RL training + ONNX browser inference demo
**Researched:** 2026-03-28

---

## Critical Pitfalls

Mistakes that cause rewrites, wasted GPU hours, or a broken demo.

---

### Pitfall 1: JS Physics Mirror Diverges from Python Environment

**What goes wrong:** The Three.js / JavaScript physics simulation drifts from the Python dm_control environment. The exported ONNX policy was trained on Python observations — positions, velocities, puck state — computed by MuJoCo's integrator at a specific timestep with its specific floating-point ordering. The JS mirror computes observations differently (different integrator, different dt, different friction approximation, or different coordinate conventions), so the policy receives inputs outside its training distribution and produces nonsensical or erratic actions.

**Why it happens:** MuJoCo's exact behavior changes between versions due to floating-point operation ordering. A hand-rolled JS physics loop using Euler integration will produce different velocity values than MuJoCo's semi-implicit Euler integrator even at the same timestep. Rounding in float32 vs float64 compounds per-step. Coordinate frame conventions (Y-up vs Z-up, radians vs degrees, local vs world space) are easy to flip.

**Consequences:** Policy behavior ranges from "slightly sloppy" to completely broken — agents spinning in place, not chasing the puck, or freezing. This is the single highest-risk item in the entire project.

**Prevention:**
1. Define the canonical observation vector format (field names, units, coordinate frame, normalization) as a shared specification document before writing either environment. Do this in Phase 1.
2. In the ONNX export phase, log 50–100 observation vectors from Python rollouts. After implementing the JS mirror, replay the same initial states and diff the observation vectors numerically. Target max absolute deviation < 0.05 per element.
3. Prioritize behavioral correctness over physical realism in the JS mirror. If ice friction tuning improves realism but shifts observations, drop the realism.
4. Use float32 explicitly in both environments — Python environment should convert observations to float32 before feeding the policy, and JS should use Float32Array.
5. Accept that the JS mirror does not need to be a perfect physics sim — it only needs to keep the policy's inputs in-distribution long enough to produce a visually sensible 60-second demo.

**Detection (warning signs):**
- Policy takes the same action on every step regardless of game state
- Agents converge to a corner and stop moving
- Numerical parity test fails on any observation element by more than ~5%
- Policy actions in JS have much lower variance than in Python rollouts

**Phase:** Address in Phase 3 (ONNX export) with specification, and Phase 4 (browser) with numerical parity tests. Do not skip the parity test.

---

### Pitfall 2: Observation Normalization Not Exported with the Model

**What goes wrong:** SB3's `VecNormalize` wrapper maintains running mean and variance statistics for observations and normalizes them before feeding to the policy network. When you export the policy to ONNX via `torch.onnx.export`, you export only the neural network weights — not the normalization statistics. In browser inference, you pass raw observations directly to the ONNX model, and the policy receives values on a completely different scale than it was trained on. The network appears to work (no errors) but outputs garbage actions.

**Why it happens:** `VecNormalize` is a wrapper around the environment, not a layer inside the PyTorch model. `torch.onnx.export` traces only the `policy.actor` forward pass. The mean/variance stats live in a separate `.pkl` file that must be saved separately and applied manually at inference time.

**Consequences:** Subtle and hard to debug — the model runs without error but the policy is broken. You may spend hours thinking the JS physics mirror is wrong before tracing it to normalization.

**Prevention:**
1. Save the `VecNormalize` statistics file alongside every checkpoint: `vec_env.save("vecnormalize_stats.pkl")`.
2. During ONNX export, bake the normalization into the export: wrap `policy.actor` in a thin `nn.Module` that applies `(obs - mean) / (std + eps)` as the first operation, then export that wrapper. This is the recommended approach.
3. Validate: run the same observation through (a) `VecNormalize` + Python policy and (b) ONNX model with baked normalization. Assert outputs match within 1e-4.
4. In JS, apply the same normalization constants (mean, std) before feeding observations to the ONNX session — these can be stored as a JSON sidecar file.

**Detection (warning signs):**
- Python rollout reward is high but ONNX inference reward is near zero
- Policy outputs cluster at extreme values (saturated logits from out-of-range inputs)
- Numerical parity test passes for raw actions but policy reward diverges

**Phase:** Address in Phase 3 (ONNX export). This is the most common SB3-to-ONNX mistake.

---

### Pitfall 3: Self-Play Policy Collapse / Strategy Cycling

**What goes wrong:** With naive self-play (train against only the current policy), the agent can collapse into a degenerate strategy: e.g., both teams learn to defend perfectly, goal rate drops to zero, and no useful behavior emerges. Alternatively, the agent discovers a strategy that beats its current self, then the opponent catches up, then the strategy is abandoned — a cyclic pattern with no net skill growth. This is endemic to non-transitive game dynamics.

**Why it happens:** The agent is always training against a non-stationary opponent (itself changing). If the opponent pool is too small (just "current self"), the training signal is highly non-stationary and oscillates. Strategy cycling happens because in adversarial games, no single strategy dominates — rock beats scissors beats paper beats rock.

**Consequences:** 50–100M steps of GPU compute wasted on an agent that just pushes into corners or stands still.

**Prevention:**
1. Use a historical opponent pool: every ~500k steps, snapshot the current policy into the pool. At episode start, sample uniformly from the pool (including current self). A larger pool prevents cycling.
2. Set `play_against_latest_model_ratio` to ~0.5 — half the time train against current self for fast skill acquisition, half the time against historical pool for diversity.
3. Monitor goal rate in TensorBoard. If goal rate drops to near zero and stays there for >5M steps, the reward shaping is insufficient and agents have learned mutual passivity. Increase the puck-toward-goal shaping reward.
4. Minimum pool size before meaningful self-play benefit: 5–10 checkpoints. Don't start pool-based self-play with only 1 snapshot.
5. Consider win-rate gating: only add a policy to the pool if it achieves >60% win rate against previous pool members.

**Detection (warning signs):**
- Goal rate in TensorBoard is flat near 0 after 10M+ steps
- Episode reward improves initially then oscillates without trend
- Video of rollout shows agents not moving toward puck or clustering

**Phase:** Address in Phase 2 (training pipeline). Implement pool logic before the full training run, not after.

---

### Pitfall 4: MuJoCo Puck Physics Instability

**What goes wrong:** A flat puck (disc geometry) against boards with high-speed contacts can produce jitter, tunneling (puck passes through boards), or energy explosions. MuJoCo's contact solver uses soft constraints (solref/solimp); a flat disc contacting a flat board edge is a difficult contact configuration. High agent speeds compound this — fast-moving capsule sticks hitting a small puck create large contact forces that destabilize the simulation.

**Why it happens:** The timestep is too large for the contact stiffness, or the contact parameters are not tuned for the geometry. MuJoCo documentation explicitly states: "Divergence is a hint that the timestep is too large for the given choice of integrator." Flat-on-flat contacts (puck-board) are particularly susceptible to numerical bouncing.

**Consequences:** The physics sim diverges mid-episode (NaN state), killing training. Or the puck vibrates in place, producing invalid velocity observations that confuse the policy.

**Prevention:**
1. Start with a conservative timestep (0.005s) and test puck-board bounce behavior before attaching agents. Verify no NaNs over 10k steps at random initial conditions.
2. For the puck geometry, use a thin cylinder (not a box). Add a small chamfer/radius to edges to avoid edge-on-flat contact degenerate cases.
3. Tune `solimp` to use soft contacts for board-puck collisions. Set a restitution coefficient that damps bounces to realistic ice hockey values (coefficient of restitution ~0.6–0.7 for boards).
4. Add an `episode.should_terminate` check for NaN in observation — immediately reset episode and log it. If NaN rate > 0.1% of steps, physics is unstable and needs tuning before running full training.
5. Limit maximum agent velocity in the action space to avoid extreme contact forces. A capsule moving at 10 m/s into a puck is not physically meaningful for hockey.

**Detection (warning signs):**
- NaN observations appearing in early training steps
- TensorBoard shows episode length suddenly dropping to 1 (instant resets)
- Puck teleports or oscillates in quick back-and-forth during manual testing

**Phase:** Address in Phase 1 (environment design) before attaching any RL training.

---

### Pitfall 5: Reward Hacking — Optimizing Shaping Signals Instead of Winning

**What goes wrong:** Dense reward shaping is necessary to guide initial exploration (otherwise agents never discover goals), but agents will optimize exactly what you reward — not what you intend. Common failure modes for this project:
- **Puck-toward-goal shaping**: Agent learns to push puck toward own goal (negative goal direction) because it consistently gets a small reward for moving the puck in any direction, and moving it toward its own goal is geometrically easier.
- **Possession reward**: All 4 agents crowd the puck (maximizes possession probability for the team), eliminating any positioning strategy.
- **Anti-clustering penalty**: If too strong, agents spread out and ignore the puck entirely.
- **Step penalty**: If too strong, agent learns to end episodes early (scoring an own goal is "better" than continuing).

**Why it happens:** The shaped reward is a proxy for winning. Agents exploit the proxy. This is described in the literature as "reward hacking" — the agent achieves high proxy reward without achieving the intended goal.

**Consequences:** Agent appears to "learn" (reward goes up) but plays in a degenerate way. Can take millions of steps to detect. Looks especially bad in the portfolio demo.

**Prevention:**
1. Always track the sparse goal-scoring reward separately from shaped reward in TensorBoard. The former is ground truth; shaped reward can go up while goal rate goes to zero.
2. Make shaping rewards conditional: `puck_toward_goal` reward should only fire when the agent is the one contacting the puck (possession check), not passively.
3. Make the sign of directional rewards explicit: use `dot(puck_velocity, goal_direction)` where `goal_direction` is the vector to the *opponent's* goal, not the puck's current velocity magnitude.
4. Start shaping coefficients small and increase only if agents fail to discover puck interaction in first 5M steps. Iterative tuning beats designing the perfect reward function upfront.
5. Add a curriculum: sparse goal reward only after ~20M steps of shaping, or increase its weight over training.

**Detection (warning signs):**
- Goal rate near 0 but episode reward is high and increasing
- Agents ignore each other and move in fixed patterns
- All 4 agents consistently move toward same position
- Manual review of rollout video at 10M steps shows no puck interaction

**Phase:** Address in Phase 1 (reward design) and monitor throughout Phase 2 (training).

---

## Moderate Pitfalls

---

### Pitfall 6: SB3 Multi-Agent Wrapper Gets Observation/Action Spaces Wrong

**What goes wrong:** SB3 is a single-agent RL library. Using it for 4-agent simultaneous play requires wrapping the environment so it presents as a single agent with a larger observation/action space (4x agent obs concatenated, 4x actions). If the wrapper incorrectly concatenates observations (wrong agent order, misaligned dimensions, missing teammate/opponent observations) or splits actions back incorrectly, the policy trains on garbage inputs.

**Why it happens:** The dm_control soccer environment returns a dict of per-agent observations. The wrapper must deterministically order agents (e.g., by team then by index), concatenate observations, and split the flat action vector back into per-agent actions in the same order. Any shuffle in ordering breaks the correspondence between "what I observed" and "what action was taken."

**Prevention:**
1. Fix agent ordering at environment initialization and assert it is stable across resets.
2. Write a unit test: manually set a known observation state, step the wrapped env, log the obs and action passed to SB3, and verify they match expected values.
3. Use a decentralized wrapper (each agent is a separate SB3 instance with shared weights) rather than centralized concatenation — avoids the ordering problem entirely, and the policy is already in the right format for ONNX export (single-agent input).
4. Shared-weight decentralized execution (IPPO) is the recommended approach for this project — one policy that each agent runs independently with its local observation.

**Detection (warning signs):**
- Policy learns nothing despite correct reward signals
- Agent 1 and Agent 3 seem to behave as mirrors of each other
- Action outputs do not correlate with observations in log traces

**Phase:** Phase 1 (environment design) / Phase 2 (training integration).

---

### Pitfall 7: ONNX Export Fails on Unsupported Ops or Dynamic Control Flow

**What goes wrong:** PyTorch's trace-based ONNX export executes the model once and records the ops. If the policy network has any Python-level control flow (if/else based on input values, dynamic loops), the export will silently bake in the path taken during the single trace run, not the general behavior. Also, some PyTorch ops have no direct ONNX equivalent and will fail at export time.

**Why it happens:** SB3 PPO policies are standard MLP or CNN architectures with no dynamic control flow — this is low risk. The risk is in custom observation preprocessing layers or any LSTM/GRU usage (recurrent PPO). For standard PPO MLP policy, trace-based export is safe.

**Prevention:**
1. Stick to standard SB3 `MlpPolicy` — no recurrent layers. If recurrence is needed, test export thoroughly with `torch.onnx.export` before committing to that architecture.
2. After export, run `onnx.checker.check_model(model)` and `onnxruntime.InferenceSession(model_path)` to validate the model loads without error.
3. Validate numerical parity: run 100 identical observations through the PyTorch model and the ONNX session; assert max absolute difference < 1e-4.
4. Use opset_version=17 or higher for broadest onnxruntime-web compatibility.

**Detection (warning signs):**
- `torch.onnx.export` raises `TracerWarning` about "converting a tensor to a Python boolean"
- ONNX model outputs differ from PyTorch model by more than 1e-3 on validation set
- `onnxruntime.InferenceSession` raises an error on model load

**Phase:** Phase 3 (ONNX export pipeline).

---

### Pitfall 8: onnxruntime-web WASM Files Not Served Correctly

**What goes wrong:** onnxruntime-web requires `.wasm` binary files to be served with `Content-Type: application/wasm`. Static hosts (Vercel, Netlify) do not always set this header by default. Additionally, the WASM files must be co-located with the JS bundle or the path must be explicitly configured via `ort.env.wasm.wasmPaths`. If either condition is not met, the ONNX session fails to initialize with a cryptic error.

**Why it happens:** onnxruntime-web loads WASM at runtime, not at bundle time. Webpack or Vite may not copy the `.wasm` files to the output directory automatically. The file path resolution is relative to the bundle location, not the HTML file.

**Consequences:** Demo is completely broken in production but works locally (because local dev server serves `.wasm` with correct MIME type by default).

**Prevention:**
1. Add explicit MIME type rules to `vercel.json` or `netlify.toml`: `"Content-Type": "application/wasm"` for `*.wasm`.
2. Set `ort.env.wasm.wasmPaths` to an absolute CDN URL (e.g., jsDelivr) rather than relying on relative path resolution.
3. Test the deployed build on a real static host before announcing the demo — do not assume localhost behavior mirrors production.
4. Note: WebGPU backend cannot be used inside a Web Worker due to `import()` restrictions in `WorkerGlobalScope`. Stick to WASM backend in the Web Worker, or use the proxy worker feature instead.

**Detection (warning signs):**
- `InferenceSession.create()` throws `"no available backend found"` in production
- Network tab shows `.wasm` files returning 404 or wrong MIME type
- Demo works on localhost but is broken on deployed URL

**Phase:** Phase 4 (browser integration) / Phase 5 (deploy).

---

### Pitfall 9: ONNX Model Download Blocks Initial Page Load

**What goes wrong:** A single ONNX model for an MLP policy trained on ~50-dimensional observations is small (~1–5 MB). But if the training checkpoint system produces 6–8 models for the timeline slider, total download is potentially 10–40 MB. If these are fetched eagerly at page load, the demo is unusable on slower connections, and a 30-second spinner kills any technical credibility.

**Why it happens:** The default implementation fetches all models upfront or initializes the ONNX session synchronously before rendering anything.

**Prevention:**
1. Lazy-load ONNX weights: fetch only the final trained model at initial load. Fetch earlier checkpoints on demand when the user interacts with the timeline slider.
2. Show the Three.js scene with a visual placeholder (agents at starting positions, static) immediately. Load and initialize the ONNX session in the background.
3. Host ONNX files on Cloudflare R2 with proper `Cache-Control: public, max-age=31536000` headers. A returning visitor should load from cache.
4. Quantize the exported model to INT8 using `onnxruntime` optimization: a 4x size reduction is typical. For an MLP policy this is safe — quantization error is negligible.
5. Display an explicit loading progress indicator — a blank screen with no feedback looks broken, not slow.

**Detection (warning signs):**
- Lighthouse score shows large network payload on initial load
- Demo loads in <3s on local but 15s+ on 4G mobile simulation
- Timeline slider click causes multi-second freeze

**Phase:** Phase 4 (browser integration) and Phase 5 (deploy).

---

### Pitfall 10: Self-Play Opponent Pool Checkpoint Strategy Mistakes

**What goes wrong:** Two failure modes:
1. **Pool too small, updated too rarely:** If the pool is only updated every 5M steps and you're running 50–100M steps total, you get 10–20 checkpoints. With uniform sampling and a pool of 10, the agent spends 90% of time playing against obsolete opponents — no meaningful challenge.
2. **Pool updated too frequently:** If snapshots are taken every 50k steps, the pool fills with nearly-identical policies and diversity is lost. All opponents play the same way.

**Why it happens:** There is a tension between diversity (older checkpoints) and relevance (recent checkpoints). The right frequency depends on how fast the policy is changing.

**Prevention:**
1. Snapshot every ~500k steps (the project spec's stated interval). At 100M total steps, this yields ~200 checkpoints — too many to keep all. Maintain a sliding window of the last 20 checkpoints and a "hall of fame" of 5 particularly strong historical policies.
2. Use win-rate monitoring: if the current policy achieves >80% win rate against the pool consistently for 2M steps, add to pool and force a harder draw.
3. Set `play_against_latest_model_ratio = 0.5` as a starting value. If training is unstable, increase it (more self-play stabilizes gradients). If policy is stagnating, decrease it (more historical opponents add diversity).
4. Save pool membership metadata alongside model checkpoints so training runs are reproducible.

**Detection (warning signs):**
- Win rate against pool is always near 100% (opponents are too easy — pool is stale)
- Win rate against pool is always near 0% (opponent sampling is broken or pool is too hard)
- TensorBoard episode reward oscillates without trend for >20M steps

**Phase:** Phase 2 (training pipeline).

---

## Minor Pitfalls

---

### Pitfall 11: Portfolio Demo Looks Amateurish to Technical Audience

**What goes wrong:** The demo works technically but signals "toy project" rather than "production ML engineering" to a hiring manager. Common failure modes:
- **No explanation of what you're looking at**: The visitor sees capsules moving around and has no idea what decision they're watching. Labeling agents (Team A / Team B), showing live reward curve, and annotating "possession," "goal attempt," etc. signals ML depth.
- **Demo breaks on first visit**: A 404 on the ONNX file, a console error, or a blank canvas immediately signals low quality. Technical hiring managers open the browser console.
- **No evidence of the training process**: Showing only the final trained agent with no artifact of the training arc (the timeline slider, or at minimum a reward curve screenshot) fails to demonstrate the RL engineering — just the 3D rendering.
- **Missing the "why this is hard" narrative**: The README/page copy should explain observation space choices, reward shaping decisions, and what changed at each training milestone. Without this, it reads as "I ran a PPO tutorial."
- **Frozen or erratic agent behavior**: A broken JS physics mirror producing jerky behavior or a policy that never scores reads as a failed demo.

**Prevention:**
1. Open the browser DevTools console before every demo share and verify zero errors.
2. Add agent labels, possession indicator, and period timer as described in the spec. These are quick wins that signal domain knowledge.
3. Embed the TensorBoard reward curve as a static image with stage annotations ("5M — finds puck", "25M — team coordination", "100M — final policy"). This is the core portfolio signal.
4. Write 3–5 sentences explaining the environment design choices in the page copy, not just in the README.
5. Have a pre-recorded video fallback in case WebGPU/WASM fails on the hiring manager's machine (Chrome with strict settings, corporate firewall, etc.).

**Phase:** Phase 4 (browser integration) and Phase 5 (deploy/polish).

---

### Pitfall 12: RunPod Session Interrupted — Checkpoints Lost

**What goes wrong:** RunPod cloud instances can be interrupted (spot pricing, billing, network issues). If checkpoints are only saved locally on the instance volume, a terminated session loses all training progress.

**Prevention:**
1. Save checkpoints to RunPod's persistent volume (not the ephemeral container disk) or sync to S3/R2 every checkpoint.
2. The project spec already mandates checkpoints every 30 minutes — verify this is on persistent storage before starting the long run.
3. Test checkpoint resume before the full run: save a checkpoint at step 1000, restart the script from checkpoint, verify step count continues correctly.

**Phase:** Phase 2 (training pipeline setup).

---

### Pitfall 13: Three.js Render Loop Blocked by ONNX Inference

**What goes wrong:** If ONNX inference runs on the main thread, a 20–50ms inference call (slow WASM on mobile) will freeze the render loop, producing visible frame rate drops. Even at 5ms inference, running 4 agent inferences synchronously on the main thread adds jitter.

**Prevention:**
1. Run all ONNX inference in a Web Worker. Pass observations to worker via `postMessage`, receive actions back. The render loop never blocks.
2. Use `SharedArrayBuffer` for zero-copy transfer if inference frequency is high. Requires COOP/COEP headers (`Cross-Origin-Opener-Policy: same-origin`, `Cross-Origin-Embedder-Policy: require-corp`) — set these in Vercel/Netlify config.
3. Decouple physics step rate from render frame rate: run the physics mirror at a fixed 50Hz in the worker, render at 60fps on the main thread with interpolation.

**Phase:** Phase 4 (browser integration).

---

## Phase-Specific Warnings

| Phase | Topic | Likely Pitfall | Mitigation |
|-------|-------|---------------|------------|
| Phase 1 | Environment design | Puck physics instability with board contacts | Tune solref/solimp and timestep before attaching RL |
| Phase 1 | Observation design | Missing canonical observation spec leads to JS mirror mismatch | Write obs spec doc first; shared between Python env and JS mirror |
| Phase 1 | Reward shaping | Proxy reward exploitation — agents learn shaping instead of winning | Track sparse goal reward separately in TensorBoard from day 1 |
| Phase 2 | Self-play training | Policy collapse from insufficient opponent diversity | Implement pool before first long run; do not retrofit |
| Phase 2 | Training infrastructure | RunPod session interruption | Verify checkpoint save path is on persistent volume before full run |
| Phase 3 | ONNX export | VecNormalize stats not baked into exported model | Wrap policy in normalization module before export |
| Phase 3 | ONNX export | Numerical precision divergence between PyTorch and ONNX | Run 100-obs parity test, assert max diff < 1e-4 |
| Phase 4 | JS physics mirror | Observation value divergence from Python env | Implement numerical parity test: same initial state, compare obs vectors |
| Phase 4 | Browser runtime | WASM files not served with correct MIME type | Set explicit headers in vercel.json / netlify.toml; test on deployed URL |
| Phase 4 | Threading | ONNX inference blocks render loop | Always use Web Worker for inference; never run on main thread |
| Phase 5 | Demo polish | Technical audience sees amateurish demo signals | Zero console errors, agent labels, training curve embedded, pre-recorded fallback |

---

## Sources

- MuJoCo documentation — physics instability and contact tuning: https://mujoco.readthedocs.io/en/stable/computation/index.html
- dm_control soccer README — multi-agent observation design: https://github.com/google-deepmind/dm_control/blob/main/dm_control/locomotion/soccer/README.md
- SB3 export documentation — VecNormalize export caveats: https://stable-baselines3.readthedocs.io/en/master/guide/export.html
- Hugging Face Deep RL Course — self-play pitfalls: https://huggingface.co/learn/deep-rl-course/en/unit7/self-play
- Survey on Self-Play Methods in RL: https://arxiv.org/html/2408.01072v1
- Reward Hacking in RL (Lilian Weng, 2024): https://lilianweng.github.io/posts/2024-11-28-reward-hacking/
- PyTorch ONNX export documentation: https://docs.pytorch.org/docs/stable/onnx.html
- onnxruntime-web deployment guide: https://onnxruntime.ai/docs/tutorials/web/deploy.html
- onnxruntime-web performance diagnosis: https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html
- WebGPU in Service Worker limitation (onnxruntime GitHub): https://github.com/microsoft/onnxruntime/issues/20876
- MuJoCo WASM browser implementation: https://github.com/zalo/mujoco_wasm
- League-based self-play PPO implementation notes: https://medium.com/@kaige.yang0110/implement-self-play-ppo-via-ray-rllib-8268ed97915d
- JointPPO / IPPO for multi-agent PPO: https://arxiv.org/html/2404.11831v1
- MuJoCo floating-point divergence across versions: https://mujoco.readthedocs.io/
- Large ONNX model size in browser (onnxruntime discussion): https://github.com/microsoft/onnxruntime/discussions/24161
