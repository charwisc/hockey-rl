# Roadmap: 2v2 Hockey RL Agent — Portfolio Web Demo

## Overview

This project follows a hard pipeline dependency: build the Python dm_control environment first, lock the observation spec as a cross-boundary contract, execute the training run on RunPod (a human-executed step on a remote GPU VM), export the trained policy to ONNX with mandatory numerical parity validation, build the browser demo against the validated artifacts, then deploy with the correct COOP/COEP headers and asset hosting. Each phase delivers one coherent capability that unblocks the next. The JS physics mirror is the highest-risk item and is the first thing built in Phase 4.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Environment** - Working dm_control 2v2 hockey env with locked observation spec
- [x] **Phase 2: Training** - Self-play PPO training pipeline executed on RunPod RTX 4090 (completed 2026-03-30)
- [ ] **Phase 3: ONNX Export** - Policy exported, numerically validated, artifacts uploaded to R2
- [ ] **Phase 4: Browser Core** - JS physics mirror + onnxruntime-web inference + Three.js renderer
- [ ] **Phase 5: UI, Polish & Deploy** - Scoreboard, reward curve, Vercel deploy with COOP/COEP headers

## Phase Details

### Phase 1: Environment
**Goal**: A working, testable dm_control 2v2 hockey environment with a locked observation/action spec that serves as the cross-boundary contract for all downstream phases
**Depends on**: Nothing (first phase)
**Requirements**: ENV-01, ENV-02, ENV-03, ENV-04, ENV-05, ENV-06, ENV-07
**Success Criteria** (what must be TRUE):
  1. A Python script can step the environment for 1000 steps without NaN values or MuJoCo contact explosions
  2. Puck rebounds off boards at physically plausible angles; ice friction causes visible velocity decay
  3. The reward function emits separate scalar metrics for sparse goal-score and shaped rewards that can be independently plotted
  4. A canonical observation vector spec document exists listing every dimension by index before any JS work begins
  5. ENV-06 obs spec is frozen: adding a field requires explicit version bump, not silent float insertion
**Plans**: TBD
**UI hint**: no

### Phase 2: Training
**Goal**: A complete self-play PPO training run on RunPod RTX 4090 producing 50-100M step checkpoints — this is a human-executed step on a remote GPU VM, not an automated CI step
**Depends on**: Phase 1
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04
**Success Criteria** (what must be TRUE):
  1. TensorBoard goal-rate curve shows the sparse score metric rising over the run (agents learn to score, not just optimize shaped reward)
  2. At least one checkpoint exists past 50M steps with a .zip SB3 artifact and matching VecNormalize stats
  3. Checkpoints are labelled by step count and saved to persistent RunPod volume at 30-minute wall-time intervals
  4. TensorBoard logs are downloadable and contain episode reward, goal rate, and puck possession per checkpoint
**Plans**: 2 plans
Plans:
- [x] 02-01-PLAN.md — SB3 callbacks (self-play pool, wall-time checkpoint, TensorBoard) + HockeyEnv opponent_path bridge
- [x] 02-02-PLAN.md — train.py entry point, requirements-train.txt, integration tests

### Phase 3: ONNX Export
**Goal**: The trained SB3 policy is exported to ONNX opset 17/18, VecNormalize stats are exported as a sidecar JSON, and a numerical parity validation script confirms action delta < threshold — this gate must pass before any browser integration begins
**Depends on**: Phase 2
**Requirements**: EXPORT-01, EXPORT-02
**Success Criteria** (what must be TRUE):
  1. The parity validation script runs 100 observations through both SB3 Python inference and ONNX Runtime Python; max action delta is below the configured threshold on all 100 samples
  2. At least one ONNX file and its obs_stats.json are publicly accessible on Cloudflare R2 before Phase 4 begins
  3. The ONNX model loads in onnxruntime (Python) without warnings about unsupported opset operators
**Plans**: TBD

### Phase 4: Browser Core
**Goal**: The exported ONNX policy runs in-browser via onnxruntime-web in a Web Worker, drives four agents through a JS physics mirror, and the Three.js renderer displays the game at stable 60 fps — JS physics mirror fidelity is validated against Python rollouts before phase sign-off
**Depends on**: Phase 3
**Requirements**: BROWSER-01, BROWSER-02, BROWSER-03, UI-01, UI-02
**Success Criteria** (what must be TRUE):
  1. The JS physics mirror keeps the puck inside the rink (board bounces work), and a numerical parity test comparing JS and Python obs vectors on the same initial state shows no coordinate-frame drift
  2. Agents visibly pursue the puck and attempt to score — policy does not produce frozen or randomly-spinning behavior
  3. ONNX inference runs in a Web Worker; the Three.js render loop maintains 60 fps on desktop even while inference is executing (confirmed via browser DevTools)
  4. Live scoreboard and period timer update each game tick as DOM overlays
  5. Reward curve chart (Chart.js) renders from static TensorBoard JSON and is visible alongside the 3D scene
**Plans**: TBD
**UI hint**: yes

### Phase 5: UI, Polish & Deploy
**Goal**: The demo is live on Vercel with correct COOP/COEP headers, ONNX weights lazily loaded from Cloudflare R2, zero console errors, and the page tells the project story clearly enough for a technical hiring manager to understand in under 90 seconds
**Depends on**: Phase 4
**Requirements**: DEPLOY-01, DEPLOY-02, DEPLOY-03
**Success Criteria** (what must be TRUE):
  1. The live Vercel URL loads the rink scene immediately; ONNX model fetches asynchronously and a progress indicator appears only if the fetch exceeds 1 second
  2. SharedArrayBuffer is available in production (confirmed via `typeof SharedArrayBuffer !== 'undefined'` in browser console) — proves COOP/COEP headers are set correctly
  3. Zero console errors on the live URL in Chrome and Firefox; `.wasm` files are served with `Content-Type: application/wasm`
**Plans**: TBD
**UI hint**: yes

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Environment | 0/TBD | Not started | - |
| 2. Training | 2/2 | Complete   | 2026-03-30 |
| 3. ONNX Export | 0/TBD | Not started | - |
| 4. Browser Core | 0/TBD | Not started | - |
| 5. UI, Polish & Deploy | 0/TBD | Not started | - |
