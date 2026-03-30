# Requirements: 2v2 Hockey RL Agent — Portfolio Web Demo

**Defined:** 2026-03-28
**Core Value:** A credible, end-to-end ML engineering artifact that demonstrates RL systems fluency — environment design, reward shaping, self-play training, and production export — to technical hiring managers.

---

## v1 Requirements

### Environment

- [x] **ENV-01**: dm_control 2v2 hockey environment includes ice rink geometry with boards, goals, and face-off positions defined in MJCF XML
- [x] **ENV-02**: Four capsule agents each have a stick hitbox; simplified rigid-body physics govern movement and collisions
- [x] **ENV-03**: Puck physics include momentum, board bouncing, ice friction coefficient, and puck-stick interaction on contact
- [x] **ENV-04**: Action space per agent: 2D movement direction, speed scalar, stick swing angle (continuous)
- [x] **ENV-05**: Observation space per agent is egocentric: own pos/vel, puck pos/vel, teammate pos/vel, both opponent pos/vel (single shared policy drives all 4 agents via egocentric observations)
- [x] **ENV-06**: Canonical observation vector layout is documented as a numbered, immutable spec before any JS code is written
- [x] **ENV-07**: Shaped reward function implements: goal scored (+10), puck-toward-opponent-goal (continuous small), puck possession (continuous small), positioning quality bonus, anti-clustering penalty, per-step penalty

### Training

- [x] **TRAIN-01**: Self-play uses a historical opponent pool of ~20 checkpoints; pool updated every ~500k training steps with 50% latest / 50% historical mix
- [ ] **TRAIN-02**: SubprocVecEnv configured for 8–16 parallel environments on RunPod RTX 4090
- [x] **TRAIN-03**: Checkpoints saved every 30 minutes of wall-time, labelled by step count; target 50–100M steps in 8–10 hour run
- [x] **TRAIN-04**: TensorBoard logging records episode reward, goal rate, and puck possession stats per checkpoint

### Export

- [ ] **EXPORT-01**: PyTorch policy exported to ONNX opset 18 using `torch.onnx.export(..., dynamo=True)`; VecNormalize running statistics baked into a wrapper `nn.Module` before export (not a separate JSON artifact)
- [ ] **EXPORT-02**: Numerical parity validation script runs 100 observations through both Python SB3 inference and ONNX Runtime Python; max action delta < configurable threshold; must pass before any browser integration begins

### Browser Core

- [ ] **BROWSER-01**: JS physics mirror implements: agent movement mechanics, linear puck damping, axis-aligned board bounce, AABB goal detection — close enough to trained observation distribution that exported policy produces coherent behavior
- [ ] **BROWSER-02**: onnxruntime-web inference runs in a dedicated Web Worker using the WASM execution provider; observation `Float32Array.buffer` transferred as Transferable; inference never blocks the Three.js render loop
- [ ] **BROWSER-03**: Three.js renderer displays: ice rink with board geometry, goals, four capsule player meshes with team colors, flat puck with slight emissive material; maintains stable 60 fps on desktop

### UI

- [ ] **UI-01**: Live scoreboard and period timer displayed as DOM overlay updating each game tick
- [ ] **UI-02**: Reward curve chart (Chart.js) rendered from TensorBoard logs exported to static JSON; visible on page alongside the 3D demo

### Deploy

- [ ] **DEPLOY-01**: `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers set on Vercel or Netlify via config file; required for onnxruntime-web SharedArrayBuffer / multi-threaded WASM
- [ ] **DEPLOY-02**: ONNX model weights hosted on Cloudflare R2 public bucket (not bundled in static site)
- [ ] **DEPLOY-03**: ONNX model fetched asynchronously after initial page render; page displays rink scene immediately; progress indicator shown only if fetch exceeds 1 second

---

## v2 Requirements

### Timeline Slider

- **SLIDER-01**: Training timeline slider with 6–8 ONNX checkpoint models loaded lazily from Cloudflare R2
- **SLIDER-02**: Stage labels at each checkpoint (e.g. "Step 0 — random", "5M — finds puck", "50M — team tactics")
- **SLIDER-03**: Reward curve chart highlight synchronized to selected checkpoint position
- **SLIDER-04**: Batch ONNX export script processes all checkpoint `.pt` files to `.onnx` in one pass

### Polish

- **POLISH-01**: Ice surface uses Three.js MeshStandardMaterial with normal map texture; broadcast camera angle replaces top-down default
- **POLISH-02**: Mobile video fallback: pre-recorded mp4 of final agent at fixed camera, served via `<video>` tag behind device/touch detection
- **POLISH-03**: Architecture diagram on page (SVG: dm_control → PPO → ONNX → browser inference → Three.js) with 2–3 sentence project description and GitHub link

---

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full hockey ruleset (offsides, icing, penalties) | Adds environment complexity with zero portfolio signal; simplified physics demonstrates scope judgment |
| Human-vs-agent controls | Fragments demo narrative; portfolio story is the self-play training system, not a game |
| Multiplayer via WebRTC/sockets | Requires server or peer-to-peer layer; breaks static hosting constraint |
| Realistic character meshes / animation rigging | Capsule geometry is intentional; 3D art is a different skillset from ML engineering |
| Server-side inference (Flask, FastAPI) | Defeats the client-side ONNX architecture story; adds cost and uptime risk |
| GitHub Pages deployment | Cannot set COOP/COEP headers required for SharedArrayBuffer; eliminated by stack constraint |
| W&B / MLflow embedded in page | Heavy dependency, requires auth, breaks static hosting; Chart.js on static JSON is sufficient |
| Sound effects / music | No portfolio signal; audio bugs annoy visitors |

---

## Traceability

| Requirement | Phase | Phase Name | Status |
|-------------|-------|------------|--------|
| ENV-01 | Phase 1 | Environment | Pending |
| ENV-02 | Phase 1 | Environment | Pending |
| ENV-03 | Phase 1 | Environment | Pending |
| ENV-04 | Phase 1 | Environment | Pending |
| ENV-05 | Phase 1 | Environment | Pending |
| ENV-06 | Phase 1 | Environment | Pending |
| ENV-07 | Phase 1 | Environment | Pending |
| TRAIN-01 | Phase 2 | Training | Pending |
| TRAIN-02 | Phase 2 | Training | Pending |
| TRAIN-03 | Phase 2 | Training | Pending |
| TRAIN-04 | Phase 2 | Training | Pending |
| EXPORT-01 | Phase 3 | ONNX Export | Pending |
| EXPORT-02 | Phase 3 | ONNX Export | Pending |
| BROWSER-01 | Phase 4 | Browser Core | Pending |
| BROWSER-02 | Phase 4 | Browser Core | Pending |
| BROWSER-03 | Phase 4 | Browser Core | Pending |
| UI-01 | Phase 4 | Browser Core | Pending |
| UI-02 | Phase 4 | Browser Core | Pending |
| DEPLOY-01 | Phase 5 | UI, Polish & Deploy | Pending |
| DEPLOY-02 | Phase 5 | UI, Polish & Deploy | Pending |
| DEPLOY-03 | Phase 5 | UI, Polish & Deploy | Pending |

**Coverage:**
- v1 requirements: 21 total
- Mapped to phases: 21
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-28*
*Last updated: 2026-03-28 after roadmap creation*
