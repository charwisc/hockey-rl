# 2v2 Hockey RL Agent — Portfolio Web Demo

## What This Is

A reinforcement learning project that trains 4 agents to play 2v2 ice hockey from scratch using self-play in a dm_control/MuJoCo simulation. The trained policy is exported to ONNX and deployed as an interactive 3D web demo — running fully client-side in the browser via Three.js and onnxruntime-web — showcasing the complete arc from reward shaping to browser inference on a portfolio site.

## Core Value

A credible, end-to-end ML engineering artifact that demonstrates RL systems fluency — environment design, reward shaping, self-play training, and production export — to technical hiring managers.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] dm_control 2v2 hockey environment: ice rink with boards/goals, 4 capsule agents with stick hitboxes, realistic puck physics (momentum, board bounce, ice friction, puck-stick interaction)
- [ ] Observation/action space: agent pos/vel, puck pos/vel, teammate/opponent pos/vel; actions: 2D movement direction + speed + stick swing angle
- [ ] Shaped reward function: goal scored, puck-toward-goal, possession, positioning quality, anti-clustering penalty, step penalty
- [ ] PPO self-play training pipeline via Stable Baselines 3 with opponent pool updated every ~500k steps
- [ ] 8–16 parallel environments configured to saturate a single RTX 4090 on RunPod
- [ ] Checkpoint system: save every 30 min wall-time, labelled by step count; target 50–100M steps in 8–10 hours
- [ ] TensorBoard logging: episode reward, goal rate, puck possession per checkpoint
- [ ] PyTorch → ONNX export pipeline with numerical parity validation against Python inference
- [ ] Three.js 3D renderer: ice rink with normal-mapped shader, board/goal geometry, capsule player meshes, flat puck with emission, broadcast-angle camera
- [ ] onnxruntime-web inference running in a Web Worker (non-blocking render loop)
- [ ] Lightweight JS physics mirror of the Python environment — close enough for exported policy to behave correctly in-browser
- [ ] Live scoreboard, period timer, and reward curve chart (Chart.js)
- [ ] Mobile fallback: pre-recorded video replay of final trained agent at fixed camera angle
- [ ] Static deploy: Vercel or Netlify; ONNX model weights hosted on Cloudflare R2 with lazy loading
- [ ] Training timeline slider: 6–8 checkpoint models, scrub through training history with stage labels (e.g. "5M — finds puck", "50M — team tactics")

### Out of Scope

- Full hockey ruleset (offsides, icing, penalties) — simplified physics sim only; adds complexity without portfolio value
- Human-vs-agent or multiplayer mode — self-play showcase is the point; adding controls fragments the demo story
- Server-side inference — static hosting constraint; also demonstrates client-side capability
- Realistic player mesh / animation rigging — capsule geometry is intentional; effort better spent on training quality

## Context

- **Target audience:** Technical hiring managers evaluating ML engineering depth, not game completeness
- **MuJoCo binding:** dm_control (not mujoco-py) — actively maintained, cleaner multi-agent API, dm_env interface
- **Training infrastructure:** RunPod cloud VM with RTX 4090, single-GPU run, ~8–10 hour wall-time target
- **RL algorithm:** PPO via Stable Baselines 3 — well-understood for continuous control, large community, easy SB3 vectorized env integration
- **Browser inference:** onnxruntime-web + Web Worker to avoid blocking the Three.js render loop
- **Model hosting:** Cloudflare R2 (S3-compatible) for ONNX weights, lazily fetched per checkpoint to keep initial page load fast
- **v1 minimum viable demo:** One polished trained agent running in-browser. Timeline slider is a v1.5/v2 enhancement.
- **JS physics mirror fidelity:** Does not need to be pixel-perfect — just close enough that the exported policy produces sensible behavior. Full fidelity mismatch = biggest single technical risk.

## Constraints

- **Tech Stack**: dm_control + MuJoCo 3+ (not mujoco-py — deprecated)
- **Tech Stack**: SB3 PPO, PyTorch, onnxruntime-web, Three.js, Chart.js
- **Infra**: Single RTX 4090 on RunPod (no multi-node, no TPU)
- **Deploy**: Static site only — no server runtime; all inference client-side
- **Budget**: Cloudflare R2 free tier sufficient for 6–8 checkpoint ONNX files (~few hundred MB total)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| dm_control over mujoco-py | Actively maintained, cleaner multi-agent API, dm_env interface integrates well with SB3 wrappers | — Pending |
| PPO via SB3 | Well-supported continuous control algorithm; SB3 vectorized envs saturate GPU easily | — Pending |
| ONNX for browser inference | Framework-agnostic export; onnxruntime-web is well-supported and actively maintained | — Pending |
| Playable final agent = v1 minimum | Timeline slider is a great feature but not the core portfolio signal — ML engineering depth is | — Pending |
| Capsule geometry (not rigged mesh) | Portfolio signal is the RL training quality, not 3D art; capsules are fast to render and keep scope honest | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-28 after initialization*
