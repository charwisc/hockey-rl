# Project Research Summary

**Project:** 2v2 Hockey RL Agent — Portfolio Web Demo
**Domain:** RL training pipeline + browser ONNX inference demo
**Researched:** 2026-03-28
**Confidence:** HIGH

---

## Executive Summary

This project sits at the intersection of three distinct engineering domains: MuJoCo/dm_control environment design, PyTorch self-play training with SB3, and browser-side ML inference via onnxruntime-web and Three.js. Each domain is well-documented and the toolchain is mature, but the project's central risk is the hand-off between them — specifically, whether the JavaScript physics mirror in the browser produces observations close enough to the Python training environment that the exported ONNX policy behaves sensibly. Everything else is dependent on getting that bridge right.

The recommended approach is to build strictly in pipeline order: environment first, training second, ONNX export third, browser integration fourth, polish and deploy fifth. The JS physics scaffolding can be built in parallel during Phase 2, but it cannot be meaningfully integrated or validated until Phase 3 artifacts (ONNX model + obs_stats.json) exist. The self-play training architecture uses IPPO (Independent PPO) — one shared policy controlling all four agents via per-agent egocentric observations — which is the correct pattern for getting a single exportable ONNX model that drives all four agents in the browser.

The two risks that can invalidate significant work if not addressed early are: (1) JS physics mirror divergence, which must be caught with a numerical parity test before the ONNX artifacts are locked in; and (2) reward hacking during training, which must be monitored from day one via a separate sparse goal-rate metric in TensorBoard. The static hosting COOP/COEP header requirement for onnxruntime-web multithreading is a deployment gotcha with a known fix (Vercel vercel.json) that should be configured before the demo goes live.

---

## Key Findings

### Recommended Stack

The Python training side is version-locked around dm_control 1.0.38, which pins MuJoCo 3.6.0 exactly — install dm-control first and let it pull the MuJoCo pin. SB3 2.7.1 on Python 3.11 with PyTorch 2.11 (cu126 wheels for RTX 4090) is the training core. The ONNX export uses PyTorch 2.11's dynamo-based exporter (TorchScript path is removed); opset 17 is the recommended target for onnxruntime-web compatibility. The frontend is Vite 8 + TypeScript + Three.js r183 + onnxruntime-web 1.24.3; static hosting on Vercel with ONNX weights on Cloudflare R2.

**Core technologies with version pins:**

- `dm-control==1.0.38` (installs `mujoco==3.6.0` automatically) — physics env and multi-agent framework
- `stable-baselines3==2.7.1` + Python 3.11 + `torch==2.11.0+cu126` — PPO training, VecEnv, checkpointing
- `shimmy[dm-control]` — dm_env → Gymnasium API bridge required by SB3
- `onnx` + `onnxruntime` (Python) — export validation and parity testing
- `three@0.183.2` — 3D renderer; use `WebGPURenderer` from `three/webgpu` with WebGL2 fallback
- `onnxruntime-web@1.24.3` — browser ONNX inference; WASM backend in Web Worker
- `chart.js@4.5.1` — reward curve chart; tree-shakeable, no server dependency
- Vite 8 + TypeScript 5 — bundler and type safety for JS side
- Vercel (not GitHub Pages) — only static host that supports COOP/COEP custom headers needed for SharedArrayBuffer

**Critical version note:** Do not target ONNX opset 19+; onnxruntime-web 1.24 ceiling is opset 18. GitHub Pages cannot set COOP/COEP headers and is therefore incompatible with multi-threaded onnxruntime-web WASM.

### Expected Features

The hiring-manager bar for a demo like this is higher than a typical side project. The demo must be click-and-run, must visibly show agents pursuing the puck, and must show evidence that training actually happened (reward curves, not just the final policy).

**Must have (table stakes):**
- Live in-browser playback without a server — hiring managers spend under 90 seconds; static autoplay is the bar
- Agents visibly pursuing puck and scoring — proves the training produced coherent behavior
- Working JS physics (puck stays in rink, bounces off boards) — broken physics breaks trust immediately
- Scoreboard + period timer — without game state UI the viewer cannot tell what they're watching
- Stable 60 fps on desktop — ONNX inference must run in a Web Worker to avoid blocking the render loop
- Training reward curve (static image or Chart.js) — "show your work" for ML engineers reviewing the demo
- GitHub link + brief on-page explanation — source must be accessible; 2–3 sentences on RL + self-play + ONNX export
- ONNX export with parity validation documented — calling this out in README signals ML engineering literacy

**Should have (differentiators):**
- Training timeline slider (6–8 checkpoints) + stage labels — the single strongest storytelling device; directly shows emergent behavior arc ("5M: finds puck → 50M: team positioning")
- Reward curve synchronized to timeline selection — connects the abstract curve to visible behavior
- Broadcast camera angle (not top-down) — signals 3D rendering competence
- Ice rink normal-map shader — visual polish that distinguishes from a toy prototype
- Architecture diagram on page — one figure showing the full pipeline (dm_control → PPO → ONNX → browser) is parsed in seconds by technical reviewers

**Defer to v2+:**
- Human keyboard controls, multiplayer, full hockey ruleset, realistic character meshes, sound, leaderboard — all explicitly out of scope per PROJECT.md; each adds complexity without portfolio signal

### Architecture Approach

The system has three physically separate runtimes connected by stable data contracts: Python training (RunPod), an artifact store (Cloudflare R2), and the browser. The observation vector layout (22 floats, egocentric per-agent) and action vector layout (4 floats) are defined once in Phase 1 and must be replicated exactly in the browser's `buildObsVector()` function — any drift here breaks the deployed policy. VecNormalize stats are exported as a JSON sidecar (`obs_stats.json`) and applied in the browser before calling the ONNX session; they are not baked into the ONNX graph.

**Major components:**

1. **HockeyTask / HockeyEnv** (Python) — dm_control Composer task defining observation space, reward, and episode logic; wrapped to Gymnasium via Shimmy for SB3 consumption
2. **SelfPlayVecEnv + OpponentPoolCallback** (Python) — 8–16 SubprocVecEnv instances; pool snapshots every ~500k steps; IPPO pattern so one policy drives all 4 agents via per-agent observations
3. **OnnxExporter + parity test** (Python, offline) — wraps SB3 actor-only, exports opset 17 with dynamic batch axis, validates numerics against PyTorch; writes obs_stats.json to R2
4. **HockeyPhysics** (JS) — lightweight mirror (Euler integration, axis-aligned board bounce, circle-circle puck collision); does not need to match MuJoCo exactly, only well enough to keep observations in-distribution
5. **InferenceWorker** (JS Web Worker) — onnxruntime-web WASM session; receives Float32Array obs, returns action Float32Array; session recreated on checkpoint switch
6. **SceneRenderer** (Three.js) — rAF loop on main thread; decoupled from inference; uses previous action if worker result not yet ready
7. **UILayer** (Chart.js + DOM) — scoreboard, timer, reward curve, timeline slider; depends on game state API from HockeyPhysics

### Critical Pitfalls

Research identified 5 pitfalls with outsized consequence for this project specifically:

1. **JS physics mirror divergence** — the single highest-risk item; policy receives out-of-distribution observations and produces erratic or frozen behavior. Prevention: write canonical obs spec document in Phase 1 before building either environment; run a numerical parity test comparing Python rollout observations to JS mirror on the same initial state before declaring Phase 4 complete.

2. **VecNormalize stats not applied in browser** — model runs without error but outputs garbage; deceptively hard to debug. Prevention: export obs_stats.json alongside every ONNX checkpoint; apply `(obs - mean) / sqrt(var + 1e-8)` in JS before calling the ONNX session; confirm with Python-vs-ONNX parity test.

3. **Self-play policy collapse** — naive self-play produces mutual passivity (goal rate → 0) or cyclical strategy reversal. Prevention: implement opponent pool (snapshot every ~500k steps, 70/30 pool/current-self sampling) before the full training run, not after; monitor sparse goal rate in TensorBoard as the ground-truth signal separate from shaped reward.

4. **Reward hacking** — agents optimize shaping proxy instead of scoring; shaped reward rises while goal rate stays near zero. Prevention: track sparse goal-scoring reward separately from day 1; make directional shaping rewards conditional on possession; start shaping coefficients small.

5. **ONNX WASM deployment failure** — demo works locally but is broken in production because `.wasm` files are missing MIME type or COOP/COEP headers are not set. Prevention: set `Content-Type: application/wasm` and COOP/COEP headers in vercel.json before first deploy; test on live URL before announcing.

---

## Implications for Roadmap

The architecture research defines a clear, dependency-ordered build sequence. Phases cannot be reordered without rework.

### Phase 1: Environment Foundation

**Rationale:** Nothing can be built or validated downstream without a working Python environment. Observation spec must be locked here — it is the cross-boundary contract for all later phases.
**Delivers:** Working dm_control 2v2 hockey env (HockeyTask, HockeyEnv), canonical obs/action spec document, MuJoCo physics validated (no NaN, correct puck bounce), reward function with separate TensorBoard metrics for sparse goal rate.
**Addresses features:** Core game loop, observation space, reward shaping
**Avoids pitfalls:** Puck physics instability (tune solref/solimp before RL); reward hacking (instrument separately from day 1); obs spec mismatch (write spec before any JS work)

### Phase 2: Training Pipeline

**Rationale:** Requires Phase 1 env. Must implement opponent pool before the long training run — retrofitting self-play after 50M steps is not practical.
**Delivers:** SelfPlayVecEnv + OpponentPoolCallback, VecNormalize, CheckpointCallback (30-min intervals to persistent volume), TensorBoard logging, successful 50–100M step run on RunPod RTX 4090.
**Uses:** SB3 2.7.1, PyTorch 2.11 cu126, SubprocVecEnv (8–16 envs), TensorBoard
**Avoids pitfalls:** Self-play collapse (pool before run); checkpoint loss (persistent volume verified before full run); policy non-stationarity (pool diversity)

**Note:** JS HockeyPhysics scaffolding and Three.js scene can be built in parallel with Phase 2 using dummy/hardcoded game states, shortening total calendar time.

### Phase 3: ONNX Export Pipeline

**Rationale:** Requires trained checkpoints from Phase 2. Export is an offline step; must lock obs/action contracts before any further JS integration work.
**Delivers:** OnnxableSB3Policy wrapper, 6–8 ONNX checkpoint exports (opset 17, dynamic batch axis), obs_stats.json per checkpoint, numerical parity validation harness, all artifacts uploaded to Cloudflare R2 with manifest.json.
**Uses:** torch.onnx (dynamo=True), onnx 1.17, onnxruntime 1.20, Cloudflare R2
**Avoids pitfalls:** VecNormalize not exported (sidecar JSON + JS apply); opset ceiling (target 17, ceiling is 18); dynamic axes omitted (always export with dynamic_axes)

### Phase 4: Browser Core

**Rationale:** Requires ONNX artifacts from Phase 3 for integration, but HockeyPhysics and Three.js scaffolding can be pre-built. This is the highest-complexity frontend phase.
**Delivers:** HockeyPhysics JS mirror, InferenceWorker (onnxruntime-web WASM), Three.js SceneRenderer (rink, capsules, puck, broadcast camera), main render loop integration with worker, numerical parity test comparing JS obs vectors to Python rollouts.
**Uses:** onnxruntime-web 1.24.3, Three.js r183 (WebGPURenderer), TypeScript 5, Vite 8
**Avoids pitfalls:** Physics mirror divergence (parity test before sign-off); render loop blocking (inference always in Web Worker); WASM MIME type (set in vercel.json before testing on live URL)

### Phase 5: UI, Polish, and Deploy

**Rationale:** Depends on stable game state API from Phase 4. Low-complexity high-impact work; this phase is the difference between a working demo and a credible portfolio piece.
**Delivers:** Scoreboard + period timer, Chart.js reward curve (from exported TensorBoard JSON), training timeline slider with R2 lazy loading + stage labels, mobile video fallback (ffmpeg), Vercel static deploy with COOP/COEP headers, zero console errors, on-page architecture diagram + written explanation, GitHub README with obs/reward/self-play design documentation.
**Uses:** Chart.js 4.5.1, Cloudflare R2 (lazy load, Cache-Control headers), Vercel
**Avoids pitfalls:** Demo appearing amateurish (zero errors, agent labels, reward curve with annotations, pre-recorded fallback); WASM deployment failure (COOP/COEP in vercel.json, tested on live URL)

### Phase Ordering Rationale

- Phase 1 is the universal dependency. The observation spec written here is the data contract all other phases consume.
- Phase 2 must precede Phase 3 by definition (export requires a checkpoint). The full training run is the longest wall-clock item (~8–10 hours); the JS scaffolding can overlap to save calendar time.
- Phase 3 unlocks browser integration. The ONNX artifacts are what make the physics mirror testable with real policy behavior.
- Phases 4 and 5 are sequential (stable game state API required for UI), but both can move quickly once Phase 3 is complete.
- The opponent pool must be built before the full Phase 2 run. This is the one ordering constraint inside a phase that is easy to violate.

### Research Flags

Phases where the implementation will surface new unknowns requiring judgment:

- **Phase 1:** MuJoCo contact parameter tuning (solref/solimp) for puck-board geometry is empirical — no formula, must test. Reward shaping coefficient values are inherently project-specific and require iteration.
- **Phase 2:** Optimal SubprocVecEnv count for RTX 4090 CPU feed (8 vs 16 vs 32) needs empirical testing. Self-play pool snapshot interval (500k steps) is documented as a recommended value, not a guaranteed optimal.
- **Phase 4:** JS physics mirror fidelity threshold (how close is "close enough") has no published standard — must be determined by observing policy behavior. This is the most open-ended implementation decision in the project.

Phases with well-documented patterns (deeper research not needed):

- **Phase 3:** ONNX export is fully documented in SB3 and PyTorch docs. The OnnxableSB3Policy wrapper pattern is confirmed. R2 upload is standard S3 CLI.
- **Phase 5:** Vercel COOP/COEP config is one vercel.json block. Cloudflare R2 CORS config is documented. Chart.js reward curve rendering is standard. No novel patterns.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All versions verified via PyPI, npm, and official release pages as of 2026-03-28. Version compatibility notes confirmed via official docs. |
| Features | HIGH for table stakes, MEDIUM for differentiators | Table stakes are well-established portfolio norms. Timeline slider value proposition is inferred from industry patterns, not direct hiring manager data. |
| Architecture | HIGH | Three-runtime pipeline and IPPO pattern confirmed by SB3 docs and community implementations. Component boundaries and interfaces are cleanly specified. |
| Pitfalls | HIGH for pitfalls 1–3, MEDIUM for 4–5 | Physics mirror divergence and VecNormalize export are confirmed failure modes in the SB3 and onnxruntime communities. Reward hacking is well-documented. Pitfall severity rankings are partially judgment-based. |

**Overall confidence:** HIGH

### Gaps to Address

- **Observation dimension (22 floats):** The obs layout in ARCHITECTURE.md is labeled "illustrative — finalized in Phase 1." The exact count depends on environment design decisions. The JS mirror and ONNX export both depend on this being stable before Phase 3.
- **Optimal parallel env count:** 8–16 SubprocVecEnv instances is the stated range; the actual CPU core count on the RunPod RTX 4090 instance will determine the ceiling. Benchmark this early in Phase 2 before the full run.
- **ONNX model size:** Estimated at 1–3 MB per MLP checkpoint. If the policy architecture is larger (wider layers), lazy loading strategy may need adjustment. Validate after first export.
- **JS physics mirror threshold:** No published standard for "close enough." Plan to run a qualitative behavior test (does the policy chase the puck?) alongside the numerical parity test. If behavior is erratic despite passing numerical parity, the obs spec may have subtle coordinate-frame differences.
- **WebGPU backend in inference worker:** ARCHITECTURE.md recommends WASM over WebGPU for the inference worker. PITFALLS.md confirms WebGPU cannot be used in a WorkerGlobalScope due to import() restrictions. Stick with WASM; no decision needed.

---

## Sources

### Primary (HIGH confidence)

- [dm-control PyPI](https://pypi.org/project/dm-control/) — version 1.0.38, MuJoCo 3.6.0 pin
- [stable-baselines3 PyPI](https://pypi.org/project/stable-baselines3/) — version 2.7.1 requirements
- [SB3 export docs](https://stable-baselines3.readthedocs.io/en/master/guide/export.html) — OnnxableSB3Policy pattern, VecNormalize caveats
- [SB3 VecEnv docs](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html) — SubprocVecEnv, VecNormalize
- [PyTorch ONNX export docs](https://docs.pytorch.org/docs/stable/onnx_export.html) — dynamo=True, opset_version
- [onnxruntime-web npm 1.24.3](https://www.npmjs.com/package/onnxruntime-web) — version confirmed, Web Worker patterns
- [Three.js r183 release](https://github.com/mrdoob/three.js/releases/tag/r183) — WebGPURenderer production status
- [web.dev COOP/COEP](https://web.dev/articles/coop-coep) — SharedArrayBuffer header requirements
- [Cloudflare R2 CORS](https://developers.cloudflare.com/r2/buckets/cors/) — bucket CORS configuration
- [MuJoCo docs — contact tuning](https://mujoco.readthedocs.io/en/stable/computation/index.html) — solref/solimp, timestep stability

### Secondary (MEDIUM confidence)

- [HuggingFace Deep RL — Self-Play](https://huggingface.co/learn/deep-rl-course/en/unit7/self-play) — opponent pool pattern, snapshot interval
- [37 PPO implementation details (ICLR)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) — training stability
- [Survey on Self-Play Methods in RL](https://arxiv.org/html/2408.01072v1) — strategy cycling and pool diversity
- [Reward Hacking in RL (Lilian Weng, 2024)](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) — proxy reward failure modes
- [onnxruntime WebGPU in Worker limitation](https://github.com/microsoft/onnxruntime/issues/20876) — confirmed WASM-only for Worker

### Tertiary (LOW confidence / inference)

- ML engineer portfolio norms — inferred from recruiting community writing; "timeline slider as differentiator" is judgment-based, not validated with hiring managers

---

*Research completed: 2026-03-28*
*Ready for roadmap: yes*
