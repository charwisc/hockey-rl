<!-- GSD:project-start source:PROJECT.md -->
## Project

**2v2 Hockey RL Agent — Portfolio Web Demo**

A reinforcement learning project that trains 4 agents to play 2v2 ice hockey from scratch using self-play in a dm_control/MuJoCo simulation. The trained policy is exported to ONNX and deployed as an interactive 3D web demo — running fully client-side in the browser via Three.js and onnxruntime-web — showcasing the complete arc from reward shaping to browser inference on a portfolio site.

**Core Value:** A credible, end-to-end ML engineering artifact that demonstrates RL systems fluency — environment design, reward shaping, self-play training, and production export — to technical hiring managers.

### Constraints

- **Tech Stack**: dm_control + MuJoCo 3+ (not mujoco-py — deprecated)
- **Tech Stack**: SB3 PPO, PyTorch, onnxruntime-web, Three.js, Chart.js
- **Infra**: RTX 5090 on RunPod (no multi-node, no TPU)
- **Deploy**: Static site only — no server runtime; all inference client-side
- **Budget**: Cloudflare R2 free tier sufficient for 6–8 checkpoint ONNX files (~few hundred MB total)
<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->
## Technology Stack

## Recommended Stack
### Python Training Environment
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Python | 3.11 | Runtime | SB3 2.7.1 supports 3.10–3.12; 3.11 is the current sweet spot — 3.12 has some SB3 edge-case issues, 3.9 is EOL Oct 2025 |
| MuJoCo | 3.6.0 | Physics simulator | dm_control 1.0.38 pins to 3.6.0 exactly; do not mix versions |
| dm_control | 1.0.38 | RL environment framework | Latest (March 2026); provides dm_env interface, mjcf composer, and multi-agent locomotion patterns; actively maintained by DeepMind |
| PyTorch | 2.11 | Neural network training | Latest stable (March 2026); required for SB3 PPO; dynamo=True ONNX export is default since 2.9 |
| Stable Baselines3 | 2.7.1 | PPO implementation | Latest stable (December 2025); proven vectorized env support; SubprocVecEnv for parallel envs |
| Gymnasium | 0.29.x | Env API standard | SB3 2.7.1 target API; Shimmy converts dm_control to Gymnasium |
| Shimmy | 1.x | dm_control → Gymnasium bridge | Farama-Foundation conversion wrapper; mandatory for SB3 to consume dm_control envs |
| TensorBoard | 2.x | Training metrics | Built into SB3 logger; episode reward, goal rate, possession curves |
| CUDA | 12.8+ | GPU training | Matches RTX 5090 (Blackwell) / RunPod image; PyTorch 2.11 ships cu128 wheels |
### ONNX Export Pipeline
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| torch.onnx (dynamo=True) | built-in to PyTorch 2.11 | Model export | `dynamo=True` is the recommended path since PyTorch 2.5; TorchScript-based legacy exporter deprecated in 2.9, removed in 2.11 |
| onnx | 1.17.x | ONNX graph validation | Validate exported graph; check opset, node shapes before browser deploy |
| onnxruntime | 1.20.x | Python-side parity check | Run ONNX inference in Python to numerically validate against raw PyTorch policy |
### Frontend / Browser Demo
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Three.js | r183 (0.183.x) | 3D renderer | Latest stable (March 2026); WebGPU renderer production-ready since r171; automatic WebGL2 fallback; npm package `three` |
| onnxruntime-web | 1.24.3 | Browser ONNX inference | Latest stable; WASM backend runs on all browsers; WebGPU EP available for performance; Web Worker integration documented |
| Chart.js | 4.5.1 | Reward curve chart | Latest v4 stable; lightweight; tree-shakeable; fits static bundle |
| Vite | 8.x | Bundler / dev server | Latest stable (March 2026); ships with Rolldown (Rust bundler), 10–30x faster builds; first-class static site support |
| TypeScript | 5.x | Type safety | Standard for Vite + Three.js projects; catches observation tensor shape bugs at compile time |
### Infrastructure / Hosting
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Vercel (or Netlify) | N/A | Static site hosting | Both support custom headers (COOP/COEP) required for SharedArrayBuffer / onnxruntime-web multithreading; Vercel has simpler `vercel.json` config |
| Cloudflare R2 | N/A | ONNX model hosting | S3-compatible; free tier covers 6–8 checkpoint files (est. 50–200 MB total); no egress fees to browsers |
## Critical Version Compatibility Notes
### dm_control + MuJoCo pinning (HIGH confidence)
# This installs mujoco==3.6.0 as a dependency
### PyTorch ONNX export: dynamo=True (HIGH confidence)
### onnxruntime-web WASM threading requires COOP/COEP headers (HIGH confidence)
### Three.js WebGPU vs WebGL2 (MEDIUM confidence)
- Use `WebGPURenderer` as the primary — it supports all major browsers as of Safari 26 (September 2025)
- It automatically falls back to WebGL2 when WebGPU is unavailable
- Import via `import { WebGPURenderer } from 'three/webgpu'` (not `three`)
- WebGPU init is async — `await renderer.init()` before first render
### SB3 multi-agent pattern (HIGH confidence)
## Alternatives Considered
| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Physics / env | dm_control 1.0.38 + MuJoCo 3.6 | mujoco-py | mujoco-py is deprecated; no MuJoCo 3.x support; dm_control is the official successor |
| Physics / env | dm_control | IsaacGym / IsaacLab | GPU-parallelized but NVIDIA-only, heavier infra, unnecessary for single-env 2v2 task |
| RL algorithm | PPO via SB3 | SAC, TD3 | SAC/TD3 are off-policy — require replay buffer, slower convergence for self-play; PPO is simpler to tune for continuous multi-agent tasks |
| RL framework | SB3 | Ray RLlib | RLlib has native multi-agent + league self-play; but adds distributed complexity for a single-GPU run; SB3 is simpler, sufficient |
| RL framework | SB3 | CleanRL | CleanRL is single-file scripts without vectorized env abstractions; SB3's SubprocVecEnv is worth the dependency |
| Browser inference | onnxruntime-web | TensorFlow.js | TF.js requires model conversion to TF SavedModel then TFJS format; extra conversion step; onnxruntime-web takes ONNX directly from PyTorch |
| Browser inference | onnxruntime-web | wonnx (Rust/WebGPU) | wonnx is experimental, lower op coverage; onnxruntime-web 1.24 now has a WebGPU EP with much broader coverage |
| 3D renderer | Three.js | Babylon.js | Babylon.js is heavier (~600 KB min vs ~170 KB for Three.js core); Three.js has larger community and more tutorial resources for custom shaders |
| 3D renderer | Three.js | raw WebGL | WebGL from scratch is prohibitively complex for one portfolio project |
| Bundler | Vite 8 | Webpack 5 | Webpack 5 is slower; Vite 8's Rolldown bundler is significantly faster; both work for static sites but Vite DX is better |
| Static hosting | Vercel | GitHub Pages | GitHub Pages does not support custom HTTP response headers — cannot set COOP/COEP, so SharedArrayBuffer (and thus multi-threaded onnxruntime-web WASM) is blocked |
## Installation
### Python (training environment)
# Create environment
# Core training stack
# ONNX export + validation
### JavaScript (browser demo)
## Sources
- [dm-control PyPI — latest version and MuJoCo pin](https://pypi.org/project/dm-control/)
- [dm_control GitHub Releases](https://github.com/google-deepmind/dm_control/releases)
- [stable-baselines3 PyPI](https://pypi.org/project/stable-baselines3/)
- [SB3 Vectorized Environments docs](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)
- [PyTorch 2.11 ONNX export docs](https://docs.pytorch.org/docs/stable/onnx_export.html)
- [onnxruntime-web npm (v1.24.3)](https://www.npmjs.com/package/onnxruntime-web)
- [onnxruntime-web docs — execution providers](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)
- [COOP/COEP for SharedArrayBuffer — web.dev](https://web.dev/articles/coop-coep)
- [Setting COOP/COEP on static hosting (March 2025)](https://blog.tomayac.com/2025/03/08/setting-coop-coep-headers-on-static-hosting-like-github-pages/)
- [Three.js r183 release](https://github.com/mrdoob/three.js/releases/tag/r183)
- [Three.js 2026 WebGPU update](https://www.utsubo.com/blog/threejs-2026-what-changed)
- [Vite 8.0 announcement](https://vite.dev/blog/announcing-vite8)
- [Chart.js npm](https://www.npmjs.com/package/chart.js)
- [Shimmy dm_control multi-agent docs](https://shimmy.farama.org/environments/dm_multi/)
## Confidence Assessment
| Area | Confidence | Notes |
|------|------------|-------|
| dm_control 1.0.38 + MuJoCo 3.6.0 | HIGH | Verified via PyPI and GitHub releases page |
| SB3 2.7.1 + Python/PyTorch requirements | HIGH | Verified via PyPI |
| PyTorch 2.11 latest stable | HIGH | Verified via PyPI and PyTorch GitHub releases |
| onnxruntime-web 1.24.3 | HIGH | Verified via npm (published 21 days ago as of research date) |
| opset_version=18 ceiling for onnxruntime-web | MEDIUM | Based on PyTorch docs and community reports; validate at export time |
| Three.js r183 latest | HIGH | Verified via GitHub releases tag |
| COOP/COEP required for SharedArrayBuffer | HIGH | Web standard requirement; Vercel/Netlify both support config |
| SB3 multi-agent vectorized trick | MEDIUM | Community pattern, not official SB3 docs; verified by multiple blog posts |
| Vite 8.x latest | HIGH | Verified via npm and Vite blog |
| WebGPU production-ready all browsers | MEDIUM | Safari 26 added WebGPU Sept 2025; older mobile Safari will still fall back to WebGL2 |
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
