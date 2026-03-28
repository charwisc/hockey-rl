# Technology Stack

**Project:** 2v2 Hockey RL Agent — Portfolio Web Demo
**Researched:** 2026-03-28
**Research Mode:** Ecosystem

---

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
| CUDA | 12.x | GPU training | Matches RTX 4090 / RunPod image; PyTorch 2.11 ships cu126 wheels |

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

---

## Critical Version Compatibility Notes

### dm_control + MuJoCo pinning (HIGH confidence)
dm_control 1.0.38 requires MuJoCo 3.6.0. These are tightly version-locked — each dm_control release upgrades to one specific MuJoCo version. Do NOT install `mujoco` and `dm-control` independently with `pip install --upgrade`; always install `dm-control` first and let it pull the correct `mujoco` pin as a dependency.

```bash
pip install dm-control==1.0.38
# This installs mujoco==3.6.0 as a dependency
```

### PyTorch ONNX export: dynamo=True (HIGH confidence)
PyTorch 2.11 makes dynamo the default and removes the legacy TorchScript fallback. Use the new API:

```python
torch.onnx.export(
    policy_net,
    (dummy_obs,),
    "policy.onnx",
    dynamo=True,
    opset_version=18,   # highest opset supported by onnxruntime-web 1.24
)
```

Opset 18 is the practical ceiling for onnxruntime-web. Do NOT target opset 19+ — onnxruntime-web WASM does not support all op set 19+ operators.

### onnxruntime-web WASM threading requires COOP/COEP headers (HIGH confidence)
Multi-threaded WASM uses SharedArrayBuffer, which requires cross-origin isolation. Static hosting must set:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

For Vercel, add to `vercel.json`:
```json
{
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        { "key": "Cross-Origin-Opener-Policy", "value": "same-origin" },
        { "key": "Cross-Origin-Embedder-Policy", "value": "require-corp" }
      ]
    }
  ]
}
```

For Netlify, add to `netlify.toml`. Without these headers, onnxruntime-web silently falls back to single-threaded WASM — inference still works but is slower.

### Three.js WebGPU vs WebGL2 (MEDIUM confidence)
r183 ships both `WebGLRenderer` (stable) and `WebGPURenderer` (production-ready since r171). For this project:
- Use `WebGPURenderer` as the primary — it supports all major browsers as of Safari 26 (September 2025)
- It automatically falls back to WebGL2 when WebGPU is unavailable
- Import via `import { WebGPURenderer } from 'three/webgpu'` (not `three`)
- WebGPU init is async — `await renderer.init()` before first render

### SB3 multi-agent pattern (HIGH confidence)
SB3 PPO does not natively support multi-agent environments. The standard pattern for 2v2 with a shared policy is:

1. Build a **single** `gymnasium.Env` whose `observation_space` is one agent's obs and `action_space` is one agent's actions
2. Wrap the 4-agent game as `SubprocVecEnv` with `n_envs=4*N_parallel_games`, where each env instance represents a single player's perspective
3. One PPO instance learns a shared policy that controls all 4 agents via this vectorized trick
4. For self-play: swap the frozen opponent policy in a custom `EvalCallback` / `BaseCallback` every ~500k steps from a saved checkpoint pool

Do NOT use PettingZoo's multi-agent API with SB3 — the overhead of the ParallelEnv-to-VecEnv wrapper is significant and the parameter-sharing pattern above is simpler and faster for symmetric self-play.

---

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

---

## Installation

### Python (training environment)

```bash
# Create environment
python3.11 -m venv .venv
source .venv/bin/activate

# Core training stack
pip install dm-control==1.0.38        # pins mujoco==3.6.0 automatically
pip install torch==2.11.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install stable-baselines3==2.7.1
pip install shimmy[dm-control]        # gymnasium <-> dm_control bridge
pip install tensorboard

# ONNX export + validation
pip install onnx onnxruntime
```

### JavaScript (browser demo)

```bash
npm create vite@latest hockey-demo -- --template vanilla-ts
cd hockey-demo
npm install three@0.183.2
npm install onnxruntime-web@1.24.3
npm install chart.js@4.5.1
npm install -D @types/three
```

---

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

---

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
