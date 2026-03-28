# Feature Landscape

**Domain:** Browser-based RL portfolio demo (2v2 hockey, dm_control + ONNX + Three.js)
**Researched:** 2026-03-28
**Overall confidence:** HIGH for table stakes / MEDIUM for differentiators

---

## Table Stakes

Features a hiring manager expects to see. Missing any of these and the demo reads as unfinished or unserious.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Live in-browser agent playback | Hiring managers spend <90 seconds on a portfolio. "Click and see it run" is the bar. Static screenshots fail this. | Med | onnxruntime-web + Three.js already committed in PROJECT.md |
| Visible scoreboard + period timer | Without game state UI, the viewer cannot tell what they're watching. Hockey without a score is ambient noise. | Low | Chart.js or plain DOM; 1-2 days |
| Working physics — puck stays in rink, bounces off boards | If the puck escapes geometry or phases through boards, the demo breaks trust immediately. | High | Biggest single risk: JS physics mirror fidelity |
| Agents visibly pursuing the puck | Policy must produce coherent behavior — agents move toward puck, not randomly. Proves training worked. | N/A (training quality, not UI) | Requires successful 50–100M step run |
| Clean 3D render at stable 60 fps (desktop) | Technical audience notices jank. Three.js must not block on inference; Web Worker is mandatory here. | Med | Already in spec; Web Worker isolates ONNX from render loop |
| Page loads without server | Static deploy (Vercel/Netlify) is standard. Any server dependency is a red flag for "didn't finish the deployment story." | Low | Already in spec |
| Brief written explanation of the project | Recruiters who cannot understand what they're looking at skip it. 2–3 sentences on the page about RL + self-play + ONNX export. | Low | Copywriting, not engineering |
| Link to code (GitHub) | Engineers verify claims. A demo without source is suspicious. | Low | README should explain environment design, reward shaping, architecture |
| TensorBoard / training metrics visible somewhere | "Show your work" — ML engineers want to see training curves, not just the finished agent. Proves the training actually ran. | Low–Med | Can be static images on the page or embedded Chart.js reward curves |
| ONNX export with parity validation documented | The export is a core ML engineering skill. Must be called out — either in README or on the page. | Low (documentation) | Numerical parity test between Python and ONNX inference is the artifact |

---

## Differentiators

Features that separate this demo from the typical "CartPole with a webpage" portfolio entry. These are the signals that get a hiring manager to forward your resume.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Training timeline slider (6–8 checkpoints) | Lets the viewer directly observe emergent behavior — "5M: finds puck, 20M: shoots, 50M: team positioning." This is the single most powerful storytelling device available. No other standard portfolio demo has this. | High | Requires: multiple ONNX exports, R2 lazy loading per checkpoint, UI scrubber. Identified as v1.5/v2 in PROJECT.md — worth prioritizing if training quality is good. |
| Stage labels on the timeline | Plain checkpoint numbers are meaningless. Labels like "Discovers shooting" or "Team spacing emerges" turn a technical artifact into a narrative. | Low | Copywriting + JSON config. No eng cost beyond the slider itself. |
| Reward curve chart synchronized to timeline | When user selects a checkpoint, the chart highlights where in training that checkpoint falls. Connects the abstract curve to the visible behavior. | Med | Chart.js + checkpoint metadata JSON |
| Self-play opponent pool design documented | Self-play is technically sophisticated. Documenting the opponent pool rotation (update every ~500k steps) demonstrates awareness of non-stationarity — a known MARL pitfall that most entry-level demos ignore entirely. | Low (documentation) | Goes in README and on the page |
| Anti-clustering reward term explained | Most hockey RL demos (if they exist) don't discuss team coordination incentives. Calling out the anti-clustering penalty shows environment design sophistication. | Low (documentation) | One paragraph on the page |
| Broadcast camera angle (not top-down) | Top-down is the easy path. Broadcast angle makes the demo feel like watching actual hockey. Signals 3D rendering competence. | Med | Three.js camera positioning + orbit/follow logic |
| Ice shader with normal map | A flat grey plane reads as a prototype. A visually polished rink signals that attention to quality extends beyond just the model. | Med | Three.js ShaderMaterial or MeshStandardMaterial with normal map texture |
| "How it works" architecture section on the page | One diagram showing: dm_control → PPO training → ONNX export → browser inference → Three.js render. Technical hiring managers parse this in seconds and it confirms full-stack ML engineering literacy. | Low | SVG or image diagram; 2–4 hours |
| Lazy loading ONNX weights per checkpoint | Demonstrates awareness of performance constraints: initial page load is fast, weights are fetched on demand. This is production-minded behavior that distinguishes engineers from researchers. | Med | R2 + fetch-on-demand already in spec |
| Mobile video fallback | Covers the case where a hiring manager views on phone. A broken Three.js experience on mobile kills the demo. Pre-recorded video maintains the impression. | Med | ffmpeg record + `<video>` tag behind device detection |

---

## Anti-Features

Things to explicitly NOT build. Each has a clear reason and an alternative.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Human keyboard controls (play against agents) | Adds input handling, collision detection with human-controlled character, and UI states. Fragments the demo narrative — the portfolio story is "I built a self-play training system," not "I built a game." PROJECT.md already excludes this. | Showcase the self-play system as the star. Add a "watch mode" camera that follows the puck. |
| Full hockey ruleset (offsides, icing, power plays) | Each rule requires environment logic, UI communication, and reward shaping work. None of it signals ML engineering depth. Hiring managers do not evaluate hockey rules. | Simplified 2-goal, continuous-play physics. Document the simplification choice explicitly — it shows good scope judgment. |
| Multiplayer (two humans via WebRTC/sockets) | Requires a server or peer-to-peer networking layer, breaks static hosting constraint, and doubles the surface area. Adds weeks. | Two agents vs. two agents, fully autonomous. More impressive as RL showcase anyway. |
| Realistic character meshes / animation rigging | 3D art is a separate skillset. Capsule geometry is honest and fast. Rigged meshes for portfolio RL signal misplaced effort. | High-quality shader on the rink ice. Capsules with team color materials. The visual quality comes from lighting/materials, not mesh complexity. |
| Server-side inference (Flask / FastAPI) | Defeats the architecture story. The whole point is client-side ONNX. A server inference endpoint also introduces latency, cost, and uptime risk for a static portfolio site. | onnxruntime-web in a Web Worker. Already in spec. Document the choice and its tradeoffs. |
| Real-time retraining in the browser | No credible ML engineering use case for in-browser PPO. It would be slow, fragile, and distracting from the core artifact. | Show training progression through the checkpoint timeline instead. |
| W&B / MLflow UI embedded in the page | Heavy dependency, requires authentication, breaks static hosting. | Export training curves as static JSON; render with Chart.js. Clean and self-contained. |
| Leaderboard / user comparison features | Requires database, auth, server. Entirely orthogonal to the portfolio story. | Remove. |
| Sound effects / music | Audio bugs are annoying and add no signal. | Remove. |
| Loading spinner longer than 3 seconds on initial page | Kills first impression. If model weights block initial render, viewers close the tab. | Load a lightweight "title card" render immediately. Fetch ONNX model weights async. Show a progress indicator only if fetch takes >1 second. |

---

## Feature Dependencies

```
Three.js scene (rink + capsules + puck)
  └── Working JS physics mirror
        └── ONNX inference in Web Worker
              └── ONNX model export (Python side)
                    └── Trained policy (50–100M step run)

Scoreboard / period timer
  └── Three.js scene (needs game state)

Reward curve chart (Chart.js)
  └── TensorBoard logs exported as JSON

Training timeline slider
  └── Multiple ONNX checkpoint exports (6–8)
  └── R2 lazy loading per checkpoint
  └── Reward curve chart (to sync highlight position)
  └── Stage labels (JSON config)

Mobile video fallback
  └── Trained policy (need final agent to record)
  └── Three.js scene (record from the same renderer)

Architecture diagram on page
  └── Nothing; can be built any time
```

Key sequencing constraint: **the JS physics mirror and trained policy quality gate everything else.** Until a policy exists that produces coherent behavior against a faithful-enough physics mirror, no amount of UI work matters.

---

## MVP Recommendation

**V1 (minimum credible demo):**

1. Three.js rink with working JS physics mirror — puck stays in bounds, bounces correctly
2. onnxruntime-web Web Worker running final checkpoint policy
3. Four capsule agents with team colors visibly pursuing puck and scoring
4. Scoreboard + period timer
5. Brief reward curve chart (static image or Chart.js from exported JSON)
6. 2–3 sentence explanation + GitHub link on page
7. Mobile video fallback for the final agent

**V1.5 (strong differentiator, pursue if training produces quality behavior):**

8. Training timeline slider with 6–8 checkpoints + stage labels
9. Reward curve synchronized to checkpoint selection
10. Broadcast camera angle + ice normal map shader

**Defer:**
- "How it works" architecture diagram: low complexity, high value, but not blocking. Build after V1 ships.
- Anti-clustering / self-play documentation: write during README pass, link from page.

---

## Complexity Estimates

| Feature | Estimate | Notes |
|---------|----------|-------|
| JS physics mirror | 3–5 days | Highest risk item. Puck physics (momentum, board bounce, ice friction, stick interaction) must match Python env closely enough for exported policy to behave. |
| Three.js scene (rink + capsules + broadcast cam) | 2–3 days | Standard Three.js work. Ice shader adds 0.5 days. |
| onnxruntime-web Web Worker integration | 1–2 days | Well-documented; main risk is tensor shape mismatches at the JS/ONNX boundary. |
| Scoreboard + timer | 0.5 days | DOM or Three.js overlay. |
| Reward curve chart (Chart.js) | 0.5–1 day | Export SB3 logs to JSON; render with Chart.js. |
| Training timeline slider + R2 lazy loading | 2–3 days | R2 fetch, checkpoint state management, UI scrubber, stage label config. |
| Mobile video fallback | 0.5 days | ffmpeg screen capture + `<video>` tag + device detection. |
| Architecture diagram | 0.5 days | SVG or Figma export. |
| Static deploy (Vercel/Netlify + R2) | 0.5 days | Known pattern; low risk. |

---

## Sources

- [ML Engineer Portfolio Projects That Will Get You Hired in 2025 — Medium](https://medium.com/@santosh.rout.cr7/ml-engineer-portfolio-projects-that-will-get-you-hired-in-2025-d1f2e20d6c79)
- [Machine Learning Engineer Portfolio Playbook for SWEs — Interview Kickstart](https://interviewkickstart.com/blogs/articles/machine-learning-engineer-portfolio)
- [Interactive Deep Reinforcement Learning Demo — Inria / Flowers team](https://developmentalsystems.org/Interactive_DeepRL_Demo/)
- [ONNX Runtime Web — Microsoft Open Source Blog](https://opensource.microsoft.com/blog/2021/09/02/onnx-runtime-web-running-your-machine-learning-model-in-browser/)
- [Using WebGPU — ONNX Runtime docs](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)
- [ONNX Runtime Web unleashes generative AI in the browser — Microsoft, 2024](https://opensource.microsoft.com/blog/2024/02/29/onnx-runtime-web-unleashes-generative-ai-in-the-browser-using-webgpu/)
- [mujoco_wasm — Simulate and Render MuJoCo Models in the Browser](https://github.com/zalo/mujoco_wasm)
- [MuJoCo WASM official bindings — google-deepmind/mujoco](https://github.com/google-deepmind/mujoco/blob/main/wasm/README.md)
- [Tensorboard Integration — Stable Baselines3 docs](https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html)
- [Interactive Visualization for Debugging RL — OpenReview (NeurIPS workshop)](https://openreview.net/forum?id=ZN3s7fN-bo)
- [Client-Side AI in 2025: Running ML Models in the Browser — Medium](https://medium.com/@sauravgupta2800/client-side-ai-in-2025-what-i-learned-running-ml-models-entirely-in-the-browser-aa12683f457f)
- [The Beginner's RL Playground — Arthur Juliani / Medium](https://awjuliani.medium.com/the-beginners-rl-playground-a-simple-interactive-website-for-grokking-reinforcement-learning-b6f1edcf7c63)
