# Phase 1: Environment - Research

**Researched:** 2026-03-28
**Domain:** dm_control Composer + MuJoCo 3.6 — custom multi-agent RL environment
**Confidence:** HIGH (core architecture) / MEDIUM (puck contact tuning specifics)

---

## Summary

Phase 1 builds the foundational artifact that all downstream phases depend on: a working, testable 2v2 hockey MuJoCo environment with a locked, documented observation/action specification. No training, no browser code, no ONNX — only the Python simulation that produces stable physics, a correct reward signal, and a frozen obs vector layout.

The implementation follows the dm_control Composer pattern exactly (Entity + Arena + Task hierarchy), wraps with a thin hand-written Gymnasium adapter (not Shimmy — Shimmy's DmControlCompatibilityV0 returns dict obs and requires FlattenObservation stacked on top; a hand-written wrapper is simpler and gives full control over the obs vector layout). The Gymnasium wrapper is the single source of truth for the obs vector: it flattens named physics fields in a deterministic, documented order, and that order becomes the cross-boundary contract for ONNX export and the JS physics mirror.

Physics stability is the hardest implementation risk in this phase. The puck-board contact configuration (thin cylinder against flat wall) is a known MuJoCo instability trigger. The mitigation is conservative timestep (0.005 s), semi-implicit Euler integrator, and contact parameters tuned before any RL training is attached. NaN detection must be wired into the environment from day one.

**Primary recommendation:** Build in this order — (1) MJCF XML with arena + puck only, verify board bounces for 10k random steps with no NaN; (2) add agent capsules and actuators, verify movement; (3) wire observations and rewards; (4) write the obs spec document; (5) add Gymnasium wrapper; (6) run the full test suite.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ENV-01 | dm_control 2v2 hockey environment with ice rink geometry (boards, goals, face-off positions) in MJCF XML | Architecture section: HockeyArena class; MJCF wall/goal patterns from dm_control soccer source |
| ENV-02 | Four capsule agents each have a stick hitbox; simplified rigid-body physics govern movement and collisions | Architecture section: HockeyPlayer Entity with capsule body + stick geom + 3 actuators |
| ENV-03 | Puck physics include momentum, board bouncing, ice friction coefficient, and puck-stick interaction on contact | Physics Tuning section; solref/solimp guidance; cylinder puck geometry recommendation |
| ENV-04 | Action space per agent: 2D movement direction, speed scalar, stick swing angle (continuous) | Interface 2 in Architecture docs; 4-float action vector confirmed |
| ENV-05 | Observation space per agent is egocentric — own pos/vel, puck pos/vel, teammate pos/vel, both opponent pos/vel | Interface 1 in Architecture docs; soccer env observables pattern with framepos/framelinvel sensors |
| ENV-06 | Canonical observation vector layout documented as numbered, immutable spec before any JS code | Obs Spec section; 22-float layout with versioning protocol |
| ENV-07 | Shaped reward: goal scored (+10), puck-toward-goal (continuous), puck possession, positioning quality, anti-clustering, per-step penalty | Reward Function section; dm_control.utils.rewards.tolerance() pattern |
</phase_requirements>

---

## Project Constraints (from CLAUDE.md)

The following directives are locked and must be followed by the planner:

- **Tech Stack:** dm_control + MuJoCo 3+ (not mujoco-py — deprecated and unsupported)
- **Tech Stack:** SB3 PPO, PyTorch, onnxruntime-web, Three.js, Chart.js — no substitutions
- **Infra:** Single RTX 4090 on RunPod (no multi-node, no TPU)
- **Deploy:** Static site only — all inference client-side
- **Budget:** Cloudflare R2 free tier for ONNX files
- **GSD Workflow:** Direct repo edits outside a GSD command require explicit user permission

---

## Standard Stack

### Core (Phase 1 only)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.11 | Runtime | SB3 2.7.1 sweet spot; 3.12 has edge-case issues; 3.9 EOL |
| MuJoCo | 3.6.0 | Physics simulator | dm_control 1.0.38 pins exactly to 3.6.0 — do NOT install separately |
| dm_control | 1.0.38 | Composer Entity/Arena/Task framework | Latest stable March 2026; provides mjcf, composer, dm_env |
| Gymnasium | 0.29.x | Env API standard | SB3 2.7.1 target API |
| NumPy | (dm_control dependency) | Array ops for obs/reward | Pinned by dm_control |
| pytest | 8.x | Test runner | Standard Python test framework |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| dm_control.utils.rewards | built-in | Shaped reward helpers (tolerance, sigmoids) | Use for puck_toward_goal, positioning rewards |
| gymnasium.utils.env_checker | built-in | validate_env() | Run before plugging into SB3 |
| numpy.testing | built-in | assert_allclose, assert_array_equal | Physics stability and reward correctness tests |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-written Gymnasium wrapper | Shimmy DmControlCompatibilityV0 + FlattenObservation | Shimmy adds a dependency and wraps the obs as a dict requiring a second FlattenObservation wrapper; hand-written gives explicit control over obs vector ordering — critical for the cross-boundary contract |
| Hand-written Gymnasium wrapper | Shimmy DmControlMultiAgentCompatibilityV0 | Outputs PettingZoo ParallelEnv, not Gymnasium; requires further wrapping for SB3; overkill for shared-policy IPPO pattern |
| dm_control.composer | dm_control.rl.control | composer is the right layer for custom multi-entity environments; rl.control is for single-entity domains only |

### Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate

pip install dm-control==1.0.38    # pulls mujoco==3.6.0 as dependency
pip install gymnasium==0.29.1
pip install pytest
pip install numpy
```

**Version verification:**
```bash
python -c "import dm_control; print(dm_control.__version__)"
python -c "import mujoco; print(mujoco.__version__)"   # must be 3.6.0
```

---

## Architecture Patterns

### Recommended Project Structure

```
env/
├── hockey_arena.py       # HockeyArena(composer.Arena) — MJCF rink geometry
├── hockey_player.py      # HockeyPlayer(composer.Entity) — capsule + stick + actuators
├── hockey_puck.py        # HockeyPuck(composer.Entity) — cylinder puck
├── hockey_task.py        # HockeyTask(composer.Task) — obs, reward, termination
├── hockey_env.py         # HockeyEnv(gymnasium.Env) — thin Gymnasium wrapper
├── obs_spec.py           # OBS_SPEC: list of (name, slice, units, notes)
└── mjcf/
    ├── arena.xml         # Standalone rink XML (optional — can be inline in _build)
    ├── player.xml        # Player body XML (optional)
    └── puck.xml          # Puck XML (optional)

tests/
├── test_physics.py       # Stability: 1000-step rollout, NaN check, board bounce
├── test_observations.py  # Obs spec conformance: shape, dtype, index values
├── test_rewards.py       # Reward unit tests: sparse + shaped components
├── test_gymnasium.py     # Gymnasium API compliance: check_env, spaces
└── conftest.py           # Shared fixtures: make_env(), sample_obs()
```

### Pattern 1: dm_control Composer Class Hierarchy

```python
# Source: dm_control soccer source + ARCHITECTURE.md

from dm_control import composer
from dm_control.composer.observation import observable

class HockeyArena(composer.Arena):
    """Ice rink: flat plane + 4 board walls + 2 goals."""

    def _build(self, rink_length=30.0, rink_width=15.0, name="hockey_arena"):
        super()._build(name=name)
        # Ground plane with ice friction
        self._ground = self._mjcf_root.worldbody.add(
            'geom', name='ice', type='plane',
            size=[rink_length/2, rink_width/2, 0.1],
            friction=[0.05, 0.005, 0.0001],   # slide, torsional, rolling — ice
            rgba=[0.8, 0.9, 1.0, 1.0])

        # Four board walls as plane geoms (same pattern as dm_control soccer Pitch)
        # Each wall: type='plane', xyaxes defines orientation, pos places it at boundary
        # board_friction higher than ice — boards should arrest lateral movement
        for pos, xyaxes in self._wall_configs(rink_length, rink_width):
            self._mjcf_root.worldbody.add(
                'geom', type='plane',
                size=[1e-7, 1e-7, 1e-7],
                pos=pos, xyaxes=xyaxes,
                friction=[0.7, 0.005, 0.0001],
                rgba=[0.3, 0.3, 0.4, 1.0])

        # Goals: AABB detectors as site geoms + collision geom for post contacts
        self._home_goal_site = self._mjcf_root.worldbody.add(
            'site', name='home_goal', type='box',
            size=[0.5, 2.0, 1.0],
            pos=[-rink_length/2 + 0.5, 0, 0.5])
        self._away_goal_site = self._mjcf_root.worldbody.add(
            'site', name='away_goal', type='box',
            size=[0.5, 2.0, 1.0],
            pos=[rink_length/2 - 0.5, 0, 0.5])


class HockeyPlayer(composer.Entity):
    """Capsule body + stick geom + 3 continuous actuators."""

    def _build(self, team: int, player_idx: int, name=None):
        self._mjcf_root = mjcf.RootElement(model=name or f"player_{team}_{player_idx}")
        # Capsule body: radius 0.3m, half-height 0.4m (represents torso)
        body = self._mjcf_root.worldbody.add('body', name='body')
        body.add('geom', name='capsule', type='capsule',
                 size=[0.3, 0.4], pos=[0, 0, 0.7],
                 mass=75.0)
        # Stick hitbox: flat box extending forward from player
        body.add('geom', name='stick', type='box',
                 size=[0.1, 0.5, 0.05], pos=[0.4, 0, 0.3],
                 contype=1, conaffinity=2)   # stick collides with puck
        # Joints: x translation, y translation, z rotation
        body.add('joint', name='x', type='slide', axis=[1, 0, 0])
        body.add('joint', name='y', type='slide', axis=[0, 1, 0])
        body.add('joint', name='rot', type='hinge', axis=[0, 0, 1])
        # Actuators: velocity-controlled (direct velocity targets)
        self._mjcf_root.actuator.add('velocity', name='vx', joint='x', kv=200.0)
        self._mjcf_root.actuator.add('velocity', name='vy', joint='y', kv=200.0)
        self._mjcf_root.actuator.add('velocity', name='vrot', joint='rot', kv=50.0)

    @property
    def mjcf_model(self):
        return self._mjcf_root

    def _build_observables(self):
        return HockeyPlayerObservables(self)


class HockeyTask(composer.Task):
    """Reward computation, observation assembly, episode management."""

    def __init__(self, arena, players, puck, time_limit=60.0):
        self._arena = arena
        self._players = players   # list of 4 HockeyPlayer, indexed [0,1] = team0, [2,3] = team1
        self._puck = puck
        self._time_limit = time_limit
        self._score = [0, 0]

    @property
    def root_entity(self):
        return self._arena

    def initialize_episode_mjcf(self, random_state):
        # Attach entities to arena (called once at compile time)
        for player in self._players:
            self._arena.attach(player)
        self._arena.attach(self._puck)

    def initialize_episode(self, physics, random_state):
        # Reset to face-off positions
        # Home team: players at (-5, ±2), Away team: players at (+5, ±2)
        # Puck at (0, 0)
        self._score = [0, 0]

    def get_observation(self, physics):
        # Per-agent observations assembled by HockeyEnv, not here
        # Task provides access to physics state; env wrapper flattens to vector
        return {}

    def get_reward(self, physics):
        # Returns scalar total reward for the active training agent
        # Detailed breakdown in reward_components() for info dict logging
        pass

    def should_terminate_episode(self, physics):
        return (physics.time() >= self._time_limit or
                self._score[0] > 0 or self._score[1] > 0)
```

**Key insight:** `HockeyTask.get_observation()` returns an empty dict. The Gymnasium wrapper (`HockeyEnv`) reads physics state directly to build the flattened obs vector. This avoids the dm_env observation dict overhead and gives explicit control over the obs index layout.

### Pattern 2: Gymnasium Wrapper — Obs Vector Assembly

```python
# Source: architecture decision from ARCHITECTURE.md + Shimmy source pattern

import gymnasium as gym
import numpy as np
from dm_control import composer

class HockeyEnv(gym.Env):
    """
    Thin Gymnasium wrapper around HockeyTask.
    Presents a single-agent view: one agent's obs/action/reward.
    The 4-agent game is stepped internally; opponent actions come from
    a frozen policy pool (injected from outside during training).
    """

    OBS_DIM = 22   # Frozen from obs_spec.py — changing requires version bump

    def __init__(self, agent_idx: int, frozen_opponent_fn=None):
        self._agent_idx = agent_idx   # 0–3
        self._frozen_opponent_fn = frozen_opponent_fn or self._random_opponent
        self._dm_env = composer.Environment(HockeyTask(...))

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.OBS_DIM,), dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(4,), dtype=np.float32)
            # [move_x, move_y, speed, stick_angle]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        timestep = self._dm_env.reset()
        obs = self._build_obs(timestep)
        return obs.astype(np.float32), {}

    def step(self, action):
        # Build full 4-agent action array
        all_actions = self._assemble_all_actions(action)
        timestep = self._dm_env.step(all_actions)
        obs = self._build_obs(timestep).astype(np.float32)
        reward = self._get_reward_for_agent(timestep)
        terminated = timestep.last()
        truncated = False
        info = self._get_reward_components(timestep)   # for TensorBoard
        return obs, float(reward), terminated, truncated, info

    def _build_obs(self, timestep) -> np.ndarray:
        """
        Assemble 22-float egocentric obs vector for self._agent_idx.
        Index layout is the cross-boundary contract — see obs_spec.py.
        """
        physics = self._dm_env.physics
        agent = self._agent_idx

        obs = np.zeros(self.OBS_DIM, dtype=np.float64)
        # [0:2] own pos (world x,y)
        obs[0:2] = physics.named.data.qpos[f"player_{agent}/x"], physics.named.data.qpos[f"player_{agent}/y"]
        # [2:4] own vel
        obs[2:4] = physics.named.data.qvel[f"player_{agent}/x"], physics.named.data.qvel[f"player_{agent}/y"]
        # [4:6] teammate pos
        teammate = (agent + 1) % 2 + (agent // 2) * 2   # 0↔1, 2↔3
        obs[4:6] = ...
        # [8:12] opponent 0 pos/vel, [12:16] opponent 1 pos/vel
        # [16:18] puck pos, [18:20] puck vel
        # [20] stick_angle, [21] facing_angle
        return obs.astype(np.float32)

    def _get_reward_components(self, timestep) -> dict:
        """
        Return reward breakdown for info dict.
        SB3 Monitor wrapper reads 'episode' key; custom callback reads others.
        Keys: 'r_goal', 'r_puck_toward_goal', 'r_possession', 'r_positioning',
              'r_clustering', 'r_step_penalty'
        """
        return {
            'r_goal': ...,
            'r_puck_toward_goal': ...,
            'r_possession': ...,
            'r_positioning': ...,
            'r_clustering': ...,
            'r_step_penalty': -0.001,
        }
```

### Pattern 3: MJCF Contact Parameters for Puck Physics

```xml
<!-- Source: MuJoCo contact documentation + dm_control soccer_ball.py analysis -->
<!-- Puck: thin cylinder, not box — avoids edge-on-flat degenerate contacts -->

<worldbody>
  <!-- Global physics options: conservative timestep for contact stability -->
  <!-- Set in <option> element: timestep="0.005" integrator="implicitfast" -->

  <!-- Puck body -->
  <body name="puck" pos="0 0 0.02">
    <freejoint name="puck_free"/>
    <geom name="puck_geom"
          type="cylinder"
          size="0.05 0.01"        <!-- radius=5cm, half-height=1cm -->
          mass="0.170"            <!-- 170g: regulation ice hockey puck -->
          friction="0.05 0.005 0.0001"   <!-- low slide friction = ice surface -->
          solref="0.02 0.4"       <!-- stiffness timescale=0.02s, damp_ratio=0.4 -->
          solimp="0.9 0.95 0.001" <!-- impedance: soft enough to avoid jitter -->
          condim="6"              <!-- full 6-DOF contact for realistic spin -->
          contype="1"
          conaffinity="1"/>
  </body>

  <!-- Board wall (plane geom) -->
  <!-- Board contact: higher restitution than ice — puck bounces off boards -->
  <!-- Use geom-level contact override for board-puck pairs -->
</worldbody>

<option timestep="0.005"
        integrator="implicitfast"   <!-- MuJoCo 3.x recommended for stiff contacts -->
        cone="elliptic"             <!-- elliptic friction cone more stable than pyramidal -->
        gravity="0 0 -9.81"/>
```

**Contact parameter guidance (MEDIUM confidence — requires empirical tuning):**

- `solref="0.02 0.4"`: The first element (0.02 s) is the constraint timescale — must be >= 2x the timestep (2 × 0.005 = 0.01). The second element (0.4) is the damping ratio — lower = more bouncy. For boards: increase damping ratio to ~0.7 to reduce excessive bouncing.
- `solimp="0.9 0.95 0.001"`: The first two values control constraint impedance (higher = stiffer contact). The third is penetration allowed (0.001 m). Increasing impedance reduces slip but can cause numerical stiffness.
- `cone="elliptic"`: Verified by MuJoCo maintainer discussion to produce more stable bouncing than pyramidal (the default). Set globally in `<option>`.
- `integrator="implicitfast"`: Recommended for MuJoCo 3.x; handles stiff contacts better than `Euler`. Falls back gracefully.

**Board bounce physics:** Board walls are plane geoms (same pattern as dm_control soccer pitch). Plane geoms are infinite half-spaces — no edge artifacts. Puck-board contact is cylinder-on-plane, which is well-conditioned.

**Ice friction:** Set on the ground plane geom: `friction="0.05 0.005 0.0001"`. The first element (0.05) is the sliding friction coefficient — verified approximately correct for ice (real ice: 0.01–0.1 depending on temperature). This is what causes visible velocity decay per the success criterion.

### Pattern 4: Reward Function with Separate Scalar Metrics

```python
# Source: dm_control.utils.rewards pattern + REQUIREMENTS.md reward spec

from dm_control.utils import rewards as dmr

class HockeyTask(composer.Task):

    def get_reward_components(self, physics) -> dict:
        """
        Compute all reward components independently.
        Returns dict — each value is a scalar that can be logged separately.
        """
        puck_pos = self._get_puck_pos(physics)
        agent_pos = self._get_agent_pos(physics, self._active_agent_idx)
        opponent_goal_pos = np.array([self._rink_length/2, 0])
        own_goal_pos = np.array([-self._rink_length/2, 0])

        # Sparse: goal scored — only fires on terminal step
        r_goal = 10.0 if self._score_this_step[0] else 0.0

        # Shaped: puck toward opponent goal
        # dot(puck_vel, direction_to_opponent_goal) — fires only when agent has possession
        puck_vel = self._get_puck_vel(physics)
        goal_dir = (opponent_goal_pos - puck_pos)
        goal_dir /= (np.linalg.norm(goal_dir) + 1e-8)
        puck_toward = np.dot(puck_vel, goal_dir)
        r_puck_toward = np.clip(puck_toward, -1.0, 1.0) * 0.05 * self._agent_has_possession

        # Shaped: puck possession (binary, per-agent)
        r_possession = 0.01 if self._agent_has_possession else 0.0

        # Shaped: positioning quality (distance to ideal support position)
        # Use dm_control tolerance() for smooth gradient
        ideal_pos = self._compute_ideal_position(physics)
        dist_to_ideal = np.linalg.norm(agent_pos - ideal_pos)
        r_positioning = dmr.tolerance(dist_to_ideal, bounds=(0, 3.0), margin=5.0) * 0.005

        # Anti-clustering: penalize being too close to teammate
        teammate_pos = self._get_teammate_pos(physics)
        teammate_dist = np.linalg.norm(agent_pos - teammate_pos)
        r_clustering = -0.01 if teammate_dist < 2.0 else 0.0

        # Per-step survival penalty
        r_step = -0.001

        return {
            'r_goal': r_goal,
            'r_puck_toward_goal': float(r_puck_toward),
            'r_possession': float(r_possession),
            'r_positioning': float(r_positioning),
            'r_clustering': float(r_clustering),
            'r_step_penalty': r_step,
        }

    def get_reward(self, physics) -> float:
        """Single scalar for the dm_env TimeStep."""
        comps = self.get_reward_components(physics)
        return sum(comps.values())
```

**TensorBoard logging pattern:** The Gymnasium wrapper populates the `info` dict with all `r_*` keys from `get_reward_components()`. A custom SB3 `BaseCallback` reads `infos` from the rollout buffer and calls `self.logger.record("train/r_goal", ...)` each episode. SB3 automatically logs `episode/ep_rew_mean` from the Monitor wrapper; custom metrics require a custom callback.

### Pattern 5: Obs Spec Document Format

```python
# obs_spec.py — The cross-boundary contract. Do not modify without bumping OBS_SPEC_VERSION.

OBS_SPEC_VERSION = "1.0.0"   # Bump minor for additions; major for reorders

OBS_DIM = 22

OBS_SPEC = [
    # (index_slice, field_name, units, coordinate_frame, notes)
    (slice(0, 2),   "agent_pos",       "meters",      "world_xy",   "xy position of this agent"),
    (slice(2, 4),   "agent_vel",       "m/s",         "world_xy",   "xy velocity of this agent"),
    (slice(4, 6),   "teammate_pos",    "meters",      "world_xy",   "xy position of teammate"),
    (slice(6, 8),   "teammate_vel",    "m/s",         "world_xy",   "xy velocity of teammate"),
    (slice(8, 10),  "opponent0_pos",   "meters",      "world_xy",   "xy position of opponent index 0"),
    (slice(10, 12), "opponent0_vel",   "m/s",         "world_xy",   "xy velocity of opponent index 0"),
    (slice(12, 14), "opponent1_pos",   "meters",      "world_xy",   "xy position of opponent index 1"),
    (slice(14, 16), "opponent1_vel",   "m/s",         "world_xy",   "xy velocity of opponent index 1"),
    (slice(16, 18), "puck_pos",        "meters",      "world_xy",   "xy position of puck"),
    (slice(18, 20), "puck_vel",        "m/s",         "world_xy",   "xy velocity of puck"),
    (slice(20, 21), "stick_angle",     "radians",     "agent_local","stick rotation relative to agent facing"),
    (slice(21, 22), "facing_angle",    "radians",     "world",      "agent heading in world frame (atan2)"),
]

assert len(OBS_SPEC) == 10
assert sum(s.stop - s.start for s, *_ in OBS_SPEC) == OBS_DIM
```

**Versioning protocol (ENV-06):** The `OBS_SPEC_VERSION` string is baked into the ONNX export metadata (via `onnx.helper.make_model` `doc_string` field). If any index changes or a new field is inserted, the major version must increment. The JS physics mirror imports the same spec JSON. A CI test asserts the version strings match.

**World vs egocentric frame:** The architecture uses world-frame positions rather than agent-egocentric (rotated relative to agent facing). This simplifies the implementation and the JS mirror. The shared policy learns to use `facing_angle` to interpret world-frame positions. If training reveals the policy fails to learn directional behavior, egocentric frame is an option in v2.

### Anti-Patterns to Avoid

- **Using `dm_control.rl.control.Environment`:** This is for single-entity built-in domains (cartpole, walker, etc.). Composer is required for custom multi-entity scenes.
- **Returning obs dict from `HockeyTask.get_observation()`:** If you implement `get_observation()` in the Task, dm_env wraps it and Shimmy unpacks it as a dict. Building the flat obs vector in the Gymnasium wrapper bypasses this entirely and gives full index control.
- **Setting `contype`/`conaffinity` incorrectly:** If agent capsules and puck share the same bits, agents will collide with each other through the puck. Use distinct bits: puck=1, agents=2, boards=4, stick=8. Tune `conaffinity` to allow stick-puck and puck-board contacts only.
- **`solref[0]` < `2 * timestep`:** MuJoCo 3.x has a safety mechanism that clamps `solref[0]` to `max(solref[0], 2*timestep)`. With timestep=0.005, set `solref[0]` >= 0.01 (use 0.02 for safety margin).
- **Skipping the NaN guard:** Adding `if np.any(np.isnan(physics.data.qpos)): env.reset()` to the step loop prevents single NaN contacts from killing entire training runs.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Reward shaping curves | Custom sigmoid/exponential functions | `dm_control.utils.rewards.tolerance()` | Provides 8 verified sigmoid types, handles boundary cases, returns [0,1] |
| Obs/action space validation | Manual shape/dtype assertions | `gymnasium.utils.env_checker.check_env()` | Checks all Gymnasium API requirements SB3 depends on |
| Multi-step simulation substeps | Manual for-loop over `physics.step()` | `dm_control.composer.Environment` (set `task.physics_steps_per_control_step`) | Composer handles substep loop, observation updates, and hook timing |
| MJCF entity attachment | Manual XML string concatenation | `arena.attach(entity)` in `initialize_episode_mjcf()` | Composer manages namespace collision avoidance and site attachment automatically |
| Geom collision filtering | Manually tracking which geoms collide | MuJoCo `contype`/`conaffinity` bitmask system | Already built in; just set the bits |

**Key insight:** dm_control Composer exists specifically to handle the boilerplate of multi-entity MuJoCo environments. Using `rl.control` or raw MuJoCo bindings for this phase would require hand-rolling everything Composer provides (entity attachment, observable management, episode lifecycle).

---

## Common Pitfalls

### Pitfall 1: Puck-Board Contact Instability (NaN Explosion)

**What goes wrong:** Thin cylinder puck contacting flat board wall plane at high speed produces large contact forces that exceed the solver's constraint budget, yielding NaN state values. The episode terminates immediately, and training never gets started.

**Why it happens:** `solref[0]` < `2 * timestep` causes the safety clamp to fire, changing effective contact stiffness unexpectedly. Alternatively, the puck freejoint allows out-of-plane rotation, and the puck flips onto its edge — a degenerate contact configuration.

**How to avoid:** (1) Set `timestep=0.005`, `solref[0]=0.02` (4x the minimum). (2) Limit freejoint to 2D: lock z-rotation and tilt DOFs using `freejoint` + `equality` constraints, or switch to slide joints for x/y + hinge for z-rotation only. (3) Run 10k random-action steps on arena + puck ONLY before adding agents; assert zero NaN occurrences.

**Warning signs:** Episode length drops to 1–5 steps in first training minutes. TensorBoard ep_len_mean < 10.

### Pitfall 2: Obs Vector Index Drift

**What goes wrong:** During development, a new obs field is inserted mid-vector (e.g., `puck_height` added at index 17), silently shifting all subsequent indices. Code that reads `obs[16:18]` for puck_pos now reads the wrong values. The ONNX export bakes in the old layout, and the JS mirror uses the wrong indices.

**Why it happens:** Python list slicing makes mid-insertion easy and silent. No compile-time check that indices are stable.

**How to avoid:** (1) Define `OBS_SPEC` as the single source of truth in `obs_spec.py`. (2) `_build_obs()` must use `OBS_SPEC` slice objects, not magic numbers. (3) Any change to `OBS_SPEC` must increment `OBS_SPEC_VERSION`. (4) A test asserts that `OBS_DIM == sum of all slice lengths` and that no slice overlaps another.

**Warning signs:** Policy reward drops to near-zero after an obs change. ONNX parity test fails.

### Pitfall 3: Shaped Reward Proxy Exploitation

**What goes wrong:** The puck-toward-goal shaping reward fires for any puck velocity toward the opponent goal — including when the agent pushes the puck toward its own goal and the puck then bounces toward the opponent goal. Agents discover this and learn to rapidly bounce the puck off boards rather than drive it toward the goal.

**Why it happens:** The reward is not gated on agent-puck possession. Any puck velocity component in the right direction earns reward regardless of who caused it.

**How to avoid:** Gate `r_puck_toward_goal` on a possession flag: the reward only fires if `distance(agent_stick, puck) < 0.5 m`. Track `r_goal` (sparse) separately in TensorBoard from the first training step; if `r_goal` stays near zero while `r_puck_toward_goal` rises, the proxy is being exploited.

**Warning signs:** High episode reward but zero goal rate in TensorBoard. Agents move in repetitive patterns without approaching the opponent's goal zone.

### Pitfall 4: Gymnasium API Non-Compliance Blocks SB3

**What goes wrong:** `HockeyEnv.reset()` returns only `obs` (not the `(obs, info)` tuple). Or `step()` returns 4 values instead of 5 (old Gym API). SB3 2.7.1 uses the Gymnasium API and will raise a cryptic error or silently use wrong values.

**Why it happens:** Old tutorials use the Gym (not Gymnasium) API. The difference is `reset()` returns `(obs, info)` and `step()` returns `(obs, reward, terminated, truncated, info)`.

**How to avoid:** Run `gymnasium.utils.env_checker.check_env(env)` before any SB3 integration. It catches all common API violations.

**Warning signs:** `check_env()` raises `UserWarning` or `AssertionError`. SB3 `PPO(env=...)` constructor raises a TypeError.

### Pitfall 5: 3D Rink Dimensions Incompatible with Physics

**What goes wrong:** Rink set to 30m × 15m (real NHL rink scale: ~60m × 26m, but smaller is fine for RL). Capsule agents have radius 0.3m. With 4 agents on a 30×15 rink, agents at face-off positions (±5m, ±2m) are fine. But if rink is scaled down further for faster physics or fewer NaN contacts, agents clip through walls.

**How to avoid:** Minimum safe rink: 20m × 10m. Agent capsule radius: 0.3m. Wall thickness (plane geom): effectively infinite — no minimum needed. Verify agent starting positions don't overlap at `initialize_episode`.

---

## Observation Vector Spec (ENV-06 Cross-Boundary Contract)

This section is the canonical document. It is replicated in `obs_spec.py` and in the JSON sidecar exported with each ONNX model.

```
OBS_SPEC_VERSION = "1.0.0"
OBS_DIM = 22
DTYPE = float32
COORDINATE_FRAME = world_xy (all positions and velocities in world frame)
UNITS = meters and m/s (no normalization applied here; VecNormalize handles that)

Index  | Name              | Units  | Notes
-------|-------------------|--------|------------------------------------------
0      | agent_pos_x       | m      | Agent x position in world frame
1      | agent_pos_y       | m      | Agent y position in world frame
2      | agent_vel_x       | m/s    | Agent x velocity in world frame
3      | agent_vel_y       | m/s    | Agent y velocity in world frame
4      | teammate_pos_x    | m      | Teammate x position in world frame
5      | teammate_pos_y    | m      | Teammate y position in world frame
6      | teammate_vel_x    | m/s    | Teammate x velocity in world frame
7      | teammate_vel_y    | m/s    | Teammate y velocity in world frame
8      | opponent0_pos_x   | m      | Opponent 0 x position in world frame
9      | opponent0_pos_y   | m      | Opponent 0 y position in world frame
10     | opponent0_vel_x   | m/s    | Opponent 0 x velocity in world frame
11     | opponent0_vel_y   | m/s    | Opponent 0 y velocity in world frame
12     | opponent1_pos_x   | m      | Opponent 1 x position in world frame
13     | opponent1_pos_y   | m      | Opponent 1 y position in world frame
14     | opponent1_vel_x   | m/s    | Opponent 1 x velocity in world frame
15     | opponent1_vel_y   | m/s    | Opponent 1 y velocity in world frame
16     | puck_pos_x        | m      | Puck x position in world frame
17     | puck_pos_y        | m      | Puck y position in world frame
18     | puck_vel_x        | m/s    | Puck x velocity in world frame
19     | puck_vel_y        | m/s    | Puck y velocity in world frame
20     | stick_angle       | rad    | Stick rotation relative to agent body heading
21     | facing_angle      | rad    | Agent heading in world frame (atan2(vy, vx) of body)

AGENT ORDERING (for all 4-agent action assembly):
  Index 0 = Team 0, Player 0  (training agent when agent_idx=0)
  Index 1 = Team 0, Player 1
  Index 2 = Team 1, Player 0
  Index 3 = Team 1, Player 1

MODIFICATION PROTOCOL:
  - Any change to index assignments → bump major version (1.x.x → 2.0.0)
  - New field appended at end → bump minor version (1.0.x → 1.1.0)
  - Notes-only changes → bump patch version (1.0.0 → 1.0.1)
  - All downstream (ONNX export, JS mirror) must be updated before new version is used
```

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 | Runtime | ✗ — system has 3.12 | 3.12.3 | Create venv with python3.11 if available; 3.12 has SB3 edge-case issues but is usable for Phase 1 tests |
| dm-control 1.0.38 | Environment framework | ✗ — not installed | — | Install via pip in venv |
| mujoco 3.6.0 | Physics simulation | ✗ — not installed | — | Installed as dm-control dependency |
| gymnasium 0.29.x | Gymnasium API | ✗ — not installed | — | Install via pip |
| pytest | Test runner | ✗ — not installed | — | Install via pip |
| MuJoCo display (GLX) | Rendering only | Unknown | — | Set `render_mode=None` for headless testing; rendering not required for Phase 1 |

**Missing dependencies with no fallback:**
- None — all dependencies are pip-installable in a Python venv.

**Missing dependencies with fallback:**
- Python 3.12 is present (not 3.11). For Phase 1 (environment only), 3.12 should work fine. SB3 edge cases affect training (Phase 2), not environment construction. Recommendation: use 3.12 for Phase 1, provision 3.11 on RunPod for Phase 2.

**Missing dependencies (install in venv):**
- dm-control==1.0.38, mujoco==3.6.0 (pinned by dm-control), gymnasium==0.29.1, pytest, numpy

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.x |
| Config file | `pytest.ini` — see Wave 0 |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ -v --tb=short` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ENV-01 | Rink XML compiles without error; boards, goals, face-off sites exist in physics | unit | `pytest tests/test_physics.py::test_arena_compiles -x` | Wave 0 |
| ENV-02 | 4 capsule agents load; stick geoms present; actuators respond to inputs | unit | `pytest tests/test_physics.py::test_agents_load -x` | Wave 0 |
| ENV-03 | Puck bounces off boards at plausible angle; velocity decays each step; no NaN in 1000 steps | integration | `pytest tests/test_physics.py::test_puck_stability_1000steps -x` | Wave 0 |
| ENV-03 | Board bounce angle: reflected angle within 20% of incident angle | unit | `pytest tests/test_physics.py::test_board_bounce_angle -x` | Wave 0 |
| ENV-03 | Ice friction: puck velocity magnitude decreases monotonically when no force applied | unit | `pytest tests/test_physics.py::test_puck_friction_decay -x` | Wave 0 |
| ENV-04 | Action space shape == (4,), dtype float32, bounds [-1, 1] | unit | `pytest tests/test_gymnasium.py::test_action_space_spec -x` | Wave 0 |
| ENV-05 | Obs shape == (22,), dtype float32; obs indices match OBS_SPEC | unit | `pytest tests/test_observations.py::test_obs_shape_and_dtype -x` | Wave 0 |
| ENV-05 | Obs indices 0:2 track agent position correctly over multiple steps | integration | `pytest tests/test_observations.py::test_obs_agent_pos_tracks_physics -x` | Wave 0 |
| ENV-05 | Obs indices 16:20 track puck position and velocity correctly | integration | `pytest tests/test_observations.py::test_obs_puck_tracks_physics -x` | Wave 0 |
| ENV-06 | OBS_SPEC slice lengths sum to OBS_DIM; no overlapping slices; version string present | unit | `pytest tests/test_observations.py::test_obs_spec_integrity -x` | Wave 0 |
| ENV-07 | r_goal == 10.0 exactly when puck crosses goal line | unit | `pytest tests/test_rewards.py::test_sparse_goal_reward -x` | Wave 0 |
| ENV-07 | r_puck_toward_goal > 0 when agent has possession and puck moves toward opponent goal | unit | `pytest tests/test_rewards.py::test_puck_toward_goal_fires_with_possession -x` | Wave 0 |
| ENV-07 | r_puck_toward_goal == 0 when agent does NOT have possession (no proxy exploit) | unit | `pytest tests/test_rewards.py::test_puck_toward_goal_gated_on_possession -x` | Wave 0 |
| ENV-07 | All 6 reward components present in info dict every step | unit | `pytest tests/test_rewards.py::test_reward_components_in_info -x` | Wave 0 |
| SC-1 | 1000 steps without NaN or MuJoCo contact explosion | integration | `pytest tests/test_physics.py::test_1000_steps_no_nan -x` | Wave 0 |
| SC-3 | Sparse r_goal and shaped rewards can be independently extracted from info | integration | `pytest tests/test_rewards.py::test_reward_independent_extraction -x` | Wave 0 |
| SC-5 | OBS_SPEC_VERSION is a non-empty string; inserting a new field (simulated) fails without version bump | unit | `pytest tests/test_observations.py::test_obs_spec_version_required -x` | Wave 0 |
| Gymnasium | check_env() passes with zero warnings | integration | `pytest tests/test_gymnasium.py::test_check_env_passes -x` | Wave 0 |

### Physics Stability Test Pattern

```python
# tests/test_physics.py
import numpy as np
import pytest
from env.hockey_env import HockeyEnv

@pytest.fixture
def env():
    e = HockeyEnv(agent_idx=0)
    yield e
    e.close()

def test_1000_steps_no_nan(env):
    """Success criterion 1: 1000 steps, no NaN, no contact explosion."""
    obs, _ = env.reset(seed=42)
    assert not np.any(np.isnan(obs)), "NaN in initial observation"
    for step in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert not np.any(np.isnan(obs)), f"NaN in obs at step {step}"
        assert not np.isnan(reward), f"NaN in reward at step {step}"
        assert np.all(np.abs(obs) < 1e6), f"Explosion at step {step}: max={np.max(np.abs(obs))}"
        if terminated or truncated:
            obs, _ = env.reset()

def test_puck_friction_decay(env):
    """Success criterion 2: ice friction causes visible velocity decay."""
    obs, _ = env.reset(seed=0)
    # Manually set puck to a known velocity, then check it decays
    # (requires direct physics access)
    physics = env.unwrapped._dm_env.physics
    physics.named.data.qvel['puck/free_x'] = 5.0
    physics.named.data.qvel['puck/free_y'] = 0.0
    initial_speed = 5.0
    for _ in range(50):   # ~0.25 seconds at 0.005s timestep
        env.step(np.zeros(4))
    final_speed = np.linalg.norm([
        physics.named.data.qvel['puck/free_x'],
        physics.named.data.qvel['puck/free_y']
    ])
    assert final_speed < initial_speed * 0.9, \
        f"Puck speed did not decay: {initial_speed:.2f} -> {final_speed:.2f}"

def test_board_bounce_angle(env):
    """Success criterion 2: puck rebounds off boards at physically plausible angle."""
    # Set puck heading toward right board wall at 45 degrees
    # After bounce, x-velocity should flip sign; y-velocity should be approximately preserved
    pass  # Implementation-specific; fill in with actual physics access
```

### Sampling Rate

- **Per task commit:** `pytest tests/ -x -q` (quick: ~10 seconds)
- **Per wave merge:** `pytest tests/ -v --tb=short` (full: ~30 seconds)
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/conftest.py` — shared `env` fixture, `make_env(agent_idx)` factory
- [ ] `tests/test_physics.py` — physics stability tests (ENV-01, ENV-02, ENV-03, SC-1, SC-2)
- [ ] `tests/test_observations.py` — obs spec integrity and value tracking (ENV-05, ENV-06)
- [ ] `tests/test_rewards.py` — reward component isolation (ENV-07, SC-3)
- [ ] `tests/test_gymnasium.py` — API compliance via check_env (ENV-04, Gymnasium compatibility)
- [ ] `pytest.ini` — configure testpaths and markers
- [ ] Python venv with required packages — `dm-control==1.0.38`, `gymnasium`, `pytest`

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| mujoco-py | dm_control + `mujoco` Python bindings | MuJoCo 2.0 open-source release (2021) | mujoco-py is deprecated; dm_control is the official successor |
| `dm_control.rl.control.Environment` | `dm_control.composer.Environment` | dm_control 0.x → 1.x | Composer is the standard for custom multi-entity environments |
| `Euler` integrator | `implicitfast` integrator | MuJoCo 3.x | More stable for stiff contacts; recommended in MuJoCo 3.x docs |
| Pyramidal friction cone | Elliptic friction cone | MuJoCo 2.x+ | Elliptic is more stable; confirmed in maintainer discussions |
| gym.Env (old Gym) | gymnasium.Env | Gymnasium 0.26+ | `reset()` returns `(obs, info)`; `step()` returns 5 values; SB3 2.x requires Gymnasium |

**Deprecated/outdated:**
- `mujoco-py`: Deprecated, no MuJoCo 3.x support. Do not use.
- Old Gym API (`gym.Env` from `gym` package): `step()` returned 4 values. SB3 2.7.1 requires the Gymnasium API.
- `dm_control.rl.control`: Only for single-entity locomotion domains; not for custom multi-agent environments.

---

## Open Questions

1. **Python 3.11 vs 3.12 for Phase 1**
   - What we know: System has Python 3.12.3. dm_control 1.0.38 lists Python 3.10-3.12 support. SB3 2.7.1 supports 3.10-3.12.
   - What's unclear: Whether the "SB3 edge-case issues with 3.12" mentioned in STACK.md affect Phase 1 environment code (which doesn't use SB3 directly).
   - Recommendation: Use Python 3.12 for Phase 1 (environment only). The edge cases are in training integration, not environment construction. Switch to 3.11 on RunPod for Phase 2.

2. **Freejoint vs slide+hinge for puck**
   - What we know: A `freejoint` gives the puck 6 DOF including 3D rotation (puck can flip). Restricting to 2D requires either a `freejoint` with equality constraints or separate slide/hinge joints.
   - What's unclear: Whether MuJoCo 3.6 equality constraints on freejoint DOFs are stable enough, or whether the slide+hinge approach is simpler.
   - Recommendation: Use three explicit joints (`slide x`, `slide y`, `hinge z`) instead of a freejoint. This prevents 3D puck flipping entirely and removes the degenerate contact case.

3. **Exact contact bits for puck-agent interactions**
   - What we know: `contype`/`conaffinity` bitmask filtering can isolate puck-board, puck-stick, and prevent agent-agent and agent-board contacts if desired.
   - What's unclear: Whether agents should collide with boards (prevents agents from going out of bounds via physics vs. reward penalty). For simplicity, agents may clip through boards if their movement is velocity-actuated within joint limits.
   - Recommendation: Add joint limits to agent x/y slide joints to prevent wall exit. This is simpler than collision-based boundary enforcement and avoids agent-board contact forces entirely.

4. **Obs vector: world frame vs agent-egocentric**
   - What we know: ARCHITECTURE.md specifies world-frame positions. dm_control soccer uses egocentric (rotated) observations computed via framepos/framelinvel sensors. Egocentric is more rotation-invariant and better for a shared policy.
   - What's unclear: Whether the JS physics mirror can correctly replicate egocentric rotation transforms.
   - Recommendation: Start with world frame as specified in ARCHITECTURE.md. Mark this as a v2 upgrade candidate if training reveals poor directional behavior. World frame is simpler to replicate in JS.

---

## Sources

### Primary (HIGH confidence)
- dm_control GitHub source — `soccer/pitch.py`, `soccer/task.py`, `soccer/soccer_ball.py`, `soccer/observables.py`, `soccer/task_test.py` — direct source code analysis
- dm_control GitHub source — `composer/environment.py` — Environment class interface, Task-Environment relationship
- dm_control.utils.rewards — tolerance() function, sigmoid types — verified from source
- ARCHITECTURE.md (this project) — Interface 1-3 obs/action/norm spec, component hierarchy
- STACK.md (this project) — Version pins: dm_control 1.0.38, MuJoCo 3.6.0

### Secondary (MEDIUM confidence)
- MuJoCo GitHub Discussion #2081 — coefficient of restitution via solref: `solref[0]` = constraint timescale, `damp_ratio` controls energy dissipation
- MuJoCo GitHub Discussion #2347 — elastic simulation stability: `cone="elliptic"` confirmed to improve bounce stability; `integrator="RK4"` or `implicitfast` recommended
- Shimmy documentation — `DmControlCompatibilityV0` wrapper source: confirmed obs dict structure, `dm_spec2gym_space()` conversion
- SB3 documentation — confirmed `info` dict custom metrics via `BaseCallback.logger.record()`

### Tertiary (LOW confidence — requires validation)
- WebSearch: Python 3.12 SB3 edge cases — not verified against specific SB3 release notes; treat as risk to validate in Phase 2
- WebSearch: `integrator="implicitfast"` availability in MuJoCo 3.6 — mentioned in 3.x docs but specific availability in 3.6 not confirmed; fallback to `"Euler"` if not available

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — versions verified via STACK.md which was verified against PyPI/GitHub
- Architecture patterns: HIGH — based directly on dm_control soccer source code analysis
- Puck contact tuning: MEDIUM — MuJoCo contact params are empirical; specific values require in-simulation validation
- Obs spec layout: HIGH — derived from ARCHITECTURE.md Interface 1 and project requirements
- Test patterns: HIGH — based directly on dm_control soccer test source code patterns

**Research date:** 2026-03-28
**Valid until:** 2026-04-28 (stable libraries; dm_control 1.0.38 and MuJoCo 3.6.0 are pinned)
