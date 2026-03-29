"""Hockey 2v2 task: observation assembly, reward computation, episode management."""

import numpy as np
from dm_control import composer
from dm_control.utils import rewards as dmr

from env.obs_spec import OBS_SPEC, OBS_DIM

# Action mapping constants
MAX_SPEED = 5.0        # m/s max agent speed
MAX_ROT_SPEED = 3.0    # rad/s max rotation speed
POSSESSION_DIST = 0.5  # meters: stick-puck distance threshold for possession

# Face-off positions (world frame)
FACEOFF_POSITIONS = {
    # (team, player_idx): (x, y)
    (0, 0): (-5.0, 2.0),
    (0, 1): (-5.0, -2.0),
    (1, 0): (5.0, 2.0),
    (1, 1): (5.0, -2.0),
}


class HockeyTask(composer.Task):
    """2v2 Hockey task managing observation assembly, reward computation, and episode lifecycle.

    Players list convention: [team0_p0, team0_p1, team1_p0, team1_p1]
    - agent_idx 0 = team 0, player 0
    - agent_idx 1 = team 0, player 1
    - agent_idx 2 = team 1, player 0
    - agent_idx 3 = team 1, player 1
    """

    def __init__(self, arena, players, puck, time_limit=60.0):
        self._arena = arena
        self._players = players   # list of 4: [team0_p0, team0_p1, team1_p0, team1_p1]
        self._puck = puck
        self._time_limit = time_limit
        self._score = [0, 0]      # [team0, team1]
        self._goal_scored_this_step = False
        self._entities_attached = False

    @property
    def root_entity(self):
        return self._arena

    def initialize_episode_mjcf(self, random_state):
        """Attach entities to arena. Called on each reset; guard prevents double-attach."""
        if self._entities_attached:
            return
        for player in self._players:
            self._arena.attach(player)
        self._arena.attach(self._puck)
        self._entities_attached = True

    def initialize_episode(self, physics, random_state):
        """Reset to face-off: players to starting positions, puck to center."""
        self._score = [0, 0]
        self._goal_scored_this_step = False

        # Reset puck to center with zero velocity
        puck_x = self._puck.mjcf_model.find('joint', 'puck_x')
        puck_y = self._puck.mjcf_model.find('joint', 'puck_y')
        puck_rot = self._puck.mjcf_model.find('joint', 'puck_rot')
        physics.bind(puck_x).qpos = 0.0
        physics.bind(puck_y).qpos = 0.0
        physics.bind(puck_rot).qpos = 0.0
        physics.bind(puck_x).qvel = 0.0
        physics.bind(puck_y).qvel = 0.0
        physics.bind(puck_rot).qvel = 0.0

        # Reset players to face-off positions
        for player in self._players:
            team = player.team
            pidx = player.player_idx
            fx, fy = FACEOFF_POSITIONS[(team, pidx)]
            x_joint = player.mjcf_model.find('joint', 'x')
            y_joint = player.mjcf_model.find('joint', 'y')
            rot_joint = player.mjcf_model.find('joint', 'rot')
            physics.bind(x_joint).qpos = fx
            physics.bind(y_joint).qpos = fy
            physics.bind(rot_joint).qpos = 0.0
            physics.bind(x_joint).qvel = 0.0
            physics.bind(y_joint).qvel = 0.0
            physics.bind(rot_joint).qvel = 0.0

    def before_step(self, physics, action, random_state):
        """Map 12-float action array to 12 actuator controls.

        action shape: (12,) — 3 actuator targets per player x 4 players.
        The HockeyEnv wrapper handles the per-agent 4-float -> 3-actuator mapping and
        concatenation before calling this. Here we just apply the 12-float array directly.
        """
        self._goal_scored_this_step = False
        if action is not None and len(action) > 0:
            physics.set_control(action)

    def after_step(self, physics, random_state):
        """Check for goal scoring after physics step."""
        puck_x = float(physics.bind(
            self._puck.mjcf_model.find('joint', 'puck_x')).qpos[0])
        puck_y = float(physics.bind(
            self._puck.mjcf_model.find('joint', 'puck_y')).qpos[0])

        # Goal detection: puck crosses goal line within goal width
        goal_half_width = 2.0
        rink_half = self._arena.rink_length / 2

        # Home goal (team 0 defends, x = -rink_half): puck at x < -rink_half + 1.0
        if puck_x < -rink_half + 1.0 and abs(puck_y) < goal_half_width:
            self._score[1] += 1  # team 1 scores
            self._goal_scored_this_step = True

        # Away goal (team 1 defends, x = +rink_half): puck at x > rink_half - 1.0
        if puck_x > rink_half - 1.0 and abs(puck_y) < goal_half_width:
            self._score[0] += 1  # team 0 scores
            self._goal_scored_this_step = True

    def get_observation(self, physics):
        """Return empty dict — per-agent obs assembly happens in HockeyEnv wrapper."""
        return {}

    def get_reward(self, physics):
        """Single scalar reward for the dm_env TimeStep (not used directly by training)."""
        return 0.0

    def should_terminate_episode(self, physics):
        """End on goal or time limit."""
        return (self._goal_scored_this_step or
                physics.time() >= self._time_limit)

    # ---- Per-agent observation and reward methods (called by HockeyEnv) ----

    def build_obs_for_agent(self, physics, agent_idx: int) -> np.ndarray:
        """Build the 22-float egocentric obs vector for one agent.

        Uses OBS_SPEC slice layout from obs_spec.py to ensure index stability.

        Index assignments (hardcoded to match OBS_SPEC exactly):
          [0:2]   agent_pos       (world_xy meters)
          [2:4]   agent_vel       (world_xy m/s)
          [4:6]   teammate_pos    (world_xy meters)
          [6:8]   teammate_vel    (world_xy m/s)
          [8:10]  opponent0_pos   (world_xy meters)
          [10:12] opponent0_vel   (world_xy m/s)
          [12:14] opponent1_pos   (world_xy meters)
          [14:16] opponent1_vel   (world_xy m/s)
          [16:18] puck_pos        (world_xy meters)
          [18:20] puck_vel        (world_xy m/s)
          [20]    stick_angle     RESERVED/always-zero in v1.0.0 (no independent stick joint)
          [21]    facing_angle    (world radians)
        """
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        player = self._players[agent_idx]

        # Agent's own joints
        x_joint = player.mjcf_model.find('joint', 'x')
        y_joint = player.mjcf_model.find('joint', 'y')
        rot_joint = player.mjcf_model.find('joint', 'rot')

        # obs[0:2] = agent_pos
        # physics.bind(joint).qpos returns a SynchronizingArrayWrapper of shape (1,)
        obs[0] = physics.bind(x_joint).qpos[0]
        obs[1] = physics.bind(y_joint).qpos[0]
        # obs[2:4] = agent_vel
        obs[2] = physics.bind(x_joint).qvel[0]
        obs[3] = physics.bind(y_joint).qvel[0]

        # Teammate: same team, other player index
        teammate_idx = self._get_teammate_idx(agent_idx)
        tm = self._players[teammate_idx]
        tm_x = tm.mjcf_model.find('joint', 'x')
        tm_y = tm.mjcf_model.find('joint', 'y')
        # obs[4:6] = teammate_pos
        obs[4] = physics.bind(tm_x).qpos[0]
        obs[5] = physics.bind(tm_y).qpos[0]
        # obs[6:8] = teammate_vel
        obs[6] = physics.bind(tm_x).qvel[0]
        obs[7] = physics.bind(tm_y).qvel[0]

        # Opponents (indices 8:16)
        opp_indices = self._get_opponent_indices(agent_idx)
        for i, opp_idx in enumerate(opp_indices):
            opp = self._players[opp_idx]
            opp_x = opp.mjcf_model.find('joint', 'x')
            opp_y = opp.mjcf_model.find('joint', 'y')
            base = 8 + i * 4  # opp0: [8:12], opp1: [12:16]
            obs[base]     = physics.bind(opp_x).qpos[0]
            obs[base + 1] = physics.bind(opp_y).qpos[0]
            obs[base + 2] = physics.bind(opp_x).qvel[0]
            obs[base + 3] = physics.bind(opp_y).qvel[0]

        # Puck
        puck_x = self._puck.mjcf_model.find('joint', 'puck_x')
        puck_y = self._puck.mjcf_model.find('joint', 'puck_y')
        # obs[16:18] = puck_pos
        obs[16] = physics.bind(puck_x).qpos[0]
        obs[17] = physics.bind(puck_y).qpos[0]
        # obs[18:20] = puck_vel
        obs[18] = physics.bind(puck_x).qvel[0]
        obs[19] = physics.bind(puck_y).qvel[0]

        # obs[20] = stick_angle — RESERVED: always-zero in v1.0.0
        # No independent stick joint exists. The 4-float action's stick_angle input
        # controls body vrot, not a separate stick. Downstream consumers (ONNX export,
        # JS mirror) must not use this dimension for physics mirroring.
        obs[20] = 0.0

        # obs[21] = facing_angle (agent heading in world frame)
        agent_facing = float(physics.bind(rot_joint).qpos[0])
        obs[21] = agent_facing

        return obs

    def get_reward_components(self, physics, agent_idx: int) -> dict:
        """Compute all 6 reward components for one agent.

        Returns dict with keys: r_goal, r_puck_toward_goal, r_possession,
        r_positioning, r_clustering, r_step_penalty.

        All values are Python float (not np.float64) for JSON serialization.
        """
        player = self._players[agent_idx]
        team = player.team

        # Agent position
        x_joint = player.mjcf_model.find('joint', 'x')
        y_joint = player.mjcf_model.find('joint', 'y')
        agent_pos = np.array([
            physics.bind(x_joint).qpos[0],
            physics.bind(y_joint).qpos[0]
        ])

        # Puck position and velocity
        puck_x_j = self._puck.mjcf_model.find('joint', 'puck_x')
        puck_y_j = self._puck.mjcf_model.find('joint', 'puck_y')
        puck_pos = np.array([
            physics.bind(puck_x_j).qpos[0],
            physics.bind(puck_y_j).qpos[0]
        ])
        puck_vel = np.array([
            physics.bind(puck_x_j).qvel[0],
            physics.bind(puck_y_j).qvel[0]
        ])

        # Opponent goal position (where this agent wants to score)
        rink_half = self._arena.rink_length / 2
        if team == 0:
            opponent_goal = np.array([rink_half, 0.0])   # team 0 attacks +x
        else:
            opponent_goal = np.array([-rink_half, 0.0])  # team 1 attacks -x

        # --- r_goal: sparse +10 for own team scoring, -10 for opponent scoring ---
        r_goal = 0.0
        if self._goal_scored_this_step:
            if team == 0 and self._score[0] > 0:
                r_goal = 10.0
            elif team == 1 and self._score[1] > 0:
                r_goal = 10.0
            # Penalize own-goal (opponent scores)
            if team == 0 and self._score[1] > 0:
                r_goal = -10.0
            elif team == 1 and self._score[0] > 0:
                r_goal = -10.0

        # --- Possession check: is this agent's stick tip close to the puck? ---
        # Stick tip approximation: agent_pos + 0.4m in facing direction
        agent_facing = float(physics.bind(
            player.mjcf_model.find('joint', 'rot')).qpos[0])
        stick_tip = agent_pos + 0.4 * np.array([
            np.cos(agent_facing), np.sin(agent_facing)])
        has_possession = np.linalg.norm(stick_tip - puck_pos) < POSSESSION_DIST

        # --- r_puck_toward_goal: gated on possession (prevents proxy exploit) ---
        # Gating prevents agents from shadowing puck velocity without controlling it.
        goal_dir = opponent_goal - puck_pos
        goal_dist = np.linalg.norm(goal_dir)
        goal_dir = goal_dir / (goal_dist + 1e-8)
        puck_toward = float(np.dot(puck_vel, goal_dir))
        r_puck_toward_goal = np.clip(puck_toward, -1.0, 1.0) * 0.05 if has_possession else 0.0

        # --- r_possession: small bonus for being near the puck ---
        r_possession = 0.01 if has_possession else 0.0

        # --- r_positioning: tolerance-based reward for being near ideal support position ---
        # Ideal: midpoint between own goal and puck (defensive support / coverage)
        own_goal = np.array([-rink_half if team == 0 else rink_half, 0.0])
        ideal_pos = 0.5 * (own_goal + puck_pos)
        dist_to_ideal = float(np.linalg.norm(agent_pos - ideal_pos))
        r_positioning = float(dmr.tolerance(
            dist_to_ideal, bounds=(0, 3.0), margin=5.0)) * 0.005

        # --- r_clustering: penalize being too close to teammate ---
        teammate_idx = self._get_teammate_idx(agent_idx)
        tm = self._players[teammate_idx]
        tm_pos = np.array([
            physics.bind(tm.mjcf_model.find('joint', 'x')).qpos[0],
            physics.bind(tm.mjcf_model.find('joint', 'y')).qpos[0]
        ])
        teammate_dist = float(np.linalg.norm(agent_pos - tm_pos))
        r_clustering = -0.01 if teammate_dist < 2.0 else 0.0

        # --- r_step_penalty: constant per-step cost to discourage passive play ---
        r_step_penalty = -0.001

        return {
            'r_goal': float(r_goal),
            'r_puck_toward_goal': float(r_puck_toward_goal),
            'r_possession': float(r_possession),
            'r_positioning': float(r_positioning),
            'r_clustering': float(r_clustering),
            'r_step_penalty': float(r_step_penalty),
        }

    def get_total_reward(self, physics, agent_idx: int) -> float:
        """Sum of all reward components for one agent."""
        return sum(self.get_reward_components(physics, agent_idx).values())

    # ---- Helpers ----

    def _get_teammate_idx(self, agent_idx: int) -> int:
        """Return the index of the teammate.

        Team 0: agents 0 and 1 are teammates (0 <-> 1)
        Team 1: agents 2 and 3 are teammates (2 <-> 3)
        """
        if agent_idx < 2:
            return 1 - agent_idx
        else:
            return 5 - agent_idx  # 2 -> 3, 3 -> 2

    def _get_opponent_indices(self, agent_idx: int) -> list:
        """Return indices of both opponents."""
        if agent_idx < 2:
            return [2, 3]
        else:
            return [0, 1]

    @property
    def score(self):
        return list(self._score)

    @property
    def goal_scored_this_step(self):
        return self._goal_scored_this_step
