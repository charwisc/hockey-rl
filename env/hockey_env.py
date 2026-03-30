"""Gymnasium wrapper for 2v2 Hockey dm_control environment.

Hand-written (not Shimmy) for explicit control over the observation vector
layout, which is the cross-boundary contract for ONNX export and JS mirror.
"""

import gymnasium as gym
import numpy as np
from dm_control import composer

from env.hockey_arena import HockeyArena
from env.hockey_player import HockeyPlayer
from env.hockey_puck import HockeyPuck
from env.hockey_task import HockeyTask, MAX_SPEED, MAX_ROT_SPEED
from env.obs_spec import OBS_DIM


class HockeyEnv(gym.Env):
    """Single-agent Gymnasium view of the 2v2 hockey game.

    Args:
        agent_idx: Which agent (0-3) this env controls.
            0,1 = team 0; 2,3 = team 1.
        frozen_opponent_fn: Callable(obs_array) -> action_array for
            non-controlled agents. Defaults to random actions.
        time_limit: Episode time limit in seconds. Default 60.
    """

    metadata = {"render_modes": []}

    def __init__(self, agent_idx: int = 0, frozen_opponent_fn=None,
                 time_limit: float = 60.0):
        super().__init__()
        self._agent_idx = agent_idx

        # Opponent path and model-caching for set_attr bridge (SubprocVecEnv)
        self._opponent_path: str | None = None
        self._opponent_model_cache: dict = {"path": None, "model": None}
        self._use_external_opponent = frozen_opponent_fn is not None
        self._frozen_opponent_fn = frozen_opponent_fn or self._default_opponent_fn

        # Build dm_control environment
        self._arena = HockeyArena()
        self._players = [
            HockeyPlayer(team=t, player_idx=p)
            for t in range(2) for p in range(2)
        ]
        self._puck = HockeyPuck()
        self._task = HockeyTask(
            self._arena, self._players, self._puck,
            time_limit=time_limit)
        self._dm_env = composer.Environment(
            self._task, time_limit=time_limit)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(OBS_DIM,), dtype=np.float32)

        # Action: [move_x, move_y, speed, stick_angle] all in [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset episode. Returns (obs, info) per Gymnasium API."""
        super().reset(seed=seed)
        self._dm_env.reset()
        obs = self._task.build_obs_for_agent(
            self._dm_env.physics, self._agent_idx)
        return obs, {}

    def step(self, action):
        """Step with 4-float action for this agent.

        Returns (obs, reward, terminated, truncated, info) per Gymnasium API.
        """
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)

        # Build full 12-float actuator control array for all 4 agents
        all_controls = self._build_all_controls(action)

        # Step dm_control environment
        self._dm_env.step(all_controls)

        # Get obs and reward for this agent
        obs = self._task.build_obs_for_agent(
            self._dm_env.physics, self._agent_idx)

        # NaN guard: if NaN detected, reset and return zero reward
        if np.any(np.isnan(obs)):
            obs, _ = self.reset()
            return obs, 0.0, True, False, {'nan_reset': True}

        reward_components = self._task.get_reward_components(
            self._dm_env.physics, self._agent_idx)
        reward = sum(reward_components.values())

        terminated = self._task.should_terminate_episode(
            self._dm_env.physics)
        truncated = False

        info = dict(reward_components)
        info['score'] = self._task.score

        return obs, float(reward), terminated, truncated, info

    def close(self):
        """Close the dm_control environment."""
        if hasattr(self, '_dm_env'):
            self._dm_env.close()

    def _build_all_controls(self, agent_action: np.ndarray) -> np.ndarray:
        """Map 4-float agent action to 12-float actuator array for all agents.

        Each agent's 4-float action [move_x, move_y, speed, stick_angle]
        maps to 3 actuator targets [vx_target, vy_target, vrot_target]:
          vx_target = move_x * speed * MAX_SPEED
          vy_target = move_y * speed * MAX_SPEED
          vrot_target = stick_angle * MAX_ROT_SPEED
        """
        controls = np.zeros(12, dtype=np.float32)  # 3 actuators x 4 players

        for i in range(4):
            if i == self._agent_idx:
                act = agent_action
            else:
                # Get action from frozen opponent/teammate
                other_obs = self._task.build_obs_for_agent(
                    self._dm_env.physics, i)
                act = self._frozen_opponent_fn(other_obs)
                act = np.asarray(act, dtype=np.float32).clip(-1.0, 1.0)

            # Map 4-float to 3 actuator targets
            move_x, move_y, speed, stick_angle = act[0], act[1], act[2], act[3]
            speed_scaled = (speed + 1.0) / 2.0  # map [-1,1] to [0,1]
            controls[i * 3 + 0] = move_x * speed_scaled * MAX_SPEED
            controls[i * 3 + 1] = move_y * speed_scaled * MAX_SPEED
            controls[i * 3 + 2] = stick_angle * MAX_ROT_SPEED

        return controls

    @property
    def opponent_path(self) -> "str | None":
        """Current opponent checkpoint path used by the model-caching closure."""
        return self._opponent_path

    @opponent_path.setter
    def opponent_path(self, value: "str | None") -> None:
        """Set opponent path. SubprocVecEnv.set_attr('opponent_path', path)
        calls this setter in each worker process."""
        self._opponent_path = value

    def _default_opponent_fn(self, obs: np.ndarray) -> np.ndarray:
        """Opponent function driven by self._opponent_path with model caching.

        - If path is None or "random": returns random action.
        - If path matches cache: reuses loaded model (no re-load).
        - Otherwise: loads PPO from path, updates cache.
        """
        path = self._opponent_path
        if path is None or path == "random":
            return np.random.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)
        if path != self._opponent_model_cache["path"]:
            from stable_baselines3 import PPO
            self._opponent_model_cache["model"] = PPO.load(path, device="cpu")
            self._opponent_model_cache["path"] = path
        action, _ = self._opponent_model_cache["model"].predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    @staticmethod
    def _random_action(obs: np.ndarray) -> np.ndarray:
        """Fallback static random action (kept for backward compatibility)."""
        return np.random.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)
