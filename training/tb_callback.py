"""TensorBoardCustomCallback — logs hockey-specific metrics to TensorBoard.

Records per-rollout aggregate metrics beyond SB3's default logging:
  - hockey/goal_rate: fraction of completed episodes in which at least one
    goal was scored (proxy for game-state engagement quality)
  - hockey/puck_possession_rate: fraction of completed episodes where the
    agent had positive r_possession reward (proxy for puck-touching behavior)

Decision references: D-13 from 02-CONTEXT.md, TRAIN-04
"""

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class TensorBoardCustomCallback(BaseCallback):
    """Log hockey/goal_rate and hockey/puck_possession_rate each rollout.

    Accumulates per-episode binary signals across all steps in the rollout,
    then reports their means on _on_rollout_end.

    Args:
        verbose: Verbosity level. 0 = silent.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self._episode_goals: list[float] = []
        self._episode_possessions: list[float] = []

    def _on_step(self) -> bool:
        """Collect per-episode metrics from completed episodes in this step."""
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", np.zeros(0, dtype=bool))

        for info, done in zip(infos, dones):
            if done:
                score = info.get("score", [0, 0])
                goals = sum(score)
                self._episode_goals.append(float(goals > 0))
                self._episode_possessions.append(
                    float(info.get("r_possession", 0.0) > 0)
                )

        return True

    def _on_rollout_end(self) -> None:
        """Report aggregated metrics to TensorBoard at rollout end."""
        if self._episode_goals:
            self.logger.record(
                "hockey/goal_rate", float(np.mean(self._episode_goals))
            )
        if self._episode_possessions:
            self.logger.record(
                "hockey/puck_possession_rate",
                float(np.mean(self._episode_possessions)),
            )
        self._episode_goals.clear()
        self._episode_possessions.clear()
