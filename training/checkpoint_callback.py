"""WallTimeCheckpointCallback — saves policy + VecNormalize on wall-clock interval.

Saves step_{N}.zip (model weights) and step_{N}_vecnorm.pkl (observation/reward
normalization statistics) every `interval_minutes` minutes of wall-clock time.
Checkpoint names encode the total env step count for easy post-hoc analysis.

Decision references: D-09, D-10, D-11, D-12 from 02-CONTEXT.md
"""

import os
import time

from stable_baselines3.common.callbacks import BaseCallback


class WallTimeCheckpointCallback(BaseCallback):
    """Save model checkpoint + VecNormalize stats every N wall-clock minutes.

    Args:
        checkpoint_dir: Directory to write .zip and .pkl files.
            Default "/workspace/checkpoints" matches RunPod persistent volume.
        interval_minutes: How often to checkpoint in wall-clock minutes.
            Default 30.0.
        verbose: Verbosity level. 0 = silent, 1 = print on each save.
    """

    def __init__(
        self,
        checkpoint_dir: str = "/workspace/checkpoints",
        interval_minutes: float = 30.0,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir
        self.interval_seconds = interval_minutes * 60.0
        self._last_save_time: float = 0.0
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _on_training_start(self) -> None:
        """Record wall-clock time at training start to anchor the interval."""
        self._last_save_time = time.time()

    def _on_step(self) -> bool:
        """Check wall-clock elapsed time; save if interval exceeded."""
        now = time.time()
        if now - self._last_save_time >= self.interval_seconds:
            self._save()
            self._last_save_time = now
        return True

    def _save(self) -> None:
        """Write model .zip and VecNormalize .pkl to checkpoint_dir."""
        step = self.num_timesteps
        model_path = os.path.join(self.checkpoint_dir, f"step_{step}.zip")
        vecnorm_path = os.path.join(self.checkpoint_dir, f"step_{step}_vecnorm.pkl")

        self.model.save(model_path)

        vec_norm = self.model.get_vec_normalize_env()
        if vec_norm is not None:
            vec_norm.save(vecnorm_path)

        if self.verbose:
            print(f"[Checkpoint] step={step} saved to {model_path}")
