"""SelfPlayPoolCallback — manages opponent checkpoint pool during PPO training.

On every `pool_update_freq` timesteps:
  1. Saves current policy to pool_dir as pool_step_{step}.zip
  2. Evicts oldest checkpoints when pool exceeds max_pool_size
  3. Reassigns each SubprocVecEnv worker to a random pool checkpoint via set_attr

The set_attr("opponent_path", path, indices=[i]) call reaches HockeyEnv.opponent_path
setter in the worker subprocess, which uses the model-caching closure to load
the checkpoint lazily on the next step.

Decision references: D-01, D-02, D-03, D-04 from 02-CONTEXT.md
"""

import os
import random
from collections import deque

from stable_baselines3.common.callbacks import BaseCallback


class SelfPlayPoolCallback(BaseCallback):
    """Snapshot current policy to an opponent pool and reassign worker opponents.

    Args:
        pool_dir: Directory to write pool checkpoint .zip files.
        pool_update_freq: Fire every N total timesteps. Default 500_000.
        max_pool_size: Maximum number of checkpoints to retain. Oldest is
            evicted when exceeded. Default 20.
        verbose: Verbosity level. 0 = silent, 1 = print on each snapshot.
    """

    def __init__(
        self,
        pool_dir: str,
        pool_update_freq: int = 500_000,
        max_pool_size: int = 20,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.verbose = verbose
        self.pool_dir = pool_dir
        self.pool_update_freq = pool_update_freq
        self.max_pool_size = max_pool_size
        self._pool: deque = deque()
        self._latest_path: str | None = None
        os.makedirs(pool_dir, exist_ok=True)

    def _on_step(self) -> bool:
        """Check whether it's time to snapshot + reassign. Always returns True."""
        if self.num_timesteps % self.pool_update_freq < self.training_env.num_envs:
            self._snapshot_and_reassign()
        return True

    def _snapshot_and_reassign(self) -> None:
        """Save current policy to pool, evict oldest if needed, reassign opponents."""
        step = self.num_timesteps
        path = os.path.join(self.pool_dir, f"pool_step_{step}.zip")

        if not os.path.exists(path):
            # Save model; SB3 model.save appends .zip automatically if not present,
            # but we pass the full path with .zip so it saves exactly here.
            self.model.save(path)

        self._pool.append(path)
        self._latest_path = path

        # Evict oldest checkpoints beyond max_pool_size
        while len(self._pool) > self.max_pool_size:
            old = self._pool.popleft()
            if os.path.exists(old):
                os.remove(old)

        # Reassign opponents for each worker env
        pool_list = list(self._pool)
        for i in range(self.training_env.num_envs):
            # 50% chance of latest policy; otherwise sample historical pool
            if random.random() < 0.5 or len(pool_list) <= 1:
                chosen = self._latest_path
            else:
                chosen = random.choice(pool_list[:-1])
            self.training_env.set_attr("opponent_path", chosen, indices=[i])

        if self.verbose:
            print(f"[SelfPlay] step={step} pool_size={len(self._pool)}")
