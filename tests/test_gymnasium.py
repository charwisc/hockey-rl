"""Gymnasium API compliance tests."""
import pytest
import numpy as np
from env.hockey_env import HockeyEnv


def test_action_space_spec(env):
    """ENV-04: Action space shape==(4,), dtype==float32, bounds [-1, 1]."""
    assert env.action_space.shape == (4,)
    assert env.action_space.dtype == np.float32
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)


def test_check_env_passes():
    """Gymnasium check_env() passes with zero warnings."""
    from gymnasium.utils.env_checker import check_env
    env = HockeyEnv(agent_idx=0)
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()
