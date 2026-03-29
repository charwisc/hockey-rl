"""Gymnasium API compliance tests."""
import pytest
import numpy as np


@pytest.mark.xfail(reason="HockeyEnv not yet implemented")
def test_action_space_spec(env):
    """ENV-04: Action space shape==(4,), dtype==float32, bounds [-1, 1]."""
    assert env.action_space.shape == (4,)
    assert env.action_space.dtype == np.float32
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)


@pytest.mark.xfail(reason="HockeyEnv not yet implemented")
def test_check_env_passes(env):
    """Gymnasium check_env() passes with zero warnings."""
    from gymnasium.utils.env_checker import check_env
    check_env(env.unwrapped, skip_render_check=True)
