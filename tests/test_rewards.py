"""Reward component isolation tests."""
import pytest


@pytest.mark.xfail(reason="HockeyEnv not yet implemented")
def test_sparse_goal_reward(env):
    """ENV-07: r_goal == 10.0 when puck crosses goal line."""
    pass  # Implemented in Plan 03


@pytest.mark.xfail(reason="HockeyEnv not yet implemented")
def test_puck_toward_goal_fires_with_possession(env):
    """ENV-07: r_puck_toward_goal > 0 when agent has possession and puck moves toward goal."""
    pass  # Implemented in Plan 03


@pytest.mark.xfail(reason="HockeyEnv not yet implemented")
def test_puck_toward_goal_gated_on_possession(env):
    """ENV-07: r_puck_toward_goal == 0 when agent does NOT have possession."""
    pass  # Implemented in Plan 03


@pytest.mark.xfail(reason="HockeyEnv not yet implemented")
def test_reward_components_in_info(env):
    """ENV-07: All 6 reward keys present in info dict every step."""
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    expected_keys = {'r_goal', 'r_puck_toward_goal', 'r_possession',
                     'r_positioning', 'r_clustering', 'r_step_penalty'}
    assert expected_keys.issubset(info.keys()), \
        f"Missing keys: {expected_keys - set(info.keys())}"


@pytest.mark.xfail(reason="HockeyEnv not yet implemented")
def test_reward_independent_extraction(env):
    """SC-3: Sparse and shaped rewards independently extractable from info."""
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    # Each reward component must be a plain float
    for key in ['r_goal', 'r_puck_toward_goal', 'r_possession',
                'r_positioning', 'r_clustering', 'r_step_penalty']:
        assert isinstance(info[key], float), f"{key} is not float"
