"""Physics stability tests for the hockey environment."""
import pytest
import numpy as np


@pytest.mark.xfail(reason="HockeyEnv not yet implemented")
def test_arena_compiles(env):
    """ENV-01: Arena XML compiles; boards, goals, face-off sites exist."""
    obs, _ = env.reset()
    physics = env.unwrapped._dm_env.physics
    # Verify arena geoms exist
    assert physics.named.model.geom_type['ice'] is not None
    assert physics.named.model.site_size['home_goal'] is not None
    assert physics.named.model.site_size['away_goal'] is not None


@pytest.mark.xfail(reason="HockeyEnv not yet implemented")
def test_agents_load(env):
    """ENV-02: 4 capsule agents load with stick geoms and actuators."""
    physics = env.unwrapped._dm_env.physics
    for i in range(4):
        team = i // 2
        idx = i % 2
        prefix = f"player_{team}_{idx}"
        # Capsule geom exists
        # Stick geom exists
        # 3 actuators (vx, vy, vrot) exist


@pytest.mark.xfail(reason="HockeyEnv not yet implemented")
def test_puck_stability_1000steps(env):
    """ENV-03 + SC-1: 1000 steps with random actions, no NaN, no explosion."""
    obs, _ = env.reset(seed=42)
    assert not np.any(np.isnan(obs)), "NaN in initial obs"
    for step in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert not np.any(np.isnan(obs)), f"NaN at step {step}"
        assert not np.isnan(reward), f"NaN reward at step {step}"
        assert np.all(np.abs(obs) < 1e6), f"Explosion at step {step}"
        if terminated or truncated:
            obs, _ = env.reset()


@pytest.mark.xfail(reason="HockeyEnv not yet implemented")
def test_board_bounce_angle(env):
    """ENV-03: Puck rebounds off board wall; reflected angle within tolerance."""
    pass  # Requires direct physics manipulation — implemented in Plan 02


@pytest.mark.xfail(reason="HockeyEnv not yet implemented")
def test_puck_friction_decay(env):
    """ENV-03: Puck velocity decays due to ice friction when no force applied."""
    pass  # Requires direct physics manipulation — implemented in Plan 02


@pytest.mark.xfail(reason="HockeyEnv not yet implemented")
def test_1000_steps_no_nan(env):
    """SC-1: Alias for puck_stability_1000steps with explicit NaN focus."""
    obs, _ = env.reset(seed=42)
    for step in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert not np.any(np.isnan(obs)), f"NaN at step {step}"
        if terminated or truncated:
            obs, _ = env.reset()
