"""Observation spec conformance tests."""
import pytest
import numpy as np
from env.obs_spec import OBS_SPEC, OBS_DIM, OBS_SPEC_VERSION


def test_obs_spec_integrity():
    """ENV-06: OBS_SPEC slices sum to OBS_DIM, no overlaps, version present."""
    assert isinstance(OBS_SPEC_VERSION, str)
    assert len(OBS_SPEC_VERSION) > 0
    total = sum(s.stop - s.start for s, *_ in OBS_SPEC)
    assert total == OBS_DIM, f"Slice total {total} != OBS_DIM {OBS_DIM}"
    all_indices = set()
    for s, *_ in OBS_SPEC:
        indices = set(range(s.start, s.stop))
        assert not (all_indices & indices), f"Overlap at {s}"
        all_indices |= indices


def test_obs_spec_version_required():
    """SC-5: OBS_SPEC_VERSION is non-empty string."""
    assert isinstance(OBS_SPEC_VERSION, str)
    assert len(OBS_SPEC_VERSION) >= 5  # at least "x.y.z"
    parts = OBS_SPEC_VERSION.split(".")
    assert len(parts) == 3, "Version must be semver: MAJOR.MINOR.PATCH"


def test_obs_shape_and_dtype(env):
    """ENV-05: Obs shape==(22,), dtype==float32."""
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {obs.shape}"
    assert obs.dtype == np.float32


def test_obs_agent_pos_tracks_physics(env):
    """ENV-05: obs[0:2] tracks agent position over multiple steps."""
    obs, _ = env.reset()
    physics = env.unwrapped._dm_env.physics
    task = env.unwrapped._task
    player = task._players[0]
    x_joint = player.mjcf_model.find('joint', 'x')
    y_joint = player.mjcf_model.find('joint', 'y')

    # obs[0:2] should match physics qpos right after reset
    px = float(physics.bind(x_joint).qpos[0])
    py = float(physics.bind(y_joint).qpos[0])
    assert abs(obs[0] - px) < 1e-5, f"obs[0]={obs[0]:.4f} != physics_x={px:.4f}"
    assert abs(obs[1] - py) < 1e-5, f"obs[1]={obs[1]:.4f} != physics_y={py:.4f}"

    # Step and verify obs still matches physics
    action = env.action_space.sample()
    obs2, _, _, _, _ = env.step(action)
    px2 = float(physics.bind(x_joint).qpos[0])
    py2 = float(physics.bind(y_joint).qpos[0])
    assert abs(obs2[0] - px2) < 1e-5, f"After step: obs[0]={obs2[0]:.4f} != physics_x={px2:.4f}"
    assert abs(obs2[1] - py2) < 1e-5, f"After step: obs[1]={obs2[1]:.4f} != physics_y={py2:.4f}"


def test_obs_puck_tracks_physics(env):
    """ENV-05: obs[16:20] tracks puck position and velocity."""
    obs, _ = env.reset()
    physics = env.unwrapped._dm_env.physics
    task = env.unwrapped._task
    puck_x_j = task._puck.mjcf_model.find('joint', 'puck_x')
    puck_y_j = task._puck.mjcf_model.find('joint', 'puck_y')

    # obs[16:18] should match puck position after reset
    px = float(physics.bind(puck_x_j).qpos[0])
    py = float(physics.bind(puck_y_j).qpos[0])
    assert abs(obs[16] - px) < 1e-5, f"obs[16]={obs[16]:.4f} != puck_x={px:.4f}"
    assert abs(obs[17] - py) < 1e-5, f"obs[17]={obs[17]:.4f} != puck_y={py:.4f}"

    # Set puck velocity and step, then check vel tracks
    physics.bind(puck_x_j).qvel = 3.0
    physics.bind(puck_y_j).qvel = 1.5
    action = env.action_space.sample()
    obs2, _, _, _, _ = env.step(action)
    vx2 = float(physics.bind(puck_x_j).qvel[0])
    vy2 = float(physics.bind(puck_y_j).qvel[0])
    assert abs(obs2[18] - vx2) < 1e-4, f"obs[18]={obs2[18]:.4f} != puck_vx={vx2:.4f}"
    assert abs(obs2[19] - vy2) < 1e-4, f"obs[19]={obs2[19]:.4f} != puck_vy={vy2:.4f}"
