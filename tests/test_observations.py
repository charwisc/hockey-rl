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


@pytest.mark.xfail(reason="HockeyEnv not yet implemented")
def test_obs_shape_and_dtype(env):
    """ENV-05: Obs shape==(22,), dtype==float32."""
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {obs.shape}"
    assert obs.dtype == np.float32


@pytest.mark.xfail(reason="HockeyEnv not yet implemented")
def test_obs_agent_pos_tracks_physics(env):
    """ENV-05: obs[0:2] tracks agent position over multiple steps."""
    pass  # Implemented in Plan 03


@pytest.mark.xfail(reason="HockeyEnv not yet implemented")
def test_obs_puck_tracks_physics(env):
    """ENV-05: obs[16:20] tracks puck position and velocity."""
    pass  # Implemented in Plan 03
