"""Shared fixtures for hockey environment tests."""
import pytest


@pytest.fixture
def make_env():
    """Factory fixture: returns a function that creates HockeyEnv(agent_idx).
    Deferred import — env module not yet implemented."""
    def _make(agent_idx=0, frozen_opponent_fn=None):
        from env.hockey_env import HockeyEnv
        return HockeyEnv(agent_idx=agent_idx, frozen_opponent_fn=frozen_opponent_fn)
    return _make


@pytest.fixture
def env(make_env):
    """Single HockeyEnv instance for agent 0. Auto-closes after test."""
    e = make_env(agent_idx=0)
    yield e
    e.close()
