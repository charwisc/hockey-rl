"""Shared fixtures for hockey environment tests."""
import pytest
from env.hockey_env import HockeyEnv


@pytest.fixture
def make_env():
    def _make(agent_idx=0, frozen_opponent_fn=None):
        return HockeyEnv(agent_idx=agent_idx, frozen_opponent_fn=frozen_opponent_fn)
    return _make


@pytest.fixture
def env(make_env):
    e = make_env(agent_idx=0)
    yield e
    e.close()
