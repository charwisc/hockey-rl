"""Unit tests for training callbacks and HockeyEnv.opponent_path property.

Tests are designed to run without SB3 installed locally — callback imports
are patched via sys.modules before importing training modules.
"""
import os
import sys
import time
import tempfile
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

# Mark all tests in this module
pytestmark = pytest.mark.training


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_sb3_modules():
    """Patch SB3 modules so callback imports work without SB3 installed."""
    mock_sb3 = MagicMock()
    mock_base_callback = type('BaseCallback', (), {
        '__init__': lambda self, verbose=0: None,
        '_on_step': lambda self: True,
        '_on_rollout_end': lambda self: None,
        '_on_training_start': lambda self: None,
    })
    mock_sb3.common.callbacks.BaseCallback = mock_base_callback
    with patch.dict('sys.modules', {
        'stable_baselines3': mock_sb3,
        'stable_baselines3.common': mock_sb3.common,
        'stable_baselines3.common.callbacks': mock_sb3.common.callbacks,
        'stable_baselines3.common.vec_env': mock_sb3.common.vec_env,
    }):
        # Remove any cached training module imports so they use fresh mocks
        for key in list(sys.modules.keys()):
            if key.startswith('training'):
                del sys.modules[key]
        yield mock_sb3


@pytest.fixture
def mock_model():
    """Minimal mock of an SB3 PPO model."""
    model = MagicMock()
    model.num_timesteps = 500_000
    model.save = MagicMock(side_effect=lambda p: open(p + ".zip", 'w').close() if not p.endswith('.zip') else open(p, 'w').close())
    model.get_vec_normalize_env.return_value = MagicMock()
    return model


@pytest.fixture
def mock_training_env():
    """Minimal mock of a VecEnv."""
    env = MagicMock()
    env.num_envs = 2
    env.set_attr = MagicMock()
    return env


@pytest.fixture
def mock_logger():
    logger = MagicMock()
    logger.record = MagicMock()
    return logger


# ---------------------------------------------------------------------------
# Task 1A: HockeyEnv.opponent_path property bridge
# ---------------------------------------------------------------------------

def test_opponent_path_property():
    """HockeyEnv.opponent_path is settable and readable."""
    from env.hockey_env import HockeyEnv
    env = HockeyEnv(agent_idx=0)
    assert env.opponent_path is None
    env.opponent_path = "/some/path.zip"
    assert env.opponent_path == "/some/path.zip"
    env.close()


def test_opponent_path_caching():
    """Setting same path should not trigger PPO.load; changing path should."""
    from env.hockey_env import HockeyEnv
    env = HockeyEnv(agent_idx=0)

    # Pre-populate cache
    cached_model = MagicMock()
    cached_model.predict.return_value = (np.zeros(4, dtype=np.float32), None)
    env._opponent_model_cache["path"] = "/cached.zip"
    env._opponent_model_cache["model"] = cached_model

    # Set the same path -- should use cache (no PPO.load)
    env.opponent_path = "/cached.zip"
    obs = np.zeros(22, dtype=np.float32)

    with patch('stable_baselines3.PPO') as mock_ppo:
        result = env._default_opponent_fn(obs)
        mock_ppo.load.assert_not_called()

    assert result.shape == (4,)
    cached_model.predict.assert_called_once()
    env.close()


# ---------------------------------------------------------------------------
# Task 1B: SelfPlayPoolCallback tests
# ---------------------------------------------------------------------------

def test_self_play_pool_snapshot(mock_model, mock_training_env, mock_logger):
    """SelfPlayPoolCallback._snapshot_and_reassign creates a .zip file in pool_dir."""
    from training.self_play_callback import SelfPlayPoolCallback

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = SelfPlayPoolCallback(pool_dir=tmpdir, pool_update_freq=500_000, max_pool_size=20)
        cb.model = mock_model
        cb.training_env = mock_training_env
        cb.logger = mock_logger
        cb.num_timesteps = 500_000

        cb._snapshot_and_reassign()

        # Model.save should have been called with the pool path
        mock_model.save.assert_called_once()
        saved_path = mock_model.save.call_args[0][0]
        assert "pool_step_500000" in saved_path
        assert saved_path.startswith(tmpdir)

        # set_attr called once per env
        assert mock_training_env.set_attr.call_count == mock_training_env.num_envs


def test_pool_eviction(mock_model, mock_training_env, mock_logger):
    """Pool evicts oldest checkpoints when exceeding max_pool_size."""
    from training.self_play_callback import SelfPlayPoolCallback

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = SelfPlayPoolCallback(pool_dir=tmpdir, pool_update_freq=100_000, max_pool_size=3)
        cb.model = mock_model
        cb.training_env = mock_training_env
        cb.logger = mock_logger

        # Create 5 snapshots manually to trigger eviction
        for step in range(1, 6):
            # Create the file on disk so eviction can os.remove it
            fake_path = os.path.join(tmpdir, f"pool_step_{step * 100_000}.zip")
            with open(fake_path, 'w') as f:
                f.write("fake")
            cb._pool.append(fake_path)
            cb._latest_path = fake_path

        # Pool now has 5 entries; trigger eviction by calling _snapshot_and_reassign
        cb.num_timesteps = 600_000
        cb._snapshot_and_reassign()

        # After the snapshot (adds 1 more entry initially to make 6, then evicts down to max_pool_size=3)
        assert len(cb._pool) <= cb.max_pool_size


def test_wall_time_checkpoint(mock_model, mock_training_env, mock_logger):
    """WallTimeCheckpointCallback._save creates step_{N}.zip and step_{N}_vecnorm.pkl."""
    from training.checkpoint_callback import WallTimeCheckpointCallback

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = WallTimeCheckpointCallback(checkpoint_dir=tmpdir, interval_minutes=30.0)
        cb.model = mock_model
        cb.training_env = mock_training_env
        cb.logger = mock_logger
        cb.num_timesteps = 1_000_000

        cb._save()

        # model.save called with step_1000000.zip path
        mock_model.save.assert_called_once()
        saved_path = mock_model.save.call_args[0][0]
        assert "step_1000000" in saved_path
        assert saved_path.endswith(".zip")

        # VecNormalize.save called
        vec_norm = mock_model.get_vec_normalize_env.return_value
        vec_norm.save.assert_called_once()
        vecnorm_path = vec_norm.save.call_args[0][0]
        assert "step_1000000_vecnorm" in vecnorm_path
        assert vecnorm_path.endswith(".pkl")


def test_tb_callback_logging(mock_model, mock_training_env, mock_logger):
    """TensorBoardCustomCallback records hockey/goal_rate and hockey/puck_possession_rate."""
    from training.tb_callback import TensorBoardCustomCallback

    cb = TensorBoardCustomCallback()
    cb.model = mock_model
    cb.training_env = mock_training_env
    cb.logger = mock_logger
    cb.num_timesteps = 100_000

    # Simulate two completed episodes via _on_step locals
    infos = [
        {"score": [1, 0], "r_possession": 0.5},  # goal scored, possession positive
        {"score": [0, 0], "r_possession": -0.1},  # no goal, no possession
    ]
    dones = np.array([True, True])
    cb.locals = {"infos": infos, "dones": dones}
    cb._on_step()

    # One more step with no done
    cb.locals = {"infos": [{"score": [0, 0], "r_possession": 0.0}], "dones": np.array([False])}
    cb._on_step()

    # Trigger rollout end logging
    cb._on_rollout_end()

    # Should have recorded both metrics
    recorded_keys = [call_args[0][0] for call_args in mock_logger.record.call_args_list]
    assert "hockey/goal_rate" in recorded_keys
    assert "hockey/puck_possession_rate" in recorded_keys

    # goal_rate: 1 out of 2 episodes had a goal → 0.5
    goal_rate_call = [c for c in mock_logger.record.call_args_list if c[0][0] == "hockey/goal_rate"][0]
    assert abs(goal_rate_call[0][1] - 0.5) < 1e-6

    # puck_possession_rate: 1 out of 2 episodes had r_possession > 0 → 0.5
    poss_rate_call = [c for c in mock_logger.record.call_args_list if c[0][0] == "hockey/puck_possession_rate"][0]
    assert abs(poss_rate_call[0][1] - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Slow integration test (requires real MuJoCo + SB3)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_subproc_vec_env():
    """HockeyEnv wrapped in SubprocVecEnv runs 100 steps without crash."""
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from env.hockey_env import HockeyEnv

    def make_env(idx):
        def _init():
            return HockeyEnv(agent_idx=idx % 4)
        return _init

    venv = SubprocVecEnv([make_env(i) for i in range(2)])
    obs = venv.reset()
    for _ in range(100):
        actions = np.stack([
            np.random.uniform(-1, 1, size=(4,)).astype(np.float32)
            for _ in range(2)
        ])
        obs, rewards, dones, infos = venv.step(actions)
    venv.close()


# ---------------------------------------------------------------------------
# Integration tests for train.py
# ---------------------------------------------------------------------------

def test_train_parse_args_defaults(monkeypatch):
    """train.py parse_args returns correct defaults."""
    monkeypatch.setattr("sys.argv", ["train.py"])
    # Remove cached train module so fresh import picks up the mocked SB3
    sys.modules.pop("train", None)
    from train import parse_args
    args = parse_args()
    assert args.total_steps == 100_000_000
    assert args.n_envs == 16
    assert args.resume is None


def test_train_parse_args_custom(monkeypatch):
    """train.py parse_args handles custom values."""
    monkeypatch.setattr("sys.argv", ["train.py", "--total-steps", "50000000", "--n-envs", "8"])
    sys.modules.pop("train", None)
    from train import parse_args
    args = parse_args()
    assert args.total_steps == 50_000_000
    assert args.n_envs == 8


def test_train_make_env():
    """make_env returns callable that creates HockeyEnv."""
    sys.modules.pop("train", None)
    from train import make_env
    fn = make_env(agent_idx=0)
    assert callable(fn)
    env = fn()
    assert hasattr(env, 'opponent_path')
    assert env.observation_space.shape == (22,)
    assert env.action_space.shape == (4,)
    env.close()


def test_train_imports():
    """All training module imports resolve."""
    from training.self_play_callback import SelfPlayPoolCallback
    from training.checkpoint_callback import WallTimeCheckpointCallback
    from training.tb_callback import TensorBoardCustomCallback
    assert SelfPlayPoolCallback is not None
    assert WallTimeCheckpointCallback is not None
    assert TensorBoardCustomCallback is not None
