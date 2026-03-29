"""Physics stability tests for the hockey environment."""
import pytest
import numpy as np
from dm_control import composer
from dm_control import mjcf
from env.hockey_arena import HockeyArena
from env.hockey_puck import HockeyPuck


class PuckOnlyTask(composer.Task):
    """Minimal task: just arena + puck, for physics stability testing."""

    def __init__(self):
        self._arena = HockeyArena()
        self._puck = HockeyPuck()
        self._arena.attach(self._puck)

    @property
    def root_entity(self):
        return self._arena

    def initialize_episode(self, physics, random_state):
        # Reset puck to center
        puck_x_joint = self._puck.mjcf_model.find('joint', 'puck_x')
        puck_y_joint = self._puck.mjcf_model.find('joint', 'puck_y')
        physics.bind(puck_x_joint).qpos = 0.0
        physics.bind(puck_y_joint).qpos = 0.0
        physics.bind(puck_x_joint).qvel = 0.0
        physics.bind(puck_y_joint).qvel = 0.0

    def get_observation(self, physics):
        return {}

    def get_reward(self, physics):
        return 0.0


@pytest.fixture
def puck_env():
    task = PuckOnlyTask()
    env = composer.Environment(task, time_limit=120.0)
    yield env, task
    env.close()


def test_arena_compiles(puck_env):
    """ENV-01: Arena XML compiles; boards, goals exist."""
    env, task = puck_env
    timestep = env.reset()
    physics = env.physics
    # Check timestep
    assert physics.model.opt.timestep == 0.005
    # Check goal sites exist
    home_id = physics.model.name2id('home_goal', 'site')
    away_id = physics.model.name2id('away_goal', 'site')
    assert home_id >= 0
    assert away_id >= 0


def test_puck_stability_1000steps(puck_env):
    """ENV-03 + SC-1: 10k random impulses, no NaN."""
    env, task = puck_env
    timestep = env.reset()
    rng = np.random.RandomState(42)
    for step in range(10000):
        # Apply random force to puck via velocity perturbation
        puck_x_joint = task._puck.mjcf_model.find('joint', 'puck_x')
        puck_y_joint = task._puck.mjcf_model.find('joint', 'puck_y')
        if step % 100 == 0:
            physics = env.physics
            physics.bind(puck_x_joint).qvel = rng.uniform(-5, 5)
            physics.bind(puck_y_joint).qvel = rng.uniform(-5, 5)
        timestep = env.step(np.array([]))  # no actuators in puck-only task
        assert not np.any(np.isnan(env.physics.data.qpos)), f"NaN qpos at step {step}"
        assert not np.any(np.isnan(env.physics.data.qvel)), f"NaN qvel at step {step}"


def test_puck_friction_decay(puck_env):
    """ENV-03: Puck velocity decays from ice friction."""
    env, task = puck_env
    env.reset()
    physics = env.physics
    puck_x_joint = task._puck.mjcf_model.find('joint', 'puck_x')
    puck_y_joint = task._puck.mjcf_model.find('joint', 'puck_y')
    # Set initial velocity
    physics.bind(puck_x_joint).qvel = 5.0
    physics.bind(puck_y_joint).qvel = 0.0
    initial_speed = 5.0
    # Step for 1 second (200 steps at 0.005s)
    for _ in range(200):
        env.step(np.array([]))
    final_vx = physics.bind(puck_x_joint).qvel
    final_vy = physics.bind(puck_y_joint).qvel
    final_speed = np.sqrt(final_vx**2 + final_vy**2)
    assert final_speed < initial_speed * 0.9, \
        f"Puck speed did not decay: {initial_speed:.2f} -> {final_speed:.2f}"


def test_board_bounce_angle(puck_env):
    """ENV-03: Puck bounces off right board; x-velocity reverses sign."""
    env, task = puck_env
    env.reset()
    physics = env.physics
    puck_x_joint = task._puck.mjcf_model.find('joint', 'puck_x')
    puck_y_joint = task._puck.mjcf_model.find('joint', 'puck_y')
    # Place puck near right wall, heading toward it
    physics.bind(puck_x_joint).qpos = 14.0  # rink_length/2 = 15, wall at 15
    physics.bind(puck_y_joint).qpos = 0.0
    physics.bind(puck_x_joint).qvel = 5.0   # heading toward right wall
    physics.bind(puck_y_joint).qvel = 2.0   # slight y component
    # Step until bounce occurs (puck reaches wall and reverses)
    for _ in range(500):
        env.step(np.array([]))
    final_vx = physics.bind(puck_x_joint).qvel
    # After bounce, x-velocity should be negative (reversed)
    assert final_vx < 0, f"Puck did not bounce: final vx={final_vx:.3f}"


def test_agents_load(env):
    """ENV-02: 4 capsule agents load with stick geoms and actuators."""
    env.reset()
    physics = env.unwrapped._dm_env.physics
    # 4 players x 3 actuators each = 12 total actuators
    assert physics.model.nu == 12, f"Expected 12 actuators, got {physics.model.nu}"
    # Verify physics is accessible
    assert physics is not None
    assert not np.any(np.isnan(physics.data.qpos)), "NaN in initial qpos"


def test_1000_steps_no_nan(env):
    """SC-1: Alias for puck_stability_1000steps with explicit NaN focus."""
    obs, _ = env.reset(seed=42)
    for step in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert not np.any(np.isnan(obs)), f"NaN at step {step}"
        if terminated or truncated:
            obs, _ = env.reset()
