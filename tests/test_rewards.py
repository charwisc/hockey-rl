"""Reward component isolation tests."""
import pytest
import numpy as np


def test_sparse_goal_reward(env):
    """ENV-07: r_goal == 10.0 when puck crosses goal line (team 0 scores)."""
    env.reset()
    physics = env.unwrapped._dm_env.physics
    task = env.unwrapped._task

    # Place puck beyond the away goal line: puck_x > 14.0 (rink_half=15, threshold at 14.0)
    # within goal width (abs(y) < 2.0) -> team 0 scores -> agent 0 r_goal == 10.0
    puck_x_joint = task._puck.mjcf_model.find('joint', 'puck_x')
    puck_y_joint = task._puck.mjcf_model.find('joint', 'puck_y')
    physics.bind(puck_x_joint).qpos = 14.5
    physics.bind(puck_y_joint).qpos = 0.0
    # Zero out puck velocity so it stays in goal zone through the physics step
    physics.bind(puck_x_joint).qvel = 0.0
    physics.bind(puck_y_joint).qvel = 0.0

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert info['r_goal'] == 10.0, \
        f"Expected r_goal==10.0 when puck is in away goal for team 0 agent, got {info['r_goal']}"


def test_puck_toward_goal_fires_with_possession(env):
    """ENV-07: r_puck_toward_goal > 0 when agent has possession and puck moves toward goal."""
    env.reset()
    physics = env.unwrapped._dm_env.physics
    task = env.unwrapped._task
    player = task._players[0]

    # Place agent at center facing +x (toward away goal)
    x_joint = player.mjcf_model.find('joint', 'x')
    y_joint = player.mjcf_model.find('joint', 'y')
    rot_joint = player.mjcf_model.find('joint', 'rot')
    physics.bind(x_joint).qpos = 0.0
    physics.bind(y_joint).qpos = 0.0
    physics.bind(rot_joint).qpos = 0.0  # facing +x

    # Place puck just ahead of agent's stick tip (within POSSESSION_DIST=0.5m)
    # Stick tip = agent_pos + 0.4m in facing dir = (0.4, 0)
    # So puck at (0.4, 0) = inside possession range
    puck_x_joint = task._puck.mjcf_model.find('joint', 'puck_x')
    puck_y_joint = task._puck.mjcf_model.find('joint', 'puck_y')
    physics.bind(puck_x_joint).qpos = 0.4
    physics.bind(puck_y_joint).qpos = 0.0
    # Puck moving toward away goal (+x direction)
    physics.bind(puck_x_joint).qvel = 2.0
    physics.bind(puck_y_joint).qvel = 0.0

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert info['r_puck_toward_goal'] > 0, \
        f"Expected r_puck_toward_goal > 0 with possession+puck toward goal, got {info['r_puck_toward_goal']}"


def test_puck_toward_goal_gated_on_possession(env):
    """ENV-07: r_puck_toward_goal == 0 when agent does NOT have possession."""
    env.reset()
    physics = env.unwrapped._dm_env.physics
    task = env.unwrapped._task
    player = task._players[0]

    # Place agent far from puck (no possession)
    x_joint = player.mjcf_model.find('joint', 'x')
    y_joint = player.mjcf_model.find('joint', 'y')
    physics.bind(x_joint).qpos = -10.0
    physics.bind(y_joint).qpos = 0.0

    # Puck far away, moving toward goal
    puck_x_joint = task._puck.mjcf_model.find('joint', 'puck_x')
    puck_y_joint = task._puck.mjcf_model.find('joint', 'puck_y')
    physics.bind(puck_x_joint).qpos = 5.0
    physics.bind(puck_y_joint).qpos = 0.0
    physics.bind(puck_x_joint).qvel = 3.0
    physics.bind(puck_y_joint).qvel = 0.0

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert info['r_puck_toward_goal'] == 0.0, \
        f"Expected r_puck_toward_goal==0 without possession, got {info['r_puck_toward_goal']}"


def test_reward_components_in_info(env):
    """ENV-07: All 6 reward keys present in info dict every step."""
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    expected_keys = {'r_goal', 'r_puck_toward_goal', 'r_possession',
                     'r_positioning', 'r_clustering', 'r_step_penalty'}
    assert expected_keys.issubset(info.keys()), \
        f"Missing keys: {expected_keys - set(info.keys())}"


def test_reward_independent_extraction(env):
    """SC-3: Sparse and shaped rewards independently extractable from info."""
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    # Each reward component must be a plain float
    for key in ['r_goal', 'r_puck_toward_goal', 'r_possession',
                'r_positioning', 'r_clustering', 'r_step_penalty']:
        assert isinstance(info[key], float), f"{key} is not float"
