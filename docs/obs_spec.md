# Observation Vector Specification

**Version:** 1.0.0 (OBS_SPEC_VERSION)
**Dimension:** 22 floats (OBS_DIM)
**Data type:** float32
**Coordinate frame:** world_xy (all positions and velocities in world frame)
**Units:** meters and m/s (no normalization -- VecNormalize handles that in training)

## Index Layout

| Index | Name | Units | Frame | Notes |
|-------|------|-------|-------|-------|
| 0 | agent_pos_x | m | world_xy | Agent x position |
| 1 | agent_pos_y | m | world_xy | Agent y position |
| 2 | agent_vel_x | m/s | world_xy | Agent x velocity |
| 3 | agent_vel_y | m/s | world_xy | Agent y velocity |
| 4 | teammate_pos_x | m | world_xy | Teammate x position |
| 5 | teammate_pos_y | m | world_xy | Teammate y position |
| 6 | teammate_vel_x | m/s | world_xy | Teammate x velocity |
| 7 | teammate_vel_y | m/s | world_xy | Teammate y velocity |
| 8 | opponent0_pos_x | m | world_xy | Opponent 0 x position |
| 9 | opponent0_pos_y | m | world_xy | Opponent 0 y position |
| 10 | opponent0_vel_x | m/s | world_xy | Opponent 0 x velocity |
| 11 | opponent0_vel_y | m/s | world_xy | Opponent 0 y velocity |
| 12 | opponent1_pos_x | m | world_xy | Opponent 1 x position |
| 13 | opponent1_pos_y | m | world_xy | Opponent 1 y position |
| 14 | opponent1_vel_x | m/s | world_xy | Opponent 1 x velocity |
| 15 | opponent1_vel_y | m/s | world_xy | Opponent 1 y velocity |
| 16 | puck_pos_x | m | world_xy | Puck x position |
| 17 | puck_pos_y | m | world_xy | Puck y position |
| 18 | puck_vel_x | m/s | world_xy | Puck x velocity |
| 19 | puck_vel_y | m/s | world_xy | Puck y velocity |
| 20 | stick_angle | rad | agent_local | RESERVED: always-zero in v1.0.0. No independent stick joint; action's stick_angle controls body vrot. Downstream consumers (ONNX export, JS mirror) must not use this dimension for physics mirroring. |
| 21 | facing_angle | rad | world | Agent heading (atan2) |

## Agent Ordering

| Index | Team | Player | Role |
|-------|------|--------|------|
| 0 | Team 0 | Player 0 | Training agent (when agent_idx=0) |
| 1 | Team 0 | Player 1 | Teammate or training agent |
| 2 | Team 1 | Player 0 | Opponent |
| 3 | Team 1 | Player 1 | Opponent |

## Action Vector

| Index | Name | Range | Mapped To |
|-------|------|-------|-----------|
| 0 | move_x | [-1, 1] | vx actuator target * speed * MAX_SPEED |
| 1 | move_y | [-1, 1] | vy actuator target * speed * MAX_SPEED |
| 2 | speed | [-1, 1] | Mapped to [0, 1], scales movement |
| 3 | stick_angle | [-1, 1] | vrot actuator target * MAX_ROT_SPEED (controls body rotation, NOT an independent stick joint) |

## Modification Protocol

- Any change to index assignments: bump MAJOR version (1.x.x -> 2.0.0)
- New field appended at end: bump MINOR version (1.0.x -> 1.1.0)
- Notes-only changes: bump PATCH version (1.0.0 -> 1.0.1)
- All downstream (ONNX export, JS mirror) must be updated before new version is used

## Source of Truth

The canonical definition is `env/obs_spec.py`. This document is a human-readable
rendering. If they disagree, `obs_spec.py` wins.
