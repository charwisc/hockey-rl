"""
Observation vector specification — cross-boundary contract.

This file defines the canonical layout of the 22-float observation vector
used by the hockey environment. Every downstream consumer (ONNX export,
JS physics mirror, tests) imports from here.

MODIFICATION PROTOCOL:
  - Any change to index assignments -> bump major version (1.x.x -> 2.0.0)
  - New field appended at end -> bump minor version (1.0.x -> 1.1.0)
  - Notes-only changes -> bump patch version (1.0.0 -> 1.0.1)
  - All downstream (ONNX export, JS mirror) must be updated before new version is used
"""

OBS_SPEC_VERSION = "1.0.0"

OBS_DIM = 22

# Each entry: (index_slice, field_name, units, coordinate_frame, notes)
OBS_SPEC = [
    (slice(0, 2),   "agent_pos",       "meters",  "world_xy", "xy position of this agent"),
    (slice(2, 4),   "agent_vel",       "m/s",     "world_xy", "xy velocity of this agent"),
    (slice(4, 6),   "teammate_pos",    "meters",  "world_xy", "xy position of teammate"),
    (slice(6, 8),   "teammate_vel",    "m/s",     "world_xy", "xy velocity of teammate"),
    (slice(8, 10),  "opponent0_pos",   "meters",  "world_xy", "xy position of opponent 0"),
    (slice(10, 12), "opponent0_vel",   "m/s",     "world_xy", "xy velocity of opponent 0"),
    (slice(12, 14), "opponent1_pos",   "meters",  "world_xy", "xy position of opponent 1"),
    (slice(14, 16), "opponent1_vel",   "m/s",     "world_xy", "xy velocity of opponent 1"),
    (slice(16, 18), "puck_pos",        "meters",  "world_xy", "xy position of puck"),
    (slice(18, 20), "puck_vel",        "m/s",     "world_xy", "xy velocity of puck"),
    (slice(20, 21), "stick_angle",     "radians", "agent_local", "stick rotation relative to agent heading"),
    (slice(21, 22), "facing_angle",    "radians", "world",    "agent heading in world frame (atan2)"),
]

# Self-check: no silent index drift
assert len(OBS_SPEC) == 12, f"Expected 12 fields, got {len(OBS_SPEC)}"
assert sum(s.stop - s.start for s, *_ in OBS_SPEC) == OBS_DIM, \
    f"Slice lengths must sum to {OBS_DIM}"

# Verify no overlapping slices
_all_indices = set()
for s, *_ in OBS_SPEC:
    indices = set(range(s.start, s.stop))
    overlap = _all_indices & indices
    assert not overlap, f"Overlapping indices: {overlap}"
    _all_indices |= indices
assert len(_all_indices) == OBS_DIM, "Gaps in index coverage"
