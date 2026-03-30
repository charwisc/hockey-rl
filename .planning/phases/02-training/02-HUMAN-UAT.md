---
status: partial
phase: 02-training
source: [02-VERIFICATION.md]
started: 2026-03-30T00:00:00Z
updated: 2026-03-30T00:00:00Z
---

## Current Test

[awaiting human testing — requires live RunPod session]

## Tests

### 1. TensorBoard goal-rate curve rises over training run
expected: The `hockey/goal_rate` metric shows an upward trend over a 50–100M step run — agents are learning to score, not just optimizing shaped reward

result: [pending]

### 2. At least one checkpoint exists past 50M steps
expected: `/workspace/checkpoints/step_50000000.zip` and `/workspace/checkpoints/step_50000000_vecnorm.pkl` exist (or equivalent step > 50M) after the run completes

result: [pending]

### 3. Checkpoints appear at 30-minute wall-time intervals
expected: Terminal output shows checkpoint saves approximately every 30 minutes; checkpoint timestamps confirm the interval

result: [pending]

### 4. TensorBoard logs are downloadable and contain all three metrics
expected: After `scp` or RunPod file manager download of `/workspace/tb_logs/`, TensorBoard shows `ep_rew_mean`, `hockey/goal_rate`, and `hockey/puck_possession_rate` curves per episode

result: [pending]

## Summary

total: 4
passed: 0
issues: 0
pending: 4
skipped: 0
blocked: 0

## Gaps
