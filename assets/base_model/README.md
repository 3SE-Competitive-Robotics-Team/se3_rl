# Base Model Registry

## `model_5999_gru.pt` (current)

- Source run: local TensorBoard run on `abbtask`
- Source file: `logs/rsl_rl/se3_wheel_leg/2026-06-09_05-06-14_recovery_overdrive_resume_4100_e802948_tb/model_5999.pt`
- Training task: `SE3-WheelLegged-Recovery-GRU`
- Run name: `recovery_overdrive_resume_4100_e802948_tb`
- Source commit: `e8029484`
- SHA256: `93449475a5b103925f9d2cdd0e43f172de9e74ecff3984096c4b4805280757ad`

This checkpoint is the current recovery GRU base model.

It was selected over `model_4999_gru.pt` after A/B validation on 2026-06-09:
MJLab fixed-height probes favored it on 7/8 command heights, and MuJoCo sim2sim
with height-conditioned action defaults favored it on 13/14 tested scenarios.
The largest improvement is command-height tracking across the full
0.195-0.390 m recovery range.

## `model_4999_gru.pt`

- Source W&B run: `luzhongjin365-se3/se3_wheel_leg/acttgnoq`
- Source file: `model_4999.pt`
- Training task: `SE3-WheelLegged-Recovery-GRU`
- Run name: `2026-06-07_15-25-55_recovery_5k_upright_orientation_4a43eec`
- Source commit: `4a43eec8`
- SHA256: `877c46a2eb5636071f742a812715e67d7456afb3eaae47c57f58f6b34a298dc8`

This checkpoint is retained as the previous recovery GRU base model for
comparison and rollback.

It was selected over the desktop checkpoint `C:/Users/13567/Desktop/model_4999.pt`
because the desktop checkpoint was trained without the near-upright
`orientation_l2` penalty. A/B sim2sim validation on 2026-06-08 showed that the
`acttgnoq` checkpoint has better post-recovery attitude stability, especially in
`pitch180_h022`, `vx2_yaw6_h030`, and `vx3_yaw9_h030`.
