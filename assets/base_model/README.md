# Base Model Registry

## `model_4999_gru.pt`

- Source W&B run: `luzhongjin365-se3/se3_wheel_leg/acttgnoq`
- Source file: `model_4999.pt`
- Training task: `SE3-WheelLegged-Recovery-GRU`
- Run name: `2026-06-07_15-25-55_recovery_5k_upright_orientation_4a43eec`
- Source commit: `4a43eec8`
- SHA256: `877c46a2eb5636071f742a812715e67d7456afb3eaae47c57f58f6b34a298dc8`

This checkpoint is the current recovery GRU base model for downstream warm-start
tasks such as `SE3-WheelLegged-Stair-CTBC-GRU`.

It was selected over the desktop checkpoint `C:/Users/13567/Desktop/model_4999.pt`
because the desktop checkpoint was trained without the near-upright
`orientation_l2` penalty. A/B sim2sim validation on 2026-06-08 showed that the
`acttgnoq` checkpoint has better post-recovery attitude stability, especially in
`pitch180_h022`, `vx2_yaw6_h030`, and `vx3_yaw9_h030`.

For stair training, link this directory into the stair experiment log root before
launching a formal run:

```bash
mkdir -p logs/rsl_rl/se3_wheel_leg_stair_ctbc
ln -sfn "$(pwd)/assets/base_model" logs/rsl_rl/se3_wheel_leg_stair_ctbc/base_model
```

On Windows, create an equivalent junction:

```powershell
New-Item -ItemType Junction `
  -Path logs\rsl_rl\se3_wheel_leg_stair_ctbc\base_model `
  -Target (Resolve-Path assets\base_model).Path
```
