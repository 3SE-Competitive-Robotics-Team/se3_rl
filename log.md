# Log

## 2026-06-13 - Stair TrainView floor plane fix

- Symptom: local `watch_remote` / TrainView showed the stair pit covered by a transparent flat plane at the pit top, and the robot was launched upward after spawning in the inverted-pyramid stair scene.
- Root cause: the SerialLeg MJCF used by the target repo still contained its standalone world `<geom name="floor" type="plane" ...>`. MJLab already provides terrain separately, so this infinite MJCF plane overlapped the generated stair pit and acted as an unintended collision barrier.
- Fix: `src/se3_train/robot_cfg.py` now loads the MJCF through `_serialleg_spec_for_training()` and deletes the worldbody `floor` geom before creating the entity, matching the source repo behavior.
- Follow-up: the stair training task no longer uses the migration-only stair MJCF. Stair and the main recovery task now both use the standard `serialleg_fourbar_surrogate_train.xml` through the same `get_serialleg_cfg()` entry point. The old asset remains isolated to its matching recovery-finetune state cache.
- Validation: static `ruff` check passed; compiled recovery/stair entity signatures are identical, and direct spec inspection confirmed `has_floor=False`.

## 2026-06-14 - Stair CTBC ramp diagnosis

- Symptom: `model_600` could still lean hard and rush forward while the robot body was on the pit-bottom flat area; this was not caused by the velocity curriculum, because iter 600 still commands only `vx=0.2..0.6 m/s` and `yaw=0`.
- Diagnosis: `scripts/diagnose_stair_rollout.py` compared the same checkpoint at iter 600 with CTBC full strength and at iter 399 with CTBC disabled. Forward speed stayed similar, but full CTBC raised max tilt from about 20 deg to about 37 deg and introduced raw action saturation.
- Fix: `StairClimbState` now supports a configurable CTBC feedforward ramp. The stair task starts CTBC at iter 400, ramps to full strength by iter 800, holds until iter 1000, and anneals to zero by iter 1600.
- Validation: `compileall` passed. Re-running the `model_600` diagnostic with the ramp gives `kff=0.5`, max tilt about 20 deg, and zero raw action saturation under the same fixed terrain-level watch setup.

## 2026-06-15 - Stair walk800 flat-phase curriculum

- Change: replaced the stair task's first walking phase with an 800-iteration flat-style locomotion phase. During iter 0-799 the terrain curriculum forces flat terrain, flat velocity tracking is active, and the flat wheel/leg contact penalties are kept active.
- Curriculum: walking commands ramp to `vx=(-2.5, 2.5)` and `yaw=(-3.0, 3.0)` by iter 700, with height range reaching `(0.24, 0.37)` by iter 600. At iter 800 the stair phase starts with `vx=(0.6, 1.0)`, then expands to `vx=(0.6, 2.5)` while keeping terrain-aware height enabled.
- CTBC: reverted the temporary feedforward ramp from 2026-06-14. `ff_bias()` is now full strength from iter 800 through iter 1000, then anneals to zero by iter 1600.
- Validation: `compileall` and `ruff` passed for the stair task files. Static config checks confirmed max iterations `3800`, 64-step rollout, no `ff_ramp_end_iter`, correct reward wrappers, watch terrain behavior before/after iter 800, and expected `kff` values at iter 799/800/1000/1600.

## 2026-06-15 - Remote walk800 training start

- Remote project: created `/workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg_pr24_review_walk800_20260615` and linked its `.venv` to the existing review environment. The original `se3_wheel_leg` and previous walk400 review folder were not overwritten.
- Start: stopped the old `pr24_ctbc_backlift_correct100hz_m400_plus3000_total3400_8192_gpu1to7_20260614` run on GPU 1-7, then launched `SE3-WheelLegged-Stair-GRU` from scratch with `8192` envs, `3800` iterations, and `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7`.
- Run: `2026-06-14_16-55-13_pr24_walk800_flatcurr_stair_ctbc_total3800_8192_gpu1to7_20260615`; log path `/tmp/train_pr24_walk800_flatcurr_stair_ctbc_20260615.log`.
- Verification: by iteration `4/3800`, selected GPUs were active, `Curriculum/terrain_levels/walking_phase=1`, `Stair/diag_stair_env_rate=0`, `Stair/diag_ctbc_kff=0`, and flat locomotion contact metrics were present.

## 2026-06-15 - Pulled and checked remote model_800

- Pull: copied `model_800.pt` from the remote walk800 run to `logs/remote_watch/2026-06-14_16-55-13_pr24_walk800_flatcurr_stair_ctbc_total3800_8192_gpu1to7_20260615/model_800.pt`.
- Check: ran local CUDA headless rollouts at `SE3_WATCH_ITER=799`, so the model was evaluated in the final flat walking phase with `ctbc_kff=0` and flat terrain.
- Random walk curriculum: 64 envs for 8 s, no terminations, final max tilt `7.94 deg`, mean tilt `2.58 deg`, and `tilt_gt_30_rate=0.00059`.
- Fixed forward check: requested `vx=2.0`, `yaw=0`, `height=0.36`; the diff-drive command constraint clipped actual `vx` to `1.89 m/s`. 32 envs for 8 s had no terminations, final max tilt `6.37 deg`, mean tilt `2.65 deg`, and `tilt_gt_45_rate=0.00066`.
- Note: the walk curriculum target range is configured to `2.5`, but `diff_drive_wheel_speed_fraction=0.70` caps realized straight-line command speed at about `1.89 m/s`.
