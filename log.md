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

## 2026-06-15 - Clamp stair command curriculum to 1.8 m/s

- Change: reduced the stair task command velocity upper bounds from `2.0/2.5` to `1.8` so the configured curriculum is below the diff-drive straight-line command budget of about `1.89 m/s`.
- Final walking velocity stages: iter `0` `(0.0, 0.0)`, `100` `(-0.5, 0.5)`, `250` `(-1.0, 1.0)`, `400` `(-1.5, 1.5)`, `600` `(-1.8, 1.8)`, `700` `(-1.8, 1.8)`.
- Final stair velocity stages: iter `800` `(0.6, 1.0)`, `1050` `(0.6, 1.6)`, `1300` `(0.6, 1.8)`, `1600` `(0.6, 1.8)`.
- Height stages unchanged: iter `0` `(0.22, 0.22)`, `150` `(0.22, 0.28)`, `350` `(0.23, 0.32)`, `600` `(0.24, 0.37)`, `800` `(0.24, 0.37)`.
- Validation: `compileall` and `ruff` passed for `src/se3_train/tasks/stair/env_cfg.py`; config printout confirmed the final velocity and height stages.

## 2026-06-15 - Remote obs34/v1.8 walk800 training restart

- Remote project: created `/workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg_pr24_review_walk800_obs34_v18_20260615` and synced the current local code.
- Start: launched `SE3-WheelLegged-Stair-GRU` from scratch with `8192` envs, `3800` iterations, and `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7`.
- Run: `2026-06-15_04-15-39_pr24_obs34_walk800_v18_stair_ctbc_total3800_8192_gpu1to7_20260615`; log path `/tmp/train_pr24_obs34_walk800_v18_stair_ctbc_20260615.log`; PID `1270910`.
- Verification: the training log shows actor observation shape `(34,)`, `leg_joint_pos` shape `(6,)`, actor GRU `GRU(34, 512)`, `Learning iteration 0/3800`, `Curriculum/terrain_levels/walking_phase=1`, and `Stair/diag_ctbc_kff=0`.

## 2026-06-15 - CTBC pulse profile and model_800 full resume

- Change: replaced the CTBC cosine feedforward envelope with a single-shot pulse profile. Each trigger now lasts `0.4 s`: `0.06 s` smooth attack to peak, `0.04 s` hold, then `0.30 s` smooth decay. The peak feedforward magnitude is preserved relative to the old cosine envelope.
- Config: `init_stair_climb_state` now passes `ff_period_s=0.4`, `ff_attack_s=0.06`, and `ff_hold_s=0.04` into `StairClimbState`. The previous `vx` curriculum clamp to `1.8 m/s` remains unchanged.
- Resume script: `scripts/remote_sync_start_stair.py --help` now documents the resume semantics. Warm-start starts all counters from zero. Full resume restores optimizer, runner iteration, and env `common_step_counter`, but the CTBC state object is recreated, so resuming from `model_800.pt` must use `--full-resume --stair-local-iter-offset 800`. The help also notes that `--iterations` is the number of additional iterations; from `model_800.pt` to total `3800`, use `--iterations 3000`.
- Local validation: `ruff check` and `compileall` passed. A direct `StairClimbState` check confirmed `period_steps=40`, peak profile at step `6`, profile returns to zero at step `40`, `kff_799=0`, `kff_800=1`, and `kff_1600=0`.
- Remote restart: stopped the old obs34/v1.8 run after confirming `model_800.pt` existed, then synced code into the existing remote project `/workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg_pr24_review_walk800_obs34_v18_20260615` without creating a new repository.
- New run: `2026-06-15_08-44-17_pr24_obs34_walk800_v18_ctbc_pulse04_m800_resume_total3800_8192_gpu1to7_20260615`; log path `/tmp/train_pr24_obs34_walk800_v18_ctbc_pulse04_m800_20260615.log`; PID `1646777`; GPUs `1-7`.
- Verification: the remote log shows loading `model_800.pt`, actor GRU `GRU(34, 512)`, `Learning iteration 800/3800`, `Stair/diag_ctbc_kff=1.0000`, and `Stair/diag_ctbc_local_iter=800.0156`, confirming the first 800 walking iterations were skipped and total iteration count stayed aligned.

## 2026-06-15 - Shorten CTBC pulse and advance annealing

- Change: shortened each CTBC trigger from `0.4 s` to `0.3 s`: `0.03 s` smooth attack, `0.08 s` peak hold, and `0.19 s` smooth decay.
- Annealing: CTBC remains full strength from iter `800` through `899`, linearly anneals from iter `900` to `1100`, and is zero from iter `1100` onward.
- Local validation: `ruff`, `compileall`, and direct state checks passed. The profile reaches peak at control step `3`, holds through step `10`, and returns to zero at step `30`; `kff` is `1.0/0.5/0.0` at iter `900/1000/1100`.
- Remote restart: stopped the previous `0.4 s` pulse run at about iter `1403`, synced into the existing remote project, and full-resumed again from the original walking `model_800.pt` with stair local offset `800`.
- Run: `2026-06-15_10-42-21_pr24_obs34_walk800_v18_ctbc_pulse03_ann900to1100_m800_resume_total3800_8192_gpu1to7_20260615`; log path `/tmp/train_pr24_obs34_walk800_v18_ctbc_pulse03_ann900to1100_m800_20260615.log`; PID `1813093`; GPUs `1-7`.
- Verification: the log starts at `Learning iteration 800/3800` with actor `GRU(34, 512)`, `walking_phase=0`, `diag_ctbc_local_iter=800.0156`, and `diag_ctbc_kff=1.0`. GPU 0 remains unused.
