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

## 2026-06-16 - Full resume from model_2400 for 2000 more iterations

- Diagnosis: the previous `pulse03_ann900to1100` run stopped at iter `2457/3800` with no Python traceback; zombie exit codes showed `signal=15` (`SIGTERM`). The last complete checkpoint was `model_2400.pt`.
- Start: synced current code into the existing remote project and launched a full resume from `model_2400.pt` with `8192` envs, `--iterations 2000`, `--stair-local-iter-offset 2400`, and `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7`.
- Run: `2026-06-16_03-26-35_pr24_obs34_walk800_v18_ctbc_pulse03_m2400_fullresume_plus2000_total4400_8192_gpu1to7_20260616`; log path `/tmp/train_pr24_obs34_walk800_v18_ctbc_pulse03_m2400_plus2000_20260616.log`; PID `3136340`.
- Verification: the log starts at `Learning iteration 2400/4400`, with `walking_phase=0`, `diag_ctbc_local_iter=2400.0156`, and `diag_ctbc_kff=0.0`; GPU 0 remains unused while GPUs `1-7` are active.

## 2026-06-16 - Stair mastery terrain curriculum

- Change: replaced per-env terrain row up/down with a global mastery target level. Target difficulty upgrades by one row when the target-level success window reaches at least `128` samples and success rate is `>= 80%`; failed windows do not downgrade the target.
- Sampling after the walking phase: `75%` target-level stair episodes, `20%` random lower-level stair episodes, and `5%` flat episodes. The first target level remains `0` unless `SE3_STAIR_INITIAL_TARGET_LEVEL` is set.
- Resume support: `scripts/remote_sync_start_stair.py` now exposes `--stair-initial-target-level` and documents that this runtime curriculum state is not restored by runner checkpoints.
- Validation: `ruff`, `compileall`, and a fake-env curriculum check passed. The fake-env check upgraded level `2 -> 3`, kept level `3` after a failed window, and measured sampling rates near `74.97% / 20.13% / 4.90%`.

## 2026-06-16 - Remote episode-10s level-3 full resume

- Change: the stair training environment now overrides the inherited flat-task timeout to `episode_length_s=10.0` for training while keeping play/watch at `9999.0`.
- Remote environment: the review project's `.venv` symlink was broken because the deleted walk400 project had owned the previous venv. It was relinked to `/workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg/.venv`; no missing-package install was needed.
- Start: synced current code into `/workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg_pr24_review_walk800_obs34_v18_20260615` and full-resumed from `model_2400.pt` with `--stair-local-iter-offset 2400`, `--stair-initial-target-level 3`, `8192` envs, and GPUs `1-7`.
- Run: `2026-06-16_06-50-51_pr24_obs34_walk800_v18_ctbc_pulse03_ep10_mastery_lvl3_m2400_fullresume_plus2000_total4400_8192_gpu1to7_20260616`; log path `/tmp/train_pr24_obs34_walk800_v18_ctbc_pulse03_ep10_lvl3_m2400_plus2000_20260616.log`; PID `3406417`.
- Verification: the run starts at `Learning iteration 2400/4400`; `env.yaml` records `episode_length_s: 10.0`, `flat_ratio: 0.05`, and `initial_target_level: 3`; logs show `diag_ctbc_local_iter=2400.0156`, `diag_ctbc_kff=0.0`, `target_level=3.0`, and GPU 0 unused.

## 2026-06-17 - Stair reward alignment v12

- Change: replaced the stair climb progress reward with `stair_target_progress_delta`, which rewards only new normalized radial progress toward the same `move_up_distance_ratio=0.35` target used by the mastery terrain curriculum. `stair_max_x_progress` was also changed from a height-gain proxy to the same normalized radial progress metric.
- New penalties: added `stair_no_progress`, `stair_backtrack`, and `stair_ctbc_no_progress` to penalize commanded stair episodes that stop, move back toward the pit center, or trigger CTBC without outward progress. Existing stall penalties were strengthened: `stair_riser_stall` weight `-1.0 -> -2.0`, `stair_commanded_stall` weight `-2.0 -> -3.0`.
- Reward rebalance: reduced CTBC shape rewards so they do not dominate locomotion: `stair_feet_clearance 2.0 -> 0.3`, `stair_feet_air_time 2.0 -> 0.2`, `stair_contact_number 2.0 -> 0.2`, and `stair_wheel_swing_zero_vel 0.5 -> 0.1`. The new radial progress reward weight is `5.0`.
- Curriculum: stair-phase forward speed is capped at `vx=(0.6, 1.2)` after iter `1050`; walking-phase yaw range still reaches `(-3.0, 3.0)` by iter `700`. CTBC local iteration remains aligned by full resume with `--stair-local-iter-offset 800`.
- Remote run: full-resumed from walking `model_800.pt` in the existing remote project as `2026-06-16_17-57-21_pr24_obs34_rewardalign_v12_m800_resume_total3800_8192_gpu1to7_20260617`; it completed through `model_3799.pt`. Final logs showed `target_level=9`, `level_mean` about `7.93`, `target_success_rate` about `0.83`, `flat_sample_rate` about `0.05`, `low_stair_sample_rate` about `0.19`, and `diag_ctbc_kff=0.0`.
- Validation: `compileall` and `ruff check` passed for `src/se3_train/tasks/stair/rewards.py` and `src/se3_train/tasks/stair/env_cfg.py`.
- Follow-up diagnosis: `model_3799.pt` can climb but tends to exploit diagonal motion on the square inverted-pyramid stairs. Training terrain uses `step_width=0.5 m`, while the robot collision envelope is about `0.56 m` for the base and `0.63 m` overall, so straight climbing can contact the next riser and diagonal climbing is geometrically easier. Sim2sim slow startup was not caused by `rc_off`; recorded rollouts had `output_enabled=1.0` and `rc_policy_reset=0.0`, and aligning sim2sim `--initial-base-height 0.36` with `command height=0.36` improved early flat speed.

## 2026-06-17 - Yaw PID robustness run and watch_remote fixes

- Training change: stair-phase yaw control now uses random world-frame yaw PID targets after the 800-iteration walking phase. Target bases remain `0/90/180/-90 deg`, target jitter was narrowed to `±35 deg`, and reset aligns the initial yaw to the command target with `90%` probability plus `±0.25 rad` noise. The reset randomization also keeps the command-height mismatch/no-torque cases for sim2sim-style startup robustness.
- Remote run: synced code into the existing remote project and restarted from walking `model_800.pt` as `2026-06-17_08-24-18_pr24_obs34_yawpid35_m800_restart_total3800_8192_gpu1to7_20260617` with `8192` envs on GPUs `1-7`. The run starts in the stair phase, not the walking phase.
- Target-level log interpretation: apparent rows with `target_success_rate=0` after an upgrade are a logging/window-reset artifact. `upgraded=0.0156` means one of 64 rollout steps upgraded the global target level; the next row often has an empty new target-level success window.
- Watch bug: `watch_remote` previously passed only the checkpoint filename iteration, so a warm-started `model_500.pt` was viewed as iteration `500` instead of `1300`. This incorrectly forced the local viewer back into walking/flat curriculum. `scripts/watch_remote_train_local.py` now infers the offset from the remote log, infers the current stair target level, and exports `SE3_WATCH_ITER`, `SE3_TRAIN_VIEW_ITER`, `SE3_WARM_START_ITERATION`, and `SE3_WARM_START_STEPS_PER_ITER=64`.
- Yaw debug: the robots that spun more than 360 deg in `remote_watch` were not running yaw PID. `Se3WarmStartRunner.load()` reset `common_step_counter` to zero unless `SE3_WARM_START_ITERATION` was set, so the command term stayed in the walking-phase yaw profile. After exporting the warm-start iteration, local diagnostics showed `pid_active=True`, `profile_active=False`, `spin_gt_360_rate=0`, and `spin_gt_180_rate=0` over 32 envs for 5 s.
- Viewer CTBC sync: `src/se3_train/play.py` now prefers `SE3_WATCH_ITER` over the raw checkpoint filename when syncing CTBC iteration, preventing a `model_N.pt` warm-start checkpoint from overriding CTBC/watch state back to `N`.
- NaN log diagnosis: `Stair/target_progress_mean`, `Stair/target_speed_mean`, and related stair/curriculum diagnostics became `nan` only on iterations with nonzero `Episode_Termination/catastrophic_state`. A few physically divergent envs produced non-finite root pose/velocity, and the diagnostic mean propagated those NaNs before reset.
- NaN log fix: `src/se3_train/tasks/stair/rewards.py` and `src/se3_train/tasks/stair/curriculums.py` now use finite-safe means for diagnostics, skipping NaN/Inf samples and returning `0` if no finite sample exists. This does not hide the underlying divergence; `Episode_Termination/catastrophic_state` remains the signal for that.
- Validation: `compileall` passed for `watch_remote_train_local.py`, `diagnose_stair_rollout.py`, `play.py`, `rewards.py`, and `curriculums.py`. A finite-mean check on `[1, nan, 3, inf]` returned `2.0`.
