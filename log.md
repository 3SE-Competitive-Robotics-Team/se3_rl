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

## 2026-06-17 - Encoder-level leg observation noise

- Change: moved leg position observation noise from the final 6D `[sin, cos, active]` output into the policy joint encoder layer `[LF, LB, RF, RB]`, preserving the sin/cos unit-circle structure and making active-rod noise arise from the underlying encoder readings.
- Added per-episode encoder bias sampled on reset with range `[-0.01, 0.01] rad` for the four policy leg encoders. The bias is actor-only and is not applied in `play/watch`.
- Added per-step encoder white noise starting at `[-0.01, 0.01] rad` and linearly ramping to `[-0.025, 0.025] rad` from PPO iter `800` to `1800` using `64` policy steps per iteration.
- Critic leg position observations remain clean; only the actor `leg_joint_pos` term uses the encoder bias/noise parameters. Existing leg velocity observation noise is unchanged.
- Validation: `compileall` passed for the modified observation/config files. Config checks confirmed flat/stair training actors receive the encoder noise parameters and `leg_encoder_bias` reset event, while flat/stair play configs and critic terms remain clean.

## 2026-06-18 - CTBC per-side outcome rewards and scratch restart

- Observation: in `remote_watch`, trained stair policies could climb but often preferred to square up to the step before climbing. Even with yaw-PID targets that made the robot approach the stair at an angle, many robots waited or shuffled until both sides contacted the riser, then climbed front-on.
- Observation: when only one side reached the stair first, the policy often did not immediately use that side's CTBC trigger to lift/retract and advance. This behavior did not match the desired asymmetric climbing behavior where the first side to hit the step should climb first.
- Diagnosis: the previous CTBC reward shaping mostly measured global/no-progress behavior after a trigger, so it did not strongly credit the specific triggered side for gaining wheel height or making target-direction progress. This likely allowed the policy to solve stairs by synchronized/front-on climbing instead of per-side obstacle negotiation.
- Change: removed the old `stair_ctbc_no_progress` reward term.
- Added per-side CTBC outcome terms keyed to the actually triggered side: `stair_ctbc_side_height_gain`, `stair_ctbc_side_forward_progress`, and `stair_ctbc_side_no_outcome`. They track the triggered side wheel height/progress from the CTBC trigger start and reward/penalize concrete outcome instead of global no-progress behavior.
- Validation: `compileall` passed for `src/se3_train/tasks/stair/rewards.py` and `src/se3_train/tasks/stair/env_cfg.py`; config checks confirmed the three new CTBC terms are active and `stair_ctbc_no_progress` is absent.
- Remote restart: synced current code into `/workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg_pr24_review_walk800_obs34_v18_20260615` and launched a from-scratch `SE3-WheelLegged-Stair-GRU` run with `8192` envs, `3800` iterations, and `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7`.
- Run: `2026-06-17_16-54-53_pr24_obs34_ctbc_sideoutcome_scratch_total3800_8192_gpu1to7_20260618`; log path `/tmp/train_pr24_obs34_ctbc_sideoutcome_scratch_total3800_20260618.log`; PID `2063710`.
- Verification: the remote command uses `--agent.resume False`; logs show `Learning iteration 0/3800`, `Curriculum/terrain_levels/walking_phase=1`, `Stair/diag_ctbc_kff=0.0`, `Stair/diag_ctbc_local_iter=0.0156`, actor `GRU(34, 512)`, and GPU 0 unused.

## 2026-06-18 - CTBC contact-gated time-decay reward cleanup

- Change: removed the conflicting active CTBC shape rewards `stair_feet_clearance`, `stair_feet_air_time`, `stair_contact_number`, and `stair_wheel_swing_zero_vel` from the stair task. The first two rewarded wheel lift/air time, and `stair_contact_number` treated triggered-side no-contact as a match, which conflicted with the desired side-wall-contact climbing behavior.
- Change: deleted those four reward functions from `src/se3_train/tasks/stair/rewards.py` and removed them from `__all__`.
- Change: `stair_ctbc_side_height_gain` and `stair_ctbc_side_forward_progress` now require triggered-side contact with terrain through `wheel_riser_sensor` before paying reward. The gate no longer filters for side-face normals, so low-stair top/edge contact is valid while wheel air time is still not rewarded. Both rewards also multiply by a 0.30 s CTBC-window decay weight with power `2.0`, so outcome achieved late in the trigger window is worth less.
- Change: `stair_ctbc_side_no_outcome` now checks an early window from `0.06 s` to `0.18 s` and penalizes a triggered side unless it has terrain contact plus either at least `0.025 m` height gain or `0.035 m` target-direction progress.
- Active CTBC outcome weights: `stair_ctbc_side_height_gain=0.25`, `stair_ctbc_side_forward_progress=0.8`, and `stair_ctbc_side_no_outcome=-1.5`.
- Validation: `compileall` and `ruff check` passed for `src/se3_train/tasks/stair/rewards.py` and `src/se3_train/tasks/stair/env_cfg.py`; static search confirmed the four removed reward names no longer exist in `src/se3_train/tasks/stair`, and config checks confirmed the CTBC outcome rewards no longer carry `contact_normal_z_max`.

## 2026-06-18 - Stair height range 0.04-0.22 and model_800 warm start

- Change: widened the stair terrain height curriculum range from `(0.05, 0.20)` to `(0.04, 0.22)`. The task now uses a single `_STAIR_STEP_HEIGHT_RANGE` constant for both `BoxInvertedPyramidStairsTerrainCfg.step_height_range` and the `obs_steps_climbed` diagnostic reward's `step_height_range`.
- Validation: `compileall` and `ruff check` passed for `src/se3_train/tasks/stair/env_cfg.py` and `src/se3_train/tasks/stair/rewards.py`; config checks printed terrain `step_height_range=(0.04, 0.22)` and `obs_steps_range=(0.04, 0.22)`.
- Remote source checkpoint: reused `model_800.pt` from `2026-06-17_16-54-53_pr24_obs34_ctbc_sideoutcome_scratch_total3800_8192_gpu1to7_20260618`.
- Remote start: synced current code into `/workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg_pr24_review_walk800_obs34_v18_20260615` and launched warm-start training with `--warm-start-iteration 800` and `--stair-local-iter-offset 800`, so both terrain/command curriculum and CTBC local iteration start at `800`.
- Run: `2026-06-18_09-22-10_pr24_obs34_h004022_ctbc_terraincontact_m800_warm_total3800_8192_gpu1to7_20260618`; log path `/tmp/train_pr24_obs34_h004022_ctbc_terraincontact_m800_20260618.log`; PID `3492057`; GPUs `1-7`.
- Verification note: the launch command confirmed `--agent.load-run 2026-06-17_16-54-53_pr24_obs34_ctbc_sideoutcome_scratch_total3800_8192_gpu1to7_20260618 --agent.load-checkpoint model_800.pt`. Follow-up log inspection was blocked by an SSH timeout to `target-via-phone`.

## 2026-06-18 - Stronger stair body collision and posture penalties

- Change: removed the CTBC discount from the stair wheel contact-force penalty. `contact_forces` now uses `contact_forces_stair`, which keeps the same wheel-force penalty during CTBC feedforward triggers instead of multiplying it by `1 - 0.5 * kff`.
- Change: strengthened base body terrain collision penalty for the stair task: `collision` weight `-16.0 -> -32.0`, with `use_recovery_gate=False` so `base_link` terrain contact is penalized directly.
- Change: strengthened body posture control for stair training: `tracking_orientation_l2` weight `-12.0 -> -18.0`; `bad_tilt` weight `-6.0 -> -10.0`, soft limit `10 deg -> 8 deg`, hard limit `30 deg -> 25 deg`, and max penalty `4.0 -> 6.0`.
- Validation: `compileall` and `ruff check` passed for `src/se3_train/tasks/stair/rewards.py` and `src/se3_train/tasks/stair/env_cfg.py`; config export confirmed `contact_forces_stair`, `collision=-32.0`, `tracking_orientation_l2=-18.0`, and `bad_tilt=-10.0`.

## 2026-06-18 - Stronger left-right leg symmetry penalty

- Change: added a stair-task override for `joint_mirror`, increasing the left-right hip/knee symmetry penalty from the inherited flat weight `-0.179` to `-0.35`.
- Change: wrapped the stair `joint_mirror` term with `joint_mirror_no_ctbc`, so the strengthened symmetry penalty is disabled during the CTBC trigger window and does not suppress the desired short single-side climbing motion.
- Change: the mirror shutdown gate is based on `StairClimbState.contact_triggered()` rather than `kff`, so it stays tied to the active CTBC phase window even when feedforward amplitude is annealed.
- Validation: `compileall` and `ruff check` passed for `src/se3_train/tasks/stair/rewards.py` and `src/se3_train/tasks/stair/env_cfg.py`; config export confirmed `joint_mirror=-0.35` and `func=joint_mirror_no_ctbc`.

## 2026-06-18 - Stair body height 0.385 and walking random vx profile

- Change: raised the stair command body-height curriculum maximum from `0.37` to `0.385` using `_STAIR_BODY_HEIGHT_RANGE=(0.24, 0.385)`. The height curriculum reaches this range at PPO iter `600` and keeps it at stair-phase entry iter `800`.
- Change: added a walking-phase `lin_vel_profile` to `VelocityHeightCommandTerm`. During PPO iters `0-800`, moving envs keep a randomly sampled `vx` for `0.6-2.0 s`, then resample a new random `vx` from the current walking curriculum range. The profile is disabled after iter `800`, so the stair phase keeps the normal positive speed curriculum.
- Change: raised the stair-phase speed curriculum maximum to `1.5 m/s`: iter `800` starts at `vx=(0.6, 1.0)`, then iters `1050`, `1300`, and `1600` use `vx=(0.6, 1.5)`.
- Validation: `compileall` and `ruff check` passed for `src/se3_train/mdp/commands.py` and `src/se3_train/tasks/stair/env_cfg.py`; config export confirmed `lin_vel_profile_iteration_range=(0, 800)`, `lin_vel_profile_resampling_time_range_s=(0.6, 2.0)`, velocity stages ending at `(0.6, 1.5)`, and height stages ending at `(0.24, 0.385)`.
- Remote note: an interrupted pre-`1.5 m/s` scratch launch named `pr24_obs34_h0385_randvx_scratch_total3800_8192_gpu1to7_20260618` had already started; it was stopped before the corrected run was launched.
- Remote restart: synced the corrected code into `/workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg_pr24_review_walk800_obs34_v18_20260615` and launched scratch training with `8192` envs, `3800` iterations, and `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7`.
- Run: `2026-06-18_15-47-03_pr24_obs34_h0385_randvx_v15_scratch_total3800_8192_gpu1to7_20260618`; log path `/tmp/train_pr24_obs34_h0385_randvx_v15_scratch_20260618.log`; PID `4005886`.
- Verification: remote source grep confirmed `_STAIR_BODY_HEIGHT_RANGE=(0.24, 0.385)`, `lin_vel_profile_resampling_time_range_s=(0.6, 2.0)`, and stair velocity stages `(0.6, 1.5)`. Training logs show `--agent.resume False`, `Learning iteration 0/3800`, `Curriculum/terrain_levels/walking_phase=1`, `Curriculum/command_height/height_max=0.2200`, `Stair/diag_ctbc_local_iter=0.0156`, and GPU 0 unused.
  关键对比在 rc_on 的瞬间：
 
 
  closedchain，t=1.00s：
 
 
  - x≈0.001, z≈0.140, vx≈-0.041, tilt≈4.4deg
  - active angle [1.459, 1.457]
  - output/front 状态约 [2.935, -1.370, -1.995, 1.374]
 
 
  fourbar_visualbase，t=1.00s：
 
 
  - x≈0.046, z≈0.138, vx≈0.254, tilt≈0.9deg
  - active angle [1.231, 1.263]
  - output/front 状态约 [0.703, -1.172, -0.707, 1.198]
 
 
  active angle 都还在范围内，但 front 杆绝对相位完全不是一个分支。closedchain no-torque 时前杆能甩到另一个相位，policy obs 里的 sin/cos(front_delta) 和 GRU 状态就变成训练没覆盖的分布。之后接管时：
 
 
  - closedchain t=1.5s：z≈0.293, vx≈0.17，还在恢复
  - fourbar t=1.5s：z≈0.381, vx≈1.10，已经站高并冲向台阶
 
 
  低层同 action step response 也能看到差异：同样给 action=[0,-1,0,-1,0,0]，fourbar 约 0.2s active angle 就接近 0；closedchain 到 0.2s 还在 0.4rad 左右，约 0.5s 才接近 0。
 