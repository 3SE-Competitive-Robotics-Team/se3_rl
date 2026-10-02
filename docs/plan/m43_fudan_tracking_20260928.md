# M43：M42 + 速度跟踪核换成复旦的尖核 + 有界宽核（2026-09-28，用户定）

## 背景

本仓库的速度跟踪是双核且分列：宽核 6·exp(−e²/σ)（平地 σ 0.5、静站 0.1、台阶列 1.44）+ 窄核 3·exp(−e²/0.04)（台阶列 w 1）。
台阶列 σ 1.44 是全场最平的核（半额处误差 1.0 m/s），是账本里"台阶列定价平坦"的来源（M34-7800 账本）；窄核只在 0.3 m/s 内有用。
复旦上台阶3 全地形统一用一对：

| | 核 | 权重 | 半额处误差 |
|---|---|---|---|
| tracking_lin_vel | exp(−e²/0.25) | 1.5 | 0.42 m/s |
| tracking_lin_vel_enhance | exp(−e²/2.5) − 1 | 1.5 | 1.3 m/s |

尖核给小误差处的奖励，宽核减一是有界罚（0 到 −1.5）给远端梯度；核比我们的宽核尖，噪声直接扣跟踪分，也是复旦 σ 能收住的原因之一。

## 改动（相对 M42 唯一差异：速度跟踪核）

`env_cfg(tracking_kernel="fudan")`：

- `tracking_lin_vel` → `mdp.rewards.tracking_lin_vel`，σ_move = σ_stand = 0.25、vz 0、无门控、w 1.5，全列统一（不再分台阶列）。
  仍用原函数是为了保留速度课程读的 `Locomotion/tracking_lin_vel_reward_curriculum` 日志键。
- 新增 `tracking_lin_vel_enhance`（`rough/rewards.py`）= exp(−e²/2.5) − 1，w 1.5，全列。
- 删除 `tracking_lin_vel_narrow`。
- 其余与 M42 相同（腿 scale 0.5、轮 10、腿 kp 20 / kd 1.5、action_rate −0.05、action_smoothness −0.05、upward 1.0 等）。

配置核验：相对 M42 删 1 项、加 1 项、改 1 项，其余逐位相同。

## 注意

- 平地速度课程的推进阈值 0.5 是按 σ 0.5 的核定的，σ 0.25 下同样误差的跟踪分更低，课程可能推得更慢；看 `Curriculum/command_vel/lin_vel_x_max` 到 2.4 的轮次。
- 台阶列从 σ 1.44 直接到 0.25，早期爬台阶的速度回报变少，看 stairs_up 等级起步是否变慢；有界罚在台阶列每秒最多 −1.5。
- 跟踪总权重从 6+3 变成 1.5+1.5，upward（+4）与支撑/进度奖励的相对权重随之变大。

## run 设置

`nulltask-5c45cdd89b-whtws` GPU 2–3 两卡 × 8192、5000 轮、保存 200、seed 42，在线 W&B；与 M42（GPU 0–1）并行、同批量。

## 判据

- σ：`Policy/mean_std` 对 M42 同轮次；静站抖动 `jitter_check.py`。
- 跟踪：`Locomotion/base_vx_error_abs`、`Rough/base_vx_stairs` 对 M42；课程 `lin_vel_x_max`、`stairs_up`。
- 台阶：riser_retry_diag、climb_map 对 M42 同 checkpoint。

## 启动记录

代码 commit `1361ba8`，whtws 仓库由 f4f515c 快进到同一 commit（M42 在跑，后续 M42 ONNX 的 git HEAD 会记 1361ba8）；
本地入口 CPU smoke 5 轮通过（tracking_lin_vel 1.5、tracking_lin_vel_enhance 1.5、课程键正常记录）。
启动器新增 `--allow-concurrent`（只要求申请的 GPU 显存 < 1 GiB，不再因已有 se3-train 进程拒绝），`--dry-run` 通过。

- 启动时间：2026-09-28 23:34（Pod 时区），GPU 2–3，两卡 × 8192、5000 轮、保存 200、seed 42，在线 W&B；与 M42（GPU 0–1）并行。
- run：`2026-09-28_15-34-48_rough-M43-fudantracking-seed42-2x8192-5k`，
  W&B [6lp5bbmy](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/6lp5bbmy)，PID/PGID `2302488`，
  state `/workspace/.se3-training-state/whtws/20260928T153441Z`。
- 首轮核验：第 50 轮在迭代、无 Traceback、无 nefc overflow，3.17 s/轮，GPU 2–3 各 11.0–11.4 GB、利用率 84–86%（M42 在 0–1 同时 82–83%）；
  远端 `params/env.yaml`：`tracking_lin_vel` 1.5（σ 0.25/0.25、vz 0、无门控）、`tracking_lin_vel_enhance` 1.5（σ 2.5）、
  `tracking_lin_vel_narrow` 不存在、共 25 项；腿 scale 0.5 / 轮 10、腿 stiffness 20 / damping 1.5、action_rate −0.05、action_smoothness −0.05、upward 1.0。
