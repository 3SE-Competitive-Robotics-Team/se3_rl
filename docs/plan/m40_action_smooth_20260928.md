# M40：M39 + action_smoothness 权重 −0.12 → −0.06（2026-09-28，用户定）

## 背景

M39（action_rate −0.48 → −0.10，docs/plan/m39_action_rate_20260928.md）跑满 5000 轮：课程 6.8、台阶列 vx 0.81、
撞面俯仰 ±3°、静站无抖动，σ 0.34 → 0.41；平滑罚实付从 −0.56 涨到 −0.90/s，成了台阶列最大的动作罚。用户决定把
`action_smoothness` 也减半。

## 权重选择

策略噪声的每维成本近似 k = 2·w_rate + 6·w_smooth（相邻差方差 2σ²、二阶差分 6σ²），M38 → M39 实测 k 降 45%、σ 升 21%
（指数约 0.32）。按此外推，在 action_rate −0.10 下：

| w_smooth | 腿的 k | 预计 σ |
|---|---|---|
| −0.12（M39） | 0.92 | 0.41（实测） |
| **−0.06** | 0.56 | 约 0.48 |
| −0.03 | 0.38 | 约 0.55 |
| −0.01（Flat 旧值） | 0.26 | 约 0.62 |

D 系列经验：σ 过 0.5 后站立极限环与抖动开始出现，−0.06 是这条线以内能迈的最大一步。轮分量定价（`wheel_scale` 2.0）与封顶 320 不动。

## 改动（相对 M39 唯一差异）

任务入口 `SE3-WheelLegged-Rough-Exp-ActionRate010Smooth006`：`env_cfg(action_rate_weight=-0.10, action_smoothness_weight=-0.06)`。
配置逐项对比：只有 `action_smoothness.weight −0.12 → −0.06`，其余 24 项与 M39 逐位相同。

## run 设置

`nulltask-5c45cdd89b-whtws` 七卡 × 8192、5000 轮、保存 200、seed 42，在线 W&B project `SE3-WheelLegged-Rough`，与 M38/M39 同批量。

## 判据

- 噪声：`Policy/mean_std` 不超过约 0.5、`Locomotion/max_abs_action`；`jitter_check.py` 静站俯仰 rms 与腿峰峰不比 M39（1.6–2.0° / ≤9°）明显变大。
- 台阶：课程等级、台阶列 vx 不低于 M39（6.8 / 0.81）；riser_retry_diag 首撞俯仰与首撞到过沿时间；climb_map 对 M39-4999（卡 8 / 摔 15）。
- 终止：灾难终止尾部（M39 0.046）。

## 启动记录

代码 commit `28faab7`，whtws 仓库由 a724b49 快进到同一 commit；本地入口 CPU smoke 5 轮通过（action_rate −0.1、action_smoothness −0.06）；启动器 `--dry-run` 通过。

- 启动时间：2026-09-28 19:21（Pod 时区），七卡 `--gpu-ids all` × 8192、5000 轮、保存 200、seed 42，在线 W&B。
- run：`2026-09-28_11-21-00_rough-M40-actionrate010-smooth006-seed42-7x8192-5k`，
  W&B [j5zwvw1x](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/j5zwvw1x)，PID/PGID `2216716`，
  state `/workspace/.se3-training-state/whtws/20260928T112053Z`。
- 首轮核验：第 36 轮在迭代、无 Traceback、无 nefc overflow，3.13 s/轮，七卡 13.0–13.5 GB、利用率 80–86%；
  远端 `params/env.yaml`：`action_rate.weight −0.1`、`action_smoothness.weight −0.06`。
