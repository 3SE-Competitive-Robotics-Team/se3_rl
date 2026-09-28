# M39：M38 默认 + action_rate 权重 −0.48 → −0.10（2026-09-28，用户定）

## 背景

M38（已定为默认，docs/plan/m38_upward_20260928.md）台阶列每秒账本里 action_rate 实付 −0.85，是台阶列第二大罚项
（第一是高度罚/orientation 合计）。M34-7800 反事实账本显示它在 0.73–0.89 m/s 之间不随速度变（−0.55…−0.58/s），
即它罚的是动作抖动而不是爬升本身；Flat 线 D 系列曾测得收敛后 action_rate 的 72–81% 是探索噪声地板。
用户决定把权重减到 −0.10 看一次。

## 改动（相对 M38 默认唯一差异）

任务入口 `SE3-WheelLegged-Rough-Exp-ActionRate010`：`env_cfg(action_rate_weight=-0.10)`，只改 `action_rate` 项的权重，
函数与 leg/wheel 分量参数不动；`action_smoothness`（−0.12）不动。配置逐项对比：只有 `action_rate.weight −0.48 → −0.1`。

## run 设置

`nulltask-5c45cdd89b-whtws` 七卡 × 8192、5000 轮、保存 200、seed 42，在线 W&B project `SE3-WheelLegged-Rough`，与 M38 同批量。

## 判据

- 动作噪声：`Policy/mean_std` 与 `Locomotion/max_abs_action`（动作罚减小后 σ 平衡点会上移，见记忆"σ 平衡点"）；
  确定性回放里腿关节的极限环幅值（`.scratch/m15_diag` 站立/行进峰峰）不显著变大。
- 台阶：`Rough/base_vx_stairs`、`Curriculum/terrain_levels/stairs_up` 不低于 M38 同期（0.72–0.75 / 5.4）；
  riser_retry_diag 的首撞俯仰、首撞到过沿时间；climb_map 通关格与摔倒格对 M38-4999（3 卡 / 20 摔）。
- 平地：`.scratch/m15_diag/sweep.py` 低速段实速与站立抖动。
- 终止：`Episode_Termination/catastrophic_state` 尾部（M38 4000 轮后升到 0.11）。

## 启动记录

代码 commit `a724b49`，whtws 仓库由 02946a4 快进到同一 commit；本地入口 CPU smoke 5 轮通过（action_rate −0.1）；启动器 `--dry-run` 通过。

- 启动时间：2026-09-28 12:42（Pod 时区），七卡 `--gpu-ids all` × 8192、5000 轮、保存 200、seed 42，在线 W&B。
- run：`2026-09-28_04-42-40_rough-M39-actionrate010-seed42-7x8192-5k`，
  W&B [uouikynn](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/uouikynn)，PID/PGID `2088093`，
  state `/workspace/.se3-training-state/whtws/20260928T044233Z`。
- 首轮核验：第 29 轮在迭代、无 Traceback、无 nefc overflow，3.13 s/轮，七卡 13.0–13.5 GB、利用率 84–87%；
  远端 `params/env.yaml`：`action_rate.weight −0.1`、`upward.weight 1.0`。
