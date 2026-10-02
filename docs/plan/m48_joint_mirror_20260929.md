# M48：M47 + joint_mirror −0.179 → −5（2026-09-29，用户定）

## 背景

用户在 sim2x 看 M47-2600 时认为 joint_mirror 基本没有约束腿部姿态。M47 到 2794 轮、近 200 轮均值：

| 量 | 值 |
|---|---|
| `Recovery/diag_joint_mirror`（原始值，rad²） | 0.152（每对关节左右差约 22°） |
| `Episode_Reward/joint_mirror` | −0.014 |
| `Episode_Reward/tracking_lin_vel` / `upward` | +2.27 / +2.03 |
| `flat_base_height` / `tracking_orientation_l2` | −0.29 / −0.20 |

joint_mirror = (前杆角左右差² + 主动杆角左右差²)/2 × 直立门控。复旦上台阶3 `nominal_state` = −1.0·(θ0左 − θ0右)²（只罚虚拟腿摆角），
按每对关节折算是本仓库 −0.179 的 11 倍，再按跟踪总权重（本仓库 9 对复旦 3）折算约 30 倍。

## 改动（相对 M47 唯一差异）

`env_cfg(joint_mirror_weight=-5.0)`：joint_mirror 权重 −0.179 → −5，公式不变。

| 左右差（两对相同） | −0.179 每秒 | −5 每秒 |
|---|---|---|
| 5° | −0.001 | −0.04 |
| 10° | −0.005 | −0.15 |
| 22° | −0.03 | −0.76 |
| 30° | −0.05 | −1.37 |

## 注意

- 指令有 roll ±0.1 rad，做 roll 需要左右腿长不同；joint_mirror 连主动杆角（腿长）一起罚，加权后 roll 跟踪可能变差（复旦只罚摆角）。
- 上台阶时单腿先抬、分腿也会被罚，看 stairs_up 等级与通关图是否退化（M39-4999 通关 24/24）。
- M47（删 bad_orientation）约 2800 轮按用户指令停止，未跑满；M48 与 M47 同批量，按轮次对比。

## run 设置

`nulltask-5c45cdd89b-whtws` 七卡 × 8192、5000 轮、保存 200、seed 42，在线 W&B project `SE3-WheelLegged-Rough`。

## 判据

- `Recovery/diag_joint_mirror` 降到 0.03 rad²（约 10°）以内。
- 上台阶通关图、平地速度与 yaw 跟踪、roll 跟踪对 M39-4999（`.scratch/final_pick/eval_all.py`）。

## 启动记录

代码 commit `be494d1`，whtws 仓库由 4398698 快进到同一 commit；41 项 rough 测试通过，本地入口 CPU smoke 5 轮通过。
配置核验：相对 M47 只改 joint_mirror 权重，奖励键、终止项逐项相同。M47 在 2858 轮按用户指令停止。

- 启动时间：2026-09-29 19:02（Pod 时区），七卡 × 8192、5000 轮、保存 200、seed 42，在线 W&B。
- run：`2026-09-29_11-02-47_rough-M48-nobadori-mirror5-seed42-7x8192-5k`，
  W&B [76yn2t4h](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/76yn2t4h)，PID/PGID `2683670`，
  state `/workspace/.se3-training-state/whtws/20260929T110240Z`。
- 首轮核验：第 12 轮在迭代、无 Traceback、无 nefc overflow，3.14 s/轮，七卡各 13.0–13.5 GB、利用率 84–86%；
  训练日志奖励表 `joint_mirror` −5.0，远端 env.yaml 无 `bad_orientation`。
