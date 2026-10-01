# J7：J4 + 无下蹲参考（flag 一到就蹬）（2026-10-01，用户定）

## 动机

用户要的是"jump_flag 一到就蹬，不下蹲"。J5 / J6（jump_flag 窗口 + 按状态给奖励 / 起跳段解析参考奖励，无参考帧）
都没学会起跳（J5 约 1800 轮、J6 约 1000 轮全部漏跳），用户定回到已经成功的 J4 上改，一次一个变量。

## 改动（单变量：参考轨迹）

`SE3-WheelLegged-Jump-Mimic-Exp-J7` = J4 + `reference_dir = assets/trajectories/jump_ref_v2_nocrouch_h022`：

- 参考由 `se3-jump-to --heights 0.20 0.30 0.40 0.50 --stand-height 0.22 --no-crouch` 生成：去掉 J4 参考开头的站立 0.2 s 与
  下蹲 0.35 s，第 0 帧即从 0.22 m 站姿恒加速度蹬地；腾空、缓冲、恢复、末尾站立与 v1 生成器相同。整条约 1.4 s（v1 约 2.0 s）。
- 站姿 0.22 m 取用户给的 0.20–0.24 中值（也是机器人默认站高）；J4 的站姿高度指令本就取参考站姿，随之从 0.28 变为 0.22。
- 其余全部照 J4：actor 54 维（含参考帧）、vx ±1.5、四项模仿奖励与权重、偏离终止 0.12 m、RSI 50%、执行链与 PPO。

| 目标 | 起跳速度 | 蹬地加速度 | 蹬地时长 |
|---|---|---|---|
| 0.20 m | 1.47 m/s | 7.0 m/s² | 0.209 s |
| 0.30 m | 1.72 m/s | 9.6 m/s² | 0.179 s |
| 0.40 m | 2.22 m/s | 16.0 m/s² | 0.138 s |
| 0.50 m | 2.62 m/s | 22.4 m/s² | 0.117 s |

0.50 m 的蹬地加速度高于 J4 下蹲版的 20.7（J4 该档腿电机已短时顶到 32 N·m），可能更难跟。准静态力矩检查在包络内（利用率峰值 0.72）。

## 验证

- J4 / J7 配置对比：只差参考目录与随之变化的站姿高度（0.28 → 0.22）。
- 功能检查（`.scratch/j7_check.py`，CPU 32 env）：四条参考都被触发（400 步 17 / 20 / 17 / 15 次）；RSI 写入后机身高度与腿长偏差 ≤ 4 cm；奖励与观测有限。
- `tests.test_onnx_metadata` + `tests.test_flat_baseline` 通过；CPU smoke（1 env、5 轮）通过。

## 启动记录

代码 commit `09493b2`，whtws 经 git bundle 由 `6cea122` 快进；GPU 0–5 仍被他人占用，用户定 GPU 6 单卡（样本约为 J4 的 1/6）。

- 启动时间：2026-10-01 18:07（Pod 时区），GPU 6 单卡 × 8192、5000 轮、每 200 轮保存、seed 42，从头训，
  W&B [qar24d7n](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Jump-Mimic/runs/qar24d7n)，PID/PGID `3770619`，
  state `/workspace/.se3-training-state/whtws/20261001T180708Z`。单卡 run 停训用 SIGTERM（SIGINT 不响应）。
- 首轮核验：第 84 轮无 Traceback、无 nefc overflow，1.34 s/轮，GPU 6 4.0 GB / 69%；`Jump/active_rate` 0.13，
  `mimic_deviation` 终止每次记录约 51 次（随机策略，与 J2–J4 开局同量级）。
