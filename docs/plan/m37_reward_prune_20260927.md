# M37：M35 基线上删掉五项重叠定价（2026-09-27，用户定）

## 背景

M35 配置共 29 项奖励，同一物理量被多条曲线定价。按 W&B 上 M35 末 100 轮的实付把每项的形状算出来
（`.scratch/m35_eval/`，形状表见下），用户决定删速度、倾斜、对称三组里的重叠项；正则类小项（能耗、限位、角动量等）保留。

## 形状分析

x 速度三项在平地列（宽核 6·exp(−e²/0.5)、窄核 3·exp(−e²/0.04)、违令罚 −2·((|e|−0.05)/3)²）：

| 误差 m/s | 宽核 | 窄核 | 违令罚 | 合计斜率 /(m/s) |
|---|---|---|---|---|
| 0.10 | 5.88 | 2.34 | 0 | −14 |
| 0.30 | 5.01 | 0.32 | −0.01 | −11 |
| 1.00 | 0.81 | 0 | −0.20 | −4 |
| 1.50 | 0.07 | 0 | −0.47 | −1 |

- 窄核只在 0–0.3 m/s 有用，宽核管 0.3–1.2；违令罚在 `lin_vel_scale` 3.0 下全程斜率不到 1，是常数税不是远端梯度（M35 台阶列实付 −0.68/s）。
- `tracking_lin_yaw_joint` 只在 |vx| ≥ 0.2 且 |yaw| ≥ 0.5 时开，台阶列 yaw 指令 ±0.3 永远开不了，全列实付 0.12/s，与 `tracking_ang_vel` 在同一误差上重复计酬。

倾斜：`tracking_orientation_l2`（−12·rad²）与 `bad_tilt`（−6，10° 起 barrier）在 15° 以上是同一量的两条曲线，20° 处贡献相等
（−1.46 / −1.50），30° 以上已由 `bad_orientation` 终止 + 摔倒罚接手；`ang_vel_xy` 罚的是角速度，与前两项不重合，保留。

对称：`joint_mirror`（关节角差，全列，实付 0.01–0.04/s）、`wheel_fore_aft_offset`（Δx²，平地列）、`wheel_height_diff`（Δz²，台阶列）。
用户决定只保留 joint_mirror 一条关节空间定价，两项轮几何罚删除。

## 改动（相对 M35 唯一差异：整组删除以下五项，其余 24 项函数、权重、参数逐位相同）

| 删除项 | M35 权重 | M35 全列实付 /s | 台阶列实付 /s |
|---|---|---|---|
| command_velocity_error | −2 | −0.36 | −0.68 |
| tracking_lin_yaw_joint | 2 | +0.12 | 0 |
| bad_tilt | −6 | −0.11 | −0.17 |
| wheel_fore_aft_offset | −40 | −0.03 | 0 |
| wheel_height_diff | −40 | −0.02 | −0.11 |

任务入口 `SE3-WheelLegged-Rough-Exp-HeightWindowDz5Prune5`：`env_cfg(..., dropped_rewards=ROUGH_M37_DROPPED_REWARDS)`，
在奖励表组装完成后整项删除。五项一起删，只能判断组合效果。

## run 设置

nulltask1 三卡（GPU 0–2）× 8192 envs、4000 轮、保存间隔 200、seed 42，与 M35 同批量，W&B project `SE3-WheelLegged-Rough`。

## 判据

- 与 M35 同轮次比：台阶列实际 vx、stairs_up 等级、灾难终止；台阶列净回报预计上移约 0.9/s（主要是违令罚常数税），
  宽核与支撑、进度奖励的相对权重不变。
- 删 `wheel_height_diff` 后走梯（左右轮高差 14–20 cm）是否回来：`Rough/wheel_dz_abs_stairs` 与 riser_events 的抬轮时序。
- 删 `bad_tilt` 后大倾角是否增多：`Episode_Termination/bad_orientation`、`tracking_orientation_l2` 实付。
- 每个 checkpoint 跑确定性回放（`.scratch/m34_eval/stair_ledger.py` 改路径与开关；`.scratch/m35_eval/riser_retry_diag.py`）。

## 启动记录

代码 commit `9e07b44`（子模块 `e753ce6`），Pod 已 ff 到同一 commit；本地入口 CPU smoke 5 轮通过；DryRun 通过。

- 启动时间：2026-09-27 22:02（Pod 时区），GPU 0–2，三卡 × 8192、4000 轮、保存 200、seed 42。
- run：`2026-09-27_22-02-50_rough-M37-dz5-prune5-seed42-3x8192-4k`，
  W&B [8bmlcdk4](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/8bmlcdk4)，PGID 591544，
  state `/workspace/.se3-training-state/nulltask1/20260927T140244Z-29981`。
- 首轮核验：第 21 轮在迭代、无 Traceback、无 nefc overflow，3.11 s/轮，三卡 11.6–12.0 GB、利用率 84–85%，已导出 `model_0.onnx`。
  远端 `params/env.yaml` 奖励表 24 项，五项已不在表中，`flat_base_height.dead_zone_m: 0.05`。
- 启动前 W&B 网关计划任务处于 Ready（M35/M36 停止后自行退出），`schtasks /Run` 后网关就绪、Pod 经代理可达 api.wandb.ai。
