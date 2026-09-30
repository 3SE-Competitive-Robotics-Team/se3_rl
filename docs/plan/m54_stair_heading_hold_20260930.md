# M54：M53 + 台阶列航向保持、轮前后错位放宽、出生朝向 ±15°（2026-09-30，用户定）

## 背景

用户在 sim2x 看到 M53 斜向上台阶时，没给 yaw 指令也会转一下让两轮回到同一级。记账（`.scratch/m53_yaw_ledger/`，
MJLab CPU、stairs_up 第 9 级 20 cm、vx 1.0、yaw 0、M53 model_3600）：

- 转动集中在第一次撞立面的约 0.2 s：先触立面的轮子水平接触力 100–136 N、被挡住停转，另一侧继续约 1.2 m/s，
  机身以 1–2.6 rad/s 绕被挡轮转正，随后两轮 0.1 s 内同抬 20 cm。
- 是被动接触扭转，不是主动差速：出生偏 30° 自由控制转 −24°，**强制左右轮速度动作一致反而转 −32°**；
  策略在加大被挡轮前进指令，轻微抵抗。
- 定价上没有理由去学别的上法：转动只让 tracking_ang_vel 少拿约 0.6，转完后航向变化不计价；单轮先上则被侧倾罚、
  轮前后错位罚（M49 为"不要一先一后走梯"加的）持续扣分；腿长行程 19.6 cm < 最高台阶 20 cm，斜 30° 不转正物理上
  只能大侧倾硬爬；斜向撞立面每回合只发生在第一道（转正后后面都正对）。

## 改动（相对 M53）

均只作用于台阶列（stairs_up、stairs_two_step_up，`ROUGH_REWARD_TERRAIN_TYPE_NAMES`）：

| 项 | M53 | M54 |
|---|---|---|
| 航向保持 `stair_heading_hold`（新增） | 无 | 航向误差² (rad²) × −6；目标 = 回合起点航向 + ∫yaw 指令（台阶列 yaw 指令恒 0，即出生朝向） |
| 轮前后错位 `wheel_fore_aft_offset` | −50·Δx² 全列 | 台阶列 ×0.25（`column_scaled`），其余列 −50 不变 |
| 台阶列出生朝向 | 正对某面 ±30° | ±15° |

航向保持每秒扣分：偏 15° 约 −0.41、偏 30° 约 −1.64，转完之后持续计价，策略要么抵抗被动扭转，要么转完再转回来。
轮前后错位台阶列 Δx 20 cm 每秒 −2.0 → −0.5，给单轮先上留余地。用户否决了"台阶列放宽 roll 定价"（姿态罚保持原函数）。

实现：`rough.rewards.stair_heading_hold`，目标航向缓存在 env 上，`episode_length_buf <= 1`（MJLab 先加步数、再算奖励、
最后 reset，reset 后第一步恰为 1）时取当前航向，之后按指令 yaw 角速度积分；另记 `Rough/stair_heading_error_deg`。

## 验证

- 配置核验：M54 相对 M53 新增 `stair_heading_hold`（−6，台阶两列），`wheel_fore_aft_offset` 改为 `column_scaled`（台阶两列 ×0.25、
  权重 −50），出生朝向半幅 0.524 → 0.262 rad，姿态罚仍为原 `tracking_orientation_l2`。
- 功能验证（`.scratch/m54_heading_check.py`，台阶列 + 平地列各 1 env）：静止 5 步漂 2.6° 扣 −0.012；人为转 20° 后记录误差 17.4°、
  扣 −0.553（= −6 × 0.304²）；平地列恒 0；reset 后误差归零。
- `tests.test_rough_port` + `tests.test_onnx_metadata` 70 项通过；M54 入口 CPU smoke（1 env、5 轮）通过。

## 启动记录

代码 commit `01e87e2`，Pod 由 `fd4fd76` 经 git bundle 快进。按用户指令停掉其余训练腾出七卡：M52 已于 4999 轮正常结束
（W&B 补传完成，残留僵尸进程可忽略）；M53 在 4261 轮 SIGINT 停止（最后 checkpoint `model_4200`，状态目录留 NOTE）。

- 启动时间：2026-09-30 18:45（Pod 时区），七卡 × 8192、5000 轮、保存 200、seed 42，从头训，在线 W&B。
- run：`2026-09-30_18-45-35_rough-M54-headinghold6-fa025-yaw15-seed42-7x8192-5k`，
  W&B [vjmjllbj](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/vjmjllbj)，PID/PGID `3314582`，
  state `/workspace/.se3-training-state/whtws/20260930T184528Z`。
- 首轮核验：第 14 轮在迭代、无 Traceback、无 nefc overflow，3.25–3.28 s/轮，七卡各 13.2–13.6 GB、利用率 81–85%；
  远端 env.yaml 与 M53 只差三处：出生朝向半幅 0.524 → 0.262 rad、`wheel_fore_aft_offset` 包成 `column_scaled`（台阶两列 ×0.25）、
  新增 `stair_heading_hold`（−6，台阶两列）。前 500 轮平地热身期没有台阶列，`Rough/stair_heading_error_deg` 为 0 属正常。
- 批量 7 × 8192，M53 是 3 × 8192，对比带批量混杂。
