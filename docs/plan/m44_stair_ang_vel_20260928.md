# M44：M43 + 台阶列 yaw 角速度跟踪加回来（2026-09-28，用户定）

## 背景

M2 起 `tracking_ang_vel` 在台阶列置零（`tracking_ang_vel_off_terrain`，`terrain_type_names` = 台阶类列），理由是台阶列
yaw 指令 ±0.3 rad/s 小、且撞沿时机身被动扭转会被当成"跟踪差"。M43 把速度跟踪核换成复旦的全列统一一对之后，
台阶列剩下的分列差异里 yaw 跟踪是唯一"整项置零"的跟踪项；台阶列跨立步态（左右轮错位 12–27 cm）导致的单轮撞沿扭转
在台阶列没有任何 yaw 计酬去纠正。复旦全地形 `tracking_ang_vel` 不分列。

## 改动（相对 M43 唯一差异）

`env_cfg(stair_ang_vel_tracking=True)`：`tracking_ang_vel` 去掉 `off_terrain` 包装，恢复 Flat 原函数
`mdp.rewards.tracking_ang_vel`，参数逐位不变（w 3.0、σ 0.25、sigma_cmd_scale 0.4、ratio_blend 0.2、无直立门控），全列生效。
其余与 M43 相同（复旦双跟踪核、腿 scale 0.5 / 轮 10、腿 kp 20 / kd 1.5、动作罚 −0.05/−0.05、upward 1.0）。

## 注意

- 台阶列 yaw 指令小（±0.3），核 σ 0.25 + sigma_cmd_scale 0.4 下台阶列基本是"罚扭转"：撞沿被动扭转每秒最多丢 3。
- 撞沿扭转是否因此减少看 riser_retry_diag 的扭动次数与 `Rough/base_vx_stairs`；若台阶列摔倒/卡住增多，说明策略为保 yaw 不敢撞沿。

## run 设置

`nulltask-5c45cdd89b-whtws` GPU 4–5 两卡 × 8192、5000 轮、保存 200、seed 42，在线 W&B；与 M42（0–1）、M43（2–3）并行、同批量。

## 判据

- 对 M43 同轮次：`Episode_Reward/tracking_ang_vel`、`Rough/base_vx_stairs`、`stairs_up` 等级、`Policy/mean_std`。
- riser_retry_diag / climb_map 对 M43 同 checkpoint：扭动次数、卡住、摔倒。

## 启动记录

（启动后补）
