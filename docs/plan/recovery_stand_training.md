# 纯倒地自起到站立训练计划

> 状态：已实现首版并完成首轮长训审计。本文记录新任务 `SE3-WheelLegged-Recovery-Stand-GRU` 的训练边界，先只把倒地自起网络训出来；运行时双策略切换器不在本文范围内。

## 背景

此前 `SE3-WheelLegged-Recovery-GRU` 的方向是一个 GRU policy 同时覆盖站立、低速行走和全姿态自起。现在训练目标改为分离能力：

- `Flat-GRU` 负责平地运动。
- `Recovery-Stand-GRU` 只负责从随机姿态恢复到固定站立态。
- 后续部署时再由外部状态机切换两个策略；本文不设计切换器。

这次拆分的核心原因是：倒地自起阶段允许甚至需要机身、腿和轮子大幅运动，而平地速度跟踪会把策略压向“少动、稳态、跟速度”，两者在早期探索里有明显梯度冲突。

## 术语

| 术语 | 定义 |
|---|---|
| 纯自起策略 | 只处理倒地/异常姿态到稳定站立的策略，不负责恢复期间移动、转向或平地行走 |
| 固定站立高度 | 当前共享默认高度 `0.230340071m`，来自 `se3_shared.robot.default_base_height` 和 `_DEFAULT_STANDING_HEIGHT` |
| 可交接站立态 | 倾角、高度、速度、双轮接触、非轮接触和腿部几何对齐都满足成功窗口的站稳状态 |
| 非轮接触 | 腿部、机身或其他非轮部件与地面接触；恢复过程中允许，成功窗口内不允许 |
| 腿部几何对齐 | 左右轮保持正常横向轮距，且机身坐标系下前后错位不超过阈值；不允许两条腿前后劈叉 |

## 已决策

1. 新建任务 `SE3-WheelLegged-Recovery-Stand-GRU`，不覆盖旧任务。
2. recovery policy 从零训练，不从 flat checkpoint warm-start。
3. actor 观测维度保持现有 32D contract，不添加 recovery mode flag。
4. 指令固定为 `vx=0, yaw_rate=0, pitch=0, roll=0, height=default_base_height`。
5. reset 从第 0 iter 起使用全难度随机：base 姿态、yaw、关节位置、关节速度全部随机。
6. recovery 只训练恢复到固定站立态，不训练移动或转向。
7. episode 长度先设为 `5s`，成功后提前 reset。
8. 语义终止只保留 `time_out` 和 `recovery_success`；`bad_orientation`、`leg_contact` 不作为倒地自起硬终止。
9. timeout 不给强失败惩罚，只是不拿成功奖励。
10. success bonus 第一版为 `10.0`，只在完成连续稳定窗口时给一次。
11. 不显式奖励“越快越好”，只记录 `time_to_success`。
12. 训练诊断必须按初始倾角分桶，不能只看总成功率。
13. 训练后半程加入平面 push disturbance：只扰动 root 的 `x/y/yaw` 速度，不加入 `z/roll/pitch` 冲击；课程按 PPO iter 从 0 逐步放大到 `±1.0`。

## 非目标

- 不实现运行时 supervisor / 双策略切换器。
- 不训练 recovery policy 执行平地速度跟踪。
- 不做 MoE、蒸馏或把两个策略重新合成一个 checkpoint。
- 不使用 checkpoint warm-start。
- 不把 `0°~45°` 正常姿态样本当作平地行走训练。
- 不用腿部或机身撑地作为最终成功判据。

## 任务入口

计划新增：

```bash
just smoke-recovery-stand
just train-recovery-stand
```

底层任务名：

```bash
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Recovery-Stand-GRU --env.scene.num-envs 1 --gpu-ids None
SE3_LOGGER=tensorboard WANDB_MODE=disabled uv run --env-file .env se3-train SE3-WheelLegged-Recovery-Stand-GRU --env.scene.num-envs 4096
```

训练环境不应复用旧 `Recovery-GRU` 的速度课程语义。可以复用已有 reset、reward helper 和 GRU PPO 配置，但最终 contract 以本文为准。
正式长训不依赖 W&B，默认用 TensorBoard/本地日志保证 checkpoint 保存不受网络影响。

## 观测与动作

动作仍为 6 维：

```text
[LF, LB, RF, RB, l_wheel, r_wheel]
```

腿部动作是位置目标，轮子动作是速度目标。actor 观测保持现有共享布局：

```text
base_ang_vel        3
projected_gravity   3
commands            5   [vx, yaw_rate, pitch, roll, base_height]
leg_joint_pos       4
leg_joint_vel       4
wheel_pos           2
wheel_vel           2
last_actions        6
jump_commands       3   本任务中固定为 0
```

critic 可继续使用 `base_lin_vel`、轮子接触力和 `base_height` 特权观测。

## Command Contract

本任务中 command sampler 固定输出：

```text
vx = 0.0
yaw_rate = 0.0
pitch = 0.0
roll = 0.0
height = default_base_height
jump_flag = 0.0
jump_target_height = 0.0
jump_phase = 0.0
```

`pitch=0` 和 `roll=0` 是目标机身姿态指令，不是 reset 姿态限制。reset 姿态仍然全随机。

## Reset Contract

第一版不使用课程，从第 0 iter 起直接全随机：

| 项 | 范围 | 说明 |
|---|---:|---|
| 倾角 | `0°~180°` | 全姿态均匀随机 |
| 倒向轴 | `-180°~180°` | 水平轴向全随机 |
| yaw | `-180°~180°` | 航向全随机 |
| base 高度 | 继续使用安全高度/clearance 采样 | 需要避免严重穿地 |
| base 线速度 | 约 `[-0.15, 0.15] m/s` | 沿用现有 recovery 难度 |
| base 角速度 | 约 `[-0.8, 0.8] rad/s` | 沿用现有 recovery 难度 |
| 髋关节偏移 | `[-0.50, 0.55] rad` | 从第 0 iter 使用最终范围 |
| 膝关节偏移 | `[-0.45, 0.65] rad` | 从第 0 iter 使用最终范围 |
| 关节速度 | `[-0.8, 0.8] rad/s` | 从第 0 iter 使用最终范围 |

如果 reset 后已经接近直立，也不能在第 0 步立即成功。成功终止必须满足最小执行时间和连续稳定窗口。

## 成功终止

`recovery_success` 是训练目标和未来交接条件的唯一成功定义。第一版阈值：

| 条件 | 第一版阈值 |
|---|---:|
| 倾角 | `< 15°` |
| base 高度误差 | `< 0.05m`，后续可收紧到 `< 0.03m` |
| base 角速度范数 | `< 0.5 rad/s` |
| base 线速度范数 | `< 0.2 m/s` |
| 双轮接触 | 左右轮接触力范数都 `> 1N`，仅作为接触存在检测 |
| 非轮接触 | 腿部、机身等非轮接触力都 `<= 1N` |
| 腿部几何对齐 | 左右轮横向距离在 `[0.40, 0.46]m`，前后错位 `<= 0.03m` |
| 连续满足时间 | `0.5s` |
| 最小 episode 时间 | reset 后至少经过 `0.5s` 才允许成功 |

第一版不加入 `total_wheel_force > k * mg` 的重量支撑比例。`1N` 不是支撑重量阈值，只是接触检测阈值；真正站稳由高度、姿态、速度和非轮接触共同约束。

## Termination Contract

保留：

- `time_out`
- `recovery_success`

移除或禁用：

- `bad_orientation`
- `leg_contact`
- `recovery_stagnation`
- 任何把倒地姿态或翻身接触提前杀掉的语义终止

如果框架需要保留非有限数、仿真崩溃等基础保护，它们只能作为基础设施保护，不能进入任务成功/失败统计。

## Reward Contract

第一版奖励目标：让策略从任意姿态探索到可交接站立态，而不是学会平地运动。

必须移除：

- `tracking_lin_vel`
- `tracking_ang_vel`
- 任何要求倒地恢复期间跟踪速度或 yaw rate 的奖励

保留或新增：

| 奖励 | 作用 |
|---|---|
| `upward` | 全姿态扶正主梯度 |
| `upward_progress` | 奖励姿态向直立方向改善 |
| `recovery_inverted_low_height` | 只在 `140°~170°` 倒置区平滑打开，惩罚低 base 高度，帮助策略脱离 `upward` 在完全倒置附近的低梯度区 |
| `recovery_height` | 只在 `60°~15°` 近直立区平滑打开，跟踪固定站立高度，避免倒地阶段用高度项套利 |
| `recovery_wheel_contact` | 接近直立后鼓励轮子重新成为接地点 |
| `recovery_nonwheel_clearance` | 接近直立后鼓励机身、腿部等非轮部件离地，避免靠身体/腿撑住的低趴解 |
| `recovery_stillness` | 接近直立后鼓励机身线速度和轮速降到可交接范围，避免翻正后继续滚走 |
| `recovery_leg_alignment` | 从恢复中段开始惩罚左右轮前后错位和异常轮距，避免两条腿前后劈叉 |
| `recovery_success_bonus` | 成功窗口完成后一次性给 `10.0` |
| `action_rate` | 轻量动作变化正则 |
| `leg_torques` | 轻量力矩正则 |
| `leg_power` | 轻量功率正则 |

`recovery_leg_alignment` 不能等到完全接近直立才启动。roll90 早期回放已观察到策略在约 `80°` 倾角附近通过左右腿前后错位把机身撑起，因此该惩罚从 `135°` 开始逐步打开，防止把前后劈叉学成中间支撑路径。

高度相关奖励拆成两个互补区间。倒置辅助项使用
`smoothstep((tilt_deg - 140) / 30) * clamp((0.24 - base_height) / 0.05, 0, 3)^2`，
只负责在接近完全倒置时惩罚低 base 高度；近直立高度跟踪使用
`smoothstep((60 - tilt_deg) / 45) * exp(-(base_height - target_height)^2 / 0.01)`，
只在姿态已经接近正确后接管固定高度。

过程语义：

- 恢复过程中允许腿、机身或连杆触地。
- 非轮接触不应作为强失败惩罚。
- 接近成功窗口时，非轮接触会阻止 success 计时。
- timeout 不给大负奖励，避免从零训练早期学成少动少错。

## 诊断指标

必须按初始倾角分桶统计，桶定义：

```text
0-30°
30-60°
60-90°
90-135°
135-180°
```

每个桶至少记录：

| 指标 | 用途 |
|---|---|
| `RecoveryStand/success_rate_by_tilt_bin/*` | 每个难度区间是否真正学会 |
| `RecoveryStand/time_to_success_by_tilt_bin/*` | 恢复耗时趋势 |
| `RecoveryStand/timeout_rate_by_tilt_bin/*` | 哪些姿态还失败 |
| `RecoveryStand/final_tilt_deg_by_tilt_bin/*` | timeout 前是否至少在扶正 |
| `RecoveryStand/final_height_error_by_tilt_bin/*` | 是否卡在高度条件 |
| `RecoveryStand/final_nonwheel_contact_rate_by_tilt_bin/*` | 是否靠腿/身体撑住 |
| `RecoveryStand/dual_wheel_contact_rate_by_tilt_bin/*` | 是否能回到双轮接触 |
| `RecoveryStand/wheel_fore_aft_offset_m` | 是否出现左右轮前后错位/腿部劈叉 |
| `RecoveryStand/success_condition/wheel_alignment` | 可交接窗口是否满足腿部几何对齐 |
| `Curriculum/push_disturbance/push_vel_max` | 当前 push disturbance 课程强度 |
| `PushDisturbance/active_velocity_max` | 最近一次实际推扰动的最大速度幅值 |

验收时优先看 `90-135°` 和 `135-180°`，总成功率只能作为辅助指标。

## 验收计划

### Smoke

实现后必须先跑：

```bash
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Recovery-Stand-GRU --env.scene.num-envs 1 --gpu-ids None
```

Smoke 只证明环境不崩，不能证明策略有效。

### 前 5 iter

检查：

- command 是否固定为 `0,0,0,0,default_base_height`。
- reset 是否从第 0 iter 就覆盖 `0°~180°`。
- 关节随机是否已经使用最大范围。
- `tracking_lin_vel` / `tracking_ang_vel` 是否没有出现在 reward 表。
- success 条件各子项日志是否存在且没有 NaN。
- `RecoveryStand/success_condition/wheel_alignment` 是否存在且没有 NaN。
- `Curriculum/push_disturbance/progress` 从 iter 语义推进，初期 `push_vel_max=0`。
- 没有 `bad_orientation` 或 `leg_contact` 提前终止。

### 早期训练

看方向，不要求马上成功：

- `upward` 和 `upward_progress` 有非零信号。
- 高倾角桶 final tilt 有下降趋势。
- timeout 前非轮接触不导致硬终止。
- 动作、力矩、功率没有持续打满。

### 中期训练

要求：

- `60-90°` 桶开始出现稳定成功。
- `90-135°` 桶 success rate 上升。
- `135-180°` 桶 final tilt 下降，即使成功率还低。
- `time_to_success` 不作为 reward，但应随训练自然下降。

### 最终训练

最终阈值先不写死为合并门槛；第一轮实验目标是验证纯自起分离训练是否可行。建议观察：

- `0-60°` 桶成功率接近饱和。
- `60-135°` 桶稳定上升。
- `135-180°` 桶至少能显著减少 final tilt 和 timeout 姿态错误。
- 成功窗口内非轮接触率接近 0。

## 实现步骤

1. 新增 `se3_recovery_stand_env_cfg`。
2. 新增 `se3_recovery_stand_gru_ppo_runner_cfg`，从零训练。
3. 注册 `SE3-WheelLegged-Recovery-Stand-GRU`。
4. 固定 command sampler 或在新 cfg 中固定 command 范围。
5. 配置 reset 为全姿态、全关节随机，无课程。
6. 移除速度跟踪奖励，只保留纯自起奖励。
7. 新增或扩展 success termination，支持双轮接触、非轮接触消失、线速度阈值、最小 episode 时间和连续窗口。
8. 新增分桶诊断日志。
9. 新增 `just smoke-recovery-stand` 和 `just train-recovery-stand`。
10. 跑 smoke，确认环境和日志不崩。

## 风险

| 风险 | 现象 | 处理 |
|---|---|---|
| 从零训练 + 全随机太硬 | 所有桶长期无成功 | 先看 final tilt 是否下降；若完全无梯度，再调 reward，不先改回速度任务 |
| 成功条件过严 | 已站起但 success rate 为 0 | 看子项日志定位是高度、双轮、非轮接触还是速度卡住 |
| 翻正后继续滚走 | roll90 回放能站起，但零速度命令下 base / wheel 速度仍高，`lin_vel` 子条件接近 0 | 加入只在接近直立后生效的 `recovery_stillness` 密集奖励，让低机身线速度和低轮速有可学习梯度 |
| 非轮接触卡成功 | 策略靠腿或机身撑住，不交接 | 保持“过程允许、成功窗口禁止”的语义；若策略进入近直立低趴静止解，加入近直立后生效的 `recovery_nonwheel_clearance` 正向密集奖励 |
| 两腿前后劈叉 | roll90 回放能站稳，但左右轮在机身 x 方向明显错开 | 加入近直立后生效的 `recovery_leg_alignment` 惩罚，并把 wheel alignment 纳入 success 和 sim2sim 验收 |
| 轮子接触力噪声 | 成功窗口抖动 | 先看左右轮 contact force 分布，再调接触阈值或连续窗口 |
| 动作过猛 | sim2sim 不稳、力矩打满 | 成功率起来后逐步加动作/力矩正则，不在早期压探索 |

## 与旧文档关系

`docs/plan/gru_recovery_training.md` 记录的是“单 recovery GRU 同时学习 locomotion 和 self-righting”的旧方案。本文是新的分离训练方案；实现 `SE3-WheelLegged-Recovery-Stand-GRU` 时，以本文为准。
