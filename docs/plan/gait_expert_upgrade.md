# Gait 专家升级路线

## 目标

当前 GAIT 专家要补齐四类能力：

1. 速度指令跟踪：稳定跟随前向速度和 yaw 指令，不靠 reward 总分猜测。
2. 生存能力：受到轻中等扰动后不轻易翻倒，接近失稳时能把姿态拉回来。
3. 复杂地形适应：低矮台阶、随机盒、离散障碍和粗糙地形上保持可控步态。
4. 倒地恢复：倒地后能在不硬砸机身和腿杆的前提下恢复到可运动状态。

## 已收集资料

本轮新增下载到 `/Users/flamingo/Projects/robomaster/thirdparty-rl-repos/papers`：

- `rma_rapid_motor_adaptation_legged_robots_2107.04034.pdf`
- `learning_to_walk_in_minutes_massively_parallel_deep_rl_2109.11978.pdf`
- `robust_perceptive_locomotion_quadrupedal_robots_wild_2201.08117.pdf`
- `legged_locomotion_challenging_terrains_egocentric_vision_2211.07638.pdf`
- `robot_parkour_learning_2309.05665.pdf`
- `extreme_parkour_legged_robots_2309.14341.pdf`
- `safe_rl_legged_locomotion_recovery_policy_2203.02638.pdf`
- `robust_recovery_controller_quadruped_deep_rl_1901.07517.pdf`
- `learning_to_recover_wheel_leg_coordination_fallen_robots_2506.05516.pdf`

已有资料里最相关的是：

- `gait_conditioned_rl_multi_phase_curriculum_2505.20619.pdf`：gait ID、reward routing、多阶段 curriculum。
- `phase_guided_controller_free_gait_transition_2201.00206.pdf`：相位引导和 controller-free transition。
- `unified_walking_running_recovery_state_dependent_amp_2605.18611.pdf`：按状态切换 locomotion / recovery 奖励。
- `seamless_multi_skill_learning_frontiers_2025.pdf`：多技能统一策略的 reward gating 和技能路由。
- `omnixtreme_breaking_generality_barrier_high_dynamic_humanoid_control_2602.23843.pdf`：specialist-to-unified、强随机化和后训练。

## 代码侧现状

关键入口：

- Stage1 平地基础步态：`SE3-WheelLegged-FlowMatch-Gait-Stage1-GRU`
- Stage2 低矮随机地形和最高 `8cm` 台阶：`SE3-WheelLegged-FlowMatch-Gait-Stage2-GRU`
- Stage3 `8-24cm` 上台阶专项：`SE3-WheelLegged-FlowMatch-Gait-Stage3-GRU`
- Stage env 配置：`src/se3_train/tasks/flow_match/gait_stage*/env_cfg.py`
- 共享实现：`src/se3_train/tasks/flow_match/common/`
- 指令生成：`src/se3_train/mdp/commands.py`
- mode reward routing：`src/se3_train/mdp/task_mode_rewards.py`

已确认问题：

1. `BasicCommandTerm._update_metrics()` 原来为空，训练日志缺少速度误差、yaw 误差和 mode 分桶指标。
2. 旧 GAIT 任务把阶段语义压在 `PreTrain/FineTune` 里，不利于表达平地能力、低矮地形和上台阶专项的真实课程边界。
3. GAIT 地形课程存在，但 actor 没有外部地形观测；这适合训练盲走鲁棒性，不适合高台阶/空隙这种需要提前选落点的能力。
4. 当前 bad orientation 是 delayed termination，不是恢复任务。它给了短暂纠错窗口，但不会系统性训练倒地恢复。

## 本轮已做的低风险改动

1. 在 `src/se3_train/mdp/commands.py` 增加通用诊断：
   - `Command/diag_vx_error_abs_active`
   - `Command/diag_yaw_error_abs_active`
   - `Command/diag_lateral_vel_abs`
   - `Command/diag_pitch_error_abs_deg`
   - `TaskMode/diag_gait_vx_error_abs_active`
   - `TaskMode/diag_gait_yaw_error_abs`

2. 在旧 GAIT 地形任务里恢复 push event 和逐步增强课程：
   - 0 step：无扰动
   - 6400 step：`x/y = ±0.15 m/s`
   - 12800 step：`x/y = ±0.30 m/s`
   - 19200 step：`x/y = ±0.45 m/s`

这两个改动不改变模型结构，只让训练可观测，并让旧地形阶段真正包含抗推扰动。

3. 将 GAIT 正式重构为三阶段任务：
   - Stage1：从零训练平地 `0-1.05m/s` 基础步态，后期加入 `±0.10 -> ±0.15m/s` push。
   - Stage2：接 Stage1 checkpoint，引入随机地形和最高 `8cm` 低台阶，最终比例为 `flat/random_grid/boxes/stairs_up/stairs_down = 45/25/15/10/5`。
   - Stage3：接 Stage2 checkpoint，主训 `8-24cm` 上台阶，最终比例为 `stairs_up=60%`，并保留 Stage2 中等地形恢复区。

## 已完成实验记录

### E1：有 push 课程短跑

- run：`2026-06-08_23-31-35_e1_speed_24cce8e`
- W&B：`td8gekxp`
- warm-start：`旧 GAIT 基线 checkpoint`
- 最终 checkpoint：`model_499.pt`

关键指标：

| 进度 | `lin_vel_x_max` | `push_vel_max` | `diag_gait_vx_error_abs_active` | `diag_gait_yaw_error_abs` | `diag_gait_lateral_vel_abs` | `bad_orientation` | `leg_contact` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 96/500 | 0.5641 | 0.0000 | 0.1213 | 0.2358 | 0.2624 | 0.1053 | 0.1754 |
| 499/500 | 1.5000 | 0.4500 | 0.2896 | 0.2849 | 0.2638 | 1.1129 | 0.9355 |

中段速度跟踪方向正确，但满速度、满地形和 push 同时打开后，速度误差和终止项一起回升。

### E2-B：关闭 push 课程对照

- run：`2026-06-09_00-40-11_e2_no_push_24cce8e`
- W&B：`8dgypkjw`
- warm-start：`旧 GAIT 基线 checkpoint`
- 最终 checkpoint：`model_499.pt`

关键指标：

| 进度 | `lin_vel_x_max` | `push_vel_max` | `diag_gait_vx_error_abs_active` | `diag_gait_yaw_error_abs` | `diag_gait_lateral_vel_abs` | `bad_orientation` | `leg_contact` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 116/500 | 0.6559 | 0.0000 | 0.1672 | 0.2697 | 0.2672 | 0.9375 | 0.2969 |
| 499/500 | 1.5000 | 0.0000 | 0.4814 | 0.2913 | 0.2690 | 1.6667 | 0.7460 |

关闭 push 后末段仍明显恶化，而且速度误差比 E1 更大。初步判断：当前主要瓶颈不是 push 本身，而是速度课程和地形课程耦合过猛；push 课程会增加压力，但不是唯一根因。

### E3：地形分桶短跑

- run：`2026-06-09_11-18-40_e3_terrain_29241b7`
- W&B：`5eghdi74`
- warm-start：`旧 GAIT 基线 checkpoint`
- 设置：关闭 push 课程，保留原速度和地形课程
- 最终 checkpoint：`model_499.pt`

关键指标：

| 进度 | `lin_vel_x_max` | `non_flat_ratio` | `push_vel_max` | `diag_gait_vx_error_abs_active` | `bad_orientation` | `leg_contact` |
|---|---:|---:|---:|---:|---:|---:|
| 122/500 | 0.6559 | 约 0.35 | 0.0000 | 约 0.18 | 约 0.94 | 约 0.30 |
| 295/500 | 1.4794 | 0.6883 | 0.0000 | 0.3814 | 0.9259 | 0.2778 |
| 499/500 | 1.5000 | 0.6611 | 0.0000 | 0.3763 | 0.6429 | 0.4286 |

末段地形分桶：

| 地形 | 占比 | cmd vx | actual vx | vx error | yaw error | pitch error |
|---|---:|---:|---:|---:|---:|---:|
| flat | 0.3836 | 0.6725 | 0.4600 | 0.2491 | 0.2896 | 3.5857 |
| random_grid | 0.1542 | 0.6910 | 0.3141 | 0.3917 | 0.3879 | 4.8521 |
| random_spread_boxes | 0.2894 | 0.6989 | 0.2291 | 0.4773 | 0.3186 | 2.9269 |
| open_stairs_up | 0.1318 | 0.6963 | 0.2066 | 0.4994 | 0.3504 | 3.7605 |
| open_stairs_down | 0.0410 | 0.6227 | 0.2483 | 0.3979 | 0.3357 | 3.9388 |

结论：flat 明显好于 boxes/stairs，random_grid 居中。push 已关闭仍然出现同类恶化，所以当前最强证据指向速度课程和地形课程同步推满导致的分布过载。复杂障碍不是单纯奖励权重问题；actor 当前没有 height scan，只能盲走，真正的主动跨越后续需要补地形观测或 teacher-student 蒸馏。

## E3 后课程修正

基于 E1/E2/E3，把 GAIT fine-tune 课程从“速度、地形、push 同步推满”拆成三段：

- 速度课程：`0 -> 32000 step` 从 `0.12 m/s` 拉到 `1.5 m/s`。
- 地形课程：`16000 -> 64000 step` 才从轻地形逐步拉到目标障碍占比和 level 3。
- push 课程：延后到 `32000 / 64000 / 96000 step`，对应 `0.15 / 0.30 / 0.45 m/s`。

这样 500 iter 短跑主要验证速度链路和中低地形压力，完整 2000 iter 才逐步暴露高障碍和外力扰动。下一轮短跑要观察：同样 500 iter 下 `diag_gait_vx_error_abs_active` 是否低于 E3 的 `0.3763`，flat 和 boxes/stairs 的误差差距是否收窄，`bad_orientation + leg_contact` 是否不再在课程末段一起抬升。

### E4：课程解耦短跑

- run：`2026-06-09_11-51-45_e4_decoupled_7dba7bb`
- W&B：`b1ybr5e8`
- warm-start：`旧 GAIT 基线 checkpoint`
- 设置：速度、地形、push 课程解耦；500 iter 内 push 仍为 0
- 最终 checkpoint：`model_499.pt`

关键指标：

| 进度 | `lin_vel_x_max` | `terrain_progress` | `non_flat_ratio` | `push_vel_max` | `diag_gait_vx_error_abs_active` | `bad_orientation` | `leg_contact` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 166/500 | 0.5769 | 0.0000 | 0.1517 | 0.0000 | 0.1167 | 约 0.20 | 0.0000 |
| 297/500 | 0.9385 | 0.0621 | 0.1378 | 0.0000 | 0.1761 | 0.6327 | 0.3673 |
| 499/500 | 1.4959 | 0.3314 | 0.3003 | 0.0000 | 0.2501 | 0.8727 | 0.4182 |

末段地形分桶：

| 地形 | 占比 | cmd vx | actual vx | vx error | yaw error | pitch error |
|---|---:|---:|---:|---:|---:|---:|
| flat | 0.8259 | 0.6806 | 0.4790 | 0.2311 | 0.2931 | 3.0999 |
| random_grid | 0.0915 | 0.6498 | 0.4082 | 0.2662 | 0.3234 | 3.6816 |
| random_spread_boxes | 0.0431 | 0.7573 | 0.3396 | 0.4274 | 0.3401 | 4.2523 |
| open_stairs_up | 0.0378 | 0.7405 | 0.3601 | 0.3901 | 0.3184 | 4.6607 |
| open_stairs_down | 0.0018 | 0.8648 | -0.0703 | 0.9351 | 0.3018 | 13.9741 |

结论：解耦课程有效降低速度误差，E4 末段 `0.2501` 明显低于 E3 的 `0.3763`。但 `bad_orientation` 和 `leg_contact` 仍在高速段抬升，说明剩余瓶颈已经从“速度/地形同步过载”转到“高速 gait 稳定性 + 障碍分桶鲁棒性”。下一步先跑 flat-only 高速对照：如果 flat-only 仍高 bad_orientation/leg_contact，就优先调高速稳定奖励和课程；如果 flat-only 明显稳定，则优先继续放慢/分层地形并补 height scan。

### E5：flat-only 高速隔离短跑

- run：`2026-06-09_12-24-09_e5_flatonly_ef53306`
- W&B：`4voi39ca`
- warm-start：`旧 GAIT 基线 checkpoint`
- 设置：同 E4，但地形课程强制 `flat=1.0`、`max_level=0`；500 iter 内 push 为 0
- 最终 checkpoint：`model_499.pt`

关键指标：

| 进度 | `lin_vel_x_max` | flat | non-flat | `diag_gait_vx_error_abs_active` | `bad_orientation` | `leg_contact` | `touchdown_support_error_m` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 146/500 | 0.5243 | 1.0000 | 0.0000 | 0.0835 | 0.0000 | 0.0465 | 约 0.32 |
| 276/500 | 0.8831 | 1.0000 | 0.0000 | 0.1260 | 0.1333 | 0.3333 | 约 0.35 |
| 397/500 | 1.2142 | 1.0000 | 0.0000 | 0.1752 | 约 0.38 | 最高 0.50 | 约 0.36 |
| 499/500 | 1.4987 | 1.0000 | 0.0000 | 0.1949 | 0.4545 | 0.4091 | 0.3716 |

末段平地分桶：

| 地形 | 占比 | cmd vx | actual vx | vx error | yaw error | pitch error | roll error |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat | 1.0000 | 0.6791 | 0.5152 | 0.1949 | 0.2722 | 2.8945 | 3.3885 |

结论：E5 证明 flat-only 比 E4 跟速更好，但高速段仍明显翻倒。`bad_orientation` 在 1.2 m/s 后升到 0.3-0.45，`leg_contact` 维持在 0.37-0.50，且 `touchdown_support_error_m` 长期约 0.36-0.37 m。剩余瓶颈不是复杂地形本身，而是高速 GAIT 的落脚支撑几何、姿态角速度和速度课程尾段稳定性。下一步 E6 先做 flat-only 高速稳定补丁，目标是在不牺牲速度误差的前提下，把末段 `bad_orientation` 和 `leg_contact` 明显压低。

## E6 高速稳定补丁

基于 E5，不先改观测结构和地形生成器，只在 GAIT fine-tune 里补高速稳定梯度：

- 腿杆触地从即时终止改为 8 step delayed termination，让轻微擦碰先进入惩罚/恢复窗口。
- `bad_orientation` 宽限从 8 step 提到 16 step，给策略短时回正机会。
- 姿态项加严：`tracking_orientation_l2=-24`、`bad_tilt=-14`，soft/hard limit 调到 `7°/22°`。
- 横滚/俯仰角速度惩罚加到 `ang_vel_xy=-0.8`。
- touchdown 支撑对齐权重加到 `-8`，`max_penalty=8`，直接压 E5 中长期偏高的 `touchdown_support_error_m`。

E6 仍先跑 flat-only 500 iter 对照。判断标准：在 `lin_vel_x_max≈1.5` 的末段，`diag_gait_vx_error_abs_active` 不明显高于 E5 的 `0.1949`，同时 `bad_orientation` 和 `leg_contact` 明显低于 E5 的 `0.4545 / 0.4091`。

### E6：高速稳定补丁短跑

- run：`2026-06-09_12-54-44_e6_stability_dfaf80f`
- W&B：`aa8sadvy`
- warm-start：`旧 GAIT 基线 checkpoint`
- 设置：flat-only；保留 500 iter 速度压力测试；增加 delayed leg contact、姿态/角速度/落脚支撑强约束
- 最终 checkpoint：`model_499.pt`

关键指标：

| 进度 | `lin_vel_x_max` | flat | `diag_gait_vx_error_abs_active` | `bad_orientation` | `leg_contact` | `actual_vx` | `touchdown_support_error_m` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 149/500 | 0.5326 | 1.0000 | 0.1200 | 0.0000 | 0.0000 | 0.1997 | 0.3197 |
| 292/500 | 0.9272 | 1.0000 | 0.1942 | 0.3793 | 0.0000 | 0.3043 | 0.3373 |
| 408/500 | 1.2475 | 1.0000 | 0.3006 | 0.2353 | 0.0000 | 0.3441 | 0.3462 |
| 499/500 | 1.4988 | 1.0000 | 0.3325 | 0.2826 | 0.0000 | 0.3800 | 0.3562 |

结论：E6 把 `leg_contact` 终止压到 0，说明 8 step delayed leg contact 对“轻微擦碰后继续恢复”有效；但速度跟踪明显变差，末段 `vx_error` 从 E5 的 `0.1949` 恶化到 `0.3325`，实际速度从 `0.5152` 降到 `0.3800`。全程加强姿态、角速度和 touchdown 支撑惩罚会让策略选择慢走保守解，不适合作为最终方案。E7 应保留 delayed leg contact 和更长 bad orientation 宽限，撤回过强的全局稳定惩罚，只做轻量姿态/角速度约束。

## E7 平衡补丁

基于 E6，下一轮改为：

- 保留腿杆触地 8 step delayed termination。
- 保留 `bad_orientation.max_steps=16`，继续给短时回正机会。
- 姿态项从 E6 的强约束回退到轻量约束：`tracking_orientation_l2=-20`、`bad_tilt=-11`，soft/hard limit 回到 `8°/24°`。
- 横滚/俯仰角速度惩罚从 `-0.8` 降到 `-0.5`。
- touchdown 支撑对齐回退到 E5 权重 `-4`、`max_penalty=4`，避免落脚几何项压制速度学习。

E7 通过标准：末段 `leg_contact` 终止继续显著低于 E5，`bad_orientation` 不高于 E5，同时 `diag_gait_vx_error_abs_active` 回到 E5 附近。

### E7：平衡补丁短跑

- run：`2026-06-09_13-21-24_e7_balance_a425f40`
- W&B：`t7gr77e4`
- warm-start：`旧 GAIT 基线 checkpoint`
- 设置：flat-only；保留 delayed leg contact 和更长 bad orientation 宽限；撤回 E6 过强的全局稳定惩罚
- 最终 checkpoint：`model_499.pt`

关键指标：

| 进度 | `lin_vel_x_max` | flat | `diag_gait_vx_error_abs_active` | `bad_orientation` | `leg_contact` | `actual_vx` | `touchdown_support_error_m` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 491/500 | 1.4766 | 1.0000 | 0.2339 | 0.4043 | 0.0000 | 0.4777 | 约 0.36 |
| 492/500 | 约 1.48 | 1.0000 | 0.2254 | 0.5909 | 0.0227 | 0.4897 | 约 0.36 |
| 493/500 | 约 1.48 | 1.0000 | 0.2270 | 0.2727 | 0.0000 | 0.5004 | 约 0.36 |
| 498/500 | 1.4959 | 1.0000 | 0.2517 | 0.4222 | 0.0000 | 0.4838 | 约 0.36 |
| 499/500 | 约 1.50 | 1.0000 | 0.2438 | 未记录到终止行 | 诊断比例 0.0022 | 0.4932 | 约 0.36 |

结论：E7 保住了 E6 最有效的 delayed leg contact，末段腿杆触地终止基本被压住；速度跟踪也从 E6 的 `0.3325` 恢复到 `0.2438` 附近。但它仍没有回到 E5 的 `0.1949`，`bad_orientation` 也仍在高速段反复出现。当前最稳的判断是：腿杆轻微触地恢复窗口应该保留；全局姿态/角速度/落脚支撑惩罚会牺牲速度；下一步要做速度条件化或事件条件化稳定约束，而不是继续无差别加大稳定惩罚。

## E8 风险条件化稳定补丁

基于 E7 和“平时下限不能太低”的约束，E8 不降低基础纪律，而是新增一条 GAIT 专用风险加权稳定项：

- `tracking_orientation_l2=-20`、`bad_tilt=-11`、`ang_vel_xy=-0.5` 等 E7 基础项保持不变，继续约束平时动作质量。
- 新增 `gait_risk_conditioned_stability`，所有 GAIT 样本都有 `base_scale=0.35` 的基础稳定压力。
- 当前向速度指令超过 `0.75 m/s` 后逐步增加速度风险权重，`1.35 m/s` 达到满额。
- 当机身倾角超过 `8°` 后逐步增加倾角风险权重，`20°` 达到满额。
- 新增日志：`diag_gait_risk_stability_scale`、`diag_gait_risk_speed`、`diag_gait_risk_tilt`、`diag_gait_tilt_deg`、`diag_gait_ang_vel_xy`，用于确认该项是在高速/大倾角时触发，而不是全程压制速度。

E8 判断标准：末段 `diag_gait_vx_error_abs_active` 不明显差于 E7 的 `0.2438`，同时 `bad_orientation` 低于 E7 末段的约 `0.40-0.42`；`leg_contact` 继续接近 0。如果速度明显掉到 E6 水平，说明风险加权仍太重；如果 `bad_orientation` 不降，说明仅靠姿态/角速度风险项不够，需要转向 gait mechanics 或恢复式奖励。

### E8：风险条件化稳定短跑（中止）

- run：`2026-06-09_19-37-01_e8_risk_f37adf7`
- W&B：`hlzuaeeu`
- warm-start：`旧 GAIT 基线 checkpoint`
- 设置：flat-only；保留 E7 基础项；新增 `gait_risk_conditioned_stability`
- 中止位置：约 `286/500`

关键中途指标：

| 进度 | `lin_vel_x_max` | flat | `diag_gait_vx_error_abs_active` | `bad_orientation` | `leg_contact` | `risk_scale` | `risk_speed` | `risk_tilt` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 146/500 | 0.5243 | 1.0000 | 0.0962 | 0.2045 | 0.0000 | 0.3558 | 0.0000 | 0.0089 |
| 268/500 | 0.8611 | 1.0000 | 0.1798 | 0.7000 | 0.0000 | 0.3693 | 0.0115 | 0.0235 |
| 286/500 | 0.9107 | 1.0000 | 0.2018 | 0.8421 | 0.0000 | 0.3719 | 0.0212 | 0.0223 |

结论：E8 没有跑完。`leg_contact` 继续被 delayed termination 和接触惩罚压住，但中速段 `bad_orientation` 提前恶化到 `0.70-0.84`，速度误差也比 E5 同阶段差。最关键证据是 `risk_speed` 和 `risk_tilt` 都很低，说明该项在中速失稳前几乎没有加码，主要变成了轻量事后补救。E9 需要把速度风险触发前移到中速段，同时降低总权重和角速度权重，避免回到 E6 的慢走保守解。

## E9 中速前置稳定补丁

E9 不再新增函数，只调整 E8 风险项参数，保持归因清楚：

- `weight` 从 `-4.0` 降到 `-3.0`，避免压制速度。
- `base_scale` 从 `0.35` 降到 `0.25`，平时仍有下限，但不全程重压。
- `speed_start` 从 `0.75` 前移到 `0.35`，`speed_full` 从 `1.35` 前移到 `0.90`，让中速段就开始主动管姿态。
- `speed_scale` 从 `0.35` 提到 `0.80`，把主要加码来源从倾角事后触发改为速度前馈触发。
- `tilt_start_deg` 从 `8°` 放到 `10°`，`tilt_scale` 从 `0.65` 降到 `0.35`，避免策略在正常摆动时被过早罚。
- `ang_vel_weight` 从 `0.18` 降到 `0.10`，减少对摆腿动作的压制。

E9 判断标准：在 `lin_vel_x_max≈0.9` 附近，`bad_orientation` 要明显低于 E8 的 `0.8421`，且 `diag_gait_vx_error_abs_active` 不高于约 `0.20`；末段再看是否比 E7 的 `bad_orientation≈0.40-0.42` 更低，同时速度误差不退回 E6 的 `0.3325`。

### E9：中速前置稳定短跑

- run：`2026-06-09_20-00-29_e9_midrisk_ef2993d`
- W&B：`lsfm29im`
- warm-start：`旧 GAIT 基线 checkpoint`
- 设置：flat-only；保留 delayed leg contact；风险稳定项从中速开始加码
- 最终 checkpoint：`model_499.pt`

关键指标：

| 进度 | `lin_vel_x_max` | `diag_gait_vx_error_abs_active` | `bad_orientation` | `leg_contact` | `actual_vx` | `risk_scale` | `risk_speed` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 145/500 | 0.5215 | 0.0961 | 0.0541 | 0.0000 | 未记录 | 0.2913 | 0.0510 |
| 301/500 | 0.9520 | 0.1796 | 0.2558 | 0.0000 | 未记录 | 0.5507 | 0.3710 |
| 426/500 | 1.2971 | 0.2491 | 0.2000 | 0.0000 | 未记录 | 0.6911 | 未记录 |
| 498/500 | 1.4960 | 0.2454 | 0.1951 | 0.0000 | 0.4493 | 未记录 | 未记录 |
| 499/500 | 约 1.50 | 0.2462 | 未记录到终止行 | 诊断比例 0.0016 | 0.4529 | 0.7123 | 0.5727 |

结论：E9 证明中速前置风险门控有效，E8 在 `lin_vel_x_max≈0.91` 时 `bad_orientation=0.8421`，E9 在 `lin_vel_x_max≈0.95` 时降到 `0.2558`。末段腿杆触地继续接近 0，说明 delayed leg contact 和接触惩罚应该保留。代价是速度跟踪没有回到 E5 flat-only 的 `0.1949`，末段 `actual_vx≈0.45` 也低于 E7 的约 `0.49`。所以 E9 是当前最好的稳定/速度折中，但速度梯度还需要在安全姿态下补回来。

## E10 安全跟速补丁

基于 E9 和“平时下限不能放太低”，E10 不撤掉风险稳定项，而是新增 GAIT 安全窗口速度跟踪奖励：

- 姿态倾角低于 `7°`、横滚/俯仰角速度低于 `0.8 rad/s` 时，额外给完整跟速奖励。
- 倾角到 `18°` 或角速度到 `2.0 rad/s` 时，额外跟速奖励平滑降到 0。
- E9 的 `gait_risk_conditioned_stability` 保持不变，继续提供平时下限和中速风险前馈。
- 新增项权重先设 `0.8`，只做轻量速度补偿，避免回到 E6 那种强约束慢走解。

E10 判断标准：末段 `diag_gait_vx_error_abs_active` 要低于 E9 的 `0.2462`，`actual_vx` 要高于 E9 的 `0.4529`；同时 `bad_orientation` 不应回到 E7 的 `0.40-0.42` 长期水平，`leg_contact` 诊断比例继续接近 0。前 5 iter 要确认奖励表包含 `gait_safe_tracking_lin_vel`，且 `TaskMode/diag_gait_safe_tracking_gate` 不是长期为 0。

### E10：安全跟速短跑

- run：`2026-06-09_21-03-27_e10_safetrack_61a7a58`
- W&B：`2a56z9ou`
- warm-start：`旧 GAIT 基线 checkpoint`
- 设置：flat-only；保留 E9 风险稳定项；新增 `gait_safe_tracking_lin_vel=0.8`
- 最终 checkpoint：`model_499.pt`

关键指标：

| 进度 | `lin_vel_x_max` | `diag_gait_vx_error_abs_active` | `bad_orientation` | `leg_contact` | `actual_vx` | `safe_gate` | `risk_scale` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 133/500 | 0.4884 | 0.0872 | 0.0233 | 0.0000 | 0.2271 | 0.7695 | 0.2787 |
| 259/500 | 0.8362 | 0.1471 | 0.1333 | 0.0000 | 0.3329 | 0.7613 | 0.4675 |
| 385/500 | 1.1812 | 0.2275 | 0.2143 | 0.0000 | 0.4137 | 0.7410 | 0.6544 |
| 497/500 | 1.4930 | 0.2369 | 0.3846 | 0.0000 | 0.4772 | 0.7223 | 0.7269 |
| 498/500 | 1.4958 | 0.2420 | 0.3778 | 0.0000 | 0.4774 | 0.7223 | 0.7280 |
| 499/500 | 1.4987 | 0.2357 | 0.4878 | 0.0000 | 0.4805 | 0.7236 | 0.7266 |

结论：E10 证实“安全时额外催跟速”有效，末段 `actual_vx` 从 E9 的 `0.4529` 提到 `0.4805`，`vx_error` 从 `0.2462` 降到 `0.2357`。但高速末段 `bad_orientation` 最高到 `0.4878`，比 E9 尾段样本更差，也接近 E7 的不稳定水平。腿杆触地终止仍为 0，说明问题不是腿杆蹭地，而是高速姿态恢复裕度不足。下一轮 E11 不应继续加速度奖励；更合理的是保留 E10 跟速项，把高速段风险稳定稍微加严，或让 `safe_tracking` 在高 `risk_speed` 时更快衰减。

## E11 中速精品目标

基于 E5-E10 的证据，纯 GAIT 先不追 `1.5 m/s`。从 E5 开始，高速段主要问题都集中在 `1.2-1.5 m/s`：速度越推越高时，`bad_orientation` 反复抬升；而 `0.8-1.05 m/s` 区间已经能同时保持较低腿杆触地和可接受跟速。因此 E11 把 GAIT fine-tune 的最大目标速度下调为 `1.05 m/s`，目标从“挑战极限速度”改为“中速范围高质量稳定步态”。

代码侧改动：

- `play` 和训练课程的 `lin_vel_x_range` 终点统一改为 `1.05 m/s`。
- 保留 E9 风险稳定项和 E10 安全跟速项。
- 不再额外加速度奖励，避免把 E10 末段姿态风险继续放大。

E11 判断标准：

- 末段 `lin_vel_x_max≈1.05` 时，`diag_gait_vx_error_abs_active` 目标低于 `0.16`，至少不能高于 E10 中速段的 `0.147-0.15` 太多。
- `actual_vx` 应稳定跟随到约 `0.35-0.45 m/s` 的均值区间；更重要的是相对 E9/E10 同速度段不退化。
- `bad_orientation` 目标稳定低于 `0.20`，不能出现 E10 高速尾段 `0.38-0.49` 的反弹。
- `leg_contact` 终止继续为 0，诊断触地比例维持在 `0.001-0.003` 量级以内。

## Post-train：高台阶地形课程

阶段性 flat-only 实验结束后，从 E11 `model_499.pt` 开始做完整 GAIT post-train。目标速度仍保持 `1.05 m/s`，训练重点转为逐步引入地形。

课程规格：

- 速度课程：`0 -> 32000 step` 从 `0.12 m/s` 拉到 `1.05 m/s`。
- 地形课程：`16000 -> 64000 step` 从轻比例地形拉到目标比例，`max_level: 0 -> 3`。
- 地形目标比例：`flat=0.30`、`random_grid=0.27`、`random_spread_boxes=0.19`、`open_stairs_up=0.18`、`open_stairs_down=0.06`。
- 台阶单级高度：`open_stairs_up/down` 从原 `0.02-0.09 m` 提高到 `0.08-0.24 m`。
- push 课程保留默认延后进入：`32000 / 64000 / 96000 step` 对应 `±0.15 / ±0.30 / ±0.45 m/s`。

观察重点：台阶高度提高后，优先看 `open_stairs_up/down` 分桶的 `vx_error_abs_active`、`pitch/roll_error_abs_deg`、`leg_contact_ratio` 和 `bad_orientation`。如果高台阶一开始就压垮策略，优先降低台阶比例或延后 stairs 课程，而不是回退目标速度。

## 本轮追加诊断

为支持 E3，在日志里增加 GAIT fine-tune 地形分桶：

- `TaskMode/diag_gait_terrain_<terrain>_ratio`
- `TaskMode/diag_gait_terrain_<terrain>_cmd_vx_abs`
- `TaskMode/diag_gait_terrain_<terrain>_actual_vx`
- `TaskMode/diag_gait_terrain_<terrain>_vx_error_abs_active`
- `TaskMode/diag_gait_terrain_<terrain>_yaw_error_abs`
- `TaskMode/diag_gait_terrain_<terrain>_lateral_vel_abs`
- `TaskMode/diag_gait_terrain_<terrain>_pitch_error_abs_deg`
- `TaskMode/diag_gait_terrain_<terrain>_roll_error_abs_deg`
- `TaskMode/diag_gait_terrain_<terrain>_leg_contact_ratio`
- `TaskMode/diag_gait_terrain_<terrain>_leg_contact_force_max`

`<terrain>` 取值：`flat`、`random_grid`、`random_spread_boxes`、`open_stairs_up`、`open_stairs_down`。

## 实验顺序

### E0：构造和 smoke

目的：确认新增诊断和 push 课程不会导致环境崩溃。

```bash
uv run ruff format --check src/se3_train
uv run ruff check src/se3_train
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-FlowMatch-Gait-Stage1-GRU --env.scene.num-envs 1 --gpu-ids None
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-FlowMatch-Gait-Stage2-GRU --env.scene.num-envs 1 --gpu-ids None
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-FlowMatch-Gait-Stage3-GRU --env.scene.num-envs 1 --gpu-ids None
```

Stage2/Stage3 训练时不在配置里写死 checkpoint，交接命令显式传：

```bash
uv run --env-file .env se3-train SE3-WheelLegged-FlowMatch-Gait-Stage2-GRU \
  --agent.resume True \
  --agent.load-run <stage1-run> \
  --agent.load-checkpoint <stage1-checkpoint>

uv run --env-file .env se3-train SE3-WheelLegged-FlowMatch-Gait-Stage3-GRU \
  --agent.resume True \
  --agent.load-run <stage2-run> \
  --agent.load-checkpoint <stage2-checkpoint>
```

### E1：速度跟踪短跑

目的：先确认速度链路是否真的改善，不混入大地形结论。

建议跑 300-500 iter，观察：

- `TaskMode/diag_gait_vx_error_abs_active`
- `Command/diag_vx_error_abs_active`
- `TaskMode/diag_gait_yaw_error_abs`
- `Episode_Termination/bad_orientation`
- `Episode_Termination/leg_contact`
- `TaskMode/diag_gait_leg_contact_ratio`

通过标准：前向速度误差随 iter 明显下降，终止率不随速度课程扩大而暴涨。

### E2：扰动鲁棒性短跑

目的：验证恢复 push 课程是否提升抗扰动，而不是只提高平地 reward。

对比两组：

- A：当前补丁，有 push 课程
- B：关闭 push 课程

评估固定命令：

- `vx=0.4, 0.8, 1.2 m/s`
- 每 4-5 秒随机 `x/y` 方向 root velocity push
- 记录 60 秒内 fall count、最大 tilt、恢复到 tilt < 15 deg 的时间

通过标准：A 的 fall count 明显低于 B，同时速度误差没有显著变差。

### E3：地形课程分桶评估

目的：区分“能盲走粗糙地形”和“能主动跨越障碍”。

按地形类型单独统计：

- flat
- random_grid
- random_spread_boxes
- open_stairs_up
- open_stairs_down

如果 flat 能跟速但障碍失败，优先补 terrain 分桶日志和 height scan 观测；如果 flat 也失败，先回 E1 调速度/reward。

### E4：恢复能力独立任务

倒地恢复不要直接混进 GAIT fine-tune。建议新增 `RECOVERY` 或先用单独 task 做恢复专家：

- reset 从 prone / supine / side / 半倒姿态采样
- reward 分阶段：减少倾角、提高 base height、恢复轮接触、降低冲击和腿杆接触
- 参考 `learning_to_recover_wheel_leg_coordination_fallen_robots_2506.05516.pdf` 的 episode-based dynamic reward shaping
- 后续再用 TaskMode 或 state-dependent gate 接入统一策略

通过标准：

- 随机倒地初态 5 秒内恢复成功率
- 机身/腿杆高冲击接触次数
- 平均恢复时间
- 恢复后 5 秒内速度跟踪误差

## 关键判断

1. 速度跟踪是第一优先级。速度误差没有闭环指标时，任何地形/扰动结论都不可靠。
2. 抗扰动要进入训练分布。只靠 delayed termination 给恢复机会，不等于学会 recovery。
3. 复杂障碍能力分两层：低矮粗糙可先盲走；高障碍、台阶边缘和 gap 需要地形观测或 teacher-student/perception distillation。
4. 倒地恢复是独立技能，不建议用 GAIT reward 顺手揉进去。先做 recovery 专家，再决定是否蒸馏到统一 TaskMode。
