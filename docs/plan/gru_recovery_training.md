# GRU 反倒起身训练规范

> 状态：历史方案。当前新的分离训练方向见 `docs/plan/recovery_stand_training.md`，新任务计划为 `SE3-WheelLegged-Recovery-Stand-GRU`，只训练倒地自起到固定站立态；本文保留用于追溯旧的“自起 + locomotion 同策略”方案。

## 目标

第一版按 RobotLab Go2W 的思路做最小闭环：同一个 GRU policy 从随机初始化开始训练，在同一套 locomotion objective 下同时学会站立、低速行走和任意姿态自起。

核心约束：

1. 不从已有 checkpoint 续训，不加载 `assets/base_model`。
2. 不区分 normal reset、fall reset、recovery reset。
3. 不设置 recovery mode，不给 actor 增加 mode 输入。
4. 不按初始姿态、倾角桶、roll 轴或 pitch 轴切换 reward。
5. reset、速度命令和腿关节扰动可以使用全局 iter 课程，但不能按初始姿态类型切换 reward 或命令。

当前任务入口：

```bash
just smoke-recovery
just train-recovery-light
just train-recovery
```

底层任务名：

```bash
uv run --env-file .env se3-train SE3-WheelLegged-Recovery-GRU --env.scene.num-envs 4096
```

## 网络与 PPO

当前恢复任务使用 `se3_recovery_gru_ppo_runner_cfg`：

| 项 | 当前值 | 说明 |
|---|---:|---|
| policy | `RNNModel` / GRU | 自起是强时序、强接触任务，GRU 用于记忆最近姿态和接触变化 |
| GRU hidden | 512 | 默认容量，先不引入 MoE 或双策略切换 |
| GRU layers | 1 | 降低部署和 sim2sim hidden state 维护成本 |
| MLP head | 512, 256, 128 | 统一 locomotion/self-righting policy head |
| `num_steps_per_env` | 64 | 给 BPTT 足够长的恢复窗口 |
| `learning_rate` | `SE3_RECOVERY_LEARNING_RATE` | 从零训练超参 |
| `desired_kl` | 0.008 | 控制 PPO 更新幅度，避免统一任务早期策略剧烈摆动 |
| `max_iterations` | 3000+ | 从零训练默认长度应按收敛情况延长 |
| init | random | policy、value、normalizer、GRU 参数全部随机初始化 |

正式实验必须从 iteration 0 开始。训练中断后的继续跑只允许作为同一 run 的故障恢复，不作为默认训练流程。

## 观测 Contract

当前实现不改变机器人动作语义：动作仍为 6 维 `[lf0, lf1, rf0, rf1, l_wheel, r_wheel]`，腿部是位置目标，轮子是速度目标。

actor 观测沿用平地/跳跃共享形状：

```text
base_ang_vel        3
projected_gravity   3
commands            5   [vx, yaw_rate, pitch, roll, base_height]
leg_joint_pos       4
leg_joint_vel       4
wheel_pos           2
wheel_vel           2
last_actions        6
jump_commands       3   当前任务中通常为 0
```

critic 额外有 `base_lin_vel`、轮子接触力和 `base_height` 特权观测。

约束：

- 不允许在没有重新制定从零训练方案时修改 actor 输入维度。
- command sampler 不因初始姿态特判置零。速度、yaw 和高度命令使用同一套采样规则。
- 如果要降低命令难度，只能使用全局 command curriculum 缩小所有样本的命令范围，不能按初始姿态切换命令。

## Reset 逻辑

训练端使用一套统一 reset sampler。每个 env 每次 reset 都从同一个分布采样，不记录 normal/fall/recovery source。

参考 RobotLab Go2W 的全姿态自起目标，当前实现使用“倾角 + 水平倒向 + yaw”的统一分布，并通过全局 iter 课程逐步放开难度：

```text
root pose:
  x, y: 小范围随机
  z: 低高度到站立高度附近随机
  tilt: 从低倾角课程逐步扩展到 [0, pi]
  tilt axis: 水平倒向全范围随机
  yaw: [-pi, pi]

root velocity:
  lin_vel xyz: 小范围随机
  ang_vel xyz: 小范围随机
```

如果实现上不用 Euler 直接采样，也可以用“倾角 + 水平倒向 + yaw”的等价方式生成姿态，但分布必须覆盖 0-180 deg 倾角，且不能把 roll 和 pitch 当成两类任务。

关节 reset：

- 默认回到 `default_joint_pos`，轮子位置归零。
- 大腿和小腿关节分别加随机偏移，范围随全局 iter 课程扩大。
- 腿部关节速度加小范围随机扰动。
- 第一版不使用离线状态缓存。

### 全局课程

课程只允许依赖全局 PPO iter，不允许依赖本次 reset 是“正常/倒地/恢复”或依赖 roll/pitch 轴向。

- 不设 `fallen_prob`。
- 不设 `recovery_stages`。
- 不按成功率触发切换。
- 不按姿态类型自动回退。

当前课程建议：

| iter | reset 倾角上限 | `vx` 命令范围 | yaw 命令范围 | 关节扰动 |
|---:|---:|---:|---:|---|
| 0 | 约 60 deg | 0 | 0 | 大腿/小腿小范围 |
| 250-300 | 约 90 deg | ±0.3 m/s | ±0.3 rad/s | 小范围 |
| 600-700 | 约 135 deg | ±0.5 m/s | ±0.5 rad/s | 中等范围 |
| 1000-1100 | 180 deg | ±1.0 m/s | ±0.5 rad/s | 中等范围 |
| 1500+ | 180 deg | ±1.5 m/s | ±0.75 rad/s | 大范围 |
| 2200+ | 180 deg | ±1.5 m/s | ±1.0 rad/s | 大范围 |

`init_tilt_bin` 只允许作为日志和验收分桶，不允许进入 reward、termination 或按轴向分支。课程阈值只能看全局 iter。

推荐诊断分桶：

| 分桶 | 倾角 | 用途 |
|---|---:|---|
| `upright_noise` | 0-30 deg | 正常扰动、防摔和轻微扶正 |
| `near_fall` | 30-75 deg | 倾倒边缘，训练主动回正 |
| `hard_tilt` | 75-130 deg | 任意倒向等价处理，训练真正自起 |
| `inverted` | 130-180 deg | 接近倒置和全姿态鲁棒性 |

轴对称约束：

- 不允许按 roll 轴和 pitch 轴分别设计 reward、metric、课程阈值或验收标准。
- 不允许出现 `hard_roll_*`、`hard_pitch_*`、`roll_success`、`pitch_success` 这类轴向专属指标。
- roll/pitch/yaw 只能作为姿态生成和失败复现的坐标表达；训练判断必须汇总到 `init_tilt_bin`。

## 奖励规范

奖励设计参考 RobotLab Go2W：同一张 locomotion reward 表覆盖站立、行走和倒地自起；初始姿态、倾角桶、内部计时变量不能进入 reward 公式。

RobotLab 对照关系：

- Go2W rough 配置把 reset 姿态扩到全 roll/pitch/yaw，并关闭 `illegal_contact` 终止。
- Go2W 使用 `upward` 作为全姿态直立项，权重为正。
- 速度追踪、姿态稳定、接触、默认姿态等项在 reward 函数内部乘 `upright_gate`，倒地时自然降权，站起后自然恢复；高度项保留最小 gate，避免倒置/侧躺时完全没有抬高机身的梯度。
- `is_terminated.weight = 0`，不靠终止惩罚塑造起身动作。

RobotLab 模式的核心是两个连续状态量：

```text
upright_gate = clamp(-projected_gravity_z, 0, 0.7) / 0.7
upward_score = square(1 - projected_gravity_z)
```

本仓库的 `projected_gravity_z` 约定为 -1 完全直立，+1 完全倒置。`upward_score` 是全姿态生效的直立项，最大值仍为 4；当前恢复为平方形式，倒置附近的起身探索由额外机制补足。`upright_gate` 只让速度追踪、接触和默认姿态等“已经接近站稳才有意义”的项平滑恢复，不能作为起身主梯度。高度项不能完全依赖 `upright_gate`，应保留最小 gate 作为倒地阶段的抬高梯度。

统一 reward 表：

| 类别 | 参考 RobotLab 项 | 本任务约束 |
|---|---|---|
| 全局直立项 | `upward` | 全状态生效，不乘 `upright_gate`，不按初始姿态或倾角桶加权 |
| 速度追踪 | `track_lin_vel_xy_exp`、`track_ang_vel_z_exp` | 乘 `upright_gate`；command 不因初始姿态置零，策略通过先站稳再跟踪速度来拿完整奖励 |
| 高度和姿态稳定 | `base_height_l2`、`lin_vel_z_l2`、`ang_vel_xy_l2`、`flat_orientation_l2` | 高度项使用带最小值的连续 gate；姿态/速度稳定项乘 `upright_gate` 或等价连续 gate，倒地时不应压制起身探索 |
| 接触和支撑 | `undesired_contacts`、`contact_forces` | 不作为倒地硬失败；惩罚应随 `upright_gate` 恢复，避免倒地初期强压接触 |
| 关节和动作正则 | `joint_torques_l2`、`joint_acc_l2`、`joint_power`、`action_rate_l2` | 全状态弱约束或连续 gate，权重不能大到压住起身动作 |
| 默认姿态/静止约束 | `stand_still`、`joint_pos_penalty` | 乘 `upright_gate`；倒地时不能把关节压回默认站姿 |
| 终止惩罚 | `is_terminated` | 权重为 0 或接近 0，不能靠终止惩罚教起身 |

### 2026-06-08 实验结论：`orientation_l2` 惩罚有效

在 `SE3-WheelLegged-Recovery-GRU` 的 5k 训练对比中，加入 near-upright 阶段的 `orientation_l2` 姿态惩罚是有效的。对照组为用户桌面 checkpoint `C:/Users/13567/Desktop/model_4999.pt`，该 checkpoint 未使用该 `orientation_l2` 惩罚训练；实验组为 W&B run `acttgnoq` 的 `model_4999.pt`，对应训练包含该姿态 L2 约束。

同一远程 recovery 代码和同一组 sim2sim case 下，带 `orientation_l2` 的 checkpoint 在恢复后的姿态稳定性更好：

| case | 带 `orientation_l2` 的 `acttgnoq/model_4999` | 无 `orientation_l2` 的桌面 checkpoint | 结论 |
|---|---:|---:|---|
| `roll90_h022` | final height 0.218 m, final tilt 1.77 deg | final height 0.225 m, final tilt 1.86 deg | 接近，实验组姿态略好 |
| `pitch180_h022` | final height 0.217 m, final tilt 2.05 deg | final height 0.223 m, final tilt 3.45 deg | 实验组明显更好 |
| `roll90_h034` | final height 0.339 m, final tilt 1.82 deg | final height 0.349 m, final tilt 1.61 deg | 对照组姿态略好，但高度偏高 |
| `vx2_yaw6_h030` | final height 0.295 m, final tilt 0.07 deg | final height 0.300 m, final tilt 2.26 deg | 实验组明显更稳 |
| `vx3_yaw9_h030` | final height 0.298 m, final tilt 1.97 deg | final height 0.315 m, final tilt 3.48 deg | 实验组更稳 |

结论：`orientation_l2` 不应被视为“乱加奖励”，它补的是恢复后近直立阶段的姿态收敛和高速命令下的横滚/俯仰稳定性；它不替代 `upward` 的全姿态起身梯度，也不应在倒置阶段强压探索。后续如果调整该项，必须保留“倒置自救主要靠 `upward`，近直立稳定靠 `orientation_l2`”这个分工，并继续用 pitch-flip、roll90 和高速 yaw case 做 A/B。

成功条件只用于统计和验收，不是额外 reward：

```text
tilt < 15 deg
abs(base_height - target_height) < 0.05 m
norm(base_ang_vel) < 1.5 rad/s
wheel_contact_force > 1.0 N
```

### 禁止的奖励改法

- 不要只加高度奖励。单纯高度会鼓励弹跳、架腿或错误姿态支撑。
- 不要让高度奖励在倒置和侧躺时完全为 0；它应是弱抬高梯度，不是替代 `upward` 的主目标。
- 不要区分恢复阶段奖励和正常运动奖励；所有项都属于同一套 locomotion objective。
- 不要让 `upward` 依赖 `upright_gate`。倒地时 gate 接近 0，会断掉最需要的起身主梯度。
- 不要按初始姿态、倾角桶或内部 mask 修改 reward 权重。
- 不要把 leg contact 在倒地时设为强终止或强惩罚。自起必然会经历身体/腿部接触。
- 不要用 alive reward 或 success bonus 证明恢复成功。
- 不要区分 pitch 轴和 roll 轴添加奖励、权重、成功率或 curriculum gate。

## 终止规范

第一版终止逻辑也参考 RobotLab 的简化策略：不要用 bad orientation 或非法接触把倒地样本提前杀掉。

允许终止：

- `time_out`：常规 episode 超时。
- `out_of_bounds`：位置越界或明显离开训练区域。
- `invalid_state`：NaN、Inf、严重穿模等物理状态异常。

禁用或置零：

- `bad_orientation`：不能作为终止条件。
- `leg_contact` / `illegal_contact`：不能作为倒地时的终止条件。
- stagnation 类终止：第一版不启用，避免再次引入恢复窗口。
- `is_terminated` reward：权重为 0 或接近 0。

## 监控指标

每次恢复训练启动后，前 5 iter 必须确认：

1. reward 表里统一 locomotion 项权重存在且数值正确。
2. `Reset/curriculum_tilt_max_deg` 和 `Curriculum/command_vel/progress` 符合当前 iter 阶段。
3. `Reset/init_tilt_bin_*_ratio` 在课程放开后覆盖所有倾角桶。
4. `Reset/mean_init_tilt_deg` 和 `Reset/max_init_tilt_deg` 符合当前课程预期。
5. 没有新增 NaN、环境崩溃或异常终止暴涨。

训练中优先看这些 wandb 指标：

| 指标 | 期望方向 | 含义 |
|---|---|---|
| `Reset/curriculum_tilt_max_deg` | 阶段推进 | reset 倾角课程是否按 iter 放开 |
| `Curriculum/command_vel/progress` | 阶段推进 | 速度课程是否按 PPO iter 而非仿真 step 过快展开 |
| `Reset/joint_curriculum_progress` | 阶段推进 | 大腿/小腿关节扰动课程是否生效 |
| `Reset/init_tilt_bin_*_ratio` | 后期覆盖全桶 | reset 分布是否最终覆盖全角度随机 |
| `Locomotion/upward` | 上升 | RobotLab 风格全姿态直立项是否提供主梯度 |
| `Locomotion/upright_gate` | 上升 | 常规 tracking/height/contact 项是否逐步恢复生效 |
| `Locomotion/height_gate` | 不低于最小 gate | 倒置/侧躺时高度项是否仍保留弱梯度 |
| `SelfRight/success_rate_by_tilt_bin/*` | 上升 | 各倾角桶是否真正学会 |
| `SelfRight/time_to_success_p90_by_tilt_bin/*` | 下降 | 当前难度下是否足够快地恢复 |
| `SelfRight/post_success_survival_rate` | 上升 | 起身后是否能继续稳定存在 |
| `SelfRight/height_cond_rate` | 上升 | 高度是否进入成功窗口 |
| `SelfRight/stable_cond_rate` | 上升 | 起身后是否能停住角速度 |
| `SelfRight/wheel_contact_cond_rate` | 上升 | 是否由轮子重新支撑 |
| `Termination/invalid_state` | 不上升 | 是否出现仿真异常 |
| `Termination/time_out` | 正常 | episode 是否主要自然结束 |

判断优先级：

1. 先看 reset、速度、关节课程是否按 iter 展开。
2. 再看 `upward` 是否给倒地姿态提供梯度。
3. 再看 `height_gate` 是否避免倒置/侧躺高度梯度断掉。
4. 再看成功条件中哪一个卡住。
5. 最后才调 reward 权重。

## 验收指标

### Smoke 验收

修改 reset、reward 或 termination 后，必须先跑：

```bash
just smoke-recovery
```

本地 CPU smoke 使用 8 个 env；1 个 env 对 GRU/PPO batch 过窄，可能出现 loss 统计 NaN，不能作为有效训练信号。

通过标准：

- 5 iter 正常完成。
- 无导入错误、shape 错误、NaN 崩溃。
- reward 表包含统一 locomotion 项。
- `Reset/curriculum_tilt_max_deg`、`Curriculum/command_vel/progress` 和 `Reset/joint_curriculum_progress` 日志存在。

### 训练早期验收

前 200 iter：

- 当前 iter 的 reset 分布、速度命令和关节扰动课程正确。
- `Locomotion/upward` 应有非零值并能随姿态改善上升。
- `Locomotion/height_gate` 不应低于最小 gate，倒置/侧躺时高度项不能完全为 0。
- `Termination/invalid_state` 不应异常暴涨。

### 中期验收

1200-2200 iter：

- `SelfRight/success_rate_by_tilt_bin/*` 整体上升。
- `SelfRight/time_to_success_p90_by_tilt_bin/*` 整体下降。
- `SelfRight/wheel_contact_cond_rate` 与成功率差距缩小，说明不是“看似直立但没有轮子支撑”。
- 高倾角桶不能只靠低难度桶拉高总成功率。

### 最终验收

最终 checkpoint 必须同时满足训练端和 sim2sim：

| 场景 | 通过标准 |
|---|---|
| 0-30 deg | 3 s 内恢复，恢复后稳定 1 s |
| 30-75 deg | 至少 20 次，整体成功率 >= 80% |
| 75-130 deg | 至少 20 次，倾倒方位均匀覆盖，整体成功率 >= 70% |
| 130-180 deg | 至少 10 次，整体成功率 >= 50%，失败样本要保存 Rerun |
| 起身后 locomotion | 恢复后给 `vx=0.2-0.4 m/s` 指令不立即再摔 |
| 正常站走 | 接近直立初始化下 1000 control steps 稳定，低速命令不立即摔倒 |

sim2sim 建议命令：

```bash
uv run se3-sim2sim --checkpoint <ckpt> --viewer none --max-steps 1000 --print-every 100
uv run se3-sim2sim --checkpoint <ckpt> --viewer rerun --max-steps 1000 --initial-roll-deg 90 --initial-base-height 0.16
uv run se3-sim2sim --checkpoint <ckpt> --viewer rerun --max-steps 1000 --initial-roll-deg -90 --initial-base-height 0.16
uv run se3-sim2sim --checkpoint <ckpt> --viewer rerun --max-steps 1000 --initial-pitch-deg 90 --initial-base-height 0.16
uv run se3-sim2sim --checkpoint <ckpt> --viewer rerun --max-steps 1000 --initial-pitch-deg -90 --initial-base-height 0.16
```

注意：sim2sim 参数里的 roll/pitch/yaw 只是复现具体初始姿态的坐标表达；通过率统计仍只能按 `init_tilt_bin` 汇总，不能拆成 pitch 轴和 roll 轴指标。

## 调参流程

遇到自起失败，按下面顺序排查：

1. **reset 没全开**：`Reset/init_tilt_bin_*_ratio` 是否覆盖 0-180 deg。
2. **梯度断链**：`Locomotion/upward`、`upright_gain`、`height_gain` 是否长期为 0。
3. **成功条件卡住**：比较 `SelfRight/height_cond_rate`、`SelfRight/stable_cond_rate`、`SelfRight/wheel_contact_cond_rate`。
4. **终止误杀**：`bad_orientation`、`leg_contact`、`illegal_contact` 是否仍在终止 episode。
5. **正常能力不足**：接近直立初始化下正常站立 sim2sim 是否满足验收标准。

常见症状与首选动作：

| 症状 | 首选排查 | 首选改法 |
|---|---|---|
| 起不来但一直有动作 | success 四条件 | 找卡住条件，不先加大所有 reward |
| 高倾角桶完全没进展 | `SelfRight/success_rate_by_tilt_bin/inverted` | 检查 `upward` 是否在高倾角仍有梯度 |
| 看似站起但又倒 | `SelfRight/stable_cond_rate`、角速度 | 加强低角速度/完成保持，不加高度奖励 |
| 站起但轮子没接地 | `SelfRight/wheel_contact_cond_rate` | 检查 wheel contact gate 和接触传感器 |
| 正常站走能力不足 | command tracking、sim2sim | 检查 `upright_gate` 是否恢复 tracking，不按 reset 类型改 reward |

## 修改守则

- 改 reward 前先写下假设、预测指标和反证指标。
- 一次只改一个主因：reset 分布、奖励权重、成功条件或终止逻辑不要混在一次提交里。
- 修改后必须跑 `just smoke-recovery`。
- 正式训练前 5 iter 必须确认新权重出现在 reward 表。
- 如果前 200 iter 指标方向反了，停止训练，不要让错误配置跑完整 3000 iter。

## 文件索引

| 模块 | 文件 |
|---|---|
| 任务注册 | `src/se3_train/__init__.py` |
| 自起环境配置 | `src/se3_train/env_cfg.py` |
| GRU PPO 配置 | `src/se3_train/rl_cfg.py` |
| 统一 reset sampler | `src/se3_train/mdp/events.py` |
| 统一 rewards | `src/se3_train/mdp/rewards.py` |
| 自起 terminations | `src/se3_train/mdp/terminations.py` |
| GRU sim2sim runtime | `src/se3_sim2sim/policy.py` |
| sim2sim 初始姿态 CLI | `src/se3_sim2sim/cli.py` |
| 常用命令 | `justfile` |
