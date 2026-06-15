# Recovery V2 两阶段训练方案

> 状态：Discovery 任务已实现并完成首轮短训，Deploy 阶段尚未实现。本文定义下一版 recovery 训练的目标、任务拆分、reset 分布、状态缓存格式、训练流程和验收标准。

## 背景

当前 `SE3-WheelLegged-Recovery-GRU` 使用单任务全姿态随机 reset：root 姿态由程序随机采样，root 高度再通过包络和碰撞 clearance 修正，关节姿态按课程逐步 full random。这个方案覆盖面宽，但与真机倒地后的稳定接触状态差距较大，也会让策略在还没发现基础起身动作时就同时面对姿态泛化、关节泛化、接触泛化和命令泛化。

Recovery V2 改为两阶段训练：

1. Discovery 阶段：在更容易的环境中发现起身运动。
2. Deploy 阶段：把已发现的起身运动迁移到真实倒地状态分布，并增强 sim2real 泛化。

这不是修改 actor contract 的方案。actor 观测维度、动作维度、GRU 架构、部署 I/O 合同都保持不变。

## 核心原则

1. 先发现动作，再增加真实复杂度。
2. 第一阶段不追求 sim2real，只追求稳定发现翻身、撑起、站稳的运动模式。
3. 第二阶段不再依赖“随机欧拉角 + 高度修正”假装真实倒地，而是使用离线物理 settle 后的状态缓存。
4. `pose_type`、`cache_split`、`reset_source` 只能用于 reset、日志和评估，不能进入 actor observation。
5. 旧任务 `SE3-WheelLegged-Recovery-GRU` 保留为 baseline，不直接覆盖。
6. 每次训练仍必须显式开启 MJLab Viser 值守，确认姿态、接触和奖励面板。

## 任务拆分

新增两个任务：

| 阶段 | 任务名 | 作用 | 初始化 |
|---|---|---|---|
| Discovery | `SE3-WheelLegged-Recovery-Discovery-GRU` | 标准倒地姿态中发现起身动作 | 从零训练 |
| Deploy | `SE3-WheelLegged-Recovery-Deploy-GRU` | 真实倒地状态缓存 + 更真实碰撞 + 域随机化 | warm-start Discovery checkpoint |

`SE3-WheelLegged-Recovery-Deploy-GRU` 使用 `Se3WarmStartRunner`，只加载 actor/critic 权重，不继承 optimizer、iteration 和环境计数。这样第二阶段课程从新 run 的 iter 0 开始。

## 不变的合同

动作仍为 6 维：

```text
[LF, LB, RF, RB, l_wheel, r_wheel]
```

actor 观测仍为 32 维，沿用当前 recovery/flat/jump 合同：

```text
base_ang_vel        3
projected_gravity   3
commands            5   [vx, yaw_rate, pitch, roll, base_height]
leg_joint_pos       4
leg_joint_vel       4
wheel_pos_zero      2
wheel_vel           2
last_actions        6
jump_commands       3
```

部署端 `se3-nx-recovery` 和 `se3-sim2sim` 不应因为本方案改变输入输出维度。

## 阶段一：Discovery

### 目标

让 GRU policy 在简单、稳定、低噪声的环境中先发现以下能力：

1. 从仰卧翻起。
2. 从俯卧翻起。
3. 从左右侧躺撑起。
4. 起身后保持直立高度和低速度稳定。

### 环境

| 项 | 设置 |
|---|---|
| MJCF | 默认 `fourbar-surrogate` |
| 碰撞 | 简化碰撞，优先训练速度和动作发现 |
| 地形 | 平地 |
| command | 早期 `vx=0`，`yaw_rate=0` |
| height command | 初期窄范围 `0.24-0.30 m` |
| 域随机化 | 关闭或极弱 |
| push disturbance | 关闭 |
| root 初速度 | 初期 0，后期小扰动 |

### Reset 分布

新增标准姿态 reset，不使用当前全姿态随机 reset。

推荐初始采样比例：

| `pose_type` | 含义 | 比例 |
|---:|---|---:|
| 0 | standing / near-upright | 0.05-0.10 |
| 1 | left_side | 0.15-0.20 |
| 2 | right_side | 0.15-0.20 |
| 3 | prone / 俯卧 | 0.25-0.30 |
| 4 | supine / 仰卧 | 0.25-0.30 |

姿态定义：

```text
standing: roll=0, pitch=0
left_side: roll=+90 deg, pitch=0
right_side: roll=-90 deg, pitch=0
prone: roll=0, pitch=+180 deg
supine: roll=0, pitch=-180 deg
yaw: [-pi, pi]
```

课程：

| iter | 姿态 jitter | root 速度 | 腿部关节扰动 | command |
|---:|---:|---:|---:|---|
| 0 | ±5 deg | 0 | 0 | `vx=0`, `yaw=0` |
| 300 | ±10 deg | lin ±0.03, ang ±0.10 | ±0.10 rad | `vx=0`, `yaw=0` |
| 800 | ±15 deg | lin ±0.05, ang ±0.20 | ±0.20 rad | `vx=0`, `yaw=0` |
| 1500 | ±20 deg | lin ±0.08, ang ±0.30 | ±0.25 rad | `vx/yaw` 小范围可选 |

### 奖励

Discovery 阶段保留当前 recovery 奖励的大方向，但降低强正则，让策略敢探索大幅翻身动作。

建议：

1. 增加或启用 recovery 正向奖励：
   - `recovery_upright`
   - `recovery_progress`
   - `recovery_stable_bonus`
2. 弱化早期正则：
   - `action_smoothness` 降到当前 30-50%
   - `leg_torques` 降到当前 50%
   - `leg_power` 降到当前 50%
3. 保留硬安全：
   - catastrophic state termination
   - 数值异常终止
   - 明显爆速过滤
4. 不启用按姿态类型切换 reward。

### Discovery 验收

第一阶段通过条件：

| 指标 | 门槛 |
|---|---:|
| 标准 supine 成功率 | >80% |
| 标准 prone 成功率 | >80% |
| 左右侧躺成功率 | >80% |
| standing 保持率 | >90% |
| 平均起身时间 | 随训练下降 |
| 动作饱和率 | 不持续贴近 100% |

成功定义：

```text
5 s 内进入：
  tilt < 15 deg
  abs(base_height - command_height) < 0.04 m
  base_ang_vel_xy < 1.0 rad/s
  wheel contact 合理
并保持至少 0.5 s
```

### 首轮 Discovery 候选

当前选定 `model_1500.pt` 作为 Discovery 阶段候选 checkpoint：

```text
run: logs/rsl_rl/se3_wheel_leg/2026-06-12_23-30-01_recovery_discovery_heightgate_continue_m300_4096_55f65a2
checkpoint: model_1500.pt
remote: /root/project/se3_wheel_leg/logs/rsl_rl/se3_wheel_leg/2026-06-12_23-30-01_recovery_discovery_heightgate_continue_m300_4096_55f65a2/model_1500.pt
```

选择依据：

| 指标 | 1500 附近观测值 |
|---|---:|
| `Recovery/diag_upright_15deg_rate` | `0.925-0.936` |
| `Recovery/diag_upright_30deg_rate` | `0.930-0.941` |
| `Recovery/diag_height_error_abs_m` | `0.0048-0.0053 m` |
| `Recovery/diag_height_ok_2cm_rate` | `0.956-0.965` |
| `Recovery/diag_raw_action_saturation_rate` | `0.039-0.045` |
| `Episode_Termination/catastrophic_state` | `0.000` |

`model_1500.pt` 不是最终部署模型，只作为 Deploy 阶段 warm-start 的 Discovery 候选。后续进入 Deploy 前，需要先用固定标准姿态评估和 Viser 复核动作质量；如需更激进的高峰候选，可保留 `model_1400.pt` 作为对照。

### Discovery 固定姿态评估

2026-06-13 在远端 L40S 上对 `model_1500.pt` 和 `model_1400.pt` 跑固定姿态评估：

- 每个姿态 `1024` 个 episode。
- 每个 episode `5.0s`，policy step `0.02s`。
- 成功定义：`tilt < 15deg` 且高度误差 `< 2cm` 连续保持 `0.5s`。
- 指令高度固定 `0.26m`，yaw/jitter/root velocity/joint randomization 全部关掉。
- 输出 JSON：`/root/project/se3_wheel_leg/tmp/recovery_eval_fixed_poses_model1500.json` 和 `/root/project/se3_wheel_leg/tmp/recovery_eval_fixed_poses_model1400.json`。

| checkpoint | pose | success | time_s | height_err_m | action_sat | leg_contact | early_done | final_tilt_deg |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `model_1500.pt` | standing | 1.000 | 0.120 | 0.0004 | 0.000 | 0.000 | 0.000 | 3.92 |
| `model_1500.pt` | left_side | 1.000 | 0.821 | 0.0008 | 0.113 | 0.012 | 0.000 | 3.97 |
| `model_1500.pt` | right_side | 1.000 | 0.680 | 0.0014 | 0.068 | 0.016 | 0.000 | 4.42 |
| `model_1500.pt` | prone | 1.000 | 1.805 | 0.0034 | 0.239 | 0.008 | 0.000 | 4.41 |
| `model_1500.pt` | supine | 1.000 | 1.805 | 0.0034 | 0.239 | 0.008 | 0.000 | 4.41 |
| `model_1400.pt` | standing | 1.000 | 0.120 | 0.0015 | 0.000 | 0.000 | 0.000 | 3.32 |
| `model_1400.pt` | left_side | 1.000 | 0.840 | 0.0005 | 0.110 | 0.016 | 0.000 | 3.61 |
| `model_1400.pt` | right_side | 1.000 | 0.640 | 0.0001 | 0.075 | 0.012 | 0.000 | 3.73 |
| `model_1400.pt` | prone | 1.000 | 1.728 | 0.0018 | 0.217 | 0.008 | 0.000 | 3.62 |
| `model_1400.pt` | supine | 1.000 | 1.728 | 0.0018 | 0.217 | 0.008 | 0.000 | 3.62 |

结论：`model_1500.pt` 五类固定姿态没有明显回撤，继续作为 Discovery 主候选。`model_1400.pt` 在 `right_side/prone/supine` 上略快、最终误差和动作饱和率略低，保留为备胎和后续 Deploy A/B 对照，但不替换 `model_1500.pt`。

## 离线倒地状态缓存

### 目标

生成真实物理 settle 后的倒地初始状态，作为 Deploy 阶段主要 reset 来源。

当前 `src/se3_tools/recovery_state_cache.py` 已有雏形，但需要升级：

1. 不再使用固定 `qpos` slice 作为长期格式。
2. 必须保存 `joint_names`，训练读取时按当前 MJLab asset 的 joint name 重排。
3. 必须生成 train/eval split。
4. 必须记录 `pose_type`。

### 生成流程

每个样本：

1. 从标准 `supine/prone/left_side/right_side` 姿态出发。
2. 随机 yaw。
3. 随机 policy 语义腿部 DOF，主动杆夹角保持在机械限位内。
4. base 从 `0.35-0.55 m` 高度释放。
5. 在 MuJoCo 中 settle 5-10 s。
6. 过滤不稳定样本。
7. 保存 root、joint、速度和标签。

过滤条件：

| 项 | 建议 |
|---|---|
| NaN/Inf | 必须过滤 |
| root 速度 | `norm(qvel[0:6]) < 1.0` 可先用作默认 |
| 腿部速度 | 最大绝对值 `< 3.0 rad/s` |
| base 高度 | `> 0.04 m` |
| 关节角 | 不超过合理机械范围 |
| 穿模/爆接触 | 通过最小碰撞高度和接触力阈值过滤 |

### 缓存格式

建议保存为：

```text
assets/recovery_states/serialleg_closedchain_recovery_v2.npz
```

字段：

```text
root_pos          float32 [N, 3]
root_quat         float32 [N, 4]  # MuJoCo wxyz
root_lin_vel      float32 [N, 3]
root_ang_vel      float32 [N, 3]
joint_names       str     [J]
joint_pos         float32 [N, J]
joint_vel         float32 [N, J]
pose_type         int64   [N]
split             int64   [N]     # 0=train, 1=eval
settle_steps      int64   [N]
source_mjcf       str
```

推荐规模：

| 姿态 | train | eval |
|---|---:|---:|
| supine | 10000 | 10000 |
| prone | 10000 | 10000 |
| left_side | 5000-10000 | 5000-10000 |
| right_side | 5000-10000 | 5000-10000 |

## 阶段二：Deploy

### 目标

从 Discovery checkpoint warm-start，把已发现的起身动作迁移到真实倒地状态缓存和更真实仿真设置。

### 环境

| 项 | 设置 |
|---|---|
| warm-start | Discovery checkpoint |
| reset 主来源 | recovery state cache |
| MJCF | 优先 `closedchain` / OBB 裁剪模型 |
| 碰撞 | 更真实碰撞 |
| 地形 | 平地起步，后续再加轻微地形变化 |
| 域随机化 | 第二阶段逐步打开 |
| push disturbance | 后期再打开 |

如果 closedchain 训练吞吐太低，可以拆成 Deploy-2A 和 Deploy-2B：

1. Deploy-2A：fourbar surrogate + cache + 强正则。
2. Deploy-2B：closedchain OBB + cache + sim2real 域随机化。

### Reset 分布

推荐：

| 来源 | 比例 |
|---|---:|
| train cache | 0.80-0.90 |
| 标准倒地姿态 | 0.05-0.10 |
| standing / near-upright | 0.05-0.10 |

Deploy 阶段不再使用当前全姿态欧拉角随机作为主分布。程序随机姿态只能作为少量补充或 ablation。

### Command 课程

| iter | `vx` | `yaw_rate` | height |
|---:|---:|---:|---|
| 0 | 0 | 0 | 0.24-0.30 |
| 500 | ±0.3 | ±0.3 | 0.23-0.32 |
| 1000 | ±0.5 | ±0.5 | 0.22-0.35 |
| 1500 | ±1.0 | ±0.75 | 0.205-0.37 |
| 2200 | ±1.5 | ±1.0 | 0.195-0.390 |

### 正则和域随机化

Deploy 阶段恢复或加强：

1. action smoothness
2. leg torque / leg power
3. wheel air velocity penalty
4. collision penalty
5. leg contact penalty
6. friction randomization
7. mass / inertia / COM randomization
8. PD gain randomization
9. action delay randomization

这些复杂度只在第二阶段加入，避免 Discovery 阶段被过早干扰。

## 评估与指标

### 训练日志

必须新增或确认存在以下指标：

```text
Recovery/reset_source_cache_ratio
Recovery/reset_source_standard_ratio
Recovery/reset_source_standing_ratio
Recovery/pose_type_supine_ratio
Recovery/pose_type_prone_ratio
Recovery/pose_type_left_side_ratio
Recovery/pose_type_right_side_ratio
Recovery/success_rate
Recovery/success_rate_by_pose_type/supine
Recovery/success_rate_by_pose_type/prone
Recovery/success_rate_by_pose_type/left_side
Recovery/success_rate_by_pose_type/right_side
Recovery/time_to_upright_s
Recovery/upright_hold_rate
Recovery/final_height_error_abs_m
Recovery/action_saturation_rate
Recovery/leg_contact_rate
```

`pose_type` 指标只用于日志和评估，不能进入 actor observation。

### 固定评估集

每个候选 checkpoint 至少跑：

1. Discovery 标准姿态评估。
2. Deploy held-out cache 评估。
3. sim2sim `roll90`。
4. sim2sim `pitch-flip`。
5. sim2sim `supine`。
6. sim2sim `prone`。
7. closedchain sim2sim headless。

### 最终验收

最终 checkpoint 必须满足：

| 场景 | 门槛 |
|---|---:|
| Discovery 标准姿态 | 每类 >80% |
| Deploy held-out cache | 每类 >80% |
| closedchain roll90 | 能起身并保持 |
| closedchain pitch-flip | 能起身并保持 |
| supine/prone sim2sim | 能起身并保持 |
| NX dry-run | 推理链路正常 |

## 建议命令入口

新增 just recipe：

```bash
just smoke-recovery-discovery
just train-recovery-discovery-light
just train-recovery-discovery

just gen-recovery-cache

just smoke-recovery-deploy
just train-recovery-deploy-light
just train-recovery-deploy
```

底层命令示例：

```bash
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Recovery-Discovery-GRU \
  --env.scene.num-envs 8 \
  --gpu-ids None

uv run --env-file .env se3-train SE3-WheelLegged-Recovery-Discovery-GRU \
  --env.scene.num-envs 4096

uv run se3-recovery-state-cache \
  --mjcf-variant closedchain \
  --states-per-type 20000 \
  --train-per-type 10000 \
  --settle-s 10 \
  --output assets/recovery_states/serialleg_closedchain_recovery_v2.npz

uv run --env-file .env se3-train SE3-WheelLegged-Recovery-Deploy-GRU \
  --agent.load-run <discovery_run> \
  --agent.load-checkpoint model_<iter>.pt \
  --env.scene.num-envs 4096
```

## 实施顺序

1. 新增本文档，冻结方案和术语。
2. 新增 Discovery task，先只做标准姿态 reset。
3. 新增 `just smoke-recovery-discovery`。
4. 跑 Discovery smoke，确认环境不崩溃。
5. 跑 Discovery 200-500 iter，确认起身动作开始出现。
6. 升级 `se3_tools.recovery_state_cache`，输出带 `joint_names` 和 split 的 v2 cache。
7. 用小规模 cache 跑读取 smoke。
8. 新增 Deploy task，接入 cache reset。
9. 用 `Se3WarmStartRunner` 从 Discovery checkpoint 启动 Deploy。
10. 跑 Deploy smoke，确认 cache reset、reward 和 diagnostics 正常。
11. 跑 Deploy 200-500 iter，确认 held-out cache 指标方向正确。
12. 再加入 closedchain / 更真实碰撞 / 域随机化。
13. 建立固定 sim2sim 评估脚本或 just recipe。
14. 将通过验收的 Deploy checkpoint 导出到 NX runtime。

## 风险与控制

| 风险 | 控制 |
|---|---|
| Discovery 学到过激翻身动作 | 第二阶段逐步恢复 action/torque/power 正则 |
| cache 关节列错位 | v2 cache 必须保存 `joint_names`，训练读取按名称重排 |
| cache 分布太窄 | 保留标准姿态补充 reset，并按 pose_type 统计成功率 |
| closedchain 训练太慢 | 先做 Deploy-2A，再做 Deploy-2B |
| actor 学 pose 标签 | `pose_type` 不进入 actor observation |
| command 太早打开导致恢复失败 | Deploy command 课程从 0 速度开始 |
| standing mix 过多稀释恢复学习 | standing ratio 控制在 5-10%，按日志监控 |

## 与旧方案关系

`docs/plan/gru_recovery_training.md` 描述的是当前单任务全随机 recovery baseline。Recovery V2 不删除该 baseline，而是新增一条更接近 sim2real 的两阶段路线。

短期对比方式：

1. 同样训练预算下比较 `Recovery/success_rate`。
2. 同样 checkpoint 频率下比较 sim2sim `roll90`、`pitch-flip`、`supine`、`prone`。
3. 使用 held-out cache 比较真实倒地状态泛化。

如果 Recovery V2 在 Discovery 阶段无法稳定发现动作，先调标准姿态 reset、正向 recovery reward 和弱正则，不直接进入 Deploy 阶段。
