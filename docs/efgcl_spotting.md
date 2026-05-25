# EFGCL 跳跃辅助机制

## 状态

已实现于 `SE3-WheelLegged-Jump-PreTrain-GRU`。Fine-tune、play 和 sim2sim 不启用外部辅助。

相关文件：

- `src/se3_train/mdp/efgcl_stabilizer.py`
- `src/se3_train/mdp/jump_curriculums.py`
- `src/se3_train/env_cfg.py`

## 背景

当前跳跃 PreTrain 有两个主要风险：地面期主动蹬地速度起不来，以及空中姿态不稳。`diag_max_airborne_vz` 长期停在 0.02~0.11m/s 时，策略没有形成有效起跳；`diag_tilt_airborne_deg` 升到 50° 以上时，策略即使进入空中也无法稳定姿态。

EFGCL（External Force Guided Curriculum Learning）的核心思想是：训练早期通过物理辅助让策略更频繁经历成功状态，再根据成功率逐步撤掉辅助。当前实现包含两段辅助：地面蹬地期的目标高度前馈外力，以及空中期的姿态 spotting 力矩。

## 设计目标

1. 解决 PreTrain 早期主动起跳和空中姿态探索问题。
2. 不修改 sim2sim 和最终部署动力学。
3. 不把起跳动力和姿态稳定永久外包给外部 wrench，两段辅助强度都必须能衰减到 0。
4. 日志必须能判断辅助是否正在工作、是否正在退出。

## 激活范围

EFGCL 只在以下条件同时满足时施加外部 wrench：

- 当前任务为 `SE3-WheelLegged-Jump-PreTrain-GRU`
- `play=False`
- `jump_flag=1`
- 起跳外力：参考轨迹阶段 `jump_stage == 0`，且参考 `ref_vz > 0.05m/s`
- 空中力矩：参考轨迹阶段 `jump_stage == 1`
- 对应辅助 scale 大于 0

`SE3-WheelLegged-Jump-GRU` 会显式移除：

```python
cfg.events.pop("efgcl_jump_guidance", None)
cfg.curriculum.pop("efgcl_spotting", None)
cfg.curriculum.pop("efgcl_takeoff", None)
```

## 起跳外力

起跳外力作用在 `base_link`，通过 MuJoCo 的 `xfrc_applied` 写入世界系 z 向外力。外力由目标高度、机器人总质量和辅助时长前馈计算，不读取 actual vz：

```text
f_point = mg / 2 * (1 + sqrt(1 + 8h / (g * dt^2)))
force_world_z = clamp(2 * f_point * assist_fraction, 0, max_force) * assist_scale
```

默认 `dt = 0.16s`，`assist_fraction = 0.5`，`max_force = 260N`。`2 * f_point`
对应论文中左右两个辅助施力点的合力；本仓库写入的是 `base_link` 单个 body 上的等效合力。
外力只在参考轨迹已经进入向上蹬地片段时激活，避免和预蹲蓄力冲突。

## 空中力矩

力矩作用在 `base_link`，通过 MuJoCo 的 `xfrc_applied` 写入外部 wrench。

输入量：

- `projected_gravity_b`：机身坐标系下重力方向
- `root_link_ang_vel_b`：机身角速度
- `jump_stage`：参考轨迹阶段

力矩形式：

```text
torque_body = [K * pg_y, -K * pg_x, 0] - D * ang_vel_xy
```

其中：

- `pg_x` 近似 pitch 误差
- `pg_y` 近似 roll 误差
- `K = 1.2 Nm`
- `D = 0.08 Nms`
- 单轴限幅 `max_torque_nm = 0.8`

每个 step 会先清零 `base_link` 的外部 wrench，再按条件写入新外力和力矩，避免离开辅助阶段后残留。

## 衰减逻辑

`efgcl_takeoff_curriculum` 根据主动起跳速度成功率衰减起跳外力：

```text
takeoff_success = jump_flag
                  & jump_stage == 1
                  & actual_vz > max(0.2, 0.35 * vz_ref)
```

`efgcl_spotting_curriculum` 根据空中直立成功率衰减姿态力矩：

```text
upright_success = jump_flag
                  & airborne
                  & vz > 0.2
                  & tilt < 25°
```

两段成功率都使用 EMA 平滑，并且 PreTrain 前 300 iter 不撤辅助。起跳成功率达标后，每个训练 iter 将 `efgcl_takeoff_assist_scale` 减少 `0.005`；姿态成功率达标后，每个训练 iter 将 `efgcl_assist_scale` 减少 `0.005`，直到 0。

## 日志指标

新增指标：

- `EFGCL/assist_scale`：兼容旧 dashboard 的总辅助强度，取两段 scale 的最大值
- `EFGCL/takeoff_assist_scale`：当前起跳外力辅助强度系数
- `EFGCL/upright_assist_scale`：当前空中姿态辅助强度系数
- `EFGCL/mean_force_n`：当前激活 env 的平均起跳辅助外力
- `EFGCL/mean_torque_nm`：当前激活 env 的平均空中辅助力矩
- `Jump/diag_takeoff_success_rate`：蹬地向上段实际 vz 达标的 EMA 成功率
- `Jump/diag_upright_success_rate`：空中且 vz 达标且姿态达标的 EMA 成功率

已有指标仍需一起观察：

- `Jump/diag_tilt_airborne_deg`
- `Jump/diag_jump_success_rate`
- `Jump/diag_active_takeoffs`
- `Jump/diag_active_success_rate`
- `Episode_Termination/bad_orientation`
- `Jump/diag_leg_contact_termination`

## 期望现象

有效时应看到：

1. 训练早期 `EFGCL/mean_force_n > 0`
2. `Jump/diag_max_airborne_vz` 和 `Jump/diag_active_takeoffs` 上升
3. 空中样本出现后 `EFGCL/mean_torque_nm > 0`
4. `Jump/diag_tilt_airborne_deg` 下降或峰值受控
5. 达到阈值后两个辅助 scale 分别逐步下降
6. scale 下降期间 `diag_max_airborne_vz` 和 `diag_tilt_airborne_deg` 不回到原水平

如果 `takeoff_assist_scale` 降到 0 后 `diag_max_airborne_vz` 回落，说明策略依赖了外力，需要降低初始外力、放慢衰减或继续强化主动起跳奖励。如果 `upright_assist_scale` 降到 0 后姿态重新发散，说明策略依赖了外力矩，需要降低初始力矩、放慢衰减或回头检查起跳角动量来源。

## 风险

主要风险是策略学会依赖外部 wrench。当前通过三点控制：

1. 只在 PreTrain 开启
2. 起跳外力低于整机重量，并且只在参考向上蹬地片段开启
3. 用起跳速度成功率和姿态联合成功率分别驱动衰减到 0

另一个风险是外力或力矩过强掩盖起跳阶段左右不对称。监控时要同时看 `jump_joint_mirror`、`landing_symmetry`、`diag_tilt_airborne_deg` 和 sim2sim 结果。

## 验证记录

改动后必须完成本地验证：

- `uv run ruff format ...`
- `uv run ruff check ...`
- `SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Jump-PreTrain-GRU --env.scene.num-envs 1 --gpu-ids None`

CPU smoke 的 1 env 随机样本可能抽不到 jump episode，此时 `EFGCL/mean_force_n = 0` 或 `EFGCL/mean_torque_nm = 0` 是正常现象。
