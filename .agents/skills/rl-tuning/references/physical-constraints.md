# 物理约束分析

RL 训练中，**物理上限优先于奖励设计**。在分析奖励权重和梯度流之前，先算物理极限。若物理不够，改奖励没有意义。

## 目录

1. 动能需求估算
2. 电机可用功估算
3. RSI 经验覆盖估算
4. 关节行程分析
5. 本项目具体数值

---

## 1. 动能需求估算

要让质量为 m 的机器人质心以速度 vz 离地，起跳时所需动能：

```
KE = 0.5 * m * vz²
vz_required = sqrt(2 * g * h_target)   # h_target = 轮子/质心离地目标高度

示例：
  m = 12.67 kg，h_target = 0.8m（轮子离地）
  vz = sqrt(2 * 9.81 * 0.8) = 3.96 m/s
  KE = 0.5 * 12.67 * 3.96² = 99.4 J
```

**注意**：`h_target` 是轮子离地高度，不是 base_link 高度。base_link 高度包含机身几何高度（约 0.26m），需减去才是真实离地高度。

---

## 2. 电机可用功估算

```
W_motor = 力矩上限 × 关节行程 × 关节数

估算步骤：
1. 确认 torque_limit（rated_torque or stall_torque）
2. 确认起跳行程：从 RSI 收腿姿态到最大伸展的角度差
3. W = torque × Δq × n_joints × n_legs

示例（本项目）：
  torque = 20 N·m（rated），关节行程 lf0: 0.6rad, lf1: 0.7rad，4关节
  W = (0.6 + 0.7) * 20 * 4 = 104 J... 等等
  实际：W = (0.6+0.7) * 20 * 2(关节对) * 2(腿) = 104J? 
  → 小心双重计数，实际 W = sum(τ_i * Δq_i) 逐关节求和
  → lf0: 0.6*20=12J, lf1: 0.7*20=14J, 双腿 = (12+14)*2 = 52J（rated）
                                                            104J（stall）
```

**结论**：rated_torque → 52J < 99J（不够）；stall_torque → 104J > 99J（刚好够）

---

## 3. RSI 经验覆盖估算

RSI（Reference State Initialization）注入的最大速度决定了策略能体验到的最高空中状态：

```
h_experience_max = vz_max² / (2 * g)

vz_max = 2.8 m/s → h_max = 2.8²/19.62 = 0.40m
vz_max = 3.96 m/s → h_max = 3.96²/19.62 = 0.80m（刚好覆盖）
```

**RSI-课程配合规则**：`vz_max_rsi` 必须随课程 `h_target_max` 同步更新：
```python
vz_max = sqrt(2 * g * term.cfg.jump_height_range[1])
```

若 RSI 不更新，课程扩大后：
- 策略被要求达到 h_target=0.8m 的 vz_ref=3.96 m/s
- 但训练中从未见过 vz > 2.8 m/s 的空中状态
- `jump_vel_z_tracking` 变成纯惩罚，无正向梯度

---

## 4. 关节行程分析

从 MJCF 读关节限位，确认蹬腿行程是否足够：

```
起跳行程 = |q_default - q_extended_limit|
RSI 收腿行程 = |q_rsi_folded - q_extended_limit|

本项目：
  lf1 range = [-0.6, 0.8]
  lf1 default = 0.207，RSI 收腿 = 0.5
  蹬腿方向：从 0.5 到 -0.6，行程 = 1.1 rad（不是瓶颈）
```

**检查关节限位软惩罚是否在起跳关键时刻触发**：
若 `dof_pos_limits`（weight 较大）在蹬腿末端触发，会压制完整行程。

---

## 5. 本项目具体数值

| 参数 | 数值 | 来源 |
|---|---|---|
| 机器人总质量 | ~12.67 kg | MJCF inertial 累加 |
| DM8009P rated_torque | 20 N·m | `motor.py:91` |
| DM8009P stall_torque | 40 N·m | `motor.py:89` |
| 腿部行程（rated，4关节） | ~52 J | 计算 |
| 腿部行程（stall，4关节） | ~104 J | 计算 |
| 0.8m 跳跃所需动能 | ~99 J | 计算 |
| RSI vz_max（旧，固定） | 2.8 m/s → 0.40m | `events.py:72`（旧） |
| RSI vz_max（新，动态） | sqrt(2g×h_max) | `events.py`（修复后） |
| 气弹簧弹性势能（双侧） | ~0.27 J | MJCF stiffness=2000 |

**气弹簧贡献可忽略**：0.27J / 99J = 0.27%，不是跳跃高度的瓶颈。
