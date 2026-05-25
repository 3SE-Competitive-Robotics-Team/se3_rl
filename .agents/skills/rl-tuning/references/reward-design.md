# 奖励函数设计原则

## 目录

1. 核心原则
2. 门控设计
3. 时序配合
4. 量级校准
5. 常见 Bug 模式
6. 本项目奖励架构速查

---

## 1. 核心原则

**原则 1：奖励信号必须在目标行为发生时为正**

在设计或 debug 奖励函数时，先问：当目标行为发生时（如起跳 vz=3 m/s），该奖励项返回什么值？

```python
# 验证：用真实数据手算
vz = 3.0
reward = clamp(vz, min=0) * weight  # jump_takeoff_drive
# 返回 3.0 * weight_dt > 0 ✓
```

**原则 2：奖励项不能相互矛盾**

两个同时激活的奖励项，不应在同一行为上给出相反的梯度方向。最常见：`tracking_lin_vel` 含 `vz²` 惩罚项，与起跳奖励直接冲突。

**原则 3：量级决定优先级**

策略会优先学习高权重奖励的行为。当多个奖励竞争时，检查：
```
有效权重 = weight × dt × 典型激活量级
```
两个目标行为若有效权重接近，策略可能在两者之间振荡。

---

## 2. 门控设计

门控（gate）是控制"奖励项在什么条件下激活"的机制，是奖励设计中最容易出 bug 的地方。

### 门控层次

```
jump_flag 门控（任务级）
  └─ jump_stage 门控（参考轨迹阶段：grounded/airborne/landing）
       └─ 方向门控（信号级：vz>0, q_flexion>0）
```

每一层都是必要的：
- `jump_flag` 防止行走 env 触发跳跃奖励
- `jump_stage` 防止 RSI 注入的空中帧触发地面奖励，也防止参考地面帧触发空中奖励
- 方向 避免双向行为套利（如抖腿套利）

### 门控豁免

某些行走惩罚需要在跳跃时豁免：

```python
def leg_torques_no_jump(env, command_name, ...):
    jump_flag = cmd[:, 5] > 0.5
    result = leg_torques(env, ...)
    return result * (~jump_flag).float()  # jump_flag=1 时清零
```

需要豁免的场景：起跳瞬间的峰值力矩、起跳时的关节大幅偏离、起跳时的垂直速度。

---

## 3. 时序配合

跳跃是一个时序任务（蹲下→起跳→飞行→着陆），需要为每个阶段设计奖励，并确保阶段间有正确的时序逻辑。

```
阶段 0（grounded，蓄力）：jump_squat_drive 奖励膝关节屈曲（vz≤0）
阶段 0（grounded，蹬腿）：jump_takeoff_drive 奖励质心向上速度（vz>0）
阶段 1（airborne，上升）：jump_vel_encourage/jump_knee_phase（屈曲收腿）
阶段 1（airborne，下降）：jump_knee_phase（伸腿缓冲）/knee_landing_prep
阶段 2（landing）：landing_symmetry 对称着陆
```

**关键检查**：每个阶段的奖励在上一阶段末端和下一阶段初始是否平滑过渡？是否存在某个时刻所有奖励都为零（梯度真空）？

---

## 4. 量级校准

`scale_rewards_by_dt=True` 时，实际每步奖励 = 函数返回值 × weight × dt。

```python
# 计算有效权重
control_dt = sim_dt * decimation = 0.002 * 5 = 0.01s

# 示例
jump_takeoff_drive: vz=1.5m/s × 150 × 0.01 = 2.25/step
jump_vel_z_tracking: |1.5-3.96|-0.5 × 150 × 0.01 = 2.94/step (惩罚)
```

**平衡检查**：起跳奖励的净梯度（正奖励 - 相关惩罚）是否为正？

---

## 5. 常见 Bug 模式

### Bug 1：Python bool × Tensor 门控失效
见 `references/symptom-to-hypothesis.md` 第 6 条。

### Bug 2：奖励套利（Reward Hacking）

**现象**：策略找到了一种非预期的行为来最大化奖励。

**经典案例**：奖励关节速度负分量（鼓励伸腿方向）→ 策略学会"原地抖腿"，关节速度正负交替，负分量均值非零。

**防御方法**：奖励应只有一种行为能满足。例如，改为奖励质心 vz > 0（真实起跳）而非关节速度。

### Bug 3：下限方向 vs 上限方向混淆

```python
# 常见错误：超出软限位的方向写反
out_of_limits = -(pos - limits[:, :, 0]).clip(max=0.0)   # 正确：超出下限的正量
out_of_limits += (pos - limits[:, :, 1]).clip(min=0.0)   # 正确：超出上限的正量
```

### Bug 4：reduction 维度错误

```python
force_mag = torch.norm(data.force, dim=-1)  # [num_envs, num_sensors]
airborne = (force_mag < threshold).all(dim=1)  # 必须 all(dim=1)，双轮都离地
# 错误用 any() → 单轮离地就算空中
```

### Bug 5：默认值覆盖 per-env 赋值

```python
# 错误（链式索引，返回副本不写回）
joint_pos[jump_mask][:, idx] = value

# 正确（直接 in-place）
jump_ids = jump_mask.nonzero(as_tuple=True)[0]
joint_pos[jump_ids[:, None], torch.tensor([idx])] = value
# 或
joint_pos[jump_mask, idx] = value  # 布尔索引单个维度
```

---

## 6. 本项目奖励架构速查

### 行走奖励（`env_cfg.py:194`）

| 奖励项 | 方向 | 核心作用 |
|---|---|---|
| `tracking_lin_vel` | + | x 速度跟踪，含 vz² 惩罚 |
| `tracking_height` | + | 高度跟踪（传感器：`base_height_sensor` min reduction） |
| `stand_still` | - | 静止时关节偏离惩罚（高度自适应 sigma） |
| `dof_pos_limits` | - | 关节超限惩罚 |

### 跳跃奖励（`env_cfg.py:659+`，jump_flag=1 时激活）

| 奖励项 | 阶段 | 激活条件 |
|---|---|---|
| `jump_squat_drive` | grounded 蓄力 | stage==0 & grounded & vz≤0 & q_flexion>0 |
| `jump_takeoff_drive` | grounded 蹬腿 | stage==0 & grounded & vz>0 |
| `jump_vel_encourage` | airborne | stage==1 & both_airborne |
| `jump_knee_phase` | airborne | stage==1，上升屈曲/下降伸展 |
| `jump_height_tracking` | airborne 上升 | stage==1 & vz≥0 |
| `jump_vel_z_tracking` | airborne 上升 | stage==1 & vz≥0 |
| `knee_landing_prep` | airborne 下降/landing | stage==2 or fast_descent |

---

## 7. 权重量级与配合关系

### 权重量级决定信号主次

`scale_rewards_by_dt=True` 时，有效权重 = `weight × dt × 典型激活量级`。两个奖励项若**权重差距 > 10×**，小权重项对策略基本无影响，策略会忽略它。

**配合型奖励必须保持量级接近**：
- 时序配合的两个奖励（如 `jump_squat_drive` + `jump_takeoff_drive`）应该权重相近，否则"先蹲"阶段的信号被"起跳"阶段淹没，策略不会学会完整序列
- 惩罚项与正向奖励的净梯度：如果某行为的正向奖励远大于对应的惩罚，策略会接受惩罚代价去换取奖励；反之则完全回避该行为

### 权重信号弱的诊断

`Episode_Reward/xxx` 的绝对值持续 < 0.01 且不随训练变化，通常意味着：
1. 该项激活频率极低（门控条件过于严格）
2. 权重相对其他奖励项太小（策略优先学习其他信号）
3. 典型激活量级太小（函数返回值本身就接近 0）

三种情况需要分别处理：(1) 检查门控条件，(2) 提高权重，(3) 检查函数实现。

### 惩罚项的负向局部最优

**高额惩罚项 + 目标行为强关联 = 策略主动回避该行为。**

如果某个目标行为（如高速起跳）在发生后经常触发高额终止惩罚（如 leg_contact landing），策略会学到"这个行为危险"，从而主动压低该行为的发生频率。表现为 `diag_max_airborne_vz` 振荡不爬升——策略在"探索高跳"和"规避惩罚"之间来回摆动。

**修复方向**：消除导致终止的根因（如空中姿态），而不是降低惩罚权重（降权只会让策略更激进，但着陆依然失败）。
| `landing_symmetry` | landing | stage==2 |
