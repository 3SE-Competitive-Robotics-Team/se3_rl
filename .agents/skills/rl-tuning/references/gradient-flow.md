# 梯度流分析

梯度流分析的目标：找到从"目标行为"到"策略接收到梯度信号"之间的完整路径，定位断链位置。

## 目录

1. MJLab env.step 完整生命周期（**必读，调试所有时序问题的基础**）
2. Reset 的完整行为与禁忌
3. 梯度流路径模型
4. 门控条件排查
5. RSI 与课程配合
6. 奖励冲突识别
7. 诊断工具：wandb 指标解读

---

## 1. MJLab env.step 完整生命周期

来源：`mjlab/envs/manager_based_rl_env.py:step()`，版本 mjlab 1.3.x。

```
① action_manager.process_action()
② for _ in range(decimation):
     action_manager.apply_action()
     scene.write_data_to_sim()
     sim.step()                    ← MuJoCo 物理步
     scene.update(dt)              ← 传感器数据刷新（ContactSensor.force 在此更新）
     metrics_manager.compute_substep()
③ episode_length_buf += 1
④ termination_manager.compute()   ← 读传感器（有效），判断终止，可写 extras（但 log 即将被清空）
⑤ reward_manager.compute()        ← 读传感器（有效），奖励计算
⑥ metrics_manager.compute()
⑦ _reset_idx(reset_env_ids)       ← ⚠️ Reset 在此发生（见下节）
   scene.write_data_to_sim()
⑧ sim.forward()                   ← reset 后物理前向计算
⑨ command_manager.compute()       ← _update_metrics 在此调用，传感器已失效
⑩ event_manager.apply(step/interval)
⑪ sim.sense()
⑫ observation_manager.compute()
```

**关键时序结论：**

| 调用位置 | 传感器数据 | extras["log"] |
|---|---|---|
| ④ termination_manager | **有效** | 即将被 ⑦ 清空，写入无意义 |
| ⑤ reward_manager | **有效** | 即将被 ⑦ 清空，写入无意义 |
| ⑦ _reset_idx 内部 | 被清空（cache invalidated） | **在此初始化为 `{}`** |
| ⑨ command_manager | **失效**（reset env 的传感器已重算） | 有效，此处写入才能上报 |

---

## 2. Reset 的完整行为与禁忌

`_reset_idx` 按以下顺序执行（`manager_based_rl_env.py:543`）：

```python
curriculum_manager.compute(env_ids)   # 更新课程参数
sim.reset(env_ids)                     # MuJoCo 状态清零
scene.reset(env_ids)                   # ⚠️ 所有传感器 reset() → _invalidate_cache()
event_manager.apply(mode="reset")      # 执行 reset 事件（random init 等）
# ——————————————————————————————————————
extras["log"] = dict()                 # ⚠️ log 清空
observation_manager.reset(env_ids)     # 各 manager reset，返回值写入 log
action_manager.reset(env_ids)
reward_manager.reset(env_ids)
metrics_manager.reset(env_ids)
curriculum_manager.reset(env_ids)
command_manager.reset(env_ids)         # JumpCommandTerm._resample_command 在此
event_manager.reset(env_ids)
termination_manager.reset(env_ids)
episode_length_buf[env_ids] = 0
```

### Reset 之后不能做的事

1. **不能读传感器数据**：`scene.reset()` 调用所有传感器的 `.reset()`，触发 `_invalidate_cache()`。下次访问 `.data` 时重新从 MuJoCo 计算——此时是 reset 后的新状态（无接触、新位置），原来的接触/力数据永久丢失。

2. **不能依赖 `extras["log"]` 保存 reset 之前的信息**：`extras["log"]` 在 `_reset_idx` 内部被 `= dict()` 覆盖。在 ④⑤ 写入的任何内容在 ⑦ 之后全部消失。

### Reset 之前必须做的事（数据抢救模式）

如果需要把"终止那一帧"的物理数据传递到 `command_manager._update_metrics`（⑨），标准做法是：

**在 ④ termination_manager 或 ⑤ reward_manager 中，把数据写入 `extras` 的非 log key**（如 `extras["_my_diag"]`）。`_reset_idx` 只清空 `extras["log"]`，其他 key 不受影响，可以跨 reset 存活到 ⑨。

```python
# ④ 终止函数中（reset 之前，传感器有效）：
env.extras["_my_diag"] = {"my_metric": value}   # 存入非 log key

# ⑨ command_manager._update_metrics 中（reset 之后，从非 log key 读）：
diag = self._env.extras.get("_my_diag")
if isinstance(diag, dict):
    self._env.extras["log"].update(diag)          # 搬入 log 上报
```

### 可以在 Reset 之后读的数据

- `robot.data.joint_pos/vel`（已被 reset 到新初始态，反映新 episode 状态）
- `episode_length_buf`（已被清零）
- `command_manager` 的内部状态（已被 `_resample_command` 更新）

---

## 3. 梯度流路径模型

一个行为要被策略学会，需要满足完整的三层路径：

```
目标行为（如：起跳 vz=3.96 m/s）
    ↓
策略需要"见过"该状态（经验覆盖）
    → RSI 注入或自然探索达到过该状态？
    ↓
状态发生时有正向奖励信号（信号强度）
    → 奖励函数在该状态下返回正值？
    → 门控条件正确激活？
    ↓
奖励信号不被冲突惩罚抵消（净梯度）
    → 是否有其他奖励项在目标行为时给出大额负值？
    ↓
策略接收到净正梯度 → 行为被强化
```

任何一层断链都会导致策略无法学会目标行为。

---

## 2. 门控条件排查

门控条件是最常见的断链位置。排查步骤：

**Step A：枚举所有激活条件**

```python
# 以 jump_takeoff_drive 为例
active = jump_flag & stage_grounded & (vz_w > 0)
# 三个条件：jump_flag、stage_grounded（参考阶段）、方向信号（vz_w > 0）
```

**Step B：逐条件验证**

对每个条件，问：
- 在目标行为发生时，这个条件为 True 的概率是多少？
- 是否存在边界情况导致条件异常为 False？

**Step C：检查 Python bool vs Tensor 混用**（高频 bug）

```python
# 危险模式
cond = isinstance(x, Type) and tensor_expr  # 返回 bool 或 Tensor，不确定
if isinstance(cond, torch.Tensor):          # 看似安全的检查
    mask = cond
else:
    mask = torch.ones(...)  # 实际上经常走这个分支，门控失效

# 安全模式
if isinstance(x, Type):
    mask = tensor_expr      # 确定是 Tensor
else:
    mask = torch.ones(...)
```

**Step D：RSI 期间的门控**

跳跃奖励和 EFGCL 必须使用参考轨迹阶段 `jump_stage` 门控。不要用真实接触力做
奖励阶段来源；随机 RSI 初始化到空中帧时，参考阶段应立即打开空中奖励。

检查：RSI 随机帧初始化后，`jump_stage`、`jump_phase` 和奖励激活是否与参考帧一致？

---

## 3. RSI 与课程配合

**RSI 的作用**：让策略在 reset 时直接体验目标状态（空中飞行），解决地面-空中的梯度断链。

**配合规则**：
- RSI 速度范围必须覆盖到课程当前最高目标高度
- 课程扩大时，RSI 速度上限必须同步更新

**检查方法**：
```python
# 在 events.py 中确认 vz_max 计算方式
# 正确：动态从课程读取
h_max = env.command_manager.get_term("velocity_height").cfg.jump_height_range[1]
vz_max = sqrt(2 * 9.81 * h_max)

# 错误：固定常量
vz_max = 2.8  # 课程扩到 0.8m 后，RSI 仍只覆盖到 0.4m
```

**课程速度**：`jump_height_curriculum` 只更新 `term.cfg.jump_height_range`，不会自动更新 RSI 速度。两个模块默认完全解耦。

---

## 4. 奖励冲突识别

目标行为发生时，是否有其他奖励项给出大额负值？

**常见冲突模式**：

| 冲突 | 症状 | 修复 |
|---|---|---|
| `tracking_lin_vel` 内含 `vz²` 惩罚 | 起跳时 vz 大 → 被惩罚 | `jump_flag=1` 时用 `tracking_lin_vel_no_jump` |
| `stand_still` 惩罚关节偏离 default | 蹬腿时关节大幅偏离 | `jump_flag=1` 时用 `stand_still_no_jump` |
| `leg_torques` 惩罚峰值力矩 | 起跳瞬间高力矩被惩罚 | `jump_flag=1` 时用 `leg_torques_no_jump` |
| `dof_pos_limits` 在蹬腿末端触发 | 关节行程受限 | 检查软限位是否设置合理 |

**识别方法**：在目标行为发生时（如起跳瞬间），打印各奖励项的值，找出净梯度为负的来源。

---

## 5. 诊断指标设计原则

**在任何修改之前，先问：我有没有足够精准的指标来验证这个假设？**

好的诊断指标需要满足：
1. **可区分**：能把"A 导致"和"B 导致"分开，而不是两者叠加的总和
2. **可定位**：能指向具体的代码路径（某类 env、某个 stage、某个时间窗口）
3. **验证闭环**：修改前后对比该指标，能直接判断修改是否生效

反例：`leg_contact` 终止率是所有 env 的总和，无法区分来自 jump_flag=1 还是 0，来自 RSI 注入后的前几步还是 landing 阶段。仅凭这个指标无法定位根因，也无法验证修复是否有效。

**正确做法：在怀疑某个根因时，先加一个最小诊断指标，直接测量假设涉及的那个量**，而不是根据总量指标做推断。

常见的精细化拆分维度：
- 按 `jump_flag` 拆分：`leg_contact_jump` vs `leg_contact_walk`
- 按 episode 时间窗口拆分：前 N 步（RSI 注入期）vs 中期 vs landing 期
- 按 stage 拆分：grounded / airborne / landing 各阶段的异常率

## 6. 诊断工具：wandb 指标解读

### `diag_*` 系列（scale-independent，优先看）

| 指标 | 含义 | 健康值 |
|---|---|---|
| `Jump/diag_max_airborne_vz` | 本 iter 最大空中 vz | 持续上升趋势 |
| `Jump/diag_mean_airborne_vz` | 平均空中 vz | 持续上升 |
| `Jump/diag_jump_success_rate` | vz>1.0 m/s 的空中 env 比例 | iter1000 后 >10% |
| `Jump/diag_tilt_airborne_deg` | 空中姿态倾斜角 | <15°，不应持续上升 |
| `Jump/diag_active_takeoffs` | 策略主动蹬腿次数（RSI 窗口后） | 应 >0，表示策略在自主学习 |
| `Jump/diag_complete_jumps` | 完整跳跃流程次数 | 应与 active_takeoffs 趋势一致 |

**`active_takeoffs << complete_jumps`**：说明大多数跳跃来自 RSI 而非策略主动，地面期梯度可能断链。

### 课程指标

| 指标 | 含义 |
|---|---|
| `Curriculum/jump_height/jump_height_hi` | 当前课程目标高度上限 |
| `Curriculum/jump_prob/jump_prob` | 当前跳跃触发概率 |

**`jump_height_hi` 上升但 `diag_max_airborne_vz` 不跟**：RSI-课程断链，检查 RSI vz_max 是否同步。

### 终止指标（风险信号）

| 指标 | 触发条件 | 阈值 |
|---|---|---|
| `knee_hyperextension` | 关节速度超过阈值触发 | <15%，否则检查力矩上限 |
| `bad_orientation` | 机身倾斜超过限制 | <5%，否则检查落地稳定性 |
| `leg_contact` | 腿部 body 接触地面 | 应趋近 0；若偏高先用精细拆分指标定位来源 |

### episode 级终止率 vs 步级传感器比率的解读

**这是一个常见的误读陷阱。**

`Episode_Termination/leg_contact = 30%` 意味着本 iter 内 30% 的 episode 以 leg_contact 结束。

`Jump/diag_leg_contact_landing = 0.0002` 意味着每步中仅 0.02% 的 env 在 landing 阶段有腿部接触。

两者差了 3 个数量级，但并不矛盾：`leg_contact` 终止是**瞬发**的——触地的那一步就结束 episode。所以一个 episode 以 leg_contact 终止，只会在极少的步数里贡献接触信号，但整个 episode 计入终止统计。

**解读原则**：
- `Episode_Termination/leg_contact` 高 → 说明频繁触发，但不知道在哪个阶段
- `diag_leg_contact_*` 精细指标 → 说明触地发生在哪个阶段（rsi/landing/walk）
- 两者结合才能定位根因

---

## 8. 修复验证的时序陷阱

### 权重修改后的策略热身期

修改奖励权重后，策略需要若干 iter 才能对新权重做出充分响应（PPO 每步只更新少量梯度）。

**前 5 iter 的数据只能验证"方向"，不能判断"效果"：**
- ✓ 奖励表中新权重已正确出现（确认代码在跑）
- ✓ 目标指标有初步的方向性变化（如 tilt_deg 出现更低的值）
- ✗ 不能因为前 5 iter 改善不明显就断定修改无效
- ✗ 不能因为前 5 iter 数据波动就断定修改有害

**正确的验证时间窗口**（基于本项目参数：1024 envs、decimation=5、GRU 策略）：
- 前 5 iter：确认代码正确运行，方向没有明显错误（奖励表权重是否更新）
- **150-200 iter：判断修改是否真正改变均衡点**（核心判断窗口）
  - RL 策略权重调整后会先探索（振荡增大），再收敛到新均衡，这个过程通常需要 100-200 iter
  - 50 iter 往往处于探索峰期（振荡最剧烈），用 50 iter 判断容易误判为"修改无效"
  - 1 iter = 1024 × 64 = 65536 个 policy step，GRU 梯度传播慢，需要更长热身
- 500+ iter：判断长期稳定性和是否突破能力天花板

### 局部最优的识别

`diag_max_airborne_vz` 振荡不爬升（长期停在某个值附近来回波动）是典型的局部最优信号。

**常见成因**：高性能行为（高速跳跃）遭遇负反馈（如 landing leg_contact 终止），策略学到"高速危险"，在探索高跳和规避惩罚之间来回摆动。

**识别方法**：同时看 `diag_max_airborne_vz` 和 `Episode_Termination/leg_contact`。如果两者都在振荡且相关（max_vz 高时 leg_contact 也高），说明是高跳触发惩罚导致的局部最优，需要修复惩罚来源而不是继续调整起跳奖励。
| `leg_contact` | 腿部碰地 | 应趋近 0 |
