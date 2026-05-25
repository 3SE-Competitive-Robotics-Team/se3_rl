# 症状 → 假设映射

本文件收录已在本项目中验证过的「症状 → 根因」案例，作为新问题诊断的参照。

## 目录

1. 行走高度偏低
2. 跳跃高度天花板
3. 地面期梯度断链
4. 课程推进但性能不跟
5. 惩罚项无法消除
6. 奖励门控失效（Python bool vs Tensor bug）
7. diag_max_airborne_vz 停滞 + 训练中 leg_contact 虚高
8. 跳跃高度主要来自收腿而非机身飞起
5. 惩罚项无法消除
6. 奖励门控失效（Python bool vs Tensor bug）
7. [待验证] diag_max_airborne_vz 停滞 + 行走 env 跪地
5. 惩罚项无法消除
6. 奖励门控失效（Python bool vs Tensor bug）

---

## 1. 行走高度偏低

**症状**
- `base_h` 平衡点 ~0.26m，目标 0.28m
- `tracking_height` 奖励收敛但高度系统性偏低

**根因**
弹簧 stiffness 降低后，原 `default_dof_pos`（按旧弹簧标定）对应的机身高度下移。策略已收敛到新的力平衡点，不是训练问题。

**诊断证据**
- 平衡点偏差量（2cm）与弹簧变软后膝关节静态下沉量相符
- `stand_still` 的 `default_height` 和 `height_range` 中值均为旧值 0.27m，策略被错误地引导向错误目标

**修复**
接受新平衡点，把 `height_range` 中值和 `stand_still.default_height` 改为 0.26m。无需重训。

**文件**
`src/se3_train/mdp/commands.py:26`，`src/se3_train/env_cfg.py:246,581`

---

## 2. 跳跃高度天花板

**症状**
- 轮子实际离地最高 6.6cm，目标 0.8m
- `diag_max_airborne_vz` 长期停在 ~1.4 m/s，不上升
- `jump_vel_z_tracking` 有大量惩罚但无法消除

**根因（叠加性，需同时修复）**

| 根因 | 证据 | 影响量级 |
|---|---|---|
| RSI vz_max=2.8 m/s 固定，覆盖上限 0.40m，课程已要求 0.8m | `events.py:72` | 大 |
| rated_torque=20 N·m，全行程可用功 52J < 0.8m 所需 99J | `robot.py:97-99` | 大 |
| `jump_takeoff_drive` stage 门控 bug，地面期梯度实际失效 | `jump_rewards.py:113` | 中 |
| 课程扩到 0.8m 后 tolerance=0.2 产生不可消除惩罚 | `env_cfg.py:716` | 中 |
| expand_iter=1000 太短，策略没学会 0.5m 就被要求 0.8m | `env_cfg.py:785` | 中 |

**关键思路**：先算物理上限再看奖励设计。52J < 99J 意味着不管奖励怎么设计，用 rated_torque 就跳不到 0.8m。

---

## 3. 地面期梯度断链

**症状**
- `diag_active_takeoffs` 长期为 0（策略不主动蹬腿）
- 所有跳跃都来自 RSI 注入，`diag_complete_jumps >> diag_active_takeoffs`
- `jump_takeoff_drive` 奖励项有值，但策略无响应

**根因**：Python bool 与 Tensor 的 `and` 运算返回 bool，导致 stage 门控对所有 env 统一失效。

```python
# 错误写法（常见陷阱）
on_ground = isinstance(term, JumpCommandTerm) and (term.jump_stage == 0)
# isinstance(...) 返回 True（Python bool）
# True and Tensor → 返回 Tensor（看起来正确，但后续 isinstance(on_ground, Tensor) 为 True）
# 问题：term.jump_stage == 0 是全局比较？不，实际上 and 会短路
# 真正的问题：这行得到的是 Tensor，但 if isinstance(on_ground, Tensor) 分支
# 对所有 env 用同一个 Tensor（不是 per-env bool），当 RSI 注入时整批 env 同时被错误激活

# 正确写法
if isinstance(term, JumpCommandTerm):
    stage_grounded = term.jump_stage == 0  # Tensor[bool]，per-env
else:
    stage_grounded = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
```

**文件**：`jump_rewards.py:113`（`jump_takeoff_drive`），`jump_rewards.py:164`（`jump_squat_drive`）

---

## 4. 课程推进但性能不跟

**症状**
- wandb 显示 `jump_height_hi` 已到 0.8，但 `diag_max_airborne_vz` 没有对应上升
- 策略在课程扩大后性能反而下降或停滞

**根因**：课程和 RSI 解耦，课程扩大目标但 RSI 速度上限固定。

策略被要求达到 3.96 m/s（0.8m），但从未在训练中见过 vz > 2.8 m/s 的空中状态。`jump_vel_z_tracking` 变成纯惩罚项，无正向信号。

**通用模式**：当课程修改目标范围时，所有依赖"目标状态经验"的模块都需要同步更新。

**检查清单**：
- RSI 速度上限是否随课程高度动态计算？
- 惩罚项的容忍带是否随课程扩大而放宽？
- `expand_iter` 是否给了策略足够时间掌握中间阶段？

---

## 5. 惩罚项无法消除

**症状**
- 某惩罚项（如 `jump_vel_z_tracking`）长期有较大负值
- 调大权重让该项下降，但对应能力没有提升

**根因**：目标不可达（物理或经验层面），惩罚成为固定成本而非梯度信号。

**诊断方法**：
1. 计算惩罚项的"零惩罚条件"是什么
2. 该条件在当前物理/经验约束下是否可达
3. 如果不可达 → 先修复约束，再考虑权重

**修复模式**：在根本约束修复前，暂时放宽 tolerance（如 0.2 → 0.5），避免持续高额惩罚压制探索。

---

## 6. 奖励门控失效

**通用 Bug 模式**（Python RL 代码中的高频陷阱）

```python
# 错误：Python bool 参与 Tensor 逻辑运算
flag = some_condition() and tensor_condition  # 类型不定
active = jump_flag & grounded & flag  # flag 类型错误时整行 active 错误

# 正确：分离 Python bool 检查和 Tensor 比较
if isinstance(term, ExpectedType):
    stage_mask = term.stage == 0          # Tensor[bool]
else:
    stage_mask = torch.ones(n, dtype=torch.bool, device=dev)
active = jump_flag & grounded & stage_mask  # 全部 Tensor，安全
```

---

## 7. diag_max_airborne_vz 停滞 + 训练中 leg_contact 虚高

**症状组合**
- `diag_max_airborne_vz` 长期在某个值附近振荡，无上升趋势
- `leg_contact` 终止率超过 `jump_flag_ratio`（即行走 env 也在触发）
- `diag_active_takeoffs` 平台，策略不继续进步

**根因假设模式**

训练中 `leg_contact` 高不等于基模有问题。需要先用 sim2sim（单 env，无并行噪声）验证基模行走是否真的跪地。

**诊断路径**
1. sim2sim 跑基模 500 步（`--viewer none`），观察 `base_h` 趋势和是否触发 `done_reason=leg_contact`
2. sim2sim 跑当前 Jump 模型同等步数做对比
3. 若 sim2sim 中均无 `leg_contact`：训练中的高值来自 RSI 高速注入后的着陆冲击或姿态失控，不是基模问题
4. 若 sim2sim 中基模有 `leg_contact`：基模确实有行走退化，必须先修基模再训跳跃

**`diag_max_airborne_vz` 停滞的额外检查**

当前跳跃诊断按参考轨迹 `jump_stage==1` 统计空中段，不再依赖真实接触阶段。若
`diag_max_airborne_vz` 仍停滞，优先检查参考轨迹高度范围、RSI 随机帧覆盖、起跳奖励是否在
`jump_stage==0/1` 正确激活，以及 `leg_contact` 是否只作为日志而非 done 信号。

**规则：基模 sim2sim 验证是跳跃调优的前置条件，发现 leg_contact 虚高时必须先做这一步。**

**已验证的精细拆分结论：**
- `diag_leg_contact_walk = 0`：行走 env 腿部触地为零，基模行走正常
- `diag_leg_contact_jump > 0`：leg_contact 完全来自 jump_flag=1 的 env
- `diag_leg_contact_rsi = 0`：RSI 注入期（前几步）没有腿部触地，RSI 姿态本身不是根因
- `diag_leg_contact_landing > 0`：**landing 阶段有腿部触地**，根因是着陆冲击时机身倾斜导致腿先着地

**真正根因**：空中姿态控制问题（`diag_tilt_airborne_deg` 偏高），着陆时机身有倾斜角，导致轮子未能先着地，腿部触地触发终止。解决方向：改善空中姿态奖励，而不是调整 RSI 收腿角度或域随机化范围。

---

## 8. 跳跃高度主要来自收腿而非机身飞起

**症状**
- `wheel_clr` 峰值只有 0.07-0.09m，但 `base_h` 峰值达 0.41m
- 两者差值约 0.32m（机身几何高度），说明 base_h 的提升几乎全靠收腿
- sim2sim 中腿部在离地后立刻快速收起，机身没有明显的飞起弧线
- `diag_max_airborne_vz` 训练中显示 2.5 m/s，但 sim2sim 中机身实际飞行高度远低于此

**根因：奖励设计与正确的起跳物理时序不符**

正确时序：**机身先获得足够 vz，腿才开始收起**。当前奖励在两处破坏了这个时序：

1. **蹬地阶段（grounded + vz>0）缺乏"腿保持伸展"约束**
   - `jump_takeoff_drive` 只奖励 vz>0，但不限制腿的动作
   - 策略可以在 vz 刚变正时就立刻收腿，导致地面反力作用时间缩短
   - 地面接触期越短，给机身的冲量越小

2. **`jump_knee_phase` 在 stage=1 且 vz≥0 时立即奖励收腿，时机过早**
   - 刚离地时机身 vz 可能还很小，过早收腿消耗腿部动量效率低
   - 应等 vz 超过阈值（如 1.0 m/s）后再奖励收腿

**物理原理**（见 `docs/background.md` 约束 4）：
- 蹬地期：腿伸展 → 地面反力 → 机身获得冲量（F·t = m·Δv）
- 飞行期：收腿 → 腿部质量上移 → 通过内力进一步提升机身质心
- 两个阶段都重要，但蹬地期是能量来源，收腿期只是动量分配

**诊断方法**：在 sim2sim 中打印 `dof_pos`，观察离地后多少步内膝关节开始屈曲。若离地 3-5 步内就开始屈曲，说明收腿过早。

**修复方向**：
1. 蹬地阶段（grounded + vz>0 + 接触力大）：惩罚膝关节屈曲（`jump_squat_drive` 反过来用）
2. `jump_knee_phase` 加 vz_threshold 门控：vz < threshold 时不奖励收腿
3. 地面期核心驱动信号改为地面反力法向分量（比 vz 更直接反映起跳效果）





**诊断信号**：门控奖励项有值（不为零），但调权重无效，`active.sum()` 打印出来与预期不符。
