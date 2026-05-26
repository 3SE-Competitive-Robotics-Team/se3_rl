---
name: rl-tuning
description: 强化学习训练调优方法论。当出现以下情况时使用：训练不收敛、奖励停滞或下降、策略行为不符合预期（如跳不高、走路抖动、高度偏低）、发现奖励函数 bug、需要分析梯度流断链、需要设计或修改课程、需要确定修改方向、需要判断修改是否生效。覆盖从症状到根因的完整诊断链，以及验证修复效果的方法。
---

# RL 训练调优方法论

## ⚠️ 训练监控规则（每次训练必须执行）

**启动训练后必须立即开始 5 分钟轮询，直到用户明确说停止。**

轮询命令：
```bash
sleep 300 && ssh wuyinyun "grep -E 'diag_max_airborne_vz|diag_tilt_airborne|diag_jump_success|diag_active_takeoffs|diag_leg_contact_landing|Episode_Termination/leg_contact|Episode_Termination/bad_orient|Iteration time|ETA' /tmp/train.log | tail -30"
```

每轮输出后必须先向用户汇报，再进入下一轮 sleep 300。汇报格式固定为：

1. 一张 Markdown 表格，至少包含：训练进度、ETA、GPU、`jump_flag_ratio`、`diag_jump_success_rate`、`diag_max_airborne_vz`、`diag_tilt_airborne_deg`、`diag_active_takeoffs`、`diag_leg_contact_termination`、`bad_orientation`。
2. 表格后用 2-4 条短句写清楚洞察：是否继续跑、主要风险、与上一轮相比的变化。
3. 汇报后立即启动下一轮 5 分钟等待，除非用户明确要求停止。

**修改生效验证（前 5 iter 必做）**：修改任何奖励权重或代码后，必须在前 5 iter 确认：
1. 奖励表中目标奖励项的权重是否已更新（确认新代码在跑）
2. 目标指标方向是否正确（不需要已收敛，只需方向对）
3. 无新增崩溃终止

**如果前 5 iter 方向不对，立刻停训练重新诊断，不要让错误的修改跑 3 小时。**

---

## 核心原则

**在陈述根因之前不动代码。** 根因必须满足：
1. 指向具体文件:行号或公式
2. 能解释所有已观测症状（不只是其中一个）
3. 能预测一个尚未观测到的现象

"奖励不收敛"不是根因。"`jump_takeoff_drive` 在 `jump_rewards.py:113` 的 `isinstance and Tensor` 写法导致 stage 门控对所有 env 失效，地面期梯度从未正确激活" 才是根因。

---

## 诊断流程（五步）

### Step 1 — 列出所有症状

不只记录用户报告的第一个症状，要穷举：
- 哪些指标异常？（值域、趋势、与预期的偏差量）
- 哪些指标正常？（同样重要，用于排除假设）
- 症状是持续的还是间歇的？
- 是从一开始就有，还是训练到某阶段才出现？

### Step 2 — 建立物理/数学上限

**在分析奖励设计之前，先算物理极限。**
若物理上就不可能达到目标，改奖励权重没有意义。

- 机器人总质量 × 目标速度² / 2 = 所需动能
- 电机额定力矩 × 关节行程 × 关节数 = 最大可用功
- RSI 注入速度上限 → 对应最大可体验高度
- 关节限位 → 最大蹬腿行程

参考：`references/physical-constraints.md`

### Step 3 — 追踪梯度流

从目标行为反向追踪：策略要学会这个行为，需要哪条奖励信号路径？路径上每个节点的激活条件是什么？

重点检查：
- 门控条件（`jump_flag`、`stage`、`grounded`）是否正确激活？
- 奖励在目标行为发生时是否为正？
- RSI 注入的经验是否覆盖目标状态空间？
- 课程扩大目标时，其他模块（RSI、惩罚容忍带）是否同步？

参考：`references/gradient-flow.md`

### Step 4 — 建立假设并验证

每个假设需要：
1. **预测**：如果假设正确，哪个 wandb 指标会有什么表现？
2. **反预测**：如果假设错误，哪个指标能证伪？
3. **最小证据**：能用一行代码或一个指标确认的最小验证

假设质量门控：假设必须能解释 Step 1 列出的**所有**症状。只能解释部分症状的是"症状级猜测"，不是根因。

### Step 5 — 修复与验证闭环

修复后必须：
1. `SE3_SMOKE=1` 验证环境不崩溃
2. 启动训练，在前 200 iter 内确认目标指标方向正确
3. 对比修复前后的同一指标，量化改善

---

## 快速导航

| 症状 | 首先读 |
|---|---|
| 某个能力天花板明显（跳不高、走不快） | `references/physical-constraints.md` |
| 奖励激活但策略不学 | `references/gradient-flow.md` → 梯度断链排查 |
| 奖励函数输出异常值 | `references/reward-design.md` → 常见 bug 模式 |
| 症状对不上假设 | `references/symptom-to-hypothesis.md` |
| 课程推进但性能没跟上 | `references/gradient-flow.md` → 课程-RSI 配合 |

---

## 修改方向决策树

```
症状：某能力达不到目标
│
├─ 先问：物理上可能吗？
│   ├─ 不可能 → 修改物理参数（力矩上限、关节限位）或降低目标
│   └─ 可能 → 继续往下
│
├─ 策略有没有见过目标状态？（RSI / 课程覆盖）
│   ├─ 没见过 → 修复 RSI 速度上限或课程范围
│   └─ 见过 → 继续往下
│
├─ 目标行为发生时奖励为正吗？
│   ├─ 为零或为负 → 检查门控条件 bug、奖励方向、权重符号
│   └─ 为正 → 继续往下
│
├─ 有没有与目标行为冲突的惩罚？
│   ├─ 有 → 添加 jump_flag gate 豁免冲突惩罚
│   └─ 没有 → 继续往下
│
└─ 考虑：权重比例、PD 增益、域随机化范围
```

---

## 验证修复是否生效

### 即时验证（前 5 iter）
- 目标奖励项方向正确（正向奖励为正，惩罚为负）
- 无新增崩溃终止（`bad_orientation`、`knee_hyperextension`）

### 短期验证（100-500 iter）
- `diag_*` 诊断指标有明显上升趋势
- `diag_active_takeoffs > 0`（策略在主动学习，不只靠 RSI）

### 中期验证（1000+ iter）
- 性能突破之前的天花板
- 课程推进时性能同步提升（不是脱钩）

---

## 本项目特定参考

- 机器人参数、默认姿态、PD 增益：`src/se3_shared/robot.py`
- 训练环境配置：`src/se3_train/env_cfg.py`
- 跳跃奖励函数：`src/se3_train/mdp/jump_rewards.py`
- 跳跃状态机：`src/se3_train/mdp/jump_commands.py`
- RSI 实现：`src/se3_train/mdp/events.py`
- 课程调度器：`src/se3_train/mdp/jump_curriculums.py`
- wandb 关键指标：`Jump/diag_*`（scale-independent，优先看这组）
