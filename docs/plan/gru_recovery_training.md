# GRU 反倒起身训练方案

> 状态：方案阶段，尚未实现。目标是让同一个策略同时保持正常轮腿运动能力，并学会倒地后的自主起身恢复。

## 动机

倒地起身是强时序、强接触、部分可观测任务。单帧 MLP 可以学正常行走，但恢复动作需要记住最近的接触变化、姿态变化、动作历史和身体是否正在变好。GRU 能给策略一个轻量的隐状态，适合在不引入复杂 MoE 或多策略切换的情况下增强恢复能力。

调研中没有在 `unitree_rl_mjlab` 看到 GRU/LSTM 训练配置。MJLab 当前依赖的 RSL-RL 已内置 `RNNModel`，可以直接接入。推荐从工程上最简单、部署成本最低的 `GRU + 512 hidden + 1 layer` 开始。

## 总体路线

使用一个 GRU policy 贯穿两阶段训练：

1. **正常运动预训练**：只训练站立、速度追踪、高度追踪和抗扰，得到稳定 sim2sim checkpoint。
2. **反倒起身后训练**：从预训练 checkpoint resume，混入倒地/半倒 reset，学习恢复，同时保留正常 reset 防止遗忘。

关键约束：预训练阶段就固定最终观测维度。后训练只改变 reset 分布、reward 分支和 curriculum，不改变 actor 输入维度。

## 网络配置

推荐使用 RSL-RL 的 `RNNModel`：

```python
actor = RslRlModelCfg(
    class_name="RNNModel",
    rnn_type="gru",
    rnn_hidden_dim=512,
    rnn_num_layers=1,
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.5,
        "std_type": "scalar",
    },
)

critic = RslRlModelCfg(
    class_name="RNNModel",
    rnn_type="gru",
    rnn_hidden_dim=512,
    rnn_num_layers=1,
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
)
```

训练参数建议：

- `num_steps_per_env=64`，给 BPTT 更长的时序窗口。
- 预训练学习率保持当前量级，例如 `6.5e-4`。
- 后训练学习率降到 `2e-4 ~ 3e-4`。
- `desired_kl` 可略收紧到 `0.006 ~ 0.008`，避免后训练冲坏已学 gait。

GRU 优先于 LSTM。GRU hidden state 简单，sim2sim 和导出维护成本低，恢复任务通常已经够用。

## 观测设计

从预训练开始就加入最终需要的字段：

- `base_height`：建议给 actor。起身恢复强依赖高度，不能只靠腿部关节角隐式推断。
- `contact`：至少包含轮子接触、腿部接触 bool 或归一化接触力。
- `mode`：`0=locomotion`，`1=recovery`。
- `base_lin_vel`：优先只给 critic；如果恢复能力不足，再考虑给 actor。

预训练阶段 `mode` 恒为 0，recovery reset 不启用。这样 checkpoint 可以直接 resume 到后训练。

## 阶段一：正常运动预训练

目标：训出正常站立和行走都稳定的 GRU checkpoint。

配置：

- reset 分布保持正常站立。
- 终止条件保持当前偏严格逻辑，继续强化“不摔”。
- action delay、域随机化、push curriculum 照常启用。
- reward 沿用正常 tracking reward。
- `mode=0`。

验收：

- 训练 play 模式稳定站走。
- sim2sim 默认无 GUI 跑 500-1000 步不倒。
- `leg_contact=0`，`bad_orientation=0`。
- `tracking_height`、`tracking_orientation`、`upward` 达到当前 MLP baseline 相近水平。

## 阶段二：反倒起身后训练

从阶段一 GRU checkpoint resume。

reset 混合比例：

| 阶段 | 正常 reset | recovery reset | 说明 |
|---|---:|---:|---|
| 初期 | 80% | 20% | 轻微半倒、半蹲、跪地 |
| 中期 | 70% | 30% | 加入侧倒、前趴、后仰 |
| 收尾 | 85-90% | 10-15% | 巩固正常 locomotion，减少遗忘 |

recovery episode 行为：

- `mode=1`。
- 速度命令置 0。
- 目标先是站起并稳定。
- 起身成功并稳定 `0.5-1.0s` 后，切回 `mode=0`，恢复正常速度/高度/姿态追踪。

## Recovery Reward

正常 mode 沿用当前 reward。

recovery mode 单独增加恢复奖励，并且这些恢复奖励不要乘 `upright_gate`：

- `upright_recovery`：鼓励 `projected_gravity_z` 回到直立。
- `height_recovery`：鼓励 base height 接近目标高度。
- `delta_upright`：奖励直立程度变好。
- `delta_height`：奖励高度变高。
- `low_ang_vel`：奖励角速度下降。
- `leg_clearance`：惩罚膝盖/腿部持续接触地面。
- `recovered_bonus`：直立、高度、低角速度同时满足并保持一段时间后给成功奖励。

核心原则：倒地时也必须有正向梯度。不能让 `upright_gate` 把所有追踪和恢复信号清零。

## 终止条件

拆成两层：

- `recoverable_fall`：倾斜、低高度、半倒、跪地等可恢复状态，不终止。
- `hard_failure`：NaN、严重穿模、翻滚不可恢复、长时间无恢复、极低高度持续太久，才终止。

建议后训练时保留最长恢复时间限制，例如 recovery mode 连续 `3-5s` 没有明显变好就终止，避免 PPO 学会长期躺平刷低惩罚。

## sim2sim 要求

上 GRU 前，sim2sim 必须支持 recurrent checkpoint：

- 识别 `rnn.rnn.weight_ih_l0` 等 GRU 权重。
- 构建 `EmpiricalNormalizer + GRU + MLP` 推理链。
- `PolicyRuntime.reset()` 清空 hidden state。
- robot reset 时同步 reset policy hidden state。
- telemetry 记录 `policy_type=gru`、`rnn_hidden_dim`、`rnn_num_layers`。

验证场景：

- 正常站立/行走。
- 半蹲/跪地恢复。
- 左右侧倒恢复。
- 前趴/后仰恢复。
- 起身成功后切回正常行走。

## 风险与备选

- **遗忘正常 gait**：后训练恢复比例过高或学习率过大。用混合 reset、低学习率、收尾阶段正常 reset 占比提高来控制。
- **恢复奖励被钻空子**：只奖励高度可能学会弹跳或支撑在错误姿态。需要同时约束直立、低角速度和腿部接触。
- **GRU hidden state 部署错误**：sim2sim 和部署端必须正确 reset hidden state，否则同一个 checkpoint 表现会漂。
- **单模型容量不足**：先加宽 GRU hidden 或 MLP head。确认单模型不够后，再考虑 MoE 或 locomotion/recovery 双 expert。

## 推荐执行顺序

1. 给训练端加 GRU runner 配置，但先不改环境语义。
2. 给 sim2sim 补 GRU checkpoint 推理。
3. 训练 GRU 正常运动 baseline，并做 sim2sim 验证。
4. 固定最终观测维度，加入 `mode/base_height/contact`。
5. 从 GRU baseline resume，开启 recovery reset 和 recovery reward。
6. 按比例 curriculum 后训练，定期拉 checkpoint 做四类 sim2sim 恢复验证。
