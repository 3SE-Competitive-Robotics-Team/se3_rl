# EFGCL 论文方法论要点

> 论文：Yoneda et al., "EFGCL: Learning Dynamic Motion through Spotting-Inspired External Force Guided Curriculum Learning", RA-L 2026.
> arXiv: 2605.10063

## 问题

足式机器人学习跳跃、空翻等高动态动作时，失败风险极高，导致有效探索困难——PPO 的 advantage 估计在失败主导的轨迹中趋近于零，梯度更新停滞。现有方案（参考轨迹模仿、reward shaping）要么依赖高质量参考数据，要么引入 designer bias 排除潜在最优解。

## 核心思想

受体操 spotting（教练扶运动员完成动作）启发，EFGCL 通过在训练早期施加外部辅助力，让策略**先物理体验成功状态**，再通过 curriculum 逐步撤掉辅助。论文的论证链是：

1. 策略一旦经历过足够多的高奖励轨迹，价值函数会对这些轨迹对应的状态赋予高估值
2. 高估值使偏离成功轨迹的动作产生大的负 TD error 和负 advantage，PPO 会自然保留这些有效动作序列
3. 因此关键不是教策略"正确动作"，而是让它**在早期经历足够多的高奖励状态**——外力只是状态分布扩展工具

## 方法

### 外力设计（三个要素）

外力由三个要素定义：**施力点** \(P\)、**力向量** \(F\)、**施加时序** \(T\)。论文强调外力是启发式设计的，不需要精确调参：

- Jump 任务：4 个 scapula link，竖直向上力，时间窗 (1.0s, 1.1s)
- 力大小由抛体模型前馈计算：

\[
f_{\text{jump}} = \frac{mg}{2} \left(1 + \sqrt{1 + \frac{8h_{\text{target}}}{g \cdot \Delta t^2}}\right)
\]

其中 \(\Delta t = 0.1\) s 为外力持续时长。四个施力点各承担 \(f/4\)。

### 成功率驱动的自适应 Curriculum

```
FOR i = 0, 1, 2, ...:
    在 α_i × F_assist 下训练 PPO，直到 success_rate > ζ (=0.6)
    α_{i+1} ← max(0, 1 - ε × (i+1)), ε = 0.01
    继承当前策略和价值函数进入下一阶段
```

关键设计：**每个阶段必须成功率达标后才衰减**，而不是按固定 iter 衰减。这样 curriculum 步长自动适应训练进度，保证相邻 MDP 之间的策略比 \(\pi_{i+1}/\pi_i \approx 1\)，符合 PPO 的小步更新假设。

Jump 成功率定义：\(|h_t^{\max} - h^{\text{target}}| < 0.1 \land |h_t| < 0.1\)（达到目标高度且已经落地）。

### 时间编码观测

外力按时序施加，策略需要感知时间。论文引入有界单调函数作为观测：

\[
\tau(t, \lambda) = \frac{\tilde{t}^3}{1 + \tilde{t}^3}, \quad \tilde{t} = \frac{t}{\lambda}
\]

\(\lambda\) 设为外力激活起始时间 \(t^{\text{start}} = 1.0\) s。这避免直接用 \(t\) 导致的 scale mismatch。

### 奖励函数：刻意稀疏

论文有意不引入任何中间动作引导奖励。所有任务共享同一套奖励结构：

\[
r_t = \rho_t^{\text{task}} + \rho_t^{\text{task}} \cdot \rho_t^{\text{stand}} + \lambda_\omega r_t^{\text{ang}} + r_t^{\text{common}}
\]

- \(\rho_t^{\text{task}}\)：任务完成度（Jump 为达到的高度与目标的 exp 距离）
- \(\rho_t^{\text{stand}}\)：落地后站立姿态奖励（只在任务完成后生效）
- \(r_t^{\text{ang}}\)：非目标轴角速度惩罚
- \(r_t^{\text{common}}\)：碰撞、终止、关节速度/加速度惩罚

三个任务（Jump / Backflip / Lateral-flip）只替换 \(\rho_t^{\text{task}}\) 中的 target variable，奖励权重和函数形式完全一致。论文明确指出这是刻意为之，避免 task-specific reward engineering 带来的不可比性。

### Teacher–Student 架构

Teacher 策略使用完整状态（含特权观测如 base 高度），通过 PPO + EFGCL 训练。Student 策略仅使用本体感知观测，通过监督学习（action matching + privileged obs reconstruction）从 Teacher 蒸馏。EFGCL 只在 Teacher 训练阶段生效。

## 关键实验结果

- **Jump 收敛速度约 2 倍**于 baseline（无 EFGCL）
- **Backflip 和 Lateral-flip**：baseline 完全学不会，EFGCL 能稳定学到并成功 sim-to-real
- **Ablation**：外力参数（施力点、大小、时序）在很宽范围内都鲁棒，只要外力"大致帮助完成动作"。仅在极端情况（力太小 100N / 太大 250N / 时序太短 0.05s）失败
- **价值函数加速**：EFGCL 在 200 iter 时价值估计已接近最终分布，baseline 需要 1000+ iter

## 与本仓库实现的关键差异

| 维度 | 论文 | 当前实现 |
|---|---|---|
| 奖励 | 刻意稀疏，无中间引导 | 大量密集奖励（takeoff_drive, vz_tracking, height_success 等） |
| 参考轨迹 | 明确不使用 | 使用跳跃参考轨迹库 |
| 空中辅助 | 无 | 额外万向弹簧力矩 |
| Curriculum | 阶段式（成功率达标后跳变衰减） | 连续式（每 iter EMA 衰减 0.005） |
| 时间编码 | τ(t) 观测 | 无 |
| 架构 | Teacher–Student 蒸馏 | asymmetric actor-critic |
| 外力窗口 | 仅论文固定时间窗 | 同样使用固定时间窗 |

当前方案更接近"EFGCL 起跳外力 + 传统 dense reward tracking"的混合，而非论文的纯物理引导范式。
