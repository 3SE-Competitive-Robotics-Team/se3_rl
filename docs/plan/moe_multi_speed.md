# MoE 多速度域方案

> 状态：构想阶段，尚未验证单策略容量瓶颈

## 动机

baseline 模型训到 lin_vel_x ±4.5 / ang_vel_yaw ±3.0 后，需要分别向高线速度（±8.0）和高角速度（±6.0）特化。若单策略网络容量不足以覆盖全速域，可用 MoE 结构让不同 expert 负责不同速度区间。

## 结构

```
obs (27d) → 共享 encoder (512) → router (小 MLP，输出 N 个 expert 权重)
                                 → N 个 expert head (256→128→6)
                                 → 加权求和 → action (6d)
```

N=3：expert-0 (baseline)、expert-1 (high-lin)、expert-2 (high-ang)。

Router 输入包含 commands（速度指令），使其能根据目标速度选择 expert。

## 训练流程

1. **阶段一（baseline）**：螺旋课程训 4000 iter，所有 expert 一起训练，router 学会均匀分配
2. **阶段二（后训练-高线速度）**：从 baseline checkpoint 加载，lin 推到 ±8.0，加 auxiliary loss 引导 router 偏向 expert-1
3. **阶段三（后训练-高角速度）**：从 baseline checkpoint 加载，ang 推到 ±6.0，加 auxiliary loss 引导 router 偏向 expert-2

辅助 loss：指令区间 × expert 概率的交叉熵 + load balancing loss 防 expert collapse。

## 部署

单模型，无需运行时切换。Router 根据实时指令自动 soft-mix experts，过渡平滑无抖动。

## 风险与备选

- MoE 在 RL locomotion 领域**不算主流**做法，调 router + balance loss 工程成本高
- 主流方案：分层策略（AMP/ASE）或加宽单策略网络
- **建议**：先验证单策略（加大网络 512→1024）+ 完整课程能否一次训到 ±8.0，容量确实不够再上 MoE
