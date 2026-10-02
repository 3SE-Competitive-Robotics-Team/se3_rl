# J10：J9 用一维相位代替 20 维参考帧（2026-10-02，用户定）

## 动机

J8 证明在"按时间模仿 + 偏离终止"下网络必须有时间信息（去掉参考帧后学不出）。用户提出把时间信息编码进 jump_commands
已有的第三维 jump_phase：一维、在 34 维契约内。参考是固定四条，(目标高度, 相位) 即确定参考当前状态，与 20 维参考帧在时间
信息上等价；网络需要自己记住"相位 → 参考"，换参考要重训。成立的话 rough 合入可保持 34 维、从 M54 热启动不改输入层。

## 改动（单变量：时间信息的给法）

`SE3-WheelLegged-Jump-Mimic-Exp-J10` = J9 + `reference_obs=False` + `phase_time_scale_s=1.5`：

- actor 与 critic 都去掉参考帧与参考时钟 / 编号（同 J8），actor 34 维。
- 跳跃中 jump_phase = 触发后参考时刻 / 1.5 s（固定常数而非各参考时长归一化，同一相位值对应同一物理时刻；参考最长 1.46 s，
  相位 0–0.97），不在跳时为 0。部署包络 jump_phase 放开到 (0, 0.973)。
- 其余全部照 J9：无下蹲参考（站姿 0.22 m）、无 RSI、四项模仿奖励、偏离终止 0.12 m、vx ±1.5、执行链与 PPO。

## 部署（se3-sim2x）

metadata 顶层 `jump_reference` 新增可选 `phase_time_scale_s`；runtime 播放器有此字段时 jump_phase = 参考时刻 / 该常数，
否则恒 0（旧 artifact 不变）。

顺带修正一处训练端 / runtime 的末尾差一步：参考播完判定 `t ≥ 时长` 在训练端 float32 累加 73 × 0.02 = 1.4599999 < 1.46，
比 runtime（float64）多播一步（J7 / J9 的 jump_flag 末尾同样差一步）；两端都改为留 1e-6 容差。

## 验证

- J9 / J10 配置对比：只差观测（参考帧 → 一维相位）与部署包络 jump_phase。
- 相位时序核对（`.scratch/j10_phase_check.py`，关掉偏离终止只看时序）：训练端观测里的 jump_phase 与 runtime 播放器逐步差 0，
  两端都在 0.96 之后同一步归 0。
- se3-sim2x 全部 unittest、`tests.test_onnx_metadata` + `tests.test_flat_baseline` 通过；CPU smoke（1 env、5 轮）通过。
