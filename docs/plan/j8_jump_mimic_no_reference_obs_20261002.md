# J8：J7 去掉全部参考观测（POMDP）（2026-10-02，用户定）

## 动机

用户想看 J7 的跳跃效果能否在没有参考帧的情况下学出来：部署契约回到 34 维（与 rough 同布局），不需要参考帧观测。

## 改动（单变量：观测）

`SE3-WheelLegged-Jump-Mimic-Exp-J8` = J7 + `env_cfg(reference_obs=False)`：

- actor 去掉 20 维参考帧，只剩 34 维本体（跳跃信息只有 jump_flag / 目标高度，jump_phase 恒 0）。
- critic 也去掉参考帧与参考时钟 / 参考编号（用户定：critic 也不看），只从状态估值。
- 其余全部照 J7：无下蹲参考（站姿 0.22 m）、四项模仿奖励按参考时刻计算、偏离终止 0.12 m、RSI 50%、jump_flag 在整段参考（约 1.4 s）为 1、vx ±1.5、执行链与 PPO。

这是 POMDP：奖励按参考时间算，actor 与 critic 都看不到时间。起跳段从 flag 上升沿开始单调上升，状态与时刻基本一一对应；
落地后的缓冲 / 恢复与参考末尾的站立段（flag 仍为 1）状态相似、时刻不同，可能出现时机不准、模仿奖励偏低或二次起跳。

## runtime（se3-sim2x）

`policy_descriptor` / `policy_runtime`：metadata 带顶层 `jump_reference` 但观测没有该项时不再拒绝加载；跳跃播放器照样按参考时长驱动
`velocity_height[5:8]`（jump_flag / 目标高度），只是不生成参考帧观测。J8 ONNX 为 34 维、`supports_jump=True`；J7 54 维 ONNX 照常。

## 验证

- J7 / J8 配置对比：只差观测（J8 actor 与 critic 的跳跃相关项只剩 jump_commands）。
- se3-sim2x 全部 unittest、`tests.test_onnx_metadata` + `tests.test_flat_baseline` 通过；CPU smoke（1 env、5 轮）通过，导出 ONNX 被 runtime
  加载为 34 维且可触发跳跃。
