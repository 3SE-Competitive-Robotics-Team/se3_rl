# M34：M27 去掉两项轮子几何罚的死区

用户于 2026-09-26 指定：以 M27 为基线，同时取消 `wheel_fore_aft_offset` 和
`wheel_height_diff` 的死区，重新从零训练。

任务入口：`SE3-WheelLegged-Rough-Exp-HeightWindowNoWheelDeadzone`。

| 项 | M27 | M34 | 保持不变 |
|---|---|---|---|
| wheel_fore_aft_offset | 10 cm 死区 | 0 | 权重 −40，仅平地列 |
| wheel_height_diff | 8 cm 死区 | 0 | 权重 −40，仅上台阶及二级上行列 |

保留原有门控，罚值分别变为门控后的 −40·Δx² 和 −40·Δz²。
其余沿用 M27，包括窗口高度参考与有界高度罚。旧入口保留原死区以供复现对照。

两项同时改变，只能判断组合效果，不能区分各自贡献。主要检查轮子错位与高度差、
台阶课程、速度跟踪和灾难状态终止，并以确定性 sim2sim 验证动作与通关。
去掉高度死区会对跨台阶时的小幅轮高差也计罚，可能降低爬升能力。

训练设置：用户确认 nulltask1 六卡，每卡 8192 环境、8000 轮、seed 42、每 200 轮保存，
从零训练，W&B 项目 `SE3-WheelLegged-Rough`。每轮样本量为 M27 两卡的三倍，
按相同 iteration 比较存在批量混杂，不能将全部差异归因于死区。

验证：本地 Flat 与新入口 CPU smoke 各 5 轮通过；远端新入口 GPU0、16 环境 smoke
5 轮通过，均成功导出 `model_4.onnx`。Ruff 检查通过，实际配置对比只有两项死区不同。

## 启动记录

- 代码 commit：`1a10b73`，子模块 `e753ce6`。
- 启动时间：2026-09-27 01:25:59（Pod 时区，UTC+8）。
- run：`2026-09-27_01-25-59_rough-M34-heightwindow-nodeadzone-seed42-6x8192-8k`。
- PID/PGID：`582278`。
- state：`/workspace/.se3-training-state/nulltask1/20260926T172552Z-25956`，日志为其下 `train.log`。
- W&B：[acqxnyu8](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/acqxnyu8)。
- 首轮核验：第 5 轮正常迭代，loss 有限，约 3.11 s/轮，六卡显存约 12.9–13.3 GB、利用率 82–89%。
  已导出 `model_0.onnx`，未发现 traceback 或 nefc overflow。
- 远端 `params/env.yaml` 确认两项 `dead_zone_m: 0.0`，权重均为 −40，地形范围保持不变。
