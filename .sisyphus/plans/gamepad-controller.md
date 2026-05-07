# Gamepad Controller for sim2sim

## 目标

为 `se3-sim2sim` 接入 XInput 手柄，实现实时 5D 命令控制，替代 CLI 静态 `--command` 参数。

## 映射方案

| 手柄输入 | 命令维度 | 范围 |
|---|---|---|
| 左摇杆 Y 轴 | `lin_vel_x` | ±1.5 m/s |
| 右摇杆 X 轴 | `ang_vel_yaw` | ±3.0 rad/s |
| 右摇杆 Y 轴 | `pitch` | ±0.2 rad |
| 左摇杆 X 轴 | `roll` | ±0.1 rad |
| LT/RT 触发器 | `height` | 0.22 ~ 0.32 m（LT 下蹲, RT 站高, 中位 0.27） |

死区：摇杆 ±0.08，触发器 0.02。

## 架构

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Gamepad    │────▶│  GamepadThread   │────▶│  robot.cmd  │
│  (XInput)   │     │  (独立线程,60Hz) │     │  (numpy 5D) │
└─────────────┘     └──────────────────┘     └─────────────┘
                            │
                            ▼
                    ┌──────────────────┐
                    │  rr.log cmd HUD  │
                    └──────────────────┘
```

- `GamepadThread`：独立守护线程，轮询手柄状态（60Hz），写入共享 `np.ndarray`
- `WheelLeggedRobot.command`：每步从共享数组读取最新值
- Rerun HUD：每步 `rr.log("command/...", rr.Scalar(...))` 显示当前命令

## 依赖

- `pygame`（跨平台 gamepad 支持，XInput/DInput/HID 均可）
- 备选：`inputs`（更轻量，纯 evdev/XInput，无窗口依赖）

## 实现步骤

1. 新增 `src/se3_sim2sim/gamepad.py`
   - `GamepadReader` 类：初始化 pygame joystick，轮询 axis/trigger
   - 映射表可配置（dataclass），支持不同手柄布局
   - 断连重连机制（手柄热插拔不崩溃）

2. 新增 `src/se3_sim2sim/command_source.py`
   - 抽象 `CommandSource` 协议（`get_command() -> np.ndarray`）
   - 实现：`StaticCommand`（当前行为）、`GamepadCommand`（手柄）
   - 后续可扩展：`KeyboardCommand`、`NetworkCommand`

3. 修改 `robot.py`
   - `WheelLeggedRobot.__init__` 接受 `CommandSource`
   - `observation()` 中 `self.command` 从 source 实时读取

4. 修改 `cli.py`
   - 新增 `--gamepad` flag
   - 有 flag 时实例化 `GamepadCommand`，否则用 `StaticCommand`

5. Rerun 可视化增强
   - 命令向量实时绘制（5 条 scalar 曲线）
   - 手柄连接状态指示

## 验收标准

- [ ] 手柄接入后能实时控制机器人前进/转向/俯仰/侧倾/蹲站
- [ ] 手柄断开时 graceful fallback 到零速站立（不崩溃）
- [ ] 无手柄时 `--command` 静态模式照常工作
- [ ] Rerun 中能看到命令实时曲线
