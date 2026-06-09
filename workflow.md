# SerialLeg NX 调试 Workflow

本文档用于真机调试前的最小闭环检查：

1. 拉起并对齐 NX 时间
2. 打开网页查看通信、状态和 MJCF 可视化

默认环境：

- NX SSH 别名：`serialleg-nx`
- NX runtime：`/home/amov/project/se3_wheel_leg_nx_runtime`
- NX relay URL：`http://192.168.137.100:8081`
- 本机 viewer URL：`http://127.0.0.1:8097`

## 命令运行位置约定

- **本机 PowerShell**：在 Windows 本机执行，通常位于 `D:\robomaster\good_code\2027\se3_wheel_leg`。
- **NX shell**：已经 `ssh serialleg-nx` 登录到 NX 后，在 NX 上执行。
- **本机浏览器**：在 Windows 本机浏览器或 Codex in-app browser 中打开。
- `ssh serialleg-nx "..."` 这一整行在 **本机 PowerShell** 执行；双引号里的命令由 **NX shell** 执行。
- 如果要在 **NX shell** 里直接运行依赖代理的脚本，必须从带 `RemoteForward 7897 127.0.0.1:7897` 的 SSH 会话登录；修改 SSH config 后需要 `exit` 再重新 `ssh serialleg-nx`。

## 1. 拉起时间

先让 NX 时间对齐，避免日志、延迟字段和本机观察时间错位。

运行位置：**本机 PowerShell**。其中双引号内命令在 **NX shell** 执行。

```powershell
ssh serialleg-nx "cd /home/amov/project/se3_wheel_leg_nx_runtime; ./scripts/fix_time_and_pull.sh --time-only; date -Is"
```

运行位置：**本机 PowerShell**。

```powershell
Get-Date -Format o
```

如果已经登录到 NX，也可以在 **NX shell** 里直接执行：

```bash
cd /home/amov/project/se3_wheel_leg_nx_runtime
./scripts/fix_time_and_pull.sh --time-only
date -Is
```

判定：

- NX `date -Is` 和本机 `Get-Date -Format o` 的时间应基本一致。
- 如果差了很多，先修 NX 网络/时间同步，再继续调试。

## 2. 打开网页看通信和状态

先在 NX 上启动 relay-only CDC 可视化服务。NX 只读 STM32 CDC 并转发状态，不做 MJCF 渲染。

运行位置：**本机 PowerShell**。其中双引号内命令在 **NX shell** 执行。

```powershell
ssh serialleg-nx "cd /home/amov/project/se3_wheel_leg_nx_runtime; pkill -f 'se3_deploy.visualize_cdc_state' 2>/dev/null || true; nohup ./scripts/visualize_cdc_state.sh --no-mjcf-render >/tmp/se3_cdc_visualizer.log 2>&1 &"
```

如果已经登录到 NX，也可以在 **NX shell** 里直接执行：

```bash
cd /home/amov/project/se3_wheel_leg_nx_runtime
pkill -f 'se3_deploy.visualize_cdc_state' 2>/dev/null || true
nohup ./scripts/visualize_cdc_state.sh --no-mjcf-render >/tmp/se3_cdc_visualizer.log 2>&1 &
```

然后在本机启动 MJCF viewer。此进程订阅 NX `/events`，用本机 GPU 渲染 MJCF 页面。

运行位置：**本机 PowerShell**，工作目录为训练仓库根目录 `D:\robomaster\good_code\2027\se3_wheel_leg`。

```powershell
uv run se3-visualize-cdc-state `
  --remote-url http://192.168.137.100:8081 `
  --host 127.0.0.1 `
  --viewer-port 8097
```

打开页面。

运行位置：**本机浏览器**。

```text
http://127.0.0.1:8097
```

网页检查项：

- 左上角 `source` 应为 `remote`。
- 左上角 `hz` 应接近 `50`。
- `render` 应为 `mjcf`，不是 `canvas`。
- 右侧 `Comm` 区集中显示通信健康状态和延迟信息。
- `Comm.state_hz` 应稳定在约 `50 Hz`。
- `Comm.target_age_ms` 是 STM32 侧看到的最近一次 NX target 年龄；如果 `target_valid=1`，它不应持续增长。
- `Comm.rx_to_output_ms` 是 STM32 从收到 NX target 到输出到电机链路的耗时；没有 latency 帧时显示 `-`。
- `Comm.latency_age_ms` 表示最近一次 latency 帧距离当前的时间；持续增长说明 STM32 没有继续回传 latency。
- 右侧 `joint_pos`、`joint_vel`、`wheel`、`Observation Slices` 应持续刷新。
- `Visual` / `Collision` 可切换视觉模型和碰撞体。
- `LF0`、`LF1`、`LW`、`RF0`、`RF1`、`RW` 可独立打开对应 joint 坐标系。
- 鼠标拖动 MJCF 画面可旋转视角，滚轮可缩放。

快速检查本机渲染状态。

运行位置：**本机 PowerShell**。

```powershell
Invoke-RestMethod http://127.0.0.1:8097/render_info |
  Select-Object width,height,render_fps,jpeg_quality,show_visual_model,show_collision_model,show_joint_frames |
  Format-List
```

当前默认渲染档位：

- 分辨率：`1280x720`
- 目标刷新：`50 fps`
- JPEG 质量：`95`
- 默认开关：`Visual on`、`Collision off`、所有 joint frame off

## 常见异常

`connected=False`

- NX 没有读到 `/dev/ttyACM*` 或 `/dev/ttyUSB*`。
- 检查 STM32 是否已烧录、USB 是否枚举、`ls /dev/ttyACM* /dev/ttyUSB*` 是否有设备。

`Comm.state_hz` 很低或跳变很大

- STM32 上行状态帧不是稳定 50 Hz。
- USB CDC 或 NX 进程可能卡顿，先看 `/tmp/se3_cdc_visualizer.log`。

`Comm.target_age_ms` 持续增长

- NX 没有持续下发 target。
- 检查 recovery runtime 是否在运行、遥控总开关是否打开、STM32 是否接受 target。

`Comm.rx_to_output_ms` 一直是 `-`

- STM32 没有回传 latency 帧。
- 检查 STM32 侧是否启用了 `MSG_LATENCY` 上行。

网页 `render=canvas`

- 本机 MJCF renderer 未启动或报错。
- 查看 `http://127.0.0.1:8097/render_info` 的 `error` 字段。
