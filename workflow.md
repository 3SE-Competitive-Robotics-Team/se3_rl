# SerialLeg NX 调试 Workflow

目标：先启动 NX 侧 CDC relay，再在本机打开 MJCF viewer 看真机回传状态。

默认环境：

- NX SSH：`serialleg-nx`
- NX runtime：`/home/amov/project/se3_wheel_leg_nx_runtime`
- NX relay：`http://192.168.137.100:8081`
- 本机 viewer：`http://127.0.0.1:8097`

## 1. 对齐 NX 时间

运行位置：本机 PowerShell。

```powershell
ssh serialleg-nx "cd /home/amov/project/se3_wheel_leg_nx_runtime; ./scripts/fix_time_and_pull.sh --time-only; date -Is"
Get-Date -Format o
```

如果已经登录 NX，也可以在 NX shell 里运行：

```bash
cd /home/amov/project/se3_wheel_leg_nx_runtime
./scripts/fix_time_and_pull.sh --time-only
date -Is
```

## 2. 启动 NX CDC Relay

运行位置：本机 PowerShell。双引号里的命令实际在 NX shell 执行。

```powershell
ssh serialleg-nx "cd /home/amov/project/se3_wheel_leg_nx_runtime; pkill -f 'se3_deploy.visualize_cdc_state' 2>/dev/null || true; nohup ./scripts/visualize_cdc_state.sh --local-cdc --no-mjcf-render >/tmp/se3_cdc_visualizer.log 2>&1 &"
```

如果已经登录 NX，也可以在 NX shell 里运行：

```bash
cd /home/amov/project/se3_wheel_leg_nx_runtime
pkill -f 'se3_deploy.visualize_cdc_state' 2>/dev/null || true
nohup ./scripts/visualize_cdc_state.sh --local-cdc --no-mjcf-render >/tmp/se3_cdc_visualizer.log 2>&1 &
```

检查 NX relay：

```powershell
Invoke-RestMethod http://192.168.137.100:8081/snapshot |
  Select-Object source,connected,port,seq,frame_hz,target_valid,target_age_ms,rc_switch_r,output_enabled |
  Format-List
```

## 3. 启动本机 MJCF Viewer

运行位置：本机 PowerShell，工作目录为 `D:\robomaster\good_code\2027\se3_wheel_leg`。

现在 viewer 默认订阅 NX 真机回传，所以直接运行：

```powershell
cd D:\robomaster\good_code\2027\se3_wheel_leg
$env:NO_PROXY="localhost,127.0.0.1,::1,192.168.137.100"
$env:no_proxy="localhost,127.0.0.1,::1,192.168.137.100"
uv run se3-visualize-cdc-state
```

打开页面：

```text
http://127.0.0.1:8097
```

网页检查项：

- 左上角 `source` 应为 `remote`。
- 左上角 `hz` 应接近 `50`。
- `render` 应为 `mjcf`，不是 `canvas`。
- 右侧 `Comm` 显示通信健康状态和延迟字段。
- `joint_pos`、`joint_vel`、`wheel`、`Observation Slices` 应持续刷新。
- `Visual` / `Collision` 可切换视觉模型和碰撞体。
- `Gravity` 开关只切换 `base_link` 姿态来源：打开用 `projected_gravity`，关闭用 `base_ang_vel` 积分。
- `base_link`、`LF0`、`LF1`、`LW`、`RF0`、`RF1`、`RW` 可独立打开坐标系显示。
- 鼠标拖动 MJCF 画面可旋转视角，滚轮可缩放。

快速检查本机渲染状态：

```powershell
Invoke-RestMethod http://127.0.0.1:8097/render_info |
  Select-Object ready,render_fps,attitude_source,use_gravity_attitude,width,height,error |
  Format-List
```

## 常见异常

`source=remote` 且 `connected=False`

- 本机 viewer 连到了 NX relay，但 NX relay 没读到 STM32 CDC。
- 看 NX 日志：`ssh serialleg-nx "tail -80 /tmp/se3_cdc_visualizer.log"`。

无法访问 `http://192.168.137.100:8081`

- NX relay 没启动，或本机到 NX 网络不通。
- 重新执行第 2 步。

网页 `render=canvas`

- 本机 MJCF renderer 未启动或报错。
- 查看 `http://127.0.0.1:8097/render_info` 的 `error` 字段。

`target_age_ms` 持续增长

- NX 没有持续下发 target。
- 检查 recovery runtime 是否在运行、遥控总开关是否打开、STM32 是否接受 target。
