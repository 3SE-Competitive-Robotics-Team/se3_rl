# Laptop Viser Play 值守

本文记录当前 `codex/xyh` 台阶训练的远程可视化方案：A800 只负责训练，Windows laptop 用原生 MuJoCo closedchain sim2sim 运行 `se3-sim2sim --viewer viser --stair-terrain`，开发机只转发 laptop 的 8080 端口并打开浏览器。这样避免在 A800/abbtask 上跑 Viser，也避免用 MJLab 单环境 play 作为回放瓶颈。

## 机器与目录

当前 laptop SSH 别名：

```text
laptop-imgpi2nm-shanghai
```

laptop 上所有本项目 viewer 相关文件必须放在 `E:`，不要放到 `C:`：

```text
E:\se3_stair_viewer                         # repo 副本和 .venv
E:\uv-python                                # uv 管理的 Python
E:\uv-cache                                 # uv cache
E:\se3_stair_viewer_setup                   # keepalive 脚本、cache、tmp
E:\se3_stair_viewer\logs\remote_watch\<run> # checkpoint 和 viewer 日志
```

长期值守脚本也显式设置这些环境变量，避免 Python、Rerun、matplotlib 或 uv 把缓存写回用户目录：

```cmd
set UV_CACHE_DIR=E:\uv-cache
set UV_PYTHON_INSTALL_DIR=E:\uv-python
set XDG_CACHE_HOME=E:\se3_stair_viewer_setup\cache
set PYTHONPYCACHEPREFIX=E:\se3_stair_viewer_setup\cache\pycache
set MPLCONFIGDIR=E:\se3_stair_viewer_setup\cache\matplotlib
set RERUN_CACHE_DIR=E:\se3_stair_viewer_setup\cache\rerun
set TEMP=E:\se3_stair_viewer_setup\tmp
set TMP=E:\se3_stair_viewer_setup\tmp
```

## 当前稳定启动方式

laptop 上用 Windows Scheduled Task 保活 Viser：

```text
SE3LaptopStairViser
```

当前推荐让该 task 执行 watcher：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File E:\se3_stair_viewer_setup\laptop_stair_watcher.ps1
```

watcher 会在 laptop 侧通过 A800 内网拉取最新 checkpoint，按固定间隔重启 viewer，并在 `E:\se3_stair_viewer` 中运行：

```cmd
.venv\Scripts\python.exe -u -m se3_sim2sim.cli ^
  --checkpoint E:\se3_stair_viewer\logs\remote_watch\<run>\<checkpoint>.pt ^
  --model-variant closedchain ^
  --viewer viser ^
  --device cpu ^
  --print-every 0 ^
  --stair-terrain ^
  --stair-terrain-level <level> ^
  --command 1.2 0 0 0 0.32 0 0 0
```

`--stair-terrain` 会在原生 MuJoCo `MjSpec` 编译前添加真实 worldbody box geom；这些台阶参与碰撞，并由 mjviser 从同一个 `MjModel` 渲染。不要用 Viser `server.scene.add_box` 之类的显示层 overlay 代替台阶地形：overlay 只负责显示，不参与 MuJoCo 接触，会导致机器人看起来直接穿模。

viewer 启动后，常规对比 checkpoint 直接在 Viser 的 Controls / Policy 里选择 `Checkpoint` 下拉框即可；切换成功后 sim2sim 会重新加载 policy、清空 GRU hidden 和动作历史，并 reset 当前环境。

只有需要切到另一个 run 目录时，才更新 `laptop_stair_watcher.ps1` 里的 `$RunDir`。同步到 laptop 的 `E:\se3_stair_viewer_setup\` 后重启 task：

```powershell
ssh laptop-imgpi2nm-shanghai "powershell -NoProfile -Command `"Stop-ScheduledTask -TaskName SE3LaptopStairViser; Start-Sleep -Seconds 2; Start-ScheduledTask -TaskName SE3LaptopStairViser`""
```

如果只想固定某个 checkpoint 反复观察，可把 task 临时改成 `laptop_stair_play_keepalive.ps1` 或 `laptop_stair_play_keepalive.cmd`。这两个脚本不会自动拉取最新 checkpoint，只按脚本里的 `RUN_DIR` / `CHECKPOINT` 启动；它们同样必须带 `--stair-terrain`，否则 Viser 中不会有真实台阶碰撞地形。

## 台阶地形约束

laptop 值守使用 native MuJoCo closedchain sim2sim，不直接加载 MJLab 的 terrain generator。因此台阶地形由 `se3_sim2sim.robot.WheelLeggedRobot` 在 `RobotConfig.stair_terrain=True` 时程序化添加：

- `stair_step_height_range=(0.05, 0.20)`，`stair_terrain_level=0..9` 线性映射到 5-20 cm 单级高度。
- 默认 `stair_step_count=6`、`stair_step_depth_m=0.5`、`stair_start_x_m=1.0`、`stair_half_width_m=2.0`。
- 每一级台阶是 worldbody box geom，`contype=2`、`conaffinity=1`，与原 MJCF floor 的碰撞掩码一致；机器人轮子/腿部碰撞 geom 使用 `contype=1`、`conaffinity=2`。
- mjviser 会把这些固定 geom 挂在 `/fixed_bodies` 下，并在 track camera 开启时与地面一起应用 scene offset；不需要额外 overlay。

验收时必须确认机器人会与台阶发生物理接触，而不是只看到灰色盒子。若视觉上出现“台阶跟车走”“台阶有但机器人穿过去”，优先检查是否误用了显示层 overlay，或 viewer 进程是否缺少 `--stair-terrain`。

## 本机打开 Viser

当 laptop 的 8080 隧道不稳定时，优先改用本机 native Viser：开发机直接运行
`se3-sim2sim --viewer viser`，浏览器也直接打开本机 `127.0.0.1:8080`。网络只负责低频
checkpoint 文件同步，不再承载 Viser HTTP/WebSocket 交互。

当前台阶值守推荐用本机 watcher：

```powershell
uv run python scripts\local_stair_viser_watcher.py `
  --run-dir 2026-06-17_10-13-55_stair3k_ctbcslow_m4999_8gpu4096_f4ebc01_20260617_3k `
  --terrain-level 3 `
  --interval-iters 100 `
  --poll-seconds 60 `
  --stability-seconds 10 `
  --device cpu
```

该脚本会把远端稳定的 `model_*.pt` 原子同步到
`logs\remote_watch\<run>\`，再在本机启动 native MuJoCo closedchain：

```cmd
uv run se3-sim2sim ^
  --checkpoint logs\remote_watch\<run>\model_<iter>.pt ^
  --model-variant closedchain ^
  --sim-dt 0.005 ^
  --control-decimation 4 ^
  --viewer viser ^
  --device cpu ^
  --print-every 0 ^
  --stair-terrain ^
  --stair-terrain-level <level> ^
  --stair-ctbc ^
  --command 1.2 0 0 0 0.32 0 0 0
```

默认不要固定 `--stair-ctbc-iter`，这样 Viser 面板里 `Load Latest Checkpoint` 热切换同一
run 目录下的新 checkpoint 时，CTBC 退火会跟随 policy checkpoint 的 `iter` 字段更新。只有
固定观察某个 payload 缺少 `iter` 的 checkpoint 时，才显式传 `--fixed-ctbc-iter` 给 watcher。

远端连接临时失败时，watcher 会继续使用本机已有的最新 checkpoint 保持 Viser 可用；这时画面
不会自动更新到远端最新模型，但本机交互不会因为 tunnel 抖动而断开。恢复 laptop/A800 链路后，
下一轮 poll 会继续同步新 checkpoint。

旧方案是开发机只做 laptop 的 8080 转发：

```powershell
ssh -N -L 8080:localhost:8080 laptop-imgpi2nm-shanghai
```

然后在浏览器打开：

```text
http://127.0.0.1:8080/
```

如果 8080 被旧 viewer 占用，先定位并清掉本机旧 tunnel 或旧本地 play，再重新启动上面的 SSH tunnel。

## 必查项

laptop 侧检查 task、viewer 进程和端口：

```powershell
ssh laptop-imgpi2nm-shanghai "powershell -NoProfile -Command `"Get-ScheduledTask -TaskName SE3LaptopStairViser; Get-NetTCPConnection -LocalPort 8080 -State Listen; Get-CimInstance Win32_Process | ? { `$_.CommandLine -match 'se3_sim2sim.cli' } | select ProcessId,CommandLine`""
```

本机检查 Viser HTTP 是否可达：

```powershell
$times = @()
for ($i = 0; $i -lt 12; $i++) {
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $r = Invoke-WebRequest -UseBasicParsing -Uri http://127.0.0.1:8080/ -TimeoutSec 10
  $sw.Stop()
  $times += [math]::Round($sw.Elapsed.TotalMilliseconds, 1)
}
"min={0}ms avg={1}ms max={2}ms" -f `
  (($times | Measure-Object -Minimum).Minimum), `
  ([math]::Round(($times | Measure-Object -Average).Average, 1)), `
  (($times | Measure-Object -Maximum).Maximum)
```

验收标准：

- `SE3LaptopStairViser` 状态为 `Running`。
- `se3_sim2sim.cli` 进程的 repo、Python、checkpoint 路径均在 `E:`，命令行包含 `--viewer viser --stair-terrain --stair-terrain-level <level>`。
- laptop `0.0.0.0:8080` 由 viewer 进程监听。
- 本机 `http://127.0.0.1:8080/` 返回 `200`，页面标题为 `Viser`。
- Controls / Info 面板显示 `Checkpoint iteration`、`Policy iteration`、`Selected checkpoint`、`Sim time`、`Wall time`、`Policy-step wall`，并且 `Reset Environment` 按钮能把 `Steps` 和 `Sim time` 重置到接近 0。
- Controls / Policy 面板显示 `Checkpoint` 下拉框、`Refresh Checkpoints` 和 `Load Latest Checkpoint`；选择另一个 `model_*.pt` 后，Info 面板的 `Selected checkpoint` / `Policy iteration` 会随之更新，`Reset count` 会增加。
- 画面中的台阶是固定地形：开启 track camera 时台阶与地面一起移动到相对视角下，但不会粘在机器人局部坐标；机器人接近台阶时不会穿模。
- 多次 HTTP 拉取平均延迟应在几十到一百毫秒量级；首次加载 2.8 MB 静态资源可能更慢，不作为交互延迟判断。

## Actual RT 说明

Viser 面板里的 `Actual RT` 不是网络延迟。当前 laptop viewer 使用 native MuJoCo closedchain sim2sim，`Actual RT = policy_steps_per_second * control_dt`，其中 `control_dt = 0.02s`。这个数字直接对应 closedchain policy-step RT。

2026-06-16 在 laptop 上对同一个 `model_600.pt` 做过干净的 closedchain policy-step timing：

```text
cpu:    约 75 policy steps/s，policy 约 1.65 ms，robot.step 约 11.69 ms，按当前 50 Hz 口径 RT 约 1.50x
cuda:0: 约 70 policy steps/s，policy 约 2.35 ms，robot.step 约 11.85 ms，按当前 50 Hz 口径 RT 约 1.40x
```

如果页面仍显示 `Actual RT` 约 `0.05-0.10x`，优先检查是否误启动了旧的 `se3_train.play` / MJLab play。MJLab `BaseViewer` 的 RT 代表 MJLab `env.step`，不是 sim2sim closedchain policy-step；此前 laptop 上 MJLab 单 env play 只能跑到约 `0.03-0.08x`，这正是切换到 native sim2sim Viser 的原因。

2026-06-16 实测状态：

```text
task: SE3LaptopStairViser Running
viewer: E:\se3_stair_viewer\.venv\Scripts\python.exe -m se3_sim2sim.cli ...
python: E:\uv-python\cpython-3.11-windows-x86_64-none\python.exe
checkpoint: E:\se3_stair_viewer\logs\remote_watch\2026-06-15_15-57-36_stair_ctbc_retract_lr1e5_m4999_8gpu_4096pergpu_20260615_235533\model_600.pt
HTTP: 12/12 成功，min 35.5 ms，avg 92.9 ms，max 489.7 ms；首次静态资源加载后稳定在 35-85 ms
browser: http://127.0.0.1:8080/，title=Viser
```
