---
name: remote-dev-se3
description: Use when managing se3_wheel_leg remote training on A800 Kubernetes via laptop, gpufree, wuyingyun, SSH, kubectl pods, GPU selection, training launch, logs, checkpoint pull, GitHub Release checkpoint exchange, wandb, local native Viser watch, and stopping remote jobs; do not use wuyingyun as a default target unless the user explicitly names it.
---

# SE3 远端训练

## 基本原则

- **Pod 隔离**：A800 集群为多用户共享环境。每个 machine 文件只对应一个用户的 pod，严禁操作同集群中其他用户的 pod 内进程（不得 exec、kill、cp、log 到非本文件指定的 pod）。`a800-xyh-am345` 只能操作 `abbtask-*` pod，不得触碰同 namespace 下其他 pod。
- 当前 `codex/xyh` 默认使用 `a800` 与 `gpufree`；`wuyingyun` 是无影云单卡机器，仅在用户明确指定时作为 SSH、代理、训练或 checkpoint 目标。
- 台阶训练优先使用 `scripts/remote_sync_start_stair.py`，不要重复手写同步和启动流程。
- 脚本负责远端预检查、checkpoint 检查、代码打包同步、`compileall`、训练启动，并输出日志和 `watch_remote` 指令。
- 开发机不能直连 A800。默认控制链路是：开发机 `ssh laptop-imgpi2nm-shanghai`，再由 laptop `ssh a800`，最后在 A800 宿主机执行 `kubectl exec -n gczx-project06 abbtask-79cdb78487-mgx44 -- bash`。
- 日常源码同步必须走 git：开发机提交并 push 后，控制 laptop 拉取代码，再从 laptop 同步/启动 A800。不要把本机工作区打包当作常规源码同步方式。
- A800 启动脚本默认不同步 `assets/base_model`，日常代码同步约 50 秒量级；只有需要更新或补齐远端基模时才显式传 `--sync-base-model`。
- 台阶值守默认用 GitHub Release checkpoint exchange + 本机 native MuJoCo closedchain Viser，不依赖 laptop 8080 隧道；旧的 laptop 8080 转发只作 fallback。
- 复杂远端命令必须通过 `scripts/remote_bash.ps1` 或 base64 模板发送；不要直接拼接多层 PowerShell、SSH、kubectl 和 Bash 字符串。
- 停止单个任务时只停止明确的 PID 或进程组。只有确认要清空容器内全部训练时，才考虑 `pkill -f se3-train`。

## 资源路由

- A800 细节、CUDA compat、git bundle fallback 和产物拉取边界：读 `machines/a800-xyh-am345.md`。
- gpufree 单卡训练、smoke 和计费注意事项：读 `machines/gpufree.md`。
- NX / Jetson 真机部署：读 `machines/nx.md`；不要把 NX 当 MJLab 训练服务器。
- 台阶 Viser 验收细节、Actual RT 解释和历史 laptop task：读仓库根目录的 `docs/laptop_viser_play.md`。

## A800 默认入口

| 项目 | 默认值 |
|---|---|
| laptop host | `laptop-imgpi2nm-shanghai` |
| A800 host（从 laptop 访问） | `a800` |
| namespace | `gczx-project06` |
| pod 逻辑目标 | `abbtask` |
| 当前 kubectl pod 名 | `abbtask-79cdb78487-mgx44` |
| laptop repo | `E:\se3_rl_train` |
| laptop uv Python dir | `E:\uv-python` |
| checkpoint exchange repo | `3SE-Competitive-Robotics-Team/se3_checkpoint_exchange` |
| laptop checkpoint exchange repo | `E:\se3_checkpoint_exchange` |
| local checkpoint exchange repo | `D:\RoboMaster\se3_checkpoint_exchange` |
| remote project | `/workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg` |
| CUDA compat | `/workspace/cudacompat/usr/local/cuda-12.6/compat` |
| CUDA toolkit lib | `/usr/local/cuda-12.2/lib64` |

`a800` 是 laptop 上可用的 SSH host alias，不是开发机直连目标；开发机只能先连 `laptop-imgpi2nm-shanghai`。

本 worktree 的实际 A800 目标是 `abbtask`，但 `kubectl exec` / `kubectl cp` 需要当前完整 pod 名。当前查到的是 `abbtask-79cdb78487-mgx44`。不要进入 recovery 历史入口 `gpu-8-b6994457c-kv2rj` 或其他用户容器；查询 pod 只用于确认 `abbtask-*` 的当前完整名称：

```powershell
ssh laptop-imgpi2nm-shanghai "ssh a800 `"kubectl get pods -n gczx-project06 -o wide`""
```

## 源码同步

日常源码同步按 git 流转：

1. 开发机本地完成修改、验证、提交。
2. 开发机 `git push` 当前分支。
3. 开发机通过 SSH 控制 laptop 拉取同一 commit。
4. 从 laptop 的 repo 工作区运行 A800 同步/启动脚本。

推荐 laptop 侧 repo 路径按实际部署确认。训练同步 repo 必须是一个干净 git 工作区；不要默认使用值守目录当训练同步目录。本次链路检查发现 `E:\se3_stair_viewer` 的 Git 顶层是 `E:\` 且没有可用提交，不能直接作为训练同步 repo。

```powershell
$branch = "codex/stair-visualbase-obs34"
$commit = git rev-parse HEAD
$laptopRepo = "E:\se3_rl_train"

ssh laptop-imgpi2nm-shanghai "powershell -NoProfile -Command `"cd '$laptopRepo'; git fetch origin $branch; git checkout $branch; git pull --ff-only origin $branch; git rev-parse HEAD`""
```

确认 laptop 输出的 commit 与开发机 `$commit` 一致后，再启动远端同步。只有临时排障且明确接受不可复现风险时，才使用本机 tar/scp 方式绕过 git。

laptop 用户级环境变量已设置 `UV_PYTHON_INSTALL_DIR=E:\uv-python`。如果 `uv run` 又开始扫描 `C:\Users\Lenovo\AppData\Roaming\uv\python` 并报 `os error 448`，先恢复该环境变量：

```powershell
ssh laptop-imgpi2nm-shanghai "powershell -NoProfile -Command `"[Environment]::SetEnvironmentVariable('UV_PYTHON_INSTALL_DIR','E:\uv-python','User')`""
```

## 远端命令编码

开发机执行 A800 bash 时，优先使用仓库脚本；默认路线已经是 laptop -> a800：

```powershell
$bash = @'
set -euo pipefail
cd /workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg
nvidia-smi
pgrep -af '[s]e3-train|[t]orchrunx|[t]orch.distributed.run' || true
'@

.\scripts\remote_bash.ps1 `
  -KubeNamespace gczx-project06 `
  -KubePod abbtask-79cdb78487-mgx44 `
  -NoWorkdir `
  -ScriptText $bash
```

没有脚本或需要手工排障时，使用双层 base64 模板，只修改 `$podScript` 内容：

```powershell
$podScript = @'
set -euo pipefail
cd /workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg
nvidia-smi
pgrep -af '[s]e3-train|[t]orchrunx|[t]orch.distributed.run' || true
'@

$podB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($podScript))
$hostScript = @"
set -euo pipefail
ssh a800 "kubectl exec -n gczx-project06 abbtask-79cdb78487-mgx44 -- bash -lc 'echo $podB64 | base64 -d | bash'"
"@
$hostB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($hostScript))
ssh laptop-imgpi2nm-shanghai "echo $hostB64 | base64 -d | bash"
```

这样可避免 PowerShell 反引号、变量展开、SSH 引号嵌套、Bash `$`、正则、管道和重定向被上层终端提前解释。

## 启动台阶训练

在 laptop 的 repo 根目录运行，而不是在开发机直接打包当前工作区：

```powershell
ssh laptop-imgpi2nm-shanghai "powershell -NoProfile -Command `"cd E:\se3_rl_train; uv run python scripts\remote_sync_start_stair.py --entry-host a800 --namespace gczx-project06 --pod abbtask-79cdb78487-mgx44 --task SE3-WheelLegged-Stair-GRU --load-run base_model --load-checkpoint model_500\.pt --iterations 3000 --run-name <run_name> --job-name <job_name> --watch-terrain-level 9`""
```

脚本默认排除 `assets/base_model` 以减少 laptop -> A800 传输时间。远端已有对应 checkpoint 时保持默认；只有需要把 laptop repo 里的基模同步进 A800 时加：

```powershell
--sync-base-model
```

脚本结束时必须打印：

- `TRAIN_PID`
- `TRAIN_LOG`
- `TRAIN_RUN_DIR`
- 对应的 `watch_remote_train_local.py` 指令

保留 `--gpu-ids all`，通过 `--cuda-visible-devices` 指定物理 GPU：

```powershell
ssh laptop-imgpi2nm-shanghai "powershell -NoProfile -Command `"cd E:\se3_rl_train; uv run python scripts\remote_sync_start_stair.py --entry-host a800 --pod abbtask-79cdb78487-mgx44 --gpu-ids all --cuda-visible-devices 1,2,3,4,5,6,7`""
```

未指定 `--cuda-visible-devices` 时，脚本要求没有其他活跃训练进程；指定后脚本会检查选定 GPU 的显存占用。

## 查看状态

将以下内容放入远端命令模板的 `$podScript`：

```bash
set +e
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
pgrep -af '[s]e3-train|[t]orchrunx|[t]orch.distributed.run' || true
grep -E 'Learning iteration|Mean reward|Stair/diag_ctbc_kff|Traceback|RuntimeError|ValueError' \
  /tmp/train_<job_name>.log | tail -160 || true
```

优先使用启动脚本输出的 `watch_remote_train_local.py` 指令查看 checkpoint 和日志。台阶 Viser 值守不要用 A800/MJLab play，按下方“Viser 值守”使用本机 native watcher。

## 停止训练

将以下内容放入远端命令模板的 `$podScript`，只停止对应任务：

```bash
if [ -f /tmp/train_<job_name>.pid ]; then
  pid=$(cat /tmp/train_<job_name>.pid)
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
  kill -TERM -- "-$pgid" 2>/dev/null || kill "$pid" 2>/dev/null || true
fi
```

停止后检查 `nvidia-smi`。如果只剩 zombie，不需要处理；如果 GPU 显存仍被占用，定位同一 PGID 的非 zombie 子进程后再清理。

## 拉取 checkpoint

默认使用 GitHub Release assets 中转 checkpoint，不用 laptop -> 本机 `scp`。A800/abbtask 不连 GitHub；laptop 作为数据桥从 A800 内网拉 checkpoint，再上传到私有仓库 `3SE-Competitive-Robotics-Team/se3_checkpoint_exchange` 的 Release asset；本机从 GitHub 下载。

不要把 `.pt` commit 进 git 历史；checkpoint exchange 仓库只存 Release assets。

### 1. 确认 checkpoint

checkpoint 必须用数字排序：

```powershell
.\scripts\remote_bash.ps1 `
  -KubeNamespace gczx-project06 `
  -KubePod abbtask-79cdb78487-mgx44 `
  -NoWorkdir `
  -ScriptText 'cd /workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg && find logs/rsl_rl/se3_wheel_leg/<run> -name "model_*.pt" | sort -V | tail -1'
```

### 2. A800 pod -> laptop -> GitHub Release

在本机通过 SSH 控制 laptop 执行；`gh` 在 laptop 上已安装并登录。把 `<run>`、`<model_N.pt>` 和 tag 换成实际值：

```powershell
$run = "<run>"
$checkpoint = "<model_N.pt>"
$tag = "run-<YYYYMMDD-HHMMSS>-<job>"
$pod = "abbtask-79cdb78487-mgx44"
$remoteProject = "/workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg"
$exchangeRepo = "3SE-Competitive-Robotics-Team/se3_checkpoint_exchange"

$laptopScript = @"
`$ErrorActionPreference = 'Stop'
`$run = '$run'
`$checkpoint = '$checkpoint'
`$tag = '$tag'
`$pod = '$pod'
`$remoteProject = '$remoteProject'
`$exchangeRepo = '$exchangeRepo'
`$hostTmp = "/tmp/`$tag-`$checkpoint"
`$workDir = "C:\tmp\`$tag"
`$assetPath = Join-Path `$workDir `$checkpoint

New-Item -ItemType Directory -Force -Path `$workDir | Out-Null
try {
  ssh a800 "kubectl cp gczx-project06/`${pod}:`$remoteProject/logs/rsl_rl/se3_wheel_leg/`$run/`$checkpoint `$hostTmp -n gczx-project06"
  scp "a800:`$hostTmp" `$assetPath
  gh release create `$tag `$assetPath `
    --repo `$exchangeRepo `
    --title `$tag `
    --notes "SE3 checkpoint exchange: `$run/`$checkpoint" `
    --prerelease
} finally {
  ssh a800 "rm -f `$hostTmp"
}
"@

$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($laptopScript))
ssh laptop-imgpi2nm-shanghai powershell -NoProfile -EncodedCommand $encoded
```

### 3. 本机从 GitHub 下载

如果本机有 `gh`：

```powershell
$tag = "run-<YYYYMMDD-HHMMSS>-<job>"
$checkpoint = "<model_N.pt>"
$exchangeRepo = "3SE-Competitive-Robotics-Team/se3_checkpoint_exchange"
$localDir = "logs\remote_watch\$tag"

New-Item -ItemType Directory -Force -Path $localDir | Out-Null
gh release download $tag `
  --repo $exchangeRepo `
  --pattern "$checkpoint" `
  --dir $localDir
```

当前本机可能没有 `gh`，可用 Git credential + GitHub API 下载 private Release asset：

```powershell
$tag = "run-<YYYYMMDD-HHMMSS>-<job>"
$checkpoint = "<model_N.pt>"
$exchangeRepo = "3SE-Competitive-Robotics-Team/se3_checkpoint_exchange"
$localDir = "logs\remote_watch\$tag"
New-Item -ItemType Directory -Force -Path $localDir | Out-Null

$cred = "protocol=https`nhost=github.com`n`n" | git credential fill
$token = ($cred | Where-Object { $_ -like "password=*" } | Select-Object -First 1) -replace "^password=", ""
if (-not $token) { throw "missing GitHub credential token" }

$headers = @{
  Authorization = "Bearer $token"
  Accept = "application/vnd.github+json"
  "User-Agent" = "se3-checkpoint-download"
}
$release = Invoke-RestMethod -Headers $headers -Uri "https://api.github.com/repos/$exchangeRepo/releases/tags/$tag"
$asset = $release.assets | Where-Object { $_.name -eq $checkpoint } | Select-Object -First 1
if (-not $asset) { throw "missing release asset: $checkpoint" }

curl.exe -L --fail `
  -H "Authorization: Bearer $token" `
  -H "Accept: application/octet-stream" `
  -H "User-Agent: se3-checkpoint-download" `
  "https://api.github.com/repos/$exchangeRepo/releases/assets/$($asset.id)" `
  -o "$localDir\$checkpoint"
```

### 4. 清理 laptop 中转文件

Release asset 保留作为可复用 checkpoint；只清 laptop / A800 临时文件：

```powershell
$tag = "run-<YYYYMMDD-HHMMSS>-<job>"
ssh laptop-imgpi2nm-shanghai "powershell -NoProfile -Command `"Remove-Item -LiteralPath 'C:\tmp\$tag' -Recurse -Force -ErrorAction SilentlyContinue; ssh a800 'rm -f /tmp/${tag}-*'`""
```

实测参考：20.4 MB checkpoint，A800 pod -> A800 host 约 3.0s，A800 host -> laptop 约 11.6s，laptop -> GitHub Release 约 12.4s，GitHub -> 本机约 4.9s。完整 GitHub 路线适合保留和复用；临时只看一次且不需要留档时，才考虑 laptop -> 本机 `scp` fallback。

### 5. 连续值守发布

长期 Viser 值守时，不要反复手工 `gh release create`。让能拿到 checkpoint 缓存的机器运行 publisher，把稳定的 `model_*.pt` 按间隔上传到同一个 GitHub Release：

```powershell
uv run python scripts\github_release_checkpoint_publisher.py `
  --checkpoint-dir logs\remote_watch\<run> `
  --github-release-tag run-<YYYYMMDD-HHMMSS>-<job> `
  --poll-seconds 60 `
  --stability-seconds 10 `
  --interval-iters 100
```

`--stability-seconds 10` 必须显式传入，避免发布半写入 checkpoint。publisher 优先读取 `GITHUB_TOKEN` / `GH_TOKEN`，否则使用 git credential helper；它不依赖本机安装 `gh`。

### 6. 临时 scp fallback

只在不需要 GitHub 留档或 GitHub 不可用时使用：

```powershell
$run = "<run>"
$checkpoint = "<model_N.pt>"
$pod = "abbtask-79cdb78487-mgx44"
$remoteProject = "/workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg"
$hostTmp = "/tmp/${run}_${checkpoint}"
$localDir = "logs\remote_watch\$run"

New-Item -ItemType Directory -Force -Path $localDir | Out-Null
ssh laptop-imgpi2nm-shanghai "ssh a800 `"kubectl cp gczx-project06/${pod}:$remoteProject/logs/rsl_rl/se3_wheel_leg/$run/$checkpoint $hostTmp -n gczx-project06`""
ssh laptop-imgpi2nm-shanghai "scp a800:$hostTmp $hostTmp"
scp "laptop-imgpi2nm-shanghai:$hostTmp" "$localDir\$checkpoint"
ssh laptop-imgpi2nm-shanghai "ssh a800 `"rm -f $hostTmp`"; rm -f $hostTmp"
```

## gpufree

gpufree 是 L40S 单卡 smoke、短验证和备用训练入口。按量计费时，把 GPU 时间视为最贵资源；代码修改、文档整理、依赖安装、CPU smoke 和产物同步尽量本地完成。

训练前必须加载环境：

```bash
source /root/gpufree-data/se3_env.sh
```

默认训练参数：`--gpu-ids 0`，`--env.scene.num-envs 8192`。需要 smoke 或短 benchmark 时显式降低环境数。

## Viser 值守

当前台阶远程训练不在 A800 pod 上开 MJLab Viser。A800/abbtask 只负责训练；值守默认是 GitHub Release checkpoint exchange + 本机 native MuJoCo closedchain Viser：

```powershell
uv run python scripts\local_stair_viser_watcher.py `
  --source github-release `
  --run-dir <run> `
  --github-release-tag run-<YYYYMMDD-HHMMSS>-<job> `
  --terrain-level <0..9> `
  --interval-iters 100 `
  --poll-seconds 60 `
  --stability-seconds 10 `
  --device cpu
```

本机 watcher 从 `3SE-Competitive-Robotics-Team/se3_checkpoint_exchange` 的 Release asset 拉取稳定 checkpoint，原子同步到 `logs\remote_watch\<run>\`，再在本机启动：

```powershell
uv run se3-sim2sim `
  --checkpoint logs\remote_watch\<run>\model_<iter>.pt `
  --model-variant closedchain `
  --sim-dt 0.005 `
  --control-decimation 4 `
  --viewer viser `
  --device cpu `
  --print-every 0 `
  --stair-terrain `
  --stair-terrain-level <0..9> `
  --stair-ctbc `
  --command 1.2 0 0 0 0.32 0 0 0
```

验收必须检查：

- 进程命令行包含 `--model-variant closedchain --viewer viser --stair-terrain --stair-terrain-level <level>`。
- `http://127.0.0.1:8080/` 返回 `200`，页面标题为 `Viser`。
- 台阶是真实 MuJoCo worldbody box 碰撞地形，不是 Viser overlay；机器人接近台阶时不能穿模。
- 多次 HTTP 拉取平均延迟应在几十到一百毫秒量级；首次静态资源加载可更慢。

远端连接或 GitHub 临时失败时，watcher 会回退到本机已有最新 checkpoint，保持当前 Viser 可用；这时画面不会自动更新到远端最新模型。恢复后下一轮 poll 会继续同步。

只有在 GitHub exchange 不可用或需要排查 laptop native viewer 时，才使用旧方案：Windows laptop 运行 `se3-sim2sim --viewer viser --stair-terrain`，开发机 `ssh -N -L 8080:localhost:8080 laptop-imgpi2nm-shanghai` 转发 8080。该方案不再是默认值守路径。

非台阶本地调试可继续用 `se3-play --viewer viser`。

## 必查项

- CUDA compat/toolkit 路径由启动脚本检查，不要绕过检查直接启动。
- A800 日志必须确认有效 CUDA driver，且不应出现 `CUDA Graphs disabled`。
- 无外网训练使用 `WANDB_MODE=offline`，否则可能不保存 checkpoint。
- 启动后同时验证进程、GPU 利用率、iteration 和关键诊断指标，不能只看 PID。
- 远端回放和 Rerun 录制优先使用 `.venv/bin/se3-sim2sim` 或 `.venv/bin/python`，避免裸 `uv run` 在录制时触发联网同步。
