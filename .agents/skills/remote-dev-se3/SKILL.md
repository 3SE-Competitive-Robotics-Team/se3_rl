---
name: remote-dev-se3
description: 管理 se3_wheel_leg 在 A800 Kubernetes 容器中的代码同步、训练启动、GPU 选择、日志监控、停止任务和 checkpoint 拉取。用户提到 A800、Kubernetes、target-via-phone、远端训练、checkpoint、GPU 或训练日志时使用；台阶任务优先调用 scripts/remote_sync_start_stair.py。
---

# SE3 远端训练

## 基本原则

- 台阶训练优先使用 `scripts/remote_sync_start_stair.py`，不要重复手写同步和启动流程。
- 脚本负责：远端预检查、checkpoint 检查、代码打包同步、`compileall`、训练启动，以及输出日志和 `watch_remote` 指令。
- 默认环境：
  - host：`target-via-phone`
  - namespace：`gczx-project06`
  - pod：`gpu-8-b6994457c-kv2rj`
  - project：`/workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg`
- 实际环境与默认值不一致时，通过脚本参数覆盖；不确定 pod 时先查询：

```powershell
ssh target-via-phone "kubectl get pods -n gczx-project06 -o wide"
```

- 包含引号、管道、正则、变量或多行 Bash 的远端命令必须先 base64 编码，再传给 `kubectl exec`。不要直接拼接多层 PowerShell、SSH 和 Bash 字符串。
- 不要使用 `pkill -f se3-train`，只停止明确的 PID 或进程组。

## 网络分层与 A800 隔离

- A800/abbtask 是隔离训练环境，不允许直连 GitHub，也不要通过恢复 `17892`、`7897`、`18080` 等 HTTP 代理或 SSH 反向代理让它间接出网。
- SSH 只作为控制面使用：本机先连接 laptop，再由 laptop 进入 A800/abbtask 或 llm 执行命令、查看日志、启动/停止训练和 Viser 值守。
- 代码同步的数据面不要走本机到 laptop 的 SSH 传大目录。流程应为：本机 push 到 GitHub，laptop 从 GitHub 拉取；A800/abbtask 需要更新代码时，由 laptop 生成 git bundle 或压缩包，再通过内网 SSH/kubectl cp 放入隔离环境。
- checkpoint 数据面不要走本机到 laptop 的 SSH 慢链路。流程应为：A800/abbtask 生成 checkpoint，laptop 通过内网从 A800/abbtask 拉取；laptop 再上传到私有 checkpoint 交换仓库 `3SE-Competitive-Robotics-Team/se3_checkpoint_exchange` 的 GitHub Release asset，本机从该仓库下载。
- checkpoint 交换仓库地址：`https://github.com/3SE-Competitive-Robotics-Team/se3_checkpoint_exchange`。不要把 `.pt` 直接提交进 git 历史，只允许作为 release asset 上传。
- checkpoint 交换仓库每个 run 使用独立 release/tag，建议 tag 格式为 `run-<YYYYMMDD-HHMMSS>-<task-or-job>`，并附带 `model_N.pt`、`model_N.meta.json` 和必要日志尾部。metadata 至少记录任务名、run timestamp、commit、checkpoint iter、来源机器和上传时间，保证后续 sim2sim 或部署可追溯。
- laptop 上传 checkpoint 示例：

```powershell
gh release create run-20260615-120000-flat `
  --repo 3SE-Competitive-Robotics-Team/se3_checkpoint_exchange `
  --title "run-20260615-120000-flat" `
  --notes "SE3 checkpoint exchange; see attached metadata."

gh release upload run-20260615-120000-flat `
  C:\path\to\model_1000.pt `
  C:\path\to\model_1000.meta.json `
  --repo 3SE-Competitive-Robotics-Team/se3_checkpoint_exchange `
  --clobber
```

- 本机下载 checkpoint 示例：

```powershell
gh release download run-20260615-120000-flat `
  --repo 3SE-Competitive-Robotics-Team/se3_checkpoint_exchange `
  --pattern "model_1000.*" `
  --dir logs\remote_watch\run-20260615-120000-flat
```

- llm 容器的出网策略按当前任务另行确认；不要因为 llm 可以访问外网，就默认给 A800/abbtask 打开同样链路。

## 远端命令编码

统一使用以下 PowerShell 模板。只修改 `$podScript` 内容：

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
kubectl exec -n gczx-project06 gpu-8-b6994457c-kv2rj -- bash -lc 'echo $podB64 | base64 -d | bash'
"@
$hostB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($hostScript))
ssh target-via-phone "echo $hostB64 | base64 -d | bash"
```

需要查询、停止、清理或复制文件时，将对应 Bash 片段放入 `$podScript`。这样可避免：

- PowerShell 反引号和变量展开。
- SSH 引号嵌套。
- Bash `$`、正则、管道和重定向被上层终端提前解释。
- `kubectl exec -- bash -lc` 的多层转义错误。

## 启动台阶训练

在仓库根目录运行：

```powershell
uv run python scripts/remote_sync_start_stair.py `
  --task SE3-WheelLegged-Stair-GRU `
  --load-run base_model `
  --load-checkpoint model_500\.pt `
  --iterations 3000 `
  --run-name <run_name> `
  --job-name <job_name> `
  --watch-terrain-level 9
```

脚本结束时会打印：

- `TRAIN_PID`
- `TRAIN_LOG`
- `TRAIN_RUN_DIR`
- 对应的 `watch_remote_train_local.py` 指令

## 指定 GPU

保留 `--gpu-ids all`，通过 `--cuda-visible-devices` 指定物理 GPU：

```powershell
uv run python scripts/remote_sync_start_stair.py `
  ... `
  --gpu-ids all `
  --cuda-visible-devices 1,2,3,4,5,6,7
```

脚本会检查选定 GPU 的显存占用。未指定 `--cuda-visible-devices` 时，脚本要求没有其他活跃训练进程。

## 查看状态

将以下内容放入 base64 模板的 `$podScript`：

```bash
set +e
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
pgrep -af '[s]e3-train|[t]orchrunx|[t]orch.distributed.run' || true
grep -E 'Learning iteration|Mean reward|Stair/diag_ctbc_kff|Traceback|RuntimeError|ValueError' \
  /tmp/train_<job_name>.log | tail -160 || true
```

优先使用启动脚本输出的 `watch_remote_train_local.py` 指令查看 checkpoint。

## 停止训练

将以下内容放入 base64 模板的 `$podScript`，只停止对应任务：

```bash
if [ -f /tmp/train_<job_name>.pid ]; then
  pid=$(cat /tmp/train_<job_name>.pid)
  pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
  kill -TERM -- "-$pgid" 2>/dev/null || kill "$pid" 2>/dev/null || true
fi
```

停止后检查 `nvidia-smi`。如果只剩 zombie，不需要处理；如果 GPU 显存仍被占用，定位同一 PGID 的非 zombie 子进程后再清理。

## 拉取 checkpoint

```powershell
$run = "<run>"
$checkpoint = "<model_N.pt>"
$remoteProject = "/workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg"
$hostTmp = "/tmp/${run}_${checkpoint}"
$localDir = "logs\remote_watch\$run"

New-Item -ItemType Directory -Force -Path $localDir | Out-Null
ssh target-via-phone "kubectl cp gczx-project06/gpu-8-b6994457c-kv2rj:$remoteProject/logs/rsl_rl/se3_wheel_leg/$run/$checkpoint $hostTmp -n gczx-project06"
scp "target-via-phone:$hostTmp" "$localDir\$checkpoint"
ssh target-via-phone "rm -f $hostTmp"
```

## 必查项

- checkpoint 必须用 `sort -V` 排序。
- CUDA compat/toolkit 路径由启动脚本检查，不要绕过检查直接启动。
- 无外网训练使用 `WANDB_MODE=offline`，否则可能不保存 checkpoint。
- 启动后同时验证进程、GPU 利用率、iteration 和 `diag_ctbc_kff`，不能只看 PID。
