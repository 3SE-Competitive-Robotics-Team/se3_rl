---
name: remote-dev-se3
description: se3_wheel_leg 远程训练运维。当前 codex/xyh 分支只使用 a800 与 gpufree 两台远程训练服务器；不要把 wuyinyun 作为 SSH、代理、训练、checkpoint 拉取或默认示例目标。
when_to_use: 远程训练 / 训练机 / SSH 隧道 / 训练启动 / 训练日志 / checkpoint / wandb / gpufree / a800 / GPU 机器
user-invocable: true
---

# se3_wheel_leg 远程训练机运维

当前 `codex/xyh` 分支只使用两台远程训练服务器：

| SSH 别名 | 环境 | GPU | 默认环境数 | 机器文档 |
|---|---|---|---:|---|
| `a800` | 局域网 Kubernetes 容器 | NVIDIA A800 * 4 | 每卡 4096，全局约 16384 | `machines/a800.md` |
| `gpufree` | gpufree 容器 | NVIDIA L40S * 1 | 单卡 8192 | `machines/gpufree.md` |

`wuyinyun` 不属于当前分支的可用远程训练服务器。遇到历史归档里的 `wuyinyun` 命令时，只能作为旧记录阅读，不能直接运行，也不要把它改写成默认目标。

## 先选机器

远程操作前必须明确目标机器：

- `a800`：主力多卡训练。当前台阶训练脚本默认通过 `target-via-phone` 进入 Kubernetes pod `gczx-project06/abbtask`，仓库路径为 `/workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg`。
- `gpufree`：L40S 单卡 smoke、短验证和备用训练。每次开机后从控制台刷新 SSH 连接参数，进入后先 `source /root/gpufree-data/se3_env.sh`。

具体 IP、pod、路径、CUDA 兼容库和依赖坑以对应 `machines/*.md` 为准。

## 源码同步规则

源码变更必须通过 git 同步到远程训练机，禁止用 `rsync`、`scp` 或手工复制覆盖源码文件。

标准流程：

1. 本地完成修改、验证、提交。
2. `git push` 当前分支。
3. 远程执行 `git fetch` 后 `git switch` / `git pull --ff-only` 到明确 commit。

A800 pod 访问 GitHub TLS 失败时，用 git bundle 作为 fallback：本地 `git bundle create <name>.bundle HEAD`，传到宿主机和 pod 后在 pod 内 `git fetch /tmp/<name>.bundle HEAD`，仍保持源码通过 git 更新。详细命令见 `machines/a800.md`。

checkpoint、日志、Rerun `.rrd`、summary JSON 等产物可以用 `kubectl cp`、`scp` 或 `rsync` 拉取。

## Windows PowerShell 规则

从 Windows 调远端 bash 时，默认使用 `scripts/remote_bash.ps1` 或专用脚本。不要把含 `&&`、`$()`、`$变量`、管道或多层引号的复杂 bash 直接塞进 PowerShell 双引号字符串。

推荐模板：

```powershell
$bash = @'
date
cd /workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg
git status --short --branch
'@

.\scripts\remote_bash.ps1 -HostAlias target-via-phone -ScriptText $bash
```

需要进入 Kubernetes pod 时，优先使用仓库已有的 A800 自动化脚本或 `scripts/remote_bash.ps1` 的 pod 参数，不要手写多层 `ssh "kubectl exec ... bash -lc \"...\""`。

## 训练管理

训练必须在 tmux session 或现有自动化脚本托管的后台任务中启动；禁止用裸 `nohup ... &`。停止训练时必须杀到实际 Python 子进程：

```bash
pkill -f "se3-train"
```

环境数规则：

| 服务器 | `--gpu-ids` | `--env.scene.num-envs` |
|---|---|---:|
| `a800` | `all` | 每卡 4096，除非实验脚本显式覆盖 |
| `gpufree` | `0` | 8192 |

需要 smoke 或短 benchmark 时，在命令里显式降低 `--env.scene.num-envs`。

## A800 台阶训练入口

当前台阶训练优先使用仓库脚本：

```powershell
uv run python scripts\remote_sync_start_stair.py
```

脚本默认目标：

| 参数 | 默认值 |
|---|---|
| entry host | `target-via-phone` |
| namespace | `gczx-project06` |
| pod | `abbtask` |
| remote project | `/workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg` |
| CUDA compat | `/workspace/cudacompat/usr/local/cuda-12.6/compat` |
| CUDA toolkit lib | `/usr/local/cuda-12.2/lib64` |

A800 容器必须让 compat `libcuda.so` 优先生效。训练日志里必须确认 `Driver 12.6` 且没有 `CUDA Graphs disabled`；否则立即停训并修正库路径。

## gpufree 入口

gpufree 按量计费，默认把 GPU 时间看成最贵资源。代码修改、文档整理、依赖安装、CPU smoke 和产物同步尽量在本地或无卡模式完成；只有 GPU smoke、吞吐 benchmark、长训和 checkpoint 评估需要开启 L40S。

常用命令见 `machines/gpufree.md`。训练前必须加载环境入口：

```bash
source /root/gpufree-data/se3_env.sh
```

## Viser 值守

当前台阶远程训练不在 A800/abbtask 上开 MJLab Viser。值守按 `docs/laptop_viser_play.md` 执行：

- A800 只负责训练。
- Windows laptop 运行 native MuJoCo closedchain `se3-sim2sim --viewer viser --stair-terrain`。
- 开发机只转发 laptop 的 8080 端口并打开浏览器。
- 验收时必须确认台阶是真实 MuJoCo 碰撞地形，机器人不会穿过台阶。

非台阶本地调试可继续用 `se3-play --viewer viser`。

## Checkpoint 和回放

checkpoint 文件名必须按数字排序：

```bash
find logs/rsl_rl/se3_wheel_leg -name "model_*.pt" | sort -V | tail -1
```

远端回放和 Rerun 录制优先使用已安装入口，例如 `.venv/bin/se3-sim2sim` 或 `.venv/bin/python`，避免裸 `uv run` 在录制时触发联网同步。A800 pod 的可选 `.done` / `.failed` marker 缺一个时，`kubectl cp` 的 `Cannot stat` warning 通常无害；以主产物存在且 `.done`/`.failed` 至少一个存在为准。
