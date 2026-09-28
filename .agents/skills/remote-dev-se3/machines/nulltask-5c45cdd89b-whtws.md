# nulltask-5c45cdd89b-whtws 训练容器

通用操作流程见父目录 `SKILL.md`。本文件记录此 Pod 的连接参数和实测结果；
资源占用、节点、Pod IP、代码版本均须在使用前复核。

## 连接与边界

- 第一跳：`laptop-wg`。
- Kubernetes 入口：`root@192.168.2.46:2222`，地址为 DHCP 动态值，变化时先核对主机密钥。
- Namespace：`gczx-project06`。
- Pod：`nulltask-5c45cdd89b-whtws`；container：`container-1`。
- 用户于 2026-09-27 明确授权从 `nulltask1` 进行 pod2pod 环境复制，源 Pod 保留不动。
- 只操作此目标；源 Pod 仅用于此次授权的只读复制。

从本机 PowerShell 进入容器：

```powershell
ssh -t laptop-wg "ssh -t -p 2222 root@192.168.2.46 kubectl exec -it -n gczx-project06 nulltask-5c45cdd89b-whtws -c container-1 -- bash"
```

2026-09-27 检查时 Pod 为 `Running`、就绪 `1/1`、重启次数 0；
Pod IP 为 `172.16.3.151`，节点为 `10.10.10.18`。这些值不是固定连接参数。

## CPU、内存与 GPU

以下为 2026-09-27 对 Pod spec、cgroup 和硬件的实测结果：

| 项目 | 配置 |
|---|---|
| CPU 型号 | Intel Xeon Platinum 8462Y+ |
| 宿主机 CPU | 2 路，每路 32 物理核，每核 2 线程；共 64 物理核、128 逻辑 CPU |
| 容器可见 CPU | `nproc=128`，cpuset 为 `0-127` |
| CPU request / limit | `2` / `64` CPU |
| cgroup CPU 配额 | `cpu.cfs_quota_us=6400000`，`cpu.cfs_period_us=100000`，比值为 64 |
| 内存 request / limit | `1000Mi` / `250Gi` |
| GPU request / limit | 7 / 7 |
| GPU 型号 | 7 张 NVIDIA A800 80GB PCIe |
| NVIDIA 驱动 | 535.54.03 |

可见 128 CPU 不代表能持续使用 128 CPU；总 CPU 时间受 64 CPU 配额限制。
64 CPU 也不代表独占核心，CPU request 只有 2 CPU，节点繁忙时需考虑共享竞争。
多进程训练的线程数应按整个 Pod 的配额安排，不能让每个进程都按 `nproc=128` 扩张。

smoke 完成后，3 秒 cgroup 采样约为 0.703 CPU，占配额 1.10%；当时 7 张 GPU
均为利用率 0%、显存占用 2 MiB。这只是当时快照。`kubectl top` 返回
`Metrics API not available`，当前占用应以 cgroup 差分采样为准；`ps %CPU` 是进程
生命周期平均值，`uptime` 的 load average 不能当作此 Pod 的实时 CPU 占用。

## 环境路径

- 仓库：`/workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg`。
- 虚拟环境：仓库下 `.venv`。
- uv：`/root/.local/bin/uv`。
- Python：`/root/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu`，保留源环境的别名符号链接。
- CUDA toolkit：`/usr/local/cuda-12.2`。
- CUDA compat：`/workspace/cudacompat/usr/local/cuda-12.6/compat`。

GPU 检查、smoke 和训练前，必须在同一个 shell 中检查 compat 目录并把它放到
`LD_LIBRARY_PATH` 首位；不得替换系统 CUDA 链接。uv 使用绝对路径或在当前 shell
把 `/root/.local/bin` 加到 `PATH`。

进入容器后的常用命令：

```bash
source /workspace/se3-env.sh
cd /workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg

# 验证当前环境，不触发联网同步依赖。
uv run --no-sync python --version
nvidia-smi

# 5 轮 GPU smoke，默认使用 GPU 0，不上传 W&B。
SE3_SMOKE=1 WANDB_MODE=disabled uv run --no-sync se3-train SE3-WheelLegged-Flat-MLP --env.scene.num-envs 32
```

`se3-env.sh` 检查 CUDA compat 目录并设置当前 shell 的 `PATH` 和
`LD_LIBRARY_PATH`，没有修改系统 CUDA 链接。新开 shell 后需要重新 source。

## 部署记录

2026-09-27 发起集群入口上的 `kubectl exec tar | kubectl exec tar` 复制，数据不绕经本机。
复制代码、Git 历史、资产、`.venv`、uv/uvx、uv Python 和 CUDA compat；排除训练日志、
W&B 历史数据、`.env` 和 Python 字节码缓存。源代码工作区干净，commit 为
`5af0d00cc9b39f26e2fed97f5541ba389c42590c`，子模块为
`e753ce6cbbac9425cb3b6128790a9ff63157f94b`。

复制已完成。依赖导入、7 张 GPU 的 Torch 张量运算和 Warp 初始化全部通过。
Python 3.11.15、uv 0.11.15、Torch 2.11.0+cu128、MJLab 1.5.3、MuJoCo 3.10.0、
MuJoCo-Warp 3.10.0.3、Warp 1.14.0、RSL-RL 5.4.0。

进入容器后先执行 `source /workspace/se3-env.sh`，再进入仓库，使用
`uv run --no-sync` 复用已复制的环境。

GPU 0、32 个环境的 Flat-MLP smoke 已通过 5 轮，退出码为 0，导出 `model_4.onnx`。
PPO value/surrogate loss 为有限值；课程统计 `vel_tracking_lin_vel` 在两轮显示 NaN，
未在此次环境部署中诊断原因。smoke 证明运行及导出链路可用，不代表长期训练质量验收。
验证命令：`SE3_SMOKE=1 WANDB_MODE=disabled uv run --no-sync se3-train SE3-WheelLegged-Flat-MLP --env.scene.num-envs 32`。
验证日志：`/workspace/se3-setup/smoke.log`；run：
`logs/rsl_rl/SE3-WheelLegged-Flat-MLP/2026-09-27_04-48-01`。
显式指定单卡时 CLI 参数应为 `--gpu-ids '[0]'`，或省略并使用默认值。

在线 W&B 链路于 2026-09-28 单独搭建（与 nulltask1 的链路并行、互不影响，不改 nulltask1 的脚本、链和 Secret）：

| 环节 | 值 |
|---|---|
| 本机代理 | 共用笔记本 `proxy.exe` 127.0.0.1:18787（两条网关都只"复用"，谁都不拥有它） |
| boring 隧道 | `whtws-wandb-proxy`：入口机 `0.0.0.0:38444` ← 笔记本 18787（`~/.boring.toml` 第二条，group se3） |
| 入口机防火墙 | 链 `SE3_WANDB_WHTWS`，INPUT 规则注释 `se3-whtws-wandb-gateway`，dport 38444，只放行本 Pod IP |
| 网关脚本 | 笔记本 `C:\Users\Lenovo\.local\bin\whtws-wandb-gateway.ps1`（由 nulltask1 脚本替换名字生成） |
| 计划任务 | `SE3-Whtws-WandbGateway`（登录触发、失败重启），状态 `~/.local/state/whtws-wandb-gateway/gateway.log` |
| Pod 内代理地址 | `http://10.10.10.116:38444` |
| Secret | `whtws-wandb`（namespace 同）：HTTP_PROXY/HTTPS_PROXY/http_proxy/https_proxy 已填；`WANDB_API_KEY` 2026-09-28 由用户批准从 `nulltask1-wandb` 服务端复制（值未离开集群），启动器 `--dry-run` 已通过 W&B 认证探测 |
| 启动器 | 本机 Git Bash `C:\Users\13567\.local\bin\start-whtws-training.sh`，Secret 在入口机读取经 stdin 注入 Pod，`--dry-run` 只做预检（含 W&B 认证探测） |

验证（2026-09-28 00:31）：网关日志 `Firewall refreshed: pod_ip=172.16.3.151` / `gateway is ready`；Pod 经 38444 到 api.wandb.ai 返回 404（可达），
nulltask1 经 38443 不受影响，本 Pod 走 38443 被拦截。网关任务退到 Ready 时先 `schtasks /Run /TN SE3-Whtws-WandbGateway`；
隧道 closed 时 `boring open whtws-wandb-proxy`。

现场检查可见 7 张 NVIDIA A800 80GB PCIe，驱动 535.54.03；GPU 占用是动态值，启动前复核。
`/workspace` 当前位于容器 overlay，并非已确认的持久卷；Pod 被删除重建时，环境可能丢失。

## 训练记录

| 标签 | 任务 / commit | run | PGID | state dir | 备注 |
|---|---|---|---|---|---|
| M37 | `SE3-WheelLegged-Rough-Exp-HeightWindowDz5Prune5` / `448c7e4` | `2026-09-27_14-18-53_rough-M37-dz5-prune5-seed42-7x8192-5k` | 1810096 | `/workspace/.se3-training-state/whtws/20260927T141846Z` | 2026-09-27 22:18 启动，七卡 × 8192、5000 轮、`WANDB_MODE=disabled`（无在线 W&B，看 TensorBoard）；2026-09-27 按用户指令在 2058 轮 SIGINT 停止（最后 checkpoint `model_2000`），结论见 docs/plan/m37_reward_prune_20260927.md。启动方式：stdin 脚本 `setsid nohup uv run --no-sync se3-train ...`，无 nulltask1 那套启动器 |

| M38 | `SE3-WheelLegged-Rough-Exp-HeightWindowDz5Prune5Upward1` / `25ca875` | `2026-09-27_16-45-53_rough-M38-dz5-prune5-upward1-seed42-7x8192-5k` | 1860054 | `/workspace/.se3-training-state/whtws/20260927T164546Z` | 2026-09-28 00:45 启动，七卡 × 8192、5000 轮，在线 W&B `bfla7ogz`，用 `start-whtws-training.sh` 启动 |

| M39 | `SE3-WheelLegged-Rough-Exp-ActionRate010` / `a724b49` | `2026-09-28_04-42-40_rough-M39-actionrate010-seed42-7x8192-5k` | 2088093 | `/workspace/.se3-training-state/whtws/20260928T044233Z` | 2026-09-28 12:42 启动，七卡 × 8192、5000 轮，在线 W&B `uouikynn` |

| M40 | `SE3-WheelLegged-Rough-Exp-ActionRate010Smooth006` / `28faab7` | `2026-09-28_11-21-00_rough-M40-actionrate010-smooth006-seed42-7x8192-5k` | 2216716 | `/workspace/.se3-training-state/whtws/20260928T112053Z` | 2026-09-28 19:21 启动，七卡 × 8192、5000 轮，在线 W&B `j5zwvw1x` |

| M41 | `SE3-WheelLegged-Rough-Exp-ActionRate001NoSmooth` / `8ea3c9e` | `2026-09-28_13-11-28_rough-M41-actionrate001-nosmooth-seed42-7x8192-5k` | 2253936 | `/workspace/.se3-training-state/whtws/20260928T131121Z` | 2026-09-28 21:11 启动，七卡 × 8192、5000 轮，在线 W&B `pe5zdez0`；M40 为此在 1767 轮停止 |

仓库于 2026-09-27 通过 git bundle 由 5af0d00 快进到 `448c7e4`，2026-09-28 再快进到 `25ca875`（子模块仍 `e753ce6`）。
停止：`ps -o args= -g <pgid>` 核对 run name 后 `kill -INT -- -<pgid>`；容器 PID 1 不回收僵尸，defunct 条目可忽略。
