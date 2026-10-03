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

| M42 | `SE3-WheelLegged-Rough-Exp-FudanActuation` / `f4f515c` | `2026-09-28_15-18-44_rough-M42-fudanactuation-seed42-2x8192-5k` | 2296319 | `/workspace/.se3-training-state/whtws/20260928T151837Z` | 2026-09-28 23:18 启动，GPU 0–1 两卡 × 8192、5000 轮，在线 W&B `ir84bch8`；M41 为此在 1991 轮停止 |
| M43 | `SE3-WheelLegged-Rough-Exp-FudanActuationTracking` / `1361ba8` | `2026-09-28_15-34-48_rough-M43-fudantracking-seed42-2x8192-5k` | 2302488 | `/workspace/.se3-training-state/whtws/20260928T153441Z` | 2026-09-28 23:34 启动，GPU 2–3 两卡 × 8192、5000 轮，在线 W&B `6lp5bbmy`；与 M42 并行（启动器 `--allow-concurrent`） |
| M44 | `SE3-WheelLegged-Rough-Exp-FudanActuationTrackingStairYaw` / `d79e094` | `2026-09-28_16-03-49_rough-M44-stairyaw-seed42-2x8192-5k` | 2313226 | `/workspace/.se3-training-state/whtws/20260928T160342Z` | 2026-09-29 00:03 启动，GPU 4–5 两卡 × 8192、5000 轮，在线 W&B `nirk75yr`；与 M42/M43 并行。启动器 W&B 预检改为 Pod 内 12×10 s 重试（笔记本上游晚间抖动） |
| M45 | `SE3-WheelLegged-Rough-Exp-FudanActuationHeight` / `569aaf2` | `2026-09-29_05-06-36_rough-M45-fudanheight-seed42-7x8192-5k` | 2564245 | `/workspace/.se3-training-state/whtws/20260929T050629Z` | 2026-09-29 13:06 启动，七卡 × 8192、5000 轮，在线 W&B `gyoug52h`；M42/M43/M44 已跑满退出；2026-09-29 1365 轮按用户指令停止 |
| M46 | `SE3-WheelLegged-Rough-Exp-FudanActuationHeightSupport` / `35952c0` | `2026-09-29_06-45-20_rough-M46-stairsupport-from-m45-600-seed42-7x8192` | 2597707 | `/workspace/.se3-training-state/whtws/20260929T064514Z` | 2026-09-29 14:45 启动，七卡 × 8192，从 M45 model_600 完整续训到 5000（启动器 `-L m45-model600-src -K model_600.pt -i 4400`），在线 W&B `wzyqxzgz`；2026-09-29 1318 轮按用户指令停止 |
| M47 | `SE3-WheelLegged-Rough-Exp-ActionRate010NoBadOri` / `4398698` | `2026-09-29_07-59-37_rough-M47-actionrate010-nobadori-seed42-7x8192-5k` | 2623517 | `/workspace/.se3-training-state/whtws/20260929T075930Z` | 2026-09-29 15:59 启动，七卡 × 8192、5000 轮，在线 W&B `87ahmuea`；2026-09-29 2858 轮按用户指令停止 |
| M48 | `SE3-WheelLegged-Rough-Exp-ActionRate010NoBadOriMirror5` / `be494d1` | `2026-09-29_11-02-47_rough-M48-nobadori-mirror5-seed42-7x8192-5k` | 2683670 | `/workspace/.se3-training-state/whtws/20260929T110240Z` | 2026-09-29 19:02 启动，七卡 × 8192、5000 轮，在线 W&B `76yn2t4h`；2026-09-29 1354 轮按用户指令停止 |
| M49 | `SE3-WheelLegged-Rough-Exp-ActionRate010NoBadOriWheelDx50` / `d5db384` | `2026-09-29_12-22-37_rough-M49-wheeldx50-from-m48-1200-seed42-7x8192` | 2711263 | `/workspace/.se3-training-state/whtws/20260929T122230Z` | 2026-09-29 20:22 启动，七卡 × 8192，从 M48 model_1200 完整续训到 5000（`-L m48-src -K model_1200.pt -i 3800`），在线 W&B `fxig1m4z`；2026-09-29 4014 轮按用户指令停止 |
| M50 | `SE3-WheelLegged-Rough-Exp-M50` / `db1805a` | `2026-09-29_15-36-04_rough-M50-stairyaw-facing30-seed42-3x8192-5k` | 2776868 | `/workspace/.se3-training-state/whtws/20260929T153558Z` | 2026-09-29 23:36 启动，GPU 0–2 三卡 × 8192、5000 轮，从头训，在线 W&B `1aocaulm`；此前误启动的七卡 M50（`172gl519`，state `20260929T153416Z`）在第 6 轮停止 |
| M51 | `SE3-WheelLegged-Rough-Exp-M51` / `db1805a` | `2026-09-29_15-53-16_rough-M51-randomterrain-seed42-4x8192-5k` | 2783527 | `/workspace/.se3-training-state/whtws/20260929T155309Z` | 2026-09-29 23:53 启动，GPU 3–6 四卡 × 8192、5000 轮，从头训，在线 W&B `du0d16ft`；与 M50 并行 |
| M52 | `SE3-WheelLegged-Rough-Exp-M52` / `c32740e` | `2026-09-30_11-42-56_rough-M52-nokneespring-seed42-4x8192-5k` | 3175460 | `/workspace/.se3-training-state/whtws/20260930T114249Z` | 2026-09-30 19:42 启动，GPU 0–3 四卡 × 8192、5000 轮，从头训，在线 W&B `uccav6w7`；Pod 经 git bundle 由 db1805a 快进（子模块 e753ce6 → bdff426） |
| M53 | `SE3-WheelLegged-Rough-Exp-M53` / `fd4fd76` | `2026-09-30_14-14-15_rough-M53-kneeff300-tn08-seed42-3x8192-5k` | 3225418 | `/workspace/.se3-training-state/whtws/20260930T141408Z` | 2026-09-30 22:14 启动，GPU 4–6 三卡 × 8192、5000 轮，从头训，在线 W&B `eh5l9vma`；与 M52（GPU 0–3）并行（`--allow-concurrent`），Pod 经 git bundle 由 c32740e 快进；2026-09-30 4261 轮按用户指令 SIGINT 停止（最后 model_4200） |
| M54 | `SE3-WheelLegged-Rough-Exp-M54` / `01e87e2` | `2026-09-30_18-45-35_rough-M54-headinghold6-fa025-yaw15-seed42-7x8192-5k` | 3314582 | `/workspace/.se3-training-state/whtws/20260930T184528Z` | 2026-09-30 18:45 启动，七卡 × 8192、5000 轮，从头训，在线 W&B `vjmjllbj`；Pod 经 git bundle 由 fd4fd76 快进 |
| J1 | `SE3-WheelLegged-Jump-Mimic-MLP` / `8cb1dcb` | `2026-10-01_06-34-29_jump-J1-mimic-v1ref-seed42-7x8192-5k` | 3541217 | `/workspace/.se3-training-state/whtws/20261001T063422Z` | 2026-10-01 06:34 启动，七卡 × 8192、5000 轮，从头训，在线 W&B 项目 SE3-WheelLegged-Jump-Mimic `edy6opq0`；Pod 经 git bundle 由 01e87e2 快进；2026-10-01 2928 轮按用户指令 SIGINT 停止（最后 model_2800） |
| J2 | `SE3-WheelLegged-Jump-Mimic-Exp-J2` / `e0be3e4` | `2026-10-01_07-44-16_jump-J2-mimic-h012-seed42-7x8192-5k` | 3566109 | `/workspace/.se3-training-state/whtws/20261001T074409Z` | 2026-10-01 07:44 启动，七卡 × 8192、5000 轮，从头训，在线 W&B 项目 SE3-WheelLegged-Jump-Mimic `naqomk10`；Pod 经 git bundle 由 8cb1dcb 快进（子模块 bdff426 → 446e89f） |
| J5 | `SE3-WheelLegged-Jump-Flag-MLP` / `6cae3f4` | `2026-10-01_16-27-36_jump-J5-flag-clear3cm-h020-024-vx24-seed42-1x8192-5k` | 3737879 | `/workspace/.se3-training-state/whtws/20261001T162729Z` | 2026-10-01 16:27 启动，GPU 6 单卡 × 8192、5000 轮，从头训，W&B `szt5049c`（GPU 0–5 被他人占用，`--allow-concurrent`）；首版 `541b64b`（`jm8aiws2`，state `20261001T161908Z`）因离地判定漏洞停止。**单卡 run 对 SIGINT 不响应**（发出后继续训了 190 轮），要用 SIGTERM |
| J6 | `SE3-WheelLegged-Jump-Flag-Exp-J6` / `6cea122` | `2026-10-01_17-23-23_jump-J6-flag-takeoffmimic-seed42-1x8192-5k` | 3756423 | `/workspace/.se3-training-state/whtws/20261001T172315Z` | GPU 6 单卡，W&B `7b8g10zu`；约 994 轮 SIGTERM 停止（模仿核宽对快速起跳参考太窄，全部漏跳）。J5 在 1810 轮 SIGTERM 停止 |
| J7 | `SE3-WheelLegged-Jump-Mimic-Exp-J7` / `09493b2` | `2026-10-01_18-07-15_jump-J7-mimic-nocrouch-h022-seed42-1x8192-5k` | 3770619 | `/workspace/.se3-training-state/whtws/20261001T180708Z` | 2026-10-01 18:07 启动，GPU 6 单卡 × 8192、5000 轮，从头训，W&B `qar24d7n`；= J4 + 无下蹲参考（站姿 0.22） |
| J8 | `SE3-WheelLegged-Jump-Mimic-Exp-J8` / `cdc95b4` | `2026-10-02_03-56-43_jump-J8-mimic-norefobs-h022-seed42-1x8192-5k` | 3967075 | `/workspace/.se3-training-state/whtws/20261002T035636Z` | 2026-10-02 03:56 启动，GPU 6 单卡 × 8192、5000 轮，从头训，W&B `e9xuzhwh`；= J7 + actor 与 critic 都不看参考观测（34 维 POMDP）；子模块随之更新到 `120831e`；2026-10-02 511 轮 SIGTERM 停止（POMDP 学不出） |
| J9 | `SE3-WheelLegged-Jump-Mimic-Exp-J9` / `4c7ce37` | `2026-10-02_04-10-49_jump-J9-mimic-nocrouch-norsi-seed42-1x8192-5k` | 3972586 | `/workspace/.se3-training-state/whtws/20261002T041042Z` | 2026-10-02 04:10 启动，GPU 6 单卡 × 8192、5000 轮，从头训，W&B `mcr71gg5`；= J7 去掉 RSI |
| J10 | `SE3-WheelLegged-Jump-Mimic-Exp-J10` / `e3c3a0a` | `2026-10-02_07-53-50_jump-J10-mimic-phase015-norsi-seed42-1x8192-5k` | 4046465 | `/workspace/.se3-training-state/whtws/20261002T075343Z` | 2026-10-02 07:53 启动，GPU 6 单卡 × 8192、5000 轮，从头训，W&B `ve0ypvwi`；= J9 用一维相位代替参考帧（34 维）；子模块 `2422622`；2026-10-02 2660 轮 SIGTERM 停止 |
| RJ1 | `SE3-WheelLegged-Rough-Exp-RJ1` / `e3b58ab` | `2026-10-02_08-52-51_rough-RJ1-jump-flat30-seed42-7x8192-5k` | 4066270 | `/workspace/.se3-training-state/whtws/20261002T085244Z` | 2026-10-02 08:52 启动，七卡 × 8192、5000 轮，从头训，W&B 项目 SE3-WheelLegged-Rough `z5oc5n3t`；= M54 + J10 跳跃（平地列 30% 跳跃样本） |
| Rough-obs30 | `SE3-WheelLegged-Rough` / `5fc61d4` | `2026-10-03_03-42-56_rough-obs30-noattitude-seed42-7x8192-5k` | 243373 | `/workspace/.se3-training-state/whtws/20261003T034249Z` | 2026-10-03 03:42 启动，七卡 × 8192、5000 轮，从头训，W&B 项目 SE3-WheelLegged-Rough `tlfquma2`；= RJ1 默认配置 + actor 30 维（去掉 pitch / roll 指令与 wheel_pos_zero，pitch / roll 采样恒 0）；Pod 经 git bundle 由 e3b58ab 切到 xyh/1002（子模块 d8f76f5）。RJ1 已跑满 5000 轮；2026-10-03 3815 轮按用户指令停止（SIGINT 未退出，改 SIGTERM） |
| MGPU-A | `SE3-WheelLegged-Rough` / `635be5f` | `2026-10-03_08-01-17_rough-obs30-A-split-seed42-6x1365-5k` | 349241 | `/workspace/.se3-training-state/whtws/20261003T080110Z` | 2026-10-03 16:01 启动，GPU 0–5 六卡 × 1365（总 env 约 8190，与单卡 8192 同数据量）、5000 轮、seed 42、从头训，W&B `tnt1j1nw`；多卡改进实验 A（多卡提速），对照单卡 `b7y13zoi`。首轮 1.55 s/轮（单卡 3.35 s），Loss/grad_noise_scale_actor 已记录 |

仓库于 2026-09-27 通过 git bundle 由 5af0d00 快进到 `448c7e4`，2026-09-28 再快进到 `25ca875`（子模块仍 `e753ce6`）。
停止：`ps -o args= -g <pgid>` 核对 run name 后 `kill -INT -- -<pgid>`；容器 PID 1 不回收僵尸，defunct 条目可忽略。
