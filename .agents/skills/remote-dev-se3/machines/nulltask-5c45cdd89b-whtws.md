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

### 2026-10-12 启动、10-13完成：Student128 被动 armature=0 从零训练

- 用户确认：四被动关节armature .005→0；腿60/3、轮.2、动作.25/15，其余物理/奖励/DR配置不变。任务`SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128`。
- 代码`28412c67d8c411a89c3ba12dcf7b9e4be1aab1d3`（xyh/1005），子模块70bd413；增量bundle快进，启动前工作区干净。CPU单env及七卡×8192各5轮smoke通过。
- GPU0–6、8192env/卡（共57344）、seed42、从零5000轮、save200、resumeFalse；通过CLI将续训专用的学生预热设为0，PPO从第0轮更新。
- W&B：[zz634hy4](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/zz634hy4)。PID/PGID3290123；state`/workspace/.se3-training-state/whtws/20261013T011618Z`，日志train.log与verified_launch.json。
- Run：`logs/rsl_rl/SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128/2026-10-13_09-16-26_rough-cts-student128-passivearm0-scratch-seed42-7x8192-5k`。启动UTC2026-10-13 01:16:18（用户America/New_York为10-12 21:16:18）。
- 核验第24/5000轮、7workers、loss有限、无Traceback/nefc overflow，GPU约13.4–13.9GiB/89–93%，每轮约3.2s。model0导出再次确认KP/KD/action scale；启动健康不代表抖动已改善。
- 2026-10-13 01:59:26 UTC只读更新：749/5000，七worker/GPU89–92%，最近30轮约3.698s/轮，最新model600；无Traceback/OOM/nefc或非有限loss。700–749 reward107.50（551–600为143.75），正处热身向地形逐步切换，active88.1%→58.2%、stairs_up level1.80→3.30；学生vxRMSE.1121、latentMSE.00861。尚未做新策略抗噪评测，未改动训练。
- 2026-10-13完成核验：4999/5000、最终PT/ONNX在UTC06:36:49保存，06:39:18日志成功收尾，七卡0%/2MiB，原PGID只余defunct。最后50轮reward77.035、stairs_up6.577、heading8.805°、latentMSE.004846、vxRMSE.108182。79次NaN是grad_noise_scale_actor诊断量，实际优化loss有限。
- 最终ONNX已取回本机原logs层级并在同新模型上做两组失能起身仿真：相对旧6998，起身峰值35.58→25.47°、高度23.92→22.35cm；带噪腿5–15Hz最大RMS .00619→.02150°，无噪也.02011°，未显示减抖收益或实车8.3Hz复现。未部署实车、未继续训练。
- 本机启动器副本`.scratch/passive_armature0_20261012/launch.sh`，在原启动器上只补`--agent.algorithm.estimator-warmup-iterations 0`并缩短首次状态等待；没有修改个人通用启动器或实车。

| 标签 | 任务 / commit | run | PGID | state dir | 备注 |
|---|---|---|---|---|---|
| M37 | `SE3-WheelLegged-Rough-Exp-HeightWindowDz5Prune5` / `448c7e4` | `2026-09-27_14-18-53_rough-M37-dz5-prune5-seed42-7x8192-5k` | 1810096 | `/workspace/.se3-training-state/whtws/20260927T141846Z` | 2026-09-27 22:18 启动，七卡 × 8192、5000 轮、`WANDB_MODE=disabled`（无在线 W&B，看 TensorBoard）；2026-09-27 按用户指令在 2058 轮 SIGINT 停止（最后 checkpoint `model_2000`），结论见 m37_reward_prune_20260927。启动方式：stdin 脚本 `setsid nohup uv run --no-sync se3-train ...`，无 nulltask1 那套启动器 |

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
| JumpCol10 | `SE3-WheelLegged-Rough` / `6303547` | `2026-10-03_13-21-20_rough-obs30-jumpcol10-seed42-6x2048-5k` | 453799 | `/workspace/.se3-training-state/whtws/20261003T132113Z` | 2026-10-03 21:21 启动，GPU 0–5 六卡 × 2048、5000 轮、seed 42、从头训，W&B `06j74ep2`；= 默认 rough + 专用平地跳跃列（全部 env 10%）+ 跳跃触发 0.5 Hz。首轮：第 58 轮无 Traceback、1.85 s/轮，Jump/jump_env_rate 0.100，六卡 5.3 GB / 65–68% |
。2026-10-03 第 3005 轮按用户要求停止（SIGTERM 进程组） |
| Orient24 | `SE3-WheelLegged-Rough-Exp-Orient24` / `aeec423` | `rough-orient24-actor128-seed42-7x1170-5k` | 713671 | `/workspace/.se3-training-state/whtws/20261003T181306Z` | 2026-10-04 02:13 启动，GPU 0–6 七卡 × 1170、5000 轮、seed 42、从头训，W&B `3ix86na3`；= 默认 rough（actor 128/64/32、去同步优化、旧 RSI bug 已修）+ tracking_orientation_l2 −12 → −24。Pod 经 git bundle 由 6303547 快进。首轮：第 144 轮无 Traceback、1.22 s/轮，actor 首层 30→128，jump_env_rate 0.100，七卡 4.5 GB / 75–80%。W&B 网关日志自 10-03 15:57 停更（boring open 失败），但 38444 链路可用、认证通过。2026-10-04 跑满 5000 轮正常结束 |
| Base128 | `SE3-WheelLegged-Rough` / `aeec423` | `2026-10-04_06-15-43_rough-base-actor128-orient12-seed42-7x1170-5k` | 944346 | `/workspace/.se3-training-state/whtws/20261004T061536Z` | 2026-10-04 14:15 启动，GPU 0–6 七卡 × 1170、5000 轮、seed 42、从头训，W&B `8kjb4vm4`；Orient24 的同代码基线（默认入口：actor 128/64/32、tracking_orientation_l2 −12），与 Orient24 只差姿态权重。首轮：第 45 轮无 Traceback、1.22 s/轮，actor 首层 30→128，jump_env_rate 0.100，七卡 4.5 GB / 76–81%。2026-10-04 跑满 5000 轮正常结束（W&B 上传完毕后清理残留进程组） |
| HighStand01 | `SE3-WheelLegged-Rough-Exp-HighStand01` / `1cb5d77` | `2026-10-04_08-50-49_rough-highstand01-orient24-seed42-7x1170-5k` | 996527 | `/workspace/.se3-training-state/whtws/20261004T085041Z` | 2026-10-04 16:50 启动，GPU 0–6 七卡 × 1170、5000 轮、seed 42、从头训，W&B `0yeh4dow`；= 新默认（姿态罚 −24，`55bce6d`）+ high_stand_transition_prob 0.5 → 0.1，对照基线为 Orient24 `3ix86na3`。Pod 经 git bundle 由 aeec423 快进。首轮：第 41 轮无 Traceback、1.14 s/轮，高姿态序列 waiting 4.3% / moving 6.6% |
| TSDStraight | `SE3-WheelLegged-Rough` / `6656fb5` | `2026-10-04_16-30-05_rough-tsdstraight-seed42-7x1170-5k` | 1144178 | `/workspace/.se3-training-state/whtws/20261004T162958Z` | 2026-10-05 00:30 启动，GPU 0–6 七卡 × 1170、5000 轮、seed 42、从头训，W&B `ctw47krj`；= 新默认（姿态罚 −24、high_stand 0.1、跳跃 vx 不截断 `3919db7`）+ 二级下台阶直行指令（vx ±2.4、yaw 0、无静站 / 高姿起步、出生朝向正对台阶 ±15°）。相对 HighStand01 `0yeh4dow` 差两处（跳跃 vx 不截断、二级下台阶指令）。Pod 经 git bundle 由 1cb5d77 快进。首轮：第 41 轮无 Traceback、1.12–1.18 s/轮，七卡 4.5–5.0 GB / 75–87% |
| HeightSigma07 | `SE3-WheelLegged-Rough-Exp-HeightSigma07` / `f090dbe` | `2026-10-05_06-33-48_rough-hsigma07-dr1-seed42-4x1170-5k` | 1536691 | `/workspace/.se3-training-state/whtws/20261005T063341Z` | 2026-10-05 14:33 启动，GPU 0–3 四卡 × 1170（总 4680）、5000 轮、seed 42、从头训，W&B `jrvpex39`；= 当前默认（含 DR1）+ 机身高度罚 σ 0.10 → 0.07（全列）。与下一行同 commit 并行（`--allow-concurrent`），总 env 数相同。Pod 经 git bundle 由 6656fb5 快进。首轮：第 125 轮无 Traceback、1.17–1.22 s/轮，3.3–3.7 GB / 76–80% |
| Base-DR1 | `SE3-WheelLegged-Rough` / `f090dbe` | `2026-10-05_06-35-26_rough-base-dr1-sigma010-seed42-3x1560-5k` | 1538670 | `/workspace/.se3-training-state/whtws/20261005T063519Z` | 2026-10-05 14:35 启动，GPU 4–6 三卡 × 1560（总 4680）、5000 轮、seed 42、从头训，W&B `h7tyzjql`；HeightSigma07 的同代码基线（默认 σ 0.10）。首轮：第 45 轮无 Traceback、1.28–1.30 s/轮，3.4–3.8 GB / 79–82% |
| DR1-OracleVel | `SE3-WheelLegged-Rough-Exp-DR1-OracleVel` / `0dd79e0`（分支 `xyh/1005`） | `2026-10-05_17-41-12_rough-dr1-oraclevel-actor128-seed42-6x1170-5k` | 1887243 | `/workspace/.se3-training-state/whtws/20261005T174105Z` | 2026-10-06 01:41 启动，GPU 0–5 六卡 × 1170、5000 轮、seed 42、从头训，save-interval 100，W&B `x0xavu14`；= nulltask1 DR1 Oracle（`0khyuabz`，56b9de4）+ actor 机身线速度 3 维，不可部署。启动前实时确认七卡空闲、无 se3-train；主 checkout 由另一会话的 `xyh/1002@f090dbe`（其 run 均已结束，分支保留）切到 `xyh/1005`，子模块经 bundle 同步到 70bd413（se3-sim2x main），smoke 在 Pod 上 CPU 跑过。mjlab 训练脚本把 env seed 设为 agent seed + rank，故只传 `--agent.seed` 与 nulltask1 一致。首轮：第 40 轮无 Traceback、1.23 s/轮，六卡 4.1–4.5 GB / 68–91%。2026-10-06 跑满 5000 轮正常退出 |
| Vx | `SE3-WheelLegged-Rough-Exp-Vx` / `18c5062`（分支 `xyh/1005`） | `2026-10-06_01-43-27_rough-dr1-vx-actor128-seed42-6x1170-5k` | 2055285 | `/workspace/.se3-training-state/whtws/20261006T014320Z` | 2026-10-06 09:43 启动，GPU 0–5 六卡 × 1170、5000 轮、seed 42、从头训，save-interval 100，W&B `qnd9o87r`；= 当前默认 + actor 前向速度 vx 1 维（不加 DR 参数），不可部署。与 nulltask1 的 Vel（`fpuf21dr`）并行。主 checkout `xyh/1005` 由 0dd79e0 快进到 18c5062，启动前确认干净、无 se3-train，smoke 在 Pod 上跑过。首轮：第 36 轮无 Traceback。2026-10-06 用户判断已收敛，第 2504 轮停止（SIGINT 未退，SIGTERM 进程组），最后 model_2500 |
| VxObserver-WheelTN | `SE3-WheelLegged-Rough-Exp-VxObserver-WheelTN` / `4832d15`（分支 `xyh/1005`） | `2026-10-06_09-44-35_rough-dr1-vxobs-wheeltn-h16-actor128-seed42-7x1500-5k` | 2289123 | `/workspace/.se3-training-state/whtws/20261006T094428Z` | 2026-10-06 17:44（Pod 时区）启动，GPU 0–6 七卡 × 1500、5000 轮、seed 42、save-interval 100，W&B `xdk0dnoa`；= VxObserver + 轮子 T-N 包络换手册额定点口径。主 checkout 由 18c5062 快进到 4832d15，Pod 双卡 smoke 通过。W&B 走 38444。启动时笔记本有 3 个 boring 守护进程，9/25 的旧守护进程占着 38444，CLI 看不到、`boring open` 静默失败（网关自 10-03 起报错）；2026-10-06 17:52 停掉两个旧守护进程（13736、23032），只留 sock 对应的 44660，网关随即 `boring tunnel opened`，本 run W&B 断约 1 min 后自动续传。首轮：第 34 轮无 Traceback、1.34 s/轮，七卡 5.1 GB / 64–78% |
| VxObserver-Z3 | `SE3-WheelLegged-Rough-Exp-VxObserver-Z3` / `3879d15`（分支 `xyh/1005`） | `2026-10-07_01-02-23_rough-dr1-vxobserver-z3-h16-actor128-seed42-6x1170-5k` | 2464422 | `/workspace/.se3-training-state/whtws/20261006T170216Z` | 2026-10-07 01:02（Pod 时区）启动，GPU 0–5 六卡 × 1170、5000 轮、seed 42、save-interval 100，W&B `byw4hj29`；= VxObserver（16 帧）+ 估计器额外 3 维隐向量（不 detach、PPO 端到端训练，方案 A），对照 `agqj496q`。主 checkout 由 4832d15 快进到 3879d15，Pod 双卡 smoke 通过、启动前实时确认七卡空闲。首轮第 34 轮无 Traceback |
| VxObserver-6x8192 | `SE3-WheelLegged-Rough-Exp-VxObserver` / `05dc18f`（分支 `xyh/1005`） | `2026-10-07_11-37-15_rough-dr1-vxobserver-h16-wheeltn-seed42-6x8192-5k` | 2731155 | `/workspace/.se3-training-state/whtws/20261007T033707Z` | 2026-10-07 11:37（Pod 时区）启动，GPU 0–5 六卡 × 8192、5000 轮、seed 42、save-interval 100，W&B `gxse68wn`；= VxObserver（16 帧）+ 新默认轮子包络，作为 nulltask1 CTS（`n6femb3m`，同 commit、同规模）的对照。主 checkout 由 3879d15 快进到 05dc18f，Pod 双卡 × 8192 smoke 通过、启动前实时确认七卡空闲。首轮第 13 轮无 Traceback、3.01 s/轮，六卡 13.1 GB / 91–93% |

仓库于 2026-09-27 通过 git bundle 由 5af0d00 快进到 `448c7e4`，2026-09-28 再快进到 `25ca875`（子模块仍 `e753ce6`）。
停止：`ps -o args= -g <pgid>` 核对 run name 后 `kill -INT -- -<pgid>`；容器 PID 1 不回收僵尸，defunct 条目可忽略。

## 2026-10-10 Student128 低增益从零训练

- 用户指定：腿部 Kp=30、Kd=2，轮子 Kd=0.1，七卡、每卡 8192 envs。动作缩放保留腿 0.25 / 轮 15，Student128、噪声、DR、奖励和课程沿用当前配置。
- 任务：`SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-LowGains`；源码 `f11aae54cd9edc6fa78dd9236caccadcfc7aa0e9`，子模块 `70bd4139e2a4bf9dd79420b5d8f90d8d70d6ec73`。Pod 经 git bundle 从 `c40ab76` 快进，启动时工作区干净。
- 2026-10-10 06:30:16 UTC 启动；GPU 0–6，总 envs 57,344；seed 42、5000 轮、每 200 轮保存。`resume=False`，学生预热为 0，模型随机初始化。
- W&B：[bmxgwtd9](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/bmxgwtd9)。
- PID / PGID：`819182`；state：`/workspace/.se3-training-state/whtws/20261010T063016Z`；日志 `train.log`，启动核验 `verified_launch.json`。
- Run：`logs/rsl_rl/SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-LowGains/2026-10-10_14-30-23_rough-cts-stairvx08-student128-kp30-kd2-wkd01-scratch-seed42-7x8192-5k`。
- 验证：Ruff 与配置检查通过；Flat CPU 单 env 和新任务七卡 × 8192 的 5 轮 smoke 均通过。Smoke 日志在 `/workspace/.se3-training-state/whtws/smoke-f11aae5/`。正式 run 已核验第 0 → 18 轮，无 Traceback、NaN loss、nefc overflow；七卡显存约 13.4–13.9 GiB，利用率 90–93%，每轮约 3.15 s。`model_0.onnx` 的 KP/KD 与 action scale 均已回读确认。启动成功不代表抖动已改善，需训练后在相同噪声条件下复测。

本机实际启动命令（Git Bash；W&B 凭据由入口机 Secret 注入）：

```bash
bash C:/Users/13567/.local/bin/start-whtws-training.sh \
  -t SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-LowGains \
  -n 8192 -i 5000 -s 200 -g 0,1,2,3,4,5,6 \
  -r rough-cts-stairvx08-student128-kp30-kd2-wkd01-scratch-seed42-7x8192-5k \
  -p SE3-WheelLegged-Rough -S 42 -c f11aae5
```

## CTS 学生 CAPS 微调（2026-10-10）

- 用户指定仅训练 CTS 学生历史编码器。任务 `SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-LowGains-CAPS`，源码 `e6eaa47`（CAPS 实现 `45b15e4` + runner 冻结协作修复）。已推送 `origin/xyh/1005`，Pod 经 git bundle 快进到同一提交。
- 从低增益 Student128 的原始 `model_4999.pt` 续训，七卡 × 8192，追加 1000 轮，seed 42，每 100 轮保存。学生 Adam lr=1e-4；原 CTS 隐向量/vx MSE + 权重 1 的 CAPS 空间动作一致性项，不加时间平滑项。
- 教师、共享 actor、critic、动作分布参数及观测归一化统计冻结；保持 3:1 师生采样、腿 Kp/Kd=30/2、轮 Kd=0.1、动作缩放 0.25/15。同一干净历史生成两份任务配置噪声，actor 直接输入当前帧保持相同，仅经历史路径回传。
- W&B：https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/mw18r6xm 。PID/PGID `1090132`；state `/workspace/.se3-training-state/whtws/20261010T130304Z`；日志 `train.log`，检查 `verified_launch.json`。
- Run：`logs/rsl_rl/SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-LowGains-CAPS/2026-10-10_21-03-12_rough-cts-student128-caps-studentonly-from4999-lr1e4-lambda1-seed42-7x8192-1k`。
- 验证：Ruff、Flat CPU 单 env 5 轮、CAPS 七卡 × 8192 续训 5 轮通过。smoke 日志 `/workspace/.se3-training-state/whtws/smoke-caps-e6eaa47/`。逐项比较仅 6 个学生权重/偏置张量改变，其余网络和归一化 buffer 全部逐位一致；CAPS 单独反传仅学生有梯度；零噪声损失为 0；ONNX 保持 480→6。正式 `model_5000` 再次通过冻结核对。
- 启动时推进到 5019/5999，loss 有限、无 Traceback 或 nefc overflow；约 3 s/轮，七卡约 13.3–13.7 GiB。启动验证不是减抖效果结论，后续需对照原 checkpoint 在相同噪声下评估静站/行走高频抖动与跟踪误差。
- 本轮只做 CAPS 空间项的项目适配实验，不等同于原 CAPS 论文完整复现。记录在本机 `.scratch/cts_caps_20261010/`。

## 2026-10-14 占用记录

- 12:28 UTC 七卡被 `/root/var_experiments` 的 VAR 量化实验占用（每卡约 62 GB、利用率 31–99%，当时已跑约 2 h），WheelCompliance 改到 nulltask1 训练。
- 期间曾把 `48ce114` 经 bundle fetch 成本地分支 `exp/wheel-compliance` 并在其上跑过 CPU 单 env smoke，随后已切回 `xyh/1005@28412c6`，工作区干净；`exp/wheel-compliance` 分支保留。

## 2026-10-15 WheelCompliance-LowGains-WheelLag：轮力矩环内一阶滞后 DR

- 用户定：在 WheelCompliance-LowGains（9b1ca6d，nulltask1 1twu3qrj）上只加轮执行器输出一阶滞后，τ 5–15 ms 每 env reset 均匀采样（`se3_train.actuators.LaggedDcMotorActuator`）。起因：1twu3qrj 实车静站仍 6–7 Hz 振荡，原始陀螺对照排除陀螺平滑为主因。
- 任务 `SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-WheelCompliance-LowGains-WheelLag`，代码 `97c3f7a2153d…`（本机分支 `claude/angry-robinson-716e6e`）。bundle `48ce114..97c3f7a` fetch 到 `refs/heads/exp/wc-wheellag-97c3f7a`。
- **主 checkout 已切到 detached `97c3f7a`**（原为 `xyh/1005@28412c6`，记录在 `smoke-wheellag-97c3f7a/before_checkout.txt`）；训完后切回 `xyh/1005`。
- 启动前 W&B 网关又停了（gateway.log 停在 10-12、whtws 隧道 closed）：End 任务 → boring close/open → Run，`probe=400`、`gateway is ready` 后 dry-run 认证 OK。
- 冒烟：Pod CPU 与六卡 × 9557 各 5 轮通过（`/workspace/.se3-training-state/whtws/smoke-wheellag-97c3f7a/`）。
- 2026-10-15 08:21 UTC 启动：GPU 0–5 × 9557、seed 42、5000 轮、save 200。W&B [0bo7adcx](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/0bo7adcx)。PID/PGID `1211510`；state `/workspace/.se3-training-state/whtws/20261015T082132Z`；run `logs/rsl_rl/SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-WheelCompliance-LowGains-WheelLag/2026-10-15_16-21-39_rough-cts-student128-wc-kp30-wheellag5-15ms-scratch-seed42-6x9557-5k`。第 13 轮无 Traceback / nefc / NaN，3.6 s/轮；`model_0.onnx` KP 30、KD 2/0.1、提交 97c3f7a、dirty False。预计约 13:30 UTC 跑完。
- 2026-10-15 10:49 UTC 按用户选择停掉 WheelLag 0bo7adcx（停在第 2146 轮，checkpoint 到 model_2000），改跑观测延迟版。

## 2026-10-15 WheelLag-ObsDelay：轮力矩滞后 + 物理步级观测延迟

- 用户定：WheelLag 上加观测延迟 1–3 物理步（5–15 ms），IMU 与电机反馈两组各自每 env reset 采样，只作用于 actor（仿 scutrobotlab/wheeled-legged_RL V14）。
- 任务 `SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-WheelCompliance-LowGains-WheelLag-ObsDelay`，代码 `ffb42dfdbf8d`（本机分支 `claude/angry-robinson-716e6e`）；bundle `97c3f7a..ffb42df` → `refs/heads/exp/wc-obsdelay-ffb42df`。**主 checkout 现为 detached `ffb42df`**（最初是 `xyh/1005@28412c6`），训完切回。
- 冒烟：Pod CPU 与六卡 × 9557 各 5 轮通过（`smoke-obsdelay-ffb42df/`）。
- 10:51 UTC 启动：GPU 0–5 × 9557、seed 42、5000 轮、save 200。W&B [g2ii8nbq](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/g2ii8nbq)。PID/PGID `1330814`；state `/workspace/.se3-training-state/whtws/20261015T105143Z`；run `.../WheelLag-ObsDelay/2026-10-15_18-51-50_rough-cts-student128-wc-kp30-wheellag-obsdelay1-3-scratch-seed42-6x9557-5k`。第 12 轮无 Traceback / nefc / NaN，3.7 s/轮；model_0 KP 30、KD 2/0.1、提交 ffb42df、dirty False。预计约 16:10 UTC 跑完。

## 2026-10-17 Crawl-MLP v2：不转向 + 导轮承重罚 + 左右对称罚

- 用户定：原计划 nulltask1，但 16:47 UTC 起其六卡被他人的 Rough LowGains（`rough-stiffchain-b5b8d1e`）占用，改到本机七卡、每卡 1024 env。对照第一版 nulltask1 的 [5rt7ujl1](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Crawl/runs/5rt7ujl1)（10c78283）。
- 任务 `SE3-WheelLegged-Crawl-MLP`，代码 `eb2af755f524`（本机分支 `xyh/1005`）。主 checkout 仍为 `xyh/1005@28412c6`，未动。bundle `28412c67..eb2af755`（87 KB）→ `refs/heads/exp/crawl-eb2af75`；工作副本 `/workspace/se3-worktrees/crawl-eb2af75`（detached，`.venv` 软链主 checkout，uv.lock 相同）。
- 冒烟：本机 CPU、Pod CPU 与七卡 × 1024 各 5 轮通过（`/workspace/.se3-training-state/whtws/smoke-crawl-eb2af75/`），0.82 s/轮。
- 启动器 `-c` 与 `git rev-parse --short`（7 位）做前缀比对，必须传 7 位（传 8 位 `eb2af755` 报 HEAD 不符）。参数存在本机 `.scratch/crawl_train_20261017/whtws_v2_args.txt`。
- 16:55 UTC 启动：GPU 0–6 × 1024、seed 42、3500 轮、save 100、resume False。W&B [nnbqq8h0](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Crawl/runs/nnbqq8h0)。PID/PGID `2900059`；state `/workspace/.se3-training-state/whtws/20261017T165457Z`；run `/workspace/se3-worktrees/crawl-eb2af75/logs/rsl_rl/SE3-WheelLegged-Crawl-MLP/2026-10-18_00-55-04_crawl-mlp-v2-guidesupport-sym-noyaw-seed42-7x1024-3500`。
- 核验第 64 轮：无 Traceback / nefc，0.78 s/轮，七卡约 3.4 GiB / 66–69%。预计约 17:45 UTC 跑完。
- 17:27 UTC 按用户要求停掉 v2 nnbqq8h0（第 2067 轮，最后 checkpoint `model_2000.pt`；`kill -INT -- -2900059` 后显存归零，进程组只剩 4 个 `<defunct>` 僵尸，无需处理）。

## 2026-10-17 Crawl-MLP v3：速度课程导轮承重门控

- 代码 `7fec802b8245`（本机分支 `xyh/1005`），bundle `eb2af755..7fec802b`（5 KB）→ `refs/heads/exp/crawl-7fec802`；工作副本 `/workspace/se3-worktrees/crawl-7fec802`。冒烟 Pod CPU 与七卡 × 1024 各 5 轮通过（`smoke-crawl-7fec802/`）。
- 17:29 UTC 启动：GPU 0–6 × 1024、seed 42、3500 轮、save 100。W&B [9qjvkgb3](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Crawl/runs/9qjvkgb3)。PID/PGID `2936191`；state `/workspace/.se3-training-state/whtws/20261017T172912Z`；run `/workspace/se3-worktrees/crawl-7fec802/logs/rsl_rl/SE3-WheelLegged-Crawl-MLP/2026-10-18_01-29-19_crawl-mlp-v3-velgate-seed42-7x1024-3500`。参数 `.scratch/crawl_train_20261017/whtws_v3_args.txt`。
- 17:57 UTC 按用户要求停掉 v3 9qjvkgb3（第 1748 轮，最后 checkpoint `model_1700.pt`；进程组只剩 `<defunct>` 僵尸）。

## 2026-10-17 Crawl-MLP v4：动作零点改为趴姿 + 趴姿姿态罚

- 代码 `0bd5e680e9f7`（本机分支 `xyh/1005`），bundle `7fec802b..0bd5e680`（7 KB）→ `refs/heads/exp/crawl-0bd5e68`；工作副本 `/workspace/se3-worktrees/crawl-0bd5e68`。冒烟 Pod CPU 与七卡 × 1024 各 5 轮通过（`smoke-crawl-0bd5e68/`）。
- 17:57 UTC 启动：GPU 0–6 × 1024、seed 42、3500 轮、save 100。W&B [f96ibzjn](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Crawl/runs/f96ibzjn)。PID/PGID `2971215`；state `/workspace/.se3-training-state/whtws/20261017T175747Z`；run `/workspace/se3-worktrees/crawl-0bd5e68/logs/rsl_rl/SE3-WheelLegged-Crawl-MLP/2026-10-18_01-57-55_crawl-mlp-v4-crawldefault-pose-seed42-7x1024-3500`。参数 `.scratch/crawl_train_20261017/whtws_v4_args.txt`。
- 18:18 UTC 按用户要求停掉 v4 f96ibzjn（第 1066 轮，最后 checkpoint `model_1000.pt`）：actor 18.3 万参数上不了 STM32。

## 2026-10-17 Crawl-MLP v5：actor 换成 Rough 的 128/64/32

- 代码 `78cdd4bfb425`（本机分支 `xyh/1005`），bundle `0bd5e680..78cdd4bf` → `refs/heads/exp/crawl-78cdd4b`；工作副本 `/workspace/se3-worktrees/crawl-78cdd4b`。冒烟 Pod CPU 与七卡 × 1024 各 5 轮通过（`smoke-crawl-78cdd4b/`）。
- 18:19 UTC 启动：GPU 0–6 × 1024、seed 42、3500 轮、save 100。W&B [vkdux1bg](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Crawl/runs/vkdux1bg)。PID/PGID `2995495`；state `/workspace/.se3-training-state/whtws/20261017T181921Z`；run `/workspace/se3-worktrees/crawl-78cdd4b/logs/rsl_rl/SE3-WheelLegged-Crawl-MLP/2026-10-18_02-19-28_crawl-mlp-v5-actor128-seed42-7x1024-3500`。参数 `.scratch/crawl_train_20261017/whtws_v5_args.txt`。

## 2026-10-17 Crawl-Slope-MLP v1：上坡 0–0.15 + 机身底部凸包分段

- 用户定：v5 上车测试通过后做斜坡版，MDP 同 v5，地形只要上坡（倒金字塔 0–0.15），姿态项相对坡面。代码 `c1426bee7ddd`（本机分支 `xyh/1005`），bundle `78cdd4bf..c1426bee`（65 KB）→ `refs/heads/exp/crawlslope-c1426be`；工作副本 `/workspace/se3-worktrees/crawlslope-c1426be`。冒烟 Pod CPU 与七卡 × 1024 各 5 轮通过（`smoke-crawlslope-c1426be/`，1.07 s/轮，无接触溢出）。
- 19:29 UTC 启动：GPU 0–6 × 1024、seed 42、从零 3500 轮、save 100。W&B [e4zslnec](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Crawl/runs/e4zslnec)（与平地 crawl 同 project）。PID/PGID `3035020`；state `/workspace/.se3-training-state/whtws/20261017T192930Z`；run `/workspace/se3-worktrees/crawlslope-c1426be/logs/rsl_rl/SE3-WheelLegged-Crawl-Slope-MLP/2026-10-18_03-29-37_crawlslope-mlp-v1-up015-seed42-7x1024-3500`。参数 `.scratch/crawl_train_20261017/whtws_slope_args.txt`。

## 2026-10-18 Crawl-Slope-MLP v2：删导轮承重罚 + 可回退速度课程

- 用户定：斜坡 v1（e4zslnec）平地 +0.8/+1.5/−1.5 跟不上、速度课程锁在 1.5。v2 只在斜坡任务里删 `guide_wheel_support`，速度课程换回 `commands_vel_adaptive` 并开回退（推进 0.5 / 回退 0.35）。代码 `407fa81c6df6`（本机分支 `xyh/1005`），bundle `c1426bee..407fa81c` → `refs/heads/exp/crawlslope-407fa81`；工作副本 `/workspace/se3-worktrees/crawlslope-407fa81`。冒烟 Pod CPU 与七卡 × 1024 各 5 轮通过（`smoke-crawlslope-407fa81/`，1.06 s/轮）。
- 11:57 UTC 启动：GPU 0–6 × 1024、seed 42、从零 3500 轮、save 100。W&B [o5lggo94](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Crawl/runs/o5lggo94)。PID/PGID `3371608`；state `/workspace/.se3-training-state/whtws/20261018T115707Z`；run `/workspace/se3-worktrees/crawlslope-407fa81/logs/rsl_rl/SE3-WheelLegged-Crawl-Slope-MLP/2026-10-18_19-57-14_crawlslope-mlp-v2-nosupport-velretreat-seed42-7x1024-3500`。参数 `.scratch/crawl_train_20261017/whtws_slope2_args.txt`。

## 2026-10-19 Student128 从零训练（带 benchmark）

- 用户定：与 nulltask1 的 LowGains 同时从零训，六卡 × 8192（GPU 0–5，第 7 卡空）。代码 `098b39f`（本机分支 `xyh/rough-bench`，未合 main：benchmark + Student128 删 200 轮学生预热），bundle `8f25b91f..xyh/rough-bench` → `refs/heads/exp/rough-bench-098b39f`；工作副本 `/workspace/se3-worktrees/rough-bench-098b39f`（`.venv` 软链主 checkout，uv.lock 相同）。子模块 `submodules/se3-sim2x` 已检出 `7c3a759`（url 指向 `/workspace/se3-worktrees/.se3-sim2x-mirror`，副本不能放在 worktree 里，否则启动器判工作区不干净）。Pod 六卡 × 8192 smoke 5 轮通过（约 4.0 s/轮）。每 1000 轮与结束自动跑 full benchmark（结果在 run 目录 `onnx/bench/`，W&B `<run>-bench`）。
- 11:32 UTC 启动：任务 `SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128`，seed 42、5000 轮、save 100。W&B [cek08f79](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/cek08f79)。PID/PGID `3831751`；state `/workspace/.se3-training-state/whtws/20261019T113216Z`；run `/workspace/se3-worktrees/rough-bench-098b39f/logs/rsl_rl/SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128/2026-10-19_19-32-23_rough-cts-student128-scratch-bench-seed42-6x8192-5k`。首轮：第 16 轮无 Traceback、3.13 s/轮。参数 `.scratch/rough_bench_train_20261019/whtws_student128_args.txt`。

## 2026-10-20 倒地出生三列 + 平顺性约束课程 从零训练

- 用户定：现在就在 whtws 用新代码从零开一条，与 nulltask1 的 1a42x8jj（e5b388e，只在平地列倒地出生、无约束）对照。代码 `4e74e45`（本机分支 `xyh/1020`，未推送）相对 e5b388e 多两处：倒地出生扩到 obstacles + wave（出生高度按地面高度图修正，占全体约 7.5%）、倒地平顺性约束 + 收紧课程 s（侧躺/倒置 3 s 内起身率门控）。不是单变量对照。
- 开训前实时查：七卡 2 MiB / 0%，无训练进程。bundle `8f25b91f..xyh/1020` → `refs/heads/exp/rough-fallcon-4e74e45`；工作副本 `/workspace/se3-worktrees/rough-fallcon-4e74e45`（`.venv` 软链主 checkout，子模块 7c3a759 从 `.se3-sim2x-mirror` 检出）。六卡 × 8192 smoke 5 轮通过（3.23 s/轮，倒地出生 7.7%），日志 `/workspace/.se3-training-state/whtws/smoke-4e74e45/`。
- 12:51 UTC 启动：GPU 0–5 × 8192（第 7 卡空）、seed 42、5000 轮、save 100，每 1000 轮 full benchmark。W&B [b3odjuk6](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/b3odjuk6)。PID/PGID `235384`；state `/workspace/.se3-training-state/whtws/20261020T125145Z`。首轮：第 12 轮无 Traceback、3.23 s/轮，六卡 13.3–13.8 GB / 85–91%。参数 `.scratch/rough_fall_train_20261020/whtws_fallcon_args.txt`。判据：`Curriculum/fall_constraint/s` 能否收到 1；`Recovery/recovered_within_3s_{side,inverted}` 不低于 0.9；`Recovery/*_max_recovering` 的力度明显低于 1a42x8jj；benchmark 走路、台阶、跳跃不退化。

## 2026-10-21 倒地约束 CaT + 每步总奖励垫非负 从零训练

- 用户定：倒地约束从罚分改为 CaT（约束即终止）。代码 `0c644416`（本机分支 `xyh/1020`，未推送）相对 b3odjuk6（4e74e45）：删四项倒地罚分；新增 `fall_constraint` 概率终止（腿速 13.4 / 机身角速度 10 / 腿功率 400 W / 轮速 30，按 R1 参考标定，不扣 fall_penalty）；`total_reward_floor` 把每步总奖励垫到 ≥0（排在 fall_penalty 之前）；课程收紧 p_max 0→0.25，崩塌保护只在首次收紧后累计。另含 9a33227 的只紧不松课程规则（tgds7ff7 因保护 bug 作废），所以对 b3odjuk6 不是严格单变量。
- 开训前实测（`/workspace/se3-probe-20261021/fall_ledger_b3od_4999.json`，账本脚本 `.scratch/fall_cat_20261021/fall_ledger.py`）：b3odjuk6 model_4999 在 s=1 时倒地期每秒奖励 −110、站立 +14；倒置出生折扣回报 −89、侧躺 −35——不垫底时 CaT 终止是逃生通道。
- 开训前实时查：七卡 2 MiB / 0%。bundle `4e74e455..xyh/1020` → `refs/heads/exp/rough-cat-0c64441`；工作副本 `/workspace/se3-worktrees/rough-cat-0c64441`（子模块 7c3a759 从 `.se3-sim2x-mirror`）。六卡 × 8192 smoke 5 轮通过（3.3–3.6 s/轮）。启动器 `-c` 要传 7 位短哈希（按 `git rev-parse --short` 前缀比较，8 位会被判不符）。
- 14:00 UTC+8 启动：GPU 0–5 × 8192、seed 42、5000 轮、save 100，每 1000 轮 full benchmark。W&B [ff0nkdzb](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/ff0nkdzb)。PID/PGID `700287`；state `/workspace/.se3-training-state/whtws/20261021T055957Z`；run `logs/rsl_rl/SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128/2026-10-21_14-00-04_rough-cts-student128-fall3col-cat-rewfloor-scratch-bench-seed42-6x8192-5k`。启动器回连被 reset（exit 255），训练已起来、未重复启动。首轮：第 18 轮无 Traceback、3.28 s/轮，六卡 13.3–13.8 GB / 88–91%。参数 `.scratch/fall_cat_20261021/whtws_cat_args.txt`。判据：翻身能否在 p_max=0 阶段被发现（垫底后时间压力只剩约 14 分/s）；s / p_max 收到哪里；`FallConstraint/*` 越界率与终止率；eval_recovery_effort.py 实测峰值腿速 ≤13.4、角速度 ≤10、功率 ≤400 W；benchmark 不退化。
- ff0nkdzb 跑满 5000 轮正常退出。结论：门槛 0.85 下 s 全程为 0、CaT 从未生效；垫底把倒地期梯度抹平（所有倒地状态奖励都是 0），倒置 3 s 内起身率 1000 轮 0.03（b3odjuk6 0.65）、5000 轮 0.68（b3odjuk6 0.97），峰值腿速仍 20 rad/s。

## 2026-10-21 CaT + 垫底 + 起身进度奖励 + 门槛 0.6 从零训练

- 用户定：在 ff0nkdzb 基础上加两处。代码 `e06a29f2`（xyh/1020，未推送）：`fall_upright_progress`（(1 − g_z)²/4 × 20/s，只发倒地出生 episode，排在 total_reward_floor 之后不参与垫底）；收紧门槛 0.85 → 0.6。相对 ff0nkdzb 是两个变量。
- bundle `0c644416..xyh/1020` → `refs/heads/exp/rough-catprog-e06a29f`；工作副本 `/workspace/se3-worktrees/rough-catprog-e06a29f`。开训前实时查七卡 2 MiB / 0%；六卡 × 8192 smoke 5 轮通过（3.2 s/轮）。
- 22:54 UTC+8 启动：GPU 0–5 × 8192、seed 42、5000 轮、save 100，每 1000 轮 full benchmark。W&B [1nvtka54](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/1nvtka54)。PID/PGID `996029`；state `/workspace/.se3-training-state/whtws/20261021T145414Z`；run `logs/rsl_rl/SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128/2026-10-21_22-54-21_rough-cts-student128-fall3col-cat-rewfloor-progress-gate06-scratch-bench-seed42-6x8192-5k`。首轮：第 21 轮无 Traceback、3.23 s/轮，六卡 13.3–13.8 GB / 89–91%。参数 `.scratch/fall_cat_20261021/whtws_catprog_args.txt`。判据：1000 轮倒置 3 s 起身率接近 b3odjuk6（0.65）；s 能否离开 0 并收上去；收紧后 `FallConstraint/*` 越界率下降、`Recovery/*_max_recovering` 降到腿速 ≤13.4、角速度 ≤10；benchmark 不退化。
- 2026-10-22 按用户指令停止 1nvtka54（SIGTERM 进程组即退），停在 2818 轮，最后 model_2800。结论：进度奖励（平方形状）没帮上忙，倒置 3 s 起身率 2537 轮仍只有 0.03；用户决定倒地约束改走拉格朗日乘子。
- 2026-10-22 标定探针（`/workspace/se3-probe-20261022/`，脚本 `.scratch/fall_lagrange_20261022/`）：b3odjuk6 model_4999 的约束量分布（probe_b3od.json）与腿速按地形列拆分（legprobe_b3od.json）。

## 2026-10-22 Student128-StudentPPO smoke（未开训）

- 用户定的单变量对照：学生隐向量 z_S 不 detach、由 PPO 端到端训练（去掉隐向量 MSE，v̂x 仍由 vx MSE 监督），其余与 Student128 逐项相同；MSE 臂即 nulltask1 正在跑的 p0cdt5d9（dfde5296）。代码 `c8968b6d`（xyh/1020，未推送），相对 dfde5296 的 src 只多这一个提交；MSE 模式 CPU smoke 5 轮 553 个数值与改动前逐位相同。
- 开训前实时查七卡空闲。bundle `e06a29f2..xyh/1020` → `refs/heads/exp/rough-studentppo-c8968b6`；工作副本 `/workspace/se3-worktrees/rough-studentppo-c8968b6`（子模块 7c3a759 从 `.se3-sim2x-mirror`）。任务 `SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-StudentPPO` 六卡 × 8192 smoke 5 轮通过（无 Traceback、无 nefc 溢出、3.24–3.33 s/轮，多卡梯度平均不卡），日志 `/workspace/.se3-training-state/whtws/smoke-studentppo-c8968b6/`。用户要求先不开训；开训时启动器 `-c c8968b6`（7 位短哈希）、`-R` 指这个 worktree。
- 14:42 UTC+8 按用户指令开训：GPU 0–5 × 8192、seed 42、5000 轮、save 100，每 1000 轮 full benchmark，worktree `rough-studentppo-c8968b6`（开训前实时复查七卡空闲、HEAD=c8968b6d、工作区干净），对照 nulltask1 的 p0cdt5d9。W&B [bg8nkked](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/bg8nkked)。PID/PGID `1353969`；state `/workspace/.se3-training-state/whtws/20261022T064242Z`；run `logs/rsl_rl/SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-StudentPPO/2026-10-22_14-42-49_rough-cts-student128-studentppo-fall3col-lagrange4-scratch-bench-seed42-6x8192-5k`。首轮：第 29 轮无 Traceback、无 nefc 溢出、3.31 s/轮，六卡 13.5–14.0 GB / 86–91%。参数 `.scratch/student_ppo_20261022/whtws_studentppo_args.txt`。判据见 memory cts-student-encoder-mse-vs-ppo：同轮次只比学生路径（benchmark、sim2x yaw/vx/鲁棒、师生奖励差、IMU 偏差漂移、起身、隐向量探针）；另盯 `Loss/estimator_latent_std`（各向同性约 0.18，第 29 轮 0.057）是否塌缩、`estimator_vx_rmse` 是否被 PPO 梯度拖坏。
- 2026-10-22 15:1x 按用户指令停止 bg8nkked（SIGTERM 进程组即退），停在 608 轮，最后 model_600。原因：c8968b6d 带拉格朗日累计的 nan bug（第 400 轮 J 已出现 NaN，武装后 λ 会永久变 nan、约束被悄悄关掉），且原对照 p0cdt5d9 已停、换成带修复的 16jwmav8（704d362a，nulltask1）。

## 2026-10-22 Student128-StudentPPO（防 nan 版）从零训练

- 代码 `704d362a`（xyh/1020，未推送）= c8968b6d + 拉格朗日累计防 nan 修复，与 nulltask1 的 16jwmav8 同一提交、只差 StudentPPO 开关（单变量）。bundle `c8968b6d..xyh/1020` → `refs/heads/exp/rough-studentppo-704d362`；工作副本 `/workspace/se3-worktrees/rough-studentppo-704d362`（detached 在 704d362a）。六卡 × 8192 smoke 5 轮通过（3.22 s/轮，`FallLagrange/nonfinite_envs` 0）。
- 15:20 UTC+8 启动：GPU 0–5 × 8192、seed 42、5000 轮、save 100，每 1000 轮 full benchmark。W&B [brosdfzt](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/brosdfzt)。PID/PGID `1371047`；state `/workspace/.se3-training-state/whtws/20261022T072038Z`；run `logs/rsl_rl/SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-StudentPPO/2026-10-22_15-20-45_rough-cts-student128-studentppo-fall3col-lagrange4-nanfix-scratch-bench-seed42-6x8192-5k`。首轮：第 27 轮无 Traceback、无 nefc 溢出、3.28 s/轮，六卡 13.5–14.0 GB / 85–91%。参数 `.scratch/student_ppo_20261022/whtws_studentppo2_args.txt`。对照 16jwmav8，判据见 memory cts-student-encoder-mse-vs-ppo。
- 2026-10-22 按用户指令停止 brosdfzt（SIGTERM 进程组即退；whtws 状态目录没有 pgid 文件，要用启动输出里的 PGID），停在 2923 轮，最后 model_2900。约 2000 轮武装后同样崩溃（地形等级 5.8→0.08），根因同 16jwmav8。

## 2026-10-22 倒地拉格朗日约束 v2 + StudentPPO smoke（未开训）

- 代码 `777ebff3`（同 nulltask1 的 MSE 臂，只差 StudentPPO 开关）。bundle `da4ce5a8..xyh/1020` → `refs/heads/exp/rough-studentppo2-777ebff`；工作副本 `/workspace/se3-worktrees/rough-studentppo2-777ebff`。StudentPPO 六卡 × 8192 smoke 5 轮通过，日志 `/workspace/.se3-training-state/whtws/smoke-studentppo2-777ebff/`。开训时启动器 `-c 777ebff`、`-R` 指这个 worktree。
- 19:31 UTC+8 按用户指令开训：StudentPPO，GPU 0–5 × 8192、seed 42、5000 轮、save 100，每 1000 轮 full benchmark，worktree `rough-studentppo2-777ebff`（开训前实时复查七卡空闲、HEAD=777ebff3、工作区干净）。W&B [b4o38249](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/b4o38249)。PID/PGID `1504289`；state `/workspace/.se3-training-state/whtws/20261022T113116Z`；run `logs/rsl_rl/SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-StudentPPO/2026-10-22_19-31-23_rough-cts-student128-studentppo-fall3col-lagrange4v2-scratch-bench-seed42-6x8192-5k`。首轮：第 32 轮无 Traceback、无 nefc 溢出、3.34 s/轮，六卡 13.5–14.0 GB / 86–97%。参数 `.scratch/fall_lagrange_20261022/whtws_studentppo_v2_args.txt`。对照 nulltask1 的 2y0rh8j8。

## 2026-10-23 倒地起身脚本教师：开环探针与 FallTeacher smoke（未开训）

- 开环探针（不进训练代码）：`.scratch/fall_teacher_probe_20261023/teacher_probe.py` 在 worktree `rough-studentppo2-777ebff` 上跑（`PYTHONPATH` 指 worktree src），两轮共 10 组、每组单卡 16384 env、约 2 分钟，结果在 `/workspace/.se3-training-state/whtws/probe-teacher-20261023/`。开跑前实时查七卡空闲（b4o38249 已跑完）。
- 代码 `bf82021a`（xyh/1020，未推送）：新增 `SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacher`。bundle `777ebff3..xyh/1020` → `refs/heads/exp/rough-fallteacher-bf82021`；工作副本 `/workspace/se3-worktrees/rough-fallteacher-bf82021`（子模块从 `.se3-sim2x-mirror`）。六卡 × 8192 smoke 5 轮通过（无 Traceback、无 nefc 溢出，3.27–3.38 s/轮，倒地出生 7.7%、教师接管约 46%），日志 `/workspace/.se3-training-state/whtws/smoke-fallteacher-bf82021/`。开训时启动器 `-c bf82021`、`-R` 指这个 worktree。
- 13:00 UTC+8 按用户指令开训（不加对照）：FallTeacher，GPU 0–5 × 8192、seed 42、5000 轮、save 100，每 1000 轮 full benchmark，worktree `rough-fallteacher-bf82021`（开训前实时复查七卡空闲、HEAD=bf82021a、工作区干净）。W&B [kf3o74t7](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/kf3o74t7)。PID/PGID `1968403`；state `/workspace/.se3-training-state/whtws/20261023T045952Z`；run `logs/rsl_rl/SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacher/2026-10-23_13-00-00_rough-cts-student128-fallteacher-fall3col-nolagrange-scratch-bench-seed42-6x8192-5k`。首轮：第 14 轮无 Traceback、无 nefc 溢出、3.32 s/轮，六卡 13.4–13.8 GB / 72–92%。参数 `.scratch/fall_teacher_probe_20261023/whtws_fallteacher_args.txt`。起身看 `FallTeacher/{on,off}_recovered_within_8s_*`（教师交还时是蹲坐，3 s 口径偏低）。

## 2026-10-23 倒地教师 BC + 腿正功率终止 smoke（未开训）

- 代码 `051f1f49`（xyh/1020，未推送）：新增 `SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-PowerLimit`（M 内腿正功率连续 3 步 > 400 W 终止、单独罚 300），对照 nulltask1 的 FallTeacherBC m36nvar6。
- 开跑前实时查：GPU 0–5 是 kf3o74t7，GPU 6 空（2 MiB）。bundle `bf82021a..xyh/1020` → `refs/heads/exp/rough-powerlimit-051f1f4`；工作副本 `/workspace/se3-worktrees/rough-powerlimit-051f1f4`（子模块从 `.se3-sim2x-mirror`）。GPU 6 单卡 × 8192 smoke 5 轮通过（无 Traceback、无 nefc 溢出，3.2 s/轮，与 kf3o74t7 同时跑），日志 `/workspace/.se3-training-state/whtws/smoke-powerlimit-051f1f4/`。
- 随机初始化策略下：M 内当步超限约 3%；结束的 M 段里功率终止约 25%（接教师 21–25%、不接教师 44–51%），每轮功率终止 0.3–0.8 个 / 卡。开训时启动器 `-c 051f1f4`、`-R` 指这个 worktree。
- 15:20 UTC+8 按用户指令停 kf3o74t7（停在约 2175 轮，SIGINT 后进程全退、七卡回到 2 MiB），随即开训 PowerLimit：GPU 0–5 × 8192、seed 42、5000 轮、save 100，`-c 051f1f4`、`-R /workspace/se3-worktrees/rough-powerlimit-051f1f4`。W&B [k4re9m75](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/k4re9m75)；PID/PGID `2065551`；state `/workspace/.se3-training-state/whtws/20261023T072133Z`；run `logs/rsl_rl/SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-PowerLimit/2026-10-23_15-21-40_rough-cts-student128-fallteacherbc-powerlimit400w3-fall3col-nolagrange-scratch-bench-seed42-6x8192-5k`。第 9 轮无 Traceback、无 nefc 溢出。对照 nulltask1 的 m36nvar6（FallTeacherBC）。

## 2026-10-23 PowerLimit 去编码器 MLP 版 smoke（未开训）

- 代码 `b0b7c8de`（xyh/1020，未推送）：新增 `SE3-WheelLegged-Rough-Exp-StairVx08-FallTeacherBC-PowerLimit-MLP`——actor 换普通 MLP 128/64/32、单帧 30 维输入（env 去掉 vx_observer / cts），critic、奖励、终止、倒地教师与模仿损失与 PowerLimit（k4re9m75）逐项相同（配置差异核对 `.scratch/fall_teacher_probe_20261023/check_powerlimit_mlp_cfg.py`）。
- 开跑前实时查：GPU 0–5 是 k4re9m75，GPU 6 空。bundle `051f1f49..xyh/1020` → `refs/heads/exp/rough-powerlimitmlp-b0b7c8d`；工作副本 `/workspace/se3-worktrees/rough-powerlimitmlp-b0b7c8d`。GPU 6 单卡 × 8192 smoke 5 轮通过（无 Traceback、无 nefc 溢出，3.04–3.06 s/轮，PowerLimit 同条件 3.2 s/轮；ONNX 导出正常），日志 `/workspace/.se3-training-state/whtws/smoke-powerlimitmlp-b0b7c8d/`。开训时启动器 `-c b0b7c8d`、`-R` 指这个 worktree。

## 2026-10-23 PowerLimit600-MLP 开训

- k4re9m75 已跑满 5000 轮（All workers completed successfully），七卡空。
- 代码 `7d64ddd1`（xyh/1020，未推送）：env_cfg 增加 `fall_power_limit_w`，新入口 `SE3-WheelLegged-Rough-Exp-StairVx08-FallTeacherBC-PowerLimit600-MLP`（限值 600 W，其余与 PowerLimit-MLP 逐项相同，配置核对只差限值）。bundle `b0b7c8de..xyh/1020` → `refs/heads/exp/rough-powerlimit600mlp-7d64ddd`；工作副本 `/workspace/se3-worktrees/rough-powerlimit600mlp-7d64ddd`。
- 20:57 UTC+8 按用户指令开训：GPU 0–6 × 1170（每卡 8192/7，合计 8190）、seed 42、5000 轮、save 100，`-c 7d64ddd`。W&B [cpzyma4c](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/cpzyma4c)；PID/PGID `2295857`；state `/workspace/.se3-training-state/whtws/20261023T125750Z`；run `logs/rsl_rl/SE3-WheelLegged-Rough-Exp-StairVx08-FallTeacherBC-PowerLimit600-MLP/2026-10-23_20-57-57_rough-mlp-fallteacherbc-powerlimit600w3-fall3col-nolagrange-scratch-bench-seed42-7x1170-5k`。第 29 轮无 Traceback、无 nefc 溢出。对照 nulltask1 的 d37wocyd（400 W，6×1365）。
- 22:30 UTC+8 按用户指令停 cpzyma4c（停在第 2853 轮，SIGINT 后进程全退、七卡 2 MiB）。CTS 600 W 版改在 nulltask1 跑（tf2qsg6r）。

## 2026-10-23 PowerLimit600-Alpha800 smoke 与开训

- 代码 `8e49e9c1`（xyh/1020，未推送）：env_cfg 增加 `fall_teacher_alpha_iters`，新入口 `SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-PowerLimit600-Alpha800`（α 400 轮起退火、600 轮 0.45、800 轮 0；其余与 PowerLimit600 tf2qsg6r 逐项相同，配置核对只差 α 调度）。bundle `7d64ddd1..xyh/1020` → `refs/heads/exp/rough-pl600alpha800-8e49e9c`；工作副本 `/workspace/se3-worktrees/rough-pl600alpha800-8e49e9c`。
- 开跑前七卡 2 MiB、无 se3-train。六卡 × 8192 smoke 5 轮通过（无 Traceback、无 nefc 溢出，3.33 s/轮），日志 `/workspace/.se3-training-state/whtws/smoke-pl600alpha800-8e49e9c/`。
- 22:34 UTC+8 按用户指令开训：GPU 0–5 × 8192、seed 42、5000 轮、save 100，`-c 8e49e9c`。W&B [dqkkieq0](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/dqkkieq0)；PID/PGID `2379420`；state `/workspace/.se3-training-state/whtws/20261023T143356Z`；run `logs/rsl_rl/SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-PowerLimit600-Alpha800/2026-10-23_22-34-04_rough-cts-student128-fallteacherbc-powerlimit600w3-alpha400to800-fall3col-nolagrange-scratch-bench-seed42-6x8192-5k`。第 11 轮无报错。对照 nulltask1 的 tf2qsg6r（α 600–1900）。

## 2026-10-24 PersistBC 核对与 smoke（未开训）

- 代码 `e75312af`（xyh/1020，未推送）：新入口 `SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-PowerLimit600-Alpha800-PersistBC`（在 dqkkieq0 入口上：模仿损失不乘 α、α=0 后教师继续推进出目标但不执行、教师目标限幅开环 ±0.3 rad；critic 教师接管维只记 α>0 的执行，旧入口不变）。bundle `8e49e9c1..xyh/1020` → `refs/heads/exp/rough-persistbc-e75312a`；工作副本 `/workspace/se3-worktrees/rough-persistbc-e75312a`。
- 七卡空。CPU 核对（Pod CPU，`check_persist_bc.py`）通过：配置只差三项；α=0 时模仿损失照算；α=0.85 / 0 两种下执行动作与公式一致（α=0 逐位等于策略动作）、目标都在实测 ±0.3 rad 内；旧入口 α=0 教师不接管。六卡 × 8192 smoke 5 轮通过（3.50 s/轮、无 Traceback / nefc），日志 `/workspace/.se3-training-state/whtws/smoke-persistbc-e75312a/`。开训时启动器 `-c e75312a`、`-R` 指这个 worktree。
- 13:53 UTC+8 按用户指令开训 PersistBC：GPU 0–5 × 8192、seed 42、5000 轮、save 100，`-c e75312a`。W&B [2tnlfwoi](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/2tnlfwoi)；PID/PGID `2799652`；state `/workspace/.se3-training-state/whtws/20261024T055303Z`；run `logs/rsl_rl/SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-PowerLimit600-Alpha800-PersistBC/2026-10-24_13-53-10_rough-cts-student128-fallteacherbc-pl600-alpha800-persistbc-leash03-fall3col-scratch-bench-seed42-6x8192-5k`。第 10 轮无报错。对照 dqkkieq0。

## 2026-10-24 recover 组入口核对与 smoke（未开训）

- 代码 `d3a8e86`（xyh/1020，未推送；062be2fb 新增 recover 组 + d3a8e86 站稳 / 截止按步数比较）：入口 `SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-Alpha800-Recover`。bundle `e75312af..xyh/1020` → `refs/heads/exp/rough-recover-d3a8e86`；工作副本 `/workspace/se3-worktrees/rough-recover-d3a8e86`（另有中间版 rough-recover-062be2f 可删）。
- 逻辑核对改在 GPU 6 上跑（用户指出不必用 CPU：核对是同一状态上实现 vs 参考，不依赖物理确定性）：配置差异只在预期项；dqkkieq0 model_4999 执行 620 步，站稳 / 失败 / 结束判定与独立参考逐步一致，recover 组总奖励 = recover 各项、走路组 recover 项为 0，±20 一次性奖惩、零速指令、按组倒地出生、critic 三维都对（站稳 28 次、正常结束 14 次；又倒路径未覆盖）。GPU 6 单卡 × 8192 smoke 5 轮通过（3.27 s/轮，倒地出生 10.0%），日志 `/workspace/.se3-training-state/whtws/smoke-recover-d3a8e86/`。
- 17:59 UTC+8 按用户指令停 2tnlfwoi（PersistBC，停在第 3717 轮，SIGINT 后进程全退、七卡 2 MiB），随即开训 Recover：GPU 0–5 × 8192、seed 42、5000 轮、save 100，`-c d3a8e86`。W&B [gyi9exiz](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/gyi9exiz)；PID/PGID `2955353`；state `/workspace/.se3-training-state/whtws/20261024T095924Z`；run `logs/rsl_rl/SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-Alpha800-Recover/2026-10-24_17-59-32_rough-cts-student128-fallteacherbc-alpha800-recovergroup-scratch-bench-seed42-6x8192-5k`。第 10 轮无报错。对照 dqkkieq0。
- 按用户指令停 gyi9exiz（Recover，停在第 3624 轮，最后存档 model_3600.pt；SIGINT 后进程全退、七卡 2 MiB）。用户认可的起身版本是 model_2200；转向改看 nulltask1 的 Recover-YawGate q59ces4z。

## 2026-10-24 Recover-AMP 数据集、核对与 smoke（未开训）

- 代码 `a2ab4b52`（xyh/1020，未推送）：从 867c7299 之前恢复 AMP（判别器多卡同步、归一化冻结在专家统计），新入口 `SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-Recover-AMP` = Recover（gyi9exiz）去掉倒地教师与模仿损失 + recover 组 AMP。bundle `d3a8e86..xyh/1020` → `refs/heads/exp/rough-recoveramp-a2ab4b5`；工作副本 `/workspace/se3-worktrees/rough-recoveramp-a2ab4b5`。
- 参考数据集：`scripts/record_recover_amp_dataset.py` 用 gyi9exiz model_2200（sha256 `85a8abab…`）在 GPU 6 录，recover 组 1401 env 成功 0.991，取 1024 段 / 94348 帧（出生 → 站稳满 1 s，段长中位 1.48 s）。`assets/amp/` 在 .gitignore 里，数据集不进 git：原件 `/workspace/.se3-training-state/whtws/recover-amp-a2ab4b5/recover.pkl`（sha256 `db7c051e…`），已拷到工作副本 `assets/amp/recover_gyi9exiz_model2200/recover.pkl`；换机器 / 新 worktree 要手动拷。
- GPU 6 逻辑核对（`.scratch/amp_recover_20261024/check_recover_amp.py`）全过：amp 观测组 vs 独立参考 3.8e-6、掩码 = recover 组、非 recover 组风格奖励恒 0、gyi9exiz 在本入口的起身帧与数据集最大均值差 0.04σ、判别器分得开、存档往返一致。
- 六卡 × 8192 smoke 5 轮通过（3.3–3.7 s/轮、无 Traceback / nefc、判别器多卡同步不卡、ONNX 导出正常），日志 `/workspace/.se3-training-state/whtws/smoke-recoveramp-a2ab4b5/`。开训时启动器 `-c a2ab4b5`、`-R` 指这个 worktree。
- 2026-10-25 01:12 UTC+8 按用户指令开训 Recover-AMP：GPU 0–5 × 8192、seed 42、5000 轮、save 100，`-c a2ab4b5`、`-R /workspace/se3-worktrees/rough-recoveramp-a2ab4b5`（开训前实时复查七卡 2 MiB、无活着的 se3-train、HEAD=a2ab4b5、工作区干净、数据集在位）。W&B [6iwgpa2m](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/6iwgpa2m)；PID/PGID `3172654`；state `/workspace/.se3-training-state/whtws/20261024T171224Z`；run `logs/rsl_rl/SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-Recover-AMP/2026-10-25_01-12-32_rough-cts-student128-recovergroup-amp-gyi2200-scratch-bench-seed42-6x8192-5k`。第 10 轮无 Traceback / nefc。参数 `.scratch/amp_recover_20261024/whtws_recoveramp_args.txt`。对照 gyi9exiz（教师 + 模仿损失）。看 `Recover/*`、`Loss/amp/*`（expert_score − policy_score 是否收敛到小差距、style_reward），按姿态评测用 `policy_recovery_eval.py`。

## 2026-10-25 Recover-AMP-Distill 核对与 smoke（未开训）

- 代码 `69c9e0f9`（xyh/1020，未推送）：Se3PPO 支持网络老师在线蒸馏（冻结同结构老师按部署路径前向、只在掩码组样本上算 Huber，挂 symmetry 钩子）；新入口 `SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-Recover-AMP-Distill` = Recover-AMP + recover 组蒸馏 gyi9exiz model_2200（系数 0.2，全程保持），配置核对 env 逐项相同、rl 只多 policy_distill。bundle `a2ab4b52..xyh/1020` → `refs/heads/exp/rough-recoverampdistill-69c9e0f`；工作副本 `/workspace/se3-worktrees/rough-recoverampdistill-69c9e0f`，`assets/amp/recover_gyi9exiz_model2200/` 下已放数据集与老师 `gyi9exiz_model_2200.pt`（gitignore，换 worktree 要拷）。
- 续训源：6iwgpa2m model_1100 已拷到 `logs/rsl_rl/<Distill 任务>/6iwgpa2m-model1100-src/`（开训时用启动器 `-L 6iwgpa2m-model1100-src -K model_1100.pt`；换更新的 checkpoint 要另拷）。
- GPU 6 逻辑核对（`.scratch/amp_recover_20261024/check_distill.py`）全过：老师与 model_2200 actor 逐位一致、冻结且不在优化器、部署路径动作与 gyi9exiz 推理策略差 0、掩码全 0 时蒸馏项 0、老师无梯度、续训恢复判别器（更新计数 1101）、update 后 Loss/imitation 0.109 且老师不变。
- GPU 6 单卡 × 8192 续训 smoke 通过（1100→1105 轮，3.6 s/轮，无 Traceback / nefc，imitation 0.079 → 0.039），日志 `/workspace/.se3-training-state/whtws/recover-ampdistill-69c9e0f/smoke.log`。GPU 0–5 当时在跑 6iwgpa2m，未做六卡 smoke（蒸馏项不引入额外跨卡通信，梯度走 PPO 原有平均）。
- 2026-10-25 按用户指令停 6iwgpa2m（Recover-AMP，停在约 1967 轮，最后存档 model_1900.pt；SIGINT 后进程全退、七卡 2 MiB）。结论：纯 AMP 走路全面领先 gyi9exiz，但倒置起身到 1900 轮仍约 0、侧躺 0.65；改走 nulltask1 的在线蒸馏 mb6sjulr。

## 2026-10-25 代码整理后 smoke（b108b75f）

- 代码 `b108b75f`（xyh/1020，未推送）：删拉格朗日 / 功率终止 / PersistBC / StudentPPO / yaw_gate / recovery_mimic 与 9 个实验入口，env 默认 recover 组 + AMP，Student128 即最终配方。bundle `69c9e0f9..xyh/1020` → `refs/heads/exp/rough-cleanup-b108b75`；工作副本 `/workspace/se3-worktrees/rough-cleanup-b108b75`，`assets/amp/recover_gyi9exiz_model2200/` 已拷数据集与老师（gitignore）。
- smoke 全过（无 Traceback / nefc，ONNX 导出正常），日志 `/workspace/.se3-training-state/whtws/smoke-cleanup-b108b75/`：Student128 六卡 × 8192（3.4 s/轮，老师载入、掩码组 recover_mask，imitation ≈ 0.10，AMP 风格奖励 0.67）；Recover、Recover-AMP、基础 Rough 各单卡 × 8192（3.0–3.6 s/轮）。

## 2026-10-25 四臂消融（whtws 两臂）

- 代码 `fbcf241f`（xyh/1020，未推送）= b108b75f 整理 + 两个消融入口（−AMP、−recover 组；起身掩码组在没有 recover 组时改用 M）。bundle `b108b75f..xyh/1020` → `refs/heads/exp/rough-ablate-fbcf241`；工作副本 `/workspace/se3-worktrees/rough-ablate-fbcf241`（数据集与老师已拷，sha256 同前）。GPU 6 核对 `.scratch/cleanup_20261025/check_no_recover_mask.py` 全过（掩码每步与 M 逐位相等、M 外风格奖励 0、蒸馏只来自 M），两臂 3 卡 × 16384 smoke 通过（并行 5.5 s/轮）。
- 20:43 UTC+8 按用户指令开训（各从零、seed 42、5000 轮、save 100、每卡 16384，四臂同时跑）：
  - −AMP：`…-Student128-Ablate-NoAMP`，GPU 0–2，W&B [n36if5d6](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/n36if5d6)，PID/PGID `3582442`，state `/workspace/.se3-training-state/whtws/20261025T124321Z`。
  - −recover 组：`…-Student128-Ablate-NoRecoverGroup`，GPU 3–5（--allow-concurrent），W&B [vsjweqne](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/vsjweqne)，PID/PGID `3584364`，state `/workspace/.se3-training-state/whtws/20261025T124456Z`。
  - 参数 `.scratch/cleanup_20261025/whtws_{noamp,norecover}_args.txt`；对照 nulltask1 的完整 ajb8j1ol 与 −蒸馏 2j6xrn05。

## 2026-10-27 Recover-Impact：低撞击起身老师

- 代码 `abb19c7f`（xyh/1020，未推送）= gyi9exiz 配方（d3a8e86）+ recover 组撞击罚：机身与四根腿杆对地接触力超过 1.5 倍体重（≈187 N）的部分，权重 −0.15 / (N·s)，只在 recover 组。配置核对与 gyi9exiz 只差这一项。子模块指针改到 `711257f`（xyh/1026-sim2x，sim2x 默认行为不变）。
- bundle `fbcf241f..xyh/1020` → `refs/heads/exp/rough-impact-abb19c7`；子模块 bundle `7c3a759..xyh/1026-sim2x` 进 `/workspace/se3-worktrees/.se3-sim2x-mirror`。工作副本 `/workspace/se3-worktrees/rough-impact-abb19c7`（子模块初始化要 `git -c protocol.file.allow=always submodule update --init`；数据集与老师已拷）。中间版 `rough-impact-wip`（fbcf241 + 补丁，src 与 abb19c7f 逐字相同）可删。
- 撞击罚标定：gyi9exiz model_2200 在本入口回放，倒置每局撞击罚中位 −8.1（超阈冲量 53.7 N·s）、侧躺 −2.8，走路组该项恒 0（`impact_ledger.py`）。GPU 3 单卡 × 1024 smoke 5 轮通过。
- 15:21 UTC+8 按用户指令开训：GPU 0–5 × 8192、seed 42、5000 轮、save 100、从零，`-c abb19c7`、`-R /workspace/se3-worktrees/rough-impact-abb19c7`（开训前实时复查七卡 2 MiB、无 se3-train）。W&B [m6rxc7v9](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/m6rxc7v9)；PID/PGID `439951`；state `/workspace/.se3-training-state/whtws/20261027T072146Z`；run `logs/rsl_rl/SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-Alpha800-Recover-Impact/2026-10-27_15-21-53_rough-cts-student128-fallteacherbc-alpha800-recovergroup-impact015-scratch-bench-seed42-6x8192-5k`。第 13 轮无 Traceback / nefc，3.4 s/轮，六卡 13.4–13.8 GB。参数 `.scratch/fall_teacher_probe_20261023/whtws_recoverimpact_args.txt`。对照 gyi9exiz；选 checkpoint 用 `policy_recovery_eval_impact.py`（机身峰值力、>3W 步数、落地速度）+ `Recover/*` 起身率。
- 2026-10-27 按用户指令停 m6rxc7v9（第 3264 轮，最后存档 model_3200；SIGINT 后七卡 2 MiB）。老师 800 轮退场后倒置起身率掉到 0、侧躺 0.19：γ=0.99 下 8 s 截止的失败罚折扣后只剩约 −0.4，翻身的撞击罚当场扣，策略选择不翻。用户判断撞击来自教师规定死的轨迹，改走下面的随机方向教师。

## 2026-10-27 Recover-RandDir：6 月随机方向倒地教师

- 代码 `bd159342`（xyh/1020，未推送）：`FallTeacherCfg.june_random()`——只接倒置（倾角 > 135°）、每局随机 ±1 方向、π rad/s 扫到 2π、倾角 < 37° 后前杆停住、连续 20 步即交还、可重接；入口 `…-Student128-FallTeacherBC-Alpha800-Recover-RandDir` = gyi9exiz 入口只换教师，不加撞击罚。bundle `abb19c7f..xyh/1020` → `refs/heads/exp/rough-randdir-bd15934`；工作副本 `/workspace/se3-worktrees/rough-randdir-bd15934`（子模块 711257f，数据集与老师已拷）。
- GPU 6 核对（`.scratch/fall_teacher_probe_20261023/check_randdir.py`，日志 `/workspace/se3-evals/randdir_20261027/check.log`）全过：配置只差 7 个教师字段；Recover 入口新代码与改动前（abb19c7f）影子教师 600 步逐位一致；开环 α=1、策略恒 0、只倒置：方向 +1 占 0.508、一局内不变、左右前杆物理同向，翻正交还率 +1 0.89 / −1 0.90（中位 1.8 / 2.3 s）。六卡 × 8192 smoke 5 轮通过（3.35 s/轮），日志 `/workspace/.se3-training-state/whtws/smoke-randdir-bd15934/`。
- 20:04 UTC+8 按用户指令开训：GPU 0–5 × 8192、seed 42、5000 轮、save 100、从零，`-c bd15934`、`-R /workspace/se3-worktrees/rough-randdir-bd15934`。W&B [3zqym1vn](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/3zqym1vn)；PID/PGID `607707`；state `/workspace/.se3-training-state/whtws/20261027T120449Z`；run `logs/rsl_rl/SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-Alpha800-Recover-RandDir/2026-10-27_20-04-56_rough-cts-student128-fallteacherbc-alpha800-recovergroup-randdir-june-scratch-bench-seed42-6x8192-5k`。第 10 轮无 Traceback / nefc。参数 `.scratch/fall_teacher_probe_20261023/whtws_recoverranddir_args.txt`。对照 gyi9exiz；看 `Recover/*`、`Recovery/recovered_rate_{inverted,side}`、`FallTeacher/dir_{pos,neg}_*`，挑 checkpoint 用 `policy_recovery_eval_impact.py`（机身峰值力对比 gyi9exiz 1250 N）。
- 撞击评测（GPU 6，`/workspace/se3-evals/impact_20261027/randdir.json`）：model_2300 倒置起身 0.99，机身峰值力中位 1021 N / p90 1816 N，98% 超 3 倍体重，下落速度 1.19 m/s（gyi9exiz 2200：1250 / 2703 N、98.7%、1.74 m/s）——轻一些，但仍砸地。2026-10-27 按用户指令停 3zqym1vn（第 3180 轮，最后存档 model_3100；SIGINT 后七卡 2 MiB）。

## 2026-10-27 Recover-RandDir-FreeFold-Impact：夹角归策略 + 撞击罚

- 代码 `437a66a0`（xyh/1020，未推送）：`FallTeacherCfg.fold='policy'`——教师只给前杆转动，后杆目标 = 策略后杆 + 跟转量（k=−1），执行夹角恰为策略夹角；模仿损失加 `back_follow`，后杆只监督跟转量；recover 组撞击罚权重按折扣口径取 −0.05（`ROUGH_RECOVER_IMPACT_WEIGHT_DISCOUNTED`）。相对 RandDir 两处改动（用户定）。bundle `bd159342..xyh/1020` → `refs/heads/exp/rough-freefold-437a66a`；工作副本 `/workspace/se3-worktrees/rough-freefold-437a66a`。
- GPU 6 核对（`.scratch/fall_teacher_probe_20261023/check_freefold.py`，日志 `/workspace/se3-evals/freefold_20261027/check{,2}.log`）全过：配置只差 fold / 撞击罚 / back_follow；Recover 与 RandDir 入口新代码与 bd159342 影子 600 步逐位一致（含执行动作，reset 步的 env 不比）；夹角逐步等于策略夹角、前杆等于教师；back_follow 单元检查过。**但夹角放开后教师开环（α=1）倒置翻正率只有 0.03–0.06（策略恒 0）/ 0.04–0.12（随机），收夹角时是 0.89–0.90**——翻身要靠策略自己学会收腿，用户知情后仍定开训。六卡 × 8192 smoke 5 轮通过（3.36 s/轮），日志 `/workspace/.se3-training-state/whtws/smoke-freefold-437a66a/`。
- 23:34 UTC+8 按用户指令开训：GPU 0–5 × 8192、seed 42、5000 轮、save 100、从零，`-c 437a66a`、`-R /workspace/se3-worktrees/rough-freefold-437a66a`。W&B [6hog7h75](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/6hog7h75)；PID/PGID `752348`；state `/workspace/.se3-training-state/whtws/20261027T153447Z`；run `logs/rsl_rl/SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-Alpha800-Recover-RandDir-FreeFold-Impact/2026-10-27_23-34-55_rough-cts-student128-fallteacherbc-alpha800-recovergroup-randdir-freefold-impact005-scratch-bench-seed42-6x8192-5k`。第 9 轮无 Traceback / nefc。参数 `.scratch/fall_teacher_probe_20261023/whtws_freefold_args.txt`。对照 3zqym1vn；先看 800 轮前 `Recovery/recovered_rate_inverted` 能否涨起来（策略要学会收夹角），再用 `policy_recovery_eval_impact.py` 比机身峰值力。
- 6hog7h75 跑满 5000 轮：倒置起身 2800 轮前 ≈0，2900–4400 轮涨到 0.6–0.8（4200 最好，站稳率 0.996），4450 轮后掉回约 0.15；侧躺前期被撞击罚压到 0.22（3zqym1vn 同期 0.48），后期 0.8–0.9。用户在 sim2x 看 model_4200 嫌动作怪：夹角放开后教师演示不出翻身，策略自己摸出的翻法不像样且不稳。

## 2026-10-28 Recover-RandDir-Impact：收夹角教师 + 撞击罚

- 代码 `f7b16577`（xyh/1020，未推送）：入口 `…-Recover-RandDir-Impact` = RandDir（3zqym1vn）只加 recover 组撞击罚 −0.05，教师不变（收夹角）。配置核对（`.scratch/fall_teacher_probe_20261023/check_randdir_impact.py`）只差撞击罚一项。bundle `437a66a0..xyh/1020` → `refs/heads/exp/rough-rdimpact-f7b1657`；工作副本 `/workspace/se3-worktrees/rough-rdimpact-f7b1657`。六卡 × 8192 smoke 5 轮通过（3.34 s/轮），日志 `/workspace/.se3-training-state/whtws/smoke-rdimpact-f7b1657/`。
- 13:08 UTC+8 按用户指令开训：GPU 0–5 × 8192、seed 42、5000 轮、save 100、从零，`-c f7b1657`、`-R /workspace/se3-worktrees/rough-rdimpact-f7b1657`。W&B [hiphaauf](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/hiphaauf)；PID/PGID `1135722`；state `/workspace/.se3-training-state/whtws/20261028T050844Z`；run `logs/rsl_rl/SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-Alpha800-Recover-RandDir-Impact/2026-10-28_13-08-51_rough-cts-student128-fallteacherbc-alpha800-recovergroup-randdir-impact005-scratch-bench-seed42-6x8192-5k`。启动时 W&B 初始化慢，约 2 min 后才出第 0 轮。参数 `.scratch/fall_teacher_probe_20261023/whtws_rdimpact_args.txt`。对照 3zqym1vn：起身率保不保得住、砸地是否减轻（用户看 sim2x 判断）。
- hiphaauf：W&B 只到第 195 轮（笔记本 USB 网卡 WLAN 2 无线电被关，W&B 网关断外网；入口机 SSH 2222 又间歇超时约 15 min），Pod 训练照常。Pod 日志：老师退场后 800–950 轮倒置起身 0.82–0.86、侧躺 0.72–0.77，没躺平；用户在 sim2x 看 model_900 嫌动作变怪——腿杆撬地被罚、轮子不罚，冲击转到轮子上。2026-10-28 按用户指令停 hiphaauf（第 1337 轮，最后存档 model_1300；SIGINT 后七卡 2 MiB）。

## 2026-10-28 Recover-RandDir-Impact-BaseWheel：撞击罚只计机身 + 两轮

- 代码 `b65fac2b`（xyh/1020，未推送）：env_cfg 加 `recover_impact_sensors`，新入口 `…-Recover-RandDir-Impact-BaseWheel` = RandDir-Impact 只把撞击罚传感器从（机身、腿杆）换成（机身 `collision_sensor`、两轮 `wheel_sensor`），阈值每部位 1.5 倍体重、权重 −0.05 不变。bundle `f7b16577..xyh/1020` → `refs/heads/exp/rough-basewheel-b65fac2`；工作副本 `/workspace/se3-worktrees/rough-basewheel-b65fac2`。
- GPU 6 核对（`.scratch/fall_teacher_probe_20261023/check_base_wheel.py`，日志 `/workspace/se3-evals/basewheel_20261028/check2.log`）：配置只差 sensor_names；在奖励函数调用的同一时刻独立重算，300 次调用最大差 0、走路组恒 0，recover 组超阈 env·步机身 7185 / 轮子 2529 / 腿杆 633（腿杆不计）。注意：step 返回后再读传感器会和奖励时刻对不上（有 env reset 时整场景 forward 重算接触力），这类核对要包住奖励函数做。GPU 6 单卡 × 8192 smoke 5 轮通过（3.25 s/轮）。
- 14:36 UTC+8 按用户指令开训：GPU 0–5 × 8192、seed 42、5000 轮、save 100、从零，`-c b65fac2`、`-R /workspace/se3-worktrees/rough-basewheel-b65fac2`。W&B [w0wawesv](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/w0wawesv)；PID/PGID `1191104`；state `/workspace/.se3-training-state/whtws/20261028T063555Z`；run `logs/rsl_rl/SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-Alpha800-Recover-RandDir-Impact-BaseWheel/2026-10-28_14-36-03_rough-cts-student128-fallteacherbc-alpha800-recovergroup-randdir-impact005-basewheel-scratch-bench-seed42-6x8192-5k`。第 8 轮无 Traceback / nefc。参数 `.scratch/fall_teacher_probe_20261023/whtws_basewheel_args.txt`。对照 hiphaauf / 3zqym1vn，用户看 sim2x 判断动作。
- w0wawesv 跑满 5000：900–2200 轮站稳率 0.97–1.0、倒置 0.83–0.88、侧躺 0.75–0.89；2500 轮起倒置塌到 ≈0（约 35% 局躺到 8 s 截止）。评测（GPU 0–2，`/workspace/se3-evals/impact_20261027/w_*.json`，脚本 `policy_recovery_eval_impact_wheel.py` 加了轮子对地力）：model_1000 倒置机身峰值 1081 / 2144 N、98% 超 3W、轮子峰值 321 / 566 N，对照 3zqym1vn 2300 机身 1013 / 1749 N、轮子 197 / 367 N——−0.05 撞击罚没压住砸地，轮子力反而更高。注意 GPU 6 被他人占满（约 80 GB），评测改用空卡。

## 2026-10-28 TeacherStay：老师不退场 + 撞击罚 −0.15（机身 + 两轮）

- 代码 `00d38a3c`（xyh/1020，未推送）：入口 `…-FallTeacherBC-Recover-RandDir-Impact015-BaseWheel-TeacherStay` = BaseWheel 改两处：α 调度轮次取 int32 上限（α 全程 0.85，老师不退场）、撞击罚 −0.05 → −0.15。用户选方案 1（训练里老师常驻、上车只跑学生，训练 / 部署不一致是已知风险）。配置核对（`.scratch/fall_teacher_probe_20261023/check_teacher_stay.py`）只差这两处。bundle `b65fac2b..xyh/1020` → `refs/heads/exp/rough-teacherstay-00d38a3`；工作副本 `/workspace/se3-worktrees/rough-teacherstay-00d38a3`。六卡 × 8192 smoke 5 轮通过（3.32 s/轮），日志 `/workspace/.se3-training-state/whtws/smoke-teacherstay-00d38a3/`。
- 21:56 UTC+8 按用户指令开训：GPU 0–5 × 8192、seed 42、5000 轮、save 100、从零，`-c 00d38a3`、`-R /workspace/se3-worktrees/rough-teacherstay-00d38a3`。W&B [aoj8n9nr](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/aoj8n9nr)；PID/PGID `1463925`；state `/workspace/.se3-training-state/whtws/20261028T135621Z`；run `logs/rsl_rl/SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-Recover-RandDir-Impact015-BaseWheel-TeacherStay/2026-10-28_21-56-28_rough-cts-student128-fallteacherbc-teacherstay085-recovergroup-randdir-impact015-basewheel-scratch-bench-seed42-6x8192-5k`。第 9 轮无 Traceback / nefc。参数 `.scratch/fall_teacher_probe_20261023/whtws_teacherstay_args.txt`。训完看两样：Pod 上开着老师评测撞击（学生 + 老师，需给评测脚本加接老师选项），学生单独拉 ONNX 开 sim2x。
- 2026-10-28 用户更正：要的是"模仿损失不退场、老师执行照常退场"，TeacherStay 理解错了；按指令停 aoj8n9nr（第 880 轮，最后存档 model_800；SIGINT 后 GPU 0–5 回 2 MiB）。停前不接老师的倒置组 3 s 起身已到 0.88–0.95。

## 2026-10-28 LossStay：模仿损失常驻、老师执行照常退场

- 代码 `459eaeea` + `adfbd292`（xyh/1020，未推送）：`FallTeacherCfg.supervise_after_retire`（α=0 后状态机继续推进、只监督不执行）与 `target_leash_rad`（开环目标限在实测 ±0.3 rad，只在 α=0 后生效——随机方向教师 π rad/s 下 α=0.85 时约 27% 接管步会顶到 0.3，一直限会改掉教师轨迹）；监督目标组权重列常驻模式下恒 1（模仿损失不乘 α）；env_cfg `fall_teacher_persist_loss`。入口 `…-Alpha800-Recover-RandDir-Impact015-BaseWheel-LossStay` = BaseWheel + 模仿损失常驻 + 撞击罚 −0.15（机身 + 两轮）。bundle 到 `refs/heads/exp/rough-lossstay-adfbd29`；工作副本 `/workspace/se3-worktrees/rough-lossstay-adfbd29`。
- GPU 0 核对（`.scratch/fall_teacher_probe_20261023/check_loss_stay.py`，日志 `/workspace/se3-evals/lossstay_20261028/check2.log`）全过：配置只差三项；BaseWheel、Recover、LossStay（α>0）三入口与 00d38a3c 影子 600 步逐位一致（状态机、教师动作、监督目标、执行动作）；α=0 时执行动作 = 策略动作、critic 接管维恒 0、监督权重恒 1、接管 env 目标都在 ±0.3 rad 内。六卡 × 8192 smoke 5 轮通过（3.33 s/轮），日志 `/workspace/.se3-training-state/whtws/smoke-lossstay-adfbd29/`。
- 23:01 UTC+8 按用户指令开训：GPU 0–5 × 8192、seed 42、5000 轮、save 100、从零，`-c adfbd29`、`-R /workspace/se3-worktrees/rough-lossstay-adfbd29`。W&B [zxmvuilz](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/zxmvuilz)；PID/PGID `1491835`；state `/workspace/.se3-training-state/whtws/20261028T150059Z`；run `logs/rsl_rl/SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-Alpha800-Recover-RandDir-Impact015-BaseWheel-LossStay/2026-10-28_23-01-06_rough-cts-student128-fallteacherbc-alpha800-lossstay-leash03-recovergroup-randdir-impact015-basewheel-scratch-bench-seed42-6x8192-5k`。第 10 轮无 Traceback / nefc。参数 `.scratch/fall_teacher_probe_20261023/whtws_lossstay_args.txt`。看点：800 轮老师退场后倒置起身率能否保住（−0.15 撞击罚曾让 m6rxc7v9 躺平）、Loss/imitation 是否维持低位。GPU 6 仍被他人占用约 80 GB。
- zxmvuilz 跑满 5000（正常结束）：老师 800 轮退场后倒置起身率 900 轮起 ≈0、全程没回来（侧躺 0.36 → 0.78，约 1/3 局躺到 8 s 截止）。退场后 `FallTeacher/policy_teacher_abs_err` 只有 0.7（动作单位，≈0.18 rad）、接管占比约 0.22——学生"跟得上"限幅后的目标却不翻：±0.3 rad 限幅让目标只领先实测 0.3 rad，学生几乎不动也能把模仿损失压低，模仿损失拉不动；撞击罚 −0.15 下 PPO 更偏好不翻。2026-10-29 跑完时 whtws 七卡已被他人 var_sampling 任务占满（各约 80 GB）。

### 2026-10-30 旧 recovery 机制删除核对

- bundle `adfbd292..xyh/1020` → `refs/heads/exp/recovery-cleanup-e8420b9`；工作副本 `/workspace/se3-worktrees/eq-05654e5b`（删前）与 `eq-e8420b97`（删后），核对完可删。
- CPU 逐位比对（`/workspace/se3-evals/recovery_cleanup_20261030/`，`record.py` 64 env、episode 缩到 2 s、随机动作 200 步、每任务 128 次 reset）：Flat、Crawl、Crawl-Slope、Jump-Mimic、Rough、Rough-GRU、Student128、FallTeacherBC-Alpha800-Recover 全部观测组 / 奖励 / 终止逐位一致（`compare.txt`）。WheelDog 在 CPU 上新旧都报 `deg2rad(float)`（自身 rewards 的旧问题），且只用共享 events 里未改动的域随机化函数，不受影响。
- GPU 0–4 各 × 8192 smoke 5 轮通过：Flat-MLP、Crawl-MLP、Jump-Mimic-MLP、Student128、FallTeacherBC-Alpha800-Recover（日志 `smoke-{0..4}.log`）。

### 2026-10-30 Recover-Catch（倒地教师 catch 释放）

- 开环扫参与录像在 `/workspace/se3-evals/catch_sweep_20261030/`（`catch_sweep.py`、`record_teacher.py`；工作副本 `eq-catch` = e8420b97 + 未提交的 fall_teacher.py）。
- bundle `e8420b97..xyh/1020` → `refs/heads/exp/recover-catch-2af06e12`；工作副本 `/workspace/se3-worktrees/recover-catch-2af06e12`（`.venv` 软链主 checkout）。GPU 0 × 1024 smoke 5 轮通过（`/workspace/se3-evals/recover_catch_smoke_20261030/smoke.log`，seg_pause_frac 0）。
- 21:28 UTC+8 按用户指令开训 `SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-Alpha800-Recover-Catch`：GPU 0–5 × 8192、seed 42、5000 轮、save 100，`-c 2af06e1 -R` 上述工作副本，与 gyi9exiz 同规模，只差倒地教师释放方式。W&B [uglay7z1](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/uglay7z1)；PID/PGID `2592653`；state `/workspace/.se3-training-state/whtws/20261030T132802Z`；run `logs/rsl_rl/SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-Alpha800-Recover-Catch/2026-10-30_21-28-09_rough-cts-student128-fallteacherbc-alpha800-recovergroup-catch-scratch-seed42-6x8192-5k`（在工作副本下）。第 10 轮无 Traceback。启动器不传 `-p` 时 W&B 项目默认取任务名，对照 run 要显式 `-p SE3-WheelLegged-Rough`。
