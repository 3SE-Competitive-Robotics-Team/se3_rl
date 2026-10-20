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

