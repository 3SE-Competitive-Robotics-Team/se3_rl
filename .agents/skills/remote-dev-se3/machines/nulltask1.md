# nulltask1 K8s 机器备忘录

> 通用操作流程见父目录 `SKILL.md`，本文件只记录 `nulltask1` 特有的连接参数和目标边界。

## 连接参数

| 项目 | 值 |
|---|---|
| 第一跳 SSH 别名 | `laptop-wg` |
| Kubernetes 入口 | `192.168.2.46`（2026-09-26 由 .47 变回 .46，2026-09-05 曾由 .46 变为 .47；DHCP 分配；主机密钥不变，主机名 host-10-10-10-116；地址是动态值，连不上先 keyscan 核对密钥再改） |
| 入口 SSH 用户 | `root` |
| 入口 SSH 端口 | `2222` |
| Kubernetes namespace | `gczx-project06` |
| Pod | `nulltask1` |
| Container | `container-1` |
| 连接链路 | 本机 → `laptop-wg` → `root@192.168.2.46:2222` → `kubectl` |
| 仓库路径 | `/workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg` |
| 仓库分支/commit | `xyh/925` / `79fff6f88f970ded7faa7ee697c12d2a205fe015`，子模块 `e753ce6`（2026-09-26 bundle 同步，M33 在跑时不要切 commit；此前为 `spring_final@738268f`，分支仍保留；动态值，用前 `git rev-parse HEAD` 复核） |
| se3-sim2x commit | `5386e4354877b7674ee1b9ea94631b83b141d8cb`（2026-09-25，main 记录的 gitlink） |
| 虚拟环境 | `/workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg/.venv` |
| uv | `/root/.local/bin/uv` |
| uv Python | `/root/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu` |
| 系统 CUDA toolkit | `/usr/local/cuda-12.2` |
| CUDA compat | `/workspace/cudacompat/usr/local/cuda-12.6/compat` |
| W&B 入口机内网代理 | `10.10.10.116:38443` |
| W&B Kubernetes Secret | `nulltask1-wandb` |

入口使用已有 SSH 凭据；密码、私钥和临时验证码不得写入本文件。`laptop-wg` 上的
`abbtask` SSH 别名带有指向其他 Pod 的 `RemoteCommand`，操作 `nulltask1` 时不得复用。

## CUDA 运行约束（强制）

`nulltask1` 运行本仓库时必须使用 CUDA 12.6 compat。所有 SE3 smoke、训练，
以及 Torch、Warp 或 MJLab GPU 检查命令，都必须在同一条命令或同一个 shell
session 内先执行：

```bash
compat=/workspace/cudacompat/usr/local/cuda-12.6/compat
test -d "$compat" || { echo "CUDA compat 不存在: $compat" >&2; exit 1; }
export LD_LIBRARY_PATH="$compat:${LD_LIBRARY_PATH:-}"
test "${LD_LIBRARY_PATH%%:*}" = "$compat" || { echo "CUDA compat 未位于 LD_LIBRARY_PATH 首位" >&2; exit 1; }
```

- compat 路径必须是 `LD_LIBRARY_PATH` 的第一项，不得追加在其他 CUDA 库之后。
- 按单条命令或当前 session 传入上述环境；不得替换 `/usr/local/cuda`、修改
  系统链接，或改动容器全局配置。
- 未通过上述目录与首项检查时，不得启动 SE3 smoke、训练或 GPU 验证。

## 只读检查

Pod 状态：

```powershell
ssh laptop-wg "ssh -o BatchMode=yes -o ConnectTimeout=10 -p 2222 root@192.168.2.46 kubectl get pod -n gczx-project06 nulltask1 -o wide"
```

Pod 事件：

```powershell
ssh laptop-wg "ssh -o BatchMode=yes -o ConnectTimeout=10 -p 2222 root@192.168.2.46 kubectl get events -n gczx-project06 --field-selector involvedObject.name=nulltask1 --sort-by=.lastTimestamp"
```

GPU 状态：

```powershell
ssh laptop-wg "ssh -o BatchMode=yes -o ConnectTimeout=10 -p 2222 root@192.168.2.46 kubectl exec -n gczx-project06 nulltask1 -c container-1 -- nvidia-smi"
```

## 网络访问

- 2026-08-31 检查时，Pod 直连 `github.com` 和 `pypi.org` 会在 TLS 阶段被重置。
- `https://download.pytorch.org/whl/cu128` 与
  `https://mirrors.aliyun.com/pypi/simple` 可访问。PyPI 包安装应按命令显式传入阿里云
  mirror，不修改全局 uv/pip 配置。
- `uv.lock` 中的 wheel 直链仍可能绕过 mirror 指向 `files.pythonhosted.org`。精确同步时先
  通过 mirror 预热差异包，再执行 `uv sync --offline --frozen`。两个本地项目都使用
  `uv-build==0.11.8`；离线构建需先把它安装进 `.venv`，并传入 `--no-build-isolation`。

## 入口机 IP 是 DHCP 动态值（2026-09-05 由 .46 变为 .47，2026-09-26 变回 .46）

2026-09-26 这次的现象：laptop-wg 掉线重连后自身变为 `192.168.2.78`，入口 `.47` ping 不通、ARP 无记录、SSH 超时。
在笔记本上用 .NET `SendPingAsync` 并发扫一遍 /24（几秒）再看 `arp -a`，存活主机只有 7 台；`.46` 以
`StrictHostKeyChecking=yes` 连上、hostname 为 `host-10-10-10-116`（known_hosts 里旧的 `.46` 记录仍有效）。
不要用 `ssh-keyscan -f` 扫整段：Windows 版逐个串行等超时，几分钟不返回。
入口 IP 一变，笔记本上的 W&B gateway / boring 隧道连不上入口机，W&B 页面会把在跑的 run 标成 crashed，
但 Pod 上训练照常（M29 在 W&B 停在第 7 轮时，Pod 上已跑到 1716 轮）。

症状：laptop-wg 能 ping 通网关 `192.168.2.1`，但对入口机 ARP `Incomplete`、SSH 在 banner 阶段超时，
同网段的其他服务器（如 VNC `192.168.2.196`）同样连不上。处理顺序：

1. 在笔记本上 `ssh-keyscan -p 2222 -t ed25519 <新IP>`，与 `known_hosts` 里旧 IP 的记录比对，密钥一致才认定是同一台入口机
   （2026-09-05 核对：`AAAAC3Nza…Zt5s` 一致，主机名 `host-10-10-10-116`，内网 `10.10.10.116` 不变）。
2. 写死 IP 的地方共 5 处，全部改掉并各留 `.bak-<日期>-entry46` 备份：
   - 本机 `C:\Users\13567\.local\bin\start-nulltask1-training.ps1` 的 `$EntryHost`
   - 笔记本 `C:\Users\Lenovo\.local\bin\nulltask1-wandb-gateway.ps1` 的 `$entryHostName`
   - 笔记本 `C:\Users\Lenovo\.boring.toml` 的 `host`
   - 笔记本 `C:\Users\Lenovo\.local\bin\nulltask1-sim2x-tunnel.ps1` 的 `root@IP`
   - 本文件
3. `schtasks /End` 再 `/Run` 两个计划任务 `SE3-Nulltask1-WandbGateway`、`SE3-Nulltask1-Sim2xTunnel`，
   看 gateway.log 出现 `boring tunnel opened` / `gateway is ready`，入口机 `ss -ltn` 有 38443，
   `iptables -S SE3_WANDB_NULLTASK1` 放行的是当前 Pod IP。
4. 在笔记本上跑 PowerShell 多行脚本时用 `powershell -EncodedCommand <UTF-16LE base64>`，不要在 bash 里拼 `$` 变量。

## W&B 实时接入

`nulltask1` 直连 `api.wandb.ai` 会被重置，实时链路为：

```text
nulltask1 -> 10.10.10.116:38443 -> boring SSH 反向隧道
          -> laptop-wg 127.0.0.1:18787 -> W&B Public Cloud
```

`laptop-wg` 上的受管组件：

| 项目 | 值 |
|---|---|
| boring tunnel | `nulltask1-wandb-proxy` |
| boring config | `C:\Users\Lenovo\.boring.toml` |
| gateway script | `C:\Users\Lenovo\.local\bin\nulltask1-wandb-gateway.ps1` |
| Scheduled Task | `SE3-Nulltask1-WandbGateway` |
| HTTP CONNECT proxy | `proxy.py==2.4.10`, `127.0.0.1:18787` |
| 入口机防火墙链 | `SE3_WANDB_NULLTASK1` |

- W&B 0.28.0 对带认证 HTTPS proxy 仍有公开兼容性问题（
  [`wandb/wandb#10367`](https://github.com/wandb/wandb/issues/10367)）。因此本机代理本身
  只监听 `laptop-wg` loopback，入口机的 `38443` 由 `SE3_WANDB_NULLTASK1`
  严格限制为当前 `nulltask1` Pod IP，其他来源一律 `DROP`。
- gateway 每 60 秒重新查询 Pod IP 并原子刷新专用防火墙链。Pod 重建、
  入口机内网 IP 变化或 `laptop-wg` 未登录时，必须重新做在线预检。
- API key 和 proxy URL 只存在 namespace `gczx-project06` 的 Secret
  `nulltask1-wandb` 中；不得输出 Secret YAML、写入 profile、仓库、命令参数或日志。
- 正式训练必须从 Secret 通过 stdin 注入 `WANDB_API_KEY` 与 proxy 变量，
  不得在 `kubectl exec -- env ...` 参数中展开凭据。

在线训练进程必须使用：

```bash
export WANDB_MODE=online
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_USERNAME=luzhongjin365-se3
export WANDB_ENTITY=luzhongjin365-se3
export WANDB_PROJECT=<task-or-explicit-project>
export WANDB_DIR=/workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg
export WANDB_INIT_TIMEOUT=120
export WANDB__SERVICE_WAIT=120
export SE3_LOGGER=wandb
export NO_PROXY='127.0.0.1,localhost,10.247.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.svc,.cluster.local'
export no_proxy="$NO_PROXY"
```

`HTTP_PROXY` / `HTTPS_PROXY` 及小写形式从 Secret 注入。`NO_PROXY` 不得省略
Kubernetes Service/Pod 网段，否则 W&B 采集 K8s metadata 时会把
`10.247.0.1:443` 误送到外部代理，导致 service timeout。

启动在线训练前必须同时确认：Scheduled Task 为 `Running`、入口机
`38443` 正在监听、专用防火墙链的 allow IP 等于当前 Pod IP，且容器内
W&B authenticated API probe 成功。任一检查失败时必须改用 `WANDB_MODE=offline`，
不得让训练在断开的 online 模式下等待上传。

2026-08-31 已完成 1 iteration CPU 端到端 probe：W&B run
[`j4fvac7n`](https://wandb.ai/luzhongjin365-se3/se3-wandb-integration-probe/runs/j4fvac7n)
状态为 `finished`，history 1 行，已上传 `model_0.pt` 和其他 6 个配置/日志文件。

## 一键在线训练启动器

本机个人启动器位于：

```text
C:\Users\13567\.local\bin\start-nulltask1-training.ps1
```

该脚本只服务于本 profile，不放进仓库公共 `scripts/`。它先检查 laptop Scheduled
Task、Pod/Container、入口代理监听、防火墙 allow IP、Secret 字段、远端代码干净度、
CUDA compat 首项、GPU 占用和 W&B authenticated API，然后才用 `setsid + nohup`
启动训练。Secret 值由入口机直接通过 stdin 注入 Pod，不进入参数、文件、profile、
输出或训练日志。

### 解释器要求（强制）：必须用 PowerShell 7

启动器**不能**用 Windows PowerShell 5.1 运行，两处硬阻塞，且第 2 条无法绕过：

1. 脚本是 UTF-8 **无 BOM**。5.1 按 ANSI 解码无 BOM 的 `.ps1`，中文字符串全部变成
   乱码，乱码字节被解析成 `&` 和引号，在语法阶段就失败。加 UTF-8 BOM 可修复这一层。
2. 脚本用 `ProcessStartInfo.StandardInputEncoding` 向 `ssh` 子进程注入 stdin，
   即 Secret 注入路径。该属性是 .NET Core 2.1+ API，在 .NET Framework 上不存在
   （实测 `.NET Framework 4.8.09221` 下 `StandardOutputEncoding` 存在、
   `StandardInputEncoding` 不存在）。5.1 只能跑在 .NET Framework 上，会在
   `Test-GatewayScheduledTask` 处直接抛
   `The property 'StandardInputEncoding' cannot be found on this object`。
   修这一条必须改动 Secret 注入代码，不得为了跑通而擅自改写。

本机唯一可用的 pwsh 是其他工具自带的 runtime，**不在 PATH 上**：

```text
C:\Users\13567\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe
```

版本 7.6.4 / .NET 10，`StandardInputEncoding` 存在，2026-09-01 实测可正常完成
`-DryRun` 与正式启动。注意 `where.exe pwsh`、`Program Files\PowerShell\*`、
`Program Files (x86)`、`WindowsApps\Microsoft.PowerShell_*`、chocolatey、scoop、
dotnet tools 全部扫不到它，只有全盘 `where /R C:\ pwsh.exe` 能找出来；同一次扫描
另有 `C:\Windows.old\...\WindowsApps\pwsh.exe`，是旧系统残留的执行别名，不可用。

`laptop-wg` 同样是 PowerShell 5.1（`5.1.26100.9168`），且其 `.local\bin` 下没有本
启动器。启动器只在本机运行，`ssh laptop-wg` → 入口机 → `kubectl` 两跳由脚本内部
完成；不要试图把它搬到 `laptop-wg` 上执行。

调用方式：

```powershell
& 'C:\Users\13567\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe' `
  -NoProfile -File 'C:\Users\13567\.local\bin\start-nulltask1-training.ps1' `
  -Task <task> -NumEnvs <n> -Iterations <n> -SaveInterval <n> -GpuIds all -DryRun
```

脚本输出含中文，用非 UTF-8 控制台承接会显示为乱码；需要可读输出时改用
`-Command` 并先设 `[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)`，
或把输出 `Out-File -Encoding utf8` 后再读。

从头训练示例：

```powershell
start-nulltask1-training.ps1 `
  -Task SE3-WheelLegged-Flat-GRU `
  -NumEnvs 1024 `
  -Iterations 5000 `
  -SaveInterval 100 `
  -GpuIds all
```

只读预检使用相同参数并追加 `-DryRun`；它不会创建训练进程或远端状态文件。

2026-09-02 新增（备份 `start-nulltask1-training.ps1.bak-20260902-1132`）：

- `-AllowConcurrent`：允许与已在运行的训练并发；跳过"已有训练进程"拒绝，GPU compute process 检查只针对
  申请的 GPU 子集。必须配合显式 GpuIds 列表，PowerShell 侧要加引号：`-GpuIds '2,3'`（不加引号会被解析成数组）。
- `--gpu-ids` 显式列表按 mjlab `TYRO_FLAGS`（UsePythonSyntaxForLiteralCollections）以 Python 字面量传入
  （`--gpu-ids [2,3]`）；`0,1` 与 `0 1` 两种写法都会被 tyro 拒绝（Unrecognized options）。
- 六卡三并发示例：三次调用分别 `-GpuIds '0,1'`、`'2,3'`、`'4,5'` + `-AllowConcurrent`，共用 `-WandbProject`。
- run_dir 检测已改为"启动前目录快照 + 新出现目录"；此前的 `-newer marker` 在同任务已有 run 持续写 checkpoint 时会
  把旧 run 目录写进 state 文件（2026-09-02 `20260902T182152Z-27447` 已手工改正）。
- 代码同步（Pod 不通 GitHub）：本地 `git bundle create <old>..spring_final`，`base64 -w0 | ssh laptop-wg "ssh ... kubectl exec -i ... -- tee /workspace/x.b64"`
  流入，再用 `scripts/remote_bash.ps1`（两跳 + kubectl，脚本 ≤ 数 KB，超长命令行会 "filename too long"）在 Pod 内校验 sha256 并 `merge --ff-only`。
默认强制从头训练。续训必须同时显式传入 `-Resume`、`-LoadRun` 和
`-LoadCheckpoint`；远端代码脏时默认拒绝，只有人工确认后才可追加 `-AllowDirty`。

每次实际启动的 PID、PGID、外层日志和识别到的 run directory 记录在：

```text
/workspace/.se3-training-state/nulltask1/<launch-id>/
```

脚本返回可直接复制的日志监控命令。训练进程由 `setsid` 脱离，因此启动检查结束后
两跳 SSH 都会关闭；W&B 实时上传继续复用 laptop 上独立常驻的 boring tunnel。

## 2026-09-03 抖动对照实验（6 并发，每卡一个 run）

仓库 commit `c72c4e2`（`feat(flat): 新增抖动对照实验的 5 个单变量任务入口`）。
每个 run 单卡 × 16384 envs × 5000 轮，样本量与此前 2 卡 × 8192 的 E0 基线一致（64 × 16384 = 1.05M/轮），
seed 全部用默认 42，因此 A0 是五个变体的精确对照。ETA 约 9 小时，显存约 10.7 GB/卡。

| 标签 | task | GPU | run name | PGID | state dir |
|---|---|---|---|---|---|
| A0 基线 | `SE3-WheelLegged-Flat-MLP` | 0 | `flat-A0-base-1x16384-5k` | 84000 | `20260903T165416Z-11265` |
| A1 速度死区 | `...-Flat-Exp-CmdDeadband` | 1 | `flat-A1-cmddeadband-1x16384-5k` | 84654 | `20260903T165449Z-23782` |
| A2 轮离地罚 | `...-Flat-Exp-WheelContact` | 2 | `flat-A2-wheelcontact-1x16384-5k` | 85136 | `20260903T165516Z-15507` |
| B1 倾角 barrier | `...-Flat-Exp-TiltBarrier` | 3 | `flat-B1-tiltbarrier-1x16384-5k` | 85714 | `20260903T165555Z-09719` |
| B2 动作延迟 | `...-Flat-Exp-ActionDelay` | 4 | `flat-B2-actiondelay-1x16384-5k` | 86249 | `20260903T165628Z-18033` |
| B3 yaw 课程 | `...-Flat-Exp-YawCurriculum` | 5 | `flat-B3-yawcurriculum-1x16384-5k` | 86733 | `20260903T165656Z-25349` |

启动前按用户授权终止了三个已跑完但卡在 W&B 上传的进程组（PGID 72578 / 73471 / 78359），
SIGINT 后 15 s 内全部退出，6 张卡显存回到 2 MiB；sim2x sidecar PGID 82778 未受影响。

### W&B gateway 的一个坑

2026-09-06 第三次出现，且是**新的失效模式**：`boring` 隧道 half-open。计划任务 Running、gateway.log 每 60 s
正常追加 `Firewall refreshed`、入口机 `ss -ltn` 有 38443、`boring list` 也显示隧道在线，但 Pod 经
`http://10.10.10.116:38443` 发 HTTP 请求全部 25 s 超时（TCP 连得上，不转发）。后果：D8 训练早已跑完并导出
model_4999，进程却卡在 W&B 上传约 12 小时不退出，占着 6 张卡的显存。

判据：`boring list` 的 Status 列显示隧道已开了很久（本次 16h01m），而 Pod 内用 urllib 请求
`https://api.wandb.ai/graphql` 超时；正常时应在 2 s 内返回 HTTPError 405。
End+Run 计划任务**修不好**，因为 `Ensure-BoringTunnel` 只在 `boring list` 报 `closed` 时才重开，
half-open 会被当成健康，日志里只有 `gateway is ready` 而没有 `boring tunnel opened`。
修复必须强制重建隧道，在笔记本上执行：

```
C:\Users\Lenovo\.local\bin\boring.exe close nulltask1-wandb-proxy
C:\Users\Lenovo\.local\bin\boring.exe open  nulltask1-wandb-proxy
```

重开后 `boring list` 的 Status 归零计时，Pod 侧探测应立刻返回 405/404。本地 proxy（18787）本身是好的，不用动。

2026-09-25 第四次：gateway.log 停在 09-22 18:25，`boring list` 为 `closed`，计划任务仍 Running。根因是
gateway 刷防火墙时起的 `ssh.exe` 子进程（09-22 18:26:03 启动）挂住不返回，主循环卡死三天、隧道断了没人重开。
`/End` 能杀掉 gateway 本体，但挂住的 `ssh.exe` 孤儿会残留（无害）。`/End` 后立刻 `/Run` 会撞上旧进程尚未退出，
新实例记 `Another gateway instance is active; exiting` 直接退出；隔几秒再 `/Run` 一次即出现
`boring tunnel opened` / `gateway is ready`。


2026-09-05 第二次出现：入口机 IP 变更后 gateway.log 在 10:34 停止追加，计划任务仍显示 Running，
入口机 38443 无监听；当时在跑的 D6 只把 history 同步到 1991 轮，后半段曲线没有上传（Pod 上的 checkpoint 与 ONNX 齐全）。
修 IP 后 End+Run 即恢复。启动器的 DryRun 会做 W&B 认证探测，偶发一次 `URLError`（exit 37）是链路瞬断，重试即可。


laptop-wg 上的 Scheduled Task `SE3-Nulltask1-WandbGateway` 可能显示 `State=Running`，
但入口机 38443 已经没有监听，启动器在 Pod 预检里以 exit 22 拒绝
（`入口机 38443 未监听，W&B boring tunnel 不可用`）。判据是
`~\.local\state
ulltask1-wandb-gateway\gateway.log` 停止追加（本次停在 2026-09-02T18:18+08:00）。
修复：`schtasks /End /TN SE3-Nulltask1-WandbGateway` 再 `/Run`，日志出现
`boring tunnel opened` / `gateway is ready` 即恢复；本地 proxy（18787）会被复用，不必单独重启。

## 2026-09-04 课程对照实验 C 批（6 并发，commit 596d85d）

单卡 16384 envs × 5000 轮，seed 默认 42（C6 用 7）。**六个 run 全部落在同一个 W&B project
`SE3-WheelLegged-Flat`**：启动器 `-WandbProject` 会 export `WANDB_PROJECT`，而
`bind_task_name` 是 `os.environ.get("WANDB_PROJECT", task_name)`，不传就退化成任务名、散成多个 project
（B 批六个就是这么散掉的）。**以后同族对照一律显式传 `-WandbProject`。**

| 标签 | task | GPU | run name | PGID | state dir |
|---|---|---|---|---|---|
| C2 课程回退 | `...-Exp-CurriculumRetreat` | 0 | `flat-C2-currretreat-1x16384-5k` | 100537 | `20260904T043353Z-06421` |
| C1 yaw 独立门控 | `...-Exp-YawGate` | 1 | `flat-C1-yawgate-1x16384-5k` | 101060 | `20260904T043422Z-12169` |
| C5 死区+倾角合并 | `...-Exp-DeadbandTilt` | 2 | `flat-C5-deadbandtilt-1x16384-5k` | 101592 | `20260904T043451Z-25256` |
| C3 yaw 步长 0.25 | `...-Exp-YawStep` | 3 | `flat-C3-yawstep-1x16384-5k` | 102125 | `20260904T043530Z-05447` |
| C4 阈值 0.75 | `...-Exp-AdvanceThreshold` | 4 | `flat-C4-advthresh-1x16384-5k` | 102654 | `20260904T043558Z-15367` |
| C6 基线 seed 7 | `SE3-WheelLegged-Flat-MLP` | 5 | `flat-C6-base-seed7-1x16384-5k` | 103193 | `20260904T043657Z-07299` |

### 启动器 ExtraArgs 的两个坑

- 用 `& pwsh -File <脚本> ... -ExtraArgs @('--agent.seed','7')` 会被 `-File` 拆成两个位置参数，
  报 `A positional parameter cannot be found that accepts argument '7'`。改用 `-Command "& '<脚本>' ..."`。
- 启动器只接受 `--key` 或 `--key=value` 形式（脚本第 185 行校验），所以必须写 `--agent.seed=7`，
  不能写成 `--agent.seed 7` 两个 token。

## 2026-09-04/05 D 批：JointAction 语义与 σ 平衡点实验（6 卡 × 8192 envs，W&B project `SE3-WheelLegged-Flat`）

| 标签 | task / commit | steps/env | W&B run | PGID | state dir | 备注 |
|---|---|---|---|---|---|---|
| D1 | `...-Exp-JointAction` / `84d0cc2` | 64 | `44qpsou9` | – | `20260904T130740Z-31846` | 2352 轮 crashed |
| D2 | `...-Exp-JointAction` / `46f67f9` | 24 | `l2h4spch` | 122553 | `20260904T161122Z-23631` | 5000 轮 finished，σ 平衡点实验的对照组 |
| D3 | `...-Exp-JointAction` / `46f67f9` | 64 | `dwttj7so` | 128755 | `20260905T010531Z-18412` | 2026-09-05 约 100 轮时按用户指令 SIGINT 停止（6 s 内退出） |
| D4 | `...-Exp-JointActionWheelPrice` / `be3f5c8` | 24 | `istnlw0b` | 131389 | `20260905T015012Z-14152` | 5000 轮 finished（2.4 h，进程组正常退出、W&B 已同步）。相对 D2 唯一差异：action_rate 轮分量 1/9→1.0、action_smoothness 轮分量 0.222→2.0 |
| D5 | `...-Exp-JointActionWheelPriceCriticLr` / `8e9a18a` | 24 | `mnj9qqbr` | 138789 | `20260905T082338Z-20489` | 2026-09-05 16:23 启动，17:40 按用户指令在 2712 轮 SIGINT 停止（checkpoint 到 model_2700）。相对 D4 唯一差异：critic 固定 LR 6.5e-4。1368 轮时 actor LR 未塌（D4 同期已贴地板）、Loss/value 无尖峰，但 reward 落后 D4 约 8 分 |
| D6 | `...-Exp-JointActionWheelPricePoseHold` / `f8f2fea` | 24 | `kbkd4wxa` | 142226 | `20260905T094223Z-04944` | 2026-09-05 17:42 启动，跑满 4999 轮（checkpoint 齐全），21:33 在最后收尾时被 SIGINT，W&B 末段同步可能不完整。1300 轮 sim2x：腿峰峰 11.7°（D4 24°），极限环 1.0 Hz 仍在；joint_pos_penalty 实付 0.35/s 不降。相对 D4 唯一差异：加 joint_pos_penalty -1.0（腿姿态回默认，静止 ×5，recovery 线同参数）。判据：确定性 sim2x 站立的 0.67 Hz 极限环消失（腿峰峰 24° → 个位数）、跟踪分与倒地率不掉 |
| D7 | `...-Exp-JointActionWheelPriceNoCmdErr` / `26cd229` | 24 | `flat-D7-jointaction-wheelprice1-nocmderr-steps24-6x8192-5k` | 146629 | `20260905T133441Z-28352` | 2026-09-05 21:34 启动。相对 D4 唯一差异：删除 command_velocity_error（该项 99% 代价来自指令阶跃后 1 s 的瞬态）。判据：跟踪分不掉、动作罚与轮抖动下降、倒地率不升。2026-09-06 00:00 跑满 4999 轮（2.35 h，LR 全程未贴地板，Loss/value 0.36）；终点 sim2x：站立倾角 1.5°、腿峰峰 8.2° @0.92 Hz（D4 24.4°）、前进 1.5 倾角 0.55°、无倒地，全面优于 D4 |
| D8 | `...-Exp-JointActionWheelPriceNoCmdErr` / `99b37a5`（子模块 `d65b9ff`） | 24 | `flat-D8-jointaction-wheelprice1-nocmderr-balancedpose-h038-steps24-6x8192-5k`（W&B `ezm9068i`） | 151365 | `20260905T164241Z-04862` | 2026-09-06 00:42 启动（用户批准）。相对 D7 差异两项：默认站姿静平衡重标定（bd5866e，整机质心正对轮轴 + 高度默认 v2）与高度指令 0.20–0.38（99b37a5）。判据：确定性 sim2x 站立俯仰均值回到 0° 附近、0.92 Hz 腿极限环幅值继续下降、跟踪与倒地率不掉、0.38 m 高度可站。2026-09-06 跑到 4999（2.33 h，W&B 记为 crashed，51 个 checkpoint 与 51 个 ONNX 齐全）。站立俯仰均值 −0.2°（1300 轮）达成判据，但 2750 轮起 LR 贴 1e-5 地板，Loss/value 尾部指数发散（每 200 轮最大值 0.6 → 1151，中位数恒 0.5），站立腿峰峰 4.7° → 7.4°，reward 峰值 123 → 终点 113 |
| D9 | `...-Exp-JointActionWheelPriceNoCmdErrCriticLr` / `5f46761` | 24 | `flat-D9-jointaction-wheelprice1-nocmderr-balancedpose-h038-criticlr-steps24-6x8192-5k`（W&B `uoor8dik`） | 156558 | `20260906T074714Z-18046` | 2026-09-06 15:47 启动。相对 D8 唯一差异：critic 固定 LR 6.5e-4（`se3_train.ppo.Se3PPO`），actor 仍走 KL 自适应。已验证 `Loss/critic_learning_rate`=6.5e-4 恒定、actor LR 独立浮动。判据：LR 贴地板后 Loss/value 尾部不再发散、reward 后期不回落、评测端站立腿峰峰保持在 1300 轮的 4.7° 水平。2026-09-06 17:0x 在 1800 轮按用户指令停止（19 个 checkpoint 与 19 个 ONNX 齐全）。**负结果**：1360 轮时 reward 88 对 D8 的 116 且自 1000 轮的 98 起持续下滑，σ 回升（腿 0.186 对 0.136）、action_rate 翻倍（-0.485 对 -0.231）、Loss/value 13.8 对 0.54、贴地板率 40%。与 D8 后期同一套退化但提前 2000 轮。结论：共用学习率原本让 actor 与 critic 同步减速，解耦后 critic 以固定 LR 超前于被压在 1e-5 的 actor，优势函数前后不一致。critic 固定 LR 两次失败（D5 落后 8 分、D9 发散），从候选中划掉 |
| D10 | `SE3-WheelLegged-Flat-MLP` / `e1946ba` | 24 | `flat-D10-merged-baseline-kyberhparams-h038-steps24-6x8192-3.5k`（W&B `2pilnc2r`） | 160076 | `20260906T084631Z-32195` | 2026-09-06 16:46 启动，**3500 轮**（不再跑满 5000，有信息量的窗口就在 3500 内）。最终形态第一步：把 D2–D8 已验证改动合并为 Flat 基线默认（joint 语义、轮分量定价 1.0/2.0、删违令罚、静平衡站姿 + 高度默认 v2、高度 0.20–0.38），并叠加对齐 kyber_rl_lab 的 PPO 超参数（lr 1e-3、entropy_coef 0.01、epochs 5、clip 0.2）。判据：复现 D8 在 1300 轮的水平（站立俯仰≈0°、腿峰峰≤5°、跟踪误差≤0.05），且学习率不再长期贴 1e-5 地板、σ 稳定在 0.2 附近。2026-09-06 按用户指令在 3356 轮停止（34 个 checkpoint 与 ONNX 到 3300）。**结论一半达成**：学习率地板治好了（3000 轮 lr 7.9e-4，D8 同期 4.8e-5 贴地板；Loss/value 稳在 0.49 无尾部发散），但评测端未复现 D8@1300：确定性站立俯仰均值 −2.8°、峰峰 5.9°、腿峰峰 9.8°，而 D8@1300 是 −0.18°/1.97°/4.72°。根因是 entropy_coef 翻倍把 σ 抬到轮 0.268/腿 0.218（D8 为 0.148/0.138），动作罚随之翻倍 |
| D11 | `SE3-WheelLegged-Flat-MLP` / `236666c` | 24 | `flat-D11-mergedbaseline-com5mm-steps24-6x8192-3.5k`（W&B `mher9vfk`） | 164005 | `20260906T102236Z-29634` | 2026-09-06 18:22 启动，3500 轮。相对 D10 唯一差异：base 质心随机化 ±20 mm → ±5 mm。量级核算：base 占整机 84.4%，±20 mm 等于整机质心 ±16.9 mm、配平倾角 ±7.8°，比刚修掉的静平衡 bug（17.2 mm / 8.0°）还大，等于把静平衡站姿的收益随机掉；±5 mm 对应 ±4.2 mm / ±2.0°。reset 腿姿全随机与其余 startup 随机化按用户指令保持不变。判据：确定性站立俯仰均值回到 0° 附近、腿峰峰接近 D8@1300 的 4.7°，且学习率仍不贴地板 |

## 2026-09-06 R 批：崎岖地形线移植（6 卡 × 8192 envs，W&B project `SE3-WheelLegged-Rough`）

| 标签 | task / commit | W&B run | PGID | state dir | 备注 |
|---|---|---|---|---|---|
| R1 | `SE3-WheelLegged-Rough` / `14ae979` | `rough-R1-scutport-terraincurriculum-stepup-6x8192-5k`（W&B `4bq51zsh`） | 168001 | `20260906T152606Z-25615` | 2026-09-06 23:26 启动，5000 轮，ETA 约 5 h。按 scutrobotlab/wheeled-legged_RL 的 V14 rough 线重写崎岖地形任务：环境继承冻结的 Flat 基线，只加带课程的地形集（平地/上下台阶/上下斜坡/随机起伏，台阶 0.02–0.20 m、踏面 1.5 m、10 级）、`terrain_levels` 地形课程、step_up 台阶前瞻状态机、能耗三项 ÷10；PPO 与 Flat 基线逐项相同，只把轮数改为 5000。显存 28.7 GB/卡（Flat 线约 10.7 GB，多出的是地形几何 + 前向射线传感器 + `contact_sensor_maxmatch` 64→500），迭代 3.38 s。判据：`Curriculum/terrain_levels/mean` 爬起来（注意它被速度课程卡着，`lin_vel_x_max` 仍为 0 时地形等级不会动）、跟踪分与倒地率不比 Flat 差太多、`Rough/step_up_detect_rate` 在台阶列非零 。2026-09-07 01:0x 按用户指令在 1092 轮停止（13 个 checkpoint）。**崩溃，无效**：986 步的 episode 在 1000 轮内掉到 23 步，`catastrophic_state` 终止 0.7 → 386，`Loss/value` 2.2 → 3.1 万，地形等级全被降回第 0 行。根因是把 `TerrainHeightSensor` 的逐射线净空当机体高度：机身陷进 hfield 后 backface 把读数钳成 0.0，无界的 `flat_base_height`（`(err/0.05)² × -4.0`）给出 -144/s（正常总奖励约 -10/s），critic 目标进万级。日志同期刷了 3 万次 hfield 接触溢出。SIGINT 120 s 未退，SIGKILL 生效 |
| R2 | `SE3-WheelLegged-Rough` / `0fe64d9` | `rough-R2-heightfix-hf02-6x8192-5k`（W&B `32eentyo`） | 171303 | `20260906T165641Z-23268` | 2026-09-07 00:56 启动，5000 轮，ETA 4:20。相对 R1 三处修复（见 commit `0fe64d9`）：机体高度改走 `mdp/terrain_height.py` 的地面估计口径（有效射线均值 + env_origins 兜底，平地逐位等价）、`flat_base_height` 加 `max_error=0.15` 误差夹紧（单项罚上限 -36/s）、hfield `horizontal_scale` 0.1 → 0.2（网格 90² → 45²，接触不再溢出）。启动后 40 轮：hfield 溢出 0、contact 溢出 0、`catastrophic_state` 0、迭代 3.11 s（R1 3.35 s）。判据同 R1，外加 `Loss/value` 全程不得进入千级 |

| R3 | `SE3-WheelLegged-Rough` / `938d0a0` | `rough-R3-forwardcmd-chebyshev-noDemote-exitTrunc-6x8192-5k` | 175757 | `20260907T064851Z-30246` | 2026-09-07 14:48（Pod 时区）启动，5000 轮。R2 诊断：课程停在 1.5 不是策略退缩，是位移判据在对称指令下是随机游走（平地列也只到 1.6）。相对 R2 改动五项（见 commit）：非平地列只发前向直行指令 vx∈[0.4,2.4]、yaw ±0.2、无静站（平地列沿用 Flat 速度课程）；课程只升不降；升级判据改切比雪夫距离 ≥4.0 m（越过最外一级台阶）；新增 `terrain_cleared` 截断（L∞>4.25 m 出块即 time_out）；stairs_up/slope_up 改反金字塔（此前列名与出生方向相反）；`Rough/` 日志键进白名单。判据：`Curriculum/terrain_levels/stairs_up`（现在才是爬台阶列）1000 轮内明显超过 R2 的 2.3；`Episode_Termination/terrain_cleared` 非零；`Rough/step_up_detect_rate` 台阶列非零；`wall_blocked` 比 R2 下降。2026-09-07 16:3x 在 1114 轮按用户指令停止（SIGINT 120 s 未退，SIGKILL；组长 PID 留为僵尸，显存已回 2 MiB）。**结论**：其余四列正常升级（slope_up 4.9），但 stairs_up 死在第 1 行（4 cm）：model_500 本机回放显示策略在坑底平台边缘 0.9–0.99 m 反复撞退，6 cm 轮子靠滚动翻不过 4 cm 竖直沿；另两处副作用：平地速度课程门看全体跟踪分被地形列拖在 0.3，`lin_vel_x_max` 整场 0；地形列从第 0 轮给满 0.4–2.4，500 轮策略对 vx≥1 原地不动。首轮 level 全为 1.0 是去掉首次 reset 保护的 bug（6184d51 已修） |
| R4 | `SE3-WheelLegged-Rough` / `2fde172` | `rough-R4-ctbc-flatgate-vxramp-6x8192-5k`（W&B `6qvid339`） | 178804 | `20260907T084146Z-03607` | 2026-09-07 16:41（Pod 时区）启动，5000 轮。相对 R3 五项：首次 reset 不结算课程（6184d51）；移植 stair 线 CTBC（3d4da2b，79b7386：只在 stairs_up 列触发，500 轮前满幅、500→1500 线性退火、之后关闭；actor 3 维扩展槽 jump_commands→ctbc）；平地速度课程只看平地列跟踪分；地形列 vx 上限跟随平地课程（起点 0.4 定速）。判据：`Curriculum/command_vel/lin_vel_x_max` 在 500 轮内爬到 2.4（R3 恒 0）；`Rough/ctbc_trigger_rate` 在退火前非零、`Rough/ctbc_kff` 500→1500 线性降；`Curriculum/terrain_levels/stairs_up` 1500 轮前离开第 0–1 行，且退火结束后不回落。2026-09-07 21:3x 跑满 4999 轮 finished。**结论**：其他五列全部到顶（level 5.6–6.6 = 随机重掷均衡位，与平地列同），stairs_up 200 轮起冻在 2.1（6 cm）零升级；CTBC 满幅时 6 cm 只能抬一侧轮子；且基础行走退化：每步跟踪分只有 R2 的 0.4–0.6，4000 轮策略在平地 vx 0.7 不走、坑底也不走；100–700 轮 catastrophic 0.2–0.4（R3 为 0），怀疑第 0 轮起注入 CTBC 把策略教成回避接触 |
| R5 | `SE3-WheelLegged-Rough` / `91a1c72` | `rough-R5-critic-heightscan-6x8192-5k`（W&B `osb92rg1`） | 183144 | `20260907T134626Z-11659` | 2026-09-07 21:46（Pod 时区）启动，5000 轮。相对 R4 唯一变量：critic 特权观测加 77 点地形高度扫描（照 yly-true/fudan_rl_wheel_leg，机身系 x ±0.5/y ±0.3 m 网格，值为各点相对脚下的抬升，打空/自击记 0），actor 34 维不变，critic 87→164。CTBC 与课程配置同 R4。判据：`Loss/value` 与跟踪分相对 R4 的变化；stairs_up 是否离开 2.1；每轮时间相对 R4 的 3.3 s（射线 19→96 条/env）。迭代 3.42 s（+4%）。2026-09-07 22:3x 在 665 轮按用户指令停止（SIGKILL）：与 R4 同样的早期退化（tracking_lin_vel 0.21，R4 同期 0.6，R3 1.0），critic 观测没改变走势。ONNX 已拉回本机 logs/rsl_rl/SE3-WheelLegged-Rough/<run>/onnx/（R5 model_600、R4 model_900 与 model_4999）。CTBC 前馈换算链已核对：方向/侧别/FK/IK/joint 语义换算全部正确，指令抬 12 cm 实际 19 cm 是气弹簧+PD 稳态偏差（零动作稳态角与指令默认差 0.3 rad），非 CTBC 问题 |

| R6 | `SE3-WheelLegged-Rough` / `0f781eb` | `rough-R6-noctbc-heightscan-6x8192-5k`（W&B `w96f0fip`） | 185782 | `20260907T144855Z-05791` | 2026-09-07 22:49（Pod 时区）启动，5000 轮。= R5 关掉 CTBC（默认 ctbc_enabled=False）：R3 的分列直行指令/只升不降/切比雪夫清块/出块截断/列名改正，加已知课程修正（首次 reset 不结算、平地速度门只看平地列、地形列 vx 随课程），加 critic 77 点高度扫描；无前馈。判据：基础行走恢复到 R3 水平（每步跟踪分 ≥1.2e-3、100–700 轮 catastrophic≈0、model_500 平地 vx 0.5–0.7 能跑）；stairs_up 预期仍卡在 ≤2 cm，本轮不考核台阶。2026-09-07 23:1x 在约 300 轮按用户指令停止。**结论**：catastrophic 回到 R3 量级（最高 0.21/窗口，多数为 0），确认 R4/R5 的 catastrophic 来自第 0 轮注入 CTBC；但 250 轮地形均值冲到 4.2 后出现『站着不动』（is_alive 0.98、清块 24→0.3/窗口、每步跟踪分 0.53e-3→0.29e-3），与 R4 同型。共同点：地形列 vx 上限随课程（≤1.6）比 R3 的 0.4–2.4 好跟，清块快、课程升得快，策略在会稳走之前被推上难地形 |
| R7 | `SE3-WheelLegged-Rough` / `8cb40b1` | `rough-R7-flatwarmup500-thr075-6x8192-5k`（W&B `qydz5fxo`） | 188110 | `20260907T151644Z-08142` | 2026-09-07 23:16（Pod 时区）启动，5000 轮。相对 R6 两个变量（用户定）：前 500 轮全部 env 在平地列（之后各 env 在下一次 reset 时换回原列第 0 行，速度课程延续不重置），Flat 速度课程推进阈值 0.5→0.75。判据：500 轮时平地跟踪达到 Flat 基线量级；换列后 catastrophic 不抬头、清块率不归零、跟踪分不腰斩；`Curriculum/flat_warmup/active` 在 500 轮后一个 episode 内降到 0。**结果**：Pod 上跑满 4999 轮（model_4999.onnx 已导出），但 W&B 只有到 952 轮、状态 crashed——笔记本 gateway 19:54 起停止刷新，上传卡住，进程挂在 W&B 收尾占着显存约 6 h；2026-09-08 09:3x 清掉。曲线（到 952 轮）：热身期 catastrophic 0、reward 88；500 轮换列后 166 轮内地形均值冲到 3.6、跟踪分从 2.35e-3 掉到 0.79e-3、catastrophic 0.2，stairs_up 卡 1.19（4 cm）——热身只推迟了『升级过快→策略退化』，没解决；台阶列仍是物理上限。结论：转 AMP |
| A1 | `SE3-WheelLegged-Rough-AMP` / `9378ec9` | `rough-A1-amp-stairsup-flatwarmup500-6x8192-5k`（W&B `4skuhyag`，project SE3-WheelLegged-Rough） | 192457 | `20260908T013539Z-26867` | 2026-09-08 09:35（Pod 时区）启动，5000 轮。= R7 配置 + AMP（judge 只对 stairs_up 列：amp_mask 观测组；示范 assets/amp/fudan_stairs20_20260907/amp_training.pkl，20 段复旦 12–20 cm 爬阶，含镜像 40 段 2702 帧，未重定向到 SerialLeg；reward_weight 3.0、热身 100 次判别器更新、每轮 2 步、batch 4096、lr 1e-4、R1 10）。数据集 pkl 在 Pod 上是未跟踪文件，已写进 .git/info/exclude 以通过启动器的干净度检查。判据：500 轮换列后 `Loss/amp/style_reward` 落在 (0,1) 中间而不是贴 0；`Curriculum/terrain_levels/stairs_up` 离开 1.2；换列后跟踪分不腰斩 | **结果**（跑到 1599 轮时人工停，2026-09-08 11:0x；下面数字取 1073 轮）：跟踪分守住 ~1.8e-3/步、catastrophic≈0、其他列升满，stairs_up 冻在 1.07；AMP：style_reward 0.60、expert 0.22 / policy −0.25、disc loss 1.31——判别器没把两边分开也没被骗。专家 vs 策略特征统计：base_velocity_x 0.62 vs 0.03、base_velocity_z 0.12 vs 0、omega_y std 1.27 vs 0.30、轮心 vx/vz std 0.6–0.9 vs 0.15、wheel_spin 22±41 vs 0.8±4.5（打滑/悬空读数，非风格）；轮心 x/z 几何一致，两车同尺寸不需重定向。结论：去轮速、加长窗口 → A2 |
| A2 | `SE3-WheelLegged-Rough-AMP` / `2fe15f3` | `rough-A2-amp-nospin-5frames-6x8192-5k`（W&B `up8z7hpe`，project SE3-WheelLegged-Rough） | 195442 | `20260908T031013Z-17934` | 2026-09-08 11:10（Pod 时区）启动，5000 轮。相对 A1 一个变量『判别器输入』：帧去掉 left/right_wheel_spin（19→17 维，`AMP_DISCRIMINATOR_FIELDS`，env 观测组与数据集 fields 同一份）+ 窗口 transition_frames 2→5（40→100 ms，判别器输入 85 维）。判据同 A1：换列后 style_reward 在 (0,1) 中间且 policy_score 向 expert 靠；`Curriculum/terrain_levels/stairs_up` 离开 1.07；跟踪分不腰斩 | **结果**（862 轮时用户停，2026-09-08 12:0x）：热身期 vx 上限推到 1.2（A1 1.0，种子差异），520–540 轮换列时 catastrophic 计数 0.8–1.2/步（≈25 env/轮，A1 为 0），650 轮回 0；掉分主要是机身高度/腿蹭地/碰撞惩罚翻倍。830 轮：每步任务奖励 0.035（A1 0.053）、stairs_up 1.22（A1 1.06）、style_reward 0.58、expert 0.20 / policy −0.29、disc loss 1.27——判别器比 A1 分得稍快但仍在梯度最大区间。台阶列 AMP 每步 +0.035（上限 0.06）与三项地形惩罚（−0.03～−0.05/步）同量级、净收益≈0 → A3 加权重 |
| A3 | `SE3-WheelLegged-Rough-AMP` / `b5e11e3` | `rough-A3-amp-w15-6x8192-5k`（W&B `ph7qi5q1`，project SE3-WheelLegged-Rough） | 198090 | `20260908T041041Z-20285` | 2026-09-08 12:10（Pod 时区）启动，5000 轮。相对 A2 一个变量：AMP `reward_weight` 3.0→15.0（每步上限 0.06→0.30，约为任务奖励的 8 倍）。判据：换列后台阶列策略愿意抬轮/俯仰（腿蹭地、碰撞惩罚上升但 style_reward 上升、policy_score 向 0 靠）；`stairs_up` 脱离 1.2；平地/其他列跟踪不因 AMP 权重变差（AMP 只在 stairs_up 列，应无影响） | **结果**（702 轮时用户停，2026-09-08 13:0x）：换列不摔（catastrophic 全程 0，高度/碰撞惩罚只有 A2 一半），style_reward 0.65、policy_score −0.18（A2 0.60 / −0.25）——策略更像专家；但 stairs_up 1.03（560 轮起冻住，A2 同期 1.21），slope_up 2.87（580 轮起冻住，A2 6.3），random_rough 3.7（A2 6.8），每步清块 3.1（A2 7.6）。读法：AMP 只在 stairs_up 列，但 actor/advantage 归一化全列共用，大奖励外溢；也可能是种子噪声，单点分不清。用户改地形分配 → A4 |
| A4 | `SE3-WheelLegged-Rough-AMP` / `5214450` | `rough-A4-amp-w15-upcols-6x8192-5k`（W&B `o6k2mf55`，project SE3-WheelLegged-Rough） | 200745 | `20260908T051302Z-23302` | 2026-09-08 13:13（Pod 时区）启动，5000 轮。相对 A3 一个变量：地形列 env 分配 flat 25 / stairs_up 20→**40** / stairs_down 20→**0** / slope_up 15→**30** / slope_down 15→**0** / random_rough 5（mjlab 每列仍留 1 个 env，列与日志键不变；每卡 stairs_up 3275、slope_up 2456、flat 2047、random_rough 410、下行列各 1）。判据：AMP 样本占 40% 后 stairs_up 是否脱离 1.2；slope_up 不再冻在 2.9；flat 跟踪不变 | **结果**（2757 轮时用户停，2026-09-08 16:0x）：热身期速度课程只在 350 轮推了一档到 0.2（EMA 擦线 0.751），随即平地上一波翻倒（bad_orientation 10/步），EMA 掉回 0.6–0.74 贴阈值下方，1000 轮才到 0.4——整轮指令 vx≤0.4，失去对照价值。换列时 75% env 一起掉坑，catastrophic 9/步、530 轮归零；stairs_up 1.02 不动，slope_up 5.98、random_rough 6.37；style_reward 0.74（四轮最高）。15:25–15:47 入口机 2222 拒绝 22 min，W&B 标 crashed，gateway 自动重开隧道后补传、状态回 running。**根因**：R7 起阈值 0.75 而 vx=0 阶段 EMA 天花板 0.72–0.75，首次推进 200–350 轮且靠种子；Flat D10（阈值 0.5，W&B Flat/2pilnc2r）45 轮推进、350 轮到 2.4、1000 轮追平；前 50 轮 EMA 曲线两边完全重合 → 改回 0.5 |
| A5 | `SE3-WheelLegged-Rough-AMP` / `b9aaa7b` | `rough-A5-amp-w15-upcols-thr05-6x8192-5k`（W&B `koqt59w7`，project SE3-WheelLegged-Rough） | 204304 | `20260908T080745Z-15051` | 2026-09-08 16:07（Pod 时区）启动，5000 轮。相对 A4 一个变量：速度课程推进阈值 0.75→0.5（Flat 默认）。判据：热身期 vx 在 100 轮内过 1.0、350 轮左右到 2.4，499 轮跟踪≈D10 同期（每秒 2.1）；换列后台阶列指令上限 2.4，stairs_up 是否脱离 1.2。**热身正确、换列后仍塌**：499 轮 vx 上限 2.4、每步 0.093，与 Flat D10 同期一致，证明阈值就是之前卡住的唯一原因；但换列后跟踪从 2.2 掉到 0.5 再没回来，清块 1–3/步，stairs_up 560 轮冲到 1.39 后到 2000 轮一直冻着，style_reward 0.74→0.60 而 policy_score −0.10→−0.27（判别器在拉开距离），全程 catastrophic 为 0。2026-09-08 19:1x 在 3200 轮按用户指令 SIGINT 停止（6 卡显存回 2 MiB，checkpoint 与 ONNX 齐到 model_3200）。诊断：`tracking_lin_vel` 高斯核 σ_move=0.08 在误差 >0.4 m/s 处没有梯度，台阶列换列后长期落在那一段 |
| A6 | `SE3-WheelLegged-Rough-AMP` / `1bdc226` | `rough-A6-amp-floor-cmderr15-noheightpen-6x8192-5k`（W&B `z9vexbn3`，project SE3-WheelLegged-Rough） | 208077 | `20260908T111457Z-32033` | 2026-09-08 19:15（Pod 时区）启动，5000 轮。相对 A5 三个变量（用户定，一次性打包，不是单变量对照）：① 删掉 step_up 前瞻状态机，台阶抬升改用地形感知高度下限（按列按行把高度指令采样下界顶到 `台阶高 + 0.02 + 0.12`，stairs_up 行 0-1 不生效、行 9 为 0.34 m，上界仍 0.38），随之删掉 `wheel_forward_sensor`、`wall_blocked` 终止与 `-NoStepUp` 入口；② `command_velocity_error` 只在 stairs_up 列加回来（权重 -2.0、死区 (0.05,0.10)、封顶 9 用历史值，唯一改动是 `lin_vel_scale` 0.5→1.5——0.5 在误差 1.55 m/s 就顶封顶，而台阶列误差长期 1.5–2.4，会全程贴封顶：既没梯度，18/s 的常数负奖励又大过全部正项约 10/s，早终止在数值上更优；1.5 让误差 1.5/2.0/2.4 分别对应 -1.9/-3.4/-4.9 每秒，全程留在二次区间）；③ `flat_base_height` 在 stairs_up 列置零。注意 `height_conditioned_action_default` 在 flat/rough 线是 False、`stand_still` 要 ‖cmd[:2]‖≤0.1（台阶列 vx≥0.4 永不触发）、`joint_pos_penalty` 权重为 None，所以 ③ 之后 stairs_up 列没有任何奖励再读高度指令，① 的下限只剩观测这一条通道。启动验证：iteration 3 时 `Episode_Reward/command_velocity_error` 为 0（热身期全员平地列，掩码正确关着）、`flat_base_height` -0.82（平地列照常生效）、amp 观测 17 维、6 卡 28.7 GB/82–85%。判据：换列后跟踪分是否还塌到 0.5；`Rough/command_velocity_error_terrain` 是否停在 1–5 的量级而不是贴封顶 9；`Episode_Termination/*` 不因常数负奖励而抬头（尤其看存活步数是否缩短）；stairs_up 是否离开 1.39；`Rough/height_cmd_terrain_mean` 随台阶行数上抬。**两个机制性担心都排除，但没解决根因**：`Rough/command_velocity_error_terrain` 全程 0.92–1.08（封顶 9 一次没贴），×2.0 = −1.9~−2.2/s，与按 scale 1.5 预算的 −1.9 吻合，说明 0.5→1.5 是对的；存活 0.932 对 A5 的 0.942，episode 930/1000，没有出现担心的「早死更优」。stairs_up 600 轮冲到 1.675 后冻住（A5 是 1.391 冻到 2100、2400 才 1.765），领先一个身位但同样冻着；按 is_alive 归一化的每秒跟踪分到 1200–1299 轮已与 A5 追平（0.696 对 0.713），catastrophic 0.034 对 A5 0.013（绝对值仍小）；`Rough/height_cmd_terrain_mean` 0.292 = 采样区间中点，说明地形感知高度下限在 stairs 等级 1.67（第 1–2 行、台阶 5–6 cm）算出的下限 0.20–0.214 根本没顶起来，这一项目前是空转。2026-09-08 21:0x 在 1800 轮按用户指令 SIGINT 停止（checkpoint 与 ONNX 齐到 model_1800，六卡显存回 2 MiB）。**本轮最大的收获是本机 sim2x 确定性回放的诊断**（平地、无探索噪声）：A6 model_500 在 2 m/s 上误差 0.02，换列 100 轮后的 model_600 只跑得出 1.07 m/s（误差 0.93）、model_1000 在指令 2.0 下只有 0.29 m/s；A5 同型（model_1000 误差 1.80，到 model_2000 才恢复到 0.13）。所以「换列后跟踪暴跌」不是 episode 变短的记账假象，策略是真坏、连纯平地都坏，且与 A6 的三处改动无关——是换列本身。机制：核 exp(-(err²+2vz²)/0.08) 在误差 0.8 以上恒为 0（既无分也无梯度），而罚项一个不少，那七成 env 上唯一有梯度的方向是「把动作压小、别动」，共享 actor 学进去后平地一起退化；与 R4/R6 记的「地形均值冲到 4.2 后站着不动」是同一个东西 |
| A7 | `SE3-WheelLegged-Rough-AMP` / `a1ebf8f` | `rough-A7-amp-vx08-ramp500-novz-6x8192-5k`（W&B `vs7p5uuc`，project SE3-WheelLegged-Rough） | 211078 | `20260908T130510Z-18077` | 2026-09-08 21:05（Pod 时区）启动，5000 轮。按 A6 的回放诊断改四处（用户定，三个训练变量一次打包，非单变量）：① 地形列 vx (0.4,2.4)→(0.4,0.8) 且与平地课程脱钩（`terrain_lin_vel_x_follow_curriculum=False`）——A6 的 `Rough/command_velocity_error_terrain`=0.95 反解出地形列误差约 1.5 m/s、指令均值 1.4，误差和指令一样大，原来上限跟随**平地**课程（350 轮就到 2.4），等于给了一个做不到的数；② 换列改线性 ramp，地形 env 比例 500→1000 轮从 0 涨到 1（逐 env 固定阈值、单调不回头，`ROUGH_FLAT_WARMUP_RAMP_ITERATIONS=500`）；③ 非平地列关掉核里的 vz 项（`terrain_vz_weight=0`，平地仍 2.0）——爬升必须有垂直速度而核按 vz² 扣分；④ 日志：`Locomotion/` 六键进白名单 + 新增按列拆开的 `Rough/tracking_lin_vel_{terrain,flat}`、`Rough/{cmd_vx,base_vx,base_vx_error}_terrain`（A6 时这些只能靠本机回放反推）。启动验证：iteration 3 时 `Curriculum/flat_warmup/progress`=0、`active`=1（全员平地，ramp 在起点），三个 `Rough/*_terrain` 键均为 0（地形列暂无 env，掩码正确），amp 观测 17 维，6 卡 28.7 GB/84–93%。判据：**`Rough/base_vx_error_terrain` 是否落进 0.5 以内**（A6 反解值 1.5，这是本轮的核心判据）；`Rough/tracking_lin_vel_terrain` 是否明显非零；`Locomotion/base_vx_error_abs` 在 500–1000 轮 ramp 期间是否不再出现 A6 那样的断崖；平地能力用 sim2x 回放 model_1000/1500 复测 2 m/s 误差（A6 同期是 1.71）；stairs_up 能否离开 1.675。**核心判据全过，只剩台阶列没动**：`Rough/base_vx_error_terrain` 0.375→0.370（判据是「落进 0.5 以内」，A6 反解值 1.5），`Rough/tracking_lin_vel_terrain` 0.43→0.454（A6 约 0），是同期 `_flat` 0.653 的 70%；`cmd_vx_terrain` 0.598（设计值 0.6）、`base_vx_terrain` 0.278（A6 约 0，机器人真的在地形上前进了）；`catastrophic_state` 全程 **0**、`leg_contact` 0，清块 7.4–8.0（A6 只有 1–2.6）；地形均值 4.17、slope_up 6.39、flat 6.1。`is_alive` 0.66 低于 A6 的 0.91 但不是坏事——终止构成里是清块截断在拉，不是摔。**平地塌陷彻底消失**：sim2x 确定性回放（平地、无噪声）model_500/700/1000/1200/2700 在 2 m/s 上的误差是 0.14/0.04/0.09/0.14/**0.03**，全程没离开 0.04–0.14；A6 峰值 1.71、A5 峰值 1.80。model_2700 是这条线最好的一个。**唯一失败项**：stairs_up 1250 轮 1.092 → 2768 轮 1.108，1500 轮几乎不动，排除了 ramp 造成的到岗延迟，坐实是 6 cm 轮子靠 0.8 m/s 的动量翻不过 4 cm 立面——这正是提方案时写明的代价（台阶列不再练高速冲坡）。2026-09-09 00:0x 在 3113 轮按用户指令 SIGINT 停止（checkpoint 与 ONNX 齐到 model_3100，六卡显存回 2 MiB）。本机已拉回 model_500/700/1000/1200/2700 |
| A8 | `SE3-WheelLegged-Rough-AMP` / `e5ff896` | `rough-A8-amp-stairvx24-h035-6x8192-5k`（W&B `udfklqkv`，project SE3-WheelLegged-Rough） | 214673 | `20260908T160301Z-25673` | 2026-09-09 00:03（Pod 时区）启动，5000 轮。相对 A7 一个改动（用户定）：把 `stairs_up` 从通用地形列拆出来单独定价——vx 1.0–2.4（与专家数据 Fudan 12–20 cm 爬升的 1.5–2.4 同段）、机身高度指令 0.35–0.38；其余地形列（斜坡、起伏）保持 A7 的 (0.4, 0.8) 低速档，平地列不动。实现：vx 走 `refresh_terrain_override` 里压在通用覆盖之上的二次 `set_velocity_ranges`（必须后设），高度走 `_apply_stair_height` 在基类采样之后按列改写，并同步刷新高度条件默认腿姿缓存。两个已知副作用：① 台阶列高度下界 0.35 高于地形感知抬高下限在最高难度行的 0.34，A6 引入的那条下限在本列被完全吞掉；② vx 1.0–2.4 把台阶列推回「指令不可达 → 核 exp(-err²/0.08) 恒零」的区间（`base_vx_terrain` 0.28 对指令均值 1.7，误差约 1.4），与 A6 的区别是只有 40% 的 env 在其中（A6 为 75%）、迁移有 ramp、且 `command_velocity_error` 专补远场梯度（封顶约 −4.9/s）。另注：0.35–0.38 显著高于专家的 0.22–0.27，AMP 风格奖励与任务指令的姿态目标更不一致，本轮未动 AMP。启动验证：iteration 1 时 `flat_warmup/progress`=0、`active`=1，`Rough/*_terrain` 均为 0（地形列暂无 env），`tracking_lin_vel_flat` 0.84，amp 观测 17 维，6 卡 28.7 GB/85%。判据：**stairs_up 能否离开 1.11**（本轮唯一目标）；同时守住 A7 已经拿到的东西——`Rough/tracking_lin_vel_flat` 不掉、sim2x 平地回放 2 m/s 误差保持 0.1 量级、`catastrophic_state` 不抬头；`Rough/base_vx_terrain` 在台阶列会被高指令拉高还是仍趴在 0.28。**指令翻倍但速度纹丝不动**：`cmd_vx_terrain` 0.598→1.178，而 `base_vx_terrain` 0.278→0.285，即「设到什么速度域都不动」；从罚值反推台阶列基本静止（静止对应罚 1.21，实测 1.2414）。`base_vx_error_terrain` 0.370→0.947，核 exp(-0.95²/0.08) 归零。stairs_up 1.108→1.386（1139 轮，ramp 已满）。平地未被带塌但也没长进：sim2x model_700 在 2 m/s 上误差 0.12，`tracking_lin_vel_flat` 0.577；catastrophic 全程 0。**逐项排查**：`flat_wheel_contact`(−0.27) gate 在 ‖cmd[:3]‖<0.08 的 idle 上，台阶列指令 1.0–2.4 永不触发，全是平地列贡献；`flat_base_height` 台阶列已置零（A6）；`flat_leg_contact`/`contact_forces`/`dof_pos_limits`/能耗三项 ≤0.007。唯一能定量的是 `command_velocity_error`：台阶列 1.2414×2.0 = **−2.48/s**，整张表最大单项，且是 A8 把台阶列 vx 0.8→2.4 而没有跟着重新定标 `lin_vel_scale`(=1.5，按 A7 的 0.4–0.8 定的) 造成的，约 5 倍于 A7。但它的梯度指向「跑快」（0→0.6 m/s 净赚 +1.44/s），单独解释不了原地不动；其余罚项按列的读数当时没有日志，查不下去。2026-09-09 01:1x 在 1375 轮按用户指令停止（checkpoint 到 model_1300）。本机已拉回 model_700 |
| A9 | `SE3-WheelLegged-Rough-AMP` / `5a515a7` | `rough-A9-amp-stairs70-rwsplit-6x8192-5k`（W&B `a40r9aa1`，project SE3-WheelLegged-Rough） | 217451 | `20260908T171916Z-09758` | 2026-09-09 01:19（Pod 时区）启动，5000 轮。相对 A8 两处（用户定）：一个训练变量 + 一份纯诊断。① 地形 env 分配改 flat 30% / stairs_up 70%，slope_up、random_rough、两个下行列 proportion 全设 0（列与日志键保留，mjlab 每列仍留 1 个 env）；ramp 沿用 500→1000，实测 512 env 下 750 轮 35% 台阶、1000 轮 69.3%/29.9%。顺带修掉 A8 的观测问题——slope_up 爬到 6.4 级正常清块，把 `Rough/*_terrain` 的「非平地」口径整个稀释了，slope 一去该口径实质等于台阶列。② 新增 `events.log_reward_split_by_column`：每步把奖励表**每一项**在台阶列上的均值记一份（`Rough/rw_<项名>_stairs`，23 项），直接读 RewardManager 的 `_step_reward` 缓冲做掩码均值，不重算奖励、不改奖励数学。本地 24 env 零动作冒烟已看到台阶列 `command_velocity_error` −3.34/s、`tracking_lin_vel` 0.00002、`flat_base_height` 0。**未改**：`command_velocity_error` 的 `lin_vel_scale` 仍是 1.5（用户定：先加日志定位再决定是否重新定标）；台阶列占比 40%→70% 后该项的全体均值会从 −0.66/s 涨到约 −1.74/s。启动验证：iteration 1 时 `flat_warmup/progress`=0、`active`=1，全部 `Rough/rw_*_stairs` 为 0（台阶列暂无 env，掩码正确），amp 观测 17 维，6 卡 28.7 GB/85–86%。判据：**看 `Rough/rw_*_stairs` 里哪一项在台阶列最负**——这是本轮唯一目的；其次 stairs_up 能否离开 1.39、`Rough/base_vx_terrain` 能否离开 0.28。**跑满 5000 轮，分列日志给出确诊**。台阶列（4800–4999，每秒）：正项 3.753 = `tracking_ang_vel` **2.739** + `is_alive` 1.000 + `tracking_lin_vel` 0.015；负项 3.700 = `command_velocity_error` **2.496** + `action_rate` 0.327 + `tracking_orientation_l2` 0.245 + `action_smoothness` 0.228 + `flat_wheel_contact` 0.150 + `collision` 0.124 + 其余 0.130；**净 +0.053**。即原地不动是稳定的正收益均衡：`tracking_ang_vel` 占该列正奖励的 73%，而它**静止就能拿满**（yaw 指令 ±0.2、σ=0.25，不动时 yaw 恒 0、误差约 0.1、核 0.96），配上 is_alive 的 +1.0 白拿 3.74/s；爬台阶要拿摔倒风险去换 `command_velocity_error` 那点梯度（0.05→0.6 m/s 净赚 +1.30/s），理性选择就是不动。台阶列实际 vx 全程 **0.052**（指令 1.70）且随训练下降（0.056→0.054→0.052），`tracking_lin_vel` 仅 0.015/4.0，核彻底死掉；stairs_up 1.269→1.590，catastrophic 0.017→0.032，AMP style 0.646→**0.494**（专家在爬、策略在站，判别器越拉越开）；平地列不受影响（`tracking_lin_vel_flat` 0.624→0.689）。**另一条直接对照**：A7 台阶列指令 0.6 时实际 vx 是 0.278，A8/A9 提到 1.7 后掉到 0.052——把指令提高反而慢了五倍，因为唯一奖励「移动」的 `tracking_lin_vel` 被推出了 exp 核的有效区 |
| A10 | `SE3-WheelLegged-Rough-AMP` / `7dab497` | `rough-A10-amp-noyawwage-cmderr3-6x8192-5k`（W&B `pkdl1i8k`，project SE3-WheelLegged-Rough） | 221719 | `20260909T043441Z-08537` | 2026-09-09 12:34（Pod 时区）启动，5000 轮。相对 A9 两处（用户定，按 A9 确诊直接对症）：① 台阶列取消「不动的工资」——`stair_ang_vel_yaw_range=(0,0)` 固定 yaw 指令为 0（指令侧），并把 `tracking_ang_vel` 换成 `rewards.tracking_ang_vel_off_terrain` 在台阶列置零（奖励侧）；两侧都要，缺一份工资就还在。实测台阶列该项 0.00000、平地列 1.66841，平地逐位不变。② `command_velocity_error` 的 `lin_vel_scale` 1.5→3.0。理由是①把台阶列正项从 3.753 压到 1.015 之后，若仍留着 −2.496 的罚，净收益就是 **−2.69/s**——整段 20 s episode 累计约 −54，而首步摔倒终止 ≈0（catastrophic 为失败终止，PPO 按 0 值 bootstrap），故意撞死在数值上严格占优；这是 A6 就写下的风险，此前靠 +3.74 的正项垫着没发作。3.0 之后误差 1.65→−0.57/s、满指令不动 2.35→−1.23/s，台阶列净约 −0.76/s：仍为负（必须真的往前爬拿到 `tracking_lin_vel` 才转正，正是要的梯度方向），但不诱发自杀。测试加硬约束：该项最坏情况乘权重必须 < 1.5/s。未改：stairs_up 的 vx 仍 1.0–2.4、高度仍 0.35–0.38、配比仍 flat 30 / stairs_up 70。启动验证：iteration 0 时 `flat_warmup/progress`=0、`active`=1，全部 `Rough/rw_*_stairs` 为 0，amp 17 维，6 卡 28.7 GB/83%。判据：**`Rough/base_vx_terrain` 能否离开 0.05**（本轮核心，A7 在指令 0.6 下是 0.278）；`Rough/rw_command_velocity_error_stairs` 应落在 −0.6~−1.2；`Episode_Termination/catastrophic_state` 是否失控上涨（自杀模式的判据，A9 是 0.032）；`Rough/rw_is_alive_stairs` 若跌破 1.0 说明 episode 在被主动缩短；stairs_up 能否离开 1.59 |
| J3 | `SE3-WheelLegged-Jump-Mimic-Exp-J3` / `4a43f13` | `jump-J3-mimic-vx15-seed42-6x8192-5k`（W&B `cjilu9wl`，project SE3-WheelLegged-Jump-Mimic） | 748772 | `20261001T101530Z-18005` | 2026-10-01 18:15（Pod 时区）启动，六卡 × 8192、5000 轮、seed 42、从头训。**在独立 worktree `/workspace/se3-worktrees/j3-4a43f13` 里跑**（主 checkout 停在 ly 的 `ly/policy-network-comparison@b4a379e`，不动）：代码经 bundle fetch 到 `refs/heads/se3-j3-sync`，`git worktree add --detach`，`.venv` 软链到主 checkout（两边 uv.lock / pyproject 相同；启动器设 PYTHONPATH=<worktree>/src，已核实 se3_train/se3_shared 从 worktree 导入），run 目录在 worktree 的 logs/ 下。启动前 GPU 0/1/3/4 被 4 个 A4-x 假占用：训练 03:5x 已跑满，卡在退出时的 W&B 上传（38443 网关自 09-29 起断），14 h 占着显存、0% 利用率；End+Run 网关后它们 18:13 前自行传完退出。首轮核验：第 37 轮无 Traceback、1.37 s/轮，六卡 6.1–6.5 GB/70–73%。2026-10-01 18:47 按用户指令在 1396 轮 SIGINT 停止（最后 model_1200），让给 J4 |
| J4 | `SE3-WheelLegged-Jump-Mimic-Exp-J4` / `6f558da` | `jump-J4-mimic-vx15-ref050-seed42-6x8192-5k`（W&B `6upe0z1l`，project SE3-WheelLegged-Jump-Mimic） | 751429 | `20261001T104827Z-02527` | 2026-10-01 18:48（Pod 时区）启动，六卡 × 8192、5000 轮、seed 42、从头训；= J3 + 0.50 m 参考。worktree `/workspace/se3-worktrees/j4-6f558da`（bundle 4a43f13..6f558da 更新到 `refs/heads/se3-j3-sync`，.venv 软链）。首轮：第 40 轮无 Traceback、1.37 s/轮 |
| Rough-obs30-1x | `SE3-WheelLegged-Rough` / `5fc61d4` | `rough-obs30-noattitude-seed42-1x8192-5k`（W&B `b7y13zoi`，project SE3-WheelLegged-Rough） | 777918 | `20261003T042941Z-16941` | 2026-10-03 12:29（Pod 时区）启动，GPU 0 单卡 × 8192、5000 轮、seed 42、从头训；与 whtws 七卡同配置（`tlfquma2`）对照多卡是否有效。worktree `/workspace/se3-worktrees/rough-obs30-5fc61d4`（bundle 0e5e63a..xyh/1002 到 `refs/heads/se3-obs30-sync`，.venv 软链，uv.lock / pyproject 一致，se3_train/se3_shared 从 worktree 导入）。启动前两条 W&B 网关都假活（日志停在 10-01 / 10-02），End+Run 后 nulltask1 恢复；whtws 的新网关因 38444 仍被旧 boring 会话占用而报 open failed，但该会话在用、tlfquma2 正常上传。首轮：第 13 轮无 Traceback、3.35 s/轮，GPU 0 11 GB / 84%；2026-10-03 3189 轮按用户指令 SIGTERM 停止 |
| Rough-Actor128 | `SE3-WheelLegged-Rough-Exp-Actor128` / `1545d9f` | `rough-obs30-actor128-seed42-5x8192-5k`（W&B `4am48uzq`，project SE3-WheelLegged-Rough） | 778989 | `20261003T045507Z-25107` | 2026-10-03 12:55（Pod 时区）启动，GPU 1–5 五卡 × 8192、5000 轮、seed 42、从头训（`-AllowConcurrent`，与 GPU 0 的单卡对照并行）；= 默认 rough（30 维 actor）+ actor 隐藏层 128/64/32、critic 512/256/128 不变。worktree `/workspace/se3-worktrees/rough-actor128-1545d9f`（bundle 5fc61d4..xyh/1002，.venv 软链）。首轮：第 9 轮无 Traceback、3.24 s/轮，五卡 12 GB / 82–92%；2026-10-03 2803 轮按用户指令停止 |
| MGPU-B | `SE3-WheelLegged-Rough` / `635be5f` | `rough-obs30-B-mb24-seed42-6x8192-5k` | 782023 | `20261003T080313Z-10186` | 2026-10-03 16:03（Pod 时区）启动，六卡 × 8192、5000 轮、seed 42、从头训，`--agent.algorithm.num-mini-batches=24`（每卡 mini-batch 8192，六卡平均后等效 4.9 万，与单卡相同；每轮梯度步数 20 → 120）；多卡改进实验 B，对照七卡 `tlfquma2`。worktree `/workspace/se3-worktrees/rough-mgpu-635be5f`（bundle 1545d9f..xyh/1002，.venv 软链）；2026-10-03 861 轮按用户指令 SIGINT 停止。结论：学习率被 KL 压到约 0.0005（七卡 0.0009），效果与七卡持平、前期更慢；actor 梯度噪声尺度从 2 万涨到 60 万，多出的梯度步被信赖域抵消 |
| Actor128-A | `SE3-WheelLegged-Rough-Exp-Actor128` / `635be5f` | `rough-obs30-actor128-split-seed42-6x1365-5k`（W&B `qtpyzxtf`，project SE3-WheelLegged-Rough） | 784532 | `20261003T090531Z-18498` | 2026-10-03 17:05（Pod 时区）启动，六卡 × 1365、5000 轮、seed 42、从头训；= whtws 实验 A（`tnt1j1nw`）只换 actor 隐藏层 128/64/32，单变量看小网络的最终上限。worktree `/workspace/se3-worktrees/rough-mgpu-635be5f`。首轮：第 8 轮无 Traceback、1.51 s/轮，actor 首层 30→128，六卡 4.6 GB / 63–68%，噪声尺度已记录 |

R 批实测的两条负结果：(1) mjwarp 的 broadphase 三档在本配置（8192 envs 单卡、383 个 geom）下
**默认的 nxn 最快**——94.4 / 118.5 / 141.6 ms/step（nxn / sap_tile / sap_segmented），不要换 SAP；
崎岖地形相对平地多出的迭代耗时在窄相（box + hfield 接触），不在宽相配对枚举。
(2) `contact_sensor_maxmatch` 只是溢出上界（`mujoco_warp/_src/sensor.py:2436`），不在热路径循环里，
64 → 500 只吃显存不吃时间。

σ 诊断结论见 `src/se3_train/tasks/flat/env_cfg.py` 的 `FLAT_ACTION_PENALTY_WHEEL_PRICING` 注释：
熵奖励按归一化维度给、(15/45)² 折价让轮噪声便宜 9 倍，平衡点 σ_wheel≈0.85-0.9；D4 预测≈0.28。
判据：D4 的 `Policy/wheel_std` 在 episode 长度打满（约 300-500 轮）后不再回升，与 D2 同轮次对比。

D4 最终结果（最后 500 轮均值，D4 对 D2）：轮 σ 0.256 对 0.922，腿 σ 0.212 对 0.285；mean_reward 81.8 对 60.2；
tracking_lin_vel 2.76 对 2.55，tracking_ang_vel 2.66 对 2.56，command_velocity_error -1.44 对 -1.49，flat_wheel_contact -0.44 对 -0.63，
ang_vel_xy -0.071 对 -0.096，action_rate -0.44 对 -0.66（轮分量定价高 9 倍仍更低）。存活与课程相同。
LR 自 3500 轮起 97% 轮次贴 1e-5 地板，Loss/value 中位数稳定 1.95 但 4250 轮后出现尖峰（最高 27，对 reward 无影响）。
最终 checkpoint `model_4999.pt` / `onnx/model_4999.onnx`。

D4 中期结果（4200 轮，W&B `istnlw0b`）：轮 σ 0.25 / 腿 σ 0.20，D2 同轮次 0.965 / 0.305；mean_reward 84 对 54，
tracking_lin_vel 2.78 对 2.52，command_velocity_error -1.42 对 -1.51，存活与课程相同。副作用：σ 缩到 0.25 以下后
KL 自适应学习率贴 1e-5 地板（3580 轮起 99% 轮次），训练进入爬行但 reward 仍缓慢上升。ONNX 每 100 轮落在
`logs/rsl_rl/SE3-WheelLegged-Flat-Exp-JointActionWheelPrice/2026-09-05_09-50-19_flat-D4-.../onnx/`，本机 4040 已验证 HTTP 200。

D6 终点评测（model_4999，确定性 sim2x 1 kHz）：站立腿峰峰 5.7°（D4 24.4°、D2 3.9°）、俯仰峰峰 3.8°，
但站立俯仰均值 +5.0°（D4 +1.5°、D2 −1.9°），腿被拉回默认姿态后机身用前倾配平；行进 1.0 m/s 腿峰峰 10.4°（D4 15.3°），
跟踪误差 0.04-0.09，无倒地。

## 在独立 worktree 里训练（主 checkout 被别人占着时）

启动器 `start-nulltask1-training.ps1` 自 2026-10-01 起有可选 `-Repository`（默认主 checkout），仓库检查改为 `test -e $repo/.git`（worktree 的 .git 是文件）。做法：bundle fetch 到一个新分支 ref（不动别人的 HEAD），`git worktree add --detach /workspace/se3-worktrees/<名字> <commit>`，`ln -sfn <主 checkout>/.venv <worktree>/.venv`（先确认两边 uv.lock 相同）。`ExtraArgs` 必须走 `pwsh -Command` 传 `@('--agent.seed=42','--env.seed=42')`，`-File` 传不了数组。

## Pod 内执行多行脚本、传文件、拉 ONNX（stdin 方式）

从本机 Git Bash 直接：

```bash
ssh laptop-wg "ssh -o BatchMode=yes -o ConnectTimeout=25 -p 2222 root@192.168.2.46 kubectl exec -i -n gczx-project06 nulltask1 -c container-1 -- bash -s" < script.sh
```

脚本全部经 stdin 进入 Pod 里的 `bash -s`，`|`、`$()`、引号、heredoc 都不会被 laptop-wg 的 cmd.exe/PowerShell 提前展开，
不需要 `scripts/remote_bash.ps1`。链路不稳，外面套 2-3 次重试并以脚本自己打印的结束标记判成功。三种常用脚本：

- **同步代码**：本地 `git bundle create x.bundle <pod_head>..spring_final`，`base64 -w0` 后作为 heredoc 嵌进脚本；
  Pod 内先核对 `HEAD == pod_head` 且工作区干净，再 `base64 -d`、`sha256sum -c`、`git bundle verify`、
  `git fetch <bundle> spring_final` + `git merge --ff-only FETCH_HEAD`。几 MB 以内十几秒完成（含二进制 STL 也可）。
- **停训练**：`ps -o args= -p <pgid>` 核对 run name 后 `kill -INT -- -<pgid>`，六卡 run 6 s 内退出、显存回 2 MiB；
  SIGINT 后 40 轮 3 s 未退再 SIGKILL。
- **拉 ONNX 到本机**：Pod 内 `echo "BEGIN <name> <sha256>"; base64 -w0 <file>; echo; echo "END <name>"`，
  本地正则切出、校验 sha256，写到 `logs/rsl_rl/<experiment>/<run>/onnx/`（与训练目录同形态），本机 sim2x 每秒重扫即可看到。
  单个 ONNX 约 737 KB，经 0.4 MB/s 的链路十几秒。

## sim2x 查看：当前做法与已停用的 4040 链路

**当前做法（2026-09-05 起）**：Pod 上的 sidecar 已按用户指令停掉，改在本机跑 sim2x：

```bash
uv run --no-sync --with-editable ./submodules/se3-sim2x se3-sim2x-browser --run-root logs/rsl_rl --host 127.0.0.1 --port 8080 --scan-interval 1.0 --physics-hz 500 --label local
```

浏览器打开 `http://127.0.0.1:8080/`，按 experiment / run_id / ONNX 切换；ONNX 用上一节的方法拉到
`logs/rsl_rl/<experiment>/<run>/onnx/`。定量评测用 `scripts/record_sim2x_video.py`（MP4 + 分阶段指标 + `--csv`，
`--action-noise` 可叠加训练式探索噪声）、`scripts/analyze_sim2x_sway.py`（站立摆动频谱/幅值）、
`scripts/eval_policy_mjlab_stand.py`（训练模拟器对照，`--stochastic` 按训练 σ 采样）。

**为什么放弃远程 4040**：Viser 每次连接要整包推送全部视觉网格。原始 SolidWorks 机身网格 13 块共 117 MB STL、234 万三角形，
Pod 内 sidecar 本地都只能约 1 MB/s 产出（物理线程占 GIL），再经 laptop→入口机约 0.4 MB/s 且会闪断的链路根本传不完。
仓库 841303b 已把视觉网格抽稀到 23.6 万面（动力学逐位不变），场景约 4 MB，但链路本身仍是瓶颈，本机直跑最省事。
D 系列 ONNX 需要子模块 `14eb892` 的 `serialleg_joint.v1` 解码器，旧 sidecar 加载不了。

**已停用但仍安装着的 4040 链路**（重新启用前先经 `pods/exec` 起 sidecar，run-root 必须是
`/workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg/logs/rsl_rl`，不能传上一级 `logs/`，否则报"未找到 ONNX"）：

| 项目 | 状态 |
|---|---|
| Pod sidecar | 已停（最后一次 state dir `/workspace/.se3-sim2x-state/nulltask1/20260905T0730Z-1khz-lod`，label `nulltask1-cpu-1khz-lod`，PID 137723 已退出） |
| 本机 4040 计划任务 | `SE3-Nulltask1-Sim2xLocalTunnel` 仍在，supervisor `C:\Users\13567\.local\bin\nulltask1-sim2x-local-tunnel.ps1` |
| laptop 14040 计划任务 | `SE3-Nulltask1-Sim2xTunnel` 仍在（2026-09-26 已改指向 .46） |
| 入口 socat 28080 | PID `2304800`（动态） |
| Pod TCP relay | `/workspace/.se3-sim2x-state/nulltask1/tcp_relay.py` |

当前 namespace 身份没有 `pods/portforward` 权限，标准 `kubectl port-forward` 会返回
`cannot create resource "pods/portforward"`。为保持 Viser 不对 Pod/集群网段开放，当前
链路改用已授权的 `pods/exec` 做逐连接 TCP bridge：

```text
本机 127.0.0.1:4040
  -> SSH -> laptop-wg 127.0.0.1:14040
  -> SSH -> 入口机 127.0.0.1:28080
  -> socat -> kubectl exec -i
  -> Pod 内 uv Python TCP relay -> 127.0.0.1:8080
```

所有监听均为 loopback。各层组件（历史记录，PID 动态）：

| 组件 | 位置 |
|---|---|
| 本机外层 SSH | 由 Scheduled Task `SE3-Nulltask1-Sim2xLocalTunnel` 守护（2026-09-02 起），supervisor 脚本 `C:\Users\13567\.local\bin\nulltask1-sim2x-local-tunnel.ps1`，每 15 秒检查 `4040` listener 并重连；PID 动态见 `C:\Users\13567\.local\state\nulltask1-sim2x\tunnel.pid`，日志 `supervisor.log` |
| laptop inner-tunnel task | `SE3-Nulltask1-Sim2xTunnel`，当前 `Running` |
| laptop supervisor | `C:\Users\Lenovo\.local\bin\nulltask1-sim2x-tunnel.ps1` |
| laptop inner-tunnel log | `C:\Users\Lenovo\.local\state\nulltask1-sim2x-tunnel\tunnel.log` |
| laptop 内层 SSH | 由 Scheduled Task 动态创建；以 `127.0.0.1:14040` listener owner 为准 |
| 入口 socat | PID `2304800`，监听 `127.0.0.1:28080` |
| 入口 listener supervisor | `/root/.local/bin/se3-nulltask1-sim2x-entry-server.sh` |
| 入口 bridge | `/root/.local/bin/se3-nulltask1-sim2x-entry-bridge.sh` |
| Pod TCP relay | `/workspace/.se3-sim2x-state/nulltask1/tcp_relay.py` |
| 本机入口 | `http://127.0.0.1:4040/` |

上述 PID 是动态运行状态，恢复或停止前必须同时核对 PID、command line 和监听端口，
不得按名称全局 `pkill`。HTTP 首页约 2.75 MB；经逐连接 exec bridge 首次加载较慢，
健康检查只读取响应头，2026-09-02 已实测返回 `HTTP/1.1 200 OK`。WebSocket 建立后
交互流量较小。browser 每秒重扫一次 ONNX，启动时为 `model_1000.onnx`，验证链路时
训练最新已到 `model_1400.onnx`，可在 Viser GUI 中切换。

入口机的 2222 端口会间歇出现 SSH banner timeout / connection refused。一次性
内层 SSH 因 `ServerAliveInterval=15`、`ServerAliveCountMax=3` 在约 45 秒失联后退出，
而本机外层仍会保留 `4040` listener，形成“端口在但页面打不开”的假健康。当前 laptop
Scheduled Task 每 15 秒单次重连；入口恢复后自动重建 `14040 -> 28080`。入口 server
脚本遇到已有且 command line 匹配的 socat listener 时会复用并保持 SSH session，只有
listener 真正消失后才创建新的，避免 `Address in use` 重启循环。判断恢复必须以本机
`4040` 实际 HTTP 200 为准，不能只看任一层 PID 或监听端口。

## 动态参数

- 2026-08-31 检查时 Pod 分配了 6 张 NVIDIA A800-SXM4-80GB；每次使用前必须重新查询。
- Kubernetes node、Pod IP、GPU 占用和运行进程都是动态状态，不写成固定连接参数。
- 2026-08-31 已从 `abbtask` 通过 Pod-to-Pod tar 流复制 `.venv`、uv/uvx、uv Python 和
  CUDA compat；源环境保留未删除。
- 2026-08-31 已更新至 MJLab 1.5.3、MuJoCo-Warp 3.10.0.3、MuJoCo 3.10.0、
  Warp-Lang 1.14.0、RSL-RL 5.4.0、Viser 1.0.30、ONNX 1.22.0 与 ONNX Runtime 1.27.0；
  当前 `uv.lock` 精确同步后为 136 个包，依赖检查通过。
- 2026-08-31 已通过 Git bundle 安装上述主仓库与 submodule commit，工作树干净；未包含
  本地未跟踪文件或 submodule 脏改动。
- 2026-09-01 已修正 laptop gateway 的持久化恢复：当 `proxy.exe` bootstrap PID 已退出、
  但 loopback worker 仍健康时，先用 W&B HTTPS CONNECT probe 验证再复用；入口 SSH 或
  boring 临时不可达时留在 Scheduled Task 内循环重试，不再让任务直接退出。修改前备份为
  `C:\Users\Lenovo\.local\bin\nulltask1-wandb-gateway.ps1.bak-20260901-1225`。
- 系统驱动/toolkit 为 CUDA 12.2。显式把 CUDA 12.6 compat 放在 `LD_LIBRARY_PATH` 首位后，
  Torch CUDA kernel 与 Warp 1.14.0 初始化均通过，Warp 识别 Toolkit 12.9 / Driver 12.6
  及 6 张 A800。
- Flat-GRU 单环境 CPU smoke 已完成 5/5 轮并导出 checkpoint/ONNX，run 为
  `logs/rsl_rl/SE3-WheelLegged-Flat-GRU/2026-08-31_21-08-40`。单环境 smoke 的 loss 为
  `nan`，只证明启动与运行链路不崩溃，不作为训练质量验收。
- 2026-09-26：入口机 LAN IP 由 `.47` 变回 `192.168.2.46`（主机密钥与主机名不变），laptop-wg 自身为 `.78`。
- 2026-09-07：入口 `.47:2222` 连续数分钟 Connection refused，全网段扫描只有 `.116` 开 2222 但 banner 超时且 MAC 不同（不是入口机）；几分钟后 `.47` 自行恢复，IP 未变。遇到 refused 先重试 5-10 分钟再怀疑换 IP。W&B gateway 计划任务当时已退到 Ready，End+Run 后 `gateway is ready`。
- 2026-09-05：入口机 LAN IP 由 `192.168.2.46` 变为 `.47`（见上文），Pod IP `172.16.7.177` 未变；Pod 上 sidecar 已停，
  sim2x 改本机运行；仓库同步到 `26cd229`。
- 2026-09-05 23:05（本机 11:05 EDT，本机时区是美东，Pod 是 UTC+8）：本地提交 `bd5866e`（默认站姿静平衡重标定 + 高度默认 v2）、`d214cee`（被动释放录像脚本）、
  `99b37a5`（Flat 高度指令 0.20–0.38 + ONNX 部署包络；子模块 `d65b9ff`）。2026-09-06 00:40（本机 12:40 EDT）D7 结束后
  已把 Pod 同步到 `99b37a5` + 子模块 `d65b9ff`（两个 bundle 走 stdin；gitlink 变化会让 `git fetch` 按需递归拉子模块远端而失败，
  必须加 `--recurse-submodules=no`）。D8 启动脚本在本机 `.scratch/launch_d8.ps1`（RunName
  `flat-D8-jointaction-wheelprice1-nocmderr-balancedpose-h038-steps24-6x8192-5k`），本机没有系统 pwsh 7，能用的是 `C:/Users/13567/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe`；
  Windows PowerShell 5.1 跑不了启动器（UTF-8 无 BOM 按 ANSI 解析 + `StandardInputEncoding` 仅 .NET Core 有）。D8 已于 00:42 启动。D7 跑完之前不要同步：导出器把 `git rev-parse HEAD` 与 dirty 标志写进 ONNX metadata
  溯源。同步时主仓库与子模块各出一个 bundle（子模块在 Pod 上是独立 git 目录）。
- checkpoint 输入路径仍未登记；启动正式训练前必须现场确认，不得从其他 machine
  profile 借用。
