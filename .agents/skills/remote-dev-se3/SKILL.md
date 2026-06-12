---
name: remote-dev-se3
description: se3_wheel_leg 远程训练机与 NX 真机部署运维。管理 SSH 隧道、代理穿透、训练启动/停止/监控、checkpoint 拉取、sim2sim 验证、NX 直连和真机部署准备。codex/xyh 分支当前只使用 a800 与 gpufree 两台远程训练服务器；wuyinyun 仅作历史归档。
when_to_use: 远程训练 / 训练机 / SSH 隧道 / 代理穿透 / 训练启动 / 训练日志 / checkpoint / wandb / gpufree / a800 / NX / Jetson / 真机部署 / 阿里云 / 腾讯云 / GPU 机器
user-invocable: true
---

# se3_wheel_leg 远程训练机运维

每台训练机的具体参数（IP、用户名、SSH 别名、GPU 型号）在 `machines/` 子目录对应文件中。
本文档是通用范式入口，所有机器共用同一套操作流程。

## codex/xyh 当前远程服务器

| SSH 别名 | 云厂商 | GPU | 备忘录 |
|---|---|---|---|
| `a800` | 局域网 Kubernetes 容器 | NVIDIA A800 * 4，每卡 4096 envs | `machines/a800.md` |
| `gpufree` | gpufree 容器 | NVIDIA L40S * 1，单卡 8192 envs | `machines/gpufree.md` |

`codex/xyh` 个人工作分支没有 `wuyinyun` 远程训练服务器。`machines/wuyinyun.md` 和 `docs/wuyinyun.md` 只保留历史归档，不作为 SSH、代理、训练、checkpoint 拉取或默认示例目标。

`serialleg-nx` 是 Jetson Orin NX 真机部署目标，不是 MJLab 训练服务器；涉及真机部署时才看 `machines/nx.md`。

执行远程操作前必须先选定 `a800` 或 `gpufree`。如果历史命令片段中仍出现 `wuyinyun`，在 `codex/xyh` 分支下视为归档示例，不能直接运行，也不能把它替换成默认目标；应改用对应机器文档里的命令。

> 新增机器时：在 `machines/` 下新建对应 `.md` 文件，并在上表添加一行。

---

## 网络代理架构

远程训练机处于内网，无法直接访问 PyPI / GitHub / wandb。
解决方案：**SSH 反向隧道把本机已有的 HTTP 代理（端口 7890）直接暴露给远程机器。**

```
本机 HTTP 代理 (127.0.0.1:7890)   ← 系统级科学上网，Python 进程维护
        │
        └── SSH -R 17890:127.0.0.1:7890 → 远程机器 (127.0.0.1:17890)
```

> 端口约定：本机代理监听 **7890**，远程隧道落点 **17890**。

### 验证本机代理可用

```bash
lsof -i :7890 | grep LISTEN
curl -s -o /dev/null -w "%{http_code}" --proxy http://127.0.0.1:7890 --max-time 8 https://api.wandb.ai
```

### 建立 / 重建隧道

```bash
pkill -f "ssh.*17890.*7890" 2>/dev/null; ssh -f -N -R 17890:127.0.0.1:7890 wuyinyun && echo "tunnel ok"

# 验证远程可用（404 = 正常）
ssh wuyinyun "curl -s -o /dev/null -w '%{http_code}' --proxy http://127.0.0.1:17890 --max-time 8 https://api.wandb.ai"
```

---

## 项目部署

### 源码同步规则

**源码变更必须通过 git 协同到远程训练机，禁止用 `rsync` / `scp` / 手工复制直接覆盖源码文件。**

正确流程：
1. 本地完成修改、验证、提交到当前分支。
2. `git push` 推送分支。
3. 远程训练机执行 `git fetch`，再 `git switch` / `git pull --ff-only` 更新到该提交。

只有 checkpoint、日志、回放、导出的分析产物允许用 `rsync` 拉取或上传；源码一律走 git，保证远程训练 run 可追溯到明确 commit。

### Windows PowerShell 调用规范

从 Windows PowerShell 调远端训练机时，默认使用 `scripts/remote_bash.ps1`，不要把复杂 bash 直接塞进 `ssh "..."`
字符串里。Windows PowerShell 5.x 不支持 `cmd1 && cmd2` 作为本地命令连接符，双引号字符串和双引号 here-string 还会提前展开 `$变量` 与 `$(...)`。

推荐模板：

```powershell
$bash = @'
date
cd ~/project/se3_wheel_leg
grep -R "reward" logs | tail -20
echo "remote time: $(date +%H:%M:%S)"
'@

.\scripts\remote_bash.ps1 -HostAlias wuyinyun -ScriptText $bash -UseProxy
```

规则：
- bash 脚本块用单引号 here-string：`@' ... '@`。
- 复杂命令写成本地 `.sh` 后用 `-ScriptPath` 执行，避免多层转义。
- 需要进 Kubernetes pod 时使用 `-KubePod` / `-KubeNamespace` / `-KubeContainer`，pod 内路径不同则加 `-Workdir` 或 `-NoWorkdir`；不要手写 `kubectl exec "... $(...) | ..."` 套娃命令。

### 首次部署

```bash
ssh wuyinyun "
  mkdir -p ~/project &&
  cd ~/project &&
  git clone git@github.com:3SE-Competitive-Robotics-Team/se3_wheel_leg.git &&
  echo 'WANDB_API_KEY=<your_key>' > ~/project/se3_wheel_leg/.env
"
```

### 安装 / 更新依赖

```bash
# 首次安装（需代理，5-10 分钟）
ssh wuyinyun "source ~/.local/bin/env && cd ~/project/se3_wheel_leg && \
  HTTPS_PROXY=http://127.0.0.1:17890 HTTP_PROXY=http://127.0.0.1:17890 uv sync 2>&1 | tail -5"

# 更新代码 + 依赖
ssh wuyinyun "source ~/.local/bin/env && cd ~/project/se3_wheel_leg && git pull && uv sync 2>&1 | tail -3"
```

慢下载排查顺序：
1. 先验证远端代理：`curl --proxy http://127.0.0.1:17890 https://pypi.org/simple/` 应返回 `200`。
2. 有锁文件时用 `uv sync --frozen`，避免慢网环境刷新锁文件。
3. 保留 uv 缓存，必要时显式设置 `UV_CACHE_DIR=$HOME/.cache/uv`；不要为了“干净”加 `--no-cache`。
4. 产物拉取用 `rsync -avz --partial --progress --timeout=60`，源码仍只走 git。

---

## 训练管理

> **强制规范**：
> - **所有训练必须在 tmux session 里启动**，禁止使用 `nohup` 或裸 SSH 后台。
>   tmux 采用 client-server 架构，`new-session -d` 完全不需要 PTY，SSH 断开后 session 和训练进程继续存活。
> - **禁止使用 `WANDB_MODE=offline`**，所有训练必须 wandb 在线可观测。
>   wandb 网络问题时，正确做法是检查并重建 tunnel，而不是切 offline。

### env 数量规则

| 服务器 | GPU | `--env.scene.num-envs` 语义 | 默认值 |
|---|---|---|
| `a800` | A800 * 4 | 每张卡 / 每个 rank 的环境数 | **4096**（全局约 16384 envs） |
| `gpufree` | L40S * 1 | 单卡环境数 | **8192** |

不再按白天 / 夜间自动切换 `1024` / `4096` 档位。需要 smoke 或短 benchmark 时，在命令里显式设置更小的 `--env.scene.num-envs`。

### 常用任务名

| 任务 | 说明 |
|---|---|
| `SE3-WheelLegged-Flat-GRU` | GRU 行走基模（优先训练，作为跳跃 pretrain 的起点） |
| `SE3-WheelLegged-Jump-PreTrain-GRU` | 跳跃预训练（依赖 GRU 行走基模） |
| `SE3-WheelLegged-Jump-GRU` | 跳跃精细训练（依赖 PreTrain checkpoint） |

### 启动训练（标准流程）

```bash
ssh <SSH_ALIAS> "bash -s" << 'ENDSSH'
WANDB_KEY=$(grep WANDB_API_KEY ~/project/se3_wheel_leg/.env | cut -d= -f2-)

# 清理同名旧 session
tmux kill-session -t train 2>/dev/null

# 创建新 session（detached，不需要 PTY）
tmux new-session -d -s train -x 220 -y 50

# 注入环境变量和训练命令
tmux send-keys -t train "export HTTP_PROXY=http://127.0.0.1:17890" Enter
tmux send-keys -t train "export HTTPS_PROXY=http://127.0.0.1:17890" Enter
tmux send-keys -t train "export WANDB_API_KEY=${WANDB_KEY}" Enter
# A800 容器依赖 CUDA Forward Compatibility，必须让 compat libcuda 通过 ldconfig 生效。
# 不要 export /usr/local/cuda-*/lib64 或 /usr/local/nvidia/lib64 到 LD_LIBRARY_PATH，
# 否则会优先加载宿主 12.2 libcuda，Warp 会禁用 CUDA Graphs。
tmux send-keys -t train "unset LD_LIBRARY_PATH" Enter
tmux send-keys -t train "source ~/.local/bin/env" Enter
tmux send-keys -t train "cd ~/project/se3_wheel_leg" Enter
tmux send-keys -t train "uv run --env-file .env se3-train <TASK_NAME> --env.scene.num-envs <NUM_ENVS>" Enter

tmux list-sessions
ENDSSH
```

启动后必须在 tmux 输出中确认 Warp 报 `Driver 12.6`，且没有 `CUDA Graphs disabled`。如果仍看到 `Driver 12.2 < 12.4`，先停止训练并修正容器库路径，不要继续长训。

> `gpufree` 不使用上面的 A800 compat 处理；按 `machines/gpufree.md` 先 `source /root/gpufree-data/se3_env.sh`，让 `.venv` 中的 cuDNN / CUDA 依赖排在系统库前面。

### 查看训练状态

```bash
# 查看 session 最新输出
ssh wuyinyun "tmux capture-pane -t train -p | tail -30"

# 列出所有 session
ssh wuyinyun "tmux list-sessions"

# GPU 状态
ssh wuyinyun "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader"
```

### attach 进入 session 交互查看（人工）

```bash
ssh wuyinyun -t "tmux attach -t train"
# 退出但保持 session：Ctrl+B 然后 D
```

### 停止训练

```bash
# 发送 Ctrl+C 终止训练进程
ssh wuyinyun "tmux send-keys -t train C-c"
sleep 3

# 确保 Python 子进程也终止（uv run 会 fork）
ssh wuyinyun "pkill -f 'se3-train'"

# 关闭 session
ssh wuyinyun "tmux kill-session -t train"
```

> `kill $PID` 只杀 `uv run` 的 shell wrapper，实际 Python 训练子进程不会终止。
> 多次"重启"后会有多个训练进程争抢 GPU，务必用 `pkill -f 'se3-train'`。

---

## Checkpoint 管理

### 查看训练 run

```bash
ssh wuyinyun "ls -t ~/project/se3_wheel_leg/logs/rsl_rl/se3_wheel_leg/"
```

### 查看某 run 的 checkpoints

```bash
TIMESTAMP=<timestamp>
ssh wuyinyun "ls ~/project/se3_wheel_leg/logs/rsl_rl/se3_wheel_leg/$TIMESTAMP/ | grep model | sort -V | tail -5"
```

> checkpoint 文件名按数字排序，`sort -V` 才正确，`sort` 会把 `model_900.pt` 排在 `model_1000.pt` 后面。

### 拉取 checkpoint 到本地

```bash
TIMESTAMP=$(ssh wuyinyun "ls -t ~/project/se3_wheel_leg/logs/rsl_rl/se3_wheel_leg/ | head -1")
CKPT=model_5000.pt

mkdir -p logs/rsl_rl/se3_wheel_leg/$TIMESTAMP
rsync -avz --progress \
  wuyinyun:~/project/se3_wheel_leg/logs/rsl_rl/se3_wheel_leg/$TIMESTAMP/$CKPT \
  logs/rsl_rl/se3_wheel_leg/$TIMESTAMP/
```

### base model 资产管理

训练完成后，将最终 checkpoint 存入 `assets/base_model/` 并用后缀标注架构：

```bash
TIMESTAMP=$(ssh wuyinyun "ls -t ~/project/se3_wheel_leg/logs/rsl_rl/se3_wheel_leg/ | head -1")
rsync -avz wuyinyun:~/project/se3_wheel_leg/logs/rsl_rl/se3_wheel_leg/$TIMESTAMP/model_5000.pt \
  assets/base_model/model_5000_gru.pt
git add assets/base_model/model_5000_gru.pt
git commit -m "chore(assets): 存入 GRU 行走基模 model_5000_gru.pt"
git push
```

| 文件名规范 | 说明 |
|---|---|
| `model_*_gru.pt` | GRU 行走基模（跳跃训练的起点） |

---

## 常见问题

### wandb 网络问题排查

**规则：禁止使用 `WANDB_MODE=offline`，所有训练必须 wandb 在线可观测。**

```bash
# 验证远程能否访问 wandb（404 = 正常，000 = 不通）
ssh wuyinyun "curl -s -o /dev/null -w '%{http_code}' --proxy http://127.0.0.1:17890 --max-time 8 https://api.wandb.ai"

# 不通则重建隧道
pkill -f "ssh.*17890.*7890" 2>/dev/null; ssh -f -N -R 17890:127.0.0.1:7890 wuyinyun && echo "tunnel ok"
```

**关键陷阱**：wandb 初始化失败时 `writer=None`，RSL-RL 不会保存任何 checkpoint，但训练日志正常打印，极难察觉。
`WANDB_API_KEY` 必须在 tmux session 里显式 export，不能只靠 `--env-file .env`。

### 多个训练进程并存

```bash
ssh wuyinyun "ps aux | grep se3-train | grep -v grep"
ssh wuyinyun "pkill -f 'se3-train'"
```

### uv sync 卡住

```bash
ssh wuyinyun "ps aux | grep 'uv sync' | grep -v grep"
ssh wuyinyun "pkill -f 'uv sync'"
```

### A800 / Kubernetes pod 常见坑

- `git fetch` 报 `gnutls_handshake() failed: Error in the pull function` 时，不要继续硬拉 GitHub。按 `machines/a800.md` 用本地 `git bundle create <name>.bundle HEAD`，传到宿主机和 pod 后在 pod 内 `git fetch /tmp/<name>.bundle HEAD`，仍然保持源码通过 git 同步。
- PowerShell 调 `ssh` / `kubectl exec` 时，含 `&&`、`$()`、`$变量` 或管道的脚本统一走 `scripts/remote_bash.ps1`，不要手写嵌套双引号。
- pod 非交互 shell 可能没有 uv 的 PATH；A800 当前 uv 在 `/root/.local/bin/uv`，项目入口在 `.venv/bin/`。自动化脚本用绝对路径，不依赖 `.bashrc`。
- 远端回放或 Rerun 录制不要用会触发同步的裸 `uv run`。优先 `.venv/bin/se3-sim2sim`、`.venv/bin/python`；必须用 uv 时加 `/root/.local/bin/uv run --no-sync`。
- `kubectl cp` / tar 拉可选 marker 时，缺少 `.done` 或 `.failed` 其中一个的 `Cannot stat` 通常无害。以 `.rrd`、summary JSON、checkpoint 是否存在且非空，以及 `.done`/`.failed` 至少一个存在作为成功判据。

---

## 新增训练机流程

1. 在 `machines/` 下新建 `<alias>.md`，参考 `machines/a800.md` 或 `machines/gpufree.md` 格式
2. 配置本机 `~/.ssh/config`（HostName、User、IdentityFile）
3. 验证 SSH 连接：`ssh <alias> "whoami && nvidia-smi --query-gpu=name --format=csv,noheader"`
4. 建立代理隧道并验证连通性
5. 首次部署项目（见上方"项目部署"）
6. 只有当该机器被明确纳入 `codex/xyh` 当前远程服务器时，才在本文件顶部的服务器表格中添加一行
