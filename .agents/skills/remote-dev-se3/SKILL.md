---
name: remote-dev-se3
description: se3_wheel_leg 远程训练机运维。管理 SSH 隧道、代理穿透、训练启动/停止/监控、checkpoint 拉取、sim2sim 验证。当用户提到远程训练、训练机、GPU 机器、wuyinyun、阿里云、腾讯云、SSH 连接、代理隧道、训练日志、checkpoint 下载、wandb 看不到数据时触发。
when_to_use: 远程训练 / 训练机 / SSH 隧道 / 代理穿透 / 训练启动 / 训练日志 / checkpoint / wandb / wuyinyun / 无影云 / 阿里云 / 腾讯云 / GPU 机器
user-invocable: true
---

# se3_wheel_leg 远程训练机运维

每台训练机的具体参数（IP、用户名、SSH 别名、GPU 型号）在 `machines/` 子目录对应文件中。
本文档是通用范式入口，所有机器共用同一套操作流程。

## 当前已注册机器

| SSH 别名 | 云厂商 | GPU | 备忘录 |
|---|---|---|---|
| `wuyinyun` | 无影云 | RTX 5880 Ada 48GB | `machines/wuyinyun.md` |

> 新增机器时：在 `machines/` 下新建对应 `.md` 文件，并在上表添加一行。

---

## 网络代理架构

远程训练机处于内网，无法直接访问 PyPI / GitHub / wandb。
解决方案：**SSH 反向隧道把本机 tinyproxy 代理暴露给远程机器。**

```
本机 tinyproxy (127.0.0.1:18080)
        │
        └── SSH -R 17890:127.0.0.1:18080 → 远程机器 (127.0.0.1:17890)
```

> 端口约定：本机 tinyproxy 监听 **18080**，远程隧道落点 **17890**。
> 如机器有本地代理（如 Mihomo），可能有其他端口（见各机器备忘录）。

### 一次性初始化（本机）

```bash
brew install tinyproxy

cat > /tmp/tinyproxy.conf << 'EOF'
Port 18080
Listen 127.0.0.1
Timeout 600
Allow 127.0.0.1
LogLevel Critical
EOF
```

### 每次会话前：建立隧道

```bash
# 替换 <alias> 为机器 SSH 别名（如 wuyinyun）

# 1. 启动本机代理
pkill -f tinyproxy 2>/dev/null
tinyproxy -c /tmp/tinyproxy.conf
sleep 1

# 2. 验证本机代理正常
curl -s -o /dev/null -w "%{http_code}" --proxy http://127.0.0.1:18080 https://pypi.org/simple/
# 期望: 200

# 3. 建立反向隧道
pkill -f "ssh.*17890.*18080" 2>/dev/null
ssh -f -N -R 17890:127.0.0.1:18080 <alias>

# 4. 验证远程可用
ssh <alias> "curl -s -o /dev/null -w '%{http_code}' --proxy http://127.0.0.1:17890 --max-time 5 https://pypi.org/simple/"
# 期望: 200
```

### 一键重建隧道

```bash
# 替换 <alias>
pkill -f tinyproxy 2>/dev/null; tinyproxy -c /tmp/tinyproxy.conf; sleep 1; pkill -f "ssh.*17890" 2>/dev/null; ssh -f -N -R 17890:127.0.0.1:18080 <alias> && echo "tunnel ok"
```

---

## 项目部署

### 首次部署

```bash
ssh <alias> "
  mkdir -p ~/project &&
  cd ~/project &&
  git clone git@github.com:3SE-Competitive-Robotics-Team/se3_wheel_leg.git &&
  echo 'WANDB_API_KEY=<your_key>' > ~/project/se3_wheel_leg/.env
"
```

### 安装 / 更新依赖

```bash
# 首次安装（torch cu128 + cudnn 约 3GB，需代理，5-10 分钟）
ssh <alias> "
  source ~/.local/bin/env &&
  cd ~/project/se3_wheel_leg &&
  HTTPS_PROXY=http://127.0.0.1:17890 HTTP_PROXY=http://127.0.0.1:17890 uv sync 2>&1 | tail -5
"

# 更新代码 + 依赖
ssh <alias> "source ~/.local/bin/env && cd ~/project/se3_wheel_leg && git pull && uv sync 2>&1 | tail -3"
```

### 安装 zellij（首次）

```bash
ssh <alias> "HTTPS_PROXY=http://127.0.0.1:17890 HTTP_PROXY=http://127.0.0.1:17890 \
  bash <(curl -L https://github.com/zellij-org/zellij/releases/latest/download/zellij-x86_64-unknown-linux-musl.tar.gz \
  | tar xz -C ~/.local/bin/ zellij)"
```

---

## 训练管理

> **重要**：zellij 不会 source `.bashrc`，必须在命令中显式设置 `PATH`。
> wandb 上传需要代理，必须带 `HTTP_PROXY`/`HTTPS_PROXY`。

### 启动训练

```bash
# 两步：先建后台会话，再在其中执行训练命令
ssh <alias> "zellij attach --create-background train"
ssh <alias> "zellij --session train action run -- bash -c '
  export PATH=\$HOME/.local/bin:\$PATH
  export HTTP_PROXY=http://127.0.0.1:17890
  export HTTPS_PROXY=http://127.0.0.1:17890
  cd ~/project/se3_wheel_leg &&
  source ~/.local/bin/env &&
  uv run --env-file .env se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1024 2>&1 | tee /tmp/train.log
'"
```

### 查看日志

```bash
ssh <alias> "tail -f /tmp/train.log"          # 实时跟踪
ssh <alias> "tail -50 /tmp/train.log"         # 最新状态
```

### 查看训练进程

```bash
ssh <alias> "ps aux | grep se3-train | grep -v grep"
ssh <alias> "zellij list-sessions"
```

### 停止训练

```bash
ssh <alias> "zellij kill-session train"
# 确保进程真正终止（uv run 会启子进程，kill 会话不够时用）
ssh <alias> "pkill -f 'se3-train'"
```

### 进入会话交互调试

```bash
ssh <alias>
# 进入后：
zellij attach train
# 退出但保持会话：Ctrl+O 然后 D
```

### GPU 状态

```bash
ssh <alias> "nvidia-smi"
ssh <alias> "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader"
```

---

## Checkpoint 管理

### 查看训练 run

```bash
ssh <alias> "ls -t ~/project/se3_wheel_leg/logs/rsl_rl/se3_wheel_leg/"
```

### 查看某 run 的 checkpoints

```bash
TIMESTAMP=<timestamp>
ssh <alias> "ls ~/project/se3_wheel_leg/logs/rsl_rl/se3_wheel_leg/$TIMESTAMP/ | grep model | sort -V | tail -5"
```

### 拉取 checkpoint 到本地

```bash
TIMESTAMP=$(ssh <alias> "ls -t ~/project/se3_wheel_leg/logs/rsl_rl/se3_wheel_leg/ | head -1")
CKPT=model_4400.pt   # 替换为目标 checkpoint

mkdir -p logs/rsl_rl/se3_wheel_leg/$TIMESTAMP
scp <alias>:~/project/se3_wheel_leg/logs/rsl_rl/se3_wheel_leg/$TIMESTAMP/$CKPT \
    logs/rsl_rl/se3_wheel_leg/$TIMESTAMP/
```

> checkpoint 文件名按数字而非字典序排序，`sort -V` 才正确，`sort` 会把 `model_900.pt` 排在 `model_1000.pt` 后面。

---

## 本地 sim2sim 验证

```bash
# 带 rerun 可视化
uv run se3-sim2sim --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_<step>.pt --max-steps 3000

# 无头模式
uv run se3-sim2sim --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_<step>.pt --viewer none --max-steps 200
```

---

## 快速备忘（替换 `<alias>`）

```bash
# 一键检查机器状态
ssh <alias> "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader && zellij list-sessions 2>/dev/null && tail -3 /tmp/train*.log 2>/dev/null | grep -E 'iteration|reward|ETA'"

# 一键重建隧道
pkill -f tinyproxy 2>/dev/null; tinyproxy -c /tmp/tinyproxy.conf; sleep 1; pkill -f "ssh.*17890" 2>/dev/null; ssh -f -N -R 17890:127.0.0.1:18080 <alias> && echo "tunnel ok"

# 一键拉代码重启训练
ssh <alias> "source ~/.local/bin/env && cd ~/project/se3_wheel_leg && git pull && \
  zellij kill-session train 2>/dev/null; \
  zellij attach --create-background train && \
  zellij --session train action run -- bash -c '\
    export PATH=\$HOME/.local/bin:\$PATH && \
    export HTTP_PROXY=http://127.0.0.1:17890 && \
    export HTTPS_PROXY=http://127.0.0.1:17890 && \
    source ~/.local/bin/env && \
    cd ~/project/se3_wheel_leg && \
    uv run --env-file .env se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1024 2>&1 | tee /tmp/train.log\
  '"
```

---

## 常见问题

### uv sync 下载超时

```bash
# 检查隧道是否存活
ssh <alias> "curl -s -o /dev/null -w '%{http_code}' --proxy http://127.0.0.1:17890 --max-time 3 https://pypi.org/simple/"

# 不通则重建隧道（见上方一键命令）
```

### wandb 看不到数据

1. 检查 `.env`：`ssh <alias> "cat ~/project/se3_wheel_leg/.env"`
2. 检查 wandb run 目录：`ssh <alias> "ls ~/project/se3_wheel_leg/wandb/"`
3. 检查 wandb 可达性：`ssh <alias> "curl -s -o /dev/null -w '%{http_code}' --proxy http://127.0.0.1:17890 --max-time 5 https://api.wandb.ai"`
4. **关键陷阱**：wandb 初始化失败时 `writer=None`，RSL-RL 不会保存任何 checkpoint，但训练日志正常打印，极难察觉。无外网时改用 `WANDB_MODE=offline`。

### 多个训练进程并存

```bash
ssh <alias> "ps aux | grep se3-train | grep -v grep"
ssh <alias> "kill <PID1> <PID2>"
# 或
ssh <alias> "pkill -f 'se3-train'"
```

> `kill $PID` 只杀 `uv run` 的 shell wrapper，实际 Python 进程不会终止。多次"重启"后会有多个训练进程争抢 GPU。

### 训练崩溃：`AttributeError: module 'warp' has no attribute 'context'`

warp-lang >= 1.13.0，`pyproject.toml` 已锁定 `<1.13.0`。验证：

```bash
ssh <alias> "source ~/.local/bin/env && cd ~/project/se3_wheel_leg && uv pip show warp-lang | grep Version"
```

修复：`ssh <alias> "source ~/.local/bin/env && cd ~/project/se3_wheel_leg && uv sync"`

### nohup 后台日志为空

`uv run` 内部再启子进程，`nohup` 无法可靠捕获输出。改用 zellij。

---

## 新增训练机流程

1. 在 `machines/` 下新建 `<alias>.md`，参考 `machines/wuyinyun.md` 格式
2. 配置本机 `~/.ssh/config`（HostName、User、IdentityFile）
3. 验证 SSH 连接：`ssh <alias> "whoami && nvidia-smi --query-gpu=name --format=csv,noheader"`
4. 建立代理隧道并验证连通性
5. 首次部署项目（见上方"项目部署"）
6. 在本文件顶部"当前已注册机器"表格中添加一行
