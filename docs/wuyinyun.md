# wuyinyun 训练机器运维笔记

## 基本信息

| 项目 | 值 |
|---|---|
| IP | 172.31.1.74 |
| 用户名 | zhongjin_lu |
| SSH 别名 | wuyinyun |
| 系统 | Ubuntu 22.04，Linux 5.15.0-125-generic |
| GPU | NVIDIA RTX 5880 Ada 48GB，sm_89 |
| CUDA toolkit | 12.8（`/usr/local/cuda-12.8`） |
| CUDA driver | 570.172.18 |
| 主机名 | pntlgaolhyg69qw |
| 仓库路径 | `~/project/se3_wheel_leg` |
| uv 路径 | `~/.local/bin/uv`（需 `source ~/.local/bin/env` 激活） |
| tmux | 可用，训练用 tmux 会话持久化 |

---

## SSH 连接

### 本机密钥

本机密钥：`~/.ssh/wuyinyun`（ed25519），公钥已写入远程 `~/.ssh/authorized_keys`。

本机 `~/.ssh/config` 配置：

```
Host wuyinyun
    HostName 172.31.1.74
    User zhongjin_lu
    IdentityFile ~/.ssh/wuyinyun
    StrictHostKeyChecking no
```

直接 `ssh wuyinyun` 即可连接。

### 验证连通性

```bash
ping -c 4 172.31.1.74
ssh wuyinyun "whoami && hostname && nvidia-smi --query-gpu=name --format=csv,noheader"
```

---

## 网络代理

远程机器处于公司内网，**直接访问 PyPI / GitHub 间歇性超时**，但访问 SwanLab 正常（国内）。

**解决方案：SSH 反向隧道 + 本机直连代理**

公司 WiFi 可直连外网，使用 `tinyproxy` 在本机建一个 HTTP 代理，通过 SSH 反向隧道暴露给远程机器。

### 一次性初始化（本机）

```bash
# 安装 tinyproxy（只需一次）
brew install tinyproxy

# 写入配置文件（只需一次）
cat > /tmp/tinyproxy.conf << 'EOF'
Port 18080
Listen 127.0.0.1
Timeout 600
Allow 127.0.0.1
LogLevel Critical
EOF
```

### 每次会话前执行

```bash
# 1. 启动本机代理
pkill -f tinyproxy 2>/dev/null
tinyproxy -c /tmp/tinyproxy.conf
sleep 1

# 2. 验证本机代理正常
curl -s -o /dev/null -w "%{http_code}" --proxy http://127.0.0.1:18080 https://pypi.org/simple/
# 期望输出: 200

# 3. 建立 SSH 反向隧道（后台持久运行）
pkill -f "ssh.*17890.*18080" 2>/dev/null
ssh -f -N -R 17890:127.0.0.1:18080 wuyinyun

# 4. 验证远程机器能通过隧道访问外网
ssh wuyinyun "curl -s -o /dev/null -w '%{http_code}' --proxy http://127.0.0.1:17890 --max-time 5 https://pypi.org/simple/"
# 期望输出: 200
```

### 使用代理执行远程命令

```bash
ssh wuyinyun "HTTPS_PROXY=http://127.0.0.1:17890 HTTP_PROXY=http://127.0.0.1:17890 <命令>"
```

> **注意**：Mihomo（Clash Meta，端口 7890）也可以作为代理源，但不如公司 WiFi 直连稳定。tinyproxy 方案走的是公司 WiFi 出口，更可靠。

---

## GitHub SSH（远程机器侧）

远程机器的 GitHub SSH 密钥：`~/.ssh/github_wuyinyun`（ed25519）。
已添加到 GitHub 账号 `XiaoPengYouCode`，名称 `wuyinyun`。

GitHub SSH 22 端口被防火墙封锁，需走 443 端口。远程机器 `~/.ssh/config` 已配置：

```
Host github.com
    HostName ssh.github.com
    Port 443
    User git
    IdentityFile ~/.ssh/github_wuyinyun
    StrictHostKeyChecking no
```

验证：

```bash
ssh wuyinyun "timeout 10 ssh -o ConnectTimeout=8 -T git@github.com 2>&1; echo exit:$?"
# 期望：Hi XiaoPengYouCode! You've successfully authenticated...
```

---

## 项目部署

### 首次部署

```bash
# 确保隧道已建立（见上方"每次会话前执行"）

ssh wuyinyun "
  mkdir -p ~/project &&
  cd ~/project &&
  git clone git@github.com:XiaoPengYouCode/se3_wheel_leg.git &&
  echo 'SWANLAB_API_KEY=<your_key>' > ~/project/se3_wheel_leg/.env
"
```

### 安装依赖

```bash
ssh wuyinyun "
  source ~/.local/bin/env &&
  cd ~/project/se3_wheel_leg &&
  HTTPS_PROXY=http://127.0.0.1:17890 HTTP_PROXY=http://127.0.0.1:17890 uv sync 2>&1 | tail -5
"
```

依赖较大（torch cu128 + cudnn 等共约 3GB），需要代理，首次约 5-10 分钟。后续有缓存会很快。

### 更新代码

```bash
ssh wuyinyun "source ~/.local/bin/env && cd ~/project/se3_wheel_leg && git pull && uv sync 2>&1 | tail -3"
```

---

## 训练管理

### 启动训练（推荐用 tmux 保持会话）

```bash
ssh wuyinyun "
  source ~/.local/bin/env &&
  tmux new-session -d -s train 'cd ~/project/se3_wheel_leg && uv run --env-file .env se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1024 2>&1 | tee /tmp/train.log'
"
```

### 查看训练日志

```bash
# 实时跟踪
ssh wuyinyun "tail -f /tmp/train.log"

# 只看最新状态
ssh wuyinyun "tail -50 /tmp/train.log"
```

### 查看训练进程

```bash
ssh wuyinyun "ps aux | grep se3-train | grep -v grep"
ssh wuyinyun "tmux ls"
```

### 停止训练

```bash
ssh wuyinyun "tmux kill-session -t train"
```

### 进入训练 tmux 会话（交互调试）

```bash
ssh wuyinyun
# 进入后：
tmux attach -t train
# 退出但保持会话：Ctrl+B 然后 D
```

### 查看 GPU 状态

```bash
ssh wuyinyun "nvidia-smi"
ssh wuyinyun "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader"
```

---

## 依赖版本注意事项

### warp-lang 必须锁定 < 1.13.0

warp 1.13.0 移除了 `wp.context` API，mjlab 1.3.0 在 GPU 路径依赖此 API，会导致训练启动崩溃。已在 `pyproject.toml` 锁定：

```toml
"warp-lang>=1.12.0,<1.13.0"
```

macOS CPU 路径不触发此问题，因此本地不会复现。

### PyTorch 版本

`pyproject.toml` 中按平台自动选择：
- macOS → `pytorch-cpu` index（torch CPU 版）
- Linux → `pytorch-cu128` index（CUDA 12.8，支持 torch>=2.7，兼容 mjlab>=1.3.0）

---

## checkpoint 管理

### 查看所有训练 run

```bash
ssh wuyinyun "ls -t ~/project/se3_wheel_leg/logs/rsl_rl/se3_wheel_leg/"
```

### 查看某次 run 的 checkpoints

```bash
ssh wuyinyun "ls ~/project/se3_wheel_leg/logs/rsl_rl/se3_wheel_leg/<timestamp>/"
```

### scp 最新 checkpoint 到本地

```bash
# 先确认最新 run
TIMESTAMP=$(ssh wuyinyun "ls -t ~/project/se3_wheel_leg/logs/rsl_rl/se3_wheel_leg/ | head -1")

# 查看最新 checkpoint
ssh wuyinyun "ls ~/project/se3_wheel_leg/logs/rsl_rl/se3_wheel_leg/$TIMESTAMP/ | grep model | sort -V | tail -5"

# scp 下来（以 model_4400.pt 为例）
mkdir -p logs/rsl_rl/se3_wheel_leg/$TIMESTAMP
scp -i ~/.ssh/wuyinyun zhongjin_lu@172.31.1.74:~/project/se3_wheel_leg/logs/rsl_rl/se3_wheel_leg/$TIMESTAMP/model_4400.pt \
    logs/rsl_rl/se3_wheel_leg/$TIMESTAMP/
```

---

## 本地 sim2sim 验证

```bash
# 带 rerun 可视化
uv run se3-sim2sim --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_<step>.pt --max-steps 3000

# 无头模式（不开 rerun）
uv run se3-sim2sim --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_<step>.pt --viewer none --max-steps 200
```

---

## 常见问题排查

### uv sync 下载超时

**现象**：`operation timed out` / `client error (Connect)` 反复出现。

**原因**：PyPI 大文件（torch wheel、cudnn 等）下载不稳定。

**解决**：确保 tinyproxy + SSH 反向隧道已建立，带代理执行 `uv sync`。

```bash
# 检查隧道是否存活
ssh wuyinyun "curl -s -o /dev/null -w '%{http_code}' --proxy http://127.0.0.1:17890 --max-time 3 https://pypi.org/simple/"

# 隧道不通则重建
pkill -f "ssh.*17890"
tinyproxy -c /tmp/tinyproxy.conf 2>/dev/null || true
ssh -f -N -R 17890:127.0.0.1:18080 wuyinyun
```

### 训练启动崩溃：`AttributeError: module 'warp' has no attribute 'context'`

**原因**：warp-lang 版本 >= 1.13.0，已被 pyproject.toml 锁定到 1.12.x，通常不会再出现。

**验证**：`ssh wuyinyun "source ~/.local/bin/env && cd ~/project/se3_wheel_leg && uv pip show warp-lang | grep Version"`

**修复**：`uv sync`（会自动降级到锁定版本）。

### 训练参数名报错：`Unrecognized options: --runner.max-iterations`

正确参数名为 `--agent.max-iterations`，但默认值已是 5000，**无需显式传入**。

### `nohup` 后台训练日志为空

`uv run` 内部会再启一个进程，`nohup` 有时无法正确捕获子进程输出。**推荐用 tmux**，完全规避此问题。

### 多个训练进程并存

```bash
ssh wuyinyun "ps aux | grep se3-train | grep -v grep"
# 确认后杀掉多余进程
ssh wuyinyun "kill <PID1> <PID2>"
```

### SwanLab 看不到数据

1. 检查 `.env` 文件是否存在且 key 正确：`ssh wuyinyun "cat ~/project/se3_wheel_leg/.env"`
2. 检查 swanlog 目录是否有新 run：`ssh wuyinyun "ls ~/project/se3_wheel_leg/swanlog/"`
3. SwanLab 上传需要访问 `swanlab.cn`，可直连（无需代理）：`ssh wuyinyun "curl -s -o /dev/null -w '%{http_code}' https://swanlab.cn"`
4. 如果 run 目录存在但控制台没数据，等 1-2 分钟后刷新，上传有延迟。

### ContactSensor 初始化失败

如果训练日志出现传感器找不到 body 的报错，检查 MJCF 中的 body 名称是否与 `env_cfg.py` 中 `ContactMatch pattern` 匹配：
- 碰撞传感器：`base_link|lf0_Link|lf1_Link|rf0_Link|rf1_Link`
- 轮子传感器：`l_wheel_Link|r_wheel_Link`

---

## 快速备忘

```bash
# 一键检查机器状态
ssh wuyinyun "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader && tmux ls 2>/dev/null && tail -3 /tmp/train*.log 2>/dev/null | grep -E 'iteration|reward|ETA'"

# 一键重建代理隧道
pkill -f tinyproxy 2>/dev/null; tinyproxy -c /tmp/tinyproxy.conf; sleep 1; pkill -f "ssh.*17890" 2>/dev/null; ssh -f -N -R 17890:127.0.0.1:18080 wuyinyun && echo "tunnel ok"

# 一键拉代码重启训练
ssh wuyinyun "source ~/.local/bin/env && cd ~/project/se3_wheel_leg && git pull && tmux kill-session -t train 2>/dev/null; tmux new-session -d -s train 'cd ~/project/se3_wheel_leg && uv run --env-file .env se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1024 2>&1 | tee /tmp/train.log'"
```
