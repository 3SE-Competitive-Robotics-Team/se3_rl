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

---

## 训练管理

> **强制规范**：
> - **所有训练必须在 tmux session 里启动**，禁止使用 `nohup` 或裸 SSH 后台。
>   tmux 采用 client-server 架构，`new-session -d` 完全不需要 PTY，SSH 断开后 session 和训练进程继续存活。
> - **禁止使用 `WANDB_MODE=offline`**，所有训练必须 wandb 在线可观测。
>   wandb 网络问题时，正确做法是检查并重建 tunnel，而不是切 offline。

### env 数量规则

| 时段 | num_envs | 说明 |
|---|---|---|
| 夜间（22:00-08:00 CST） | **4096** | 无人使用，充分利用 GPU |
| 白天（08:00-22:00 CST） | **1024** | 保留响应余量 |

启动前先检查远程时间：`ssh wuyinyun "date '+%H:%M %Z'"`

### 常用任务名

| 任务 | 说明 |
|---|---|
| `SE3-WheelLegged-Flat-GRU` | GRU 行走基模（优先训练，作为跳跃 pretrain 的起点） |
| `SE3-WheelLegged-Jump-PreTrain-GRU` | 跳跃预训练（依赖 GRU 行走基模） |
| `SE3-WheelLegged-Jump-GRU` | 跳跃精细训练（依赖 PreTrain checkpoint） |
| `SE3-WheelLegged-Flat` | MLP 行走（已有基模 `*_mlp.pt`，一般不需要重新训） |

### 启动训练（标准流程）

```bash
ssh wuyinyun "bash -s" << 'ENDSSH'
WANDB_KEY=$(grep WANDB_API_KEY ~/project/se3_wheel_leg/.env | cut -d= -f2-)

# 清理同名旧 session
tmux kill-session -t train 2>/dev/null

# 创建新 session（detached，不需要 PTY）
tmux new-session -d -s train -x 220 -y 50

# 注入环境变量和训练命令
tmux send-keys -t train "export HTTP_PROXY=http://127.0.0.1:17890" Enter
tmux send-keys -t train "export HTTPS_PROXY=http://127.0.0.1:17890" Enter
tmux send-keys -t train "export WANDB_API_KEY=${WANDB_KEY}" Enter
tmux send-keys -t train "export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:\$LD_LIBRARY_PATH" Enter
tmux send-keys -t train "source ~/.local/bin/env" Enter
tmux send-keys -t train "cd ~/project/se3_wheel_leg" Enter
tmux send-keys -t train "uv run --env-file .env se3-train <TASK_NAME> --env.scene.num-envs <NUM_ENVS>" Enter

tmux list-sessions
ENDSSH
```

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
| `model_*_mlp.pt` | MLP 行走基模 |
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

---

## 新增训练机流程

1. 在 `machines/` 下新建 `<alias>.md`，参考 `machines/wuyinyun.md` 格式
2. 配置本机 `~/.ssh/config`（HostName、User、IdentityFile）
3. 验证 SSH 连接：`ssh <alias> "whoami && nvidia-smi --query-gpu=name --format=csv,noheader"`
4. 建立代理隧道并验证连通性
5. 首次部署项目（见上方"项目部署"）
6. 在本文件顶部"当前已注册机器"表格中添加一行
