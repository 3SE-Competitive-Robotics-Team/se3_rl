# gpufree 机器备忘录

> 通用操作流程见父目录 `SKILL.md`，本文件只记录 gpufree 特有的参数和坑。

## 基本信息

| 项目 | 值 |
|---|---|
| 主机名 | `gpufree-container` |
| SSH 别名 | `gpufree` |
| SSH 地址 | 控制台动态生成；每次开机后从实例详情重新复制 SSH 命令 |
| 用户名 | `root` |
| 系统 | Ubuntu 容器环境，Linux 5.15.0-176-generic |
| GPU | NVIDIA GeForce RTX 4090 24GB；控制台当前有 5 卡实例可用于长训 |
| NVIDIA driver | 580.126.09 |
| 仓库路径 | `/root/gpufree-data/se3_wheel_leg` |
| 数据盘 | `/root/gpufree-data`，约 49GB |
| uv 路径 | `/root/.local/bin/uv` |
| 环境入口 | `source /root/gpufree-data/se3_env.sh` |

## 本机 SSH 配置

gpufree 实例开关机后 SSH 端口和初始密码可能变化。本机 `~/.ssh/config` 只保留 `gpufree` 的通用安全选项，不固定 `HostName` 和 `Port`。每次连接前先在控制台实例详情中重新复制 SSH 命令和密码；密码只在交互提示中临时输入，禁止写入本文档、shell 脚本、SSH config 或聊天记录。

`~/.ssh/config`：

```sshconfig
Host gpufree
    User root
    StrictHostKeyChecking no
    UserKnownHostsFile NUL
    PreferredAuthentications password,publickey
    PubkeyAuthentication yes
    IdentitiesOnly no
    # 每次从 gpufree 控制台刷新 HostName / Port，或直接使用控制台给出的 ssh 命令。
    HostName gpufree-console-refresh-required.invalid
    Port 22
```

连接流程：

```bash
# 1. 从 https://www.gpufree.cn/console/instances 复制最新命令，例如：
ssh root@<console-host> -p <console-port>

# 2. 如果需要使用 gpufree alias，先临时更新 ~/.ssh/config 的 HostName / Port。
# 3. 连接后再验证：
ssh root@<console-host> -p <console-port> "hostname; whoami"
```

## 环境状态

已完成：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
cd /root/gpufree-data/se3_wheel_leg
uv sync
```

依赖验证：

```text
torch 2.11.0+cu128
mujoco 3.8.0
mjlab 1.3.0
rerun-sdk 0.31.4
warp-lang 1.12.1
```

## 省钱运行策略

gpufree 按量计费时，默认把 GPU 时间看成最贵资源。代码修改、依赖安装、仓库同步、文档整理和 CPU smoke 都不要占用 GPU 开机时间；只有正式 GPU smoke、吞吐 benchmark、长轮训练和 checkpoint 评估需要开 RTX 4090 实例。

控制台实例备注：

| 实例ID | 配置 | 用途 |
|---|---|---|
| `mhkty1yi-gvnhcs30` | RTX 4090 24GB * 5 | 长轮训练优先使用 |
| `oj45dlbh-xvkf612m` | RTX 4090 24GB * 1 | 单卡调试或备用 |

推荐生命周期：

1. **本地改代码**：在本机完成代码修改、文档修改、ruff、CPU smoke 和提交。远程源码同步必须走 git，不用 GPU 实例在线等待。
2. **无卡模式开机准备环境**：需要整理远程仓库、安装依赖、写 `.env`、配置 SSH key、执行 `uv sync`、创建 `se3_env.sh`、拉取 checkpoint 或做 CPU smoke 时，用控制台的无卡模式开机。无卡模式下不要期待 `nvidia-smi` 可用，训练命令必须带 `--gpu-ids None` 且 `num_envs=1`。
3. **GPU 模式短验证**：准备开训前再把 5 卡实例切到 GPU 模式。先跑 GPU smoke 和 20-50 iter benchmark，确认 `torch.cuda.device_count()==5`、wandb 有日志、checkpoint 能保存、Rerun 监控能产物落盘。
4. **GPU 模式长训**：只有确认短验证方向正确后才启动长轮训练。训练必须在 tmux 中运行，必须同步拉取 checkpoint/Rerun。
5. **收尾立刻关机**：训练停止、checkpoint 和 Rerun 拉回本地后，先确认无 `se3-train` 进程和无未同步产物，再从控制台关机。不要让 GPU 实例空转等下一次改代码。

无卡模式可做：

```bash
ssh gpufree "source /root/gpufree-data/se3_env.sh && cd /root/gpufree-data/se3_wheel_leg && git status --short --branch"
ssh gpufree "source /root/gpufree-data/se3_env.sh && cd /root/gpufree-data/se3_wheel_leg && uv sync"
ssh gpufree "source /root/gpufree-data/se3_env.sh && cd /root/gpufree-data/se3_wheel_leg && SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Recovery-GRU --env.scene.num-envs 1 --gpu-ids None"
```

GPU 模式开训前检查：

```bash
ssh gpufree "source /root/gpufree-data/se3_env.sh && cd /root/gpufree-data/se3_wheel_leg && uv run python -c 'import torch; print(torch.cuda.device_count())'"
ssh gpufree "source /root/gpufree-data/se3_env.sh && cd /root/gpufree-data/se3_wheel_leg && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
```

5 卡训练命令示例：

```bash
ssh gpufree "source /root/gpufree-data/se3_env.sh && cd /root/gpufree-data/se3_wheel_leg && tmux new-session -d -s recovery4096 'source /root/gpufree-data/se3_env.sh && cd /root/gpufree-data/se3_wheel_leg && uv run --env-file .env se3-train SE3-WheelLegged-Recovery-GRU --gpu-ids all --env.scene.num-envs 1024'"
```

注意：MJLab 多卡模式会给每张卡启动一个训练进程，`--env.scene.num-envs 1024` 是**每张卡 1024 个环境**，5 卡全局约 5120 个环境。不要直接把单卡的 `4096` 原样搬到 5 卡，除非已经明确要用全局约 20480 个环境；那样课程阶段、`max_iterations`、`save_interval` 和评估频率都要按总样本量重新缩放。

## cuDNN 环境坑

gpufree 系统路径里有 cuDNN 9.8：

```text
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.8.0
```

当前 PyTorch 编译目标是 cuDNN 9.19。若直接使用默认 `LD_LIBRARY_PATH`，GRU 模型迁移到 CUDA 时会报：

```text
RuntimeError: cuDNN version incompatibility: PyTorch was compiled against (9, 19, 0) but found runtime version (9, 8, 0)
```

正确做法是在训练前加载 gpufree 环境入口，让 `.venv` 中 PyTorch 随包的 NVIDIA libs 排在系统库前面：

```bash
source /root/gpufree-data/se3_env.sh
```

该文件会设置：

```bash
export PATH="$HOME/.local/bin:$PATH"
export UV_LINK_MODE=copy
export LD_LIBRARY_PATH="<repo>/.venv/lib/python3.11/site-packages/nvidia/*/lib:/opt/vulkan/x86_64/lib:/usr/lib/x86_64-linux-gnu"
```

## 常用命令

只读状态检查：

```bash
ssh gpufree "source /root/gpufree-data/se3_env.sh && cd /root/gpufree-data/se3_wheel_leg && git status --short --branch && nvidia-smi"
```

CPU smoke：

```bash
ssh gpufree "source /root/gpufree-data/se3_env.sh && cd /root/gpufree-data/se3_wheel_leg && SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1 --gpu-ids None"
```

GPU smoke：

```bash
ssh gpufree "source /root/gpufree-data/se3_env.sh && cd /root/gpufree-data/se3_wheel_leg && SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 32"
```

启动训练建议使用 tmux：

```bash
ssh gpufree "source /root/gpufree-data/se3_env.sh && cd /root/gpufree-data/se3_wheel_leg && tmux new-session -d -s train 'source /root/gpufree-data/se3_env.sh && cd /root/gpufree-data/se3_wheel_leg && uv run --env-file .env se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1024'"
```

查看训练：

```bash
ssh gpufree "tmux capture-pane -t train -p | tail -40"
ssh gpufree "nvidia-smi"
```

停止训练：

```bash
ssh gpufree "tmux send-keys -t train C-c; sleep 3; pkill -f 'se3-train'; tmux kill-session -t train"
```

## 已验证

- `ssh gpufree` 免密登录可用。
- `uv sync` 完成。
- Torch、Warp、MuJoCo、MJLab import 正常。
- CPU smoke 通过：`SE3-WheelLegged-Flat-GRU`，1 env，5 iter。
- GPU smoke 通过：`SE3-WheelLegged-Flat-GRU`，32 env，5 iter。
