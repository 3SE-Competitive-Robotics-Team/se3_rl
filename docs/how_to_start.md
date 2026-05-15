# How to Start

这份文档带你从一台空机器开始，走完一次完整实验：装工具、配 W&B、跑 smoke、跑训练、找 checkpoint、跑 sim2sim。读完并跑完这些步骤，你就有了本地可复现的训练闭环。

## 0. 确认机器条件

训练需要 Linux + NVIDIA GPU + CUDA，推荐 8GB 以上显存，常规训练用 1024 个环境。

macOS 可以跑依赖安装、代码检查、CPU smoke、诊断工具和 sim2sim，但不适合正式训练。CPU 训练只用于调试。

## 1. 安装基础工具

### 安装 Git

macOS:

```bash
xcode-select --install
```

或：

```bash
brew install git
```

Ubuntu / Debian:

```bash
sudo apt update
sudo apt install git
```

Windows 推荐安装 Git for Windows：

https://git-scm.com/download/win

确认 Git 可用：

```bash
git --version
```

第一次用 Git，配置提交身份：

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

需要向仓库推送代码，还需要让项目管理员把你的 GitHub 账号加入组织或仓库，仓库地址：

https://github.com/3SE-Competitive-Robotics-Team/se3_wheel_leg

### 安装 uv

官方安装文档：https://docs.astral.sh/uv/getting-started/installation/

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

安装完成后重新打开终端，确认 `uv` 可用：

```bash
uv --version
```

### 安装 just

本仓库用 [just](https://github.com/casey/just) 统一命令入口，免记长命令。你只需要记住几个词：`just setup`、`just smoke`、`just train`、`just sim`。

macOS:

```bash
brew install just
```

Ubuntu / Debian (>= 24.04):

```bash
sudo apt install just
```

其它 Linux / Windows 以及更多安装方式参考：https://github.com/casey/just?tab=readme-ov-file#installation

确认 just 可用：

```bash
just --version
```

## 2. 克隆仓库

```bash
git clone https://github.com/3SE-Competitive-Robotics-Team/se3_wheel_leg.git
cd se3_wheel_leg
```

## 3. 安装项目依赖

一条命令完成依赖安装 + pre-commit hook 配置：

```bash
just setup
```

这等价于： `uv sync && uv run prek install`。

成功后确认环境就绪：

```bash
just check
```

`just check` 会检查 Python、uv、核心依赖、GPU/CUDA、`.env`、prek hook 状态。

之后每次 `git commit` 都会自动运行 ruff format 和 ruff check。

需要撤销 hook 时：

```bash
uv run prek uninstall
```

第一次提交前，可以手动跑一次全量检查：

```bash
uv run prek run --all-files
```

如果 hook 修改了文件，查看差异后重新暂存：

```bash
git status
git diff
```

## 4. 申请并配置 W&B

训练默认把指标上传到 Weights & Biases 项目：

https://wandb.ai/3se-competitive-robotics-team/se3_wheel_leg

W&B API key 文档：https://docs.wandb.ai/quickstart

需要做三件事：

1. 注册或登录 W&B 账号。
2. 让项目管理员把你的账号加入团队或项目。
3. 在 W&B 个人设置里创建 API key，只保存到本地 `.env`，不要提交。

从模板创建 `.env`：

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 key：

```bash
WANDB_API_KEY=你的_wandb_api_key
```

`.env` 已被 `.gitignore` 忽略。

训练机器无法访问 W&B 时，在 `.env` 加入离线模式：

```bash
WANDB_MODE=offline
```

**离线模式很重要**：不设 `WANDB_MODE=offline` 而 W&B 在线初始化失败时，训练会继续跑，但 checkpoint 一个都不会保存，日志里也看不出异常。网络不稳定时先加这行，事后再用 `wandb sync` 上传。正式训练前先确认在线上传可用，或把 `WANDB_MODE=offline` 作为保险。

## 5. 跑一次 smoke

smoke 只跑 5 轮，不上传 W&B，确认环境、MJCF、依赖和训练入口没有崩。改训练代码后先跑一次。

```bash
just smoke       # CPU smoke（任何机器都可以跑）
just smoke-gpu   # GPU smoke（有 NVIDIA GPU 时）
```

训练循环正常结束，没有 Python traceback、MuJoCo 报错或 CUDA 报错，就可以进入正式训练。

## 6. 开始第一次完整训练

```bash
just train       # 平地训练
just train-rough # 崎岖地形训练
```

`just train` 会自动检查 `.env` 是否存在，没有时会提示。

第一次建议先跑平地任务。训练会：

- 用 `src/se3_train/rl_cfg.py` 里的 PPO 配置。
- 默认跑 5000 轮。
- 每 100 轮保存一次 checkpoint。
- 把训练指标上传到 W&B。
- 把本地产物写入 `logs/rsl_rl/se3_wheel_leg/<timestamp>/`。

训练过程中打开 W&B 项目页，确认新 run 出现，reward、loss、episode length 等曲线持续更新。

**关于进程管理**：通过 SSH `nohup ... &` 启动训练后，不能直接 `kill` `uv run` 对应的 PID，那只会杀掉 wrapper 进程，实际的 Python 训练子进程还在跑。用下面这条命令杀干净：

```bash
pkill -f "se3-train"
```

## 7. 找到训练产物

checkpoint 在：

```text
logs/rsl_rl/se3_wheel_leg/<timestamp>/
```

常见文件：

```text
model_0.pt
model_100.pt
model_200.pt
...
model_4900.pt
model_4999.pt
params/
```

按数字排序查看 checkpoint：

```bash
ls logs/rsl_rl/se3_wheel_leg/<timestamp>/model_*.pt | sort -V
```

不要用字典序判断最新模型，`model_900.pt` 会排在 `model_1000.pt` 后面。

## 8. 跑第一次 sim2sim

sim2sim 用标准 MuJoCo CPU，可以在训练机器或本地电脑上跑。

```bash
just sim                          # 自动选最新 checkpoint，Rerun 可视化
just sim-headless                 # 无 GUI，快速验证（自动选 checkpoint）
just sim-ckpt logs/.../model_4999.pt  # 指定 checkpoint，Rerun 可视化
just sim-headless-ckpt logs/.../model_4999.pt  # 指定 checkpoint，无 GUI
```

sim2sim 默认启用 yaw PID 闭环，目标 yaw 为 0 度。需要开环回放或其它高级参数时，仍可直接用 `uv run`：

```bash
uv run se3-sim2sim \
    --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_4999.pt \
    --no-yaw-pid \
    --viewer none \
    --max-steps 200
```

## 9. 第一次实验完成标准

同时满足以下条件，这次入门实验就算完成：

- `just setup` 成功，`just check` 全部通过。
- `uv run prek run --all-files` 通过。
- `just smoke` 正常结束。
- W&B 项目页能看到正式训练 run。
- `logs/rsl_rl/se3_wheel_leg/<timestamp>/` 下生成 checkpoint。
- `just sim-headless` 能跑完，终端没有 traceback。

完成后，你有了修改训练代码、重新训练、再用 sim2sim 验证的完整闭环。

## 10. 常见问题

### just 是什么？

[just](https://github.com/casey/just) 是一个命令运行器，类似于 Makefile 但更简单。运行 `just` 查看所有可用命令，`just <命令>` 执行。所有命令定义在仓库根目录的 `justfile` 中。

### 找不到 `just`

参考 [安装 just](#安装-just) 一节。已安装但终端找不到时，重新打开终端再试。

### 找不到 `uv`

重新打开终端再试 `uv --version`。如果还是不可用，检查安装脚本输出里提示的路径是否在 `PATH` 中。

### 找不到 `prek`

先确认已在仓库根目录执行过 `uv sync`。本仓库不要求全局安装 `prek`，用：

```bash
uv run prek --version
```

### 提交时 ruff 自动修改了文件

这是正常行为。`prek` 会先格式化，再 lint 自动修复。查看改动后重新暂存：

```bash
git status
git diff
git add <被修改的文件>
git commit
```

### W&B 没有新 run

确认 `.env` 存在（`just train` 会检查），`WANDB_API_KEY` 正确，账号已加入团队，训练机器能访问 W&B。

### 训练没有 checkpoint

先确认不是 smoke 模式（smoke 不下发 checkpoint）。正式训练必须用 `just train`（会自动带 `--env-file .env`）。如果 W&B 在线初始化失败，训练可能继续跑但 checkpoint 不会保存，网络不稳定时把 `WANDB_MODE=offline` 写入 `.env` 后重新训练。

### CUDA 或 GPU 报错

确认机器是 Linux + NVIDIA GPU，驱动、CUDA、PyTorch 与项目依赖匹配。只想验证代码入口时，先用 CPU smoke：

```bash
just smoke
```

### sim2sim 找不到 checkpoint

显式传入 checkpoint 路径：

```bash
just sim-ckpt logs/rsl_rl/se3_wheel_leg/<timestamp>/model_4999.pt
```

如果 checkpoint 在远程训练机上，先把对应的 `model_*.pt` 和 `params/` 目录同步到本地。
