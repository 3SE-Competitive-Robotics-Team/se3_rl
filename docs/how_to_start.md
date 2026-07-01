# How to Start

这份文档带你从一台空机器开始，走完一次完整实验：装工具、配 W&B、跑 smoke、跑训练、找 checkpoint、跑 sim2sim。读完并跑完这些步骤，你就有了本地可复现的训练闭环。

## 0. 确认机器条件

训练需要 Linux + NVIDIA GPU + CUDA，推荐 8GB 以上显存。本地/单卡常规训练示例用 1024 个环境；`codex/xyh` 远程正式训练使用 `a800` 四卡每卡 4096 envs，或 `gpufree` L40S 单卡 8192 envs。

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

## 2. 克隆仓库

```bash
git clone https://github.com/3SE-Competitive-Robotics-Team/se3_wheel_leg.git
cd se3_wheel_leg
```

## 3. 安装项目依赖

安装依赖并配置 pre-commit hook：

```bash
uv sync
uv run prek install
```

成功后确认环境就绪：

```bash
uv run python --version
uv --version
uv run python -c "import mujoco, torch; from importlib.metadata import version; print('mujoco:', mujoco.__version__); print('torch:', torch.__version__); print('rerun-sdk:', version('rerun-sdk'))"
uv run python -c "import torch; print('CUDA 可用:', torch.cuda.is_available()); print('GPU 数量:', torch.cuda.device_count())"
```

这些命令会检查 Python、uv、核心依赖和 GPU/CUDA 状态。`.env` 与 W&B 在正式训练前单独确认。

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

## 4. 跑一次 smoke

smoke 只跑 5 轮，不上传日志，确认环境、MJCF、依赖和训练入口没有崩。改训练代码后先跑一次。

```bash
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-FlowMatch-Wheel-GRU --env.scene.num-envs 1 --gpu-ids None
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-FlowMatch-Wheel-GRU --env.scene.num-envs 1024
```

训练循环正常结束，没有 Python traceback、MuJoCo 报错或 CUDA 报错，就可以进入正式训练。

## 5. 开始第一次完整训练

```bash
uv run se3-train SE3-WheelLegged-FlowMatch-Wheel-GRU --env.scene.num-envs 1024
uv run se3-train SE3-WheelLegged-Rough --env.scene.num-envs 1024
```

运行前确认 `.env` 存在，并已写入 W&B 配置。

第一次建议先跑平地任务。训练会：

- 用对应任务 `src/se3_train/tasks/<task>/rl_cfg.py` 里的 PPO 配置。
- 默认跑 5000 轮。
- 每 100 轮保存一次 checkpoint。
- 把训练指标上传到 W&B。
- 把本地产物写入 `logs/rsl_rl/se3_wheel_leg/<timestamp>/`。

训练过程中打开 W&B 项目页，确认新 run 出现，reward、loss、episode length 等曲线持续更新。

**关于进程管理**：通过 SSH `nohup ... &` 启动训练后，不能直接 `kill` `uv run` 对应的 PID，那只会杀掉 wrapper 进程，实际的 Python 训练子进程还在跑。用下面这条命令杀干净：

```bash
pkill -f "se3-train"
```

## 6. 找到训练产物

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

## 7. 跑第一次 sim2sim

sim2sim 用标准 MuJoCo CPU，可以在训练机器或本地电脑上跑。

```bash
uv run se3-sim2sim --max-steps 3000 --course walk-sweep
uv run se3-sim2sim --viewer none --max-steps 200 --print-every 20 --course walk-sweep
uv run se3-sim2sim --checkpoint logs/.../model_4999.pt --max-steps 3000 --course walk-sweep
uv run se3-sim2sim --checkpoint logs/.../model_4999.pt --viewer none --max-steps 200 --print-every 20 --course walk-sweep
```

sim2sim 默认启用 yaw PID 闭环，目标 yaw 为 0 度。需要开环回放或其它高级参数时，仍可直接用 `uv run`：

```bash
uv run se3-sim2sim \
    --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_4999.pt \
    --no-yaw-pid \
    --viewer none \
    --max-steps 200
```

## 8. 第一次实验完成标准

同时满足以下条件，这次入门实验就算完成：

- `uv sync` 和环境检查命令全部通过。
- `uv run prek run --all-files` 通过。
- CPU smoke 正常结束。
- W&B 项目页能看到正式训练 run。
- `logs/rsl_rl/se3_wheel_leg/<timestamp>/` 下生成 checkpoint。
- 无 GUI sim2sim 能跑完，终端没有 traceback。

完成后，你有了修改训练代码、重新训练、再用 sim2sim 验证的完整闭环。

## 9. 常见问题

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

### 训练没有 checkpoint

先确认不是 smoke 模式（smoke 不下发 checkpoint）。检查 `logs/rsl_rl/` 目录下是否有对应 run 的 checkpoint 文件。

### CUDA 或 GPU 报错

确认机器是 Linux + NVIDIA GPU，驱动、CUDA、PyTorch 与项目依赖匹配。只想验证代码入口时，先用 CPU smoke：

```bash
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-FlowMatch-Wheel-GRU --env.scene.num-envs 1 --gpu-ids None
```

### sim2sim 找不到 checkpoint

显式传入 checkpoint 路径：

```bash
uv run se3-sim2sim --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_4999.pt --max-steps 3000 --course walk-sweep
```

如果 checkpoint 在远程训练机上，先把对应的 `model_*.pt` 和 `params/` 目录同步到本地。
