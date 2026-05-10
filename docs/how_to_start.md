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

## 2. 克隆仓库

```bash
git clone https://github.com/3SE-Competitive-Robotics-Team/se3_wheel_leg.git
cd se3_wheel_leg
```

## 3. 安装项目依赖

```bash
uv sync
```

成功后确认命令入口已注册：

```bash
uv run se3-train --help
uv run se3-sim2sim --help
uv run prek --version
```

本仓库用 `prek.toml` 管理提交前检查，`prek` 已放在 dev 依赖里，`uv sync` 后通过 `uv run prek` 使用。提交前 hook 会依次运行 `uv run ruff format .` 和 `uv run ruff check . --fix`。

把 `prek` 接入 Git：

```bash
uv run prek install
```

之后每次 `git commit` 都会自动对本次提交包含的文件运行 `prek`。

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

CPU smoke（任何机器都可以跑）：

```bash
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1 --gpu-ids None
```

有 NVIDIA GPU 时可以跑 GPU smoke：

```bash
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1024
```

训练循环正常结束，没有 Python traceback、MuJoCo 报错或 CUDA 报错，就可以进入正式训练。

## 6. 开始第一次完整训练

平地任务：

```bash
uv run --env-file .env se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1024
```

崎岖地形任务：

```bash
uv run --env-file .env se3-train SE3-WheelLegged-Rough --env.scene.num-envs 1024
```

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

无 GUI 快速验证：

```bash
uv run se3-sim2sim \
    --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_4999.pt \
    --viewer none \
    --max-steps 200 \
    --print-every 20
```

打开 Rerun 可视化：

```bash
uv run se3-sim2sim \
    --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_4999.pt \
    --max-steps 3000
```

不传 `--checkpoint` 时，程序会自动选择 `logs/rsl_rl/se3_wheel_leg/` 下最新 run 里编号最大的 checkpoint：

```bash
uv run se3-sim2sim --viewer none --max-steps 200 --print-every 20
```

sim2sim 默认启用 yaw PID 闭环，目标 yaw 为 0 度。需要开环回放时加 `--no-yaw-pid`：

```bash
uv run se3-sim2sim \
    --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_4999.pt \
    --no-yaw-pid \
    --viewer none \
    --max-steps 200
```

## 9. 第一次实验完成标准

同时满足以下条件，这次入门实验就算完成：

- `uv sync` 成功。
- `uv run prek install` 成功，`uv run prek run --all-files` 通过。
- smoke 训练正常结束。
- W&B 项目页能看到正式训练 run。
- `logs/rsl_rl/se3_wheel_leg/<timestamp>/` 下生成 checkpoint。
- `se3-sim2sim` 能加载 checkpoint 并跑到指定 `--max-steps`。
- sim2sim 终端没有 traceback，`done_reason=max_steps` 或正常结束。

完成后，你有了修改训练代码、重新训练、再用 sim2sim 验证的完整闭环。

## 10. 常见问题

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

确认 `.env` 存在，训练命令带了 `--env-file .env`，`WANDB_API_KEY` 正确，账号已加入团队，训练机器能访问 W&B。

### 训练没有 checkpoint

先确认不是 smoke 模式，正式训练必须用 `uv run --env-file .env se3-train ...`。如果 W&B 在线初始化失败，训练可能继续跑但 checkpoint 不会保存，网络不稳定时把 `WANDB_MODE=offline` 写入 `.env` 后重新训练。

### CUDA 或 GPU 报错

确认机器是 Linux + NVIDIA GPU，驱动、CUDA、PyTorch 与项目依赖匹配。只想验证代码入口时，先用 CPU smoke：

```bash
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1 --gpu-ids None
```

### sim2sim 找不到 checkpoint

显式传入 checkpoint 路径：

```bash
uv run se3-sim2sim --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_4999.pt --viewer none --max-steps 200
```

如果 checkpoint 在远程训练机上，先把对应的 `model_*.pt` 和 `params/` 目录同步到本地。
