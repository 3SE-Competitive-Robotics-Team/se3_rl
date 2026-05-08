# How to Start

这份文档面向第一次接触本仓库的新人，目标是从空机器开始，完成一次可记录的训练实验，并用训练出的 checkpoint 跑一次 sim2sim。

## 0. 确认机器条件

训练需要 Linux + NVIDIA GPU + CUDA。推荐使用 8GB 以上显存的 GPU，常规训练环境数用 1024。

macOS 可以运行依赖安装、代码检查、CPU smoke、诊断工具和 sim2sim，但不适合作为正式训练机器。CPU 训练只用于调试，速度很慢。

## 1. 安装基础工具

### 安装 Git

Git 用来克隆仓库、提交代码和推送到 GitHub。

macOS 可以用 Xcode Command Line Tools 或 Homebrew 安装：

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

第一次使用 Git 时，配置提交身份：

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

如果你需要向仓库推送代码，还需要让项目管理员把你的 GitHub 账号加入组织或仓库，并确认你可以访问：

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

安装完成后，重新打开终端，确认 `uv` 可用：

```bash
uv --version
```

## 2. 克隆仓库

```bash
git clone https://github.com/3SE-Competitive-Robotics-Team/se3_wheel_leg.git
cd se3_wheel_leg
```

## 3. 安装项目依赖

本项目固定通过 `uv` 管理 Python 和依赖：

```bash
uv sync
```

成功后可以确认命令入口已注册：

```bash
uv run se3-train --help
uv run se3-sim2sim --help
uv run prek --version
```

本仓库使用 `prek.toml` 管理提交前检查。`prek` 已经放在 dev 依赖里，`uv sync` 后通过 `uv run prek` 使用。

提交前 hook 会运行：

- `uv run ruff format .`
- `uv run ruff check . --fix`

把 `prek` 接入 Git：

```bash
uv run prek install
```

这一步会安装 `prek` 的 Git shim。之后每次 `git commit` 都会自动对本次提交包含的文件运行 `prek`，新人不需要记住每次手动执行检查。

需要撤销 hook 时：

```bash
uv run prek uninstall
```

第一次提交前，可以手动跑一次全量检查：

```bash
uv run prek run --all-files
```

如果 hook 修改了文件，先查看差异，再重新暂存和提交：

```bash
git status
git diff
```

## 4. 申请并配置 W&B

训练默认把指标上传到 Weights & Biases 项目：

https://wandb.ai/3se-competitive-robotics-team/se3_wheel_leg

W&B API key 文档：https://docs.wandb.ai/quickstart

新人需要做三件事：

1. 注册或登录 W&B 账号。
2. 让项目管理员把你的账号加入团队或项目。
3. 在 W&B 个人设置里创建 API key，并只保存到本地 `.env`。

从模板创建 `.env`：

```bash
cp .env.example .env
```

编辑 `.env`：

```bash
WANDB_API_KEY=你的_wandb_api_key
```

`.env` 包含密钥，已经被 `.gitignore` 忽略，不能提交。

如果训练机器无法直接访问 W&B，可以在 `.env` 里加入离线模式：

```bash
WANDB_MODE=offline
```

离线模式会保留本地 wandb 记录，网络恢复后再同步。正式训练前优先确认在线上传可用。

## 5. 跑一次 smoke

smoke 用来确认环境、MJCF、依赖和训练入口没有崩。它只跑 5 轮，不上传 W&B，适合每次改训练代码后先跑。

CPU smoke:

```bash
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1 --gpu-ids None
```

有 NVIDIA GPU 时也可以跑 GPU smoke:

```bash
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1024
```

看到训练循环正常结束，没有 Python traceback、MuJoCo 报错或 CUDA 报错，就可以进入正式训练。

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

- 使用 `src/se3_train/rl_cfg.py` 里的 PPO 配置。
- 默认跑 5000 轮。
- 每 100 轮保存一次 checkpoint。
- 把训练指标上传到 W&B。
- 把本地产物写入 `logs/rsl_rl/se3_wheel_leg/<timestamp>/`。

训练过程中打开 W&B 项目页，确认新 run 出现，并且 reward、loss、episode length 等曲线持续更新。

## 7. 找到训练产物

训练完成后，checkpoint 在：

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

sim2sim 使用标准 MuJoCo CPU，可以在训练机器或本地电脑上跑。

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

如果不传 `--checkpoint`，程序会自动选择 `logs/rsl_rl/se3_wheel_leg/` 下最新 run 里的最高编号 checkpoint：

```bash
uv run se3-sim2sim --viewer none --max-steps 200 --print-every 20
```

sim2sim 默认启用 yaw PID 闭环，目标 yaw 为 0 度。需要开环回放时加：

```bash
uv run se3-sim2sim \
    --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_4999.pt \
    --no-yaw-pid \
    --viewer none \
    --max-steps 200
```

## 9. 第一次实验完成标准

一次新人入门实验完成，需要同时满足：

- `uv sync` 成功。
- `uv run prek install` 成功，并且 `uv run prek run --all-files` 通过。
- smoke 训练成功结束。
- W&B 项目页能看到正式训练 run。
- `logs/rsl_rl/se3_wheel_leg/<timestamp>/` 下生成 checkpoint。
- `se3-sim2sim` 能加载 checkpoint 并跑到指定 `--max-steps`。
- sim2sim 终端没有 traceback，`done_reason=max_steps` 或正常结束。

完成这些步骤后，就具备了修改训练代码、重新训练、再用 sim2sim 验证的基本闭环。

## 10. 常见问题

### 找不到 `uv`

重新打开终端后再试一次 `uv --version`。如果仍不可用，检查安装脚本输出里提示的安装路径是否在 `PATH` 中。

### 找不到 `prek`

先确认已经在仓库根目录执行过：

```bash
uv sync
```

本仓库不要求全局安装 `prek`，使用：

```bash
uv run prek --version
```

### 提交时 ruff 自动修改了文件

这是正常行为。`prek` 会先格式化，再运行 lint 自动修复。查看改动后重新暂存：

```bash
git status
git diff
git add <被修改的文件>
git commit
```

### W&B 没有新 run

确认 `.env` 存在，并且训练命令带了 `--env-file .env`。再确认 `WANDB_API_KEY` 正确、账号已加入团队，并且训练机器能访问 W&B。

### 训练没有 checkpoint

先确认训练不是 smoke 模式。正式训练必须使用 `uv run --env-file .env se3-train ...`。如果 W&B 在线初始化失败，训练可能继续跑但 checkpoint 保存异常；网络不稳定时把 `WANDB_MODE=offline` 写入 `.env` 后重新训练。

### CUDA 或 GPU 报错

确认机器是 Linux + NVIDIA GPU，并且驱动、CUDA、PyTorch 与项目依赖匹配。只想验证代码入口时，先用 CPU smoke：

```bash
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1 --gpu-ids None
```

### sim2sim 找不到 checkpoint

显式传入 checkpoint 路径：

```bash
uv run se3-sim2sim --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_4999.pt --viewer none --max-steps 200
```

如果 checkpoint 在远程训练机上，需要先把对应的 `model_*.pt` 和必要的 `params/` 记录同步到本地。
