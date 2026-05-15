# SE3 Wheel Leg

SerialLeg 轮腿机器人（6 DOF）的强化学习训练与 sim2sim 验证框架。训练端用 MJLab（MuJoCo-Warp GPU 加速）跑 PPO，验证端用标准 MuJoCo CPU，两端通过 `se3_shared` 共享机器人常量、观测维度和动作延迟配置，保证 sim2sim gap 可控。

## 前置条件

- Python 3.11（`>=3.11,<3.12`）
- [uv](https://github.com/astral-sh/uv) 包管理器
- 训练：Linux + NVIDIA GPU + CUDA 12.4+
- sim2sim / 诊断工具：macOS、Linux、Windows（WSL）均可

## 项目结构

```text
src/
├── se3_shared/    # 训练和 sim2sim 共享配置，包含关节语义、PD 增益、动作缩放、延迟参数
├── se3_train/     # MJLab 训练环境，含 MDP（奖励、观测、事件）和 PPO 配置
├── se3_sim2sim/   # MuJoCo CPU 验证，Rerun 可视化
└── se3_tools/     # 关节方向和默认姿态诊断工具
```

机器人模型位于 `assets/robots/serialleg/`，训练产物默认写入 `logs/rsl_rl/se3_wheel_leg/`。

## 环境准备

```bash
uv sync
uv run prek install
```

`uv run prek install` 把提交前检查接入 Git，之后每次 `git commit` 自动运行 ruff format 和 ruff check。

## Quick Start（四步跑通）

本仓库用 [just](https://github.com/casey/just) 统一命令入口，免记长命令。

```bash
just setup     # 装依赖 + 配置 pre-commit hook
just smoke     # CPU smoke 验证环境
just train     # 开始平地训练（需要 GPU + .env）
just sim       # 加载 checkpoint，Rerun 可视化回放
```

运行 `just` 查看所有可用命令。

训练指标上传到 W&B 项目：[se3_wheel_leg](https://wandb.ai/3se-competitive-robotics-team/se3_wheel_leg)。

## 常用命令

### Setup & 检查

```bash
just setup       # uv sync + prek install
just check       # 环境健康检查（Python / GPU / W&B / prek）
```

### Smoke 验证

修改训练代码后先跑这个，5 轮训练验证环境不崩溃，不上传 W&B。

```bash
just smoke       # CPU smoke
just smoke-gpu   # GPU smoke
```

### 训练

需要 `.env` 以上传指标到 W&B。`just train` 会自动检查 `.env` 是否存在。

```bash
just train       # Flat 地形，1024 envs
just train-rough # Rough 地形，1024 envs
just train-cpu   # CPU 调试训练（极慢）
```

### Sim2sim 验证

```bash
just sim                          # 自动选最新 checkpoint，Rerun 可视化
just sim-ckpt logs/.../model_4999.pt  # 指定 checkpoint
just sim-headless                 # 无 GUI，快速验证
```

### 代码质量

```bash
just fmt         # ruff 格式化
just lint        # ruff lint + 修复
just check-code  # 格式化 + lint（提交前执行）
```

### 清理

```bash
just clean       # 清理 logs/ wandb/ replays/
```

### 原始 uv 命令（备选）

如需自定义参数，仍可直接使用 `uv run`：

```bash
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1 --gpu-ids None
uv run --env-file .env se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1024
uv run --env-file .env se3-train SE3-WheelLegged-Rough --env.scene.num-envs 1024
uv run se3-sim2sim --checkpoint logs/rsl_rl/se3_wheel_leg/<run>/model_4999.pt --max-steps 3000
uv run se3-sim2sim --checkpoint logs/rsl_rl/se3_wheel_leg/<run>/model_4999.pt --viewer none --max-steps 200
```

## 动作延迟

训练端和 sim2sim 共享 `se3_shared.ActionDelayConfig`，默认 5 ms，reset 时在 4-6 ms 随机采样。sim2sim 支持命令行覆盖：

```bash
uv run se3-sim2sim --action-delay-ms 5 --action-delay-min-ms 4 --action-delay-max-ms 6
uv run se3-sim2sim --no-action-delay
```

## 文档

- [新手入门](docs/how_to_start.md)
- [训练指南](docs/train.md)
- [训练性能记录](docs/perf.md)
- [wuyinyun 训练机器运维笔记](docs/wuyinyun.md)
- [GRU 反倒起身训练方案](docs/plan/gru_recovery_training.md)
- [MoE 多速度域方案](docs/plan/moe_multi_speed.md)
- [膝关节弹簧建模方案](docs/plan/knee_spring_modeling.md)
- [碰撞模型优化记录](docs/todo/collision_model_optimization.md)

## 注意事项

- 所有 Python 命令通过 `just` 或 `uv` 执行，不直接用 `python` 或 `pip`。
- `.env`、`logs/`、`wandb/`、Rerun 回放文件不应提交。
- 训练 checkpoint 较大，分享仓库时单独传 `model_*.pt`，不要提交到 Git。
- 无外网环境训练用 `WANDB_MODE=offline`，否则 W&B 初始化失败会导致 checkpoint 无法保存。
