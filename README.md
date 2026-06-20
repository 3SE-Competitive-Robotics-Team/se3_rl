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
├── se3_shared/     # 训练和 sim2sim 共享配置，包含关节语义、PD 增益、动作缩放、延迟参数
├── se3_train/      # MJLab 训练环境，含 MDP（奖励、观测、事件）和 PPO 配置
├── se3_sim2sim/    # MuJoCo CPU 验证，Rerun 可视化
├── se3_tools/      # 关节方向和默认姿态诊断工具
├── se3_jump_to/    # 跳跃参考轨迹生成与回放
└── se3_flow_match/ # Flow Matching 蒸馏、数据采集、监控与 play 工具
```

机器人模型位于 `assets/robots/serialleg/`，训练产物默认写入 `logs/rsl_rl/se3_wheel_leg/`。

## 环境准备

```bash
uv sync
uv run prek install
```

`uv run prek install` 把提交前检查接入 Git，之后每次 `git commit` 自动运行 ruff format 和 ruff check。

## Quick Start

本仓库按功能包直接调用对应 CLI，避免把训练、验证、诊断和工具脚本塞进单一任务入口。

```bash
uv sync
uv run prek install
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-FlowMatch-Wheel-GRU --env.scene.num-envs 1 --gpu-ids None
uv run --env-file .env se3-train SE3-WheelLegged-FlowMatch-Wheel-GRU --env.scene.num-envs 1024
uv run se3-sim2sim --max-steps 3000 --course walk-sweep
```

训练指标上传到 W&B 项目：[se3_wheel_leg](https://wandb.ai/3se-competitive-robotics-team/se3_wheel_leg)。

## 常用命令

### Setup & 检查

```bash
uv sync
uv run prek install
uv run python --version
uv run python -c "import mujoco, torch; from importlib.metadata import version; print('mujoco:', mujoco.__version__); print('torch:', torch.__version__); print('rerun-sdk:', version('rerun-sdk'))"
uv run python -c "import torch; print('CUDA 可用:', torch.cuda.is_available()); print('GPU 数量:', torch.cuda.device_count())"
```

### Smoke 验证

修改训练代码后先跑这个，5 轮训练验证环境不崩溃，不上传 W&B。

```bash
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-FlowMatch-Wheel-GRU --env.scene.num-envs 1 --gpu-ids None
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-FlowMatch-Wheel-GRU --env.scene.num-envs 1024
```

### 训练

需要 `.env` 以上传指标到 W&B。正式训练前先确认 `.env` 存在。

```bash
uv run --env-file .env se3-train SE3-WheelLegged-FlowMatch-Wheel-GRU --env.scene.num-envs 1024
uv run --env-file .env se3-train SE3-WheelLegged-Rough --env.scene.num-envs 1024
uv run --env-file .env se3-train SE3-WheelLegged-FlowMatch-Wheel-GRU --env.scene.num-envs 1 --gpu-ids None
```

### Sim2sim 验证

```bash
uv run se3-sim2sim --max-steps 3000 --course walk-sweep
uv run se3-sim2sim --checkpoint logs/.../model_4999.pt --max-steps 3000 --course walk-sweep
uv run se3-sim2sim --viewer none --max-steps 200 --print-every 20 --course walk-sweep
```

### 代码质量

```bash
uv run ruff format .
uv run ruff check . --fix
uv run prek run --all-files
```

### 清理

```bash
rm -rf logs/ wandb/ replays/ MUJOCO_LOG.TXT
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
- [训练任务架构](docs/task_architecture.md)
- [wuyinyun 训练机器运维笔记](docs/wuyinyun.md)
- [GRU 反倒起身训练方案](docs/plan/gru_recovery_training.md)
- [MoE 多速度域方案](docs/plan/moe_multi_speed.md)
- [膝关节弹簧建模方案](docs/plan/knee_spring_modeling.md)
- [碰撞模型优化记录](docs/todo/collision_model_optimization.md)

## 注意事项

- 所有 Python 命令通过 `uv` 执行，不直接用 `python` 或 `pip`。
- `.env`、`logs/`、`wandb/`、Rerun 回放文件不应提交。
- 训练 checkpoint 较大，分享仓库时单独传 `model_*.pt`，不要提交到 Git。
- W&B 初始化或运行期写入失败时，runner 会自动降级到本地 TensorBoard，训练和 checkpoint 保存继续进行；远程长训仍建议先确保代理可用，避免丢在线日志。
