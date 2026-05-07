# SE3 Wheel Leg

轮腿机器人 SerialLeg 的强化学习训练与 sim2sim 验证仓库。训练端使用 MJLab（MuJoCo-Warp GPU 加速），验证端使用标准 MuJoCo CPU，并通过 `se3_shared` 共享机器人参数、观测维度和动作延迟配置。

## 项目结构

```text
src/
├── se3_shared/    # 训练和 sim2sim 共享配置
├── se3_train/     # MJLab 训练环境、奖励、事件和 PPO 配置
├── se3_sim2sim/   # MuJoCo CPU 验证与 Rerun 可视化
└── se3_tools/     # 关节方向和默认姿态诊断工具
```

机器人模型位于 `assets/robots/serialleg/`，训练产物默认写入 `logs/rsl_rl/se3_wheel_leg/`。

## 环境准备

```bash
uv sync
```

训练需要 Linux + NVIDIA GPU + CUDA；macOS 适合运行 sim2sim、诊断工具和 CPU smoke。

训练指标上传到 Weights & Biases 项目：[se3_wheel_leg](https://wandb.ai/3se-competitive-robotics-team/se3_wheel_leg)。

## 常用命令

```bash
# 训练环境 smoke，修改训练代码后优先跑这个
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1 --gpu-ids None

# GPU 训练
uv run --env-file .env se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1024

# sim2sim 验证，默认读取最新 checkpoint
uv run se3-sim2sim --viewer none --max-steps 200

# 指定 checkpoint 并打开 Rerun
uv run se3-sim2sim --checkpoint logs/rsl_rl/se3_wheel_leg/<run>/model_4999.pt --max-steps 3000

# 提交前检查（ruff format + ruff check）
prek run --all-files

# 也可以直接跑底层命令
uv run ruff format .
uv run ruff check . --fix
```

## 动作延迟

训练端和 sim2sim 共享 `se3_shared.ActionDelayConfig`。默认启用 5 ms 动作延迟，并在 reset 时从 4-6 ms 范围内随机采样。sim2sim 可通过命令行覆盖：

```bash
uv run se3-sim2sim --action-delay-ms 5 --action-delay-min-ms 4 --action-delay-max-ms 6
uv run se3-sim2sim --no-action-delay
```

在 `sim_dt=0.002` 时，延迟会量化到物理步数，4-6 ms 对应 2-3 个物理步。

## 文档

- [How to Start](docs/how_to_start.md)
- [训练指南](docs/train.md)
- [GRU 反倒起身训练方案](docs/plan/gru_recovery_training.md)
- [碰撞模型优化记录](docs/todo/collision_model_optimization.md)

## 注意事项

- 所有 Python 命令通过 `uv` 执行。
- `.env`、`logs/`、`wandb/`、Rerun 回放文件和本地 AI 工具配置不应提交。
- 训练 checkpoint 较大，分享仓库时建议单独传递需要验证的 `model_*.pt`。
