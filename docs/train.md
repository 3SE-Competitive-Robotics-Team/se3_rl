# 训练指南

## 环境准备

### 机器人模型文件

训练和仿真需要的 MJCF 与 mesh 文件已经放在 `assets/robots/serialleg/`。如果后续重新导出模型，保持 MJCF 中的关节名和 mesh 相对路径不变。

### 安装依赖

```bash
uv sync
```

## 训练命令

### Smoke 模式（验证环境）

修改训练代码后，先运行 smoke 模式验证环境不会崩溃：

```bash
# CPU smoke 模式（推荐，快速验证）
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1 --gpu-ids None

# GPU smoke 模式
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1024
```

**特点**：
- 仅训练 5 轮
- 不上传到 wandb（仅本地 tensorboard）
- 用于验证代码修改不会导致环境崩溃

### GPU 训练（推荐）

需要 NVIDIA GPU + CUDA 12.4+，推荐环境数 1024，训练约 2-3 小时（RTX 3090/4090）。

```bash
# 平地训练
uv run --env-file .env se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1024

# 崎岖地形训练
uv run --env-file .env se3-train SE3-WheelLegged-Rough --env.scene.num-envs 1024
```

### CPU 训练

macOS 或无 GPU 时可使用 CPU 模式，但速度极慢，仅用于调试：

```bash
# 平地训练（CPU 模式）
uv run --env-file .env se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1 --gpu-ids None

# 崎岖地形训练（CPU 模式）
uv run --env-file .env se3-train SE3-WheelLegged-Rough --env.scene.num-envs 1 --gpu-ids None
```

**注意**：CPU 模式建议将环境数设为 1，否则会非常慢。

## 训练参数

训练步数在 `src/se3_train/rl_cfg.py` 中配置：

```python
max_iterations=2000,  # 默认 2000 轮
save_interval=100,    # 每 100 轮保存一次 checkpoint
```

## 训练产物

**文件格式**：PyTorch checkpoint（`.pt` 文件）

**保存位置**：`logs/rsl_rl/se3_wheel_leg/<timestamp>/`

**保存规则**：每 100 轮保存一次

```
logs/rsl_rl/se3_wheel_leg/2026-05-05_23-13-57/
├── model_0.pt        # 第 0 轮
├── model_100.pt      # 第 100 轮
├── model_200.pt      # 第 200 轮
├── ...
├── model_1900.pt     # 第 1900 轮
├── model_1999.pt     # 默认 2000 轮训练的最终模型
└── params/           # 配置文件
```

## 评估/回放

项目不使用 `se3-play`。训练完成后，使用自研 `se3-sim2sim` workflow 进行回放和验证，Rerun 负责可视化体验：

```bash
uv run se3-sim2sim --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_1999.pt --max-steps 3000
```

如果不传 `--checkpoint`，程序会自动选择 `logs/rsl_rl/se3_wheel_leg/` 下编号最高的 `model_*.pt`；编号相同时选择较新的 run。

需要保存 Rerun 回放文件时：

```bash
uv run se3-sim2sim --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_1999.pt --max-steps 3000 --rerun-record replays/se3_wheel_leg.rrd
```

`replays/` 和 `.rrd` 文件属于本地验证产物，默认不提交。

## Sim2Sim 验证

使用 MuJoCo 进行 sim2sim 验证（纯 CPU，macOS 可运行）：

```bash
uv run se3-sim2sim --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_1999.pt --viewer none --max-steps 200 --print-every 20
```

训练默认只上传 wandb 指标曲线，不开启视频。策略回放、轨迹检查、控制量曲线和可复查的 `.rrd` 文件统一由 `se3-sim2sim` 的 Rerun workflow 产出。

## 常见问题

### ImportError: cannot import name 'XmlMotorActuatorCfg'

mjlab 版本更新，API 已变化。使用 `XmlActuatorCfg` 替代。

### ValueError: Error opening file '../meshes/...'

缺少机器人 mesh 文件，参考「复制机器人模型文件」章节。

### IndexError: list index out of range (GPU)

无可用 NVIDIA GPU，使用 `--gpu-ids None` 切换到 CPU 模式。
