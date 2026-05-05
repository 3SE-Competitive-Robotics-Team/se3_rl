# 训练指南

## 环境准备

### 复制机器人模型文件

训练和仿真需要机器人的 mesh 文件。从 `wheel-legged-gym` 项目复制：

```bash
cp -r ../wheel-legged-gym/resources/robots/serialleg/meshes assets/robots/serialleg/
```

### 安装依赖

```bash
uv sync
```

## 训练命令

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
├── model_2000.pt     # 第 2000 轮（最终模型）
└── params/           # 配置文件
```

## 评估/回放

训练完成后，使用 checkpoint 文件进行评估：

```bash
uv run se3-play SE3-WheelLegged-Flat --checkpoint-file logs/rsl_rl/se3_wheel_leg/<timestamp>/model_2000.pt
```

## Sim2Sim 验证

使用 MuJoCo 进行 sim2sim 验证（纯 CPU，macOS 可运行）：

```bash
uv run se3-sim2sim --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_2000.pt
```

## 常见问题

### ImportError: cannot import name 'XmlMotorActuatorCfg'

mjlab 版本更新，API 已变化。使用 `XmlActuatorCfg` 替代。

### ValueError: Error opening file '../meshes/...'

缺少机器人 mesh 文件，参考「复制机器人模型文件」章节。

### IndexError: list index out of range (GPU)

无可用 NVIDIA GPU，使用 `--gpu-ids None` 切换到 CPU 模式。
