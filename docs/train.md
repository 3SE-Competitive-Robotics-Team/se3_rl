# 训练指南

## 环境准备

### 机器人模型文件

训练和仿真用的 MJCF 与 mesh 文件已放在 `assets/robots/serialleg/`。重新导出模型时，保持 MJCF 中的关节名和 mesh 相对路径不变。

### 安装依赖

```bash
uv sync
```

## 训练命令

### Smoke 模式（验证环境）

修改训练代码后先跑一次 smoke，5 轮训练，不上传 wandb，确认环境不崩：

```bash
# CPU smoke（任何机器都能跑）
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1 --gpu-ids None

# GPU smoke
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1024
```

### GPU 训练

需要 NVIDIA GPU + CUDA 12.4+，环境数推荐 1024，RTX 3090/4090 约 2-3 小时跑完。

```bash
# 平地训练
uv run --env-file .env se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1024

# 崎岖地形训练
uv run --env-file .env se3-train SE3-WheelLegged-Rough --env.scene.num-envs 1024
```

### 远程多卡训练

gpufree 等按量计费机器遵循“本地改代码、无卡模式准备、GPU 模式短验证、GPU 模式长训、产物同步后立即关机”的流程，具体见 `.agents/skills/remote-dev-se3/machines/gpufree.md`。

MJLab 多卡训练使用 `--gpu-ids all`。多卡时 `--env.scene.num-envs` 是每张卡的环境数，不是全局环境数。例如 5 张 RTX 4090 上设置 `1024`，全局约为 `5 * 1024 = 5120` 个环境。

```bash
uv run --env-file .env se3-train SE3-WheelLegged-Recovery-GRU --gpu-ids all --env.scene.num-envs 1024
```

如果把单卡 `4096` 直接搬到 5 卡，会变成全局约 `20480` 个环境。除非已经重新缩放课程阶段、总迭代数、保存间隔和评估频率，否则优先用每卡 `1024` 做 20-50 iter benchmark。

### CPU 训练

macOS 或无 GPU 时可跑，速度很慢，只用于调试。环境数设为 1，否则会更慢：

```bash
# 平地训练（CPU 模式）
uv run --env-file .env se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1 --gpu-ids None

# 崎岖地形训练（CPU 模式）
uv run --env-file .env se3-train SE3-WheelLegged-Rough --env.scene.num-envs 1 --gpu-ids None
```

## 训练参数

轮数和保存间隔在对应任务的 `src/se3_train/tasks/<task>/rl_cfg.py` 里配置：

```python
max_iterations=5000,  # 默认 5000 轮
save_interval=100,    # 每 100 轮保存一次 checkpoint
```

## 训练产物

checkpoint 保存在 `logs/rsl_rl/se3_wheel_leg/<timestamp>/`，每 100 轮一个 `.pt` 文件：

```
logs/rsl_rl/se3_wheel_leg/2026-05-05_23-13-57/
├── model_0.pt
├── model_100.pt
├── model_200.pt
├── ...
├── model_4900.pt
├── model_4999.pt     # 5000 轮训练的最终模型
└── params/
```

## 评估/回放

项目不用 `se3-play`，回放和验证走 `se3-sim2sim`，Rerun 负责可视化：

```bash
uv run se3-sim2sim --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_4999.pt --max-steps 3000
```

不传 `--checkpoint` 时，程序自动选 `logs/rsl_rl/se3_wheel_leg/` 下编号最高的 `model_*.pt`，编号相同时选较新的 run。

需要保存 Rerun 回放文件：

```bash
uv run se3-sim2sim --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_4999.pt --max-steps 3000 --rerun-record replays/se3_wheel_leg.rrd
```

`replays/` 和 `.rrd` 文件是本地验证产物，不提交。

## Sim2Sim 验证

纯 MuJoCo CPU，macOS 可跑：

```bash
uv run se3-sim2sim --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_4999.pt --viewer none --max-steps 200 --print-every 20
```

训练只上传 wandb 指标曲线，不录视频。策略回放、轨迹检查、控制量曲线和 `.rrd` 文件都由 `se3-sim2sim` 的 Rerun 产出。

## 常见问题

### ImportError: cannot import name 'XmlMotorActuatorCfg'

mjlab 版本更新，API 已变化，改用 `XmlActuatorCfg`。

### ValueError: Error opening file '../meshes/...'

缺少机器人 mesh 文件，参考「机器人模型文件」章节。

### IndexError: list index out of range (GPU)

没有可用的 NVIDIA GPU，加 `--gpu-ids None` 切到 CPU 模式。
