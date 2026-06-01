# 训练指南

## 环境准备

### 机器人模型文件

训练和仿真用的 MJCF 与 mesh 文件已放在 `assets/robots/serialleg/`。重新导出模型时，保持 MJCF 中的关节名和 mesh 相对路径不变。

当前默认模型是闭链四连杆无气弹簧常力版本，用来先单独验证闭链机构对平地基模的影响：

```text
assets/robots/serialleg/mjcf/serialleg_closed_chain_v3_train.xml
```

带 300 N 气弹簧常力的同构 MJCF 必须保留为对照模型，后续只有在默认闭链基模稳定后再显式启用：

```text
assets/robots/serialleg/mjcf/serialleg_closed_chain_v3_train_spring.xml
```

训练端默认 `SE3_ROBOT_MJCF_VARIANT=closedchain` 使用无气弹簧常力版本；需要 A/B 对照时用 `SE3_ROBOT_MJCF_VARIANT=closedchain_spring` 或直接通过 `SE3_ROBOT_MJCF` 指向带弹簧文件。

policy 动作顺序固定为 `[LF, LB, RF, RB, l_wheel, r_wheel]`，其中 `LB/RB` 对应 `l_drive_bar_Joint/r_drive_bar_Joint`。闭链限位语义是同侧两根主动杆夹角；当前装配分支下左腿为 `LF-LB`，右腿为 `RB-RF`，允许范围为 `0.0~1.7 rad`，当前默认夹角为 `1.22 rad`，不是后主动杆的绝对角。

当前无气弹簧默认站姿按“base_link 距地约 0.23 m、轮心接近整机质心投影、base/腿部几何离地”的几何平衡点重标定：

```text
default_dof_pos = [-0.2275, -1.4475, 0.2275, 1.4475, 0.0, 0.0]
default_output_knee_pos = [-1.163001511, 1.163000657]
default_coupler_pos = [1.296413806, -1.296416824]
default_base_height = 0.230340071 m
```

这只是两轮倒立系统的 reset 几何基点；零轮速开环 PD 仍不能替代策略的轮子平衡反馈。旧开链模型仍保留为显式回退 variant：

```bash
SE3_ROBOT_MJCF_VARIANT=openchain uv run se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1 --gpu-ids None
uv run se3-sim2sim --model assets/robots/serialleg/mjcf/serialleg_fidelity_cylinder_wheels.xml --viewer none --max-steps 200
```

闭链模型基础检查：

```bash
uv run python scripts/check_closedchain_model.py
uv run se3-joint-viewer --closedchain-spring
```

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

# 开链回退 smoke（仅用于 A/B 定位）
SE3_SMOKE=1 SE3_ROBOT_MJCF_VARIANT=openchain uv run se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1 --gpu-ids None

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

A800 四卡倒地自起训练当前推荐使用每卡 `2048` 个环境。2026-06-01 benchmark 显示 `2048 env/rank` 约 `155.7k steps/s`，比 `1024 env/rank` 的约 `91.8k steps/s` 明显更快，同时等样本量仍保留约 `1500` 次 PPO update。`3072` 和 `4096` 吞吐更高，但等样本量时 update 次数更少，先作为吞吐实验或夜间试训，不直接替代主线长训。详细数据见 `docs/perf.md`。

```bash
# A800 四卡倒地自起推荐长训档位
SE3_LOGGER=tensorboard ./.venv/bin/se3-train \
  SE3-WheelLegged-Recovery-Stand-GRU \
  --gpu-ids all \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 1500
```

如果训练容器使用 CUDA Forward Compatibility，启动训练前确认 Warp 报 `Driver 12.6`，且没有 `CUDA Graphs disabled`。容器默认 `LD_LIBRARY_PATH=/usr/local/nvidia/lib64` 可能覆盖 ldconfig 中的 compat `libcuda`，必要时在训练 shell 中 `unset LD_LIBRARY_PATH`。

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
