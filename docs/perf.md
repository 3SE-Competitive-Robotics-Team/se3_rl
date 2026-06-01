# 训练性能记录

记录各设备的训练吞吐，用于横向对比 Jetson Orin NX、RTX 4060/4060 Ti、RTX 4090/5090 等。

## 当前基线

记录日期：2026-05-06

设备：

- 机器：Apple M5 MacBook Air
- CPU：10 核
- 内存：16GB
- 训练方式：CPU 训练
- 操作系统：macOS

训练命令：

```bash
uv run --env-file .env se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 512 --gpu-ids None
```

训练配置：

- 环境数：`512`
- 每个环境每轮采样步数：`32`
- 总训练轮次：`5000`
- 每轮采样量：`512 * 32 = 16384` steps
- 总目标采样量：`512 * 32 * 5000 = 81920000` steps

## 已观测吞吐

来自训练日志：

```text
Learning iteration 23/5000
Total steps: 393216
Steps per second: 607
Collection time: 26.419s
Learning time: 0.560s
Iteration time: 26.98s
```

吞吐计算：

```text
512 * 32 / 26.98 ~= 607 steps/s
```

`Total steps` 与第 23 轮日志一致（iteration 从 0 开始计数，`Learning iteration 23` 对应已完成 24 轮采样）：

```text
512 * 32 * 24 = 393216
```

## 完整训练耗时估算

按 `607 steps/s` 估算：

```text
81920000 / 607 = 134958s ~= 37.5h
```

当前本地 CPU 基线：

```text
512 env, CPU, Apple M5：约 607 steps/s
```

## 后续对比口径

更换设备后，记录训练日志中的这一行：

```text
Steps per second: <value>
```

同时记录：

- 设备型号
- 是否使用 GPU
- 环境数
- `num_steps_per_env`
- `Collection time`
- `Learning time`
- `Iteration time`
- 显存或内存占用

只有环境数、训练配置和任务一致时，`Steps per second` 才适合直接横向比较。

## A800 倒地自起多卡 benchmark

记录日期：2026-06-01

设备：

- 机器：局域网 Kubernetes 训练容器
- GPU：NVIDIA A800-SXM4-80GB * 4
- 代码：`377beb1`
- 任务：`SE3-WheelLegged-Recovery-Stand-GRU`
- 多卡方式：`--gpu-ids all`
- 说明：`--env.scene.num-envs` 是每张卡的环境数，不是全局环境数。

CUDA Forward Compatibility 必须生效。该容器默认 `LD_LIBRARY_PATH=/usr/local/nvidia/lib64` 会优先加载宿主 12.2 `libcuda`，导致 Warp 报 `Driver 12.2` 并禁用 CUDA Graphs。benchmark 前使用 ldconfig 注册 `/workspace/cudacompat/usr/local/cuda-12.6/compat`，并在训练 shell 中 `unset LD_LIBRARY_PATH`，确认 Warp 报：

```text
CUDA Toolkit 12.9, Driver 12.6
```

测试命令形态：

```bash
SE3_LOGGER=tensorboard ./.venv/bin/se3-train \
  SE3-WheelLegged-Recovery-Stand-GRU \
  --gpu-ids all \
  --env.scene.num-envs <env_per_rank> \
  --agent.max-iterations <iters>
```

结果取训练后段稳定窗口均值：

| env/rank | 全局 env | iter | 吞吐 | Iteration time | Collection time | Learning time | GPU 平均利用率 | 单卡显存峰值 | 等同 `1024*3000` 样本量的估算耗时 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 4096 | 200 | `91.8k steps/s` | `2.86s` | `2.41s` | `0.44s` | `69.8%` | `6.3GB` | `2.38h` |
| 2048 | 8192 | 200 | `155.7k steps/s` | `3.37s` | `2.79s` | `0.58s` | `72.9%` | `9.1GB` | `1.40h` |
| 3072 | 12288 | 100 | `186.7k steps/s` | `4.21s` | `3.44s` | `0.78s` | `77.7%` | `13.2GB` | `1.17h` |
| 4096 | 16384 | 60 | `215.5k steps/s` | `4.87s` | `3.94s` | `0.93s` | `76.6%` | `17.9GB` | `1.01h` |

结论：

- A800 上 GPU 没有被完全打满。即使 `4096 env/rank`，平均利用率仍约 `77%`，显存也远低于 80GB。
- 主要时间在 rollout collection，不在 PPO learning。最后一轮典型比例是 collection 数秒、learning 不到 1 秒。
- `2048 env/rank` 是当前推荐长训档位：吞吐从 `1024` 的约 `92k` 提升到约 `156k steps/s`，但等样本量时仍有约 `1500` 次 PPO update。
- `3072` 和 `4096` 适合作为吞吐实验或夜间试训档位。等样本量时 PPO update 只有约 `1000` / `750` 次，可能改变策略质量，不能直接替代 `1024 env/rank * 3000 iter` 的训练结论。

推荐用法：

```bash
# 同等样本量的快速长训，约 1.4 小时
SE3_LOGGER=tensorboard ./.venv/bin/se3-train \
  SE3-WheelLegged-Recovery-Stand-GRU \
  --gpu-ids all \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 1500

# 更保守，样本更多，约 1.9 小时
SE3_LOGGER=tensorboard ./.venv/bin/se3-train \
  SE3-WheelLegged-Recovery-Stand-GRU \
  --gpu-ids all \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 2000
```

不要只按“相同 iter 数”比较不同环境数。多卡下每轮样本量为：

```text
num_gpus * env_per_rank * num_steps_per_env
```

本任务当前 `num_steps_per_env=64`，因此 `2048 env/rank * 1500 iter` 与 `1024 env/rank * 3000 iter` 的总样本量接近。
