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
uv run --env-file .env se3-train SE3-WheelLegged-Flat --env.scene.num-envs 512 --gpu-ids None
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
