# MJLab 训练运行时升级计划

## 目标

不切换到 UniLab，不改变 SerialLeg 的 MJLab/MuJoCo-Warp 仿真平台。只吸收 UniLab 暴露出的工程经验，把 SE3 的 MJLab 训练运行时做深：

- 启动时记录 CPU 可见数量、cgroup CPU 配额、CUDA 可见设备。
- 每轮 PPO 拆分记录 rollout collection、return 计算、policy update 和总吞吐。
- 保持 RSL-RL 原有日志口径，避免破坏已有 benchmark 解析。
- 后续在这些指标基础上做 env/rank、num_steps_per_env、save/log 策略自动化。

## 已完成的第一刀

新增 `Se3ProfiledOnPolicyRunner`，通过 MJLab 任务注册已有的 `runner_cls` seam 接入，不 fork MJLab 训练入口，不修改仿真后端。

所有 SE3 MJLab 任务现在都会使用 SE3 runner：

- 普通 flat/rough/recovery-stand 任务：`Se3ProfiledOnPolicyRunner`
- recovery/jump warm-start 任务：`Se3WarmStartRunner`，继承同一套 profile 能力

新增日志字段：

| 字段 | 含义 |
| --- | --- |
| `Runtime/cpu_visible_count` | `os.cpu_count()` 看到的 CPU 数 |
| `Runtime/cpu_affinity_count` | Linux affinity 允许的 CPU 数 |
| `Runtime/cpu_quota_count` | cgroup v2 `cpu.max` 推导的 CPU 配额 |
| `Runtime/cpu_effective_count` | 训练应参考的有效 CPU 数 |
| `Perf/collect_s` | rollout collection 耗时 |
| `Perf/returns_s` | GAE/return 计算耗时 |
| `Perf/update_s` | PPO update 耗时 |
| `Perf/iteration_s` | 单轮总耗时 |
| `Perf/steps_per_second` | 本 rank 的 `num_envs * num_steps_per_env / iteration_s` |

RSL-RL 原生 console 中的 `Collection time` 和 `Learning time` 仍保持旧口径，其中 `Learning time = returns_s + update_s`。

## 为什么先做 profile runner

UniLab 这次的问题不是“仿真平台天然更快”，而是它让 CPU/GPU pipeline 的瓶颈暴露得足够清楚：线程数、collection time、learning time、GPU 利用率一对齐，就能看出错误配置。MJLab 当前训练虽然已经有 `Collection time` 和 `Learning time`，但缺少资源快照和 return/update 拆分，无法判断：

- 是 GPU 仿真没喂满，还是 PPO update 太重。
- 是 cgroup 配额错判，还是 env/rank 不合适。
- 是多卡同步开销，还是单卡 rollout 本身慢。

`Se3ProfiledOnPolicyRunner` 不改变训练数学，只把这些运行时事实记录下来，风险最小。

## 后续切片

1. 资源感知 launcher：启动前根据 GPU 数、显存、CPU 配额生成推荐 `env.scene.num-envs` 和 `agent.max-iterations`。
2. benchmark parser：从训练日志/TensorBoard 中提取 tail-window 均值，自动写入 `docs/perf.md`。
3. profile-driven presets：把 A800/A100/4090 的推荐档位固化成 `just train-profile <profile>`。
4. 深层 pipeline 评估：只有 profile 证明 PPO update 或 CPU 后处理成为瓶颈时，再考虑更激进的采样/学习重叠。

## 验收

第一刀验收只看“行为不变、日志更多”：

- `se3-train` smoke 不崩。
- TensorBoard/W&B 出现 `Runtime/*` 和新增 `Perf/*` 字段。
- 旧的 `Steps per second`、`Collection time`、`Learning time` 仍正常打印。
- 同一任务同一 seed 下，训练曲线不应因 runner profile 产生系统性变化。
