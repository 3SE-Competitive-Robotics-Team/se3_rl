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

## A100 远端验证

记录时间：2026-06-02。

远端机器：`root@120.209.70.195 -p 30369`，A100-SXM4-80GB * 2。测试时两张 GPU 空闲，单卡测试固定 `CUDA_VISIBLE_DEVICES=0`。

测试 commit：

```text
b4e2852 enh(se3_train): 为 MJLab 训练增加运行时画像
```

测试命令：

```bash
SE3_LOGGER=tensorboard CUDA_VISIBLE_DEVICES=0 \
uv run --no-sync se3-train SE3-WheelLegged-Flat-GRU \
  --env.scene.num-envs 8192 \
  --agent.max-iterations 60 \
  --agent.save-interval 1000
```

训练启动时 runtime summary：

```text
[SE3 Runtime] cpu_visible=112, cpu_affinity=112, cpu_quota=24.00, cpu_effective=24.00, cuda_visible=0
```

结果：

| 指标 | warm 平均 | tail-20 平均 | 最后一轮 |
| --- | ---: | ---: | ---: |
| `Iteration time` | `12.448s` | `12.363s` | `12.330s` |
| `Collection time` | `10.883s` | `10.814s` | `10.798s` |
| `Learning time` | `1.565s` | `1.549s` | `1.531s` |
| `Steps per second` | `42.12k` | `42.41k` | `42.53k` |
| `Perf/returns_s` | - | `0.0068s` | `0.0067s` |
| `Perf/update_s` | - | `1.542s` | `1.525s` |

结论：

- profile runner 在 A100 真实训练中可用，`Runtime/*` 和新增 `Perf/*` 已写入 TensorBoard。
- cgroup 配额被正确识别为 24 核，避免再次把 `os.cpu_count()==112` 误当成可用 CPU。
- `returns_s` 只有毫秒级，`update_s` 约 1.5s；主要瓶颈仍是 MJLab rollout collection，tail-20 约 10.8s。
- 单卡 `8192 envs x 64 steps` 吞吐约 `42.4k steps/s`，与此前同机 MJLab baseline 一致，说明 profile runner 没有引入可见吞吐回退。

双卡 `--gpu-ids all` 首次尝试在 torchrunx rank 同步阶段失败，rank0 看到 rank1 断开，未进入训练循环。该问题与 profile runner 的单卡验证无关，但需要单独排查 torchrunx/rank1 启动日志和容器 OpenGL/EGL 依赖。
