> 固化自 `.tmp/full_perf_pipeline/FINAL_REPORT.md`（2026-08-29 性能画像会话产物；raw 数据在
> `.tmp/full_perf_pipeline/stages/*/report.json`，属临时区不入库）。采纳决策见文末追记。

# Recovery-Loco Grouped History-MLP 完整性能报告

日期：2026-08-29  
任务：`SE3-WheelLegged-Recovery-Loco-Grouped-MLP`  
固定 commit：`190fc4c44818a675440fca4936308c76cb8c52ea`  
固定规模：8192 envs × 24 policy steps  
固定 checkpoint：job 6 `model_4999.pt`  
硬件：NVIDIA GeForce RTX 4090 24 GB

## 结论

主要瓶颈是 `env.step()`，不是 History-MLP actor 或 PPO update。基线画像中，actor 约 22.3 ms、`env.step()` 约 2.072 s、rollout storage 约 16.3 ms、returns 约 2.57 ms、PPO update 约 191.8 ms；`env.step()` 占 Collection 约 96.85%。

最终判断：

- **可直接采用的配置优化**：`nconmax=48, njmax=192`。Collection 对称改善 1.768%，吞吐提升 1.613%，显存减少约 96 MiB，全部 run 无容量 overflow。
- **高价值工程候选**：把纯诊断 GPU→CPU 同步改成 GPU 累积、iteration 末批量传输。临时上限实验改善 Collection 2.612%，两组配对方向一致。
- **高价值版本候选**：MJLab 1.5.3 + MJWarp 3.10.0.3 + MuJoCo 3.10.0 + Warp 1.14.0。Collection 改善 12.681%，纯 MJWarp physics step 改善 13.03%。正式迁移仍需把完整依赖和 API migration 一并验收。
- **有条件的碰撞优化**：base 从 32 个 CoACD hull 改成一个 box。Collection 改善 2.501%，但 active contacts/world 减少 25.681%，当前 checkpoint rollout reward 中位数下降 8.387%；必须用 box 重训并做物理验收后才能采用。
- **拒绝**：中心 3 rays、solver `12/8`、solver `20/20`。
- **拒绝当前 3.12 混装方式**：MJLab 1.4 + MJWarp/MuJoCo 3.12 + Warp 1.16 不兼容；不是性能结论，而是明确的 world 维桥接错误。

## MJWarp 官方五段诊断

同一实际 MJLab 环境导出的模型包含 6 actuators；不是裸 MJCF。基线使用 MJWarp 3.8.1、MuJoCo 3.8.0、Warp 1.12.1，同时启用：

```text
--event_trace --measure_alloc --measure_solver --memory
```

其中 `--measure_alloc` 输出 nacon 与 nefc 两段，因此总共得到以下五段画像。

### 1. Event trace

- physics step：202.70 ns/world-step
- solve：76.98 ns
- collision：17.79 ns
- primitive narrowphase：14.20 ns
- forward：191.18 ns

结论：solve 是最大单项，collision 不是主导；但 solver 上限不能只按平均迭代数直接压低，必须看端到端 ABBA。

### 2. nacon allocation

- mean：14.67 contacts/world
- p95：21
- 原始容量：64/world

这给 `nconmax=48` 留有充足余量。训练 ABBA 中候选没有 nacon overflow。

### 3. nefc allocation

- mean：31.95
- p95：56
- 原始容量：256/world
- `48/192` 候选训练快照峰值：114

候选距 192 仍有明显余量，全部 run 的 nefc overflow 为 0。

### 4. Solver iterations

- mean：2.53
- p95：5
- 配置上限：`iterations=100, ls_iterations=50`

平均值低不代表可把上限设成 12。`12/8` 候选的有效快照全部撞到 12，端到端反而变慢 3.473%；`20/20` 也没有收益。

### 5. Memory

MJWarp 3.8 的报告：

- Model：63.18 MiB
- Data：408.92 MiB
- `efc.J`：128 MiB
- Other：606.96 MiB
- 报告总量：1079.06 MiB

不同 MJWarp 版本对 Other/total 的记账口径发生变化，不直接比较人类可读 total。稳定可比项显示 3.10 的 Model 基本不变、Data 增加 11.27%；训练进程的 `nvidia-smi` 中位数为 2770 → 2804 MiB（+34 MiB）。

## 逐阶段 ABBA

每个阶段都是 A→B→B→A；每个 run 30 iteration，排除前 5 个 warm-up，因此每个变体有 50 个有效 iteration。下表的“Collection 变化”使用两组相邻 AB/BA 的对称几何均值。

| 阶段 | A → B | Collection 变化 | 95% moving-block 区间 | steps/s 变化 | 判断 |
|---|---|---:|---:|---:|---|
| Raycast | 19 rays → 3 center rays | -0.909% | -1.283% ～ +0.203% | +0.880% | 拒绝，方向不稳定 |
| Capacity | 64/256 → 48/192 | -1.768% | -2.542% ～ -1.151% | +1.613% | 采用 |
| Solver | 100/50 → 12/8 | +3.473% | +2.960% ～ +3.778% | -2.803% | 拒绝 |
| Solver 补测 | 100/50 → 20/20 | +0.660% | +0.190% ～ +0.923% | -0.683% | 拒绝 |
| Base collision | 32 CoACD → 1 box | -2.501% | -3.007% ～ -2.173% | +2.402% | 性能通过，物理有条件 |
| 同步点 | 当前诊断同步 → 无诊断同步上限 | -2.612% | -2.953% ～ -1.991% | +2.535% | 工程候选 |
| 版本 | 1.4/3.8 → 1.5.3/3.10 | -12.681% | -13.115% ～ -12.121% | +13.085% | 性能通过，需正式迁移 |

### Raycast

- 两组 Collection 配对：-2.877%、+1.098%，方向相反。
- Collection/update 归一化效应仅 -0.059%。
- 保留完整 19 rays；缩射线不是当前瓶颈。

### Capacity

- 两组配对：-0.810%、-2.717%，方向一致。
- GPU 显存：2922 → 2826 MiB。
- nacon/nefc overflow 全部为 0。
- 采用 `48/192` 作为后续累计基线。

### Solver

- `12/8`：Collection +3.473%，稳定变慢；solver max 撞 12，rollout reward 也略降。
- `20/20`：Collection +0.660%；两组配对为 +2.626%、-1.269%，没有可复现收益。
- 保留 `100/50`。

### Base 单 box

临时 box：0.5 × 0.24 × 0.186 m，中心 z = -0.03 m。

- 两组 Collection 配对：-1.075%、-3.907%。
- GPU 显存：2826 → 2770 MiB。
- active contacts/world：-25.681%。
- 当前 checkpoint rollout reward/step 中位数：0.288519 → 0.264320（-8.387%）。

这是碰撞语义变化，不是等价网格优化。性能结果只能支持“值得单独重训”，不能支持直接替换现有训练模型。

### 同步点

- 两组 Collection 配对：-2.596%、-2.628%，高度一致。
- PPO update 基本不变，显存不变。
- 临时实验通过屏蔽纯诊断 extras、`.item()`/`.tolist()` 和非有限计数测得上限。

正式实现不能简单隐藏 extras。正确结构是 GPU 侧累积计数/和/最大值，只在 iteration 末或低频窗口批量传输一次。

### 版本升级

训练 ABBA：

- Collection：1.854737 → 1.622958 s，对称改善 12.681%。
- 两组配对：-12.328%、-13.032%。
- steps/s：+13.085%。
- PPO update：-1.381%。
- rollout reward/step 中位数：0.273873 → 0.277402（+1.289%），短窗口内未见退化。
- GPU 显存：2770 → 2804 MiB。

纯 MJWarp 对照使用版本匹配、模型元数据完全相同的 MJB：

| 指标 | 3.8 | 3.10.0.3 | 变化 |
|---|---:|---:|---:|
| physics step | 202.70 ns | 176.29 ns | -13.03% |
| solve | 76.98 ns | 53.96 ns | -29.90% |
| collision | 17.79 ns | 17.96 ns | +1.00% |
| solver niter mean | 2.53 | 2.21 | -12.48% |
| steps/s | 4.740 M | 5.470 M | +15.41% |

训练端与官方 testspeed 的改善量一致，版本收益主要来自 solve/implicit 路径，不来自 collision。

实验为保持 checkpoint、actor 与 PPO 对照不变，故意把 `rsl-rl-lib` 固定在 5.2.0；正式采用 MJLab 1.5.3 时，还需要评估其声明的 RSL-RL 5.4.0 依赖以及本仓库自定义 runner 的 API migration。当前结果证明 simulator-stack 的性能收益，不等价于已完成生产依赖升级。

## 3.12 混装失败的根因

最初候选是 MJLab 1.4 + MJWarp/MuJoCo 3.12 + Warp 1.16：

- job 40、41 在 8192 env startup 均触发 PyTorch CUDA `IndexKernel` 越界。
- job 43 加 `CUDA_LAUNCH_BLOCKING=1` 后，首个真实失败点是 `randomize_friction` 写 `geom_friction[env_ids, :, 0]`。
- MJLab 1.4 只识别 stride-0 shared field；新版 MJWarp 把 shared model field 表示成带 `_is_batched` 标记的真实 size-1 world 维，导致 8192 env 索引 size-1 tensor。
- 该问题与 MJLab 官方 [PR #1091](https://github.com/mujocolab/mjlab/pull/1091) 的描述和修复完全一致。

官方版本契约：MJLab 1.4 对应 3.8；1.5 对应 3.10；1.6 对应 3.11。截至本次调查，MJLab main 仍未声明支持 3.12。因此 3.12 混装被拒绝，正式版本 ABBA 改用最小官方支持升级 1.5.3/3.10.0.3。

## 累计收益

以下是各阶段 ABBA 对称效应的乘法估计，不是同一个直接端到端 run 的简单首尾比值：

| 组合 | Collection 估计变化 | steps/s 估计变化 | 适用条件 |
|---|---:|---:|---|
| Capacity | -1.768% | +1.613% | 可直接配置采用 |
| Capacity + Sync | -4.334% | +4.189% | Sync 需正式异步化实现 |
| Capacity + Sync + Version | -16.465% | +17.823% | 不改变碰撞几何；需同步与版本迁移 |
| Capacity + Box + Sync | -6.726% | +6.692% | Box 需重训与物理验收 |
| Capacity + Box + Sync + Version | -18.554% | +20.654% | 全部候选均通过各自验收后 |

不应把最后一行当成当前代码已经获得的收益：本轮没有修改正式源码，base box 也尚未通过任务语义验收。

## 建议落地顺序

1. 正式配置改为 `48/192`，按仓库要求做 smoke。
2. 把诊断指标改成 GPU 累积、iteration 末批量传输，再做同规格 ABBA。
3. 建独立升级分支迁到 MJLab 1.5.3 / MJWarp 3.10.0.3，处理完整依赖并做 smoke、短训与 sim2sim 回归。
4. base box 放在独立实验分支，从头或足够长时间重训；验收倒地恢复、翻滚、侧撞、底盘刮碰、轮腿自碰与 sim2sim。
5. 保留 19 rays 与 solver `100/50`，不投入实现工作。

## 产物与工作区状态

- 所有脚本、覆盖层配置、MJB 元数据和报告只在 `.tmp/full_perf_pipeline/`。
- 正式源码、`pyproject.toml`、`uv.lock` 和 tracked Git 文件均未修改。
- `git status --short --untracked-files=all` 为空；`.tmp` 被 Git 忽略。
- 版本兼容调查见 `version_upgrade_investigation.md`。
- 各阶段完整 raw summary 见 `stages/*/report.json`，人类可读摘要见 `stages/*/report.md`。

## 采纳决策追记（2026-08-29）

用户裁定：① 48/192 进正式配置 + smoke；② 诊断同步异步化 + 同规格复测；
③ 独立分支迁 MJLab 1.5.3/MJWarp 3.10.0.3；④ 射线与 solver 保持不动；
**base box 碰撞几何优化放弃**（+2.4% 吞吐 vs 整轮重训 + 物理验收 + reset cache 作废，收益不抵成本）。
