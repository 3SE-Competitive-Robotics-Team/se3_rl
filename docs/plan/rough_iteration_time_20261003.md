# Rough 单轮耗时拆解与去同步优化（2026-10-03）

## 背景

JumpCol10 六卡 × 2048（W&B `06j74ep2`）每轮 2.33 s，用户要求拆清楚一轮时间花在哪、能否再优化，并在 whtws 上实测。
本轮只做**不改训练行为**的实现优化（用户批准的第 1、2 项）；会改变实验的杠杆（每卡 env 数、地形 geom 数）
和 mjlab 内部同步（第 3 项）不在本轮范围。

## 一轮时间花在哪（改动前）

| 部分 | 耗时 | 依据 |
|---|---|---|
| PPO 更新 + 日志 | ≈0.15 s/轮 | W&B `Perf/update_s` 0.08、cProfile `logger.log` 0.07 |
| 采样 24 步 | 2.24 s/轮（93 ms/步） | W&B `Perf/collect_s` |
| └ `env.step` | ≈70 ms/步（单卡 2048） | `scripts/bench_rough_sim.py --profile-managers --event-trace` |
| 　物理 4 子步 | 18 ms | 913 个地形 geom、4.7 万宽相配对；平地只要 8 ms |
| 　`sim.forward` | 5–11 ms | mjlab 每步一次完整 `mj_forward`（含求解器） |
| 　reset 路径 | 20+ ms | 每步约 6 个 env 结束；cProfile `_reset_idx` 26 ms/步 |
| 　奖励 / 观测 / 终止 / 射线 | 11 / 3.4 / 2.7 / 3.7 ms | 奖励每次调用约 1255 个 kernel |

根因是 **CPU 受限**：单卡 2048 env 每步墙钟约 94 ms，GPU kernel 区间并集只有 32 ms（`torch.profiler`）；
每步约 5800 个 kernel、**267 次 GPU→CPU 同步**（`torch.cuda.set_sync_debug_mode`）。每次同步 CPU 都要等 GPU 把
排队的物理与奖励跑完，两边无法流水。8192 env/卡时样本翻 4 倍，每步只从 70 涨到 117 ms，同样说明是固定开销。
6 进程并发时每进程仍 68 ms/步，不是进程间抢 CPU。

同步的写法来源：

1. Python 元组 / 列表做高级索引（`x[:, (0, 2)]`、`joint_pos[:, leg_ids]`）——每次在 CPU 建索引再拷到 GPU；
2. 每次调用 `torch.tensor(常数, device=cuda)` / `torch.as_tensor(...)`；
3. `x[ids] = 0.0` 这类按索引写 Python 标量——标量先做成 CPU 张量再 pageable 拷贝，CUDA 在这种拷贝前同步整条流；
4. 布尔掩码写入（`x[mask] = y`，内部 `nonzero`）、`.any()` 分支、`.item()` 日志。

## 改动（逐位等价）

- `se3_shared.torch_constants`：`device_constant` / `device_index` 按 (值, 设备, dtype) 缓存常量与索引张量；
  `fourbar` / `height_default` / `leg_policy` 的常量、`joint_indices.tensor_ids` 全部走缓存，`[:, (0, 2)]` 改切片 `[:, 0::2]`。
- 观测、终止、奖励、台阶奖励里的元组索引改缓存索引张量；按列几何、高姿态起步序列、台阶领奖缓存改 `torch.where`。
- reset 路径的标量写入统一改 `index_fill_` / `masked_fill_`（actions、recovery_state、commands、jump_commands、
  rough commands、跳跃时钟、events）；events 里 56 处采样区间 `torch.tensor(float(x))` 改缓存常量。
  `env_ids[mask]` 之后按子集个数采样的地方保留——那里的同步是保持随机数消耗顺序所必需的。
- `terminations.leg_contact` 的跳跃诊断只在 `enable_jump_metrics` 打开时计算（rough 关闭，原来每步 7 次 `.item()` 算完没人读）。
- `columns.column_mask` 按 (`terrain_types` 身份, 版本号) 缓存：每步 15 次调用只在 reset / 课程换列后重算。
- 四连杆换算左右腿合并成一次 `[N, 2]` 计算（`_side_active_angles_torch` 等），`coupler_jacobian` 的 hi / lo 合并一次。

## 验证

- **整环境逐位比对**（`.scratch/perf/equiv_check.py`）：GPU 上 MuJoCo-Warp 物理有原子加的非确定性（同一份代码两次运行
  第 0 步就差 1e-3），所以在 CPU 上跑 rough、32 env × 1100 步（后一半 env 大动作逼出摔倒，覆盖 19 次终止 + 29 次超时重置），
  逐步比对 actor / critic 观测、奖励、终止、关节与根状态、指令、日志。去同步与列掩码缓存全部逐位一致。
  注意 CPU 上记录张量要 `.clone()`：`.cpu()` 对 CPU 张量不复制，记下的全是同一块缓冲区的引用，比对会恒等。
- **四连杆合并只做函数级比对**：PyTorch CPU 的 `cos/sin/atan2` 对连续张量走 SLEEF 向量化、尾部元素走标量路径，
  改变张量形状会让个别元素差 1 ulp（CPU 上 float64 n=13 差 4e-13），所以 CPU 整环境比对不适用；
  GPU 上逐元素数学函数与形状无关，`.scratch/perf/fb_check.py` 在 6 种大小 × 2 种精度下 84 项全部逐位一致。
- 102 个单元测试通过；Rough、Flat-MLP、Jump-Mimic-MLP、Jump-FineTune-GRU CPU smoke 通过。

## 结果（whtws，单卡 2048 env，同 seed 从头训 60 轮，取第 11–60 轮均值）

| 版本 | 单轮时间 | 每步同步 |
|---|---|---|
| 旧代码 `6303547` | 1.736 s | 267 |
| 去同步 | 1.507 s | 82 |
| + 列掩码缓存 | 1.482 s | — |
| + 四连杆左右合并 | **1.433 s（−17.5%）** | — |

## 剩余杠杆（未做）

1. ~~旧跳跃 RSI 残留~~ 已修（见下节，改变训练行为）。
2. **mjlab 内部同步约 40 次/步**：`RewardManager.reset` 逐项 `x[ids] = 0.0`（12.5 次/步）、射线 `_extract_yaw_rotation`
   的 `.any()`、官方 `terrain_levels_vel`、curriculum / termination / event manager。需给上游提 PR 或打补丁。
3. 会改变实验的：每卡 env 数（单样本吞吐好约 2.5 倍，改 PPO batch）、地形 geom 数（物理 18 → 8 ms，改任务分布）。

## 行为修复：跳跃中途被终止的 env 误走旧跳跃线 RSI（用户批准，单独提交）

reset 事件先预采样指令（`_pre_resample_command_for_reset`）再写跳跃维度（`clock.write_dims`），这时跳跃时钟还没 reset，
跳跃中途被终止（模仿偏离 / 摔倒）的 env 被写成 `jump_flag=1`；`reset_root_state_full` / `reset_joints` 随即把它当旧跳跃线
RSI 样本，从旧参考轨迹（`DEFAULT_JUMP_TRAJ_PATHS`）第 0 帧注入机身高度、速度与关节角，而不是正常 reset 分布。
whtws 实测（2048 env × 600 步、随机动作）：旧代码 733 次跳跃中途 reset 全部走了旧 RSI，修复后 0 次。

修法：`RoughJumpCommandTerm` 在 reset 预采样时先 `clock.reset(env_ids)` 再写跳跃维度（根因）；旧 RSI 只对
`enable_jump_lifecycle=True` 的 `JumpCommandTerm` 生效（PreTrain / FineTune 不变）。Jump-Mimic 任务用自己的
`reset_jump_mimic` 事件，不受影响。修复后每步同步 82 → 76.5，单卡 2048 env 训练 1.425 s/轮。

## 工具

- `.scratch/perf/remote.sh '<命令>'`：把本地 `src/se3_train`、`src/se3_shared`、`.scratch/perf` 推到 whtws
  `/workspace/se3-worktrees/perf-base` 并执行（worktree 软链主 checkout 的 `.venv`，必须设 `PYTHONPATH=<worktree>/src`，
  否则导入的是主 checkout 的源码）。
- `.scratch/perf/sync_audit.py`（按代码位置统计每步同步）、`gpu_busy.py`（GPU 忙碌 vs 墙钟）、
  `call_counts.py`（每步函数调用次数与耗时）、`equiv_check.py`、`fb_check.py`；`scripts/profile_env_terms.py`（逐项计时）。
