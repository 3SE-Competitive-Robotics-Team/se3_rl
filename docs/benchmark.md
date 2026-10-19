# Rough benchmark（se3-bench）

训练后自动评测：在原生 MuJoCo + `se3_runtime` 部署链路上回放训练导出的 ONNX，按固定场景出指标，
不用每次把模型拉回本地开 sim2x。只读 ONNX 契约，MLP / GRU / History-MLP（CTS 学生）都能跑。
代码在 `src/se3_tools/bench/`，当前版本 `rough-v1`（`spec.BENCH_VERSION`）。

## 五项评测

| 项 | 场景 | 头条指标（W&B `bench/<suite>/...`） |
|---|---|---|
| A 平地跟踪 | 平地 vx × 高度网格、yaw 网格（含包络顶端 ±12 rad/s），阶跃后取最后 5 s 稳态 | `A/vx_err_mean`、`A/yaw_top_ratio`、`A/height_abs_err_mean`、`A/pitch_rate_band_rms_max`（5–15 Hz 俯仰角速度）、`A/falls` |
| B 静站与起步 | 零指令静站 30 s，再阶跃前进；对照只站 1 s 的冷起步 | `B/drift_speed_max`、`B/drift_m_max`、`B/yaw_drift_deg_max`、`B/t90_after_stand_mean_s`、`B/t90_degrade_mean_s` |
| C 台阶 | 训练地形台阶四列（上行 / 二级上行 / 下行 / 二级下行）逐级，助跑开局与静止开局；上行列记立面事件 | `C/stairs_up_max_level`、`C/stairs_up_r<N>_clear_rate`、`C/static_start_gap`、`C/target_style_rate`、`C/jump_onto_rate`、`C/down_fall_rate` |
| D 跳跃 | 平地站 2 s 后按参考目标离地间隙起跳（0.2–0.5 m） | `D/success_rate`、`D/clearance_err_mean_m` |
| E 鲁棒 | WideDR 那组 11 种扰动 plant × 平地跟踪 / 静站 / 侧推 / 第 6 级台阶 | `E/falls`、`E/flat_vx_err_mean`、`E/stairs_r6_clear_rate` |

判据要点：
- 台阶走出该块 = 机身到出生点切比雪夫距离 ≥ 4.0 m（与 Rough 训练升级判据一致），一条 20 s。
- 台阶场景每次按当前 checkout 的 `rough_terrains_cfg()` 现生成（`scenes.build_stair_grid`），与训练地形逐块一致。
- 助跑开局先用 0.24 m 低姿态起步 0.6 s 再切目标高度，把「爬升能力」与「高姿态静止起步死锁」分开测；
  `C/static_start_gap` = 助跑通过率 − 静止通过率。
- 立面事件：触面前 0.5 s 双轮同时离地 > 2 cm 记「跳上去」；机身前沿过沿早于双轮抬起 6 cm 记「机身先过沿」；
  左右轮抬起时间差 ≤ 0.1 s 且前两者满足记「目标上台阶姿态」（`C/target_style_rate`）。
- 摔倒 = 机身倾角 > 60°。默认确定性回放；full 档台阶另加 σ = 0.21 的训练式动作噪声各 3 次（`C/noise_up_clear_rate`）。

## 两档

- **quick**（只用于手动快速检查）：A 粗网格、B、C 第 3/6/9 级、D 两档目标。约 50 条。
- **full**（训练中每 1000 轮、训练结束、手动）：A 全网格、B、C 全部 9 级 + 噪声重复、D 全部、E。约 320 条。

实测耗时（2026-10-19，nulltask1 128 核，StairVx08 model_4999，不录像）：

| 进程数 | quick（52 条） | full（324 条） |
|---|---|---|
| 8 | 10.8 s | — |
| 16 | 9.0 s | 20.2 s |
| 32 | 9.0 s | 14.9 s |
| 64 | 10.1 s | 17.4 s |
| 96 | 10.0 s | 21.3 s |

quick 16 进程以上就卡在最长一条（30 s 静站 + 起步）上；full 32 进程最快，再多反而被进程启动开销拖慢。
训练钩子默认 worker 数 = CPU 核数的 1/4，128 核 Pod 上正好是 32。

## 用法

```bash
# 手动评测一个 ONNX（结果写到 ONNX 旁的 bench/，含 JSON 与录像）
uv run python -m se3_tools.bench.cli run --onnx logs/rsl_rl/<exp>/<run>/onnx/model_3000.onnx --suite full
# 对比两份结果的头条指标
uv run python -m se3_tools.bench.cli compare <a.json> <b.json>
```

`uv sync` 之后也可以直接用 `se3-bench run ...`。

## 训练中自动触发

`RslRlOnPolicyRunnerCfg.bench_interval`（Rough 系入口默认 1000，其余任务 0 = 关闭）。runner 每次导出 ONNX 后由
`se3_train.bench_hook.maybe_launch_bench` 判断：迭代数是 `bench_interval` 的整数倍、以及训练结束的最后一次保存，都跑 full（2026-10-19 起）。

- 独立进程（新 session、`nice -n 10`、单线程 BLAS），训练不等它；worker 数默认 CPU 核数的 1/4（`--agent.bench-workers` 覆盖）。
- 互斥锁 `onnx/bench/.lock`：上一次评测没跑完时本次排队。
- 结果：`onnx/bench/model_<N>_<suite>.json`、`.log`、`videos/`；logger 为 W&B 时另记到同 group 的 `<run>-bench`
  评测 run（横轴 iteration，含逐条结果表和三段录像：第 9 级台阶、跳跃、静站后起步）。渲染失败只跳过录像，
  结果里记 `video_error`。**nulltask1 / whtws 目前录不了像**：容器的 `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`
  不含 `graphics`，NVIDIA OpenGL 库没挂进容器，EGL 起不来（补 glvnd 厂商配置也不行）；要录像得让 Pod 规格加上
  `graphics` 能力，或在容器里装 OSMesa 走软件渲染。
- 关闭：`--agent.bench-interval 0`；smoke（`SE3_SMOKE=1`）默认不触发，`SE3_BENCH_IN_SMOKE=1` 可打开验证链路。

## 改场景

改任何网格、时长、判据或 plant 都要升 `BENCH_VERSION`；不同版本的数字不直接比较（compare 会警告）。
