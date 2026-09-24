# Rough 训练 iteration 时间复盘（2026-09-13）

问题：W&B 上 rough 的 `Perf/iteration_s` 偏长（M1 hqcn4y2f 约 2.9 s，A28 只有 2.4 s，Flat 1.58 s）。
要找不改训练结果的降法，并对照 mjlab / mujoco_warp 官方资料看地形有没有更省的用法。只调研 + 写基准脚本，没有改训练配置、没有提交任务。

## 结论

1. **时间几乎全在 collection，且与地形 geom 数近似线性。** learning 恒 0.12 s。8192 env/卡、24 步/轮：

   | 地形 | 地形 geom | 宽相配对 `nxn_geom_pair_filtered` | collection s/轮 | 来源 |
   |---|---|---|---|---|
   | Flat（1 个 plane） | 1 | 56 | 1.46 | 2026-09-06 实测 |
   | 两列 flat + stairs_up（144 box） | 144 | 8 064 | 2.17–2.29 | A27 / A28（1k–1.5k 轮） |
   | 六列（274 box + 30 hfield 45×45） | 304 | 17 024 | 3.03–3.15（1k 轮）；M1 415 轮时 2.79 | A21–A26、M1 |

   机器人有 56 个碰撞 geom（54 mesh + 2 cylinder），宽相配对 = 56 × 地形 geom 数 + 自碰对，mjwarp NXN 核的线程数 = 世界数 × 配对数。

2. **六列里四列是"死列"。** `ROUGH_TERRAIN_PROPORTIONS` 把 stairs_down / slope_up / slope_down / random_rough 设为 0，mjlab 课程模式仍各给 1 个 env（共 4/8192），
   但它们贡献 160 个 geom（130 个 box + 30 个 hfield，11.6 万三角形）、把宽相配对翻倍。A27/A28 用的是只有两列的生成器，671db84 重写换回了六列版
   `rough_terrains_cfg()`，M1 的 2.9 s 就是这么回来的。同阶段对比：删掉四列可省约 0.8 s/轮，5000 轮约 1 小时。

3. **官方资料里没有"更省的地形写法"，只有更贵的。** mjlab `ROUGH_TERRAINS_CFG` 是 20 列 × 10 行、8 m 块、hfield 0.1 m 分辨率，官方 Go1/G1 rough 直接吃这个开销；
   `terrain.rst` / `faq.rst` 没有性能条目。mujoco_warp 文档的性能条目只有一条：`nconmax` / `njmax`
   "Memory and computation scales with the values of these parameters. For best performance, the values of these parameters should be set as small as possible
   while ensuring the simulation does not exceed these limits."，配 `mjwarp-testspeed --measure_alloc` 量实际接触/约束数。

4. **以下不是杠杆，别再花时间：**
   - 宽相换 SAP：mujoco_warp 3.10 `_src/io.py:664` 自动选择规则是配对数 < 250 000 用 NXN，我们 8k–17k，远在阈值内；最多做 5 分钟 A/B。
   - 射线：mjlab 1.5.3 的 `RayCastSensor` 走 `SensorContext` + `refit_bvh` + `_ray_bvh`（BVH），`sim.sense()` 每个 env 步只调一次，106 条射线不是热点。
   - `contact_sensor_maxmatch=500`：只是溢出上界，不在热路径。
   - 多卡：数据并行，每卡 8192 env，iteration 时间不随卡数下降。

## 不改训练结果就能做的

| 措施 | 预期 | 代价 / 验证 |
|---|---|---|
| 删掉比例为 0 的四列（`rough_terrains_cfg` 只生成 proportion > 0 的子地形，或直接恢复两列） | 2.9–3.1 → 2.3–2.4 s/轮（A27/A28 已证明） | 4 个 env 换列、`Curriculum/terrain_levels/{stairs_down,slope_*,random_rough}` 日志键消失；`tests/test_rough_port.py:218` 断言六列名要改 |
| 按官方建议缩 `nconmax` / `njmax`（现在沿用 Flat 的 256 / 1040；官方 rough 35 / 1500，go1 None / 300） | 未知，mjwarp 文档说计算量随之缩 | 先用基准脚本量每世界最大 ncon / nefc，取 2× 余量；溢出会丢接触，必须留余量 |
| 宽相 `sap_tile` / `sap_segmented` A/B | 大概率无差或更慢 | 基准脚本一次跑完 |

## 会改任务的（属于实验变量，不能混进性能优化）

减 `num_rows`（10 行 → 课程粒度变粗）、加宽踏面 / 改 `platform_width` / `border_width`（每块台阶 13 个 box 里 4 个是块边框）、
降 hfield `horizontal_scale`、减 critic 射线（77 条）——都改机器人看到的地形或观测。见 `experiment-discipline-single-variable`。

## 实测脚本

`scripts/bench_rough_sim.py`：只建 `ManagerBasedRlEnv`（不建 runner、不上传 W&B），零动作 + 0.1 噪声，量 `env.step` / `sim.step` / `sim.sense` 毫秒，
折算 24 步一轮的 collection 秒数，并输出每世界最大 ncon / nefc（定 nconmax / njmax 用）。本机没有 GPU，只做了 CPU 冒烟（2 env × 2 步，默认与 sap_tile 两条路径都能跑）。

```bash
uv run python scripts/bench_rough_sim.py --device cuda:0 --num-envs 8192 --steps 96 --variants 6col-default,2col-default,2col-sap_tile,flat-default --event-trace
```

`--event-trace` 直接调 `mjwarp.step` 打印宽相 / 窄相 / 求解器各阶段的平均毫秒。需要一张空闲卡，跑一遍约 5–10 分钟；跑不跑、何时跑由用户定。

## 追问：能不能做到和 Flat 一样快（2026-09-13 晚）

做不到，除非改机器人碰撞模型或改地形本身。两列 rough 与 Flat 的 0.7–0.8 s/轮差距由四部分构成，能不改训练结果就去掉的只有一小部分：

| 组成 | Flat | 两列 rough | 能不能不改训练结果地去掉 |
|---|---|---|---|
| 宽相配对（每世界每子步的球包围测试） | 56 | 8 064（机器人 56 个碰撞 geom × 144 个地形 box） | 不能。mjwarp 模型全世界共享，每个世界都要对全部地形 geom 测一遍；只能靠减 geom（死列已删）或减机器人碰撞 geom（改物理） |
| 球包围测试通过后的 OBB 测试 | 每世界 1–2 对 | 每世界 170–250 对（长 box 的包围球太大：网格边框 50 m、平地块 6.4 m、台阶环 4.6 m） | 能。开 `broadphase_filter=("plane","sphere","aabb","obb")` 后本机模拟只剩 1–2 对进 OBB；过滤器是保守的，接触结果不变。收益要实测 |
| 窄相 | 轮-平面是 PRIMITIVE 特化 | 轮-box、腿-box 是 CONVEX（GJK/EPA） | 不能，box 地形就是这样 |
| 传感器与项数 | 3 个高度传感器 19 条射线 | 5 个 106 条射线 + 多 3 个奖励、2 个终止、2 个课程项 | 射线已是 BVH 且每步一次；项数是任务定义 |

另外两个官方性能旋钮对 Flat 和 rough 同样有效，能把 rough 的绝对时间压下去，但它们不是"地形"问题：

1. **`njmax` / `nconmax`。** mujoco_warp 求解器有 8 处 kernel 直接按 `(nworld, njmax)` 起线程（`_src/solver.py:1502,1509,2039,2058,3906,3927,4354,4362`），约束相关 6 处按 `naconmax`。
   我们沿用 Flat 的 njmax=1040 / nconmax=256；本机 2 个世界静立时 nefc=48。官方 velocity 任务 njmax=1500 / nconmax=35，go1 用 300 / None。
   溢出时 mjwarp 会打印 "nefc overflow - please increase njmax" 并置 `d.overflow`，该世界那一步的物理不对，所以必须先量训练态的每世界最大 nefc / ncon 再留 2–3 倍余量。
2. **求解器迭代上限 `iterations`。** mjwarp 用 `wp.capture_while` 循环到**所有**世界收敛或达到 `opt.iterations`（`_src/solver.py:3992`），8192 个世界里最慢的那个决定整批跑几轮。
   我们用 MuJoCo 默认 100 / ls 50，官方 mjlab velocity 用 10 / 20。它只影响没在上限内收敛的世界，是否等价要看基准里的 `solver_niter_max`：最大值远小于上限，缩上限就是零成本。
   需要 CUDA 驱动 ≥ 12.4 才有条件图，否则 mjlab 会关掉 CUDA graph、mjwarp 退回固定跑满 `iterations` 轮的分支，那样 100 轮就是纯浪费；基准输出 `use_cuda_graph` / `graph_conditional` / `cuda_driver` 三个字段就是查这个。

基准脚本已加：变体后缀 `+aabb`；全局 `--njmax --nconmax --iterations --ls-iterations`；输出 `solver_niter_max/mean`、`overflow_worlds`、CUDA graph 状态。建议在 Pod 空闲卡上按这个顺序跑：

```bash
uv run python scripts/bench_rough_sim.py --device cuda:0 --num-envs 8192 --steps 96 --variants flat-default,2col-default,2col-default+aabb,2col-sap_tile --event-trace
uv run python scripts/bench_rough_sim.py --device cuda:0 --num-envs 8192 --steps 96 --variants 2col-default+aabb --njmax 300 --nconmax 64
uv run python scripts/bench_rough_sim.py --device cuda:0 --num-envs 8192 --steps 96 --variants 2col-default+aabb --iterations 10 --ls-iterations 20
```

基准用的是零动作加噪声，接触数比真训练少；nefc / ncon 的余量要按训练态定，最稳的办法是在 runner 里顺手记一条 `Perf/nefc_max`（未做）。

会改物理的大招只有一个值得记下：机器人 54 个 mesh 碰撞 geom 换成十来个原语，配对数降 5 倍、窄相从 GJK 变原语特化，rough 能接近 Flat；但接触几何变了，Flat 基线上的结论要重跑，属于新实验线。

## 实测（2026-09-13 晚，nulltask1 GPU 4，A800，8192 env，24 步/轮折算；0–3 号卡的 M1 同时在训）

第一轮：地形与宽相。基准里的 collection 比训练少了策略推理与日志，但比例与训练一致（六列/两列 = 1.35，训练 1.3–1.35）。

| 变体 | 地形 geom | 宽相配对 | env.step ms | 物理 ms（4 子步） | 射线 ms | collection s/轮 |
|---|---|---|---|---|---|---|
| flat | 80 | 56 | 43.8 | 16.4 | 2.8 | 1.05 |
| 六列（M1 现状） | 383 | 17 024 | 102.7 | 50.3 | 4.2 | 2.46 |
| 两列 | 223 | 8 064 | 76.2 | 27.8 | 3.8 | 1.83 |
| 两列 + AABB 过滤 | 223 | 8 064 | 75.7 | 29.1 | 3.9 | 1.82 |
| 两列 + sap_tile | 223 | 8 064 | 82.4 | 34.2 | 3.8 | 1.98 |

删四个死列 −26%；AABB 过滤零收益（球测试本身才是大头，OBB 那 200 对不值钱）；SAP 慢 8%，与 mjwarp 自己的阈值判断一致。

第二轮：官方旋钮（都在两列上）。每世界最大 nefc 40–89、最大接触 7–18，都没溢出（`overflow_worlds`=0）。

| 设置 | env.step ms | 物理 ms | collection s/轮 |
|---|---|---|---|
| njmax 1040 / nconmax 256（现状） | 76.1 | 28.6 | 1.83 |
| njmax 256 / nconmax 64 | 70.8 | 24.8 | 1.70 |
| njmax 128 / nconmax 32 | 70.9 | 23.6 | 1.70 |
| iterations 10 / ls 20（njmax 1040） | 77.0 | 26.3 | 1.85 |
| njmax 256 + iterations 10 | 71.1 | 22.6 | 1.71 |
| flat，njmax 256 / nconmax 64 | 39.5 | 13.1 | 0.95 |

njmax 1040→256 再省 7%（flat 也省 9%），128 与 256 无差，取 256 留 3–5 倍余量。求解器迭代上限 10 在 CUDA graph 下没有收益（`solver_niter_max` 本来只有 9–11），不改。

第三、四轮：把 env.step 拆到 manager 与单项。两列 env.step 76 ms = 物理 4 子步 ≈ 28 + `sim.forward` ≈ 5–19 + 射线 3.8 + manager ≈ 22。manager 里：

| manager | flat ms | 两列 ms | 每次调用主机同步 |
|---|---|---|---|
| command.compute | 7.7 | 7.8 | `.item()` 184 次、`nonzero` 79 次、stream sync 162 次 |
| reward.compute | 5.5 | 8.3 | `.item()` 2 / 14 次 |
| observation.compute | 3.2 | 3.3 | 0 次 `.item()` |
| termination.compute | 1.8 | 1.9 | `.item()` 18 次（`catastrophic_state` 0.94 ms） |

**command.compute 的 7.8 ms 里 7.4 ms 是 `velocity_height._update_metrics`**，flat 与 rough 一样：每步 184 次 `.item()` 把 GPU 流水线打断 160 多次，占 flat 一步的 18%、两列的 10%，纯日志，不进任何奖励或观测。
rough 多出的 3 ms 奖励在 `stair_support_height`（1.0）、`tracking_lin_vel` 的分列版（0.94 对 flat 0.52）、`stair_climb_progress`（0.4）和 14 次 `.item()`。

### 建议（按收益）

1. rough 覆盖层删四个比例为 0 的列：−0.63 s/轮（基准），对应训练约 2.9 → 2.3 s。
2. rough 覆盖层设 `cfg.sim.njmax = 256`、`cfg.sim.nconmax = 64`：再 −0.13 s/轮；首个 run 盯日志里有没有 "nefc overflow" / "broadphase overflow"。Flat 若也改，同样 −9%。
3. `velocity_height._update_metrics` 去掉逐项 `.item()`（张量累加、迭代末再落主机）：flat/rough 各 −0.18 s/轮，结果不变，但动的是共享 mdp 代码，要用户点头。
4. 不做：AABB 过滤、SAP、求解器迭代上限、射线。

三项都做，两列 rough 从现状六列的 2.46 到约 1.5 s/轮（基准口径，−40%）；flat 从 1.05 到约 0.77。

脚本：`scripts/bench_rough_sim.py`（变体 / 旋钮 / event-trace / `--profile-managers`）与 `scripts/profile_env_terms.py`（逐项计时 + 主机同步计数）。Pod 上原始日志在 `/workspace/.se3-bench/round{1,2,3,4}*.{log,jsonl}`。

## 待办（2026-09-13 用户定：先记录，后续再改）—— 当晚已全部改完，见文末「改动记录」

### 1. 删掉四个比例为 0 的地形列

现状 [terrains.py:41](../../src/se3_train/tasks/rough/terrains.py) 的 `ROUGH_TERRAIN_PROPORTIONS`：flat 0.30、stairs_up 0.70，以下四列为 0，
mjlab 课程模式仍各分 1 个 env（共 4/8192，0.05%），但几何全量生成，宽相配对 8 064 → 17 024：

| 列 | 几何 | geom 数 |
|---|---|---|
| stairs_down | 正金字塔台阶，10 行 × 13 box | 130 |
| slope_up | 反金字塔斜坡 hfield（45×45），10 行 | 10 |
| slope_down | 正金字塔斜坡 hfield（45×45），10 行 | 10 |
| random_rough | 随机起伏 hfield（45×45），10 行 | 10 |

这四列是 2026-09-09 A9 时把比例设 0 的，当时保留列只为日志键不变；671db84 重写沿用了这份表。

改法：`rough_terrains_cfg()` 只生成 proportion > 0 的子地形（或直接删掉四个条目），同步改：
- [env_cfg.py:89](../../src/se3_train/tasks/rough/env_cfg.py) `ROUGH_ALL_TERRAIN_TYPE_NAMES` 六个名字（`columns.py` 对缺失列名会跳过，不改也不报错，但常量应与实际列一致）；
- [tests/test_rough_port.py:218](../../tests/test_rough_port.py) 断言 `list(gen.sub_terrains) == list(ROUGH_ALL_TERRAIN_TYPE_NAMES)`，224–234 行断言 stairs_down / slope_up / slope_down 的类型；
- W&B 里 `Curriculum/terrain_levels/{stairs_down,slope_up,slope_down,random_rough}` 四个键消失，对比旧 run 时知道即可。

预期：基准 2.46 → 1.83 s/轮；训练口径 M1 的 2.9 → 约 2.3 s（A27/A28 两列实跑 2.29–2.41）。这是对照 M1 的性能变量，不与奖励/课程改动混在同一个 run。

### 2. rough 覆盖层缩 njmax / nconmax

在 [env_cfg.py:159](../../src/se3_train/tasks/rough/env_cfg.py) `cfg.sim.contact_sensor_maxmatch` 旁加 `cfg.sim.njmax = 256`、`cfg.sim.nconmax = 64`（现沿用 Flat 的 1040 / 256）。
实测每世界最大约束 40–89、最大接触 7–18；128 与 256 同速，取 256 留 3–5 倍余量。首个 run 看日志有没有 mjwarp 的
"nefc overflow - please increase njmax" / "broadphase overflow"。预期再 −0.13 s/轮；Flat 若同改 −9%，但那是冻结基线，另议。

**2026-09-13 晚 M1 真配置溢出验证（用户要求）**：`scripts/check_sim_overflow.py` 在 GPU 4 上用 M1 的 model_2400 按训练方式采样动作，
六列地形、8192 env、起步行 0–9 均匀（比 M1 的 `max_init_terrain_level=0` 更狠）、2000 步 = 两个完整 episode（31 k 次回合结束，
末态各行分布 [2528, 722, 924, 912, 750, 1927, 429, 0, 0, 0]，即 7–9 行的 env 都经历过最高台阶再被降级）：

| 设置 | 每世界约束峰值 nefc | 每步峰值 p50 / p90 / p99 | 每世界接触峰值 | 接触池占用 | 宽相候选峰值 | overflow |
|---|---|---|---|---|---|---|
| njmax 1040 / nconmax 256（现值） | 54 | 29 / 45 / 46 | 10 | 16 933 / 2 097 152 | 27 378 | 0 |
| njmax 256 / nconmax 64 | 50 | 29 / 37 / 46 | 9 | 16 916 / 524 288 | 27 115 | 0，五类标志全 0，日志无 mjwarp 溢出 printf |

njmax 256 是峰值的 4.7 倍，nconmax 64 是每世界接触峰值的 6 倍、总池只用 3%。结论：可以安全改。

### 3. 去掉 `JumpCommandTerm._update_metrics` 的逐项 `.item()`

[jump_commands.py:570](../../src/se3_train/mdp/jump_commands.py) 起每步约二十个 `Jump/diag_*` 指标逐个 `.item()`，
实测每步 184 次 `.item()`、162 次 stream sync、7.4 ms，flat 与 rough 相同，纯日志。改法二选一：保持 0 维张量交给 RSL-RL logger
迭代末统一取均值（基类 `reset` 已按此约定，见 [commands.py:174](../../src/se3_train/mdp/commands.py)），或跳机制关闭时整段跳过。
顺带：`catastrophic_state` 终止项每步 18 次 `.item()`（0.9 ms）。预期 flat / rough 各 −0.18 s/轮。动的是共享 mdp，要单独提交、跑全量测试。

三项合计：两列 rough ≈ 1.5 s/轮（基准口径，较现状 −40%），flat ≈ 0.77。不做：AABB 过滤、SAP、求解器迭代上限、射线数。

## 改动记录（2026-09-13 晚，用户批准三项一起改；未提交）

1. **删死列**：[terrains.py](../../src/se3_train/tasks/rough/terrains.py) 的 `ROUGH_TERRAIN_PROPORTIONS` 只剩 flat 0.30 / stairs_up 0.70，
   `rough_terrains_cfg()` 只生成这两列，hfield 相关 preset、`_slope`、`_HF_HORIZONTAL_SCALE` 一并删除（0.2 m 分辨率的教训写进模块 docstring）；
   `stair_only_terrains_cfg()`（台阶定向评测，含 stairs_down）不变。[env_cfg.py](../../src/se3_train/tasks/rough/env_cfg.py) 的
   `ROUGH_ALL_TERRAIN_TYPE_NAMES` 改为 `tuple(ROUGH_TERRAIN_PROPORTIONS)`，与地形表同源。
2. **约束池 / 接触池**：env_cfg 新增 `ROUGH_NJMAX = 256`、`ROUGH_NCONMAX = 64`，在 `contact_sensor_maxmatch` 旁写进 `cfg.sim`，
   台阶定向评测任务同用。首个 run 盯日志有没有 "nefc overflow" / "broadphase overflow"。
3. **Jump/* 诊断**：[jump_commands.py](../../src/se3_train/mdp/jump_commands.py) 的 `_update_metrics`、`_takeoff_reward_diagnostics`、
   `_symmetry_diagnostics`、`_mean_on_mask` 全部改成 0 维张量、无 `.item()`、无按数据分支（掩码均值用 clamp 分母，空中最大 vz 用 where），
   EMA 改成 (ema, seen) 张量对（`*_ema` 键改为每步都报，未命中前为 0）；跳跃线沿用。行走线 [flat/env_cfg.py](../../src/se3_train/tasks/flat/env_cfg.py)
   的 JumpCommandCfg 设 `enable_jump_metrics=False`（recovery / stair 线早已如此；log_filter 本来就裁掉 Jump/*，rough 通过字段拷贝继承）。
   `Jump/diag_leg_contact_*` 由终止项写入、仍有 `.item()`，不在本次范围（0.3 ms）。

测试：`tests/test_rough_port.py` 改六列断言为两列、新增 `test_sim_pool_sizes_follow_overflow_measurement`（钉 256 / 64 且不低于实测峰值 3 倍）和
`test_jump_metrics_are_host_sync_free`（打开诊断后禁掉 `.item()` 跑一遍，键齐、全是 0 维有限张量）；`tests/test_flat_baseline.py` 钉
`enable_jump_metrics=False`。全量 `unittest discover` 105 个用例通过；三个基准脚本按新默认（变体名 `rough`）冒烟通过。

预期（基准口径）：rough 2.46 → 约 1.5 s/轮，训练口径 M1 的 2.9 → 约 1.9；flat 因 `enable_jump_metrics=False` 也省约 0.18 s/轮。
对照 M1 时这是三个性能变量叠加的 run，奖励 / 课程 / 观测都没动；四个课程日志键 `Curriculum/terrain_levels/{stairs_down,slope_up,slope_down,random_rough}` 消失。

## 参考

- mujoco_warp 文档（性能调优、nconmax/njmax、testspeed）：https://mujoco.readthedocs.io/en/latest/mjwarp/index.html
- mujoco_warp 宽相实现与自动选择：`.venv/Lib/site-packages/mujoco_warp/_src/collision_driver.py`、`_src/io.py:664`
- mjlab 地形文档：https://github.com/mujocolab/mjlab/blob/main/docs/source/terrain.rst ；FAQ：https://github.com/mujocolab/mjlab/blob/main/docs/source/faq.rst
- mjlab 性能讨论（Go1 平地，6–7 倍差距是 mujoco_warp 装错版本）：https://github.com/mujocolab/mjlab/discussions/220
- 本仓库官方基座调研：`docs/plan/rough_official_base_survey_20260913.md`
