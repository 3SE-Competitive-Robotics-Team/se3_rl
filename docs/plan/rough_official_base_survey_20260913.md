# Rough 训练可用的官方基座调研（2026-09-13）

问题：有没有成熟的官方 rough 训练代码（地形生成、课程升降级等），让我们只叠自己的 trick 和机器人适配。只调研，没有改代码、没有升级依赖。

## 结论

- **有，就是本地已装的 mjlab 1.5.3 自带的 `mjlab.tasks.velocity`。** 它是 Isaac Lab 官方 velocity rough 任务在 MuJoCo-Warp 上的移植，manager 体系、`TerrainGeneratorCfg`、`terrain_levels_vel` 的语义与 Isaac Lab 一一对应；本仓库所有任务都已经写在这套 API 上，所以它是唯一"拿来就能接"的官方基座。
- **它没有的**：任何轮腿或轮式机器人示例（asset zoo 只有 Go1 四足、G1 人形、YAM 机械臂）、机身高度指令、按地形列分列的指令或奖励、平地热身。这些是我们要自己叠的部分。
- **其他候选都不在我们的仿真器上**：Isaac Lab 是设计源头但跑在 Isaac Sim；legged_gym 跑在已停更的 Isaac Gym，复旦仓库是它的派生；SCUT wheeled-legged_RL 跑在 Isaac Lab，我们的 rough 已经从它移植过一次；mujoco_playground 的 rough 地形是静态 hfield XML、没有课程、训练栈是 JAX/brax。上游 mjlab 最新版 1.6.0（2026-08-09）的 tasks 目录仍只有 cartpole / manipulation / tracking / velocity，没有新增 rough 相关任务。

## mjlab velocity 任务提供了什么（按本地 1.5.3 逐文件核对）

| 官方模块 | 内容 | 与我们的关系 |
|---|---|---|
| `mjlab.terrains.config` | 17 种地形 preset（金字塔台阶正/反、开放台阶、随机台阶、斜坡、随机起伏、波浪、离散障碍、Perlin、随机格、踏石、窄梁、嵌套环、倾斜格）、`ROUGH_TERRAINS_CFG`（7 类 10 行）、`STAIRS_TERRAINS_CFG`（平地 + 三档台阶，课程模式） | 我们的 `terrains.py` 已经在用同一套生成器，只是自己手写了 6 列配置 |
| `TerrainGeneratorCfg(curriculum=True)` | 每种子地形独占一列、难度沿行插值、`proportion` 只管 env 分配 | 已在用 |
| `tasks.velocity.mdp.terrain_levels_vel` | 走过半块地形升一级；走的距离不到"指令速度 × 回合时长"的一半降一级；`update_env_origins` 在到顶后随机回到某一级；首次 reset 冻结 | 我们自己写了只升不降版，并在 A28-cap3 用 `max_level` 堵掉了到顶随机回级。复旦对照报告里"课程有升有降、到顶后重新随机采样"正是官方默认行为 |
| `tasks.velocity.mdp.commands_vel` | 按训练步数分阶段放开速度范围 | 我们是 EMA 表现驱动的自适应版（`mdp/curriculums.py`） |
| `UniformVelocityCommandCfg` | `rel_standing_envs` / `rel_heading_envs` / `rel_forward_envs`（一定比例 env 只发前向指令） | 我们是自定义 `VelocityHeightCommandCfg`（带高度指令、地形感知高度下限、按列覆盖） |
| `terminations.terrain_edge_reached`（time_out）与 `out_of_terrain_bounds` | 走到块边缘按截断结算、出网格截断 | 与我们的 `terrain_cleared` 语义相同 |
| `envs.mdp.height_scan` + `RayCastSensorCfg(include_geom_groups=(0,))` | actor 与 critic 都看高度扫描，带噪声 | 我们只给 critic，且因为机器人 geom 也在 group 0 要自己过滤自击 |
| `MetricsManager` / `MetricsTermCfg` | 逐步指标累积、按回合平均记日志 | 我们的分列奖励日志是用 interval 事件写 `extras["log"]` 硬凑的 |
| `envs.mdp.randomize_terrain` | play 模式每次 reset 随机换地形 | 我们没有 |
| `config/go1`、`config/g1` | 两份 250 行左右的机器人适配示例 | 可作为 SerialLeg 适配的模板 |
| `rl/runner.VelocityOnPolicyRunner` | 保存 checkpoint 时导出带 metadata 的 ONNX 并上传 W&B | 我们有自己的 `Se3ProfiledOnPolicyRunner` 与 `se3.meta.v1` 契约 |

官方奖励表（`track_linear_velocity`、`track_angular_velocity`、`upright`、`variable_posture`、`feet_air_time`、`feet_clearance`、`feet_swing_height`、`feet_slip`、`soft_landing`、`self_collision_cost` 等）是为足式设计的，轮腿基本用不上；这一层仍然要用我们冻结的 Flat 基线。

## 我们 rough 包与官方件的逐项对照

| 我们的模块（行数） | 官方对应 | 建议 |
|---|---|---|
| `terrains.py`（175） | preset 组合 + `TerrainGeneratorCfg` | 保留我们的几何参数（9 m 块、1.5 m 踏面、2 m 平台、0.02–0.20 m 阶高），改用 preset 写法，约 30 行 |
| `curriculums.terrain_levels`（75） | `terrain_levels_vel` | 直接换官方，恢复降级与到顶随机回级；Chebyshev 与 Euclid 在直行指令下等价 |
| `curriculums.flat_warmup`（120） | 无；官方用 `max_init_terrain_level` + 列比例起步 | 我们的 trick，是否保留待定（A15 线用了它，A18 起 warm-start 实验都关了它） |
| `terminations.terrain_cleared`（38） | `terrain_edge_reached` | 直接换官方 |
| `observations.height_scan_obs`（93） | `envs.mdp.height_scan` | 修 MJCF geom group 后直接换官方（见下） |
| `events.log_reward_split_by_column` 等（101） | `MetricsTermCfg` | 改成 metrics 项 |
| `commands.py`（365） | `UniformVelocityCommandCfg` 只有全局 `rel_forward_envs` | 保留：高度指令、地形感知高度下限、台阶列前向高速覆盖是有证据的 trick；高姿起步转移按验收表定 |
| `rewards.py` 三个按列包装 + `stair_rewards.py`（429） | 无 | 保留，但 7 份列掩码合成 1 个 helper |
| `ctbc.py`（213） | 无 | 删（R4/R5 判负） |
| `env_cfg.py`（710） | `make_velocity_env_cfg()` 的写法：工厂返回基线，机器人配置只改需要的字段 | 重写成薄覆盖 |

## 两条路

**A（推荐）：保留冻结的 Flat 基线 env，rough 只做一层薄覆盖。** 机器人、`SerialLegDelayedAction`、34 维 actor 契约、域随机化、奖励定价全部沿用 Flat；rough 层只换地形、`terrain_levels_vel`、`terrain_edge_reached`、`out_of_terrain_bounds`、官方 `height_scan`（critic）、metrics 诊断，再叠上表中"保留"的 trick。估计 rough 包从 2388 行降到 500 行以内，ONNX 契约与 sim2x runtime 不动。

**B：从 `make_velocity_env_cfg()` 起步，像 Go1 那样写 SerialLeg 配置。** 代价：官方 actor 观测含 `base_lin_vel` 与 `height_scan`（部署没有这两样）、没有高度指令、动作项 / 延迟 / 气弹簧补偿 / T-N 限幅都要重接，`se3.meta.v1` 契约和 sim2x runtime 全部重做，Flat 基线上 D2 到 D11 的结论作废。不推荐。

## 走 A 路之前要做的适配

1. **MJCF geom group。** `serialleg_closed_chain_v3_train_obb_trim.xml` 的 81 个 geom 都没写 group（默认 0），与地形同组，官方 `height_scan` 会打到自己的腿和轮子；mjlab 约定 collision 用 group 3、visual 用 group 2、射线只看 group 0。改 XML 的 group 不影响物理，但 sim2x / viewer / 各传感器要过一遍。改完就能删掉我们的自击过滤。
2. **升降级课程依赖前向指令。** 官方降级判据用"指令速度 × 时长"，对称随机指令下位移是随机游走（我们已知）。保留地形列前向指令覆盖，或改用官方 `rel_forward_envs`（全局比例，不分列）。
3. **速度课程二选一。** 官方按步数分阶段，简单可复现；我们的 EMA 自适应版有已知的种子敏感性（阈值 0.75 时靠种子）。
4. **版本不用动。** 留在 1.5.3；1.6.0 换 MuJoCo 3.11 并有 CommandTerm / CollisionCfg 破坏性改动（见 `mjlab_1_5_3_upgrade_investigation.md`），本事不需要升级。

## 参考

- mjlab 仓库与发布记录：https://github.com/mujocolab/mjlab/releases （最新 v1.6.0，2026-08-09）
- mjlab 论文（含地形网格与课程说明）：https://arxiv.org/pdf/2601.22074
- Isaac Lab `terrain_levels_vel`（语义与 mjlab 相同）：https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/curriculums.py
- legged_gym（升降级课程的源头，Isaac Gym）：https://github.com/leggedrobotics/legged_gym
- mujoco_playground（静态 hfield，无课程）：https://github.com/google-deepmind/mujoco_playground
- 其他轮腿开源（均非 MuJoCo）：https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot 、https://github.com/limxdynamics/pointfoot-legged-gym

## 实施记录（2026-09-13，A 路线已落地，未提交）

- `tasks/rough` 从 12 个文件 2388 行减到 10 个文件 1428 行（含注释与 docstring；A 路线预估的 500 行没算说明性注释）：`terrains.py` 改用官方 preset；`curriculums.py` 只剩平地热身，升降级用 `terrain_levels_vel`；`terrain_cleared` 换成 `terrain_edge_reached`（`threshold_fraction=1.0`）加 `out_of_terrain_bounds`；critic 高度扫描换成 `RayCastSensorCfg` + 官方 `height_scan`；新增 `columns.py` 合并 7 份列掩码；删除 `ctbc.py`、`observations.py`、`terminations.py`。
- 共享代码里只被 rough 实验用过的 A14（`randomize_reset_last_actions`）、A19（`command_speed_gate_range` / `min_clearance`）已删。
- geom group 改在内存里做：`robot_cfg.get_serialleg_closedchain_cfg(collision_geom_group=3)`，MJCF 文件不动，只有 rough 传这个参数；Flat 的传感器读数不受影响。
- 两处与调研文写法不同的细节：
  1. 出块截断门槛取块半边长的 1.0 倍而不是官方默认 0.95。官方升级判据是"欧氏距离 > 块半边长"，截断门槛低于它就永远升不了级；取 1.0 时直行到块边缘的那一步同时满足截断与升级。
  2. `flat_warmup` 排在 `terrain_levels` **之后**：先结算升降级再换列，换列那次 reset 的位移是旧地块上的，不能拿来升级。
- 分列奖励诊断仍走 interval 事件：`MetricsManager` 只对全部 reset 的 env 求均值，做不出"只看台阶列"的均值。
- 官方 `terrain_levels_vel` 的降级判据读 `command[:, :2]`，在本仓库指令布局里是 (vx, yaw)：台阶列 yaw 恒 0 不受影响，平地列 yaw 会抬高应走距离，但平地列各行几何相同、升降无意义。
- 默认定价取 A15；相对 A15 的唯一差别是课程从只升不降改成官方升降级。这个默认没有 W&B run 对应，第一次训练要当新基线看。
- 验证：`tests/test_rough_port.py` 重写（24 用例）与 `test_terrain_height` / `test_amp` / `test_flat_baseline` / `test_critic_learning_rate` / `test_training_queue` 全过；Rough 与 Flat-MLP 的 CPU smoke（1 env、5 轮）各导出 model_4.onnx；ruff 通过。
