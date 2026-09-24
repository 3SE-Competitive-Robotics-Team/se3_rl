# M24：新增二级台阶训练列，上行/下行两条 + 等级门控（2026-09-21，用户定）

对照基线：M23（W&B `7om3sfb1`，commit 44b6237，六卡 8000 轮，跑到 2600 轮时用户在 sim2x 里判定效果满意）。
**本改动只落代码，未启动训练**；M23 仍在跑。

## 改动

训练地形从五列变六列，新增 `stairs_two_step`，复刻 `assets/robots/serialleg/mjcf/serialleg_fudan_sim2sim_steps.xml`
里那道"二级台阶"。新地形类 `TwoStepStairsTerrainCfg`（`rough/terrains.py`），环形铺法与 `stairs_up` 一致
（机器人生在中心、向外走），外圈顶面对齐 z=0 与相邻块平滑衔接。

| 难度 | 第一级高 | 第一级踏面 | 第二级高 | 第二级踏面 | 外圈下降 |
|---|---|---|---|---|---|
| 0.0 | 0.05 m | 0.60 m | 0.04 m | 0.60 m | 0.05 m |
| 1.0 | **0.20 m** | **0.60 m** | **0.15 m** | **0.15 m** | **0.05 m** |

难度 1 精确等于源几何，射线剖面实测：`x=1.00` 上 0.200、`x=1.62` 上 0.150、`x=1.76` 下 0.050，终点 +0.300。
源场景（机器人沿 +x 走）是：地面 → 上 0.20（踏面 0.60）→ 上 0.15（踏面 0.15）→ 下 0.05 进入长平台。
第二级踏面收窄（0.60 → 0.15）是这一列的核心课程量。

### 为什么它和 stairs_up 是两类问题

第二级不是普通台阶，而是一道 **0.15 m 宽的凸棱**：上 15 cm 之后立刻下 5 cm。轮距 0.433 m 决定了
两只轮不可能同时站在上面，所以它考的是"跨棱"，而 `stairs_up` 的等高宽踏面（0.7 m）金字塔考的是"爬楼梯"。

### 列名开关统一

新增 `terrains.ROUGH_STAIR_LIKE_COLUMNS = ("stairs_up", "stairs_two_step")`，作为"哪些列算上台阶"的唯一定义。
原先散落在两个文件里的八处开关全部改为引用它，免得以后再加列时漏掉某一处：

| 开关 | 位置 |
|---|---|
| 台阶专项奖励（进度、双轮支撑）与台阶列置零项、宽核 σ=1.44 | `env_cfg.ROUGH_REWARD_TERRAIN_TYPE_NAMES` |
| 窄核速度跟踪 w=1（M22） | `env_cfg.ROUGH_TRACKING_LIN_VEL_NARROW_STAIR_COLUMNS` |
| 左右轮高度差罚（M23） | `env_cfg.ROUGH_WHEEL_HEIGHT_DIFF_COLUMNS` |
| 支撑面高度参考（M21） | `env_cfg.ROUGH_BASE_HEIGHT_SUPPORT_COLUMNS` |
| 接触税豁免 | `env_cfg.ROUGH_CONTACT_TAX_FREE_COLUMNS`（再并上 stairs_down） |
| 台阶指令（vx 0.4–2.4 前向、yaw ±0.3） | `commands.ROUGH_STAIR_COMMAND_TERRAIN_NAMES` |
| 地形感知高度下限 | `commands.ROUGH_TERRAIN_STEP_HEIGHT_TYPE_NAMES` |
| 按平地方式发指令的列（不含新列） | `commands.ROUGH_TERRAIN_COMMAND_FLAT_NAMES` |

高度下限读的是 `step_height_range`，新地形把它定义为**第一级**高度 (0.05, 0.20)，即两级里更高的那一级，
难度 1 时下限仍是 0.20 + 0.02 + 0.12 = 0.34，与 stairs_up 一致。

### 台阶奖励的等效几何

`stair_climb_progress` / `stair_support_height` 的 `_geometry()` 原本假设等高等宽多级，对两级不等高算不对。
给 `SubTerrainCfg` 加了一个可选钩子 `se3_stair_geometry(difficulty)`，有则优先用它自报的
`(阶高, 第一道立面半径, 爬升段长度, 可奖励级数)`，没有则走原公式（其余五列行为逐位不变）。

二级台阶自报的阶高取**第二级**（难度 1 时 0.15）：`stair_support_height` 用 `floor(rise / 阶高)` 数级数，
而 rise 在第一级是 0.20、第二级是 0.35，取 0.15 能分别数成 1 和 2；取第一级的 0.20 会把第二级也数成 1，第二级就白爬了。
`start = 1.0`（中心平台半边）、`length = 0.60 + 第二级踏面`、`count = 2`。

## 代价

| | 五列（M23） | 六列（M24） |
|---|---|---|
| 地形 geom（10 行合计） | 450 | **620** |
| 每块 geom | flat 1 / stairs_up 21 / stairs_down 21 / slope 1 | 再加 stairs_two_step **17** |
| 预计单轮 | 2.59 s | 约 2.9–3.0 s（按 geom 数外推，未实测） |
| 8000 轮 ETA | 5.9 h | 约 6.5 h |

env 分配比例从 stairs_up 与两条坡各匀出来：flat 0.15、stairs_up 0.30、stairs_two_step 0.20、
stairs_down 0.15、slope_up 0.10、slope_down 0.10。

## 验收（启动后）

1. 难度 1 的剖面仍是 0.20 / 0.15 / −0.05（`.scratch` 的射线剖面脚本）。
2. `Curriculum/terrain_levels/stairs_two_step` 能爬升；`Rough/rw_stair_*_stairs` 不因新列拖低。
3. 新列的确定性回放：需要先生成一个二级台阶的 sim2x 评测场景（`scripts/make_stair_scene.py` 需扩展）。
4. 原五列不退：平地传递曲线与 20 cm 通关按 M23 口径复测。

## 风险

- 新列与 stairs_up 共享同一套台阶奖励与指令，"跨棱"和"爬楼梯"两种流形可能互相干扰，
  像 M18 那样把一列学到的习惯带到另一列。先看两列课程等级是否都能涨。
- 难度 1 的第二级踏面 0.15 m 小于轮径 0.12 m 的两倍，可能根本上不去，课程会卡在中段。
  若 `terrain_levels/stairs_two_step` 长期停在 5 以下，把 `second_step_width_range` 下限放宽到 0.25。


## 补充（同日，用户追加要求）

### 1. 拆成上行与下行两条列

`TwoStepStairsTerrainCfg` 加 `descending` 开关，同一道剖面正反两走：

| 列 | 出生高度 | 向外依次 | 对应源场景 |
|---|---|---|---|
| `stairs_two_step_up` | −0.30 m | 上 0.20（踏面 0.60）→ 上 0.15（踏面 **0.15**）→ 下 0.05 | 机器人从地面往平台爬 |
| `stairs_two_step_down` | +0.30 m | 上 0.05（踏面 **0.15**）→ 下 0.15（踏面 0.60）→ 下 0.20 | 机器人从平台往台阶方向走 |

两列的外圈都锚在 z=0，与相邻地形块平滑衔接。射线剖面逐项实测通过。

**待遇按用户要求分开**：上行列进 `ROUGH_STAIR_LIKE_COLUMNS`，与 `stairs_up` 完全同待遇（台阶专项奖励与宽核 σ、
台阶指令 vx 0.4–2.4 前向 + yaw ±0.3、地形感知高度下限、窄核 w=1、支撑面高度参考、轮高度差罚、接触税豁免）；
下行列进 `ROUGH_TERRAIN_COMMAND_FLAT_NAMES`，按平地待遇（速度跟平地课程到 ±2.4、有偏航跟踪、无台阶专项奖励、
不抬高度下限），另外像 `stairs_down` 一样豁免接触税——下行时轮子离地与碰撞是地形本身造成的，不豁免等于按地形罚钱。

### 2. 等级门控 `curriculums.two_step_gate`

`stairs_up` 的平均难度等级达到 `ROUGH_TWO_STEP_GATE_LEVEL = 5.0`（9 级里的中段，约 11 cm 阶高）之前，
两条二级台阶列的 env 暂放到各自的"母列"：上行去 `stairs_up`，下行去 `flat`。达标后一次性放开，**不再回收**
（等级会随策略波动，反复迁移会让这些 env 的课程等级与 episode 统计来回重置）。

两个实现要点：
- 判据只统计"本来就属于 `stairs_up`"的 env，不受门控期迁进来的样本影响。
- 必须排在 `flat_warmup` **之后**，且只处理已结束热身的 env：热身期全体在平地列，这时迁移会把还在热身的 env
  提前拽到 `stairs_up`；原始列名也优先复用热身记下的那份，否则第一次运行时 clone 到的是"热身把大家改成 flat 之后"的快照。
- 日志：`Curriculum/two_step_gate/opened`、`Curriculum/two_step_gate/gate_level`。

### 3. 最终比例与代价

| 列 | flat | stairs_up | two_step_up | two_step_down | stairs_down | slope_up | slope_down |
|---|---|---|---|---|---|---|---|
| 比例 | 0.15 | 0.28 | 0.15 | 0.10 | 0.14 | 0.09 | 0.09 |
| 每块 geom | 1 | 21 | 17 | 17 | 21 | 1 | 1 |

十行合计 geom **450 → 790**（+76%）。单轮预计从 2.59 s 涨到 3.2 s 上下，8000 轮 ETA 约 7.1 h。
门控期（stairs_up 均级 < 5）这两列没有 env，但几何仍然生成，所以这份开销从第 0 轮就在。

### 4. 测试

全量 **41 例**连跑五次通过，新增：
- 两列的剖面（上行 −0.30/−0.10/+0.05/0，下行 +0.30/+0.35/+0.20/0）与环宽（窄棱 0.15 在内、宽踏面 0.60 在外）。
- `TwoStepGateRuntimeTests`：未达标时两列为空且 env 落在母列、达标后拿回自己的 env 且等级归 0、等级掉回去不回收；
  以及门控在课程表里排在 `flat_warmup` 之后。
- 共享运行时 fixture 关掉门控（否则那两列没有 env，无法验证各列奖励行为），门控由上面的专门用例钉住。
- 顺带把 M19 轮错位罚那条同样脆弱的 `_step_reward` 逐位比对改成"纯函数精确比对 + 符号检查"，与 M21/M22/M23 一致。

## 启动记录（2026-09-22）

- M23 在 3181 轮 SIGINT 停掉（进程组 4 s 清空、六卡释放）。commit `738268f`，bundle 同步到 nulltask1
  （远端干净，HEAD 44b6237 → 738268f）。本地 41 例连跑五次通过，CPU smoke 五轮通过。
- 启动器 DryRun 通过后正式启动：`.scratch/launch_m24.ps1`，GPU 0–5 × 8192，8000 轮，保存间隔 200，seed 42。
- run `2026-09-22_00-56-17_rough-M24-twostep-updown-gate5-wheeldz40-seed42-6x8192-8k`，W&B `x87zbdbf`，
  PID/PGID 405176，state `/workspace/.se3-training-state/nulltask1/20260921T165610Z-29409`。
- 第 12 轮：**3.16 s/轮**（M23 五列是 2.59 s，+22%，与 geom 450 → 790 的预估一致），ETA 7:09，
  六卡各约 12.9–13.3 GB（M23 约 12.0 GB）、利用率 81–85%，无 `nefc overflow`、无报错。
  门控按预期关着：`Curriculum/two_step_gate/opened` 0、`gate_level` 0（还在平地热身，stairs_up 均级 0）。
- 后续观察点：约 500–1000 轮热身结束后 `gate_level` 开始涨；`opened` 翻到 1 的那一轮即二级台阶两列上线，
  之后看 `Curriculum/terrain_levels/stairs_two_step_up` / `_down` 能否跟着爬，以及 stairs_up 的等级是否被拖慢。

## 出生点与助跑距离（三条台阶列一致）

中心平台都是 2.0 m 见方（半边 1.0 m），reset 时 xy 各随机 ±0.1 m、yaw 全随机，所以到第一道立面：
轴向 0.90–1.10 m，对角最远 1.51 m。

| 列 | 轴向 1.00 m | 之后 | 再之后 |
|---|---|---|---|
| `stairs_up` | 上 0.20（踏面 0.70） | 每 0.70 m 再上 0.20，共 4 级 | — |
| `stairs_two_step_up` | 上 0.20（踏面 0.60） | 1.60 m 处上 0.15（踏面 **0.15**） | 1.75 m 处下 0.05，外圈 2.25 m |
| `stairs_two_step_down` | 上 0.05（踏面 **0.15**） | 1.15 m 处下 0.15（踏面 0.60） | 1.75 m 处下 0.20，外圈 2.25 m |

下行列的窄棱在**第一道**：跨上去的瞬间前轮已经悬在下一级上方，比上行列（窄棱在第二道）更紧。


## 收尾（2026-09-22）：跑满 8000 轮，但门控有 bug

- 8000 轮跑满自行退出，耗时 **11:26:17**（末段 3.65 s/轮，比第 12 轮的 3.16 s 又慢了，因为门控打开后
  env 真正散到七列）。无 `nefc overflow`、无报错，41 个 checkpoint 齐全，六卡已释放。W&B `x87zbdbf` state=finished。

### 结果：新列可学，没有拖垮原有列

| 轮次 | stairs_up | two_step_up | two_step_down |
|---|---|---|---|
| 1000 | 4.38 | 5.07 | 4.55 |
| 2000 | 6.25 | 6.31 | 5.83 |
| 4000 | 6.42 | 6.29 | 5.92 |
| 7000 | 6.54 | 6.44 | 5.91 |
| 末值 | **6.57** | **6.32** | **5.97** |

上行列几乎追平 `stairs_up`，下行列稳定在 5.9。计划里担心的"0.15 m 窄棱可能根本上不去、课程卡在中段"没有发生。

同轮次对照 M23（3080–3180 窗口）：`stairs_up` 6.27 对 5.63、台阶列实速 0.438 对 0.412，两项都更好；
但 `rw_stair_climb_progress_stairs` 0.60 对 0.81、`rw_stair_support_height_stairs` 1.45 对 1.86、
`Train/mean_reward` 29.7 对 34.9 都更低——这三项是按 `ROUGH_REWARD_TERRAIN_TYPE_NAMES`（两列）取的均值，
样本构成变了（`stairs_up` 比例 0.35 → 0.28，又多出一条更难的上行列），不能直接当退化读。

### bug：门控在第 499 轮之前就打开了，实际没起作用

`Curriculum/two_step_gate/gate_level` 在热身期（第 0–1039 轮）就已经是 6.0，第 499 轮之前 `opened` 就翻成 1，
而那时官方的 `Curriculum/terrain_levels/stairs_up` 还是空值（热身期没有 env 在那一列）。

根因：门控判据读的是 `terrain.terrain_levels[original == stairs_up 列]`，按**原始列归属**取等级。
但热身期这些 env 全被 `flat_warmup` 放在**平地列**上，而官方的 `terrain_levels_vel` 在热身期照常结算升降级——
平地上走得远就升级，所以它们的 `terrain_levels` 早早涨到 6。门控读到的是"这些 env 在平地上的等级"，
不是"stairs_up 这一列的难度"。

修法（一行）：判据改成按**当前所在列**取，
`gate_mask = terrain.terrain_types == names.index(gate_terrain_name)`。热身期没有 env 在 stairs_up 列，
mask 为空、level 保持 0，门控不会打开；热身结束后统计的就是真实的台阶难度（包含门控期迁进来的 two_step_up env，
它们和原生 env 在同一列同一难度，统计它们没问题）。

影响：本轮训练里两条二级台阶列从热身结束（约 1039 轮）起就有 env，等于**没有门控**。
从等级曲线看没造成灾难（两列都涨到 6 附近），所以这次的结果仍可用；但"等级到 5 才开放"这个要求没有真正验证过。
