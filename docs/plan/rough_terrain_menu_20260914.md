# mjlab 可用地形清单与难度参数（2026-09-14，供选型，未改代码）

课程语义：`TerrainGeneratorCfg(curriculum=True)` 下每种子地形独占一列，行号映射难度
`difficulty = row / (num_rows − 1)`，我们 `num_rows=10`，所以 row 0 是 range 下界、row 9 是上界。
升降级由官方 `terrain_levels_vel` 管（走够远升一行，走不动降一行）。

**`ROUGH_TERRAIN_PROPORTIONS` 的键就是列数**：比例设 0 不能关掉一列，只是白占 geom（四个零比例列曾多花 0.6 s/轮），
不要的列必须不放进字典。

## Box 类（纯 box geom，射线与碰撞便宜，推荐优先）

| preset | difficulty 0 → 1 插的是什么 | 固定参数 | 备注 |
|---|---|---|---|
| `flat` | 无 | — | 当前占 0.30 |
| `pyramid_stairs_inv` | `step_height_range[0] → [1]` | `step_width`、`platform_width`、`border_width` | **当前 stairs_up**，出生在坑底向外爬升 |
| `pyramid_stairs` | 同上 | 同上 | 正金字塔，出生在顶部平台向外下行（下台阶） |
| `random_stairs` | 采样高度 × (0.5 + 0.5·d) | `step_width` 默认 0.8 | 每级高度随机，不是规则楼梯 |
| `open_stairs` | 高度 `range[0]→[1]`，**同时**宽度 `range[1]→[0]` | `step_thickness` 0.05、`inverted` | 悬空踏板（有厚度、下面是空的），越难越高越窄 |
| `box_random_grid` | `grid_height_range[0] → [1]`，格高在 ±该值内均匀采样 | `grid_width` | 棋盘式高低块，最接近"碎石地/砖堆" |
| `random_spread_boxes` | 块数 × (0.5+0.5·d)，块高 × (0.2+0.8·d) | `box_width/length/yaw_range` | 地面上散落箱子，可保留平地底面 |
| `stepping_stones` | 石块尺寸 `max → min`（缝隙随之变大），高度/位移扰动 × d | `stone_height`、`floor_depth` | 石块间是深坑，掉下去就是终止 |
| `narrow_beams` | 梁宽 `max → min` | `num_beams`、`spacing`、`beam_height` | 独木桥，梁间是深坑 |
| `tilted_grid` | 倾角 `tilt_range_deg × d`，高差 `height_range × d` | `grid_width` | 每格随机倾斜的板块 |
| `nested_rings` | 环宽 `max → min`，缝 `gap_min → gap_max` | `num_rings`、`height_range` | 同心圆台阶/沟 |

## Hfield 类（高度场，便宜但有碰撞坑）

| preset | difficulty 0 → 1 | 备注 |
|---|---|---|
| `hf_pyramid_slope` / `_inv` | `slope_range[0] → [1]`（rise/run 比值） | 上坡 / 下坡，A9 前用过 |
| `random_rough` | **默认忽略 difficulty**（`scale_with_difficulty=False` 时全程满幅） | 要课程必须显式开 `scale_with_difficulty=True` |
| `wave_terrain` | `amplitude_range[0] → [1]` | `num_waves` 控制波数 |
| `discrete_obstacles` | `obstacle_height_range[0] → [1]` | 平地上立柱/台块 |
| `perlin_noise` | `height_range[0] → [1]` | 连续起伏，比 random_rough 平滑 |

**hfield 的硬约束**：`horizontal_scale` 必须取 0.2 m。MuJoCo 凸体-hfield 按 AABB 覆盖的格子逐格生成三角棱柱、
单对上限 50 个三角形；机身最大碰撞块 0.52 m 宽在 0.1 m 格子下要 98 个三角形，接触会被直接丢弃、机身穿进地形
（R1 崩溃的物理侧诱因）。0.2 m 格子只要 32 个。

## 本机器人的几何门槛（选参数时的硬约束）

| 量 | 值 | 含义 |
|---|---|---|
| 轮半径 | 0.06 m | 障碍高超过约 0.12 m（两倍轮半径）就必须靠腿抬，纯滚不上去 |
| 轮距（横向） | 0.433 m | `stepping_stones`/`narrow_beams` 的石块或梁宽必须 > 0.433 才能双轮同时落，否则本质是走独轮 |
| 基座碰撞盒纵向 | 0.52 m | 踏面/块宽小于它时，跨立面必然同时压两级（M4 的 0.5 m 踏面就是这个情况） |
| 机身指令高度 | 0.20–0.38 m | 地形感知下限 = 障碍高 + 0.02 + 0.12，20 cm 障碍就把下限顶到 0.34 |
| 单块地形 | 9×9 m，边框 0.5，平台 2.0 | 可用半径 (9−1−2)/2 = 3 m 放地形 |

## 加一列时必须一起决定的四件事

1. **列比例**：`ROUGH_TERRAIN_PROPORTIONS` 里给多少 env（当前 flat 0.30 / stairs_up 0.70）。
2. **指令覆盖**：`commands.py` 里非平地列默认 vx 0.4–0.8、yaw ±0.2；台阶列单独是 vx 1.0–2.4、yaw 0。新列走哪一档，
   要不要进 `ROUGH_TERRAIN_STEP_HEIGHT_TYPE_NAMES`（决定是否按障碍高抬机身高度下限）。
3. **奖励生效范围**：台阶专项奖励（`stair_climb_progress`/`stair_support_height`）、三项置零（`is_alive`/轮离地/碰撞）、
   yaw 跟踪置零、运动核放宽，现在都只认 `("stairs_up",)`。新列要不要纳入，每一项都要单独定。
4. **迭代时间**：box 类每块 geom 数直接决定墙钟（1.5 m 踏面 13/级、0.7 m 21/级、0.5 m 29/级；
   四列零比例地形曾占 160 geom、多 0.6 s/轮）。加列前可用 `scripts/bench_rough_sim.py` 实测。

## 已知与当前配置的关系

- 现在只有两列：`flat` 0.30、`stairs_up` 0.70（`pyramid_stairs_inv`，step_height 0.02–0.20、step_width 0.7、platform 2.0、border 0.5）。
- A9 之前有过 `stairs_down`/`slope_up`/`slope_down`/`random_rough` 四列，比例设 0 后仍白占 geom，2026-09-13 删除。
- 高姿死锁（`m5_highstand_deadlock_20260914.md`）与列无关，但会影响新列的选型：凡是把机身高度下限顶到 0.34 以上的障碍高，
  当前策略在那一行就是走不动的。
