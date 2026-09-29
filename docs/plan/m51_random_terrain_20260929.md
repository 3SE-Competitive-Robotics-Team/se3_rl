# M51：M50 + 三列随机地形（筹划中，2026-09-29，用户定）

## 背景

M50 之前的七列都是规则几何（平台、台阶、光滑坡），没有随机起伏或不规则小障碍。actor 是盲的，真机遇到接缝、线槽、
起伏路段都是训练里没见过的情况。用户决定加随机地形，新地形的 MDP 与 flat 保持一致，高度参考先不变。
与 M50 分开成单独一条（M50 已叠三处改动，混在一起无法归因）。

## 改动（相对 M50 唯一差异：地形集）

`env_cfg(random_terrains=True)` → `terrains.rough_terrains_cfg(random_terrains=True)`，在七列之后加三列，全部 heightfield、格子 0.2 m
（0.1 m 格子会让机身碰撞块超过 50 个三角形上限而丢接触），外圈边 0.5 m：

| 列 | mjlab 地形 | 参数（随难度 0 → 1 线性） |
|---|---|---|
| random_rough | HfRandomUniformTerrainCfg | 每格高度 U(0, 上限)，上限 0 → 5 cm，noise_step 5 mm |
| wave | HfWaveTerrainCfg | 幅值 0 → 8 cm，9 m 地块 4 个波（波长约 2.25 m） |
| obstacles | HfDiscreteObstaclesTerrainCfg | fixed 模式只有凸起、不挖坑；高 2 → 10 cm，宽 0.4–1.0 m，每块 40 个，中心 2 m 出生平台留空 |

生成核验（`.scratch/m51/terrain_stats.py`，相对出生点的地面高度）：

| 列 | 第 0 级 | 第 5 级 | 第 9 级 |
|---|---|---|---|
| random_rough | 平 | ±1.1–1.4 cm | ±2.5 cm（峰峰 5 cm） |
| wave | 平 | ±4.0 cm | ±7.9 cm（峰峰 16 cm，最大坡度约 12.7°） |
| obstacles | 最高 2 cm | 最高 6 cm | 最高 10 cm |

整张地形 824 个 geom、50 个 heightfield（新增 30 个）。

比例（合计 1.0）：上台阶两列 0.43 不动；从 flat、两坡、二级下台阶匀出 0.20，stairs_down 0.14 保留（下台阶是共同短板）。

| flat | stairs_up | two_step_up | two_step_down | stairs_down | slope_up | slope_down | random_rough | wave | obstacles |
|---|---|---|---|---|---|---|---|---|---|
| 0.07 | 0.28 | 0.15 | 0.06 | 0.14 | 0.05 | 0.05 | 0.08 | 0.07 | 0.05 |

## 新列的 MDP（与 flat 一致）

- 指令：三列名加进 `commands.ROUGH_TERRAIN_COMMAND_FLAT_NAMES`，按平地发（vx 随速度课程到 2.4、yaw 随平地课程、静站与高姿起步转移都采）。
  名单里写了当前地形集没有的列名会被跳过，其他入口不受影响。
- 奖励：不进台阶分列定价、台阶专项奖励与接触税置零名单，和 flat 同一套（含 flat_wheel_contact、collision、is_alive）。
- 高度参考不变：非台阶列仍是机身正下方单点射线。开过障碍时参考会随障碍高度跳变，这是已知代价，先观察。
- 速度课程信号仍只读 flat 列（flat 比例降到 0.07，七卡 × 8192 仍约 4000 个 env）。

## 待办

- 启动前用 `scripts/bench_rough_sim.py` 实测迭代时间（heightfield 与轮子碰撞在 mjwarp 上的开销未测）。

## 验证

rough 41 项测试通过（默认地形集不变）；M51 入口 CPU smoke 通过；相对 M50 奖励、终止、课程、观测、动作逐项相同。
