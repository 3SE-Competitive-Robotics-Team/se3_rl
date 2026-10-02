# M25–M27：推力修正后的新基线，与台阶速度上限、窗口高度口径两个单变量对照（2026-09-25，用户定）

对照来源：复旦仓库台阶训练复核（yly-true/fudan_rl_wheel_leg `8204e853`，上台阶 v3 快照），
旧评审见 `docs/plan/fudan_stair_training_review_20260913.md`。

## 三处改动

1. **推力课程轮次换算（bug 修复，对三条 run 都生效）**：推力课程按 PPO 轮次分档但漏传 `steps_per_policy_iter`，
   落到默认 64，而 Flat/Rough 的 rollout 是 24，课程时钟慢 2.67 倍：首档（2000 轮 ±0.3 m/s）实际 5333 轮才出现，
   D11 Flat 基线与之前所有 5000 轮 Rough 都**从未推过**，M13–M24 的 8000 轮 run 只在最后约 2700 轮推过 ±0.3。
   修正后第 2000 轮 ±0.3、第 5000 轮 ±0.5。runner 启动时校验所有按轮次推进的课程/事件与 `num_steps_per_env` 一致。
2. **台阶列逐 env 速度上限**（`ROUGH_STAIR_SPEED_CAP_ENABLED`，M26 打开）：每个 env 记一个 vx 上限，台阶列按
   [0.4, 上限] 采样；每个 episode 结束时按达成率 r = Σmax(vx,0)/Σ指令 调整：r < 0.4 降 0.25、r ≥ 0.7 升 0.1，
   夹在 [1.0, 2.4]，初值 2.4，台阶列上不足 1 s 的 episode 不调。阈值取复旦的降级线 40% 与指令扩张线 70%。
   复旦只在"第 0 级失败 / 最高级通关"时调速度；我们的官方升降级只看 20 s 内能否走到地块边缘（平均 0.225 m/s 即可），
   管不到速度跟踪，所以改为按达成率连续调。
3. **上台阶列高度口径**（`ROUGH_STAIR_HEIGHT_REFERENCE`，M27 取 `"window"`）：地面参考从 M21 的两轮支撑面
   改为机身周围 77 点窗口均值（复用 critic 高度扫描，网格与复旦一致），罚从夹 ±0.15 m 的二次罚改为有界的
   −4·(1 − exp(−e²/0.1²))：小误差曲率不变，最大 −4/s（原 −9/s）。这是"参考面 + 有界"两个变量打包成一个实验。

另外 `*_terrain` 速度日志 M9 起混入了按平地方式发 ±2.4 对称指令的下台阶与坡道列，不能当台阶列欠速的证据；
新增只看上台阶列的 `Rough/{cmd_vx,base_vx,base_vx_error}_stairs`（三条 run 都有）。

## run 设置

nulltask1，三条并发，每条 2 卡 × 8192 envs、8000 轮、保存间隔 200、seed 42，W&B project `SE3-WheelLegged-Rough`。
三条共用同一 commit，用任务入口区分（中途切 commit 会污染在跑 run 的 ONNX 溯源）。

| 标签 | 任务 | 相对 M25 的唯一差异 | GPU |
|---|---|---|---|
| M25 | `SE3-WheelLegged-Rough` | —（门控修正 d9c1116 + 推力修正后的新基线） | 0,1 |
| M26 | `SE3-WheelLegged-Rough-Exp-StairSpeedCap` | 台阶列逐 env 速度上限 | 2,3 |
| M27 | `SE3-WheelLegged-Rough-Exp-HeightWindow` | 上台阶列高度窗口口径 + 有界罚 | 4,5 |

已知混杂：每轮样本 2 × 8192 × 24 = 39 万，是 M24（六卡）的 1/3；M25 相对 M24/`7c29rmkd` 还叠加了门控修正与推力修正。
**三条之间可比，不与 M24 及之前的 run 直接比较。**

## 判据

- **M25**：`Curriculum/two_step_gate/opened` 在热身结束后、`Curriculum/terrain_levels/stairs_up` 过 5 时才翻 1
  （M24 的曲线约在 1000–1500 轮）；`Curriculum/push_disturbance/push_vel_max` 在 2000 轮升到 0.3、5000 轮升到 0.5，
  推力上线后 `catastrophic_state` 与摔倒不失控。给出台阶列速度日志的基线水平。
- **M26**：同轮次 `Rough/base_vx_error_stairs` 低于 M25；`Curriculum/stair_speed_cap/cap_mean` 从 2.4 回落后稳定、
  `ratio_mean` 上升；`Curriculum/terrain_levels/stairs_up` 不低于 M25（降速不能伤爬台阶）。
- **M27**：`Rough/rw_flat_base_height_stairs` 比 M25 轻；`Rough/base_height_err_window_stairs` 与
  `_support_stairs` 分开看；立面动作按 M21 的口径（riser_events：触面前离地 ≤2 cm、触面后 ≤0.15 s 双轮抬起、
  轮高差 ≤8 cm），爬升不比 M21 慢；平地列逐位不变，平地跟踪不应变化。

W&B 曲线不能替代确定性回放：2000/4000/6000/7999 各跑一次台阶通关（`scripts/eval_stair_climb.py`，8 次/级）、
立面事件（riser_events）与平地/台阶固定指令扫描，三条同轮次对比。

## 启动记录

2026-09-26 00:51–00:52（Pod 时区）启动，commit `2abae57`（Pod 仓库 `xyh/925`，子模块 `5386e43`），启动器 DryRun 全部通过。
启动前 boring 隧道已开 10 h、Pod 侧 W&B 探测 5 次失败 2 次且其余 9 s 才返回，两条 DryRun 因此 exit 37；
`boring close/open` 重建后 5/5 在 2 s 左右返回 405（与 `boring-tunnel-half-open` 同一失效模式）。

| 标签 | run | W&B | PGID | state dir |
|---|---|---|---|---|
| M25 | `2026-09-26_00-51-35_rough-M25-pushfix-gatefix-seed42-2x8192-8k` | `2ww9pseu` | 553712 | `20260925T165128Z-06322` |
| M26 | `2026-09-26_00-51-57_rough-M26-stairspeedcap-seed42-2x8192-8k` | `9mtmomzq` | 554768 | `20260925T165150Z-03898` |
| M27 | `2026-09-26_00-52-21_rough-M27-heightwindow-seed42-2x8192-8k` | `eo712lx2` | 555927 | `20260925T165214Z-30361` |

首轮核验：三条都在迭代、无报错，热身期 3.11–3.18 s/轮（与 `7c29rmkd` 热身期同），每卡 11.2–11.6 GB、利用率 82–88%。
热身结束 env 分到七列后按 `7c29rmkd` 的经验会升到约 3.8 s/轮，8000 轮约 8 h。
