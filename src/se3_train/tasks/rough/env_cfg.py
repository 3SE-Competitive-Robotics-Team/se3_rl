"""崎岖地形行走任务环境配置：冻结的 Flat 基线 + 一层薄的 rough 覆盖。

覆盖层里能用 mjlab 官方件的都用官方件：地形 preset 与课程模式生成器（terrains.py）、地形难度升降级
`terrain_levels_vel`（走过半块升级、走不到指令距离一半降级、到顶后随机回级）、出块截断
`terrain_edge_reached` 与出网格截断 `out_of_terrain_bounds`、critic 的高度扫描 `height_scan`。
本仓库自己的部分只剩有实验证据的几项（证据见 docs/plan/stair_training_wandb_review_20260913.md）：

- 指令（commands.py）：机身高度指令 + 地形感知高度下限；上台阶列独立采样前进与偏航速度，其余列沿用平地指令。
- 奖励（stair_rewards.py / rewards.py）：台阶进度与双轮支撑两项专项奖励；台阶列置零高度罚与 yaw 工资、
  放宽运动核；速度违令罚；非平地列关 vz 项。A12 十组消融证明前四项缺一即不上台阶。
- 课程（curriculums.py）：前 500 轮平地热身 + 500 轮 ramp；平地速度课程只看平地列（events.py）。

默认定价取从零训练最好的 A15（W&B h85eljnj）：非台阶列高度 σ 0.10、运动核分母 0.5、平地 vz 项 0、
违令罚全列、能耗三项与 Flat 同价。相对 A15 的差别：课程换成官方升降级；以及 M2（2026-09-13）的台阶列定价——
台阶列 is_alive / flat_wheel_contact / collision 置零，加 mjlab `is_terminated` 摔倒罚（见 ROUGH_STAIRS_ZEROED_REWARDS 注释）。
M3：台阶列加回机身高度罚（见 ROUGH_BASE_HEIGHT_SUPPORT_COLUMNS 注释）。
M6：非台阶列运动核 0.5 → 1.0，补起步段梯度（见 ROUGH_OFF_STAIR_TRACKING_SIGMA_MOVE 注释）。
M7：速度跟踪权重 4 → 6，把运动从每秒亏 15 翻成赚（见 ROUGH_TRACKING_LIN_VEL_WEIGHT 注释）。
M8：平地列注入高姿起步转移，解开探索瓶颈（见 ROUGH_HIGH_STAND_TRANSITION_PROB 注释）。
M9：补下台阶与上下坡三列；新列按平地方式发指令（±2.4 + yaw），接触税置零扩到下台阶。
M10：修 M9 的 bug——下行地形正常往下走会跌破 catastrophic_state 的 −0.5 m 下限被判物理发散
（见 ROUGH_CATASTROPHIC_MIN_BASE_HEIGHT 注释）。
M11：非台阶列运动核退回 0.5，找回中高速段的分辨率（见 ROUGH_OFF_STAIR_TRACKING_SIGMA_MOVE 注释）。
M12：njmax 256 → 512，五列地形下 256 一直在溢出丢约束（见 ROUGH_NJMAX 注释）。
M15：全程叠加窄核速度跟踪（w=1、σ=0.04）；M17：窄核权重 1 → 3（见 ROUGH_TRACKING_LIN_VEL_NARROW_WEIGHT 注释）。
M18：左右轮前后错位罚 w·Δx²，台阶列置零；M19：改成只罚平地列 + 10 cm 死区（见 ROUGH_WHEEL_OFFSET_WEIGHT 注释）。
M20：窄核速度跟踪在上台阶列置零；M22：台阶列恢复到 w=1（见 ROUGH_TRACKING_LIN_VEL_NARROW_STAIR_WEIGHT 注释）。
M23：上台阶列新增左右轮高度差罚，堵死走梯（见 ROUGH_WHEEL_HEIGHT_DIFF_WEIGHT 注释）。
M24：新增二级台阶上行/下行两列（复旦那道凸棱），上行列按 stairs_up 待遇、下行列按平地待遇；
八处台阶开关统一引用 terrains.ROUGH_STAIR_LIKE_COLUMNS；两列由 curriculums.two_step_gate 门控，
stairs_up 均级到 5 才开放（见 ROUGH_TWO_STEP_GATE_LEVEL）。
M21：上台阶列机身高度罚的地面参考改为轮子支撑面（见 ROUGH_BASE_HEIGHT_SUPPORT_COLUMNS 注释）。

机器人实体与 Flat 同一个 MJCF，只把碰撞 geom 从 group 0 改到 group 3（内存里改，不动文件），
让 `include_geom_groups=(0,)` 的高度射线只看地形，不再打到自己的腿和轮子。
"""

from __future__ import annotations

from dataclasses import fields, replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import height_scan
from mjlab.envs.mdp.rewards import is_terminated
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.mdp.curriculums import terrain_levels_vel
from mjlab.tasks.velocity.mdp.terminations import out_of_terrain_bounds, terrain_edge_reached
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from se3_train.robot_cfg import get_serialleg_closedchain_cfg
from se3_train.tasks.flat.env_cfg import (
    FLAT_ACTION_SMOOTHNESS_SPRING,
    FLAT_CMD_VEL_DEADBAND,
    FLAT_COMMAND_VELOCITY_ERROR_WEIGHT_LEGACY,
    FLAT_WHEEL_ACTION_SCALE,
)
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg

from . import curriculums, events, rewards, stair_rewards
from .commands import (
    ROUGH_BODY_COLLISION_BOTTOM_OFFSET,
    ROUGH_STAIR_ANG_VEL_YAW_RANGE,
    ROUGH_STAIR_COMMAND_TERRAIN_NAMES,
    ROUGH_STAIR_HEIGHT_RANGE,
    ROUGH_STAIR_LIN_VEL_X_RANGE,
    ROUGH_TERRAIN_ANG_VEL_YAW_RANGE,
    ROUGH_TERRAIN_COMMAND_FLAT_NAMES,
    ROUGH_TERRAIN_HEIGHT_CLEARANCE,
    ROUGH_TERRAIN_LIN_VEL_X_RANGE,
    ROUGH_TERRAIN_STEP_HEIGHT_TYPE_NAMES,
    RoughCommandCfg,
)
from .terrains import (
    ROUGH_STAIR_LIKE_COLUMNS,
    ROUGH_TERRAIN_PROPORTIONS,
    ROUGH_TWO_STEP_DOWN_COLUMN,
    ROUGH_TWO_STEP_GATE_LEVEL,
    ROUGH_TWO_STEP_UP_COLUMN,
    rough_terrains_cfg,
)

# mjlab 资产库约定：碰撞 geom group 3、视觉 geom group 2、射线传感器只看 group 0（地形）。
ROUGH_ROBOT_COLLISION_GEOM_GROUP = 3
# 全部 env 从最简单一行起步，难度由官方课程逐级放开（它到顶后会随机回级，起点不必随机）。
ROUGH_MAX_INIT_TERRAIN_LEVEL = 0
# 三个接触传感器的 secondary 都是 pattern="terrain"，生成器地形有几百个 geom，64 个匹配槽会溢出
# （运行时刷 "contact match overflow"，接触力读数不可信）。mjlab 自己的 rough velocity 任务同样取 500。
ROUGH_CONTACT_SENSOR_MAXMATCH = 500
# mjwarp 的约束池 / 接触池按每世界上限分配，求解器 kernel 也按 (nworld, njmax) 起线程，官方文档说这两个值
# "越小越快，前提是不溢出"。沿用 Flat 的 1040 / 256 时 rough 每轮多 0.13 s。
# 2026-09-13（两列地形）用 M1 的 model_2400 压测得 256 / 64 零溢出。
#
# **2026-09-15 五列地形后 njmax 256 不够**：M9 / M10 / M11 的 train.log 里 "nefc overflow - please
# increase njmax to N" 分别刷了 3900 / 32433 / 10892 次，N 实测 294–402。M11 从 754 轮（平地热身一结束、
# 第一次踩上五列地形）就开始溢出。溢出时约束被丢弃，接触力不可信，那三轮的物理与结论都要打折。
# 取 512 覆盖实测最大 402 并留余量；nconmax 那一路从未报过溢出，维持 64。
#
# **压测方法的教训**：4096 env × 1000–2000 步的 check_sim_overflow.py 峰值只有 45–68，据此判断"够用"是错的——
# 训练是 8192 env × 4 卡跑几千轮，撞到的极端情况远多于压测；而且脚本报的 overflow_worlds 即使在放大池后
# 仍出现，当时被误判成"与池无关"。**换地形后以真实训练日志里的 nefc overflow 为准**，压测只能证伪不能证明够用。
ROUGH_NJMAX = 512
ROUGH_NCONMAX = 64
# 课程按训练轮次计数用的每轮步数，与 rl_cfg 的 num_steps_per_env 一致（由测试钉住）。
ROUGH_STEPS_PER_POLICY_ITER = 24
ROUGH_FLAT_WARMUP_ITERATIONS = 500
ROUGH_FLAT_WARMUP_RAMP_ITERATIONS = 500
# 出块截断的门槛：块半边长的比例。官方升级判据是"到出生点的欧氏距离 > 块半边长"，截断门槛不能低于它，
# 否则永远升不了级；取 1.0 时直行到块边缘的那一步同时满足截断与升级。
ROUGH_TERRAIN_EDGE_THRESHOLD_FRACTION = 1.0

# 分列定价生效的列、平地速度课程读的列。
ROUGH_REWARD_TERRAIN_TYPE_NAMES = ROUGH_STAIR_LIKE_COLUMNS
ROUGH_CURRICULUM_SIGNAL_TERRAIN_NAMES = ("flat",)
ROUGH_CURRICULUM_TRACKING_LOG_KEY = "Locomotion/tracking_lin_vel_reward_curriculum"
ROUGH_VZ_FLAT_TERRAIN_TYPE_NAMES = ("flat",)
# 训练地形的全部列，与 terrains.ROUGH_TERRAIN_PROPORTIONS 的键同源（2026-09-13 起只有 flat 与 stairs_up）。
ROUGH_ALL_TERRAIN_TYPE_NAMES = tuple(ROUGH_TERRAIN_PROPORTIONS)

# 台阶专项奖励（A12；消融 A7/A8：撤掉任一项台阶列速度归零）。
ROUGH_STAIR_CLIMB_PROGRESS_WEIGHT = 3.0
ROUGH_STAIR_SUPPORT_HEIGHT_WEIGHT = 4.0
# 速度违令二次罚（A6 起台阶列，A15 起全六列）。误差归一化尺度 3.0：0.5 时台阶列全程贴封顶 9、
# 梯度没了且 −18/s 的常数负奖励会教出自杀策略（A10）；3.0 时误差 1.65 → −0.57/s、2.35 → −1.23/s。
ROUGH_COMMAND_VELOCITY_ERROR_WEIGHT = FLAT_COMMAND_VELOCITY_ERROR_WEIGHT_LEGACY
ROUGH_COMMAND_VELOCITY_ERROR_LIN_SCALE = 3.0
# A15 非台阶列定价。A13b flat 列账本（96 env，指令 vx 2.0 / h 0.38）：站着不动净 +2.407/s、走路净
# −2.932/s，走路必然产生的机身起伏被高度罚、核里的 vz 项、姿态罚罚了三遍。σ 0.05→0.10 收回 +3.11/s，
# 运动核 0.08→0.5 与 vz 2.0→0 合计收回 +2.97/s；违令罚扩到全列只打在"不动"那边。
ROUGH_BASE_HEIGHT_SIGMA = 0.10
# 非台阶列运动核。A15 定 0.5；M6（2026-09-14）为补高姿起步梯度放到 1.0；
# M11（2026-09-15 用户定）退回 0.5——高姿起步已由 M8 的 high_stand_transition 解决，
# 而 1.0 的副作用是中高速段没有分辨率：M10-4000 实测五列在指令 2.0 下全线欠速 0.48–0.74
# （平地 1.52、slope_down 1.26），平地扫描里指令 0.8 冲到 1.37、1.2 只有 1.57，
# 实际速度被压在 1.4–1.6 一条带上出不去。算术原因：σ=1.0、指令 2.0 时跑 1.5 能拿 0.78 的核值、
# 跑满才 1.0，差距只有两成；σ=0.5 时同样差距是 0.45 对 1.0。
# 这一项是 M6 单独改的，退回去不影响 M7 的跟踪权重 6 与 M8 的起步转移。
ROUGH_OFF_STAIR_TRACKING_SIGMA_MOVE = 0.5
# M7（2026-09-14 用户定）：速度跟踪权重 4 → 6（Flat 基线是 4）。M6-1200 在真实 env 的全项账本
# （平地列、去 push，docs/plan/m6_ledger_full_20260914.md）：高姿站着不动 171.25/s、低姿跑 1.7 m/s
# 156.22/s —— 站着不动是全局最优，不是局部最优。跑起来赚 tracking +111.6，却亏掉轮离地 −36.9、
# 高度 −25.9、摔倒 −20.8、碰撞 −16.7、腿触地 −12.0 等共 −136，净亏 15。缺口只有 15/s：
# 权重 4→6 让运动那侧多 +71.7 而静止只多 +15.9，差值 −15 → +41，余量够覆盖策略不成熟期。
# 只在收益一侧加码，不放松任何安全约束（轮离地、碰撞、摔倒罚都原样保留）。
ROUGH_TRACKING_LIN_VEL_WEIGHT = 6.0
# 叠加精细速度跟踪，静站与运动均生效；0.04 对应 0.2 m/s 误差尺度。
# M17（2026-09-20 用户定）：权重 1.0 → 3.0，相对 M15（4flo3d80）的唯一改动。依据是 M15-7999 的反事实账本
# （docs/plan/m15_tracking_diagnosis_20260920.md 评测四）：奖励最优已经是"跟上指令"，但跟准比策略自己选的冻结/慢模态
# 每秒只多赚 +1.1…+2.3（总回报约 10/s），宽核 σ=0.5 在 ±0.4 m/s 内近似平顶；w=3 把这份激励抬到 +2.4…+3.7/s
# （30 cm/0.4：1.8 → 3.3；38 cm/0.4：1.1 → 2.4；38 cm/−0.4：2.3 → 3.7）。σ 不动，宽核不动。
# 验收：30 cm 的 0.2–1.2 传递曲线是否离开 0.08/0.32/0.59/1.76 的台阶；台阶 0.34 下 0.8–1.4 通关不退。
ROUGH_TRACKING_LIN_VEL_NARROW_WEIGHT = 3.0
# M20（2026-09-21 用户定）：窄核在上台阶列置零，其余四列保持 w=3。立面事件量化（docs/plan/m19 附录、
# .scratch/m18_eval/riser_events.py）：用户要的流形是"贴地撞面、机身靠动量先过台阶沿、腿一收双轮同抬"，M15 在立面处
# 正是如此（触面前离地 0 cm、触面后 0.06 s 双轮抬起、左右轮高差 5 cm）；这种爬法每道立面必掉一次速（M15 爬升段最低
# 0.15 m/s），w=3 的窄核把掉速罚贵了，策略改成不掉速的走梯（M17，双腿分开）或起跳（M18，台阶列轮离地罚 M2 起为零、跳是免费的）。
# 台阶列窄核奖励 M15 0.07 → M17 0.68 是它在台阶上为窄核优化的证据。
# M22（2026-09-21 用户定）：台阶列从"置零"恢复到 **w=1**（M15 的值，不是 M17 的 3），其余四列仍是 3；
# 这是相对 M21（ee88a1d）的唯一改动。依据是 M21-2600 的评测与归因（docs/plan/m21_height_support_ref_20260921.md 末两节）：
# 置零后台阶列只剩 σ=1.44 的宽核与违令罚，策略在 v=1.0 下每道立面要 2.0 s（M15-7999 是 0.75 s）、爬升段均速 0.38
# （M15 0.86），抬轮从触面后 0.06 s 推迟到 0.30 s；给 v=1.4 时立面间隔回到 0.9–1.3 s、抬轮 0.12–0.20 s，
# 说明慢是"没有速度激励"而不是姿态缺陷。同一慢推姿态泄漏到平地低速段（指令 0.4 实速 0.23，M17-2000 是 0.41；
# 轮子动作指令只有一半、前倾深 4–8°），M18/M20 都没有这个退化。
# 取 1 不取 3：M15（w=1 全列）是唯一同时具备目标流形与高速的模型，M17 的 w=3 才把策略推向走梯/起跳。
# 分列权重靠 rewards.column_scaled 实现（RewardTermCfg 只有一个 weight）。
ROUGH_TRACKING_LIN_VEL_NARROW_STAIR_COLUMNS: tuple[str, ...] = ROUGH_STAIR_LIKE_COLUMNS
ROUGH_TRACKING_LIN_VEL_NARROW_STAIR_WEIGHT = 1.0
# M18（2026-09-21 用户定）：左右轮前后错位罚。M17-7999 评测（docs/plan/m17_narrow_w3_20260920.md 附录）：
# 前进时左右轮心在机身系里错开 +20…+29 cm（跨立步态，轮心距 0.433 → 0.49–0.52 m），38 cm 静站 −26 cm，
# 倒退翻成右轮在前；M15 在 22/30 cm 也有 2–11 cm、38 cm 16–20 cm。机理：两轮前后错开后俯仰不再是倒立摆，
# 跟踪近似静定，窄核 w=3 抬高了它的价值。joint_mirror −0.179 是关节空间量，Δx=28 cm 只花 0.18/s，等于免费。
# 直接罚几何量 w·Δx²：w=40/m² 使 25 cm 花 2.5/s（与跟踪激励 +2.4…+3.7 同量级）、5 cm 只花 0.1/s；
# 台阶列置零（爬升时一条腿先上，Δx 周期摆到 ±20 cm 是动作本身）。相对 M17 的唯一改动。
ROUGH_WHEEL_OFFSET_WEIGHT = 40.0
# M19（2026-09-21 用户定）：改形，权重不变。M18 跑到 4468 轮的中途核查：平地 |Δx| 已压到 1–5 cm，但共享策略把
# "两轮齐平"学成全局习惯，台阶上也齐平，M17 那种一只轮先上 20 cm（轮高差 p95 20 cm）的走梯方式没了，只剩
# 深前倾（−21°）双轮同抬的"扑着上"，用户判定动作流形不对。改成：只罚平地列（坡列、下台阶列不罚），
# 并加 10 cm 死区——静站/平地行驶时 20–29 cm 的错位仍花 0.4–1.4/s，10 cm 以内免费，爬梯的一先一后不再被同一习惯压平。
ROUGH_WHEEL_OFFSET_DEAD_ZONE_M = 0.10
ROUGH_WHEEL_OFFSET_COLUMNS: tuple[str, ...] = ("flat",)
# M23（2026-09-21 用户定）：上台阶列新增左右轮**高度**差罚，堵死走梯（一只轮先上一级、另一只在下一级推地）。
# 为什么罚 Δz 不罚 Δx：爬升段两种流形的 |Δz| p95 切得很干净——目标流形 M15-7999 2.7/4.2、M18 0.2–2.7、
# M21-2600 1.2/4.1，走梯 M17-7999 14.1/20.4、M22-1200 14.0/18.4；而前后错位 |Δx| 在两组完全重叠
# （M15 均值 −10.5/−9.2 比 M22 的 −7.4/−7.8 还大），罚 Δx 会先把目标流形罚掉（docs/plan/m23_wheel_dz_20260921.md）。
# 机制（.scratch/m22_eval/leg_pair.py）：M22 两条腿是同时收的（时差 0.00–0.08 s），慢的是后轮——
# 左轮 0.12–0.24 s 抬起、右轮要 0.18–0.50 s，因为两轮前后错开后前轮先够到立面。根因是撞面速度不足
# （M15 1.24–1.31 m/s 靠动量整体越沿，M22 只有 0.86–0.93、20 cm 第一道甚至 −0.10 被弹回），
# 动量不够时"双轮同抬"要更大的瞬时减速，错开左右轮则不掉速，而台阶列的 Δx 罚 M19 起就关了，错开是免费的。
# 定价：死区 0.08 留在 M15 的 p95（4.2 cm）之上两倍，爬升摆动免费；12 cm 罚 0.064/s、14 cm 0.144/s、
# 18 cm 0.40/s、22 cm 0.78/s，与台阶列窄核收益（0.125/s）同量级，够抵消"错开省速度"的好处。
ROUGH_WHEEL_HEIGHT_DIFF_WEIGHT = 40.0
ROUGH_WHEEL_HEIGHT_DIFF_DEAD_ZONE_M = 0.08
ROUGH_WHEEL_HEIGHT_DIFF_COLUMNS: tuple[str, ...] = ROUGH_STAIR_LIKE_COLUMNS
ROUGH_TRACKING_LIN_VEL_NARROW_SIGMA = 0.04
ROUGH_FLAT_VZ_WEIGHT = 0.0
# 台阶列运动核分母（A11）：误差约 1 m/s 时仍有半额奖励，给低速前进提供可区分的回报。
ROUGH_STAIR_TRACKING_SIGMA_MOVE = 1.44
# 非平地列 tracking_lin_vel 核里的 vz 系数（A7）：爬台阶和上坡必须有垂直速度。
ROUGH_TERRAIN_VZ_WEIGHT = 0.0
# M2（2026-09-13 用户定）：台阶列上置零的三项。M1 hqcn4y2f 的台阶列账本（每秒贡献）显示"冻住不动"净 +0.94/s
# （is_alive +1.0、零速时跟踪核仍 +0.54、违令罚 −0.60），而"努力爬"在 1600 轮净 −0.4/s：轮离地罚 −0.7、
# base 碰地形罚 −0.5 正好罚在抬轮跨立面这个动作上，is_alive 则是"站着的工资"（与 A10 删掉 yaw 工资同理）。
# 官方升降级把台阶 env 堆在成功率约一半的行，按此账本"尝试"要成功率超过约 40% 才划算，去掉工资后约 12%。
# 确定性回放在 1400 轮长出常数动作不动点（docs/plan/m1_model800_vs_1400_20260913.md）就是这个失衡的产物。
ROUGH_STAIRS_ZEROED_REWARDS = ("is_alive", "flat_wheel_contact", "collision")
# 三项接触税在哪些列置零（2026-09-15）：上下台阶都会抬轮跨立面、机身也会蹭到台阶，
# 坡面不给——坡上轮子本来就该一直着地，置零等于放掉唯一的接触约束。
ROUGH_CONTACT_TAX_FREE_COLUMNS = (
    *ROUGH_STAIR_LIKE_COLUMNS,
    "stairs_down",
    ROUGH_TWO_STEP_DOWN_COLUMN,
)

# catastrophic_state 的机身高度下限（2026-09-15 修 M9 的 bug）。Flat 基线取 −0.5 m，判据是
# 世界 z 减 env 原点，本意是"掉出地图或物理发散"。但下行地形的出生点在**顶部平台**
# （pyramid_stairs 与 hf_pyramid_slope 都是正金字塔），机器人正常往下走就会跌破 −0.5 被判发散：
# M9（W&B upidjdsa）1697 轮时 catastrophic 2.63/轮、平均回报从 34 掉到 8，而课程等级照升
# （升级看水平位移），就是这个误判。按当前几何，可用半径 3.0 m、每侧 4 级台阶：
#   stairs_down 最难级底 −0.80 m（7 级时下完 3 级就触发）
#   slope_down  最难级底 −1.05 m（6 级时离中心 2.1 m 就触发）
# 取 −1.5 m 覆盖最深的 −1.05 并留 0.45 m 余量；平地与上行地形行为不变。
# 上限 3.0 m 不动：slope_up 最高 +1.2、stairs_up 最高 +0.8，都在内。
# **以后再加下行地形，先按 (每侧级数 × 最大阶高) 或 (最大坡度 × 可用半径) 核这条下限。**
ROUGH_CATASTROPHIC_MIN_BASE_HEIGHT = -1.5
# 摔倒罚（一次性，按事件计）：工资拿掉后台阶列每秒净值接近 0 甚至为负，非超时终止按 0 自举就等于"免费退出"，
# 提前摔死会变便宜（A10 的自杀策略）。mjlab `is_terminated` 对所有非 time_out 终止（灾难、倾倒）记 1；
# RewardManager 按 dt 缩放奖励，所以权重取 −ROUGH_FALL_PENALTY / step_dt，使每次终止恰好扣 ROUGH_FALL_PENALTY。
# 量级取剩余 episode 可能负值的上界：20 s × 0.5/s = 10。
ROUGH_FALL_PENALTY = 10.0
# M3（2026-09-13 用户定）：台阶列加回机身高度罚。M1/M2 这一项在台阶列置零（base_height_penalty_off_terrain），
# M2-4999 确定性回放在平地中速指令（0.8–2.0）进 0.8 Hz 弹跳极限环（z 极差 20 cm），台阶列高度不受约束是怀疑对象；
# M3 起全列生效（σ 取 ROUGH_BASE_HEIGHT_SIGMA，误差夹 ±0.15 m）。
# M21（2026-09-21 用户定）：上台阶列的地面参考从"base_link 正下方射线（base_height_sensor）"改为"两轮下方射线"
# （轮子支撑面，现成的 stair_reward_height 传感器，取有效命中均值）。旧口径下机身沿一越过台阶边，参考地面瞬间抬一阶，
# 误差夹满 0.15 → 峰值 −9/s，直到机身升完这一阶：M15-7999 第 9 级每道 20 cm 立面跌落 20.3–20.5 cm、恢复 0.38–0.40 s、
# 每道累计 −2.7…−3.1，整段四级 −10，而台阶列专项奖励只有约 +2.7/s（docs/plan/m20_narrow_off_stairs_20260921.md 末段）。
# 它把用户要的"机身先过沿、轮子随后收腿提上来"的过渡期按最高费率罚，过渡越慢罚越多，奖励天然偏向缩短过渡的
# 起跳（M18）/走梯（M17）。支撑面口径：轮子还在下一阶时参考不跳，机身前倾压紧时误差只是几厘米，轮子过沿时机身已随之升起。
# 平地上两种口径逐位相同；其余四列仍用原口径。σ、夹紧、权重 −4 不变，不加死区（单变量）。
ROUGH_BASE_HEIGHT_SUPPORT_COLUMNS: tuple[str, ...] = ROUGH_STAIR_LIKE_COLUMNS
ROUGH_BASE_HEIGHT_SUPPORT_SENSOR = "stair_reward_height"
# M8（2026-09-14 用户定）：平地列注入高姿起步转移。M7-1200 的噪声扫描（.scratch/m7_explore.py，
# 无限平面、16 env）显示这是探索瓶颈而不是定价问题：h=0.38 静止起步时确定性动作回报 232.4、0 个跑起来；
# 加训练实际噪声 σ=0.31 后只有 1/16 跑起来、采样里最好的 238.6 仍不如确定性的 261.8（优势全非正，
# 梯度为零）；σ 加倍到 0.62 有 11/16 动起来但回报塌到 −50.7（乱动不是行走，优势照样为负）。
# 同时该状态本身极罕见：平地列 0.30 × 高度>0.34 的 0.22 × 静站 0.10 × 高速 0.67 ≈ 0.44%，
# 每轮不到一个正样本。转移把命中率提到约 0.30×0.5=15%，正样本从 <1 变约 30 个/轮。
# 值取 A20/A21 验证过的 0.5（commit 5ec1fd7）：A15 在 0.38 m 静站后给 0.8/1.6/2.4 只能跑 0.05 m/s，
# 加转移后 A21 model_999 达 0.80/1.55/2.14 m/s；代价是 catastrophic 终止 0.03–0.08 → 0.12–0.15/轮。
# 高度区间、静站时长、切换后速度沿用 commands.py 的 A20 默认值 (0.36,0.38)/(1.5,2.5)s/(0.8,2.4)。
ROUGH_HIGH_STAND_TRANSITION_PROB = 0.5

# critic 特权地形观测：机身系 yaw 对齐网格，x ±0.5 m、y ±0.3 m、间距 0.1 m，11×7 = 77 条射线，
# 与 yly-true/fudan_rl_wheel_leg 的 measured_points_x/y 一致。只进 critic，actor 契约不变。
ROUGH_CRITIC_HEIGHT_SCAN_SENSOR_NAME = "critic_height_scan"
ROUGH_CRITIC_HEIGHT_SCAN_SIZE_M = (1.0, 0.6)
ROUGH_CRITIC_HEIGHT_SCAN_RESOLUTION_M = 0.1


def _to_rough_command_cfg(command_cfg, **overrides) -> RoughCommandCfg:
    """把 Flat 的 JumpCommandCfg 逐字段搬进 RoughCommandCfg，再覆盖 rough 要改的基类字段。

    逐字段搬运而不是重新构造，是为了让 Flat 基线以后改指令参数时 rough 自动跟随；
    RoughCommandCfg 自己新增的字段取它的默认值（commands.py 的模块常量）。
    """
    base = {f.name: getattr(command_cfg, f.name) for f in fields(command_cfg) if f.init}
    base.update(overrides)
    return RoughCommandCfg(**base)


def env_cfg(
    play: bool = False,
    *,
    terrain_generator: TerrainGeneratorCfg | None = None,
) -> ManagerBasedRlEnvCfg:
    """带官方地形课程与地形感知高度下限的崎岖地形环境配置。

    terrain_generator：None 时用 `rough_terrains_cfg()`；定向评测传 `stair_only_terrains_cfg()`。
    """
    cfg = flat_env_cfg(
        play=play,
        wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
        action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
    )

    cfg.scene.entities = {
        "robot": get_serialleg_closedchain_cfg(
            collision_geom_group=ROUGH_ROBOT_COLLISION_GEOM_GROUP
        )
    }
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=terrain_generator or rough_terrains_cfg(),
        max_init_terrain_level=ROUGH_MAX_INIT_TERRAIN_LEVEL,
    )
    cfg.sim.contact_sensor_maxmatch = ROUGH_CONTACT_SENSOR_MAXMATCH
    cfg.sim.njmax = ROUGH_NJMAX
    cfg.sim.nconmax = ROUGH_NCONMAX

    # 台阶专项奖励用的双轮传感器 + critic 高度扫描；Flat 原有传感器布局不动。
    cfg.scene.sensors = (
        *cfg.scene.sensors,
        TerrainHeightSensorCfg(
            name="stair_reward_height",
            frame=(
                ObjRef(type="body", name="l_wheel_Link", entity="robot"),
                ObjRef(type="body", name="r_wheel_Link", entity="robot"),
            ),
            ray_alignment="yaw",
            pattern=RingPatternCfg.single_ring(radius=0.01, num_samples=4),
            max_distance=2.0,
            include_geom_groups=(0,),
            reduction="min",
        ),
        ContactSensorCfg(
            name="stair_reward_contact",
            primary=ContactMatch(
                mode="body", pattern=r"^(l_wheel_Link|r_wheel_Link)$", entity="robot"
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force", "normal", "tangent"),
            reduce="maxforce",
            num_slots=4,
            global_frame=True,
        ),
        RayCastSensorCfg(
            name=ROUGH_CRITIC_HEIGHT_SCAN_SENSOR_NAME,
            frame=ObjRef(type="body", name="base_link", entity="robot"),
            ray_alignment="yaw",
            pattern=GridPatternCfg(
                size=ROUGH_CRITIC_HEIGHT_SCAN_SIZE_M,
                resolution=ROUGH_CRITIC_HEIGHT_SCAN_RESOLUTION_M,
            ),
            max_distance=2.0,
            include_geom_groups=(0,),
        ),
    )
    cfg.observations = dict(cfg.observations)
    critic = cfg.observations["critic"]
    critic_terms = dict(critic.terms)
    critic_terms["height_scan"] = ObservationTermCfg(
        func=height_scan,
        params={"sensor_name": ROUGH_CRITIC_HEIGHT_SCAN_SENSOR_NAME},
    )
    cfg.observations["critic"] = replace(critic, terms=critic_terms)

    cfg.commands = dict(cfg.commands)
    cfg.commands["velocity_height"] = _to_rough_command_cfg(
        cfg.commands["velocity_height"],
        terrain_aware_height=True,
        terrain_height_clearance=ROUGH_TERRAIN_HEIGHT_CLEARANCE,
        body_collision_bottom_offset=ROUGH_BODY_COLLISION_BOTTOM_OFFSET,
        terrain_step_height_type_names=ROUGH_TERRAIN_STEP_HEIGHT_TYPE_NAMES,
        terrain_command_flat_names=ROUGH_TERRAIN_COMMAND_FLAT_NAMES,
        high_stand_transition_prob=ROUGH_HIGH_STAND_TRANSITION_PROB,
    )

    cfg.events = dict(cfg.events)
    cfg.events["reset_stair_rewards"] = EventTermCfg(
        func=stair_rewards.reset_stair_rewards, mode="reset"
    )
    cfg.events["set_curriculum_env_mask"] = EventTermCfg(
        func=events.set_curriculum_env_mask,
        mode="startup",
        params={"terrain_type_names": ROUGH_CURRICULUM_SIGNAL_TERRAIN_NAMES},
    )
    cfg.events["log_reward_split"] = EventTermCfg(
        func=events.log_reward_split_by_column,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"terrain_type_names": ROUGH_REWARD_TERRAIN_TYPE_NAMES},
    )

    _apply_rough_rewards(cfg)

    cfg.terminations = dict(cfg.terminations)
    catastrophic = cfg.terminations["catastrophic_state"]
    cfg.terminations["catastrophic_state"] = replace(
        catastrophic,
        params={
            **catastrophic.params,
            "min_base_height": ROUGH_CATASTROPHIC_MIN_BASE_HEIGHT,
        },
    )
    cfg.terminations["terrain_edge_reached"] = TerminationTermCfg(
        func=terrain_edge_reached,
        params={"threshold_fraction": ROUGH_TERRAIN_EDGE_THRESHOLD_FRACTION},
        time_out=True,
    )
    cfg.terminations["out_of_terrain_bounds"] = TerminationTermCfg(
        func=out_of_terrain_bounds, time_out=True
    )

    if not play:
        cfg.curriculum = dict(cfg.curriculum)
        if "command_vel" in cfg.curriculum:
            params = dict(cfg.curriculum["command_vel"].params or {})
            params["tracking_log_key"] = ROUGH_CURRICULUM_TRACKING_LOG_KEY
            cfg.curriculum["command_vel"] = replace(cfg.curriculum["command_vel"], params=params)
        # 顺序有意义：先结算升降级，再由热身换列（见 curriculums.flat_warmup）。
        cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
            func=terrain_levels_vel,
            params={"command_name": "velocity_height"},
        )
        cfg.curriculum["flat_warmup"] = CurriculumTermCfg(
            func=curriculums.flat_warmup,
            params={
                "command_name": "velocity_height",
                "iterations": ROUGH_FLAT_WARMUP_ITERATIONS,
                "ramp_iterations": ROUGH_FLAT_WARMUP_RAMP_ITERATIONS,
                "steps_per_policy_iter": ROUGH_STEPS_PER_POLICY_ITER,
            },
        )
        # M24：二级台阶要等 stairs_up 均级到 5 才开放；必须排在 flat_warmup 之后（见 two_step_gate 文档）。
        cfg.curriculum["two_step_gate"] = CurriculumTermCfg(
            func=curriculums.two_step_gate,
            params={
                "command_name": "velocity_height",
                "gate_terrain_name": "stairs_up",
                "gate_level": ROUGH_TWO_STEP_GATE_LEVEL,
                "gated_columns": (
                    (ROUGH_TWO_STEP_UP_COLUMN, "stairs_up"),
                    (ROUGH_TWO_STEP_DOWN_COLUMN, "flat"),
                ),
            },
        )

    return cfg


def _apply_rough_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
    """加两项台阶专项奖励与全列违令罚，把三项 Flat 奖励换成按列包装（权重与未提及的核参数跟随 Flat）。"""
    cfg.rewards = dict(cfg.rewards)
    cfg.rewards["stair_climb_progress"] = RewardTermCfg(
        func=stair_rewards.stair_climb_progress,
        weight=ROUGH_STAIR_CLIMB_PROGRESS_WEIGHT,
        params={"terrain_type_names": ROUGH_REWARD_TERRAIN_TYPE_NAMES},
    )
    cfg.rewards["stair_support_height"] = RewardTermCfg(
        func=stair_rewards.stair_support_height,
        weight=ROUGH_STAIR_SUPPORT_HEIGHT_WEIGHT,
        params={"terrain_type_names": ROUGH_REWARD_TERRAIN_TYPE_NAMES},
    )
    cfg.rewards["command_velocity_error"] = RewardTermCfg(
        func=rewards.command_velocity_error_on_terrain,
        weight=float(ROUGH_COMMAND_VELOCITY_ERROR_WEIGHT),
        params={
            "command_name": "velocity_height",
            "terrain_type_names": ROUGH_ALL_TERRAIN_TYPE_NAMES,
            "lin_vel_scale": ROUGH_COMMAND_VELOCITY_ERROR_LIN_SCALE,
            "yaw_vel_scale": 1.0,
            "lin_deadband": float(FLAT_CMD_VEL_DEADBAND[0]),
            "yaw_deadband": float(FLAT_CMD_VEL_DEADBAND[1]),
            "max_penalty": 9.0,
        },
    )
    # M21：上台阶列高度参考改为轮子支撑面，其余列与 Flat 原函数逐位相同（σ 取 A15 的 0.10）。
    height = cfg.rewards["flat_base_height"]
    cfg.rewards["flat_base_height"] = replace(
        height,
        func=rewards.base_height_penalty_support_on_terrain,
        params={
            **height.params,
            "sigma": ROUGH_BASE_HEIGHT_SIGMA,
            "support_sensor_name": ROUGH_BASE_HEIGHT_SUPPORT_SENSOR,
            "terrain_type_names": ROUGH_BASE_HEIGHT_SUPPORT_COLUMNS,
        },
    )
    ang = cfg.rewards["tracking_ang_vel"]
    cfg.rewards["tracking_ang_vel"] = replace(
        ang,
        func=rewards.tracking_ang_vel_off_terrain,
        params={**ang.params, "terrain_type_names": ROUGH_REWARD_TERRAIN_TYPE_NAMES},
    )
    track = cfg.rewards["tracking_lin_vel"]
    cfg.rewards["tracking_lin_vel"] = replace(
        track,
        func=rewards.tracking_lin_vel_terrain_vz,
        weight=ROUGH_TRACKING_LIN_VEL_WEIGHT,
        params={
            **track.params,
            "sigma_move": ROUGH_OFF_STAIR_TRACKING_SIGMA_MOVE,
            "vz_weight": ROUGH_FLAT_VZ_WEIGHT,
            "terrain_vz_weight": ROUGH_TERRAIN_VZ_WEIGHT,
            "flat_type_names": ROUGH_VZ_FLAT_TERRAIN_TYPE_NAMES,
            "stair_sigma_move": ROUGH_STAIR_TRACKING_SIGMA_MOVE,
            "stair_type_names": ROUGH_REWARD_TERRAIN_TYPE_NAMES,
        },
    )
    # M20 把窄核在台阶列置零，M22 改成台阶列 w=1（column_scaled 缩放），其余四列仍是 w=3、逐位不变。
    cfg.rewards["tracking_lin_vel_narrow"] = RewardTermCfg(
        func=rewards.column_scaled,
        weight=ROUGH_TRACKING_LIN_VEL_NARROW_WEIGHT,
        params={
            "inner": rewards.tracking_lin_vel_narrow,
            "params": {
                "command_name": "velocity_height",
                "sigma": ROUGH_TRACKING_LIN_VEL_NARROW_SIGMA,
            },
            "terrain_type_names": ROUGH_TRACKING_LIN_VEL_NARROW_STAIR_COLUMNS,
            "scale": ROUGH_TRACKING_LIN_VEL_NARROW_STAIR_WEIGHT / ROUGH_TRACKING_LIN_VEL_NARROW_WEIGHT,
        },
    )
    cfg.rewards["wheel_fore_aft_offset"] = RewardTermCfg(
        func=rewards.wheel_fore_aft_offset,
        weight=-ROUGH_WHEEL_OFFSET_WEIGHT,
        params={
            "apply_type_names": ROUGH_WHEEL_OFFSET_COLUMNS,
            "dead_zone_m": ROUGH_WHEEL_OFFSET_DEAD_ZONE_M,
        },
    )
    # M23：上台阶列的左右轮高度差罚（走梯罚），与上面的平地前后错位罚是两件事。
    cfg.rewards["wheel_height_diff"] = RewardTermCfg(
        func=rewards.wheel_height_diff,
        weight=-ROUGH_WHEEL_HEIGHT_DIFF_WEIGHT,
        params={
            "apply_type_names": ROUGH_WHEEL_HEIGHT_DIFF_COLUMNS,
            "dead_zone_m": ROUGH_WHEEL_HEIGHT_DIFF_DEAD_ZONE_M,
        },
    )
    # M2：台阶列不发工资、不罚爬升动作；权重与原参数逐位沿用 Flat，只在台阶列乘零。
    for name in ROUGH_STAIRS_ZEROED_REWARDS:
        term = cfg.rewards[name]
        cfg.rewards[name] = RewardTermCfg(
            func=rewards.off_column,
            weight=float(term.weight),
            params={
                "inner": term.func,
                "params": dict(term.params),
                "terrain_type_names": ROUGH_CONTACT_TAX_FREE_COLUMNS,
            },
        )
    # 摔倒罚：非超时终止那一步一次性扣 ROUGH_FALL_PENALTY（权重按 dt 反缩放）。
    step_dt = float(cfg.sim.mujoco.timestep) * int(cfg.decimation)
    cfg.rewards["fall_penalty"] = RewardTermCfg(
        func=is_terminated, weight=-ROUGH_FALL_PENALTY / step_dt
    )


__all__ = [
    "ROUGH_ALL_TERRAIN_TYPE_NAMES",
    "ROUGH_BASE_HEIGHT_SIGMA",
    "ROUGH_BASE_HEIGHT_SUPPORT_COLUMNS",
    "ROUGH_BASE_HEIGHT_SUPPORT_SENSOR",
    "ROUGH_BODY_COLLISION_BOTTOM_OFFSET",
    "ROUGH_CATASTROPHIC_MIN_BASE_HEIGHT",
    "ROUGH_COMMAND_VELOCITY_ERROR_LIN_SCALE",
    "ROUGH_COMMAND_VELOCITY_ERROR_WEIGHT",
    "ROUGH_CONTACT_SENSOR_MAXMATCH",
    "ROUGH_CONTACT_TAX_FREE_COLUMNS",
    "ROUGH_CRITIC_HEIGHT_SCAN_RESOLUTION_M",
    "ROUGH_CRITIC_HEIGHT_SCAN_SENSOR_NAME",
    "ROUGH_CRITIC_HEIGHT_SCAN_SIZE_M",
    "ROUGH_CURRICULUM_SIGNAL_TERRAIN_NAMES",
    "ROUGH_CURRICULUM_TRACKING_LOG_KEY",
    "ROUGH_FALL_PENALTY",
    "ROUGH_FLAT_VZ_WEIGHT",
    "ROUGH_FLAT_WARMUP_ITERATIONS",
    "ROUGH_FLAT_WARMUP_RAMP_ITERATIONS",
    "ROUGH_HIGH_STAND_TRANSITION_PROB",
    "ROUGH_MAX_INIT_TERRAIN_LEVEL",
    "ROUGH_NCONMAX",
    "ROUGH_NJMAX",
    "ROUGH_OFF_STAIR_TRACKING_SIGMA_MOVE",
    "ROUGH_REWARD_TERRAIN_TYPE_NAMES",
    "ROUGH_ROBOT_COLLISION_GEOM_GROUP",
    "ROUGH_STAIRS_ZEROED_REWARDS",
    "ROUGH_STAIR_ANG_VEL_YAW_RANGE",
    "ROUGH_STAIR_CLIMB_PROGRESS_WEIGHT",
    "ROUGH_STAIR_COMMAND_TERRAIN_NAMES",
    "ROUGH_STAIR_HEIGHT_RANGE",
    "ROUGH_STAIR_LIN_VEL_X_RANGE",
    "ROUGH_STAIR_SUPPORT_HEIGHT_WEIGHT",
    "ROUGH_STAIR_TRACKING_SIGMA_MOVE",
    "ROUGH_STEPS_PER_POLICY_ITER",
    "ROUGH_TERRAIN_ANG_VEL_YAW_RANGE",
    "ROUGH_TERRAIN_COMMAND_FLAT_NAMES",
    "ROUGH_TERRAIN_EDGE_THRESHOLD_FRACTION",
    "ROUGH_TERRAIN_HEIGHT_CLEARANCE",
    "ROUGH_TERRAIN_LIN_VEL_X_RANGE",
    "ROUGH_TERRAIN_STEP_HEIGHT_TYPE_NAMES",
    "ROUGH_TERRAIN_VZ_WEIGHT",
    "ROUGH_TRACKING_LIN_VEL_NARROW_STAIR_COLUMNS",
    "ROUGH_TRACKING_LIN_VEL_NARROW_STAIR_WEIGHT",
    "ROUGH_TRACKING_LIN_VEL_WEIGHT",
    "ROUGH_VZ_FLAT_TERRAIN_TYPE_NAMES",
    "ROUGH_WHEEL_HEIGHT_DIFF_COLUMNS",
    "ROUGH_WHEEL_HEIGHT_DIFF_DEAD_ZONE_M",
    "ROUGH_WHEEL_HEIGHT_DIFF_WEIGHT",
    "ROUGH_WHEEL_OFFSET_COLUMNS",
    "ROUGH_WHEEL_OFFSET_DEAD_ZONE_M",
    "ROUGH_WHEEL_OFFSET_WEIGHT",
    "env_cfg",
]
