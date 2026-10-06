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
M18：左右轮前后错位罚 w·Δx²，台阶列置零；M19：改成只罚平地列 + 10 cm 死区（M37 起删除，见下）。
M20：窄核速度跟踪在上台阶列置零；M22：台阶列恢复到 w=1（见 ROUGH_TRACKING_LIN_VEL_NARROW_STAIR_WEIGHT 注释）。
M23：上台阶列新增左右轮高度差罚，堵死走梯（M37 起删除，见下）。
M24：新增二级台阶上行/下行两列（复旦那道凸棱），上行列按 stairs_up 待遇、下行列按平地待遇；
八处台阶开关统一引用 terrains.ROUGH_STAIR_LIKE_COLUMNS；两列由 curriculums.two_step_gate 门控，
stairs_up 均级到 5 才开放（见 ROUGH_TWO_STEP_GATE_LEVEL）。
M21：上台阶列机身高度罚的地面参考改为轮子支撑面（见 ROUGH_BASE_HEIGHT_SUPPORT_COLUMNS 注释）。
M27/M34：上台阶列高度参考改为复旦窗口口径 + 有界罚，两项轮几何罚死区归零（见 ROUGH_STAIR_HEIGHT_REFERENCE 注释）。
M35：窗口高度罚加 ±5 cm 死区（见 ROUGH_STAIR_HEIGHT_DEAD_ZONE_M 注释）。
M37：删五项重叠定价——command_velocity_error、tracking_lin_yaw_joint、bad_tilt、wheel_fore_aft_offset、
wheel_height_diff（见 ROUGH_DROPPED_FLAT_REWARDS 注释，docs/plan/m37_reward_prune_20260927.md）。
M38：加全局向上奖励 upward 权重 1.0（见 ROUGH_UPWARD_WEIGHT 注释）。2026-09-28 用户定 M38 为默认，临时入口全部删除，
复现各对照用对应 commit（M34 1a10b73、M35 5af0d00、M37 9e07b44、M38 25ca875）。
M39：action_rate −0.48 → −0.10（见 ROUGH_ACTION_RATE_WEIGHT 注释）。M47：删 bad_orientation 终止。
M49：删 joint_mirror，加左右轮心前后错位罚 −50（见 ROUGH_WHEEL_FORE_AFT_WEIGHT 注释）。
M50：台阶列 yaw 指令恒 0、出生朝向正对台阶、yaw 跟踪恢复 Flat 原式（见 ROUGH_STAIR_SPAWN_YAW_HALF_RANGE_DEG 注释）。
M51：加随机粗糙、波浪、离散矮障碍三列（terrains.ROUGH_RANDOM_TERRAIN_PROPORTIONS）。
M53：膝气弹簧电机侧前馈补偿 + 腿部 T-N 包络 ×0.8（见 ROUGH_LEG_TORQUE_ENVELOPE_SCALE 注释）。
M54：台阶列航向保持 −6、轮前后错位罚 ×0.25、出生朝向 ±15°（见 ROUGH_STAIR_HEADING_HOLD_WEIGHT 注释）。
RJ1：合入 J10 跳跃（见 _apply_jump_mimic）；2026-10-03 起跳跃样本为专用平地跳跃列（占全部 env 10%）、触发 0.5 Hz。2026-10-02 用户定 RJ1 为默认，M39–M54 与 RJ1 的
临时入口全部删除；M40–M46、M48、M52 的对照未采用。复现各对照用对应 commit（M39 a724b49、M47 4398698、M49 d5db384、
M50 be3498f、M51 db1805a、M53 fd4fd76、M54 01e87e2、RJ1 e3b58ab）。

DR1（2026-10-05 用户定）：rough 默认域随机化换成真 Kp/Kd、质心 ±5 cm、质量 −1…+3 kg、气弹簧 ×0.9–1.5、延迟 0–10 ms，
不做恢复系数（见 ROUGH_DR_* 注释，docs/plan/rough_wide_dr_20261003.md）。

机器人实体与 Flat 同一个 MJCF，只把碰撞 geom 从 group 0 改到 group 3（内存里改，不动文件），
让 `include_geom_groups=(0,)` 的高度射线只看地形，不再打到自己的腿和轮子。
"""

from __future__ import annotations

import math
from dataclasses import fields, replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import height_scan
from mjlab.envs.mdp.rewards import is_terminated
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
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

from se3_shared import M3508_C620_14
from se3_train.mdp import events as mdp_events
from se3_train.mdp import observations as mdp_observations
from se3_train.mdp import rewards as mdp_rewards
from se3_train.robot_cfg import get_serialleg_closedchain_cfg, serialleg_wheel_half_track
from se3_train.tasks.common.no_attitude import apply_no_attitude_layout
from se3_train.tasks.flat.env_cfg import (
    FLAT_ACTION_SMOOTHNESS_SPRING,
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
    ROUGH_STAIR_SPEED_CAP_ENABLED,
    ROUGH_TERRAIN_ANG_VEL_YAW_RANGE,
    ROUGH_TERRAIN_COMMAND_FLAT_NAMES,
    ROUGH_TERRAIN_HEIGHT_CLEARANCE,
    ROUGH_TERRAIN_LIN_VEL_X_RANGE,
    ROUGH_TERRAIN_STEP_HEIGHT_TYPE_NAMES,
    RoughCommandCfg,
    RoughJumpCommandCfg,
)
from .terrains import (
    ROUGH_RANDOM_TERRAIN_PROPORTIONS,
    ROUGH_STAIR_LIKE_COLUMNS,
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
# 训练地形的全部列，与 terrains.ROUGH_RANDOM_TERRAIN_PROPORTIONS 的键同源（M51 起十列）。
ROUGH_ALL_TERRAIN_TYPE_NAMES = tuple(ROUGH_RANDOM_TERRAIN_PROPORTIONS)

# 台阶专项奖励（A12；消融 A7/A8：撤掉任一项台阶列速度归零）。
ROUGH_STAIR_CLIMB_PROGRESS_WEIGHT = 3.0
ROUGH_STAIR_SUPPORT_HEIGHT_WEIGHT = 4.0
# 速度违令二次罚（A6 起台阶列，A15 起全六列，lin_vel_scale 3.0）M37 起删除：3.0 下 1.0 m/s 误差只罚 0.20、
# 斜率不到 1，是常数税不是远端梯度（docs/plan/m37_reward_prune_20260927.md）。
# A15 非台阶列定价。A13b flat 列账本（96 env，指令 vx 2.0 / h 0.38）：站着不动净 +2.407/s、走路净
# −2.932/s，走路必然产生的机身起伏被高度罚、核里的 vz 项、姿态罚罚了三遍。σ 0.05→0.10 收回 +3.11/s，
# 运动核 0.08→0.5 与 vz 2.0→0 合计收回 +2.97/s；违令罚扩到全列只打在"不动"那边。
# 2026-10-05（用户定）σ 0.10 → 0.07（全列，上台阶列窗口口径同样生效）。依据：加速时机身冲高 6–9 cm，反事实账本显示
# 贴着指令高度加速总奖励更高、加速不慢，但差额只占窗口 3–6%（策略没学好，不是被奖励鼓励）。同代码 4+3 卡对照
# HeightSigma07 `jrvpex39` 对基线 `h7tyzjql`（f090dbe，均含 DR1，总 env 4680），sim2x 回放：0.30 m 加速峰值高出指令
# 87–90 → 24–45 mm，0.34 m 低速稳态 +50 → +5…+21 mm；0.38 m 静站后起步到 90% 更快、倾角 11–13° → 4–6°。
# 代价：0.22/0.26 m 稳态偏低 1–2.4 cm；0.38 m 静站后 vx 0.8 稳态超速到 1.13；下台阶摔倒 4 → 9（r9 低姿态 1.2–2.0 m/s），
# 嫌疑是下行列单点射线过沿跳变被更陡的罚放大（未验证）。平地 vz 项与独立 vz 罚分不开冲高与贴高，未采用。
ROUGH_BASE_HEIGHT_SIGMA = 0.07
# 显式 vx 观测器（2026-10-05 用户定，见 se3_train.vx_observer）：actor 历史帧数与监督目标组名。
# 16 帧 = 0.32 s（50 Hz），用户定；估计器 480→128→64→1 约 7 万次乘加，MCU 上可推。
ROUGH_VX_OBSERVER_HISTORY_LENGTH = 16
ROUGH_VX_OBSERVER_TARGET_GROUP = "estimator_target"
# 轮子 T-N 包络（2026-10-06 用户定，见 robot_cfg.get_serialleg_closedchain_cfg 的 wheel_torque_envelope）：
# 默认仍是旧口径 "linear_peak"（45 rad/s 只剩 1.14 N·m）；"rated_point" 按手册额定点（469 rpm@19:1 出 3 N·m）建平台。
ROUGH_WHEEL_TORQUE_ENVELOPE = "linear_peak"
# 指令可行域的差速轮速预算（2026-10-06 用户定）："legacy" = Flat 继承的 45 rad/s、半轮距 0.20 m；
# "rated" = M3508 手册额定转速 66.65 rad/s（14:1）、MJCF 实测半轮距 0.2166 m，使用比例都保持 0.9。
# 旧口径下原地 12 rad/s 实际要 43.3 rad/s 轮速（45 的 96%，超出 0.9 预算），vx = 1 时 yaw 只能采到 7.15。
ROUGH_COMMAND_WHEEL_BUDGET = "legacy"
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
# M18/M19（左右轮前后错位罚 Δx²，平地列、10 cm 死区）与 M23（上台阶列左右轮高度差罚 Δz²、8 cm 死区）
# 于 M37 删除：用户决定对称只保留 joint_mirror 一条关节空间定价；M34 曾把两项死区归零。
# 历史与定价推导见 docs/plan/m18_wheel_offset_20260921.md、m23_wheel_dz_20260921.md、m37_reward_prune_20260927.md。
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
# 上台阶列高度罚的地面参考口径（2026-09-25 用户定做对照）：
#   "support" = M21 两轮支撑面 + 夹 ±0.15 m 的二次罚（现行默认）；
#   "window"  = 复旦口径：机身周围 77 点窗口均值（复用 critic 高度扫描）+ 有界罚 1 − exp(−e²/σ²)，
#               见 rewards.base_height_penalty_window_on_terrain。σ、权重 −4、生效列与 support 相同。
ROUGH_STAIR_HEIGHT_REFERENCE = "window"
# M35（2026-09-26 用户定）：窗口口径高度罚在上台阶列的死区（m），只在 "window" 口径下有意义；0 = 原样。
# 依据 M34-7800 反事实账本（docs/plan/m35_m36_stair_speed_20260926.md）：窗口高度罚是唯一随爬升速度明显上涨的罚项，
# 误差以过沿时机身偏低为主，±5 cm 死区去掉其 85%，动作罚（action_rate / action_smoothness）在 0.73–0.89 m/s 间不变。
ROUGH_STAIR_HEIGHT_DEAD_ZONE_M = 0.05
# M37（2026-09-27 用户定）：在 M35 基线上整组删掉五项重叠定价，其余逐位不变。
# 依据 docs/plan/m37_reward_prune_20260927.md 的形状分析：
#   command_velocity_error  lin_vel_scale 3.0 下 1.0 m/s 误差只罚 0.20、斜率不到 1，是常数税不是远端梯度（台阶列实付 −0.68/s）；
#   tracking_lin_yaw_joint  只在 |vx|≥0.2 且 |yaw|≥0.5 时开，台阶列 yaw 指令 ±0.3 永远开不了，与 tracking_ang_vel 重复计酬；
#   bad_tilt                15° 以上与 tracking_orientation_l2 是同一量的两条曲线，30° 以上已由 bad_orientation 终止接手；
#   wheel_fore_aft_offset / wheel_height_diff  用户决定对称只保留 joint_mirror 一条关节空间定价。
# M38（2026-09-28 用户定）：M37 + 全局向上奖励 `mdp.rewards.upward`（(1 − pg_z)²，直立 4、侧躺 1、倒置 0）权重 1.0。
# 形状：0–20° 内几乎是 +4/s 常数工资（18° 只少 0.2/s，是 orientation_l2 同角度的 1/6），45° 以上才有量；
# 它不受列掩码，等于给台阶列重新发 4 倍 is_alive 的工资（M2 曾把 is_alive 在台阶列置零），判据里要盯台阶列低速冻住。
# None = 不加（M37 及之前）。M38-4999 回放：撞面俯仰从 M37 的 −16…−18° 回到 −2…−10°，h=0.38 低速格子从卡变过；
# 代价是台阶列 vx 0.79–0.86 → 0.72–0.75、落差后摔倒变多（docs/plan/m38_upward_20260928.md）。2026-09-28 起默认 1.0。
ROUGH_UPWARD_WEIGHT: float | None = 1.0
# 2026-10-04 起 tracking_orientation_l2（pitch/roll L2）−12 → −24（用户定）。同代码同配置（7 卡 × 1170、actor 128）对照：
# Base128 `8kjb4vm4`（−12）对 Orient24 `3ix86na3`（−24），sim2x 地形网格 10 种地形 × r3/6/9 平均倾角 8.7° → 4.4°、
# p95 20.8° → 11.8°、通过率 93% → 100%，平地 vx 误差 0.213 → 0.082、抗推最大倾角 21.3° → 15.6°；
# 代价是起步到 90% 速度 0.61 → 0.80 s。跳跃落地变差与中等高度跑动站高两条 run 都有，与本项无关
# （docs/plan/rough_orient24_20261004.md）。
ROUGH_ORIENTATION_WEIGHT: float = -24.0
# 从 Flat 继承后整项删除的三项（另三项 rough 自己不再构造）：
#   tracking_lin_yaw_joint  只在 |vx|≥0.2 且 |yaw|≥0.5 时开，台阶列 yaw 指令恒 0 永远开不了，与 tracking_ang_vel 重复计酬；
#   bad_tilt                15° 以上与 tracking_orientation_l2 是同一量的两条曲线；
#   joint_mirror            M49 起对称只罚水平分量，改由 wheel_fore_aft_offset 计价（见 ROUGH_WHEEL_FORE_AFT_WEIGHT）。
ROUGH_DROPPED_FLAT_REWARDS: tuple[str, ...] = ("tracking_lin_yaw_joint", "bad_tilt", "joint_mirror")
# M39（2026-09-28 用户定）：action_rate 权重 −0.48 → −0.10（Flat 是 −0.48）。M34-7800 账本里 action_rate 在 0.73–0.89 m/s
# 间不随速度变（−0.55…−0.58/s），M38 台阶列实付 −0.85/s，是台阶列第二大罚项（docs/plan/m39_action_rate_20260928.md）。
ROUGH_ACTION_RATE_WEIGHT = -0.10
# M47（2026-09-29 用户定）：删 bad_orientation 终止（倾角 > 30° 连续 100 步即终止并吃 fall_penalty）。
# 失败终止只剩数值发散的 catastrophic_state，倒地 env 躺到超时，由姿态 / 触地 / upward 等奖励项计价
# （docs/plan/m47_no_bad_orientation_20260929.md）。
# M49（2026-09-29 用户定）：对称只罚水平分量——删 joint_mirror（M48 曾把它从 −0.179 加到 −5，左右差 22° 时每秒只扣
# 0.03 → 0.76），加 wheel_fore_aft_offset（左右轮心机身系前后错位 Δx²，全列、无死区、直立门控）。竖直分量（左右腿长差）
# 留给 roll 指令，平地侧倾由 tracking_orientation_l2 计价。w −50/m²：Δx 5/10/20 cm 每秒 0.13/0.5/2.0
# （docs/plan/m48_joint_mirror_20260929.md、m49_wheel_fore_aft_20260929.md）。
ROUGH_WHEEL_FORE_AFT_WEIGHT = -50.0
# M50（2026-09-29 用户定）：台阶列（stairs_up、stairs_two_step_up）不再采样 yaw 角速度指令（commands.ROUGH_STAIR_ANG_VEL_YAW_RANGE
# = (0, 0)），转向多样性改由初始朝向提供：初始朝向从全向均匀收窄为「正对某一面台阶 ± 半宽」（events.stair_facing_yaw）；
# 台阶列 yaw 角速度跟踪恢复 Flat 原式（M44 的做法），指令恒 0 时它就是台阶列的 yaw 角速度罚（docs/plan/m50_stair_yaw_20260929.md）。
# M54 把半宽从 ±30° 收到 ±15°：腿长行程 19.6 cm < 最高台阶 20 cm，斜 30° 不转正只能大侧倾硬爬。
ROUGH_STAIR_SPAWN_YAW_HALF_RANGE_DEG = 15.0
# M51（2026-09-29 用户定）：地形集加随机粗糙、波浪、离散矮障碍三列（terrains.ROUGH_RANDOM_TERRAIN_*），
# 新列 MDP 与 flat 一致：按平地发指令（commands.ROUGH_TERRAIN_COMMAND_FLAT_NAMES 已含三列名）、平地同一套奖励
# （不进台阶分列定价与接触税置零名单）、高度参考不变（非台阶列仍是机身正下方单点射线）。速度课程信号仍只读 flat 列。
# M53（2026-09-30 用户定）：plant 保留 300 N 膝气弹簧，电机侧打开前馈补偿 −F·dL/dα，F 按额定 300 N 给
# （se3_shared.RobotConfig.knee_gas_spring_force）；弹簧力 DR 照旧左右独立 ×0.9–1.1（270–330 N），额定之外的残差
# 交给策略与 PD。前馈在 PD 之后、T-N 限幅之前叠加（SerialLegDelayedAction），与真机执行方式一致；ONNX 契约随动作项
# 导出 {force: 300, compensation_enabled: true}，sim2x / 真机按契约做同一前馈。M52（训练 plant 去掉弹簧）的对照未采用。
# M53 第二处改动：腿部电机 T-N 包络按物理含义取参并整体 ×0.8——saturation_effort = 0.8·电压限零速截距
# （DM8009P V1.0 @24V 约 132 → 106）、effort_limit = 0.8·峰值 40 = 32，即 32 N·m 平台到约 11.7 rad/s 再沿反电动势下降、
# 16.76 rad/s 过零（旧口径恒扭矩区被额定 20 削平，100 rpm 下只有 15 N·m，低于达妙官方实测的至少 25 N·m）。
# 用户确认真机至少 30、35 肯定可给。见 robot_cfg.get_serialleg_closedchain_cfg（docs/plan/m53_knee_spring_feedforward_20260930.md）。
ROUGH_LEG_TORQUE_ENVELOPE_SCALE = 0.8
# M54（2026-09-30 用户定）：斜向上台阶不靠被动扭转转正。M53 记账（.scratch/m53_yaw_ledger/）显示先触立面的轮子
# 被挡住、另一侧继续走，机身被动转正（出生偏 30° 转 −24°，强制左右轮同速转 −32°），原定价只按瞬时角速度收 0.6、转完不计价；
# 而单轮先上会被轮前后错位罚持续扣分。两处改动，均只作用于台阶列（stairs_up、stairs_two_step_up）：
#   1. stair_heading_hold：航向误差平方（rad²）×(−6)，目标 = 出生朝向 + ∫yaw 指令；偏 15°/30° 每秒 −0.41/−1.64。
#   2. wheel_fore_aft_offset 在台阶列 ×0.25（Δx 20 cm 每秒 −2.0 → −0.5），给单轮先上留余地；其余列 −50 不变。
# 见 docs/plan/m54_stair_heading_hold_20260930.md。
ROUGH_STAIR_HEADING_HOLD_WEIGHT = -6.0
ROUGH_STAIR_WHEEL_FORE_AFT_SCALE = 0.25
# M8（2026-09-14 用户定）：平地列注入高姿起步转移。M7-1200 的噪声扫描（.scratch/m7_explore.py，
# 无限平面、16 env）显示这是探索瓶颈而不是定价问题：h=0.38 静止起步时确定性动作回报 232.4、0 个跑起来；
# 加训练实际噪声 σ=0.31 后只有 1/16 跑起来、采样里最好的 238.6 仍不如确定性的 261.8（优势全非正，
# 梯度为零）；σ 加倍到 0.62 有 11/16 动起来但回报塌到 −50.7（乱动不是行走，优势照样为负）。
# 同时该状态本身极罕见：平地列 0.30 × 高度>0.34 的 0.22 × 静站 0.10 × 高速 0.67 ≈ 0.44%，
# 每轮不到一个正样本。转移把命中率提到约 0.30×0.5=15%，正样本从 <1 变约 30 个/轮。
# 值取 A20/A21 验证过的 0.5（commit 5ec1fd7）：A15 在 0.38 m 静站后给 0.8/1.6/2.4 只能跑 0.05 m/s，
# 加转移后 A21 model_999 达 0.80/1.55/2.14 m/s；代价是 catastrophic 终止 0.03–0.08 → 0.12–0.15/轮。
# 高度区间、静站时长、切换后速度沿用 commands.py 的 A20 默认值 (0.36,0.38)/(1.5,2.5)s/(0.8,2.4)。
# 2026-10-04 起 0.5 → 0.1（用户定）。0.5 时任意时刻约 26% env 处在高姿态序列中（0.1 时 5.4%）。同代码对照
# Orient24 `3ix86na3`（0.5）对 HighStand01 `0yeh4dow`（0.1），sim2x 回放：跳跃落地后 |俯仰| 21–22° → 1.6–9.3°、
# 下沉 40–47 → 7–28 mm、0.5 s 内稳定；平地静站固定俯仰 −5.6/+7.5/−3.3° → −0.8/−0.6/−1.6°（0.22/0.30/0.38 m）；
# 平地 vx 误差 0.082 → 0.058；中等高度跑动站高减 1–2.5 cm（未根除）。M8 的高姿态起步能力保留（0.38 m 静站后
# 0.8/1.6/2.4 均能起步，稳态 0.84/1.53/2.12 m/s、到 90% 稍慢）；代价：wave r6/r9 0.6 m/s 卡住、两级下台阶 r9 摔倒、
# 跳跃离地平均低 1.6 cm（docs/plan/rough_highstand01_20261004.md）。
ROUGH_HIGH_STAND_TRANSITION_PROB = 0.1

# DR1（2026-10-05 用户定，原 Exp-WideDR-NoRest）：rough 默认域随机化在 Flat 继承的 DR 上改五项，其余（摩擦、惯量、电机被动参数、
# 默认关节位置、推力课程、观测噪声）不动。起因是 DR 审计（.scratch/dr_audit/verify_dr.py）：Flat 继承的 pd_gains 实际是 6 电机
# 输出力矩 ×0.9–1.1、kd 无效，质量、质心、延迟都比复旦 stairs_v3（同尺寸轮腿）窄。
#   Kp/Kd   torch 侧真正的增益 DR（events.randomize_pd_gains_torch），×0.9–1.1；原输出力矩缩放随之去掉
#   质心    ±5 mm → ±5 cm
#   质量    base 附加 −0.5…+1.5 → −1…+3 kg
#   气弹簧   ×0.9–1.1 → ×0.9–1.5（前馈补偿仍按额定 300 N，残差最大 +150 N 交给策略）
#   动作延迟 4–6 ms（按 5 ms 物理步取整恒为 1 步）→ 0–10 ms（0/1/2 步各 1/3）；play 与 ONNX 契约同步，
#           真机 runtime 默认把延迟覆盖成 0（se3_runtime_nx --action-delay-steps），不受影响
# 恢复系数不做：MuJoCo 软接触的阻尼对持续接触一直生效，压低阻尼得到的是轮地接触持续欠阻尼振荡，而不是 PhysX 那种只在撞击时
# 生效的 restitution。离线归因显示 e > 0.7 回报断崖（0.9–1.0 −76%），是 WideDR 全面退化的主因。
# 评测（docs/plan/rough_wide_dr_20261003.md）：扰动 plant 11 种 × 6 项 0 摔（旧默认 5 摔），质心 +5 cm 下 vx 误差 0.14（旧 0.56）；
# 名义平地 vx 误差约为旧默认的 2 倍。注意 NoRest（11f9d40）相对 Orient24 还混入了 3919db7 / b50ef1f / 6656fb5 三处默认改动，
# DR1 对名义性能的单独代价尚无干净对照。复现各对照：WideDR 80019b8、Oracle 1fa00ca、Rest05 17c91ba、NoRest 11f9d40。
ROUGH_DR_KP_RANGE = (0.9, 1.1)
ROUGH_DR_KD_RANGE = (0.9, 1.1)
ROUGH_DR_COM_RANGE_M = 0.05
ROUGH_DR_BASE_MASS_RANGE_KG = (-1.0, 3.0)
ROUGH_DR_KNEE_SPRING_SCALE_RANGE = (0.9, 1.5)
ROUGH_DR_ACTION_DELAY_RANGE_S = (0.0, 0.010)

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
    stair_speed_cap: bool = ROUGH_STAIR_SPEED_CAP_ENABLED,
    stair_height_reference: str = ROUGH_STAIR_HEIGHT_REFERENCE,
    stair_height_dead_zone_m: float = ROUGH_STAIR_HEIGHT_DEAD_ZONE_M,
    upward_weight: float | None = ROUGH_UPWARD_WEIGHT,
    orientation_weight: float | None = ROUGH_ORIENTATION_WEIGHT,
    high_stand_transition_prob: float = ROUGH_HIGH_STAND_TRANSITION_PROB,
    base_height_sigma: float = ROUGH_BASE_HEIGHT_SIGMA,
    oracle_dr_obs: bool = False,
    oracle_base_vel: bool = False,
    oracle_base_vx: bool = False,
    vx_observer: bool = False,
    vx_observer_history_length: int = ROUGH_VX_OBSERVER_HISTORY_LENGTH,
    wheel_torque_envelope: str = ROUGH_WHEEL_TORQUE_ENVELOPE,
    command_wheel_budget: str = ROUGH_COMMAND_WHEEL_BUDGET,
) -> ManagerBasedRlEnvCfg:
    """带官方地形课程与地形感知高度下限的崎岖地形环境配置。

    terrain_generator：None 时用 `rough_terrains_cfg()`；定向评测传 `stair_only_terrains_cfg()`。
    stair_speed_cap / stair_height_reference / stair_height_dead_zone_m / upward_weight：对照实验开关，
    默认取模块常量（window 口径 + 5 cm 死区 + upward 1.0，见各常量注释）。其余定价与执行链固定为 RJ1（见模块 docstring）。
    orientation_weight：tracking_orientation_l2（pitch/roll L2）权重，默认 ROUGH_ORIENTATION_WEIGHT（−24），
    None 沿用 Flat 的 −12。
    high_stand_transition_prob：平地列"高姿态静站 → 前进"序列的生成概率，默认 ROUGH_HIGH_STAND_TRANSITION_PROB；对照实验开关。
    base_height_sigma：机身高度罚 flat_base_height 的 σ（全列共用，上台阶列的窗口口径同样用它），默认 ROUGH_BASE_HEIGHT_SIGMA；对照实验开关。
    oracle_dr_obs：actor 额外观测真实 DR 参数（见 _apply_oracle_dr_obs），只做诊断、不可部署。
    oracle_base_vel：actor 额外观测机身系线速度 3 维（critic 同源的 base_lin_vel），只做诊断、不可部署。
    oracle_base_vx：actor 额外观测机身系前向速度 vx 1 维，只做诊断、不可部署。
    vx_observer：actor 观测改为 ROUGH_VX_OBSERVER_HISTORY_LENGTH 帧历史，并增加估计器监督目标组
    ROUGH_VX_OBSERVER_TARGET_GROUP（真实 vx，不进 actor / critic）；须配 rl_cfg.vx_observer_rl_cfg。可部署。
    vx_observer_history_length：vx_observer 的 actor 历史帧数，默认 ROUGH_VX_OBSERVER_HISTORY_LENGTH；对照实验开关。
    wheel_torque_envelope：轮子 T-N 包络，"linear_peak" | "rated_point"（见 ROUGH_WHEEL_TORQUE_ENVELOPE）。
    command_wheel_budget：指令差速轮速预算，"legacy" | "rated"（见 ROUGH_COMMAND_WHEEL_BUDGET）。
    """
    if stair_height_reference not in ("support", "window"):
        raise ValueError(
            f"stair_height_reference 只能是 'support' 或 'window'，实际为 {stair_height_reference!r}"
        )
    # stair_height_dead_zone_m 只进 "window" 口径的参数表；选 "support" 时忽略（M38 起默认死区 0.05，选回 support 不该报错）。
    cfg = flat_env_cfg(
        play=play,
        wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
        action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
    )

    cfg.scene.entities = {
        "robot": get_serialleg_closedchain_cfg(
            collision_geom_group=ROUGH_ROBOT_COLLISION_GEOM_GROUP,
            leg_torque_envelope_scale=ROUGH_LEG_TORQUE_ENVELOPE_SCALE,
            wheel_torque_envelope=wheel_torque_envelope,
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
        high_stand_transition_prob=float(high_stand_transition_prob),
        # 上限只由训练期课程项结算；play 时没有课程，打开只会空累计。
        stair_speed_cap_enabled=bool(stair_speed_cap) and not play,
    )
    if command_wheel_budget == "rated":
        command = cfg.commands["velocity_height"]
        command.diff_drive_max_wheel_speed = float(M3508_C620_14.rated_speed)
        command.diff_drive_half_track = serialleg_wheel_half_track()
    elif command_wheel_budget != "legacy":
        raise ValueError(f"未知的 command_wheel_budget {command_wheel_budget!r}")

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

    # M53：电机侧气弹簧前馈补偿（见 ROUGH_LEG_TORQUE_ENVELOPE_SCALE 上方注释）。
    delayed_action = cfg.actions["delayed_action"]
    if delayed_action.knee_gas_spring_force <= 0.0:
        raise ValueError(f"前馈补偿弹簧力必须为正数，实际为 {delayed_action.knee_gas_spring_force}")
    delayed_action.knee_gas_spring_compensation_enabled = True
    # M50 / M54：台阶列出生朝向正对台阶 ±15°（见 ROUGH_STAIR_SPAWN_YAW_HALF_RANGE_DEG）。
    reset_root = cfg.events["reset_root_state"]
    cfg.events["reset_root_state"] = replace(
        reset_root,
        params={
            **(reset_root.params or {}),
            "yaw_sampler": events.stair_facing_yaw,
            "yaw_sampler_params": {
                "terrain_type_names": ROUGH_REWARD_TERRAIN_TYPE_NAMES,
                "half_range_rad": math.radians(ROUGH_STAIR_SPAWN_YAW_HALF_RANGE_DEG),
            },
        },
    )

    _apply_rough_rewards(
        cfg,
        stair_height_reference=stair_height_reference,
        stair_height_dead_zone_m=stair_height_dead_zone_m,
        base_height_sigma=base_height_sigma,
    )
    # M49 / M54：左右轮心前后错位罚，台阶列 ×0.25（见 ROUGH_WHEEL_FORE_AFT_WEIGHT、ROUGH_STAIR_WHEEL_FORE_AFT_SCALE）。
    cfg.rewards["wheel_fore_aft_offset"] = RewardTermCfg(
        func=rewards.column_scaled,
        weight=ROUGH_WHEEL_FORE_AFT_WEIGHT,
        params={
            "inner": rewards.wheel_fore_aft_offset,
            "params": {"dead_zone_m": 0.0},
            "terrain_type_names": ROUGH_REWARD_TERRAIN_TYPE_NAMES,
            "scale": ROUGH_STAIR_WHEEL_FORE_AFT_SCALE,
        },
    )
    # M54：台阶列航向保持（见 ROUGH_STAIR_HEADING_HOLD_WEIGHT）。
    cfg.rewards["stair_heading_hold"] = RewardTermCfg(
        func=rewards.stair_heading_hold,
        weight=ROUGH_STAIR_HEADING_HOLD_WEIGHT,
        params={
            "command_name": "velocity_height",
            "terrain_type_names": ROUGH_REWARD_TERRAIN_TYPE_NAMES,
        },
    )
    # M38：全局向上奖励。
    if upward_weight is not None:
        cfg.rewards["upward"] = RewardTermCfg(func=mdp_rewards.upward, weight=float(upward_weight))
    if orientation_weight is not None:
        cfg.rewards["tracking_orientation_l2"] = replace(
            cfg.rewards["tracking_orientation_l2"], weight=float(orientation_weight)
        )
    # M39：action_rate 权重（只改权重，函数与参数不动）。
    cfg.rewards["action_rate"] = replace(
        cfg.rewards["action_rate"], weight=ROUGH_ACTION_RATE_WEIGHT
    )

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
    # M47：删倾角终止（见 ROUGH_WHEEL_FORE_AFT_WEIGHT 上方注释）。
    del cfg.terminations["bad_orientation"]

    # DR1：rough 默认域随机化（见 ROUGH_DR_* 注释）。
    _apply_rough_domain_randomization(cfg, play=play)

    # RJ1：合入 J10 的跳跃（见 _apply_jump_mimic）。
    _apply_jump_mimic(cfg)
    # 2026-10-02：观测 34 → 30 维、部署指令六维（去掉 pitch / roll 指令与 wheel_pos_zero，见 tasks.common.no_attitude）。
    apply_no_attitude_layout(cfg)
    if oracle_dr_obs:
        _apply_oracle_dr_obs(cfg)
    if oracle_base_vel:
        # DR1 OracleVel（2026-10-05，用户定）：在 Oracle 之上再给 actor 机身线速度，看 DR1 下平地跟踪变粗是否来自
        # 速度只能从轮速间接推（质心偏移配平、打滑、延迟都会扰乱轮速→机身速度的换算）。不加噪声。
        actor = cfg.observations["actor"]
        terms = dict(actor.terms)
        terms["oracle_base_lin_vel"] = ObservationTermCfg(func=mdp_observations.base_lin_vel_obs)
        cfg.observations["actor"] = replace(actor, terms=terms)
    if oracle_base_vx:
        # DR1 Vx（2026-10-05，用户定）：OracleVel 的 actor 里 vx 敏感度 2.11、vz 0.51、vy 0.28，看只给 vx 能保住多少平地跟踪收益。
        actor = cfg.observations["actor"]
        terms = dict(actor.terms)
        terms["oracle_base_lin_vel_x"] = ObservationTermCfg(
            func=mdp_observations.base_lin_vel_x_obs
        )
        cfg.observations["actor"] = replace(actor, terms=terms)
    if vx_observer:
        _apply_vx_observer_obs(cfg, history_length=vx_observer_history_length)

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
        # 台阶列逐 env 速度上限；必须排在 flat_warmup / two_step_gate 之后（它们会给 env 换列）。
        if stair_speed_cap:
            cfg.curriculum["stair_speed_cap"] = CurriculumTermCfg(
                func=curriculums.stair_speed_cap,
                params={"command_name": "velocity_height"},
            )

    return cfg


ROUGH_JUMP_MAX_HEIGHT_ERROR = 0.12
"""RJ1 跳跃偏离参考终止阈值（同 J2 起的 0.12 m）。"""
ROUGH_JUMP_PHASE_TIME_SCALE_S = 1.5
"""RJ1 jump_phase = 参考时刻 / 该常数（同 J10）。"""


def _apply_jump_mimic(cfg: ManagerBasedRlEnvCfg) -> None:
    """RJ1（2026-10-02 用户定）：按 J10 合入跳跃——34 维观测里的 jump_flag / 目标高度 / 相位、无 RSI、无下蹲参考。

    只有跳跃样本会跳（2026-10-03 起为专用平地跳跃列的全部 env，占全部 env 10%；此前为平地列 30%），
    且整回合高度指令固定 0.22（指令项见 commands.RoughJumpCommandTerm）；
    四项模仿奖励只计跳跃样本；跳跃期间屏蔽静站罚、接触力罚与速度跟踪的 vz 项（机身高度罚、轮 / 腿离地罚
    本来就按 jump_flag 屏蔽）；偏离参考提前终止不吃摔倒罚。观测维度与 M54 相同。
    """
    from se3_train.tasks.jump_mimic import mdp as jump_mdp
    from se3_train.tasks.jump_mimic.env_cfg import (
        JUMP_MIMIC_REFERENCE_HEIGHTS,
        JUMP_MIMIC_REWARD_WEIGHTS,
    )
    from se3_train.tasks.jump_mimic.reference import JumpReferenceLibrary, reference_paths

    paths = reference_paths(JUMP_MIMIC_REFERENCE_HEIGHTS)
    library = JumpReferenceLibrary(paths, "cpu")
    base = cfg.commands["velocity_height"]
    fields_ = {f.name: getattr(base, f.name) for f in fields(base) if f.init}
    ranges = dict(fields_.get("deployment_ranges") or {})
    ranges.update(
        jump_flag=(0.0, 1.0),
        jump_target_height=(0.0, float(library.target_clearance.max())),
        jump_phase=(0.0, float(library.duration.max()) / ROUGH_JUMP_PHASE_TIME_SCALE_S),
    )
    fields_.update(
        enable_jump_lifecycle=False,
        deployment_ranges=ranges,
        reference_paths=paths,
        phase_time_scale_s=ROUGH_JUMP_PHASE_TIME_SCALE_S,
    )
    cfg.commands["velocity_height"] = RoughJumpCommandCfg(**fields_)

    cfg.rewards = dict(cfg.rewards)
    for name, weight in JUMP_MIMIC_REWARD_WEIGHTS.items():
        cfg.rewards[name] = RewardTermCfg(
            func=rewards.jump_env_only,
            weight=float(weight),
            params={"inner": getattr(jump_mdp, name), "params": {}},
        )
    for name in ("stand_still", "contact_forces"):
        term = cfg.rewards[name]
        cfg.rewards[name] = RewardTermCfg(
            func=jump_mdp.not_jumping,
            weight=float(term.weight),
            params={"inner": term.func, "params": dict(term.params or {})},
        )
    track = cfg.rewards["tracking_lin_vel"]
    if track.func is not rewards.tracking_lin_vel_terrain_vz:
        raise ValueError("RJ1 只接 se3 速度跟踪核（tracking_lin_vel_terrain_vz）")
    cfg.rewards["tracking_lin_vel"] = replace(
        track, params={**track.params, "zero_vz_when_jumping": True}
    )
    fall = cfg.rewards["fall_penalty"]
    cfg.rewards["fall_penalty"] = RewardTermCfg(
        func=rewards.is_terminated_except,
        weight=float(fall.weight),
        params={"exclude_terms": ("mimic_deviation",)},
    )

    cfg.terminations = dict(cfg.terminations)
    cfg.terminations["mimic_deviation"] = TerminationTermCfg(
        func=jump_mdp.mimic_deviation,
        time_out=False,
        params={"max_height_error": ROUGH_JUMP_MAX_HEIGHT_ERROR},
    )


def _apply_rough_domain_randomization(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
    """DR1（见 ROUGH_DR_* 注释）：改 Flat 继承的 startup 事件参数，Kp/Kd 换成 torch 侧实现，放宽动作延迟。

    动作延迟属于动作项，play（评测 / ONNX 导出）同样生效，契约随之导出 0–10 ms；
    其余都是训练期 startup 事件，play 没有这些事件。
    """
    delayed_action = cfg.actions["delayed_action"]
    min_s, max_s = ROUGH_DR_ACTION_DELAY_RANGE_S
    delayed_action.action_delay_enabled = True
    delayed_action.action_delay_randomize = True
    delayed_action.action_delay_min_s = float(min_s)
    delayed_action.action_delay_max_s = float(max_s)
    delayed_action.action_delay_s = 0.5 * (float(min_s) + float(max_s))
    if play:
        return

    def _params(name: str, **overrides) -> dict:
        return {**(cfg.events[name].params or {}), **overrides}

    cfg.events["pd_gains"] = replace(
        cfg.events["pd_gains"],
        func=mdp_events.randomize_pd_gains_torch,
        params=_params("pd_gains", kp_range=ROUGH_DR_KP_RANGE, kd_range=ROUGH_DR_KD_RANGE),
    )
    cfg.events["com"] = replace(
        cfg.events["com"], params=_params("com", com_range=ROUGH_DR_COM_RANGE_M)
    )
    cfg.events["base_mass"] = replace(
        cfg.events["base_mass"],
        params=_params("base_mass", mass_range=ROUGH_DR_BASE_MASS_RANGE_KG),
    )
    cfg.events["knee_spring_force"] = replace(
        cfg.events["knee_spring_force"],
        params=_params("knee_spring_force", force_scale_range=ROUGH_DR_KNEE_SPRING_SCALE_RANGE),
    )


def _apply_vx_observer_obs(cfg: ManagerBasedRlEnvCfg, *, history_length: int) -> None:
    """显式 vx 观测器（2026-10-05，用户定，见 se3_train.vx_observer）的观测侧改动。

    actor 组整组开 16 帧历史（term-major、oldest→newest 展平；reset 后第一帧回填整段，runtime 同样处理），
    噪声逐帧在入历史前施加，与部署端一致。critic 仍是单帧。监督目标单独成组，不加噪声，
    rsl_rl 的 obs_groups 不引用它，所以只进 rollout storage、不进任何网络输入。
    """
    actor = cfg.observations["actor"]
    cfg.observations["actor"] = replace(
        actor, history_length=int(history_length), flatten_history_dim=True
    )
    cfg.observations[ROUGH_VX_OBSERVER_TARGET_GROUP] = ObservationGroupCfg(
        terms={"base_lin_vel_x": ObservationTermCfg(func=mdp_observations.base_lin_vel_x_obs)},
        concatenate_terms=True,
        enable_corruption=False,
    )


def _apply_oracle_dr_obs(cfg: ManagerBasedRlEnvCfg) -> None:
    """DR1 Oracle（2026-10-05，用户定）：actor 观测末尾追加真实 DR 参数 31 维，critic 不变。

    前置观测器（历史 → 隐向量）方案的收益上限：actor 直接知道 DR1 下的真实参数，看名义平地跟踪 / 中速前后晃能否
    回到旧 DR 水平。追加项与 critic 同源：28 维 DR 回读、气弹簧力 2 维、动作延迟 1 维（DR1 不做恢复系数，不加那一维）。
    不加噪声。导出器不认识这些项，训练期 ONNX 导出会失败（runner 捕获，checkpoint 照存）。
    此前 WideDR 上的 Oracle（1fa00ca）受恢复系数实现问题污染，结论不适用于 DR1。
    """
    actor = cfg.observations["actor"]
    terms = dict(actor.terms)
    terms["oracle_dr_params"] = ObservationTermCfg(func=mdp_observations.dr_model_params_obs)
    terms["oracle_knee_spring"] = ObservationTermCfg(
        func=mdp_observations.knee_gas_spring_force_obs
    )
    terms["oracle_action_delay"] = ObservationTermCfg(func=mdp_observations.action_delay_obs)
    cfg.observations["actor"] = replace(actor, terms=terms)


def _apply_rough_rewards(
    cfg: ManagerBasedRlEnvCfg,
    *,
    stair_height_reference: str = ROUGH_STAIR_HEIGHT_REFERENCE,
    stair_height_dead_zone_m: float = ROUGH_STAIR_HEIGHT_DEAD_ZONE_M,
    base_height_sigma: float = ROUGH_BASE_HEIGHT_SIGMA,
) -> None:
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
    # M37：从 Flat 继承的两项重叠定价整项删除（另三项 rough 不再构造）。
    for name in ROUGH_DROPPED_FLAT_REWARDS:
        del cfg.rewards[name]
    # M21：上台阶列高度参考改为轮子支撑面，其余列与 Flat 原函数逐位相同（σ 取 A15 的 0.10）。
    # "window" 口径换成机身周围 77 点窗口均值 + 有界罚，其余参数不变（见 ROUGH_STAIR_HEIGHT_REFERENCE）。
    height = cfg.rewards["flat_base_height"]
    height_params = {
        **height.params,
        "sigma": float(base_height_sigma),
        "support_sensor_name": ROUGH_BASE_HEIGHT_SUPPORT_SENSOR,
        "terrain_type_names": ROUGH_BASE_HEIGHT_SUPPORT_COLUMNS,
    }
    if stair_height_reference == "window":
        height_func = rewards.base_height_penalty_window_on_terrain
        height_params["window_sensor_name"] = ROUGH_CRITIC_HEIGHT_SCAN_SENSOR_NAME
        height_params["dead_zone_m"] = float(stair_height_dead_zone_m)
    else:
        height_func = rewards.base_height_penalty_support_on_terrain
    cfg.rewards["flat_base_height"] = replace(height, func=height_func, params=height_params)
    # M44 / M50：yaw 角速度跟踪全列同权同参（Flat 原式）；台阶列 yaw 指令恒 0，它在台阶列就是 yaw 角速度罚。
    cfg.rewards["tracking_ang_vel"] = replace(
        cfg.rewards["tracking_ang_vel"], func=mdp_rewards.tracking_ang_vel
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
            "scale": ROUGH_TRACKING_LIN_VEL_NARROW_STAIR_WEIGHT
            / ROUGH_TRACKING_LIN_VEL_NARROW_WEIGHT,
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
    "ROUGH_ACTION_RATE_WEIGHT",
    "ROUGH_ALL_TERRAIN_TYPE_NAMES",
    "ROUGH_BASE_HEIGHT_SIGMA",
    "ROUGH_BASE_HEIGHT_SUPPORT_COLUMNS",
    "ROUGH_BASE_HEIGHT_SUPPORT_SENSOR",
    "ROUGH_BODY_COLLISION_BOTTOM_OFFSET",
    "ROUGH_CATASTROPHIC_MIN_BASE_HEIGHT",
    "ROUGH_COMMAND_WHEEL_BUDGET",
    "ROUGH_CONTACT_SENSOR_MAXMATCH",
    "ROUGH_CONTACT_TAX_FREE_COLUMNS",
    "ROUGH_CRITIC_HEIGHT_SCAN_RESOLUTION_M",
    "ROUGH_CRITIC_HEIGHT_SCAN_SENSOR_NAME",
    "ROUGH_CRITIC_HEIGHT_SCAN_SIZE_M",
    "ROUGH_CURRICULUM_SIGNAL_TERRAIN_NAMES",
    "ROUGH_CURRICULUM_TRACKING_LOG_KEY",
    "ROUGH_DROPPED_FLAT_REWARDS",
    "ROUGH_DR_ACTION_DELAY_RANGE_S",
    "ROUGH_DR_BASE_MASS_RANGE_KG",
    "ROUGH_DR_COM_RANGE_M",
    "ROUGH_DR_KD_RANGE",
    "ROUGH_DR_KNEE_SPRING_SCALE_RANGE",
    "ROUGH_DR_KP_RANGE",
    "ROUGH_FALL_PENALTY",
    "ROUGH_FLAT_VZ_WEIGHT",
    "ROUGH_FLAT_WARMUP_ITERATIONS",
    "ROUGH_FLAT_WARMUP_RAMP_ITERATIONS",
    "ROUGH_HIGH_STAND_TRANSITION_PROB",
    "ROUGH_JUMP_MAX_HEIGHT_ERROR",
    "ROUGH_JUMP_PHASE_TIME_SCALE_S",
    "ROUGH_LEG_TORQUE_ENVELOPE_SCALE",
    "ROUGH_MAX_INIT_TERRAIN_LEVEL",
    "ROUGH_NCONMAX",
    "ROUGH_NJMAX",
    "ROUGH_OFF_STAIR_TRACKING_SIGMA_MOVE",
    "ROUGH_ORIENTATION_WEIGHT",
    "ROUGH_REWARD_TERRAIN_TYPE_NAMES",
    "ROUGH_ROBOT_COLLISION_GEOM_GROUP",
    "ROUGH_STAIRS_ZEROED_REWARDS",
    "ROUGH_STAIR_ANG_VEL_YAW_RANGE",
    "ROUGH_STAIR_CLIMB_PROGRESS_WEIGHT",
    "ROUGH_STAIR_COMMAND_TERRAIN_NAMES",
    "ROUGH_STAIR_HEADING_HOLD_WEIGHT",
    "ROUGH_STAIR_HEIGHT_DEAD_ZONE_M",
    "ROUGH_STAIR_HEIGHT_RANGE",
    "ROUGH_STAIR_HEIGHT_REFERENCE",
    "ROUGH_STAIR_LIN_VEL_X_RANGE",
    "ROUGH_STAIR_SPAWN_YAW_HALF_RANGE_DEG",
    "ROUGH_STAIR_SPEED_CAP_ENABLED",
    "ROUGH_STAIR_SUPPORT_HEIGHT_WEIGHT",
    "ROUGH_STAIR_TRACKING_SIGMA_MOVE",
    "ROUGH_STAIR_WHEEL_FORE_AFT_SCALE",
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
    "ROUGH_UPWARD_WEIGHT",
    "ROUGH_VX_OBSERVER_HISTORY_LENGTH",
    "ROUGH_VX_OBSERVER_TARGET_GROUP",
    "ROUGH_VZ_FLAT_TERRAIN_TYPE_NAMES",
    "ROUGH_WHEEL_FORE_AFT_WEIGHT",
    "ROUGH_WHEEL_TORQUE_ENVELOPE",
    "env_cfg",
]
