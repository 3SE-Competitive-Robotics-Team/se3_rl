"""崎岖地形行走任务环境配置。

移植自 scutrobotlab/wheeled-legged_RL 的 V14 rough 线，做法与参考仓库一致：rough 直接继承
flat 的整套配置，只换地形、加地形课程、按地形抬高高度指令下限，奖励表只放松能耗类三项，其余逐项不动。
参考仓库 rough 相对 flat 的奖励改动就只有 `wheel_power` 与 `joint_torque` 各 ÷10
（`WheelbipeV14RoughEnvCfg.__post_init__`）；不引入新的奖励项。

继承的是**冻结的 Flat 基线**（D11，见 tasks/flat/env_cfg.py 与 tests/test_flat_baseline.py），
即轮 scale 15 + 弹簧时代 action_smoothness 定价，而不是 `flat_env_cfg()` 的裸默认值
（那套是旧继承线的 scale 45 契约）。
"""

from __future__ import annotations

from dataclasses import fields, replace

from mjlab.envs import ManagerBasedRlEnvCfg
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
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from se3_train.mdp import events as shared_events
from se3_train.mdp.amp_observations import build_amp_mask_terms, build_amp_obs_terms
from se3_train.tasks.flat.env_cfg import (
    FLAT_ACTION_SMOOTHNESS_SPRING,
    FLAT_CMD_VEL_DEADBAND,
    FLAT_COMMAND_VELOCITY_ERROR_WEIGHT_LEGACY,
    FLAT_CURRICULUM_ADVANCE_THRESHOLD,
    FLAT_WHEEL_ACTION_SCALE,
)
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg
from se3_train.tasks.stair import observations as stair_observations

from . import ctbc, curriculums, events, observations, rewards, stair_rewards, terminations
from .commands import RoughCommandCfg
from .terrains import rough_terrains_cfg

# 台阶前的机身抬升：2026-09-08 用户定，删掉 step_up 前瞻状态机，改用**地形感知抬高下限**。
# 重采样时按 env 所在列与难度行算出这一级台阶需要的最低机身高度，把高度指令采样区间的
# 下界顶到该值（实现在 mdp/commands.py 的 _terrain_aware_min_height，rough 只负责配数）：
#   required = step_height + terrain_height_clearance - body_collision_bottom_offset
# 数值沿用 stair 线的标定（tasks/stair/env_cfg.py）：机体碰撞网格底面在 base_link 下方
# 0.12 m（COACD 网格 z 范围 [-0.1376, 0.1118]，取平底面而非最低角点），再留 0.02 m 余量。
# 台阶 0.02→0.20 m 对应的下限是 0.20（行 0-1 不生效）→0.34 m，始终在 height_range 上界 0.38 之内。
ROUGH_TERRAIN_AWARE_HEIGHT = True
ROUGH_TERRAIN_HEIGHT_CLEARANCE = 0.02
ROUGH_BODY_COLLISION_BOTTOM_OFFSET = -0.12
# 只在上台阶列抬高：下行列的台阶在身后，抬高只是白白升高重心（该列当前 proportion 也是 0）。
# 名字必须是 terrains.rough_terrains_cfg() 里带 step_height_range 的子地形名，
# 对不上时下限静默失效（基类默认值是 stair 线的列名），由 tests/test_rough_port.py 钉住。
ROUGH_TERRAIN_STEP_HEIGHT_TYPE_NAMES = ("stairs_up",)

# 台阶列的分列定价（2026-09-08 用户定，A6）：这两项都只改生效范围，不改数值。
# 生效列与 AMP、地形感知高度下限取同一组，默认只有上台阶列。
ROUGH_REWARD_TERRAIN_TYPE_NAMES = ("stairs_up",)
# 台阶运动核分母：误差约 1 m/s 时仍有半额奖励，给低速前进提供可区分的回报。
ROUGH_STAIR_TRACKING_SIGMA_MOVE = 1.44
# 速度违令二次罚（rewards.command_velocity_error_on_terrain）只在台阶列加回来。
# tracking_lin_vel 的高斯核 σ_move=0.08 在误差 >0.4 m/s 处没有梯度；平地上误差小且短暂，
# 所以 2026-09-06（D7 对 D4）把这一项从 Flat 删掉是对的（它 99% 的代价来自指令阶跃后 1 s 内，
# 等于奖励指令跳变后猛冲）。台阶列是另一个区间：A5 换列后跟踪误差长期大于 0.4 m/s、
# 跟踪分从 2.2 掉到 0.5 再没回来，正是核压零、没有梯度的那一段。权重与死区沿用删除前的历史值。
# None = 不加回来，退回 Flat 基线（全线都没有这一项）。
ROUGH_COMMAND_VELOCITY_ERROR_WEIGHT: float | None = FLAT_COMMAND_VELOCITY_ERROR_WEIGHT_LEGACY
# 唯一相对历史值改动的参数：误差归一化尺度 0.5 → 1.5（2026-09-08 用户定）。
# 0.5 是按平地的误差量级定的：封顶 9 在误差 1.55 m/s 处就到顶。台阶列的 vx 指令是 0.4–2.4，
# 而 A5 的策略在台阶上跑不到 2 m/s，误差长期 1.5–2.4，用 0.5 会全程贴封顶——
# 贴封顶等于常数，梯度又没了（正是这项要解决的问题），而 |weight|×9 = 18/s 的常数负奖励
# 比全部正项加起来（约 10/s，is_alive 只有 1/s）还大，早终止在数值上严格更优，会教出自杀策略。
# 1.5 是按 A7 的 vx 0.4–0.8 定的，A8 把台阶列提到 1.0–2.4 之后单 env 罚值涨到 −2.48/s（A9 实测 −2.496），
# 成了台阶列最大的单项。2026-09-09 用户定（A10）改 3.0：A10 同时把 tracking_ang_vel 在台阶列归零，
# 台阶列的正项从 3.753 掉到 1.015，若仍留着 −2.50 的罚，净收益就是 −2.69/s——一整段 episode 累计约 −54，
# 而第一步摔倒终止是 ≈0（catastrophic 是失败终止，PPO 按 0 值 bootstrap），故意撞死在数值上严格占优。
# 3.0 之后：误差 1.65（A9 实测）→ −0.57/s、2.35（满指令且不动）→ −1.23/s，净收益约 −0.76/s，
# 仍为负（必须真的往前爬拿到 tracking_lin_vel 才转正，这正是要的梯度方向），但不再诱发自杀。
# yaw 尺度不动：台阶列 yaw 指令已固定为 0，误差本来就小。
ROUGH_COMMAND_VELOCITY_ERROR_LIN_SCALE = 3.0
# 机身高度罚（flat_base_height，(clamp(err,±0.15)/0.05)² 无界二次罚，误差 0.15 m 即 36/s）
# 在台阶列置零：爬升时机身相对脚下地面的高度必然大幅偏离指令，这项罚等于按爬升幅度罚钱。
# 置零后台阶列的姿态由 AMP 风格奖励和地形感知高度下限管；其余列与 Flat 基线逐位相同。
ROUGH_ZERO_BASE_HEIGHT_ON_TERRAIN = True
# 非平地列把 tracking_lin_vel 核里的 vz 项关掉（2026-09-08 用户定，A7）。
# 核是 exp(-(err_x² + vz_weight·vz²)/σ)，爬台阶和上坡必须有垂直速度，而这项按 vz² 扣分：
# vz 0.2 m/s 把核乘掉 0.37，16° 坡上以 1 m/s 走的 vz 就是 0.28。平地列保持 2.0 不变。
# 用“非平地”取反而不是点名列，新增子地形时不用记得来加名字。
ROUGH_TERRAIN_VZ_WEIGHT = 0.0
ROUGH_VZ_FLAT_TERRAIN_TYPE_NAMES = ("flat",)

ROUGH_ALL_TERRAIN_TYPE_NAMES = (
    "flat",
    "stairs_up",
    "stairs_down",
    "slope_up",
    "slope_down",
    "random_rough",
)
"""rough_terrains_cfg() 的全部六列。给 command_velocity_error 放开列门控时用。"""

# 非台阶列的定价修正（2026-09-11 定，A15）。A13b model_1600 在 flat 列上的实测奖励账本
# （96 env，53 个在走 / 24 个卡住，指令 vx 2.0 / h 0.38，单位每秒）：
#
#   奖励项                        走 1.64 m/s   卡 0.11 m/s      差
#   flat_base_height                  -4.625       -0.473    -4.152
#   tracking_orientation_l2           -0.743       -0.019    -0.724
#   bad_tilt                          -0.572       -0.005    -0.567
#   tracking_lin_vel                  +0.115        0.000    +0.115
#   tracking_ang_vel                  +2.711       +2.328    +0.383
#   is_alive                          +1.000       +1.000         0
#   合计                              -2.932       +2.407    -5.339
#
# 站着不动净赚 +2.407/秒，走路净亏 2.932/秒——站着是这个奖励函数在非台阶列上真正的最优解，
# 死锁、速度天花板、3° 起步基域都是它的表现。根因是「走路必然产生的机身起伏」被罚了三遍：
# flat_base_height 直接罚（占缺口 78%）、tracking_lin_vel 核里的 vz 项再罚一次（实测核衰减
# 0.284 里 vz 占 0.154，比速度误差的 0.130 还多）、orientation/bad_tilt 再罚一次姿态。
# 而 tracking_lin_vel 满分只有 4.0/秒，光高度罚的差额就 4.15/秒——即使速度跟踪做到满分也不划算。
#
# 对照 A0（开着 AMP）同一份账本：走 -3.885、卡 +1.879，缺口 5.77 比 A13b 还大，但它照样走。
# 两者 RewardManager 侧几乎一样，区别只有 AMP 那笔发在管理器之外的 +9.1/秒——AMP 一直是
# 压住这个反向激励的配重，A13 拿掉它只是让问题显形。修法是把价格改对，不是把配重加回来。
#
# 台阶列早就改对了（高度罚置零、σ_move 1.44、vz 项关闭、挂违令罚），A15 把同一套扩到其余五列。
ROUGH_BASE_HEIGHT_SIGMA = None
"""非台阶列 flat_base_height 的 σ；None 跟随 Flat 基线的 0.05。A15 用 0.10（收回 +3.11/秒）。"""

ROUGH_OFF_STAIR_TRACKING_SIGMA_MOVE = None
"""非台阶列 tracking_lin_vel 的运动核分母；None 跟随 Flat 基线的 0.08。

0.08 是为 Flat 基线 ±1.0 m/s 的指令范围调的，现在课程推到 2.4 m/s，可达误差远超 0.28 m/s，
核在最需要它的地方是死的：实测走到 1.64/2.0（八成）只拿满分的 2.8%。A15 用 0.5。
"""

ROUGH_FLAT_VZ_WEIGHT = None
"""平地列 tracking_lin_vel 核里的 vz 权重；None 跟随 Flat 基线的 2.0。A15 用 0.0。

与 ROUGH_OFF_STAIR_TRACKING_SIGMA_MOVE 合计收回约 +2.97/秒。
"""

ROUGH_COMMAND_VELOCITY_ERROR_TERRAIN_NAMES = None
"""command_velocity_error 生效的列；None 表示跟随 reward_terrain_type_names（只有台阶列）。

A15 传全部六列。这是唯一不对称的一项（只打在"不动"那边），按 A10 的定标约 +1.2/秒。
"""

# 非平地列的速度指令限制：只发前向直行指令，平地列沿用 Flat 的速度课程。
# 对称随机指令下 20 s 的净位移是随机游走，地形课程的位移判据推不动（R2 平地列也只到 1.6）。
ROUGH_TERRAIN_COMMAND_OVERRIDE_ENABLED = True
# 2026-09-08 用户定（A7）：地形列 vx 从 (0.4, 2.4) 收到 (0.4, 0.8)，并与平地课程脱钩。
# 依据：A6 的 Rough/command_velocity_error_terrain=0.95 反解出地形列 vx 误差约 1.5 m/s（RMS 口径），
# 而指令均值 (0.4+2.4)/2=1.4——误差和指令一样大，机器人相对指令基本不动，
# 核 exp(-1.5²/0.08)=7e-13 精确为零：没有分，也没有梯度。
# 原来 vx 上限跟随**平地**课程当前上限，而平地 350 轮就冲到 2.4，等于换列那一刻直接给了
# 一个爬 5 cm 台阶的轮足机器人做不到的数。R4 加 follow_curriculum 的直觉对，但挂错了信号。
# 收到 0.8 之后指令均值 0.6：踏面上滚到 0.4 就是误差 0.2 → 核 0.61，梯度回来。
ROUGH_TERRAIN_LIN_VEL_X_RANGE = (0.4, 0.8)
ROUGH_TERRAIN_ANG_VEL_YAW_RANGE = (-0.2, 0.2)
ROUGH_TERRAIN_LIN_VEL_X_FOLLOW_CURRICULUM = False
# 台阶列单独定价（2026-09-08 用户定，A8）：A7 把所有非平地列 vx 收到 (0.4, 0.8) 之后，
# 平地能力（2 m/s 误差 0.03）和地形列梯度（base_vx_error_terrain 0.37）都回来了，
# 但 stairs_up 1500 轮只从 1.09 挪到 1.11——6 cm 轮子靠 0.8 m/s 的动量翻不过 4 cm 立面，
# 而 slope_up 已经 6.39、平地 6.1。所以把台阶列拆出来给高速 + 高站姿，其余地形列保持低速档。
# vx 1.0–2.4 与专家数据（Fudan 12–20 cm 爬升，1.5–2.4 m/s）同一段；高度 0.35–0.38 顶到
# Flat 上界附近换离地净空，整体高于地形感知抬高下限在最高难度行的 0.34，那条下限在本列被吞掉。
ROUGH_STAIR_COMMAND_TERRAIN_NAMES = ("stairs_up",)
ROUGH_STAIR_LIN_VEL_X_RANGE = (1.0, 2.4)
ROUGH_STAIR_HEIGHT_RANGE = (0.35, 0.38)
# 台阶列取消「不动的工资」（2026-09-09 用户定，A10）：yaw 指令固定为 0，且该项奖励在台阶列归零。
# A9 逐项拆分：台阶列 tracking_ang_vel +2.739/s，占该列正奖励 3.753 的 73%，静止即可拿满
# （yaw 指令 ±0.2、σ=0.25，不动时误差 0.1、核 0.96）；加 is_alive +1.0，站着不动净 +0.053/s 为正。
ROUGH_STAIR_ANG_VEL_YAW_RANGE = (0.0, 0.0)
ROUGH_ZERO_TRACKING_ANG_VEL_ON_TERRAIN = True

# reset 帧 last_actions 随机化（2026-09-10 定，A14）。训练里 `ActionManager.reset()` 把 `_action`
# 清零，于是「reset 帧 last_actions 恒为全 0」成了策略能依赖的模式开关：A13b model_1600 在
# vx 2.0 / h 0.38 下干净 reset 后 2.15 m/s，先站 5 秒再切同一条指令只有 0.01 m/s，而单独清
# last_actions 或单独清关节姿态都出不来（0.01 / 0.04），两个一起清才回到 2.15。默认 0 保持
# 与历史各线逐位一致，只有显式给正数才启用。范围按实测原始动作跨度（走路 [-3.62, 3.40]）定。
ROUGH_RESET_LAST_ACTION_RANGE = 0.0
ROUGH_RESET_LAST_ACTION_PROB = 1.0
# 逐项奖励的分列日志（2026-09-09 用户定，A9）：每步把奖励表每一项在台阶列上的均值记一份，
# 键名 Rough/rw_<项名>_stairs。A8 只能读出 command_velocity_error 在台阶列是 −2.48/s
# （因为它本来就只在那列生效），其余罚项被平地列稀释、无从定位。不改奖励数学。
ROUGH_REWARD_SPLIT_LOG_ENABLED = True
ROUGH_CURRICULUM_SIGNAL_TERRAIN_NAMES = ("flat",)
ROUGH_CURRICULUM_TRACKING_LOG_KEY = "Locomotion/tracking_lin_vel_reward_curriculum"

# 能耗类罚项在崎岖地形上的折价系数。参考仓库把 wheel_power 与 joint_torque
# 从 -1e-4 降到 -1e-5：上台阶本来就要更多力矩和功率，沿用平地定价会把爬升压住。
ROUGH_ENERGY_PENALTY_SCALE = 0.1
_ROUGH_ENERGY_REWARD_NAMES = ("leg_torques", "wheel_torques", "leg_power")

# 全部 env 从最简单一行起步，难度由 terrain_levels 课程逐级放开。
ROUGH_MAX_INIT_TERRAIN_LEVEL = 0

# 平地热身：前 N 轮全部 env 在平地列，之后各 env 在下一次 reset 时换回原列（2026-09-07 用户定，R7）。
# 速度课程推进阈值：R7–A4 用严格档 0.75，但 vx=0 阶段的跟踪 EMA 天花板就在 0.72–0.75，首次推进要等 200–350 轮、
# 且靠种子（A4 到 499 轮只推到 0.2）；Flat 基线 D10 用 0.5 在 45 轮推进、350 轮到 2.4、1000 轮追平跟踪。
# 2026-09-08 用户定（A5）：改回 Flat 默认 0.5。
ROUGH_FLAT_WARMUP_ITERATIONS = 500
# 换列不再一刀切：地形 env 比例在 500→1000 轮之间从 0 线性涨到 1（2026-09-08 用户定，A7）。
# A5/A6 的本机 sim2x 回放显示伤害集中在换列后那 100 轮：A6 model_500 在 2 m/s 上误差 0.02，
# model_600 掉到 0.93，A5 同型（model_1000 误差 1.80）——一次性把 75% 的 env 扔进跟不上的
# 指令里，共享 actor 连平地一起退化。0 即退回一刀切。
ROUGH_FLAT_WARMUP_RAMP_ITERATIONS = 500
ROUGH_CURRICULUM_ADVANCE_THRESHOLD = FLAT_CURRICULUM_ADVANCE_THRESHOLD

# AMP 观测组：契约 19 维运动帧（se3.amp.motion.v1，docs/amp_input.md）按 AMP_DISCRIMINATOR_FIELDS 切成 17 维（去轮速），只供判别器用。
ROUGH_AMP_OBS_GROUP = "amp"
# AMP 只对这些子地形列生效（2026-09-08 用户定：只有上台阶列）；掩码走独立观测组，判别器按它筛 env。
ROUGH_AMP_MASK_OBS_GROUP = "amp_mask"
ROUGH_AMP_TERRAIN_TYPE_NAMES = ("stairs_up",)

# CTBC：轮子顶住台阶立面时替策略把该侧轮子向后上方缩回（stair 线的 teacher-forcing，见 ctbc.py）。
# 退火按训练轮次：ann_start 之前满幅，ann_start→ann_end 线性退到 0，之后策略自己上台阶。
# 只在上台阶列触发；500 轮前满幅，500→1500 线性退火，之后关闭（2026-09-07 用户定）。
# 2026-09-07 默认关闭：R4/R5 从第 0 轮注入前馈，100–700 轮 catastrophic 0.2–0.5（R3 为 0），
# 策略学成回避接触，平地跟踪也一起退化（4000 轮策略 vx 0.7 不走）。要开显式传 ctbc_enabled=True。
ROUGH_CTBC_ENABLED = False
ROUGH_CTBC_ANN_START_ITER = 500
ROUGH_CTBC_ANN_END_ITER = 1500
ROUGH_CTBC_TERRAIN_TYPE_NAMES = ("stairs_up",)
ROUGH_CTBC_RISER_SENSOR_NAME = "wheel_riser_sensor"
ROUGH_CTBC_STEPS_PER_POLICY_ITER = 24

# critic 特权地形观测：机身系 yaw 对齐网格，x ±0.5 m、y ±0.3 m、间距 0.1 m，11×7 = 77 条射线，
# 与 yly-true/fudan_rl_wheel_leg 的 measured_points_x/y 一致。只进 critic，actor 契约不变。
ROUGH_CRITIC_HEIGHT_SCAN_ENABLED = True
ROUGH_CRITIC_HEIGHT_SCAN_SENSOR_NAME = "critic_height_scan"
ROUGH_CRITIC_HEIGHT_SCAN_SIZE_M = (1.0, 0.6)
ROUGH_CRITIC_HEIGHT_SCAN_RESOLUTION_M = 0.1

# 接触传感器匹配槽数，见 env_cfg() 内的注释。
ROUGH_CONTACT_SENSOR_MAXMATCH = 500

_ROUGH_FLAT_BASELINE = {
    "wheel_action_scale": FLAT_WHEEL_ACTION_SCALE,
    "action_smoothness": FLAT_ACTION_SMOOTHNESS_SPRING,
}


def _to_rough_command_cfg(command_cfg, **overrides) -> RoughCommandCfg:
    """把 Flat 的 JumpCommandCfg 原样搬进 RoughCommandCfg，再覆盖 rough 要改的字段。

    逐字段搬运而不是重新构造，是为了让 Flat 基线以后改指令参数时 rough 自动跟随；
    `overrides` 既能填 RoughCommandCfg 新增的分列速度字段，也能覆盖继承来的
    地形感知高度字段（后者在基类 VelocityHeightCommandCfg 上，Flat 用的是默认值）。
    """
    base = {f.name: getattr(command_cfg, f.name) for f in fields(command_cfg) if f.init}
    base.update(overrides)
    return RoughCommandCfg(**base)


def env_cfg(
    play: bool = False,
    *,
    terrain_generator: TerrainGeneratorCfg | None = None,
    terrain_height_clearance: float = ROUGH_TERRAIN_HEIGHT_CLEARANCE,
    terrain_step_height_type_names: tuple[str, ...] = ROUGH_TERRAIN_STEP_HEIGHT_TYPE_NAMES,
    reward_terrain_type_names: tuple[str, ...] = ROUGH_REWARD_TERRAIN_TYPE_NAMES,
    stair_command_terrain_names: tuple[str, ...] = ROUGH_STAIR_COMMAND_TERRAIN_NAMES,
    stair_lin_vel_x_range: tuple[float, float] = ROUGH_STAIR_LIN_VEL_X_RANGE,
    stair_height_range: tuple[float, float] = ROUGH_STAIR_HEIGHT_RANGE,
    stair_ang_vel_yaw_range: tuple[float, float] = ROUGH_STAIR_ANG_VEL_YAW_RANGE,
    zero_tracking_ang_vel_on_terrain: bool = ROUGH_ZERO_TRACKING_ANG_VEL_ON_TERRAIN,
    reward_split_log: bool = ROUGH_REWARD_SPLIT_LOG_ENABLED,
    command_velocity_error_weight: float | None = ROUGH_COMMAND_VELOCITY_ERROR_WEIGHT,
    zero_base_height_on_terrain: bool = ROUGH_ZERO_BASE_HEIGHT_ON_TERRAIN,
    energy_penalty_scale: float = ROUGH_ENERGY_PENALTY_SCALE,
    terrain_curriculum: bool = True,
    terrain_command_override: bool = ROUGH_TERRAIN_COMMAND_OVERRIDE_ENABLED,
    terrain_lin_vel_x_range: tuple[float, float] = ROUGH_TERRAIN_LIN_VEL_X_RANGE,
    terrain_ang_vel_yaw_range: tuple[float, float] = ROUGH_TERRAIN_ANG_VEL_YAW_RANGE,
    terrain_lin_vel_x_follow_curriculum: bool = ROUGH_TERRAIN_LIN_VEL_X_FOLLOW_CURRICULUM,
    flat_curriculum_signal_only: bool = True,
    ctbc_enabled: bool = ROUGH_CTBC_ENABLED,
    ctbc_ann_start_iter: int = ROUGH_CTBC_ANN_START_ITER,
    ctbc_ann_end_iter: int = ROUGH_CTBC_ANN_END_ITER,
    critic_height_scan: bool = ROUGH_CRITIC_HEIGHT_SCAN_ENABLED,
    flat_warmup_iterations: int = ROUGH_FLAT_WARMUP_ITERATIONS,
    flat_warmup_ramp_iterations: int = ROUGH_FLAT_WARMUP_RAMP_ITERATIONS,
    terrain_vz_weight: float = ROUGH_TERRAIN_VZ_WEIGHT,
    curriculum_advance_threshold: float = ROUGH_CURRICULUM_ADVANCE_THRESHOLD,
    reset_last_action_range: float = ROUGH_RESET_LAST_ACTION_RANGE,
    reset_last_action_prob: float = ROUGH_RESET_LAST_ACTION_PROB,
    base_height_sigma: float | None = ROUGH_BASE_HEIGHT_SIGMA,
    off_stair_tracking_sigma_move: float | None = ROUGH_OFF_STAIR_TRACKING_SIGMA_MOVE,
    flat_vz_weight: float | None = ROUGH_FLAT_VZ_WEIGHT,
    command_velocity_error_terrain_names: tuple[str, ...] | None = (
        ROUGH_COMMAND_VELOCITY_ERROR_TERRAIN_NAMES
    ),
    amp_enabled: bool = False,
    amp_terrain_type_names: tuple[str, ...] = ROUGH_AMP_TERRAIN_TYPE_NAMES,
) -> ManagerBasedRlEnvCfg:
    """带地形课程与地形感知高度下限的崎岖地形环境配置。

    terrain_generator：None 时用 `rough_terrains_cfg()`；定向评测可传
    `stair_only_terrains_cfg()`。
    terrain_height_clearance：机体碰撞盒底面相对单级台阶顶部的最小余量(m)，
    高度指令的采样下界按 `台阶高 + 该余量 - body_collision_bottom_offset` 抬高；
    设 0 即关掉下限，高度指令退回 Flat 的 0.20–0.38 均匀采样。
    terrain_step_height_type_names：下限生效的子地形列名，默认只有上台阶列。
    reward_terrain_type_names：下面两项分列定价生效的子地形列名，默认只有上台阶列。
    stair_command_terrain_names：单独发高速/高站姿指令的子地形列名，空元组即关闭
    （这些列退回 terrain_lin_vel_x_range 与全局 height_range）。
    stair_lin_vel_x_range / stair_height_range / stair_ang_vel_yaw_range：
    台阶列的 vx、机身高度、yaw 角速度指令范围。
    zero_tracking_ang_vel_on_terrain：把 tracking_ang_vel 在台阶列置零（取消静止即可拿满的正奖励）。
    reward_split_log：每步记一份奖励表逐项在台阶列上的均值（Rough/rw_*_stairs），纯诊断。
    command_velocity_error_weight：只在这些列生效的速度违令二次罚权重；None 即不加该项
    （退回 Flat 基线，全线都没有它），见 ROUGH_COMMAND_VELOCITY_ERROR_WEIGHT 注释。
    zero_base_height_on_terrain：把 flat_base_height 在这些列上置零；关掉即全线同价。
    energy_penalty_scale：能耗类罚项相对 Flat 基线的折价系数，1.0 即与平地同价。
    terrain_curriculum：关掉后地形难度不再随表现提升（play 模式下恒为关）。
    terrain_command_override：非平地列只发前向直行指令（vx/yaw 范围见后两个参数），
    平地列沿用 Flat 速度课程；关掉即全部列都走 Flat 的对称随机指令。
    terrain_lin_vel_x_follow_curriculum：非平地列 vx 上限跟随平地速度课程当前上限。
    flat_curriculum_signal_only：平地速度课程只按平地列的跟踪分推进（R3 里全体均值被地形列拖住，
    平地列整场 vx=0）；关掉即退回 Flat 的全体均值判据。
    ctbc_enabled：接触触发的轮端抬升前馈（ctbc.py）。关掉后不加立面传感器、不挂状态机，
    actor 的 jump_commands 扩展槽退回 Flat 的实现（恒 0），观测维数与项名都不变。
    ctbc_ann_start_iter / ctbc_ann_end_iter：前馈退火起止轮次；play 模式下按 checkpoint 轮次
    由 play.py 固定。
    critic_height_scan：critic 加 77 点地形高度扫描特权观测（observations.height_scan_obs），
    actor 不变；关掉即 critic 只有原来的标量离地高度。
    flat_warmup_iterations：前 N 轮全部 env 在平地列（curriculums.flat_warmup），0 关闭。
    flat_warmup_ramp_iterations：热身结束后地形 env 比例从 0 线性涨到 1 所用的轮数，0 即一刀切。
    terrain_vz_weight：非平地列 tracking_lin_vel 核里的 vz 系数（平地列恒为 Flat 的 2.0）。
    curriculum_advance_threshold：Flat 速度课程推进阈值（默认与 Flat 相同 0.5；R7–A4 曾用 0.75）。
    amp_enabled：加 `amp` 观测组（AMP_DISCRIMINATOR_FIELDS 切列的运动帧，mdp/amp_observations.amp_motion_frame）与 `amp_mask` 观测组
    （env 是否在 amp_terrain_type_names 列上），只供 se3_train.amp 使用，actor/critic 不看它们。
    判别器与数据集在 rl_cfg 的 amp_cfg 里配。
    amp_terrain_type_names：AMP 生效的子地形列，默认只有上台阶列；空元组即全部 env。
    """
    cfg = flat_env_cfg(
        play=play,
        curriculum_advance_threshold=float(curriculum_advance_threshold),
        **_ROUGH_FLAT_BASELINE,  # type: ignore[arg-type]
    )

    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=terrain_generator or rough_terrains_cfg(),
        max_init_terrain_level=ROUGH_MAX_INIT_TERRAIN_LEVEL,
    )
    # 三个接触传感器的 secondary 都是 `pattern="terrain"`，平面地形只有一个 terrain geom，
    # 生成器地形有几百个，64 个匹配槽会溢出（运行时刷 "contact match overflow"，
    # 接触力读数不可信）。mjlab 自己的 rough velocity 任务同样取 500。
    # 只动这一个传感器缓冲区，不碰 solver/cone/impratio，避免与 Flat 基线产生物理差异。
    cfg.sim.contact_sensor_maxmatch = ROUGH_CONTACT_SENSOR_MAXMATCH

    # 专项奖励使用独立双轮传感器，不改变 Flat 原有高度/接触观测的布局。
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
    )
    cfg.events["reset_stair_rewards"] = EventTermCfg(
        func=stair_rewards.reset_stair_rewards, mode="reset"
    )
    if reset_last_action_range > 0.0:
        cfg.events["randomize_reset_last_actions"] = EventTermCfg(
            func=shared_events.randomize_reset_last_actions,
            mode="reset",
            params={
                "action_range": float(reset_last_action_range),
                "probability": float(reset_last_action_prob),
            },
        )
    cfg.rewards["stair_climb_progress"] = RewardTermCfg(
        func=stair_rewards.stair_climb_progress, weight=3.0
    )
    cfg.rewards["stair_support_height"] = RewardTermCfg(
        func=stair_rewards.stair_support_height, weight=4.0
    )

    cfg.commands = dict(cfg.commands)
    cfg.commands["velocity_height"] = _to_rough_command_cfg(
        cfg.commands["velocity_height"],
        terrain_aware_height=ROUGH_TERRAIN_AWARE_HEIGHT,
        terrain_height_clearance=float(terrain_height_clearance),
        body_collision_bottom_offset=ROUGH_BODY_COLLISION_BOTTOM_OFFSET,
        terrain_step_height_type_names=tuple(terrain_step_height_type_names),
        terrain_command_override_enabled=terrain_command_override,
        stair_command_terrain_names=tuple(stair_command_terrain_names),
        stair_lin_vel_x_range=tuple(stair_lin_vel_x_range),
        stair_height_range=tuple(stair_height_range),
        stair_ang_vel_yaw_range=tuple(stair_ang_vel_yaw_range),
        terrain_lin_vel_x_range=tuple(terrain_lin_vel_x_range),
        terrain_ang_vel_yaw_range=tuple(terrain_ang_vel_yaw_range),
        terrain_lin_vel_x_follow_curriculum=terrain_lin_vel_x_follow_curriculum,
    )

    if reward_split_log:
        cfg.events = dict(cfg.events)
        cfg.events["log_reward_split"] = EventTermCfg(
            func=events.log_reward_split_by_column,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={"terrain_type_names": tuple(reward_terrain_type_names)},
        )

    if flat_curriculum_signal_only:
        cfg.events = dict(cfg.events)
        cfg.events["set_curriculum_env_mask"] = EventTermCfg(
            func=events.set_curriculum_env_mask,
            mode="startup",
            params={"terrain_type_names": ROUGH_CURRICULUM_SIGNAL_TERRAIN_NAMES},
        )
        if not play and "command_vel" in cfg.curriculum:
            cfg.curriculum = dict(cfg.curriculum)
            params = dict(cfg.curriculum["command_vel"].params or {})
            params["tracking_log_key"] = ROUGH_CURRICULUM_TRACKING_LOG_KEY
            cfg.curriculum["command_vel"] = replace(cfg.curriculum["command_vel"], params=params)

    # 清掉本块台阶、走到边框即截断结算课程，不进邻块（邻块是另一行难度）。
    cfg.terminations = dict(cfg.terminations)
    cfg.terminations["terrain_cleared"] = TerminationTermCfg(
        func=terminations.terrain_cleared,
        time_out=True,
    )

    if ctbc_enabled:
        _add_ctbc(
            cfg,
            ann_start_iter=ctbc_ann_start_iter,
            ann_end_iter=ctbc_ann_end_iter,
        )

    if critic_height_scan:
        _add_critic_height_scan(cfg)

    if amp_enabled:
        cfg.observations = dict(cfg.observations)
        cfg.observations[ROUGH_AMP_OBS_GROUP] = ObservationGroupCfg(
            terms=build_amp_obs_terms(),
            concatenate_terms=True,
            enable_corruption=False,
        )
        cfg.observations[ROUGH_AMP_MASK_OBS_GROUP] = ObservationGroupCfg(
            terms=build_amp_mask_terms(amp_terrain_type_names),
            concatenate_terms=True,
            enable_corruption=False,
        )

    # 上台阶要更大的力矩与功率，沿用平地定价会把爬升直接压住。
    cfg.rewards = dict(cfg.rewards)
    for name in _ROUGH_ENERGY_REWARD_NAMES:
        term = cfg.rewards[name]
        term.weight = float(term.weight) * float(energy_penalty_scale)

    _apply_terrain_column_rewards(
        cfg,
        terrain_type_names=tuple(reward_terrain_type_names),
        command_velocity_error_weight=command_velocity_error_weight,
        zero_base_height=zero_base_height_on_terrain,
        terrain_vz_weight=terrain_vz_weight,
        zero_tracking_ang_vel=zero_tracking_ang_vel_on_terrain,
        base_height_sigma=base_height_sigma,
        off_stair_tracking_sigma_move=off_stair_tracking_sigma_move,
        flat_vz_weight=flat_vz_weight,
        command_velocity_error_terrain_names=command_velocity_error_terrain_names,
    )

    if not play and terrain_curriculum:
        cfg.curriculum = dict(cfg.curriculum)
        if int(flat_warmup_iterations) > 0:
            # 必须排在 terrain_levels 之前（同一次 reset 内先换列再结算课程）。
            cfg.curriculum = {
                "flat_warmup": CurriculumTermCfg(
                    func=curriculums.flat_warmup,
                    params={
                        "command_name": "velocity_height",
                        "iterations": int(flat_warmup_iterations),
                        "ramp_iterations": int(flat_warmup_ramp_iterations),
                        "steps_per_policy_iter": ROUGH_CTBC_STEPS_PER_POLICY_ITER,
                    },
                ),
                **cfg.curriculum,
            }
        cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
            func=curriculums.terrain_levels,
            params={"command_name": "velocity_height"},
        )

    return cfg


def _apply_terrain_column_rewards(
    cfg: ManagerBasedRlEnvCfg,
    *,
    terrain_type_names: tuple[str, ...],
    command_velocity_error_weight: float | None,
    zero_base_height: bool,
    terrain_vz_weight: float,
    zero_tracking_ang_vel: bool,
    base_height_sigma: float | None = None,
    off_stair_tracking_sigma_move: float | None = None,
    flat_vz_weight: float | None = None,
    command_velocity_error_terrain_names: tuple[str, ...] | None = None,
) -> None:
    """把台阶列的两处分列定价接进奖励表（见 rewards.py 的模块 docstring）。

    两项都只改生效范围：平地热身期（全员在平地列）与其余列的定价与 Flat 基线逐位相同。
    """
    if command_velocity_error_weight is not None:
        cfg.rewards["command_velocity_error"] = RewardTermCfg(
            func=rewards.command_velocity_error_on_terrain,
            weight=float(command_velocity_error_weight),
            params={
                "command_name": "velocity_height",
                "terrain_type_names": (
                    terrain_type_names
                    if command_velocity_error_terrain_names is None
                    else tuple(command_velocity_error_terrain_names)
                ),
                "lin_vel_scale": ROUGH_COMMAND_VELOCITY_ERROR_LIN_SCALE,
                "yaw_vel_scale": 1.0,
                "lin_deadband": float(FLAT_CMD_VEL_DEADBAND[0]),
                "yaw_deadband": float(FLAT_CMD_VEL_DEADBAND[1]),
                "max_penalty": 9.0,
            },
        )
    if zero_base_height:
        # 用 replace 而不是重建：sigma / max_error / 权重继续跟随 Flat 基线。
        term = cfg.rewards["flat_base_height"]
        height_params = {**term.params, "terrain_type_names": terrain_type_names}
        if base_height_sigma is not None:
            height_params["sigma"] = float(base_height_sigma)
        cfg.rewards["flat_base_height"] = replace(
            term,
            func=rewards.base_height_penalty_off_terrain,
            params=height_params,
        )
    if zero_tracking_ang_vel:
        # 同样用 replace：σ / sigma_cmd_scale / ratio_blend / 权重继续跟随 Flat 基线。
        ang = cfg.rewards["tracking_ang_vel"]
        cfg.rewards["tracking_ang_vel"] = replace(
            ang,
            func=rewards.tracking_ang_vel_off_terrain,
            params={**ang.params, "terrain_type_names": terrain_type_names},
        )
    # 非平地列关闭 vz 项；仅上台阶列放宽运动核，其余核参数与权重继续跟随 Flat 基线。
    track = cfg.rewards["tracking_lin_vel"]
    track_params = {
        **track.params,
        "terrain_vz_weight": float(terrain_vz_weight),
        "stair_sigma_move": ROUGH_STAIR_TRACKING_SIGMA_MOVE,
        "stair_type_names": terrain_type_names,
        "flat_type_names": ROUGH_VZ_FLAT_TERRAIN_TYPE_NAMES,
    }
    if off_stair_tracking_sigma_move is not None:
        # sigma_move 在 tracking_lin_vel_terrain_vz 里只作用于台阶列之外的列
        # （台阶列走 stair_sigma_move），正好是 A15 要改的那五列。
        track_params["sigma_move"] = float(off_stair_tracking_sigma_move)
    if flat_vz_weight is not None:
        # vz_weight 只作用于 flat_type_names 那一列；其余列已由 terrain_vz_weight 管。
        track_params["vz_weight"] = float(flat_vz_weight)
    cfg.rewards["tracking_lin_vel"] = replace(
        track,
        func=rewards.tracking_lin_vel_terrain_vz,
        params=track_params,
    )


def _add_critic_height_scan(cfg: ManagerBasedRlEnvCfg) -> None:
    """给 critic 加机身周围的地形高度扫描：一个网格射线传感器 + 一个只在 critic 组的观测项。"""
    scan_sensor = TerrainHeightSensorCfg(
        name=ROUGH_CRITIC_HEIGHT_SCAN_SENSOR_NAME,
        frame=ObjRef(type="body", name="base_link", entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(
            size=ROUGH_CRITIC_HEIGHT_SCAN_SIZE_M,
            resolution=ROUGH_CRITIC_HEIGHT_SCAN_RESOLUTION_M,
        ),
        max_distance=2.0,
        include_geom_groups=(0,),
        reduction="none",
    )
    cfg.scene.sensors = (*cfg.scene.sensors, scan_sensor)
    cfg.observations = dict(cfg.observations)
    critic = cfg.observations["critic"]
    terms = dict(critic.terms)
    terms["height_scan"] = ObservationTermCfg(
        func=observations.height_scan_obs,
        params={"sensor_name": ROUGH_CRITIC_HEIGHT_SCAN_SENSOR_NAME},
    )
    cfg.observations["critic"] = replace(critic, terms=terms)


def _add_ctbc(cfg: ManagerBasedRlEnvCfg, *, ann_start_iter: int, ann_end_iter: int) -> None:
    """接 CTBC：立面接触传感器、三个事件、3 维观测替换 jump_commands、last_actions 排除前馈。"""
    # 逐槽位带法向的轮-地形接触传感器：只有法向接近水平的接触才算顶住台阶立面。
    riser_sensor = ContactSensorCfg(
        name=ROUGH_CTBC_RISER_SENSOR_NAME,
        primary=ContactMatch(
            mode="body",
            pattern=r"^(l_wheel_Link|r_wheel_Link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force", "normal", "tangent"),
        reduce="maxforce",
        num_slots=4,
        global_frame=True,
    )
    cfg.scene.sensors = (*cfg.scene.sensors, riser_sensor)

    cfg.events = dict(cfg.events)
    cfg.events["init_ctbc_state"] = EventTermCfg(
        func=ctbc.init_ctbc_state,
        mode="startup",
        params={
            "contact_window": 3,
            "force_threshold_n": 10.0,
            "ff_amplitude_rad": 1.70,
            "ff_x_m": 0.02,
            "ff_lift_m": 0.02,
            "ff_period_s": 0.60,
            "ff_rise_ratio": 0.35,
            "ff_hold_ratio": 0.0,
            "ff_wheel_action": 0.0,
            "ff_start_iter": 0,
            "ann_start_iter": int(ann_start_iter),
            "ann_end_iter": int(ann_end_iter),
            "phantom_trigger_iter": 0,
            "allow_bilateral_trigger": False,
            "profile_path": None,
        },
    )
    cfg.events["step_ctbc_state"] = EventTermCfg(
        func=ctbc.step_ctbc_state,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "wheel_sensor_name": "wheel_sensor",
            "riser_sensor_name": ROUGH_CTBC_RISER_SENSOR_NAME,
            "riser_normal_z_max": 0.5,
            "num_steps_per_env": ROUGH_CTBC_STEPS_PER_POLICY_ITER,
            "terrain_type_names": ROUGH_CTBC_TERRAIN_TYPE_NAMES,
        },
    )
    cfg.events["reset_ctbc_state"] = EventTermCfg(func=ctbc.reset_ctbc_state, mode="reset")

    # 3 维扩展槽仍叫 jump_commands（部署契约 se3-sim2x policy_contract 只认这个名字，部署端填 0），
    # 只把实现换成 CTBC 相位/触发位——退火结束后训练侧也恒 0，与部署端语义一致。
    # last_actions 改为策略原始输出，不含注入的前馈（与 stair 线一致）。
    ctbc_term = ObservationTermCfg(func=ctbc.ctbc_obs)
    last_actions_term = ObservationTermCfg(func=stair_observations.last_actions_obs)
    cfg.observations = dict(cfg.observations)
    for group_name in ("actor", "critic"):
        group_cfg = cfg.observations[group_name]
        terms = dict(group_cfg.terms)
        assert "jump_commands" in terms, f"{group_name} 观测组缺少 jump_commands 扩展槽"
        terms["jump_commands"] = ctbc_term
        terms["last_actions"] = last_actions_term
        cfg.observations[group_name] = replace(group_cfg, terms=terms)


__all__ = [
    "ROUGH_ALL_TERRAIN_TYPE_NAMES",
    "ROUGH_AMP_MASK_OBS_GROUP",
    "ROUGH_AMP_OBS_GROUP",
    "ROUGH_AMP_TERRAIN_TYPE_NAMES",
    "ROUGH_BASE_HEIGHT_SIGMA",
    "ROUGH_BODY_COLLISION_BOTTOM_OFFSET",
    "ROUGH_COMMAND_VELOCITY_ERROR_LIN_SCALE",
    "ROUGH_COMMAND_VELOCITY_ERROR_TERRAIN_NAMES",
    "ROUGH_COMMAND_VELOCITY_ERROR_WEIGHT",
    "ROUGH_CONTACT_SENSOR_MAXMATCH",
    "ROUGH_CRITIC_HEIGHT_SCAN_ENABLED",
    "ROUGH_CRITIC_HEIGHT_SCAN_RESOLUTION_M",
    "ROUGH_CRITIC_HEIGHT_SCAN_SENSOR_NAME",
    "ROUGH_CRITIC_HEIGHT_SCAN_SIZE_M",
    "ROUGH_CTBC_ANN_END_ITER",
    "ROUGH_CTBC_ANN_START_ITER",
    "ROUGH_CTBC_ENABLED",
    "ROUGH_CTBC_TERRAIN_TYPE_NAMES",
    "ROUGH_CURRICULUM_ADVANCE_THRESHOLD",
    "ROUGH_CURRICULUM_SIGNAL_TERRAIN_NAMES",
    "ROUGH_CURRICULUM_TRACKING_LOG_KEY",
    "ROUGH_ENERGY_PENALTY_SCALE",
    "ROUGH_FLAT_VZ_WEIGHT",
    "ROUGH_FLAT_WARMUP_ITERATIONS",
    "ROUGH_FLAT_WARMUP_RAMP_ITERATIONS",
    "ROUGH_MAX_INIT_TERRAIN_LEVEL",
    "ROUGH_OFF_STAIR_TRACKING_SIGMA_MOVE",
    "ROUGH_RESET_LAST_ACTION_PROB",
    "ROUGH_RESET_LAST_ACTION_RANGE",
    "ROUGH_REWARD_SPLIT_LOG_ENABLED",
    "ROUGH_REWARD_TERRAIN_TYPE_NAMES",
    "ROUGH_STAIR_ANG_VEL_YAW_RANGE",
    "ROUGH_STAIR_COMMAND_TERRAIN_NAMES",
    "ROUGH_STAIR_HEIGHT_RANGE",
    "ROUGH_STAIR_LIN_VEL_X_RANGE",
    "ROUGH_TERRAIN_ANG_VEL_YAW_RANGE",
    "ROUGH_TERRAIN_AWARE_HEIGHT",
    "ROUGH_TERRAIN_COMMAND_OVERRIDE_ENABLED",
    "ROUGH_TERRAIN_HEIGHT_CLEARANCE",
    "ROUGH_TERRAIN_LIN_VEL_X_FOLLOW_CURRICULUM",
    "ROUGH_TERRAIN_LIN_VEL_X_RANGE",
    "ROUGH_TERRAIN_STEP_HEIGHT_TYPE_NAMES",
    "ROUGH_TERRAIN_VZ_WEIGHT",
    "ROUGH_VZ_FLAT_TERRAIN_TYPE_NAMES",
    "ROUGH_ZERO_BASE_HEIGHT_ON_TERRAIN",
    "ROUGH_ZERO_TRACKING_ANG_VEL_ON_TERRAIN",
    "env_cfg",
]
