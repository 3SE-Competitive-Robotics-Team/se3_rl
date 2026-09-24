from __future__ import annotations

from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from se3_shared import JointGroup, ObservationConfig
from se3_shared import RobotConfig as SharedRobotConfig
from se3_train.mdp.actions import SerialLegDelayedActionCfg
from se3_train.robot_cfg import get_serialleg_closedchain_cfg

from . import commands, curriculums, events, observations, rewards, terminations

_ROBOT_DEFAULTS = SharedRobotConfig()
_OBS_DEFAULTS = ObservationConfig()
# 观测噪声按原始物理单位定义，再乘以观测函数内部使用的缩放系数。mjlab 流水线是
# compute → noise → clip → scale，而本仓库在 obs func 内部完成缩放（term.scale 为 None），
# 所以 noise 必须与 func 输出同量纲，等价于 legged_gym 的 noise_scales × obs_scales。
# 2026-09-02 前直接写 0.2 / 1.5，等效原始噪声被放大到 ±0.8 rad/s 与 ±6 rad/s。
_ANG_VEL_NOISE_RAD_S = 0.2
_LEG_JOINT_VEL_NOISE_RAD_S = 1.5
_ANG_VEL_NOISE = _ANG_VEL_NOISE_RAD_S * _OBS_DEFAULTS.ang_vel_scale
_LEG_JOINT_VEL_NOISE = _LEG_JOINT_VEL_NOISE_RAD_S * _OBS_DEFAULTS.leg_vel_scale
_DEFAULT_STANDING_HEIGHT = _ROBOT_DEFAULTS.default_base_height
# 高度指令范围（有速度指令与站立 env 共用，全程无课程）。2026-09-05 由 0.20–0.32 改为 0.20–0.38：
# 轮心 x 取 -29.6 mm 时腿完全伸直对应 base 0.389 m，0.38 m 处主动杆夹角只剩 5°，是可达上限附近；
# v2 高度默认在 0.38 m 的整机质心残差 +5.2 mm（约 1.1°）。
_STANDING_HEIGHT_RANGE = (0.20, 0.38)
_FLAT_LEG_ACTION_SCALE = 0.25
# 轮 action scale：基类默认沿用 RobotConfig 的 45（rough/stair/jump/flow_match 继承线契约不变）；
# Flat 任务本身（flat/__init__.py 注册）显式使用 15，与 recovery 族一致（σ-gate 诊断：scale 45 时
# 探索噪声物理轮噪 ±σ×45 rad/s 挡死平衡校正）。改 scale 必须同步按 (scale/45)² 补偿动作空间罚项，
# 见 env_cfg() 内 wheel_pricing；指令可行域预算是物理量，与 scale 解耦（见下）。
_FLAT_LEGACY_WHEEL_ACTION_SCALE = float(_ROBOT_DEFAULTS.action_scale[JointGroup.WHEEL_ACTUATORS[0]])
FLAT_WHEEL_ACTION_SCALE = 15.0
_FLAT_DIFF_DRIVE_MAX_WHEEL_SPEED_RAD_S = 45.0
# 动作空间罚项在 scale 45 时代定标的权重；轮分量按 wheel_pricing 折算。
_FLAT_ACTION_RATE_WEIGHT = -0.48
# action_smoothness 重定价（2026-09-02）：Flat 线停在 job 51 之前的 -0.01/cap 80，弹簧 plant 上噪声失去隐式
# 代价后 cap 早饱和、对噪声零梯度，wheel σ 在 2000 轮内膨胀到 0.67-1.2（recovery 线以 -0.12/cap 320
# 在 4gs3te0p 把 σ 退火到 0.24）。这里对齐 recovery 线的物理定价：weight -0.12、cap 320、轮分量按腿的
# 2 倍计价再乘 wheel_pricing（scale 15 → 2.0/9≈0.222）。噪声地板 6σ² 为动作单位量，σ_leg 0.3 / σ_wheel 1.0
# 时合计约 5，远低于 cap，梯度不截断。
# (weight, max_penalty, wheel_base)；wheel_base 再乘 wheel_pricing 得到轮分量。基类默认 LEGACY 供
# rough/jump/flow_match 继承线保持旧契约，Flat 三个任务注册时显式传 SPRING。
FLAT_ACTION_SMOOTHNESS_LEGACY = (-0.01, 80.0, 1.0)
FLAT_ACTION_SMOOTHNESS_SPRING = (-0.12, 320.0, 2.0)
# History-MLP 变体的 actor 历史帧数（34 维 × 5 = 170 维展平），与 recovery_discovery 一致。
FLAT_HISTORY_LENGTH = 5
# 自适应课程从零起步，commands_vel_adaptive() 首次调用即覆写为 (0,0)
_FLAT_INITIAL_LIN_VEL_X_RANGE = (0.0, 0.0)
_FLAT_INITIAL_ANG_VEL_YAW_RANGE = (0.0, 0.0)
# 速度课程终值，也是写进 ONNX metadata 的部署包络（lin_vel_x）。
_FLAT_MAX_LIN_VEL_X = 2.4
_FLAT_COMMAND_WHEEL_RADIUS = 0.06
_FLAT_COMMAND_HALF_TRACK = 0.20
_FLAT_COMMAND_WHEEL_SPEED_FRACTION = 0.9
# 2026-09-03 抖动对照实验的单变量旋钮。默认值 = 实验前基线（commit 9dbfe11），
# 旧 Flat-Exp 注册入口已移除，保留共享配置旋钮供继承任务使用。
# 诊断依据：日志逐项预算里 command_velocity_error 是最大单项罚（占总罚 38%）且 2k 后不再下降，
# flat_wheel_contact 是唯一越训越差的项（轮离地率 3%→6.6%），bad_tilt soft 10° 起价太晚
# （实测平均倾角已在 12° 附近），action delay 只有 4-6 ms 不到一个控制步，yaw 课程顶到 12 rad/s。
# (lin m/s, yaw rad/s) command_velocity_error 死区
FLAT_CMD_VEL_DEADBAND = (0.05, 0.10)
FLAT_CMD_VEL_DEADBAND_WIDE = (0.15, 0.30)
FLAT_WHEEL_CONTACT_WEIGHT = -10.0
FLAT_WHEEL_CONTACT_WEIGHT_HEAVY = -30.0
# (soft_limit_deg, hard_limit_deg) bad_tilt barrier
FLAT_BAD_TILT_LIMITS_DEG = (10.0, 30.0)
FLAT_BAD_TILT_LIMITS_TIGHT_DEG = (6.0, 25.0)
# None = 沿用 ActionDelayConfig 默认（4-6 ms）；元组为 (min_s, max_s)，delay_s 取中点。
FLAT_ACTION_DELAY_RANGE_S: tuple[float, float] | None = None
# 1-3 个控制步 @50 Hz，对齐 scutrobotlab/wheeled-legged_RL 的动作延迟量级。
FLAT_ACTION_DELAY_RANGE_ONE_TO_THREE_STEPS_S = (0.020, 0.060)
FLAT_MAX_ANG_VEL_YAW = 12.0
FLAT_MAX_ANG_VEL_YAW_LOW = 6.0
# 2026-09-04 C 批：yaw 上限保持 12（真实需求），改修课程的爬升方式。
# 诊断：自适应课程在前 200 轮就把 yaw 顶到 9，五个 run 的存活率随即从 0.70 崩到 0.12-0.18，
# 而课程只扩不缩、锁死在 9.0。三处缺陷分别对应下面三组常量。
FLAT_CURRICULUM_ANG_VEL_YAW_STEP = 1.0
FLAT_CURRICULUM_ANG_VEL_YAW_STEP_FINE = 0.25
FLAT_CURRICULUM_ADVANCE_THRESHOLD = 0.5
FLAT_CURRICULUM_ADVANCE_THRESHOLD_STRICT = 0.75
# yaw 上限改由 yaw 跟踪 EMA 独立驱动；此前它由线速度跟踪分推进，与 yaw 能力无关。
FLAT_CURRICULUM_YAW_GATE = False
# EMA 跌破 retreat_threshold 时回退一步（滞回），治的是冲过头之后无法退回。
FLAT_CURRICULUM_RETREAT = False
# 腿部 action 语义。默认沿用旧契约；joint 语义下四维直接是四根主动杆的绝对目标角，
# 没有夹角这个中间量，也没有解码器夹紧（隐含夹角越界交给 MJCF 的 tendon 限位承接）。
# 改这个会改变 ONNX 契约，必须重训，旧 checkpoint 与新 sim2x 不可混用。
# 2026-09-06 起 joint 成为 Flat 默认：D2 以来的全部 D 系列实验都建立在该语义上，部署契约
# serialleg_joint.v1 已在 sim2x 落地。旧 active_rod 只作为继承线的历史取值保留。
FLAT_LEG_ACTION_SEMANTICS: Literal["active_rod", "joint"] = "joint"
# 动作罚项（action_rate / action_smoothness）轮分量的定价基准。None = 按 (wheel_action_scale/45)² 折算，
# 即"同一物理轮速轨迹的罚款与 scale 无关"，是 Flat 三任务与 Exp-* 的基线。
# 2026-09-05 σ 诊断：σ 按归一化动作维度学习，熵奖励 entropy_coef 也按归一化维度给，不随折价缩放。
# 轮分量折到 1/9 后，一单位归一化轮噪声的折现代价 k_wheel=0.0051，只有腿 k_leg=0.032 的 1/6.3；
# 平衡点 σ = sqrt(entropy_coef·std_A / 2k)（std_A≈sqrt(Loss/value)≈1.4）给出轮 0.85 / 腿 0.34，
# 与 D2/A0 实测轮 0.90-0.98、腿 0.25-0.30 一致；收敛后 action_rate 的 72-81%、action_smoothness 的
# 91-104% 都是纯探索噪声地板。同一 entropy_coef、同一 -0.12/cap 320 但轮分量 2.0 的 4gs3te0p，σ 退火到 0.23。
# 1.0 = 轮分量按归一化动作单位计价：action_rate 轮 1.0、action_smoothness 轮 2.0，预测轮 σ 平衡点 0.28。
# 2026-09-06 起 1.0 成为 Flat 默认：D4 对 D2 实测轮 σ 0.922 → 0.256、腿 σ 0.285 → 0.212，
# mean_reward 60.2 → 81.8，速度/yaw 跟踪、违令罚、轮离地罚、姿态项全部更好，存活持平。
FLAT_ACTION_PENALTY_WHEEL_PRICING: float | None = 1.0
FLAT_ACTION_PENALTY_WHEEL_PRICING_UNIT = 1.0
# 折价基准的历史取值（(wheel_action_scale/45)²），供继承线与旧实验入口显式引用。
FLAT_ACTION_PENALTY_WHEEL_PRICING_LEGACY: float | None = None
# tracking_orientation_l2 权重。2026-09-05 腿部摆动诊断：D4 确定性策略站立时有 0.67 Hz 极限环（腿峰峰 24°、
# 俯仰 rms 2.2°、横滚 1.9°），训练模拟器里带 σ 采样与观测噪声时和 D2 分不开，只有确定性 rollout 才露出来。
# -12 下这段慢摆只花 0.03/s 等于免费；-120 时 0.31/s 与动作罚项同量级，D2 式安静站立仍只花 0.03/s，
# 行进俯仰 0.3-1.4° 不受影响。该项与用腿还是用轮做平衡无关，直接压机身晃动。
FLAT_TRACKING_ORIENTATION_WEIGHT = -12.0
FLAT_TRACKING_ORIENTATION_WEIGHT_STRONG = -120.0
# joint_pos_penalty（腿关节偏离高度条件默认姿态的 L2 范数，直立门控、始终生效，静止时 ×5）权重。
# None = 不加，Flat 基线只有指令为零时才生效的 stand_still。-1.0 与 recovery / recovery_discovery / stair 三条线相同。
# 2026-09-05 腿部摆动诊断的量级：D4 确定性站立慢摆 ||Δq|| 均值 0.32 rad，×5 后 1.6/s；D2 式安静站立 0.04 rad，0.18/s；
# 行进时 D4 0.49/s、D2 0.30/s。训练条件（σ 采样 + 观测噪声 + 域随机化）下两者都约 1.9/s，该项会同时把两种策略往默认姿态推。
FLAT_JOINT_POS_PENALTY_WEIGHT: float | None = None
FLAT_JOINT_POS_PENALTY_WEIGHT_RECOVERY_LINE = -1.0
# command_velocity_error（速度违令二次罚，死区外 (|err|-db)²/scale² 封顶 9）权重；None = 删除该项。
# 2026-09-05 诊断：tracking_lin_vel 的高斯核 σ=0.08 在误差 >0.4 m/s 时无梯度，该项本意是补远处梯度（52ba696）；
# 但 D4 确定性评测拆分显示它 99% 的代价来自每次指令阶跃后 1 s 内（2.4 m/s / 12 rad/s 阶跃按额定扭矩至少
# 0.4-1 s 才能跟上，二次项直接顶到封顶），稳态只有 0.003/s；训练里平均 -1.44/s 是最大单项罚，
# 等于奖励指令跳变后猛冲（高增益）。此前 A1 只放宽死区，只动了稳态那 1%。
# 2026-09-06 起删除成为 Flat 默认：D7 对 D4 实测跟踪分更高（lin 2.91/2.77、ang 2.68/2.66）、
# 动作罚减半（action_rate -0.195/-0.434）、轮离地与姿态项更好、存活持平，且学习率全程未贴地板。
FLAT_COMMAND_VELOCITY_ERROR_WEIGHT: float | None = None
# 该项的历史权重，供仍需要它的实验入口（速度死区对照）显式引用。
FLAT_COMMAND_VELOCITY_ERROR_WEIGHT_LEGACY: float | None = -2.0


def _action_delay_kwargs(action_delay_range_s: tuple[float, float] | None) -> dict[str, object]:
    """把 (min_s, max_s) 展开成 SerialLegDelayedActionCfg 的延迟字段；None 表示沿用默认。"""
    if action_delay_range_s is None:
        return {}
    min_s, max_s = (float(action_delay_range_s[0]), float(action_delay_range_s[1]))
    if min_s > max_s:
        raise ValueError(f"action_delay_range_s 需满足 min <= max，收到 {action_delay_range_s}")
    return {
        "action_delay_enabled": True,
        "action_delay_randomize": True,
        "action_delay_min_s": min_s,
        "action_delay_max_s": max_s,
        "action_delay_s": 0.5 * (min_s + max_s),
    }


def env_cfg(
    play: bool = False,
    *,
    wheel_action_scale: float = _FLAT_LEGACY_WHEEL_ACTION_SCALE,
    action_smoothness: tuple[float, float, float] = FLAT_ACTION_SMOOTHNESS_LEGACY,
    command_velocity_deadband: tuple[float, float] = FLAT_CMD_VEL_DEADBAND,
    flat_wheel_contact_weight: float = FLAT_WHEEL_CONTACT_WEIGHT,
    bad_tilt_limits_deg: tuple[float, float] = FLAT_BAD_TILT_LIMITS_DEG,
    action_delay_range_s: tuple[float, float] | None = FLAT_ACTION_DELAY_RANGE_S,
    max_ang_vel_yaw: float = FLAT_MAX_ANG_VEL_YAW,
    curriculum_ang_vel_yaw_step: float = FLAT_CURRICULUM_ANG_VEL_YAW_STEP,
    curriculum_advance_threshold: float = FLAT_CURRICULUM_ADVANCE_THRESHOLD,
    curriculum_yaw_gate: bool = FLAT_CURRICULUM_YAW_GATE,
    curriculum_retreat: bool = FLAT_CURRICULUM_RETREAT,
    leg_action_semantics: Literal["active_rod", "joint"] = FLAT_LEG_ACTION_SEMANTICS,
    action_penalty_wheel_pricing: float | None = FLAT_ACTION_PENALTY_WHEEL_PRICING,
    tracking_orientation_weight: float = FLAT_TRACKING_ORIENTATION_WEIGHT,
    joint_pos_penalty_weight: float | None = FLAT_JOINT_POS_PENALTY_WEIGHT,
    command_velocity_error_weight: float | None = FLAT_COMMAND_VELOCITY_ERROR_WEIGHT,
) -> ManagerBasedRlEnvCfg:
    """SerialLeg 轮腿机器人的平地环境配置。

    wheel_action_scale：轮 raw action → 轮速目标（rad/s）的 scale。默认 45 供继承线沿用旧契约；
    Flat 任务注册时传 FLAT_WHEEL_ACTION_SCALE=15。动作空间罚项（action_rate / action_smoothness）
    的轮分量按 (wheel_action_scale/45)² 折算，保证同一物理轮速轨迹的罚款与 scale 无关。
    action_smoothness：(weight, max_penalty, wheel_base)，见 FLAT_ACTION_SMOOTHNESS_* 注释。
    command_velocity_deadband / flat_wheel_contact_weight / bad_tilt_limits_deg /
    action_delay_range_s / max_ang_vel_yaw：抖动对照实验的单变量旋钮，默认即基线，
    见 FLAT_CMD_VEL_DEADBAND 等常量的注释。
    curriculum_*：速度课程爬升方式的单变量旋钮，默认即基线，见 FLAT_CURRICULUM_* 常量。
    leg_action_semantics：腿部 action 语义，见 FLAT_LEG_ACTION_SEMANTICS 注释。
    action_penalty_wheel_pricing：动作罚项轮分量的定价基准，None 即 (wheel_action_scale/45)²，
    见 FLAT_ACTION_PENALTY_WHEEL_PRICING 注释。
    tracking_orientation_weight：机身姿态 L2 罚权重，见 FLAT_TRACKING_ORIENTATION_WEIGHT 注释。
    joint_pos_penalty_weight：腿姿态回默认罚权重，None 不加，见 FLAT_JOINT_POS_PENALTY_WEIGHT 注释。
    command_velocity_error_weight：速度违令罚权重，None 删除该项，见 FLAT_COMMAND_VELOCITY_ERROR_WEIGHT 注释。
    """
    smooth_weight, smooth_cap, smooth_wheel_base = action_smoothness
    cmd_lin_deadband, cmd_yaw_deadband = command_velocity_deadband
    bad_tilt_soft_deg, bad_tilt_hard_deg = bad_tilt_limits_deg
    # 同一物理轮速轨迹：动作幅值 ×(45/scale)、差分平方 ×(45/scale)²，权重乘以其倒数保持定价。
    # 显式传入时改按该值计价（σ 平衡点实验，见 FLAT_ACTION_PENALTY_WHEEL_PRICING 注释）。
    if action_penalty_wheel_pricing is None:
        wheel_pricing = (float(wheel_action_scale) / _FLAT_LEGACY_WHEEL_ACTION_SCALE) ** 2
    else:
        wheel_pricing = float(action_penalty_wheel_pricing)

    scene = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": get_serialleg_closedchain_cfg()},
        num_envs=1024,
        env_spacing=3.0,
    )

    cfg = ManagerBasedRlEnvCfg(
        decimation=_ROBOT_DEFAULTS.control_decimation,
        scene=scene,
    )

    collision_sensor_cfg = ContactSensorCfg(
        name="collision_sensor",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(base_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )

    leg_contact_sensor_cfg = ContactSensorCfg(
        name="leg_contact_sensor",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(lf0_Link|lf1_Link|rf0_Link|rf1_Link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )

    wheel_sensor_cfg = ContactSensorCfg(
        name="wheel_sensor",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(l_wheel_Link|r_wheel_Link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )

    base_height_sensor_cfg = TerrainHeightSensorCfg(
        name="base_height_sensor",
        frame=ObjRef(type="body", name="base_link", entity="robot"),
        ray_alignment="yaw",
        pattern=RingPatternCfg.single_ring(radius=0.05, num_samples=4),
        max_distance=2.0,
        include_geom_groups=(0,),
        reduction="min",
    )

    # 轮子离地高度传感器:从左轮 body 向下打射线,测量真实轮子离地距离
    # 用于 jump_wheel_clr_tracking,防止策略通过收腿套利 base_link 高度
    wheel_height_sensor_cfg = TerrainHeightSensorCfg(
        name="wheel_height_sensor",
        frame=ObjRef(type="body", name="l_wheel_Link", entity="robot"),
        ray_alignment="yaw",
        pattern=RingPatternCfg.single_ring(radius=0.01, num_samples=4),
        max_distance=2.0,
        include_geom_groups=(0,),
        reduction="min",
    )

    critic_height_sensor_cfg = TerrainHeightSensorCfg(
        name="critic_height_sensor",
        frame=ObjRef(type="body", name="base_link", entity="robot"),
        ray_alignment="yaw",
        pattern=RingPatternCfg.single_ring(radius=0.15, num_samples=8),
        max_distance=2.0,
        include_geom_groups=(0,),
        reduction="mean",
    )

    cfg.scene.sensors = (
        collision_sensor_cfg,
        leg_contact_sensor_cfg,
        wheel_sensor_cfg,
        base_height_sensor_cfg,
        critic_height_sensor_cfg,
        wheel_height_sensor_cfg,
    )

    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=observations.base_ang_vel_obs,
            noise=Unoise(n_min=-_ANG_VEL_NOISE, n_max=_ANG_VEL_NOISE),
        ),
        "projected_gravity": ObservationTermCfg(
            func=observations.projected_gravity_obs,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "commands": ObservationTermCfg(func=observations.commands_obs),
        "leg_joint_pos": ObservationTermCfg(
            func=observations.leg_joint_pos_obs,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "leg_joint_vel": ObservationTermCfg(
            func=observations.leg_joint_vel_obs,
            noise=Unoise(n_min=-_LEG_JOINT_VEL_NOISE, n_max=_LEG_JOINT_VEL_NOISE),
        ),
        "wheel_pos_zero": ObservationTermCfg(func=observations.wheel_pos_obs),
        "wheel_vel": ObservationTermCfg(func=observations.wheel_vel_obs),
        "last_actions": ObservationTermCfg(func=observations.last_actions_obs),
        "jump_commands": ObservationTermCfg(func=observations.jump_commands_obs),
    }

    critic_terms = {
        **actor_terms,
        "base_lin_vel": ObservationTermCfg(func=observations.base_lin_vel_obs),
        "wheel_contact_forces": ObservationTermCfg(
            func=observations.wheel_contact_force_obs,
            params={"sensor_name": "wheel_sensor"},
        ),
        "base_height": ObservationTermCfg(
            func=observations.base_height_obs,
            params={"sensor_name": "critic_height_sensor"},
        ),
        "knee_gas_spring_force": ObservationTermCfg(
            func=observations.knee_gas_spring_force_obs,
        ),
        # ---- critic 特权包 v2（CTS RA-L 2024 特权集合 + 本仓库 DR 参数回读）----
        "motor_torques": ObservationTermCfg(func=observations.motor_torque_obs),
        "joint_acc": ObservationTermCfg(func=observations.joint_acc_obs),
        "leg_contact_forces": ObservationTermCfg(
            func=observations.contact_force_norm_obs,
            params={"sensor_name": "leg_contact_sensor"},
        ),
        "base_collision_force": ObservationTermCfg(
            func=observations.contact_force_norm_obs,
            params={"sensor_name": "collision_sensor"},
        ),
        "dr_model_params": ObservationTermCfg(func=observations.dr_model_params_obs),
    }

    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=not play,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    cfg.actions = {
        "delayed_action": SerialLegDelayedActionCfg(
            entity_name="robot",
            leg_scales=(_FLAT_LEG_ACTION_SCALE,) * 4,
            wheel_scale=float(wheel_action_scale),
            action_clip=_ROBOT_DEFAULTS.action_clip,
            leg_action_semantics=leg_action_semantics,
            **_action_delay_kwargs(action_delay_range_s),
        ),
    }

    cfg.commands = {
        "velocity_height": commands.JumpCommandCfg(
            resampling_time_range=(5.0, 5.0),
            jump_prob=0.0,  # 行走任务不触发跳跃
            # 行走线不看 Jump/* 诊断，开着时每步 184 次 .item() 占一步的 18%
            # （2026-09-13 实测 7.4 ms）；recovery / stair 线早已关闭。要看时再开，实现已改成无主机同步。
            enable_jump_metrics=False,
            lin_vel_x_range=_FLAT_INITIAL_LIN_VEL_X_RANGE,
            ang_vel_yaw_range=_FLAT_INITIAL_ANG_VEL_YAW_RANGE,
            height_range=_STANDING_HEIGHT_RANGE,
            standing_height_range=_STANDING_HEIGHT_RANGE,
            constrain_diff_drive_commands=True,
            diff_drive_wheel_radius=_FLAT_COMMAND_WHEEL_RADIUS,
            diff_drive_half_track=_FLAT_COMMAND_HALF_TRACK,
            # 指令可行域预算是物理量（M3508 能力量级 rad/s），与动作 scale 解耦；
            # 若跟随 scale 改成 15 会把 vx/yaw 指令域错误压缩到 1/3。
            diff_drive_max_wheel_speed=_FLAT_DIFF_DRIVE_MAX_WHEEL_SPEED_RAD_S,
            diff_drive_wheel_speed_fraction=_FLAT_COMMAND_WHEEL_SPEED_FRACTION,
        ),
    }
    # 写进 ONNX metadata 的最终 command 包络（课程终值）。不声明时部署端会退回旧兼容边界
    # （height 0.20–0.32 等），sim2x 会拒绝 0.32 m 以上的高度指令。只影响 metadata 与
    # 部署端输入校验/控件范围，不影响训练采样。
    _flat_command_cfg = cfg.commands["velocity_height"]
    _flat_command_cfg.deployment_ranges = {
        "lin_vel_x": (-_FLAT_MAX_LIN_VEL_X, _FLAT_MAX_LIN_VEL_X),
        "ang_vel_yaw": (-float(max_ang_vel_yaw), float(max_ang_vel_yaw)),
        "pitch": tuple(_flat_command_cfg.pitch_range),
        "roll": tuple(_flat_command_cfg.roll_range),
        "height": _STANDING_HEIGHT_RANGE,
        "jump_flag": (0.0, 0.0),
        "jump_target_height": (0.0, 0.0),
        "jump_phase": (0.0, 0.0),
    }
    cfg.rewards = {
        "tracking_lin_vel": RewardTermCfg(
            func=rewards.tracking_lin_vel,
            weight=4.0,
            params={
                "command_name": "velocity_height",
                "sigma_move": 0.08,
                "sigma_stand": 0.1,
                "vz_weight": 2.0,
                "use_upright_gate": False,
            },
        ),
        "tracking_ang_vel": RewardTermCfg(
            func=rewards.tracking_ang_vel,
            # 高 yaw 组合命令在大误差时 exp 核梯度会消失,需要保留方向性学习信号。
            weight=3.0,
            params={
                "command_name": "velocity_height",
                "sigma": 0.25,
                "sigma_cmd_scale": 0.4,
                "ratio_blend": 0.2,
                "use_upright_gate": False,
            },
        ),
        "tracking_lin_yaw_joint": RewardTermCfg(
            func=rewards.tracking_lin_yaw_joint,
            # 组合命令必须同时保住线速度和 yaw,避免策略只完成其中一轴。
            weight=2.0,
            params={
                "command_name": "velocity_height",
                "min_lin_cmd": 0.2,
                "min_yaw_cmd": 0.5,
                "lin_error_scale": 0.75,
                "yaw_error_scale": 2.0,
                "use_upright_gate": False,
            },
        ),
        "command_velocity_error": RewardTermCfg(
            func=rewards.command_velocity_error,
            weight=float(command_velocity_error_weight or 0.0),
            params={
                "command_name": "velocity_height",
                "lin_vel_scale": 0.5,
                "yaw_vel_scale": 1.0,
                "lin_deadband": float(cmd_lin_deadband),
                "yaw_deadband": float(cmd_yaw_deadband),
                "max_penalty": 9.0,
            },
        ),
        # 姿态相关项只使用惩罚语义:偏离目标姿态扣分,明显倾斜加重扣分。
        "tracking_orientation_l2": RewardTermCfg(
            func=rewards.tracking_orientation_l2,
            weight=float(tracking_orientation_weight),
            params={"command_name": "velocity_height"},
        ),
        "flat_base_height": RewardTermCfg(
            func=rewards.flat_base_height_penalty_no_jump,
            weight=-4.0,
            params={
                "command_name": "velocity_height",
                "sigma": 0.05,
                "height_sensor_name": "base_height_sensor",
            },
        ),
        "bad_tilt": RewardTermCfg(
            func=rewards.bad_tilt,
            weight=-6.0,
            params={
                "soft_limit_deg": float(bad_tilt_soft_deg),
                "hard_limit_deg": float(bad_tilt_hard_deg),
                "max_penalty": 4.0,
            },
        ),
        "ang_vel_xy": RewardTermCfg(func=rewards.ang_vel_xy, weight=-0.146),
        "angular_momentum": RewardTermCfg(
            func=rewards.angular_momentum,
            weight=-5.0e-5,
        ),
        "leg_torques": RewardTermCfg(
            func=rewards.leg_torques,
            weight=-2.0e-4,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "wheel_torques": RewardTermCfg(
            func=rewards.wheel_torques,
            weight=-1.0e-4,
            params={"max_torque": 3.0, "asset_cfg": SceneEntityCfg("robot")},
        ),
        "stand_still": RewardTermCfg(
            func=rewards.stand_still,
            weight=-1.0,
            params={
                "command_name": "velocity_height",
                "command_threshold": 0.1,
                "default_height": _DEFAULT_STANDING_HEIGHT,
                "height_tolerance": 40.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "leg_dof_acc": RewardTermCfg(
            func=rewards.leg_dof_acc,
            weight=-2.17e-7,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "leg_power": RewardTermCfg(
            func=rewards.leg_power,
            weight=-1.03e-4,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "action_rate": RewardTermCfg(
            func=rewards.action_rate,
            weight=_FLAT_ACTION_RATE_WEIGHT,
            # 轮分量按 wheel_pricing 折算（scale 15 → 1/9），腿分量不变。
            params={"leg_scale": 1.0, "wheel_scale": wheel_pricing},
        ),
        "action_smoothness": RewardTermCfg(
            func=rewards.action_smoothness,
            weight=float(smooth_weight),
            params={
                "command_name": "velocity_height",
                "max_penalty": float(smooth_cap),
                "leg_scale": 1.0,
                # 轮分量 = wheel_base × wheel_pricing（SPRING @ scale 15 → 2.0/9 ≈ 0.222，与 recovery 线同价）。
                "wheel_scale": float(smooth_wheel_base) * wheel_pricing,
            },
        ),
        "joint_mirror": RewardTermCfg(
            func=rewards.joint_mirror,
            weight=-0.179,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "dof_pos_limits": RewardTermCfg(
            func=rewards.dof_pos_limits,
            weight=-5.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "collision": RewardTermCfg(
            func=rewards.collision,
            weight=-16.0,
            params={"sensor_name": "collision_sensor", "asset_cfg": SceneEntityCfg("robot")},
        ),
        "contact_forces": RewardTermCfg(
            func=rewards.contact_forces,
            weight=-1.07e-3,
            params={
                "threshold": 35.0,
                "sensor_name": "wheel_sensor",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "flat_wheel_contact": RewardTermCfg(
            func=rewards.flat_wheel_contact_penalty,
            weight=float(flat_wheel_contact_weight),
            params={
                "command_name": "velocity_height",
                "sensor_name": "wheel_sensor",
                "force_threshold": 1.0,
            },
        ),
        "flat_leg_contact": RewardTermCfg(
            func=rewards.flat_leg_contact_penalty,
            weight=-25.0,
            params={
                "command_name": "velocity_height",
                "sensor_name": "leg_contact_sensor",
                "force_threshold": 1.0,
            },
        ),
        # 业界标准:移除显式 termination 惩罚,改用 alive reward 隐式机制
        # 摔倒 → episode 结束 → 损失后续所有 alive 累积(隐式 penalty ~= alive * remaining_steps)
        # ETH/Unitree/CMU 所有框架 termination weight = 0,这是行业共识
        "is_alive": RewardTermCfg(func=rewards.is_alive, weight=1.0),
    }
    if command_velocity_error_weight is None:
        del cfg.rewards["command_velocity_error"]
    if joint_pos_penalty_weight is not None:
        # 参数与 recovery / recovery_discovery 线完全一致。
        cfg.rewards["joint_pos_penalty"] = RewardTermCfg(
            func=rewards.joint_pos_penalty,
            weight=float(joint_pos_penalty_weight),
            params={
                "command_name": "velocity_height",
                "stand_still_scale": 5.0,
                "velocity_threshold": 0.5,
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    cfg.terminations = {
        "time_out": TerminationTermCfg(func=terminations.time_out, time_out=True),
        "catastrophic_state": TerminationTermCfg(
            func=terminations.catastrophic_state,
            time_out=False,
            params={
                "max_leg_pos_error": 3.0,
                "max_leg_vel": 120.0,
                "max_root_lin_vel": 80.0,
                "max_root_ang_vel": 500.0,
                "min_base_height": -0.5,
                "max_base_height": 3.0,
            },
        ),
        "bad_orientation": TerminationTermCfg(
            func=terminations.bad_orientation_delayed,
            time_out=False,
            params={"limit_angle": 0.5236, "max_steps": 100},
        ),
        "leg_contact": TerminationTermCfg(
            func=terminations.leg_contact,
            time_out=False,
            params={
                "sensor_name": "leg_contact_sensor",
                "force_threshold": 1.0,
                "command_name": "velocity_height",
                "terminate": False,
            },
        ),
    }

    if not play:
        cfg.curriculum = {
            "command_vel": CurriculumTermCfg(
                func=curriculums.commands_vel_adaptive,
                params={
                    "command_name": "velocity_height",
                    "lin_vel_x_step": 0.2,
                    "ang_vel_yaw_step": float(curriculum_ang_vel_yaw_step),
                    "max_lin_vel_x": _FLAT_MAX_LIN_VEL_X,
                    "max_ang_vel_yaw": float(max_ang_vel_yaw),
                    "init_lin_vel_x": 0.0,
                    "init_ang_vel_yaw": 0.0,
                    "advance_threshold": float(curriculum_advance_threshold),
                    "ema_alpha": 0.05,
                    "yaw_gate_enabled": bool(curriculum_yaw_gate),
                    "yaw_advance_threshold": float(curriculum_advance_threshold),
                    "retreat_enabled": bool(curriculum_retreat),
                },
            ),
            "push_disturbance": CurriculumTermCfg(
                func=curriculums.push_disturbance,
                params={
                    "use_iterations": True,
                    "push_stages": [
                        {
                            "step": 0,
                            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
                        },
                        {
                            "step": 2000,
                            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
                        },
                        {
                            "step": 5000,
                            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
                        },
                        {
                            "step": 10000,
                            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
                        },
                        {
                            "step": 20000,
                            "velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)},
                        },
                        {
                            "step": 40000,
                            "velocity_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0)},
                        },
                    ],
                },
            ),
        }

    if play:
        cfg.commands["velocity_height"].pitch_range = (0.0, 0.0)
        cfg.commands["velocity_height"].roll_range = (0.0, 0.0)
        cfg.commands["velocity_height"].lin_vel_x_range = (-1.0, 1.0)
        cfg.commands["velocity_height"].ang_vel_yaw_range = (0.0, 0.0)
        cfg.events = {
            "reset_scene_to_default": EventTermCfg(
                func=lambda env, env_ids: None,
                mode="reset",
            ),
            "reset_root_state": EventTermCfg(
                func=events.reset_root_state_full,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
            "reset_joints": EventTermCfg(
                func=events.reset_joints,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "align_root_height_to_wheels": True,
                    "wheel_clearance": 0.001,
                    "full_joint_randomization": True,
                    "full_front_joint_offset_range": 1.57,  # ±90° hip 随机化
                },
            ),
        }
        cfg.episode_length_s = 9999.0
    else:
        cfg.events = {
            "reset_scene_to_default": EventTermCfg(
                func=lambda env, env_ids: None,
                mode="reset",
            ),
            "reset_root_state": EventTermCfg(
                func=events.reset_root_state_full,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
            "reset_joints": EventTermCfg(
                func=events.reset_joints,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "align_root_height_to_wheels": True,
                    "wheel_clearance": 0.001,
                    "full_joint_randomization": True,
                    "full_front_joint_offset_range": 1.57,  # ±90° hip 随机化
                },
            ),
            "friction": EventTermCfg(
                func=events.randomize_friction,
                mode="startup",
                params={"friction_range": (0.2, 1.5), "asset_cfg": SceneEntityCfg("robot")},
            ),
            "restitution": EventTermCfg(
                func=events.randomize_restitution,
                mode="startup",
                params={"restitution_range": (0.0, 0.5), "asset_cfg": SceneEntityCfg("robot")},
            ),
            # 2026-08-30 发现并修复：这三个事件曾因 body id 硬编码 0（world）自 mjlab
            # 移植起从未生效（4gs3te0p 及更早 run 的 plant 均无这三项 DR）。id 解析修复后
            # 按用户决定恢复原范围启用（2026-08-31 critic v2 实验起生效）。
            "base_mass": EventTermCfg(
                func=events.randomize_base_mass,
                mode="startup",
                params={"mass_range": (-0.5, 1.5), "asset_cfg": SceneEntityCfg("robot")},
            ),
            "inertia": EventTermCfg(
                func=events.randomize_inertia,
                mode="startup",
                params={"inertia_range": (0.8, 1.2), "asset_cfg": SceneEntityCfg("robot")},
            ),
            "com": EventTermCfg(
                func=events.randomize_com,
                mode="startup",
                # ±5cm 对 7kg/40cm 级机身占比过大，7lxhzb64 学出原地摆腿探测质心的
                # 习惯（zero_hold 前杆摆动 7 倍于基线）；收到 ±2cm 保留鲁棒性、压掉探测摆。
                # 2026-09-06 再收到 ±0.5cm：同一病理在 ±2cm 下仍然成立。base_link 10.745 kg
                # 占整机 12.729 kg 的 84.4%，base 质心 x 偏 ±20 mm 等于整机质心偏 ±16.9 mm；
                # 默认站姿质心高出轮心 123 mm，对应配平倾角 ±7.8°，比本周刚修掉的静平衡 bug
                # （17.2 mm / 8.0°）还大，等于把静平衡默认站姿的收益随机掉。±5 mm 对应整机
                # ±4.2 mm、配平倾角 ±2.0°，仍覆盖真实装配误差。
                params={"com_range": 0.005, "asset_cfg": SceneEntityCfg("robot")},
            ),
            "pd_gains": EventTermCfg(
                func=events.randomize_pd_gains,
                mode="startup",
                params={
                    "kp_range": (0.9, 1.1),  # 收窄:配合 stall_torque 上限,避免 kp 偏软时振荡跪地
                    "kd_range": (0.9, 1.1),
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
            "knee_spring_force": EventTermCfg(
                func=events.randomize_knee_spring_force,
                mode="startup",
                params={
                    # 左右腿独立 ±10%：气弹簧充气压差与装配公差（ETH PEA 论文同量级）
                    "force_scale_range": (0.9, 1.1),
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
            "motor_passive_params": EventTermCfg(
                func=events.randomize_motor_passive_params,
                mode="startup",
                params={
                    # 名义值待真机辨识回填，先用宽 DR 覆盖估计区间
                    # （来源见 docs/plan/motor_passive_params.md）
                    "armature_scale_range": (0.6, 1.5),
                    "damping_scale_range": (0.5, 2.5),
                    "frictionloss_scale_range": (0.5, 2.0),
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
            "default_dof_pos": EventTermCfg(
                func=events.randomize_default_dof_pos,
                mode="startup",
                params={
                    "offset_range": (-0.05, 0.05),
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
            "push_robots": EventTermCfg(
                func=events.push_robots,
                mode="interval",
                interval_range_s=(5.0, 6.0),
                params={
                    "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
        }
        cfg.episode_length_s = 20.0

    cfg.scale_rewards_by_dt = True
    cfg.sim = SimulationCfg(
        nconmax=256,
        njmax=1040,
        mujoco=MujocoCfg(timestep=_ROBOT_DEFAULTS.sim_dt),
    )
    cfg.viewer = ViewerConfig()

    return cfg
