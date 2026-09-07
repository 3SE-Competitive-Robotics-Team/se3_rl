"""崎岖地形行走任务环境配置。

移植自 scutrobotlab/wheeled-legged_RL 的 V14 rough 线，做法与参考仓库一致：rough 直接继承
flat 的整套配置，只换地形、加地形课程、开台阶状态机，奖励表只放松能耗类三项，其余逐项不动。
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
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    TerrainHeightSensorCfg,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from se3_train.tasks.flat.env_cfg import (
    FLAT_ACTION_SMOOTHNESS_SPRING,
    FLAT_CURRICULUM_ADVANCE_THRESHOLD_STRICT,
    FLAT_WHEEL_ACTION_SCALE,
)
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg
from se3_train.tasks.stair import observations as stair_observations

from . import ctbc, curriculums, events, observations, terminations
from .commands import StepUpCommandCfg
from .terrains import rough_terrains_cfg

# 台阶前瞻辅助默认开启：这是本次移植的主体，参考仓库跑场线
# （WheelbipeV14RoughEnvCfg_v1）也是开着的。关掉即退化为纯地形课程。
ROUGH_STEP_UP_ENABLED = True

# 前瞻距离(m)，与参考仓库 `wheel_forward_scan_cfg.scan.forward_offset` 一致。
# 传感器把射线排成 [-d, 0, +d]，索引 0/1/2 = 后方/身下/前方。
ROUGH_STEP_UP_LOOKAHEAD_M = 0.5

# 非平地列的速度指令限制：只发前向直行指令，平地列沿用 Flat 的速度课程。
# 对称随机指令下 20 s 的净位移是随机游走，地形课程的位移判据推不动（R2 平地列也只到 1.6）。
ROUGH_TERRAIN_COMMAND_OVERRIDE_ENABLED = True
ROUGH_TERRAIN_LIN_VEL_X_RANGE = (0.4, 2.4)
ROUGH_TERRAIN_ANG_VEL_YAW_RANGE = (-0.2, 0.2)
# 地形列 vx 上限跟随平地速度课程当前上限；平地速度课程只按平地列的跟踪分推进。
ROUGH_TERRAIN_LIN_VEL_X_FOLLOW_CURRICULUM = True
ROUGH_CURRICULUM_SIGNAL_TERRAIN_NAMES = ("flat",)
ROUGH_CURRICULUM_TRACKING_LOG_KEY = "Locomotion/tracking_lin_vel_reward_curriculum"

# 能耗类罚项在崎岖地形上的折价系数。参考仓库把 wheel_power 与 joint_torque
# 从 -1e-4 降到 -1e-5：上台阶本来就要更多力矩和功率，沿用平地定价会把爬升压住。
ROUGH_ENERGY_PENALTY_SCALE = 0.1
_ROUGH_ENERGY_REWARD_NAMES = ("leg_torques", "wheel_torques", "leg_power")

# 全部 env 从最简单一行起步，难度由 terrain_levels 课程逐级放开。
ROUGH_MAX_INIT_TERRAIN_LEVEL = 0

# 平地热身：前 N 轮全部 env 在平地列，之后各 env 在下一次 reset 时换回原列（2026-09-07 用户定，R7）。
# 速度课程推进阈值改用 Flat 的严格档 0.75：默认 0.5 在 vx=0 阶段轻松通过，100 轮内就把 vx 放到 1.6。
ROUGH_FLAT_WARMUP_ITERATIONS = 500
ROUGH_CURRICULUM_ADVANCE_THRESHOLD = FLAT_CURRICULUM_ADVANCE_THRESHOLD_STRICT

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


def _forward_probe_sensor_cfg(name: str, lookahead_m: float) -> TerrainHeightSensorCfg:
    """身前/身下/身后三条向下射线，供 step_up 状态机做空间差分。

    frame 取 base_link 而不是轮子：MJLab 的射线起点就是 frame 位置（局部 z 偏移恒为 0），
    起点落进几何体内部时净空被钳到 0，挂在轮心量不出高于轮半径的台阶。详见 commands.py。
    """
    return TerrainHeightSensorCfg(
        name=name,
        frame=ObjRef(type="body", name="base_link", entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(2.0 * lookahead_m, 0.0), resolution=lookahead_m),
        max_distance=2.0,
        include_geom_groups=(0,),
        reduction="none",
    )


def _to_step_up_command_cfg(command_cfg, **step_up_kwargs) -> StepUpCommandCfg:
    """把 Flat 的 JumpCommandCfg 原样搬进 StepUpCommandCfg，只追加 step_up_* 字段。

    逐字段搬运而不是重新构造，是为了让 Flat 基线以后改指令参数时 rough 自动跟随。
    """
    base = {f.name: getattr(command_cfg, f.name) for f in fields(command_cfg) if f.init}
    return StepUpCommandCfg(**base, **step_up_kwargs)


def env_cfg(
    play: bool = False,
    *,
    terrain_generator: TerrainGeneratorCfg | None = None,
    step_up_enabled: bool = ROUGH_STEP_UP_ENABLED,
    step_up_lookahead_m: float = ROUGH_STEP_UP_LOOKAHEAD_M,
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
    curriculum_advance_threshold: float = ROUGH_CURRICULUM_ADVANCE_THRESHOLD,
) -> ManagerBasedRlEnvCfg:
    """带地形课程与台阶前瞻辅助的崎岖地形环境配置。

    terrain_generator：None 时用 `rough_terrains_cfg()`；定向评测可传
    `stair_only_terrains_cfg()`。
    step_up_enabled：关掉后指令项逐位退化为 Flat 的 JumpCommandTerm，用于做“只有地形课程、
    没有状态机”的单变量对照。
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
    curriculum_advance_threshold：Flat 速度课程推进阈值（默认严格档 0.75）。
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

    sensor_name = "wheel_forward_sensor"
    cfg.scene.sensors = (
        *cfg.scene.sensors,
        _forward_probe_sensor_cfg(sensor_name, step_up_lookahead_m),
    )

    cfg.commands = dict(cfg.commands)
    cfg.commands["velocity_height"] = _to_step_up_command_cfg(
        cfg.commands["velocity_height"],
        step_up_enabled=step_up_enabled,
        step_up_sensor_name=sensor_name,
        terrain_command_override_enabled=terrain_command_override,
        terrain_lin_vel_x_range=tuple(terrain_lin_vel_x_range),
        terrain_ang_vel_yaw_range=tuple(terrain_ang_vel_yaw_range),
        terrain_lin_vel_x_follow_curriculum=terrain_lin_vel_x_follow_curriculum,
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

    # 墙（地形外围 border 这类高过机身的障碍）不是策略失败，按截断 bootstrap。
    cfg.terminations = dict(cfg.terminations)
    cfg.terminations["wall_blocked"] = TerminationTermCfg(
        func=terminations.wall_blocked,
        time_out=True,
    )
    # 清掉本块台阶、走到边框即截断结算课程，不进邻块（邻块是另一行难度）。
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

    # 上台阶要更大的力矩与功率，沿用平地定价会把爬升直接压住。
    cfg.rewards = dict(cfg.rewards)
    for name in _ROUGH_ENERGY_REWARD_NAMES:
        term = cfg.rewards[name]
        term.weight = float(term.weight) * float(energy_penalty_scale)

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
    "ROUGH_FLAT_WARMUP_ITERATIONS",
    "ROUGH_MAX_INIT_TERRAIN_LEVEL",
    "ROUGH_STEP_UP_ENABLED",
    "ROUGH_STEP_UP_LOOKAHEAD_M",
    "ROUGH_TERRAIN_ANG_VEL_YAW_RANGE",
    "ROUGH_TERRAIN_COMMAND_OVERRIDE_ENABLED",
    "ROUGH_TERRAIN_LIN_VEL_X_FOLLOW_CURRICULUM",
    "ROUGH_TERRAIN_LIN_VEL_X_RANGE",
    "env_cfg",
]
