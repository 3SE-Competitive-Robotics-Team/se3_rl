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

from dataclasses import fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import GridPatternCfg, ObjRef, TerrainHeightSensorCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from se3_train.tasks.flat.env_cfg import (
    FLAT_ACTION_SMOOTHNESS_SPRING,
    FLAT_WHEEL_ACTION_SCALE,
)
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg

from . import curriculums, terminations
from .commands import StepUpCommandCfg
from .terrains import rough_terrains_cfg

# 台阶前瞻辅助默认开启：这是本次移植的主体，参考仓库跑场线
# （WheelbipeV14RoughEnvCfg_v1）也是开着的。关掉即退化为纯地形课程。
ROUGH_STEP_UP_ENABLED = True

# 前瞻距离(m)，与参考仓库 `wheel_forward_scan_cfg.scan.forward_offset` 一致。
# 传感器把射线排成 [-d, 0, +d]，索引 0/1/2 = 后方/身下/前方。
ROUGH_STEP_UP_LOOKAHEAD_M = 0.5

# 能耗类罚项在崎岖地形上的折价系数。参考仓库把 wheel_power 与 joint_torque
# 从 -1e-4 降到 -1e-5：上台阶本来就要更多力矩和功率，沿用平地定价会把爬升压住。
ROUGH_ENERGY_PENALTY_SCALE = 0.1
_ROUGH_ENERGY_REWARD_NAMES = ("leg_torques", "wheel_torques", "leg_power")

# 全部 env 从最简单一行起步，难度由 terrain_levels 课程逐级放开。
ROUGH_MAX_INIT_TERRAIN_LEVEL = 0

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
) -> ManagerBasedRlEnvCfg:
    """带地形课程与台阶前瞻辅助的崎岖地形环境配置。

    terrain_generator：None 时用 `rough_terrains_cfg()`；定向评测可传
    `stair_only_terrains_cfg()`。
    step_up_enabled：关掉后指令项逐位退化为 Flat 的 JumpCommandTerm，用于做“只有地形课程、
    没有状态机”的单变量对照。
    energy_penalty_scale：能耗类罚项相对 Flat 基线的折价系数，1.0 即与平地同价。
    terrain_curriculum：关掉后地形难度不再随表现升降（play 模式下恒为关）。
    """
    cfg = flat_env_cfg(play=play, **_ROUGH_FLAT_BASELINE)  # type: ignore[arg-type]

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
    )

    # 墙（地形外围 border 这类高过机身的障碍）不是策略失败，按截断 bootstrap。
    cfg.terminations = dict(cfg.terminations)
    cfg.terminations["wall_blocked"] = TerminationTermCfg(
        func=terminations.wall_blocked,
        time_out=True,
    )

    # 上台阶要更大的力矩与功率，沿用平地定价会把爬升直接压住。
    cfg.rewards = dict(cfg.rewards)
    for name in _ROUGH_ENERGY_REWARD_NAMES:
        term = cfg.rewards[name]
        term.weight = float(term.weight) * float(energy_penalty_scale)

    if not play and terrain_curriculum:
        cfg.curriculum = dict(cfg.curriculum)
        cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
            func=curriculums.terrain_levels,
            params={"command_name": "velocity_height"},
        )

    return cfg


__all__ = [
    "ROUGH_CONTACT_SENSOR_MAXMATCH",
    "ROUGH_ENERGY_PENALTY_SCALE",
    "ROUGH_MAX_INIT_TERRAIN_LEVEL",
    "ROUGH_STEP_UP_ENABLED",
    "ROUGH_STEP_UP_LOOKAHEAD_M",
    "env_cfg",
]
