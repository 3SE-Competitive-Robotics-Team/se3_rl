import functools
from pathlib import Path

import mujoco
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from se3_shared import DM8009P, M3508_C620_14, JointGroup
from se3_shared import RobotConfig as SharedRobotConfig

_RESOURCES = Path(__file__).resolve().parents[2] / "assets"
_MJCF_DIR = _RESOURCES / "robots" / "serialleg" / "mjcf"
_MJCF_PATH = _MJCF_DIR / "serialleg_closed_chain_v3_train_obb_trim.xml"

_ROBOT_CFG = SharedRobotConfig()

_WHEEL_JOINT_NAMES = JointGroup.WHEEL_NAMES


def _serialleg_spec_for_training(
    collision_geom_group: int | None = None,
    knee_gas_spring: bool = True,
) -> mujoco.MjSpec:
    """加载不含独立世界地面的 SerialLeg MJCF。

    MJLab 场景单独提供地形。保留 MJCF 的全局平面会覆盖 z=0 的生成台阶坑，
    使机器人与平面障碍碰撞，而不是与台阶地形碰撞。

    collision_geom_group：不为 None 时把 MJCF 里 group 0 的碰撞 geom 改到该 group。
    只改内存里的 spec，MJCF 文件不动（recovery 状态缓存按文件字节校验）。geom group 只影响
    渲染与射线传感器、不影响碰撞；rough 线用它让 `include_geom_groups=(0,)` 的高度射线只看
    地形，不再打到自己的腿和轮子，与 mjlab 资产库 collision=3 / visual=2 的约定一致。

    knee_gas_spring：False 时删掉两个膝气弹簧恒力 actuator（M52，2026-09-30）。同样只改内存 spec；
    挂点 site 与 spatial tendon 保留（tendon 无刚度/阻尼，没有 actuator 就不产生力）。
    """
    spec = mujoco.MjSpec.from_file(str(_MJCF_PATH))
    for geom in list(spec.worldbody.geoms):
        if geom.name == "floor":
            spec.delete(geom)
            break
    if not knee_gas_spring:
        for name in JointGroup.KNEE_SPRING_ACTUATOR_NAMES:
            actuator = spec.actuator(name)
            if actuator is None:
                raise ValueError(f"MJCF 缺少膝气弹簧 actuator {name!r}")
            spec.delete(actuator)
    if collision_geom_group is not None:
        for geom in spec.geoms:
            if geom.group == 0:
                geom.group = int(collision_geom_group)
    return spec


def get_serialleg_closedchain_cfg(
    *,
    wheel_kd_override: float | None = None,
    collision_geom_group: int | None = None,
    leg_kp_override: float | None = None,
    leg_kd_override: float | None = None,
    knee_gas_spring: bool = True,
    leg_torque_envelope_scale: float | None = None,
) -> EntityCfg:
    """构造固定使用正式 OBB 闭链 MJCF 的 SerialLeg 训练实体。

    leg_kp_override / leg_kd_override（M42，2026-09-28）：腿 PD 增益覆盖，None 取 se3_shared.RobotConfig（60 / 3.0）。
    覆盖值会随 actuator cfg 写进 ONNX metadata 的 KP/KD，sim2x 与真机按 metadata 执行，不需要改 runtime。
    knee_gas_spring：False = 训练 plant 去掉 300 N 膝气弹簧（M52），见 `_serialleg_spec_for_training`。
    leg_torque_envelope_scale（M53，2026-09-30）：None = 旧口径（saturation_effort 填峰值 40、effort_limit 填额定 20，
    恒扭矩区被额定削平）；给系数 k 时按物理含义取参——saturation_effort = k·电压限零速截距（DM8009P V1.0 @24V 约 132），
    effort_limit = k·峰值 40，即「k·40 平台 + 反电动势下降段」，转折与空载速度不随 k 变。两值随 actuator cfg 写进
    ONNX metadata，sim2x / 真机按同一 T-N 包络限矩。对比图见 scripts/plot_tn_envelope_proposal.py。
    """
    leg_kp = _ROBOT_CFG.leg_kp if leg_kp_override is None else float(leg_kp_override)
    leg_kd = _ROBOT_CFG.leg_kd if leg_kd_override is None else float(leg_kd_override)
    if leg_torque_envelope_scale is None:
        leg_saturation_effort = DM8009P.stall_torque
        leg_effort_limit = DM8009P.rated_torque
    else:
        scale = float(leg_torque_envelope_scale)
        if not 0.0 < scale <= 1.0:
            raise ValueError(f"leg_torque_envelope_scale 必须位于 (0, 1]，实际为 {scale}")
        leg_saturation_effort = scale * DM8009P.voltage_limited_stall_torque
        leg_effort_limit = scale * DM8009P.stall_torque
    leg_actuator_cfg = DcMotorActuatorCfg(
        target_names_expr=JointGroup.POLICY_LEG_NAMES,
        stiffness=leg_kp,
        damping=leg_kd,
        saturation_effort=leg_saturation_effort,
        velocity_limit=DM8009P.no_load_speed,
        effort_limit=leg_effort_limit,
    )
    wheel_kd = _ROBOT_CFG.wheel_kd if wheel_kd_override is None else float(wheel_kd_override)
    return EntityCfg(
        spec_fn=functools.partial(
            _serialleg_spec_for_training, collision_geom_group, knee_gas_spring
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                leg_actuator_cfg,
                DcMotorActuatorCfg(
                    target_names_expr=_WHEEL_JOINT_NAMES,
                    stiffness=0.0,
                    damping=wheel_kd,
                    saturation_effort=M3508_C620_14.stall_torque,
                    velocity_limit=M3508_C620_14.no_load_speed,
                    effort_limit=M3508_C620_14.rated_torque,
                ),
            ),
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, _ROBOT_CFG.default_base_height),
            joint_pos=_ROBOT_CFG.default_model_joint_pos,
            joint_vel={".*": 0.0},
        ),
    )
