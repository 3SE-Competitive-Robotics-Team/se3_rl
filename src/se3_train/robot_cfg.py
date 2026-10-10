import functools
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from se3_shared import DM8009P, M3508_C620_14, JointGroup
from se3_shared import RobotConfig as SharedRobotConfig

_RESOURCES = Path(__file__).resolve().parents[2] / "assets"
_MJCF_DIR = _RESOURCES / "robots" / "serialleg" / "mjcf"
_MJCF_PATH = _MJCF_DIR / "serialleg_closed_chain_v3_train_obb_trim.xml"

_ROBOT_CFG = SharedRobotConfig()

_WHEEL_JOINT_NAMES = JointGroup.WHEEL_NAMES


def _serialleg_spec_for_training(collision_geom_group: int | None = None) -> mujoco.MjSpec:
    """加载不含独立世界地面的 SerialLeg MJCF。

    MJLab 场景单独提供地形。保留 MJCF 的全局平面会覆盖 z=0 的生成台阶坑，
    使机器人与平面障碍碰撞，而不是与台阶地形碰撞。

    collision_geom_group：不为 None 时把 MJCF 里 group 0 的碰撞 geom 改到该 group。
    只改内存里的 spec，MJCF 文件不动（recovery 状态缓存按文件字节校验）。geom group 只影响
    渲染与射线传感器、不影响碰撞；rough 线用它让 `include_geom_groups=(0,)` 的高度射线只看
    地形，不再打到自己的腿和轮子，与 mjlab 资产库 collision=3 / visual=2 的约定一致。
    """
    spec = mujoco.MjSpec.from_file(str(_MJCF_PATH))
    for geom in list(spec.worldbody.geoms):
        if geom.name == "floor":
            spec.delete(geom)
            break
    if collision_geom_group is not None:
        for geom in spec.geoms:
            if geom.group == 0:
                geom.group = int(collision_geom_group)
    return spec


@functools.lru_cache(maxsize=1)
def serialleg_wheel_half_track() -> float:
    """MJCF 默认姿态下左右轮心横向间距的一半 (m)。腿在机身 x-z 平面内运动，轮距不随姿态变。"""
    model = mujoco.MjModel.from_xml_path(str(_MJCF_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ys = [
        float(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name), 1])
        for name in ("l_wheel_Link", "r_wheel_Link")
    ]
    return float(np.abs(ys[0] - ys[1]) / 2.0)


def get_serialleg_closedchain_cfg(
    *,
    leg_kp_override: float | None = None,
    leg_kd_override: float | None = None,
    wheel_kd_override: float | None = None,
    collision_geom_group: int | None = None,
    leg_torque_envelope_scale: float | None = None,
    wheel_torque_envelope: Literal["linear_peak", "rated_point"] = "linear_peak",
) -> EntityCfg:
    """构造固定使用正式 OBB 闭链 MJCF 的 SerialLeg 训练实体。

    leg_kp_override / leg_kd_override / wheel_kd_override：任务级名义增益；None 沿用共享配置。
    增益随机化围绕这些名义值采样，ONNX metadata 从执行器配置导出同一组值。

    leg_torque_envelope_scale（M53，2026-09-30）：None = 旧口径（saturation_effort 填峰值 40、effort_limit 填额定 20，
    恒扭矩区被额定削平）；给系数 k 时按物理含义取参——saturation_effort = k·电压限零速截距（DM8009P V1.0 @24V 约 132），
    effort_limit = k·峰值 40，即「k·40 平台 + 反电动势下降段」，转折与空载速度不随 k 变。两值随 actuator cfg 写进
    ONNX metadata，sim2x / 真机按同一 T-N 包络限矩。对比图见 scripts/plot_tn_envelope_proposal.py。

    wheel_torque_envelope（2026-10-06 用户定）：轮子 M3508 的 T-N 包络。
      "linear_peak"（旧口径）：saturation_effort = 峰值 3.32，从零速线性降到空载 68.5 rad/s，再被额定 2.21 截顶，
                               22.9 rad/s 起就低于额定，45 rad/s 只剩 1.14 N·m；与手册额定点（469 rpm@19:1 仍出 3 N·m）不符。
      "rated_point"：saturation_effort = 过手册额定点与空载点的反电动势线截距（约 82 N·m），effort_limit = 额定 2.21，
                     即额定平台延续到额定转速 66.65 rad/s、末端降到空载。低速力矩与旧口径相同，只改高速段。
    """
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
        stiffness=_ROBOT_CFG.leg_kp if leg_kp_override is None else float(leg_kp_override),
        damping=_ROBOT_CFG.leg_kd if leg_kd_override is None else float(leg_kd_override),
        saturation_effort=leg_saturation_effort,
        velocity_limit=DM8009P.no_load_speed,
        effort_limit=leg_effort_limit,
    )
    wheel_kd = _ROBOT_CFG.wheel_kd if wheel_kd_override is None else float(wheel_kd_override)
    if wheel_torque_envelope == "linear_peak":
        wheel_saturation_effort = M3508_C620_14.stall_torque
    elif wheel_torque_envelope == "rated_point":
        wheel_saturation_effort = M3508_C620_14.rated_point_stall_torque
    else:
        raise ValueError(f"未知的 wheel_torque_envelope {wheel_torque_envelope!r}")
    return EntityCfg(
        spec_fn=functools.partial(_serialleg_spec_for_training, collision_geom_group),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                leg_actuator_cfg,
                DcMotorActuatorCfg(
                    target_names_expr=_WHEEL_JOINT_NAMES,
                    stiffness=0.0,
                    damping=wheel_kd,
                    saturation_effort=wheel_saturation_effort,
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
