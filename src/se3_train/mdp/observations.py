"""SE3 轮腿机器人的观测函数。

观测空间:
- actor: 34D (腿部前杆使用 sin/cos 相位观测)
- critic: actor + 特权信息
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch

from se3_shared import (
    DM8009P,
    M3508_C620_14,
    JointGroup,
    ObservationConfig,
    policy_leg_phase_active_obs_torch,
)
from se3_shared import RobotConfig as SharedRobotConfig
from se3_train.mdp.contact_utils import (
    contact_force_nonfinite_env_mask,
    finite_contact_force_norm,
)
from se3_train.mdp.joint_indices import (
    leg_actuator_ids,
    policy_leg_joint_ids,
    tensor_ids,
    wheel_actuator_ids,
    wheel_joint_ids,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

# 模块级观测配置，作为缩放系数的单一来源
_OBS_CFG = ObservationConfig()
_NOMINAL_KNEE_SPRING_FORCE = SharedRobotConfig().knee_gas_spring_force
_WHEEL_CONTACT_FORCE_NONFINITE_TOTAL_ATTR = "_wheel_contact_force_nonfinite_total"
_WHEEL_CONTACT_FORCE_SAMPLE_TOTAL_ATTR = "_wheel_contact_force_sample_total"
_DEFAULT_CONTACT_DEBUG_LOG_INTERVAL_STEPS = 256


def _finite_clamp(value: torch.Tensor, limit: float | None = None) -> torch.Tensor:
    """把观测限制在有限范围内，避免单个发散 env 污染整批 PPO。"""
    bound = float(_OBS_CFG.clip_value if limit is None else limit)
    return torch.nan_to_num(value, nan=0.0, posinf=bound, neginf=-bound).clamp(-bound, bound)


def base_ang_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """基座坐标系下的角速度,缩放 0.25。"""
    robot = env.scene["robot"]
    return _finite_clamp(robot.data.root_link_ang_vel_b * _OBS_CFG.ang_vel_scale)


def projected_gravity_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """投影到基座坐标系下的重力向量。"""
    robot = env.scene["robot"]
    return _finite_clamp(robot.data.projected_gravity_b, limit=1.0)


def commands_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """速度/姿态/高度指令,缩放 (2.0, 0.25, 5.0, 5.0, 5.0)。

    取前 5 维: [lin_vel_x, ang_vel_yaw, pitch, roll, height]
    兼容跳跃任务（8 维指令），跳跃扩展维度由 jump_commands_obs 单独输出。
    """
    cmd = env.command_manager.get_command("velocity_height")
    scale = torch.tensor(list(_OBS_CFG.command_scale), device=cmd.device)
    return _finite_clamp(cmd[:, :5] * scale)


def leg_joint_pos_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """腿部主动杆相位和主动杆夹角观测，6D。"""
    robot = env.scene["robot"]
    leg_ids = policy_leg_joint_ids(robot)
    return _finite_clamp(
        policy_leg_phase_active_obs_torch(
            robot.data.joint_pos[:, leg_ids],
            robot.data.default_joint_pos[:, leg_ids],
        )
    )


def leg_joint_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """腿部主动杆速度,缩放 0.25,4D。"""
    robot = env.scene["robot"]
    leg_ids = policy_leg_joint_ids(robot)
    return _finite_clamp(robot.data.joint_vel[:, leg_ids] * _OBS_CFG.leg_vel_scale)


def wheel_pos_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """保留轮子位置观测槽位，但固定为 0，避免连续转角无限累计。"""
    robot = env.scene["robot"]
    wheel_ids = wheel_joint_ids(robot)
    return torch.zeros_like(robot.data.joint_pos[:, wheel_ids])


def wheel_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """轮子关节速度,缩放 0.05（MJCF 已修正轴方向）。"""
    robot = env.scene["robot"]
    return _finite_clamp(robot.data.joint_vel[:, wheel_joint_ids(robot)] * _OBS_CFG.wheel_vel_scale)


def last_actions_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """上一步的 6 个动作。"""
    return _finite_clamp(env.action_manager.action)


def processed_last_actions_obs(
    env: ManagerBasedRlEnv,
    action_name: str = "delayed_action",
) -> torch.Tensor:
    """返回动作项处理后、FIFO 延迟前的上一条 6D command。

    scripted teacher 训练必须让 actor 看见实际送入延迟队列的动作，否则 teacher
    相位与方向会变成隐藏状态。teacher 关闭时该值退化为普通 clipped policy action，
    因而不改变部署时的 34D 单帧观测契约。
    """

    action_term = env.action_manager.get_term(action_name)
    action = getattr(action_term, "raw_action", None)
    if not isinstance(action, torch.Tensor) or action.shape != env.action_manager.action.shape:
        raise RuntimeError(f"动作项 {action_name} 未提供匹配的 raw_action")
    return _finite_clamp(action)


# --- Critic 特权观测 ---


def base_lin_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """基座坐标系下的线速度,3D（特权信息,actor 不可见）。"""
    robot = env.scene["robot"]
    return _finite_clamp(robot.data.root_link_lin_vel_b)


def wheel_contact_force_obs(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """轮子地面接触力标量,2D（特权信息）。"""
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, 2, device=env.device)
    _record_wheel_contact_force_nonfinite(env, data.force)
    return _finite_clamp(finite_contact_force_norm(data.force))


def knee_gas_spring_force_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """左右膝气弹簧当前恒力，按额定值归一，2D（特权信息,critic 专用）。

    值来自 randomize_knee_spring_force 的采样缓存；任务未启用该 DR 事件时
    弹簧力就是 MJCF 额定值，退化为常数 1。
    """
    values = getattr(env, "_knee_spring_force", None)
    if not isinstance(values, torch.Tensor) or values.shape[0] != env.num_envs:
        return torch.ones(env.num_envs, 2, device=env.device)
    return _finite_clamp(values / _NOMINAL_KNEE_SPRING_FORCE)


def motor_torque_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """六电机实测输出力矩，按各自额定值归一，6D（特权信息，critic 专用）。

    只统计电机 actuator（按名解析，气弹簧 actuator 不在内），CTS（RA-L 2024）
    特权集合中的关节力矩项。
    """
    robot = env.scene["robot"]
    leg_ids = tensor_ids(leg_actuator_ids(robot), device=env.device)
    wheel_ids = tensor_ids(wheel_actuator_ids(robot), device=env.device)
    leg_torque = robot.data.actuator_force[:, leg_ids] / float(DM8009P.rated_torque)
    wheel_torque = robot.data.actuator_force[:, wheel_ids] / float(M3508_C620_14.rated_torque)
    return _finite_clamp(torch.cat([leg_torque, wheel_torque], dim=-1))


def joint_acc_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """六个 policy 关节的角加速度，×0.0025 缩放，6D（特权信息，critic 专用）。"""
    robot = env.scene["robot"]
    leg_ids = tensor_ids(policy_leg_joint_ids(robot), device=env.device)
    wheel_ids = tensor_ids(wheel_joint_ids(robot), device=env.device)
    acc = torch.cat([robot.data.joint_acc[:, leg_ids], robot.data.joint_acc[:, wheel_ids]], dim=-1)
    return _finite_clamp(acc * 0.0025)


def contact_force_norm_obs(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """指定接触传感器逐 body 的法向合力范数，/100 N 缩放（特权信息，critic 专用）。

    leg_contact_sensor 为 4D（四条腿 Link），collision_sensor 为 1D（base_link）。
    """
    data = env.scene.sensors[sensor_name].data
    return _finite_clamp(finite_contact_force_norm(data.force) / 100.0)


_DR_PARAM_CACHE_ATTR = "_dr_model_param_obs_cache"


def _resolve_dr_param_indices(env: ManagerBasedRlEnv) -> dict:
    """解析 DR 回读观测所需的全局模型索引与默认值，按 env 缓存。

    名称均按 '/' 后缀匹配（scene 内实体带 'robot/' 前缀）。
    """
    cache = getattr(env, _DR_PARAM_CACHE_ATTR, None)
    if isinstance(cache, dict):
        return cache

    mj_model = env.sim.mj_model

    def _find(obj_type: mujoco.mjtObj, count: int, name: str) -> int:
        for idx in range(count):
            full = mujoco.mj_id2name(mj_model, obj_type, idx)
            if full and full.split("/")[-1] == name:
                return idx
        raise ValueError(f"模型缺少 {obj_type} 对象 {name}")

    base_bid = _find(mujoco.mjtObj.mjOBJ_BODY, mj_model.nbody, "base_link")
    friction_gid = _find(mujoco.mjtObj.mjOBJ_GEOM, mj_model.ngeom, "l_wheel_Link_collision_0")
    kp_aid = _find(mujoco.mjtObj.mjOBJ_ACTUATOR, mj_model.nu, JointGroup.POLICY_LEG_NAMES[0])
    dof_adr = []
    for name in (*JointGroup.POLICY_LEG_NAMES, *JointGroup.WHEEL_NAMES):
        jid = _find(mujoco.mjtObj.mjOBJ_JOINT, mj_model.njnt, name)
        dof_adr.append(int(mj_model.jnt_dofadr[jid]))

    def _default(field: str) -> torch.Tensor:
        return torch.as_tensor(env.sim.get_default_field(field), device=env.device)

    cache = {
        "base_bid": base_bid,
        "friction_gid": friction_gid,
        "kp_aid": kp_aid,
        "dof_adr": torch.tensor(dof_adr, device=env.device, dtype=torch.long),
        "default_body_mass": _default("body_mass")[base_bid],
        "default_body_ipos": _default("body_ipos")[base_bid],
        "default_body_inertia": _default("body_inertia")[base_bid],
        "default_friction": _default("geom_friction")[friction_gid, 0],
        "default_kp": _default("actuator_gainprm")[kp_aid, 0],
        "default_kd": _default("actuator_biasprm")[kp_aid, 2],
        "default_armature": _default("dof_armature")[dof_adr],
        "default_damping": _default("dof_damping")[dof_adr],
        "default_frictionloss": _default("dof_frictionloss")[dof_adr],
    }
    setattr(env, _DR_PARAM_CACHE_ATTR, cache)
    return cache


def _ratio_to_default(value: torch.Tensor, default: torch.Tensor) -> torch.Tensor:
    """按默认值归一；默认值为 0 的量（如轮子被动阻尼）返回常数 1。"""
    safe = torch.where(default.abs() > 1e-12, default, torch.ones_like(default))
    ratio = value / safe
    return torch.where(default.abs() > 1e-12, ratio, torch.ones_like(ratio))


def dr_model_params_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """域随机化后的模型参数回读，28D（特权信息，critic 专用）。

    直接从 per-world 模型字段读取，无需事件另存缓冲；未启用对应 DR 的任务
    退化为常数（比值 1 / 偏移 0）。布局：
    [0]     接触摩擦系数比值（randomize_friction 对全部 robot geom 写同值）
    [1]     base 质量比值
    [2:5]   base 质心偏移 / 0.05 m
    [5:8]   base 惯量比值
    [8:10]  腿部 PD kp/kd 缩放（randomize_pd_gains 每 env 单值广播）
    [10:28] 六电机 armature/damping/frictionloss 比值（逐关节独立 DR）
    """
    cache = _resolve_dr_param_indices(env)
    model = env.sim.model
    friction = _ratio_to_default(
        model.geom_friction[:, cache["friction_gid"], 0], cache["default_friction"]
    )
    base_mass = _ratio_to_default(model.body_mass[:, cache["base_bid"]], cache["default_body_mass"])
    com_offset = (model.body_ipos[:, cache["base_bid"]] - cache["default_body_ipos"]) / 0.05
    inertia = _ratio_to_default(
        model.body_inertia[:, cache["base_bid"]], cache["default_body_inertia"]
    )
    kp = _ratio_to_default(model.actuator_gainprm[:, cache["kp_aid"], 0], cache["default_kp"])
    kd = _ratio_to_default(model.actuator_biasprm[:, cache["kp_aid"], 2], cache["default_kd"])
    dof_adr = cache["dof_adr"]
    armature = _ratio_to_default(model.dof_armature[:, dof_adr], cache["default_armature"])
    damping = _ratio_to_default(model.dof_damping[:, dof_adr], cache["default_damping"])
    frictionloss = _ratio_to_default(
        model.dof_frictionloss[:, dof_adr], cache["default_frictionloss"]
    )
    return _finite_clamp(
        torch.cat(
            [
                friction.unsqueeze(-1),
                base_mass.unsqueeze(-1),
                com_offset,
                inertia,
                kp.unsqueeze(-1),
                kd.unsqueeze(-1),
                armature,
                damping,
                frictionloss,
            ],
            dim=-1,
        )
    )


def _record_wheel_contact_force_nonfinite(env: ManagerBasedRlEnv, force: torch.Tensor) -> None:
    """累计 wheel contact force 的非有限值触发次数，并写入训练日志。"""
    step = int(getattr(env, "common_step_counter", 0))
    interval = max(
        1,
        int(
            getattr(
                env,
                "_se3_contact_debug_log_interval_steps",
                _DEFAULT_CONTACT_DEBUG_LOG_INTERVAL_STEPS,
            )
        ),
    )
    if interval > 1 and (step - 1) % interval != 0:
        return

    # 累计值保持 0 维 GPU tensor：``.item()`` 会在 rollout 内制造主机同步点；
    # RSL-RL logger 在迭代末对 extras 张量统一转换。
    nonfinite_envs = contact_force_nonfinite_env_mask(force).sum().float()
    previous_total = getattr(env, _WHEEL_CONTACT_FORCE_NONFINITE_TOTAL_ATTR, None)
    if not isinstance(previous_total, torch.Tensor):
        previous_total = torch.zeros((), device=env.device)
    total = previous_total + nonfinite_envs
    samples = float(getattr(env, _WHEEL_CONTACT_FORCE_SAMPLE_TOTAL_ATTR, 0)) + env.num_envs
    setattr(env, _WHEEL_CONTACT_FORCE_NONFINITE_TOTAL_ATTR, total)
    setattr(env, _WHEEL_CONTACT_FORCE_SAMPLE_TOTAL_ATTR, samples)

    if hasattr(env, "extras"):
        env.extras.setdefault("log", {}).update(
            {
                "Debug/wheel_contact_force_nonfinite_last_envs": nonfinite_envs,
                "Debug/wheel_contact_force_nonfinite_total_envs": total,
                "Debug/wheel_contact_force_obs_total_envs": samples,
                "Debug/wheel_contact_force_nonfinite_rate": total / max(samples, 1.0),
            }
        )


def base_height_obs(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """底盘离地高度（多射线取均值），标量（特权信息，critic 专用）。"""
    from mjlab.sensor import TerrainHeightSensor

    sensor: TerrainHeightSensor = env.scene[sensor_name]
    return torch.nan_to_num(sensor.data.heights, nan=0.0, posinf=0.0, neginf=0.0)


def jump_commands_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """跳跃指令观测，3D：[jump_flag, jump_target_height, jump_phase]。

    jump_flag:          0/1，本 episode 是否触发跳跃
    jump_target_height: 目标跳跃高度（m），范围 0.1~0.6
    jump_phase:         0→1 连续相位，从参考轨迹第 0 帧开始按 motion 时间推进

    要求指令必须为 8 维（JumpCommandTerm），否则直接报错。
    """
    cmd = env.command_manager.get_command("velocity_height")
    if cmd.shape[1] != 8:
        raise ValueError(
            f"jump_commands_obs 要求 8 维指令 (JumpCommandTerm),"
            f" 实际得到 {cmd.shape[1]} 维。请将 velocity_height 指令替换为 JumpCommandCfg。"
        )
    return _finite_clamp(cmd[:, 5:8])
