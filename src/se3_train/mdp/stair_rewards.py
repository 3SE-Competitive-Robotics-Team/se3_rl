"""台阶训练奖励和诊断项。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from se3_train.mdp.joint_indices import wheel_joint_ids

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _get_stair_state(env: ManagerBasedRlEnv):
    """读取当前环境的 CTBC 状态机；没有台阶状态时返回 None。"""
    return getattr(env, "stair_climb_state", None)


def _upright_gate(env: ManagerBasedRlEnv) -> torch.Tensor:
    """只在接近直立时施加行走类惩罚，避免倒地恢复样本被压制。"""
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    return torch.clamp(-pg_z, 0.0, 0.7) / 0.7


def _ctbc_triggered_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
    """返回 CTBC 前馈当前是否激活，兼容参考代码的 contact_triggered 名称。"""
    state = _get_stair_state(env)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    active_mask = getattr(state, "active_mask", None)
    if callable(active_mask):
        return active_mask().to(device=env.device, dtype=torch.bool)

    contact_triggered = getattr(state, "contact_triggered", None)
    if callable(contact_triggered):
        return contact_triggered().to(device=env.device, dtype=torch.bool)

    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def _terrain_type_mask(
    env: ManagerBasedRlEnv,
    terrain_type_names: tuple[str, ...] | list[str] | None,
) -> torch.Tensor:
    """按 terrain generator 的子地形名称生成逐 env mask。"""
    if not terrain_type_names:
        return torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    terrain = getattr(env.scene, "terrain", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    if not isinstance(terrain_types, torch.Tensor):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    cfg = getattr(terrain, "cfg", None)
    generator_cfg = getattr(cfg, "terrain_generator", None)
    sub_terrains = getattr(generator_cfg, "sub_terrains", {}) or {}
    selected = {str(name) for name in terrain_type_names}
    terrain_types = terrain_types.to(device=env.device)
    if terrain_types.shape != (env.num_envs,):
        terrain_types = terrain_types.reshape(-1)
        if terrain_types.numel() == 1:
            terrain_types = terrain_types.expand(env.num_envs)
        elif terrain_types.numel() != env.num_envs:
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for terrain_index, terrain_name in enumerate(sub_terrains):
        if str(terrain_name) in selected:
            mask = mask | (terrain_types == terrain_index)
    return mask


def stair_climb_height(
    env: ManagerBasedRlEnv,
    command_name: str = "velocity_height",
    max_gain: float = 0.35,
    forward_gate_start: float = 0.10,
    forward_gate_width: float = 0.25,
    terrain_type_names: tuple[str, ...] | list[str] | None = None,
) -> torch.Tensor:
    """奖励真实台阶爬升，原地抬高机身不再给分。"""
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    root_pos = robot.data.root_link_pos_w
    origins = env.scene.env_origins
    base_z = root_pos[:, 2] - origins[:, 2]
    forward_x = root_pos[:, 0] - origins[:, 0]
    height_gain = torch.clamp(base_z - cmd[:, 4], min=0.0, max=float(max_gain))
    forward_gate = torch.clamp(
        (forward_x - float(forward_gate_start)) / max(float(forward_gate_width), 1.0e-6),
        min=0.0,
        max=1.0,
    )
    terrain_mask = _terrain_type_mask(env, terrain_type_names)
    climb = height_gain * forward_gate * terrain_mask.float() * _upright_gate(env)
    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        success_like = (height_gain > 0.12) & (forward_x > 0.25) & terrain_mask
        env.extras["log"].update(
            {
                "Stair/stair_climb_height": climb.mean().item(),
                "Stair/stair_height_gain_raw": height_gain.mean().item(),
                "Stair/stair_base_z": base_z.mean().item(),
                "Stair/stair_command_height": cmd[:, 4].mean().item(),
                "Stair/stair_forward_x": forward_x.mean().item(),
                "Stair/stair_forward_gate": forward_gate.mean().item(),
                "Stair/stair_terrain_mask_rate": terrain_mask.float().mean().item(),
                "Stair/stair_success_like_rate": success_like.float().mean().item(),
            }
        )
    return climb


def stair_forward_distance(
    env: ManagerBasedRlEnv,
    max_progress: float = 1.0,
    terrain_type_names: tuple[str, ...] | list[str] | None = None,
) -> torch.Tensor:
    """只在台阶地形奖励向 +x 通过台阶入口的位移。"""
    robot = env.scene["robot"]
    forward_x = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    terrain_mask = _terrain_type_mask(env, terrain_type_names)
    reward = torch.clamp(forward_x, min=0.0, max=float(max_progress))
    return reward * terrain_mask.float() * _upright_gate(env)


def stair_forward_progress(
    env: ManagerBasedRlEnv,
    command_name: str = "velocity_height",
    sigma: float = 0.25,
) -> torch.Tensor:
    """台阶前进速度跟踪奖励，保留向前梯度。"""
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    error = torch.square(robot.data.root_link_lin_vel_b[:, 0] - cmd[:, 0])
    reward = torch.exp(-error / float(sigma))
    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"]["Stair/forward_speed_error"] = torch.sqrt(error).mean().item()
    return reward


def stair_contact_diagnostics(
    env: ManagerBasedRlEnv,
    wheel_sensor_name: str = "wheel_sensor",
    leg_sensor_name: str = "leg_contact_sensor",
    collision_sensor_name: str = "collision_sensor",
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """记录台阶训练关心的接触诊断，返回零奖励。"""
    wheel_contact = _contact_any(env, wheel_sensor_name, force_threshold)
    leg_contact = _contact_any(env, leg_sensor_name, force_threshold)
    collision_contact = _contact_any(env, collision_sensor_name, force_threshold)
    nonwheel = leg_contact | collision_contact
    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Stair/wheel_contact_rate": wheel_contact.float().mean().item(),
                "Stair/nonwheel_contact_rate": nonwheel.float().mean().item(),
                "Stair/leg_contact_rate": leg_contact.float().mean().item(),
                "Stair/base_collision_rate": collision_contact.float().mean().item(),
            }
        )
    return torch.zeros(env.num_envs, device=env.device)


def stair_yaw_diagnostics(
    env: ManagerBasedRlEnv,
    command_name: str = "velocity_height",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """临时记录台阶 play 自转来源，不参与奖励。"""
    robot = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_yaw = cmd[:, 1]
    base_yaw_rate = robot.data.root_link_ang_vel_b[:, 2]
    wheel_vel = robot.data.joint_vel[:, wheel_joint_ids(robot)]
    wheel_yaw_drive = wheel_vel[:, 0] + wheel_vel[:, 1]
    wheel_forward_drive = wheel_vel[:, 0] - wheel_vel[:, 1]
    action = env.action_manager.action
    wheel_action_yaw_drive = action[:, 4] + action[:, 5]
    wheel_action_forward_drive = action[:, 4] - action[:, 5]
    leg_action_mirror = torch.mean(
        torch.abs(torch.stack((action[:, 0] + action[:, 2], action[:, 1] + action[:, 3]), dim=1)),
        dim=1,
    )
    ctbc_active = _ctbc_triggered_mask(env)

    action_term = _serial_action_term(env, robot)
    raw_wheel_yaw_drive = wheel_action_yaw_drive
    raw_leg_action_mirror = leg_action_mirror
    delayed_wheel_yaw_drive = wheel_action_yaw_drive
    if action_term is not None:
        raw_action = getattr(action_term, "raw_action", None)
        delayed_action = getattr(action_term, "delayed_action", None)
        if isinstance(raw_action, torch.Tensor) and raw_action.shape[-1] >= 6:
            raw_wheel_yaw_drive = raw_action[:, 4] + raw_action[:, 5]
            raw_leg_action_mirror = torch.mean(
                torch.abs(
                    torch.stack(
                        (raw_action[:, 0] + raw_action[:, 2], raw_action[:, 1] + raw_action[:, 3]),
                        dim=1,
                    )
                ),
                dim=1,
            )
        if isinstance(delayed_action, torch.Tensor) and delayed_action.shape[-1] >= 6:
            delayed_wheel_yaw_drive = delayed_action[:, 4] + delayed_action[:, 5]

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        log = {
            "StairYaw/cmd_yaw_mean": cmd_yaw.mean().item(),
            "StairYaw/cmd_yaw_abs_mean": torch.abs(cmd_yaw).mean().item(),
            "StairYaw/base_yaw_rate_mean": base_yaw_rate.mean().item(),
            "StairYaw/base_yaw_rate_abs_mean": torch.abs(base_yaw_rate).mean().item(),
            "StairYaw/base_yaw_rate_abs_ctbc": _masked_mean(torch.abs(base_yaw_rate), ctbc_active),
            "StairYaw/base_yaw_rate_abs_no_ctbc": _masked_mean(
                torch.abs(base_yaw_rate), ~ctbc_active
            ),
            "StairYaw/ctbc_active_rate": ctbc_active.float().mean().item(),
            "StairYaw/wheel_vel_yaw_drive_mean": wheel_yaw_drive.mean().item(),
            "StairYaw/wheel_vel_yaw_drive_abs_mean": torch.abs(wheel_yaw_drive).mean().item(),
            "StairYaw/wheel_vel_yaw_drive_abs_ctbc": _masked_mean(
                torch.abs(wheel_yaw_drive), ctbc_active
            ),
            "StairYaw/wheel_vel_yaw_drive_abs_no_ctbc": _masked_mean(
                torch.abs(wheel_yaw_drive), ~ctbc_active
            ),
            "StairYaw/wheel_vel_forward_drive_abs_mean": torch.abs(wheel_forward_drive)
            .mean()
            .item(),
            "StairYaw/action_wheel_yaw_drive_abs_mean": torch.abs(wheel_action_yaw_drive)
            .mean()
            .item(),
            "StairYaw/action_wheel_forward_drive_abs_mean": torch.abs(wheel_action_forward_drive)
            .mean()
            .item(),
            "StairYaw/raw_action_wheel_yaw_drive_abs_mean": torch.abs(raw_wheel_yaw_drive)
            .mean()
            .item(),
            "StairYaw/delayed_action_wheel_yaw_drive_abs_mean": torch.abs(delayed_wheel_yaw_drive)
            .mean()
            .item(),
            "StairYaw/leg_action_mirror_abs_mean": leg_action_mirror.mean().item(),
            "StairYaw/raw_leg_action_mirror_abs_mean": raw_leg_action_mirror.mean().item(),
        }
        log.update(_terrain_yaw_logs(env, torch.abs(base_yaw_rate), torch.abs(wheel_yaw_drive)))
        env.extras["log"].update(log)

    return torch.zeros(env.num_envs, device=env.device)


def ctbc_active_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
    """返回 CTBC 当前是否激活，用于后续门控奖励。"""
    return _ctbc_triggered_mask(env)


def leg_torques_no_ctbc(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """CTBC 抬腿周期内暂时豁免腿部力矩惩罚。"""
    from se3_train.mdp import rewards

    penalty = rewards.leg_torques(env, asset_cfg=asset_cfg)
    triggered = _ctbc_triggered_mask(env)
    return penalty * (~triggered).float() * _upright_gate(env)


def leg_power_no_ctbc(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """CTBC 抬腿周期内暂时豁免腿部功率惩罚。"""
    from se3_train.mdp import rewards

    penalty = rewards.leg_power(env, asset_cfg=asset_cfg)
    triggered = _ctbc_triggered_mask(env)
    return penalty * (~triggered).float() * _upright_gate(env)


def stand_still_no_ctbc(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    default_height: float = 0.27,
    height_tolerance: float = 40.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """CTBC 抬腿周期内不惩罚偏离默认站姿。"""
    from se3_train.mdp import rewards

    penalty = rewards.stand_still(
        env,
        command_name=command_name,
        command_threshold=command_threshold,
        default_height=default_height,
        height_tolerance=height_tolerance,
        asset_cfg=asset_cfg,
    )
    triggered = _ctbc_triggered_mask(env)
    return penalty * (~triggered).float()


def action_rate_no_ctbc(env: ManagerBasedRlEnv) -> torch.Tensor:
    """CTBC 抬腿周期内降低动作变化率惩罚，保留少量平滑约束。"""
    from se3_train.mdp import rewards

    penalty = rewards.action_rate(env)
    triggered = _ctbc_triggered_mask(env).float()
    scale = 1.0 - 0.8 * triggered
    return penalty * scale * _upright_gate(env)


def joint_pos_penalty_no_ctbc(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    stand_still_scale: float = 5.0,
    velocity_threshold: float = 0.5,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """CTBC 抬腿周期内豁免默认腿姿惩罚，避免压制跨台阶动作。"""
    from se3_train.mdp import rewards

    penalty = rewards.joint_pos_penalty(
        env,
        command_name=command_name,
        asset_cfg=asset_cfg,
        stand_still_scale=stand_still_scale,
        velocity_threshold=velocity_threshold,
        command_threshold=command_threshold,
    )
    triggered = _ctbc_triggered_mask(env)
    effective = penalty * (~triggered).float()
    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "Stair/joint_pos_penalty_effective": effective.mean().item(),
                "Stair/joint_pos_penalty_ctbc_exempt_rate": triggered.float().mean().item(),
            }
        )
    return effective


def contact_forces_no_ctbc(
    env: ManagerBasedRlEnv,
    threshold: float,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """CTBC 抬腿周期内放宽轮子接触力惩罚，仍约束极端硬撞。"""
    from se3_train.mdp import rewards

    penalty = rewards.contact_forces(
        env,
        threshold=threshold,
        sensor_name=sensor_name,
        asset_cfg=asset_cfg,
    )
    triggered = _ctbc_triggered_mask(env).float()
    return penalty * (1.0 - 0.5 * triggered)


def stair_feet_clearance(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    h_min: float = 0.03,
    h_max: float = 0.25,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """CTBC 激活时奖励轮子抬离地面到合理高度。"""
    del sensor_name
    triggered = _ctbc_triggered_mask(env).float()
    if not bool(triggered.any()):
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene[asset_cfg.name]
    if not hasattr(env, "_stair_wheel_body_ids"):
        body_ids, _ = robot.find_bodies(("l_wheel_Link", "r_wheel_Link"), preserve_order=True)
        env._stair_wheel_body_ids = body_ids
    wheel_z = torch.nan_to_num(
        robot.data.body_com_pos_w[:, env._stair_wheel_body_ids, 2],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    origin_z = env.scene.env_origins[:, 2].unsqueeze(1)
    wheel_height = wheel_z - origin_z
    in_range = ((wheel_height > float(h_min)) & (wheel_height < float(h_max))).float()
    return in_range.sum(dim=-1) * triggered


def stair_feet_air_time(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """CTBC 激活时奖励轮子短暂离地，鼓励形成抬轮动作。"""
    del asset_cfg
    triggered = _ctbc_triggered_mask(env).float()
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = torch.linalg.norm(data.force, dim=-1).reshape(env.num_envs, -1)[:, :2]
    in_air = (force_mag < 1.0).float()
    return in_air.sum(dim=-1) * triggered


def stair_contact_number(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """CTBC 激活时奖励摆动轮/支撑轮接触状态匹配。"""
    del asset_cfg
    state = _get_stair_state(env)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = torch.linalg.norm(data.force, dim=-1).reshape(env.num_envs, -1)[:, :2]
    in_contact = force_mag > 1.0
    ff_phase = getattr(state, "_ff_phase", None)
    if not isinstance(ff_phase, torch.Tensor):
        return torch.zeros(env.num_envs, device=env.device)
    expected_swing = ff_phase[:, :2] >= 0
    expected_stance = ~expected_swing
    match = (in_contact & expected_stance) | (~in_contact & expected_swing)
    reward = match.float() - 1.3 * (~match).float()
    return reward.sum(dim=-1) * _ctbc_triggered_mask(env).float()


def stair_wheel_swing_zero_vel(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """CTBC 激活时让摆动轮少空转。"""
    del sensor_name
    state = _get_stair_state(env)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)
    ff_phase = getattr(state, "_ff_phase", None)
    if not isinstance(ff_phase, torch.Tensor):
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene[asset_cfg.name]
    wheel_vel = robot.data.joint_vel[:, wheel_joint_ids(robot)]
    active_side = (ff_phase[:, :2] >= 0).float()
    return torch.exp(-(active_side * wheel_vel**2).sum(dim=-1)) * _ctbc_triggered_mask(env).float()


def _contact_any(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    force = torch.linalg.norm(data.force, dim=-1).reshape(env.num_envs, -1)
    return (force > float(force_threshold)).any(dim=1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    """计算 mask 内均值；空 mask 返回 0。"""
    mask = mask.to(device=values.device, dtype=torch.bool).reshape(-1)
    values = values.reshape(-1)
    if not bool(mask.any()):
        return 0.0
    return values[mask].float().mean().item()


def _serial_action_term(env: ManagerBasedRlEnv, robot: object) -> object | None:
    """找到 SerialLeg action term，读取 raw/delayed action 诊断。"""
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None:
        return None
    for term_name in action_manager.active_terms:
        term = action_manager.get_term(term_name)
        if getattr(term, "_entity", None) is robot:
            return term
    return None


def _terrain_yaw_logs(
    env: ManagerBasedRlEnv,
    yaw_rate_abs: torch.Tensor,
    wheel_yaw_abs: torch.Tensor,
) -> dict[str, float]:
    """按每类地形拆分 yaw 诊断，定位是否某个地形触发自转。"""
    terrain = getattr(env.scene, "terrain", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    cfg = getattr(terrain, "cfg", None)
    generator_cfg = getattr(cfg, "terrain_generator", None)
    sub_terrains = getattr(generator_cfg, "sub_terrains", {}) or {}
    if not isinstance(terrain_types, torch.Tensor) or not sub_terrains:
        return {}
    terrain_types = terrain_types.to(device=env.device).reshape(-1)
    logs: dict[str, float] = {}
    for terrain_index, terrain_name in enumerate(sub_terrains):
        mask = terrain_types == terrain_index
        safe_name = str(terrain_name).replace("/", "_")
        logs[f"StairYaw/terrain_{safe_name}_base_yaw_abs"] = _masked_mean(yaw_rate_abs, mask)
        logs[f"StairYaw/terrain_{safe_name}_wheel_yaw_abs"] = _masked_mean(wheel_yaw_abs, mask)
    return logs
