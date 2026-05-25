"""EFGCL 跳跃辅助。

该模块只在 PreTrain 训练期使用。PreTrain 按 EFGCL 论文的 Jump 任务范式设计：
- 在固定时间窗内施加竖直外力，让策略早期体验成功跳跃状态。
- 外力由成功率课程逐步衰减到 0，最终策略必须靠轮地接触主动起跳。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

from se3_train.mdp.jump_commands import JumpCommandTerm

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_UPRIGHT_ASSIST_SCALE_ATTR = "_efgcl_assist_scale"
_UPRIGHT_SUCCESS_EMA_ATTR = "_efgcl_upright_success_ema"
_MEAN_TORQUE_ATTR = "_efgcl_mean_torque_nm"
_TAKEOFF_ASSIST_SCALE_ATTR = "_efgcl_takeoff_assist_scale"
_TAKEOFF_SUCCESS_EMA_ATTR = "_efgcl_takeoff_success_ema"
_MEAN_FORCE_ATTR = "_efgcl_takeoff_mean_force_n"
_TAKEOFF_ASSISTED_MASK_ATTR = "_efgcl_takeoff_assisted_mask"
_TAKEOFF_ASSIST_BODY_IDS_ATTR = "_efgcl_takeoff_assist_body_ids"
_PREV_JUMP_FLAG_ATTR = "_efgcl_prev_jump_flag"
_HEIGHT_PREV_JUMP_FLAG_ATTR = "_efgcl_height_prev_jump_flag"
_MAX_BASE_HEIGHT_ATTR = "_efgcl_max_base_height"
_MAX_WHEEL_HEIGHT_ATTR = "_efgcl_max_wheel_height"
_G = 9.81
_TAKEOFF_ASSIST_BODY_NAMES: tuple[str, ...] = (
    "base_force_pt_fl",
    "base_force_pt_fr",
    "base_force_pt_bl",
    "base_force_pt_br",
)


def _get_jump_term(env: ManagerBasedRlEnv, command_name: str) -> JumpCommandTerm | None:
    """读取跳跃指令项；非跳跃任务返回 None。"""
    if not hasattr(env, "command_manager"):
        return None
    try:
        term = env.command_manager.get_term(command_name)
    except Exception:
        return None
    return term if isinstance(term, JumpCommandTerm) else None


def _log_efgcl_metrics(
    env: ManagerBasedRlEnv,
    upright_assist_scale: float,
    takeoff_assist_scale: float,
    mean_torque_nm: float,
    mean_force_n: float,
    upright_success_rate: float,
    takeoff_success_rate: float,
    takeoff_assisted_ratio: float,
) -> None:
    """写入 EFGCL 训练诊断指标。"""
    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                # assist_scale 保留为兼容旧 dashboard 的总辅助强度。
                "EFGCL/assist_scale": max(upright_assist_scale, takeoff_assist_scale),
                "EFGCL/upright_assist_scale": upright_assist_scale,
                "EFGCL/takeoff_assist_scale": takeoff_assist_scale,
                "EFGCL/mean_torque_nm": mean_torque_nm,
                "EFGCL/mean_force_n": mean_force_n,
                "EFGCL/takeoff_assisted_ratio": takeoff_assisted_ratio,
                "Jump/diag_upright_success_rate": upright_success_rate,
                "Jump/diag_takeoff_success_rate": takeoff_success_rate,
            }
        )


def _robot_mass(env: ManagerBasedRlEnv, num_envs: int, device: torch.device) -> torch.Tensor:
    """读取每个 env 的机器人总质量，兼容域随机化后的 batched model。"""
    body_mass = env.sim.model.body_mass
    mass = body_mass.sum(dim=1) if body_mass.ndim == 2 else body_mass.sum().expand(num_envs)
    return mass.to(device=device)


def _projectile_feedforward_force(
    target_height: torch.Tensor,
    robot_mass: torch.Tensor,
    assist_duration_s: float,
    assist_force_fraction: float,
    max_force_n: float,
) -> torch.Tensor:
    """按 EFGCL 论文的跳跃外力公式计算前馈辅助力。

    论文在左右肩胛点各施加一次向上力：
        f = mg / 2 * (1 + sqrt(1 + 8h / (g * dt^2)))

    这里返回的是总合力，后续会平均分到四个对称挂点，避免单点受力引入额外力矩。
    actual vz 不参与计算，避免外力变成速度误差补偿器。
    """
    dt = max(float(assist_duration_s), 1e-3)
    height = torch.clamp(target_height, min=0.01)
    root = torch.sqrt(1.0 + 8.0 * height / (_G * dt * dt))
    force_per_point = robot_mass * _G * 0.5 * (1.0 + root)
    force = 2.0 * force_per_point * float(assist_force_fraction)
    return torch.clamp(force, min=0.0, max=float(max_force_n))


def _fixed_time_window(
    term: JumpCommandTerm,
    start_s: float,
    end_s: float,
    policy_dt_s: float,
) -> torch.Tensor:
    """按 EFGCL 论文的固定时间窗判断当前是否施加起跳辅助。"""
    dt = max(float(policy_dt_s), 1.0e-4)
    start_step = max(0, round(float(start_s) / dt))
    end_step = max(start_step + 1, round(float(end_s) / dt))
    return (term.traj_step >= start_step) & (term.traj_step < end_step)


def _after_fixed_time_window(
    term: JumpCommandTerm,
    end_s: float,
    policy_dt_s: float,
) -> torch.Tensor:
    """固定起跳辅助窗口结束后的成功率评估区间。"""
    dt = max(float(policy_dt_s), 1.0e-4)
    end_step = max(1, round(float(end_s) / dt))
    return term.traj_step >= end_step


def _wheel_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float,
) -> torch.Tensor:
    """计算左右轮底部相对地面的最小高度。"""
    robot = env.scene[asset_cfg.name]
    attr_name = f"_efgcl_wheel_body_ids_{asset_cfg.name}"
    body_ids = getattr(env, attr_name, None)
    if not isinstance(body_ids, list) or len(body_ids) != 2:
        body_ids, _ = robot.find_bodies(("l_wheel_Link", "r_wheel_Link"), preserve_order=True)
        if len(body_ids) != 2:
            return torch.zeros(env.num_envs, device=env.device)
        setattr(env, attr_name, body_ids)
    wheel_pos_w = robot.data.body_link_pos_w[:, body_ids, :]
    ground_z = env.scene.env_origins[:, 2].unsqueeze(1)
    wheel_bottom_h = wheel_pos_w[:, :, 2] - ground_z - float(wheel_radius)
    return torch.min(wheel_bottom_h, dim=1).values


def _takeoff_assist_body_ids(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> list[int]:
    """获取起跳外力使用的四个对称挂点。"""
    attr_name = f"{_TAKEOFF_ASSIST_BODY_IDS_ATTR}_{asset_cfg.name}"
    body_ids = getattr(env, attr_name, None)
    if isinstance(body_ids, list) and len(body_ids) == 4:
        return body_ids

    robot = env.scene[asset_cfg.name]
    body_ids, body_names = robot.find_bodies(_TAKEOFF_ASSIST_BODY_NAMES, preserve_order=True)
    if len(body_ids) != 4:
        raise RuntimeError(f"必须找到四个对称受力点，实际找到: {body_names}")
    setattr(env, attr_name, body_ids)
    return body_ids


def _update_jump_max_heights(
    env: ManagerBasedRlEnv,
    jump_flag: torch.Tensor,
    base_height: torch.Tensor,
    wheel_height: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """维护每次跳跃窗口内的 base 和轮组最大高度。"""
    max_base = getattr(env, _MAX_BASE_HEIGHT_ATTR, None)
    max_wheel = getattr(env, _MAX_WHEEL_HEIGHT_ATTR, None)
    if not isinstance(max_base, torch.Tensor) or max_base.shape != jump_flag.shape:
        max_base = base_height.clone()
    if not isinstance(max_wheel, torch.Tensor) or max_wheel.shape != jump_flag.shape:
        max_wheel = wheel_height.clone()

    prev_jump = getattr(env, _HEIGHT_PREV_JUMP_FLAG_ATTR, None)
    if not isinstance(prev_jump, torch.Tensor) or prev_jump.shape != jump_flag.shape:
        prev_jump = torch.zeros_like(jump_flag)
    new_jump = jump_flag & ~prev_jump

    max_base = torch.where(new_jump, base_height, max_base)
    max_wheel = torch.where(new_jump, wheel_height, max_wheel)
    max_base = torch.where(jump_flag, torch.maximum(max_base, base_height), base_height)
    max_wheel = torch.where(jump_flag, torch.maximum(max_wheel, wheel_height), wheel_height)

    setattr(env, _MAX_BASE_HEIGHT_ATTR, max_base.detach())
    setattr(env, _MAX_WHEEL_HEIGHT_ATTR, max_wheel.detach())
    setattr(env, _HEIGHT_PREV_JUMP_FLAG_ATTR, jump_flag.detach())
    return max_base, max_wheel


def _takeoff_assist_probability(
    assist_scale: float,
    min_unassisted_probe_ratio: float,
) -> float:
    """把课程 scale 转成每次跳跃的起跳辅助采样概率。

    即使 scale=1，也保留一部分无辅助 probe。这样 PPO 会持续看到 sim2sim 同构
    的样本，EFGCL 只扩展状态分布，不把全部起跳信用交给外力。
    """
    probe_ratio = max(0.0, min(1.0, float(min_unassisted_probe_ratio)))
    return max(0.0, min(float(assist_scale), 1.0 - probe_ratio))


def _unassisted_takeoff_success_window(
    active_airborne: torch.Tensor,
    assisted_mask: torch.Tensor,
) -> torch.Tensor:
    """只用无辅助样本评估起跳撤力成功率。"""
    return active_airborne & ~assisted_mask


def _sample_takeoff_assisted_mask(
    env: ManagerBasedRlEnv,
    jump_flag: torch.Tensor,
    assist_probability: float,
) -> torch.Tensor:
    """按每个 jump episode 采样是否施加起跳辅助。"""
    previous_jump_flag = getattr(env, _PREV_JUMP_FLAG_ATTR, None)
    if (
        not isinstance(previous_jump_flag, torch.Tensor)
        or previous_jump_flag.shape != jump_flag.shape
    ):
        previous_jump_flag = torch.zeros_like(jump_flag)

    assisted_mask = getattr(env, _TAKEOFF_ASSISTED_MASK_ATTR, None)
    if not isinstance(assisted_mask, torch.Tensor) or assisted_mask.shape != jump_flag.shape:
        assisted_mask = torch.zeros_like(jump_flag)

    new_jump = jump_flag & ~previous_jump_flag
    if new_jump.any():
        sampled = torch.rand(jump_flag.shape, device=jump_flag.device) < assist_probability
        assisted_mask = torch.where(new_jump, sampled, assisted_mask)

    assisted_mask = assisted_mask & jump_flag
    setattr(env, _TAKEOFF_ASSISTED_MASK_ATTR, assisted_mask.detach())
    setattr(env, _PREV_JUMP_FLAG_ATTR, jump_flag.detach())
    return assisted_mask


def apply_efgcl_jump_guidance(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    command_name: str,
    max_takeoff_force_n: float = 260.0,
    takeoff_assist_start_s: float = 1.0,
    takeoff_assist_end_s: float = 1.1,
    policy_dt_s: float = 0.01,
    takeoff_assist_duration_s: float = 0.10,
    takeoff_assist_force_fraction: float = 0.5,
    min_unassisted_takeoff_probe_ratio: float = 0.35,
    min_ref_takeoff_vz: float = 0.05,
    takeoff_success_vz_ratio: float = 0.35,
    takeoff_success_min_vz: float = 0.2,
    stiffness_nm: float = 1.2,
    damping_nms: float = 0.08,
    max_torque_nm: float = 0.8,
    success_tilt_limit_deg: float = 25.0,
    success_vz_threshold: float = 1.0,
    success_height_tolerance: float = 0.10,
    base_height_offset: float = 0.26,
    wheel_radius: float = 0.059,
    ema_alpha: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """施加 EFGCL 起跳外力和空中直立力矩。

    PreTrain 使用论文式固定时间窗，而不是参考轨迹相位：
    - 起跳外力默认在 1.0s 到 1.1s 施加。
    - 力大小由目标高度、机器人质量和辅助时长按抛体模型前馈计算。
    - 成功率按本次跳跃窗口内轮组和 base 的最大高度是否接近目标评估。

    空中力矩模型是虚拟万向弹簧：
        torque_body = [K * pg_y, -K * pg_x, 0] - D * ang_vel_xy

    projected_gravity_b 的直立目标为 [0, 0, -1]。小角度下 pg_x/pg_y
    近似 pitch/roll 误差，起跳外力会平均分到四个对称挂点，外部力矩在世界系
    写入 MuJoCo xfrc_applied。
    """
    del env_ids  # step event 对所有 env 生效。

    robot = env.scene[asset_cfg.name]
    root_body_id = 0
    takeoff_body_ids = _takeoff_assist_body_ids(env, asset_cfg)
    apply_body_ids = [*takeoff_body_ids, root_body_id]

    # 外部力矩是持久数据，每步先清零，避免离开空中阶段后残留。
    zero_force = torch.zeros(env.num_envs, len(apply_body_ids), 3, device=env.device)
    zero_torque = torch.zeros_like(zero_force)
    robot.data.write_external_wrench(
        force=zero_force,
        torque=zero_torque,
        body_ids=apply_body_ids,
    )

    term = _get_jump_term(env, command_name)
    if term is None:
        _log_efgcl_metrics(env, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return

    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    force_window = _fixed_time_window(
        term,
        takeoff_assist_start_s,
        takeoff_assist_end_s,
        policy_dt_s,
    )
    active_takeoff_window = jump_flag & force_window
    after_force_window = jump_flag & _after_fixed_time_window(
        term,
        takeoff_assist_end_s,
        policy_dt_s,
    )
    airborne = after_force_window
    active_airborne = jump_flag & airborne
    takeoff_assist_scale = float(getattr(env, _TAKEOFF_ASSIST_SCALE_ATTR, 1.0))
    upright_assist_scale = float(getattr(env, _UPRIGHT_ASSIST_SCALE_ATTR, 1.0))
    assist_probability = _takeoff_assist_probability(
        takeoff_assist_scale,
        min_unassisted_takeoff_probe_ratio,
    )
    takeoff_assisted_mask = _sample_takeoff_assisted_mask(env, jump_flag, assist_probability)

    pg = robot.data.projected_gravity_b
    vz_w = robot.data.root_link_lin_vel_w[:, 2]
    h_target = cmd[:, 6]
    active_takeoff = active_takeoff_window

    base_height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    wheel_h = _wheel_height(env, asset_cfg, wheel_radius)
    max_base_h, max_wheel_h = _update_jump_max_heights(env, jump_flag, base_height, wheel_h)

    del min_ref_takeoff_vz, takeoff_success_vz_ratio, takeoff_success_min_vz
    target_base_h = h_target + float(base_height_offset)
    height_tol = float(success_height_tolerance)
    takeoff_success_window = after_force_window
    takeoff_success = (
        takeoff_success_window
        & (torch.abs(max_wheel_h - h_target) < height_tol)
        & (torch.abs(max_base_h - target_base_h) < height_tol)
    )
    if takeoff_success_window.any():
        batch_takeoff_rate = (
            takeoff_success.float().sum() / takeoff_success_window.float().sum().clamp_min(1.0)
        )
        previous_takeoff = getattr(env, _TAKEOFF_SUCCESS_EMA_ATTR, None)
        if previous_takeoff is None:
            takeoff_success_ema = batch_takeoff_rate
        else:
            takeoff_success_ema = (
                1.0 - ema_alpha
            ) * previous_takeoff + ema_alpha * batch_takeoff_rate
        setattr(env, _TAKEOFF_SUCCESS_EMA_ATTR, takeoff_success_ema.detach())
        takeoff_success_rate = float(takeoff_success_ema.item())
    else:
        previous_takeoff = getattr(env, _TAKEOFF_SUCCESS_EMA_ATTR, None)
        takeoff_success_rate = 0.0 if previous_takeoff is None else float(previous_takeoff.item())

    tilt_rad = torch.acos(torch.clamp(-pg[:, 2], -1.0, 1.0))
    success_tilt_limit = torch.deg2rad(torch.tensor(success_tilt_limit_deg, device=env.device))
    upright_success = (
        active_airborne & (vz_w > success_vz_threshold) & (tilt_rad < success_tilt_limit)
    )

    if active_airborne.any():
        batch_rate = upright_success.float().sum() / active_airborne.float().sum().clamp_min(1.0)
        previous = getattr(env, _UPRIGHT_SUCCESS_EMA_ATTR, None)
        if previous is None:
            success_ema = batch_rate
        else:
            success_ema = (1.0 - ema_alpha) * previous + ema_alpha * batch_rate
        setattr(env, _UPRIGHT_SUCCESS_EMA_ATTR, success_ema.detach())
        upright_success_rate = float(success_ema.item())
    else:
        previous = getattr(env, _UPRIGHT_SUCCESS_EMA_ATTR, None)
        upright_success_rate = 0.0 if previous is None else float(previous.item())

    force_w = zero_force.clone()
    total_force_w = torch.zeros(env.num_envs, 3, device=env.device)
    force_active = active_takeoff & takeoff_assisted_mask
    if takeoff_assist_scale > 0.0 and force_active.any():
        mass = _robot_mass(env, env.num_envs, env.device)
        total_force_w[:, 2] = (
            _projectile_feedforward_force(
                target_height=h_target,
                robot_mass=mass,
                assist_duration_s=takeoff_assist_duration_s,
                assist_force_fraction=takeoff_assist_force_fraction,
                max_force_n=max_takeoff_force_n,
            )
            * force_active.float()
        )
        point_force_w = total_force_w[:, 2] / float(len(takeoff_body_ids))
        force_w[:, : len(takeoff_body_ids), 2] = point_force_w.unsqueeze(1).expand(
            -1,
            len(takeoff_body_ids),
        )
        force_w *= takeoff_assist_scale
        mean_force_n = float(
            torch.norm(total_force_w[force_active] * takeoff_assist_scale, dim=1).mean().item()
        )
    else:
        mean_force_n = 0.0

    ang_vel_b = robot.data.root_link_ang_vel_b
    torque_b = torch.zeros(env.num_envs, 3, device=env.device)
    torque_b[:, 0] = stiffness_nm * pg[:, 1] - damping_nms * ang_vel_b[:, 0]
    torque_b[:, 1] = -stiffness_nm * pg[:, 0] - damping_nms * ang_vel_b[:, 1]
    torque_b = torch.clamp(torque_b, min=-max_torque_nm, max=max_torque_nm)
    torque_b *= active_airborne.float().unsqueeze(-1) * upright_assist_scale

    torque_w = quat_apply(robot.data.root_link_quat_w, torque_b)
    force_w[:, len(takeoff_body_ids), :] = 0.0
    torque_w_full = torch.zeros_like(force_w)
    torque_w_full[:, len(takeoff_body_ids), :] = torque_w
    robot.data.write_external_wrench(force=force_w, torque=torque_w_full, body_ids=apply_body_ids)

    if upright_assist_scale > 0.0 and active_airborne.any():
        mean_torque = torch.norm(torque_b[active_airborne], dim=1).mean()
        mean_torque_nm = float(mean_torque.item())
    else:
        mean_torque_nm = 0.0
    setattr(env, _MEAN_TORQUE_ATTR, mean_torque_nm)
    setattr(env, _MEAN_FORCE_ATTR, mean_force_n)
    if jump_flag.any():
        takeoff_assisted_ratio = float(takeoff_assisted_mask[jump_flag].float().mean().item())
    else:
        takeoff_assisted_ratio = 0.0
    _log_efgcl_metrics(
        env,
        upright_assist_scale,
        takeoff_assist_scale,
        mean_torque_nm,
        mean_force_n,
        upright_success_rate,
        takeoff_success_rate,
        takeoff_assisted_ratio,
    )


def apply_airborne_upright_spotting(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    command_name: str,
    stiffness_nm: float = 1.2,
    damping_nms: float = 0.08,
    max_torque_nm: float = 0.8,
    success_tilt_limit_deg: float = 25.0,
    success_vz_threshold: float = 1.0,
    ema_alpha: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """兼容旧配置名：只传空中参数时仍走完整 EFGCL 事件。"""
    apply_efgcl_jump_guidance(
        env=env,
        env_ids=env_ids,
        command_name=command_name,
        stiffness_nm=stiffness_nm,
        damping_nms=damping_nms,
        max_torque_nm=max_torque_nm,
        success_tilt_limit_deg=success_tilt_limit_deg,
        success_vz_threshold=success_vz_threshold,
        ema_alpha=ema_alpha,
        asset_cfg=asset_cfg,
    )
