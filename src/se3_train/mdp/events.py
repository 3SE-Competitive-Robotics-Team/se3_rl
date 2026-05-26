"""SE3 轮腿机器人的域随机化事件。

与原始 Isaac Gym 实现配置保持一致。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    euler_xyz_from_quat,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
)

from se3_shared import JointGroup
from se3_train.mdp.jump_commands import JumpCommandTerm
from se3_train.mdp.jump_trajectories import (
    DEFAULT_JUMP_TRAJ_HEIGHTS,
    DEFAULT_JUMP_TRAJ_PATHS,
    JumpTrajLibrary,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _pre_resample_jump_command_for_reset(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str = "velocity_height",
) -> None:
    """在 reset 写状态前预采样跳跃指令，保证 RSI 读取新 episode 的 jump_flag。"""
    if not hasattr(env, "command_manager"):
        return
    try:
        term = env.command_manager.get_term(command_name)
    except Exception:
        return
    if isinstance(term, JumpCommandTerm):
        term.pre_resample_for_reset(env_ids)


def reset_root_state_full(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """重置 base 到默认站立状态,yaw 随机,xy 小偏移。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    _pre_resample_jump_command_for_reset(env, env_ids)

    asset: Entity = env.scene[asset_cfg.name]
    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    root_states = default_root_state[env_ids].clone()

    n = len(env_ids)
    pos = root_states[:, 0:3].clone()
    pos[:, 0] += sample_uniform(
        torch.tensor(-0.1, device=env.device),
        torch.tensor(0.1, device=env.device),
        (n,),
        env.device,
    )
    pos[:, 1] += sample_uniform(
        torch.tensor(-0.1, device=env.device),
        torch.tensor(0.1, device=env.device),
        (n,),
        env.device,
    )
    pos[:, 0:3] += env.scene.env_origins[env_ids]

    # 仅随机化 yaw,保持直立。
    yaw = sample_uniform(
        torch.tensor(-torch.pi, device=env.device),
        torch.tensor(torch.pi, device=env.device),
        (n,),
        env.device,
    )
    roll = torch.zeros(n, device=env.device)
    pitch = torch.zeros(n, device=env.device)
    quat_delta = quat_from_euler_xyz(roll, pitch, yaw)
    default_quat = root_states[:, 3:7]
    new_quat = quat_mul(default_quat, quat_delta)

    vel = torch.zeros(n, 6, device=env.device)

    # jump_flag=1 的 episode：从参考轨迹初始化。
    #
    # 设计原则：
    #   - PreTrain 可从随机参考帧开始，覆盖起跳、空中和落地状态分布
    #   - 注入参考帧的 base_pos_z、base_vel、q_ref、q_vel，确保状态与参考相位一致
    #   - 起点帧存入 env._rsi_traj_frame[env_ids] 供 reset_joints 读取
    #   - rsi_takeoff_prob < 1.0 时，部分 jump episode 回退到预蹲姿态，仅用于显式消融
    if hasattr(env, "command_manager"):
        try:
            term = env.command_manager.get_term("velocity_height")
            rsi_takeoff_prob = (
                term.cfg.rsi_takeoff_prob if isinstance(term, JumpCommandTerm) else 1.0
            )
            cmd = env.command_manager.get_command("velocity_height")
            jump_mask = cmd[env_ids, 5] > 0.5
            if jump_mask.any():
                # 决定哪些 env 使用轨迹起点初始化（非全量时随机跳过部分）
                rsi_mask = jump_mask
                if rsi_takeoff_prob < 1.0:
                    rand = torch.rand(len(env_ids), device=env.device)
                    rsi_mask = jump_mask & (rand < rsi_takeoff_prob)

                # 初始化轨迹起点帧缓存（供 reset_joints 读取）
                if not hasattr(env, "_rsi_traj_frame"):
                    env._rsi_traj_frame = torch.full(
                        (env.num_envs,), -1, dtype=torch.long, device=env.device
                    )
                env._rsi_traj_frame[env_ids] = -1  # 默认不做 RSI

                if rsi_mask.any():
                    # 按 jump_target_height 最近邻匹配轨迹，可从随机帧初始化。
                    traj_paths = (
                        term.cfg.traj_paths
                        if isinstance(term, JumpCommandTerm)
                        else DEFAULT_JUMP_TRAJ_PATHS
                    )
                    traj_heights = (
                        term.cfg.traj_target_heights
                        if isinstance(term, JumpCommandTerm)
                        else DEFAULT_JUMP_TRAJ_HEIGHTS
                    )
                    library = JumpTrajLibrary.get(traj_paths, traj_heights, str(env.device))
                    h_targets = cmd[env_ids[rsi_mask], 6]
                    n_rsi = rsi_mask.sum().item()
                    frame_rsi = torch.zeros(n_rsi, dtype=torch.long, device=env.device)
                    if isinstance(term, JumpCommandTerm) and term.cfg.rsi_random_frame:
                        max_step = library.n_steps_for(h_targets) - 1
                        phase_lo, phase_hi = term.cfg.rsi_frame_phase_range
                        lo_step = torch.clamp((max_step.float() * phase_lo).round().long(), min=0)
                        hi_step = torch.clamp(
                            (max_step.float() * phase_hi).round().long(),
                            min=0,
                        )
                        hi_step = torch.maximum(lo_step, hi_step)
                        rand = torch.rand(n_rsi, device=env.device)
                        frame_rsi = (
                            (lo_step.float() + rand * (hi_step - lo_step).float()).round().long()
                        )

                    ref_pos, ref_vel, _, _, _, _ = library.gather(h_targets, frame_rsi)

                    # 写入参考线速度和 base 高度；xy 位置仍用当前 env 原点附近的小随机偏移。
                    vel[rsi_mask, 0:3] = ref_vel
                    pos[rsi_mask, 2] = ref_pos[:, 2] + env.scene.env_origins[env_ids][rsi_mask, 2]

                    # 缓存帧号供 reset_joints 使用，并同步 command 里的参考相位。
                    env._rsi_traj_frame[env_ids[rsi_mask]] = frame_rsi
                    if isinstance(term, JumpCommandTerm):
                        term.set_reference_frame(env_ids[rsi_mask], frame_rsi)
        except Exception:
            pass

    asset.write_root_link_pose_to_sim(torch.cat([pos, new_quat], dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(vel, env_ids=env_ids)

    # 6DoF base tracking 需要以 reset 后的随机 xy/yaw 为参考零点。
    # 直接跟世界系 x=0/yaw=0 会和 reset_root_state_full 的随机化冲突。
    if not hasattr(env, "_jump_pose_ref_pos_w"):
        env._jump_pose_ref_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
    if not hasattr(env, "_jump_pose_ref_yaw"):
        env._jump_pose_ref_yaw = torch.zeros(env.num_envs, device=env.device)
    _, _, yaw_ref = euler_xyz_from_quat(new_quat)
    env._jump_pose_ref_pos_w[env_ids] = pos
    env._jump_pose_ref_yaw[env_ids] = yaw_ref


def reset_joints(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """重置关节位置到默认站立姿态(default_joint_pos)附近小范围随机。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    joint_pos[:, JointGroup.WHEELS] = 0.0

    # jump_flag=1 的 episode：RSI 关节角——从参考轨迹对应帧注入
    # reset_root_state_full 已缓存帧号到 env._rsi_traj_frame，供此处读取
    # 确保关节角与注入的 base_pos_z/base_vel_z 对应，初始状态在轨迹流形上
    # 若缓存不存在（非 RSI 帧或行走 episode），回退到固定预蹲姿态
    _jump_hip_fallback = 0.85  # rad，无轨迹数据时的回退预蹲髋角
    _jump_knee_fallback = 0.55  # rad，无轨迹数据时的回退预蹲膝角
    if hasattr(env, "command_manager"):
        try:
            cmd = env.command_manager.get_command("velocity_height")
            jump_mask = cmd[env_ids, 5] > 0.5
            if jump_mask.any():
                rsi_frames = getattr(env, "_rsi_traj_frame", None)
                # rsi_done_local：local index 中已从轨迹注入关节角的位置（用于计算回退 mask）
                rsi_done_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)

                if rsi_frames is not None:
                    frames = rsi_frames[env_ids]
                    rsi_done_mask = jump_mask & (frames >= 0)
                    if rsi_done_mask.any():
                        term = env.command_manager.get_term("velocity_height")
                        traj_paths = (
                            term.cfg.traj_paths
                            if isinstance(term, JumpCommandTerm)
                            else DEFAULT_JUMP_TRAJ_PATHS
                        )
                        traj_heights = (
                            term.cfg.traj_target_heights
                            if isinstance(term, JumpCommandTerm)
                            else DEFAULT_JUMP_TRAJ_HEIGHTS
                        )
                        library = JumpTrajLibrary.get(traj_paths, traj_heights, str(env.device))
                        h_targets = cmd[env_ids[rsi_done_mask], 6]
                        _, _, q_ref, q_vel, _, _ = library.gather(h_targets, frames[rsi_done_mask])
                        # q_ref/q_vel: [lf0, lf1, lw, rf0, rf1, rw]
                        joint_pos[rsi_done_mask, JointGroup.LEGS[0]] = q_ref[:, 0]
                        joint_pos[rsi_done_mask, JointGroup.LEGS[1]] = q_ref[:, 1]
                        joint_pos[rsi_done_mask, JointGroup.LEGS[2]] = q_ref[:, 3]
                        joint_pos[rsi_done_mask, JointGroup.LEGS[3]] = q_ref[:, 4]
                        joint_vel[rsi_done_mask, JointGroup.LEGS[0]] = q_vel[:, 0]
                        joint_vel[rsi_done_mask, JointGroup.LEGS[1]] = q_vel[:, 1]
                        joint_vel[rsi_done_mask, JointGroup.LEGS[2]] = q_vel[:, 3]
                        joint_vel[rsi_done_mask, JointGroup.LEGS[3]] = q_vel[:, 4]

                # 未做 RSI 的 jump env 默认保持站立姿态。
                # 只有显式启用部分 RSI 时，未注入的样本才回退到预蹲姿态做消融。
                term = env.command_manager.get_term("velocity_height")
                use_fallback_squat = not (
                    isinstance(term, JumpCommandTerm) and term.cfg.rsi_takeoff_prob <= 0.0
                )
                fallback_mask = jump_mask & ~rsi_done_mask & use_fallback_squat
                if fallback_mask.any():
                    joint_pos[fallback_mask, JointGroup.LEGS[0]] = _jump_hip_fallback
                    joint_pos[fallback_mask, JointGroup.LEGS[1]] = _jump_knee_fallback
                    joint_pos[fallback_mask, JointGroup.LEGS[2]] = _jump_hip_fallback
                    joint_pos[fallback_mask, JointGroup.LEGS[3]] = _jump_knee_fallback
        except Exception:
            pass

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def push_robots(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """随机速度推动。"""
    asset: Entity = env.scene[asset_cfg.name]
    vel_w = asset.data.root_link_vel_w[env_ids]

    range_list = [
        velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=env.device)
    vel_w += sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=env.device)
    asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)


def randomize_friction(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    friction_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """随机化几何体摩擦系数。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    _ = env.scene[asset_cfg.name]
    n = len(env_ids)

    friction = sample_uniform(
        torch.tensor(friction_range[0], device=env.device),
        torch.tensor(friction_range[1], device=env.device),
        (n, 1),
        env.device,
    )

    # 写入该实体的所有几何体。
    geom_ids = asset_cfg.geom_ids
    if isinstance(geom_ids, slice):
        env.sim.model.geom_friction[env_ids, :, 0] = friction
    else:
        for gid in geom_ids:
            env.sim.model.geom_friction[env_ids, gid, 0] = friction.squeeze(-1)


def randomize_restitution(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    restitution_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """随机化几何体恢复系数。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    _ = env.scene[asset_cfg.name]
    n = len(env_ids)

    _ = sample_uniform(
        torch.tensor(restitution_range[0], device=env.device),
        torch.tensor(restitution_range[1], device=env.device),
        (n, 1),
        env.device,
    )

    geom_ids = asset_cfg.geom_ids
    if isinstance(geom_ids, slice):
        env.sim.model.geom_margin[env_ids, :] = 0.0
        # MuJoCo 没有完全相同的逐几何体恢复系数;
        # 我们使用 solref/solimp 来设置接触属性。
        # 这是随机化范围的占位符。
    else:
        for gid in geom_ids:
            env.sim.model.geom_margin[env_ids, gid] = 0.0


def randomize_base_mass(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    mass_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """随机化基座连杆附加质量。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    _ = env.scene[asset_cfg.name]
    n = len(env_ids)

    default_mass = env.sim.get_default_field("body_mass")
    base_body_idx = 0  # base_link 是第 0 个 body。

    added_mass = sample_uniform(
        torch.tensor(mass_range[0], device=env.device),
        torch.tensor(mass_range[1], device=env.device),
        (n,),
        env.device,
    )

    env.sim.model.body_mass[env_ids, base_body_idx] = default_mass[base_body_idx] + added_mass


def randomize_inertia(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    inertia_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """随机化基座连杆惯性。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    n = len(env_ids)
    default_inertia = env.sim.get_default_field("body_inertia")
    base_body_idx = 0

    scale = sample_uniform(
        torch.tensor(inertia_range[0], device=env.device),
        torch.tensor(inertia_range[1], device=env.device),
        (n, 3),
        env.device,
    )

    env.sim.model.body_inertia[env_ids, base_body_idx] = default_inertia[base_body_idx] * scale


def randomize_com(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    com_range: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """随机化基座连杆质心偏移。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    n = len(env_ids)
    default_ipos = env.sim.get_default_field("body_ipos")
    base_body_idx = 0

    offset = sample_uniform(
        torch.tensor(-com_range, device=env.device),
        torch.tensor(com_range, device=env.device),
        (n, 3),
        env.device,
    )

    env.sim.model.body_ipos[env_ids, base_body_idx] = default_ipos[base_body_idx] + offset


def randomize_pd_gains(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    kp_range: tuple[float, float],
    kd_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """随机化 PD 增益(缩放默认增益)。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    _ = env.scene[asset_cfg.name]
    n = len(env_ids)

    kp_scale = sample_uniform(
        torch.tensor(kp_range[0], device=env.device),
        torch.tensor(kp_range[1], device=env.device),
        (n, 1),
        env.device,
    )
    kd_scale = sample_uniform(
        torch.tensor(kd_range[0], device=env.device),
        torch.tensor(kd_range[1], device=env.device),
        (n, 1),
        env.device,
    )

    # 随机化执行器增益(gainprm 和 biasprm)。
    default_gainprm = env.sim.get_default_field("actuator_gainprm")
    default_biasprm = env.sim.get_default_field("actuator_biasprm")

    actuator_ids = asset_cfg.actuator_ids
    if isinstance(actuator_ids, slice):
        env.sim.model.actuator_gainprm[env_ids, :, 0] = default_gainprm[:, 0] * kp_scale
        env.sim.model.actuator_biasprm[env_ids, :, 1] = default_biasprm[:, 1] * kp_scale
        env.sim.model.actuator_biasprm[env_ids, :, 2] = default_biasprm[:, 2] * kd_scale
    else:
        for aid in actuator_ids:
            env.sim.model.actuator_gainprm[env_ids, aid, 0] = default_gainprm[
                aid, 0
            ] * kp_scale.squeeze(-1)
            env.sim.model.actuator_biasprm[env_ids, aid, 1] = default_biasprm[
                aid, 1
            ] * kp_scale.squeeze(-1)
            env.sim.model.actuator_biasprm[env_ids, aid, 2] = default_biasprm[
                aid, 2
            ] * kd_scale.squeeze(-1)


def randomize_default_dof_pos(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    offset_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """随机化默认关节位置。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    n = len(env_ids)

    offset = sample_uniform(
        torch.tensor(offset_range[0], device=env.device),
        torch.tensor(offset_range[1], device=env.device),
        (n, len(JointGroup.ALL)),
        env.device,
    )

    default_joint_pos = asset.data.default_joint_pos.clone()
    default_joint_pos[env_ids[:, None], torch.tensor(JointGroup.ALL, device=env.device)] += offset

    # 裁剪到关节限制范围内。
    soft_limits = asset.data.soft_joint_pos_limits
    if soft_limits is not None:
        ctrl_idx = torch.tensor(JointGroup.ALL, device=env.device)
        default_joint_pos[env_ids[:, None], ctrl_idx] = torch.clamp(
            default_joint_pos[env_ids[:, None], ctrl_idx],
            soft_limits[env_ids[:, None], ctrl_idx, 0],
            soft_limits[env_ids[:, None], ctrl_idx, 1],
        )

    asset.data.default_joint_pos[env_ids] = default_joint_pos[env_ids]
