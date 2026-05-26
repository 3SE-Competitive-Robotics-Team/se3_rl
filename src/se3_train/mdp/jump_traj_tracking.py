"""跳跃参考轨迹 tracking 奖励。

将 TO 生成的参考轨迹（base_pos、base_vel、q_ref）逐帧接入奖励，
引导策略复现完整的蹲→跳→收腿→展腿→缓冲动作时序。

设计原则：
- 多条轨迹按 jump_target_height 最近邻匹配
- 用 traj_step 计步器索引当前帧，traj_step 由 reference motion 时间推进
- 三个维度分离，权重独立可调：高度、速度、关节角
- 容忍带（tolerance）防止在轨迹误差较大时奖励函数反向梯度过强
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from se3_shared import JointGroup
from se3_train.mdp.jump_commands import JumpCommandTerm
from se3_train.mdp.jump_trajectories import JumpTrajLibrary

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _get_jump_term(env: ManagerBasedRlEnv, command_name: str) -> JumpCommandTerm:
    term = env.command_manager.get_term(command_name)
    assert isinstance(term, JumpCommandTerm)
    return term


def _as_tuple(value: str | tuple[str, ...]) -> tuple[str, ...]:
    """兼容单轨迹字符串和多轨迹 tuple 配置。"""
    if isinstance(value, str):
        return (value,)
    return tuple(value)


# ---------------------------------------------------------------------------
# tracking 奖励函数
# ---------------------------------------------------------------------------


def _height_library(
    env: ManagerBasedRlEnv,
    traj_path: str | tuple[str, ...],
    traj_target_height: float,
    traj_target_heights: tuple[float, ...] | None,
) -> tuple[JumpTrajLibrary, tuple[float, ...]]:
    """按配置取轨迹库，并兼容单轨迹旧参数。"""
    traj_paths = _as_tuple(traj_path)
    heights = traj_target_heights if traj_target_heights is not None else (traj_target_height,)
    return JumpTrajLibrary.get(traj_paths, tuple(heights), str(env.device)), tuple(heights)


def _pose_tracking_baseline(
    env: ManagerBasedRlEnv,
    robot: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """读取 reset 时缓存的 base pose 参考零点。

    reset_root_state_full 会随机 xy 和 yaw。6DoF tracking 必须跟踪相对这个零点的
    参考位移，否则奖励会把合法随机初始状态当成误差。
    """
    ref_pos = getattr(env, "_jump_pose_ref_pos_w", None)
    ref_yaw = getattr(env, "_jump_pose_ref_yaw", None)
    if ref_pos is None:
        default_root_state = robot.data.default_root_state
        ref_pos = default_root_state[:, 0:3].clone()
        if hasattr(env.scene, "env_origins"):
            ref_pos[:, 0:3] = ref_pos[:, 0:3] + env.scene.env_origins[:, 0:3]
    if ref_yaw is None:
        ref_yaw = torch.zeros(env.num_envs, device=env.device)
    return ref_pos, ref_yaw


def traj_base_pose_6d_tracking(
    env: ManagerBasedRlEnv,
    command_name: str,
    traj_path: str | tuple[str, ...],
    sigma_xy: float = 0.20,
    sigma_z: float = 0.10,
    sigma_rot: float = 0.45,
    xy_weight: float = 0.20,
    z_weight: float = 0.50,
    rot_weight: float = 0.30,
    traj_target_height: float = 0.6,
    traj_target_heights: tuple[float, ...] | None = None,
    height_match_tol: float = 0.15,
    grounded_weight: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """base_link 轨迹跟踪：只用 xyz 正向 tracking。

    姿态质量由 jump_orientation / jump_tilt_barrier 等惩罚项负责。这里不再
    通过姿态正奖励给分，避免策略把“获得好姿态奖励”理解成可接受坏姿态。
    """
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    h_target = cmd[:, 6]

    term = _get_jump_term(env, command_name)
    library, _ = _height_library(env, traj_path, traj_target_height, traj_target_heights)
    ref_pos, _, _, _, ref_height, _ = library.gather(h_target, term.traj_step)
    ref_start_pos, _, _, _, _, _ = library.gather(h_target, torch.zeros_like(term.traj_step))
    height_match = torch.abs(h_target - ref_height) <= height_match_tol

    robot = env.scene[asset_cfg.name]
    pose_ref_pos, _ = _pose_tracking_baseline(env, robot)

    target_pos = pose_ref_pos.clone()
    target_pos[:, 0:2] = pose_ref_pos[:, 0:2] + (ref_pos[:, 0:2] - ref_start_pos[:, 0:2])
    height_scale = h_target / ref_height
    default_base_z = robot.data.default_root_state[:, 2]
    if hasattr(env.scene, "env_origins"):
        default_base_z = default_base_z + env.scene.env_origins[:, 2]
    target_pos[:, 2] = default_base_z + (ref_pos[:, 2] - default_base_z) * height_scale

    pos = robot.data.root_link_pos_w
    xy_err_sq = torch.sum((pos[:, 0:2] - target_pos[:, 0:2]) ** 2, dim=1)
    z_err_sq = (pos[:, 2] - target_pos[:, 2]) ** 2

    # 早期 PostTrain 可能仍传入第一版严格参数（xy=0.08, rot=0.25）。
    # 这些参数会让 xy/yaw 误差把 base tracking 奖励压到接近 0。这里设置下限，
    # 保证旧配置也能得到分项 reward 的稳定梯度。
    sigma_xy_eff = max(float(sigma_xy), 0.20)
    sigma_z_eff = max(float(sigma_z), 0.10)

    xy_reward = torch.exp(-xy_err_sq / (sigma_xy_eff**2))
    z_reward = torch.exp(-z_err_sq / (sigma_z_eff**2))
    total_weight = max(float(xy_weight) + float(z_weight), 1.0e-6)
    reward = (float(xy_weight) * xy_reward + float(z_weight) * z_reward) / total_weight

    # 接地期降权：准备段只提供弱时序引导，主要约束集中在飞行和落地。
    in_air_or_landing = term.jump_stage >= 1
    weight_mask = torch.where(
        in_air_or_landing, torch.ones_like(reward), torch.full_like(reward, grounded_weight)
    )

    return reward * weight_mask * (jump_flag & height_match).float()


def traj_base_z_tracking(
    env: ManagerBasedRlEnv,
    command_name: str,
    traj_path: str | tuple[str, ...],
    sigma: float = 0.10,
    traj_target_height: float = 0.6,
    traj_target_heights: tuple[float, ...] | None = None,
    height_match_tol: float = 0.15,
    grounded_weight: float = 0.05,
) -> torch.Tensor:
    """base_link 高度跟踪：exp(-err²/sigma²)，飞行+着陆期激活，接地期极小权重。

    高度是最直观的跳跃质量度量：蹲下时跟低值、起跳时跟上升值、飞行时跟峰值、着陆时跟缓冲值。
    用指数形状（而非 L1/L2 惩罚）保证近零误差时梯度大、远离时梯度不爆炸。

    grounded_weight：接地期的奖励缩放系数（默认 0.05）。
    接地期 traj_step 会从轨迹第 0 帧推进到站立段末帧。
    通过大幅降低接地期权重，只给完整起跳准备动作提供微弱引导信号，主力奖励集中在飞行+着陆阶段。

    height_match_tol：只对目标高度在 [traj_target_height ± tol] 范围内的 env 激活 tracking。
    其余 env jump_flag 仍为 1 但不激活此奖励，避免低目标高度 env 跟高轨迹发生冲突。
    """
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    h_target = cmd[:, 6]

    term = _get_jump_term(env, command_name)

    traj_paths = _as_tuple(traj_path)
    heights = traj_target_heights if traj_target_heights is not None else (traj_target_height,)
    library = JumpTrajLibrary.get(traj_paths, tuple(heights), str(env.device))
    ref_pos, _, _, _, ref_height, _ = library.gather(h_target, term.traj_step)
    height_match = torch.abs(h_target - ref_height) <= height_match_tol

    robot = env.scene["robot"]
    base_z = robot.data.root_link_pos_w[:, 2]
    default_base_z = robot.data.default_root_state[:, 2]
    height_scale = h_target / ref_height
    ref_z = default_base_z + (ref_pos[:, 2] - default_base_z) * height_scale

    err = base_z - ref_z
    reward = torch.exp(-(err**2) / (sigma**2))

    # 接地期降权：只给准备动作引导，防止原地不动拿满分
    in_air_or_landing = term.jump_stage >= 1
    weight_mask = torch.where(
        in_air_or_landing, torch.ones_like(reward), torch.full_like(reward, grounded_weight)
    )

    return reward * weight_mask * (jump_flag & height_match).float()


def traj_vz_tracking(
    env: ManagerBasedRlEnv,
    command_name: str,
    traj_path: str | tuple[str, ...],
    std_grounded: float = 0.45,
    std_takeoff: float = 0.60,
    std_air: float = 0.55,
    std_landing: float = 0.45,
    traj_target_height: float = 0.6,
    traj_target_heights: tuple[float, ...] | None = None,
    height_match_tol: float = 0.15,
) -> torch.Tensor:
    """垂直速度跟踪：阶段相关 std 的 exp 奖励，全阶段激活。

    vz 跟踪保证动作的加速/减速节奏正确：
    - 蹲下段：vz 应为负（向下），跟踪到位防止抖动
    - 起跳段：vz 应快速增大，跟踪起跳加速度
    - 飞行段：vz 从正到负，跟踪抛物线节奏
    - 着陆段：vz 为负（向下缓冲），跟踪缓冲速度

    早期用 L1 惩罚时，策略只看见“哪里扣分少”，容易学成低速贴轨迹。
    这里改为 Unitree tracking 风格的 exp(-err²/std²) 正奖励，并按阶段设置
    std：起跳和空中略宽，着陆略紧。这样能保留时序引导，同时减少 PPO 更新时
    对蹬地速度的硬压制。
    """
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    h_target = cmd[:, 6]

    term = _get_jump_term(env, command_name)

    traj_paths = _as_tuple(traj_path)
    heights = traj_target_heights if traj_target_heights is not None else (traj_target_height,)
    library = JumpTrajLibrary.get(traj_paths, tuple(heights), str(env.device))
    _, ref_vel, _, _, ref_height, _ = library.gather(h_target, term.traj_step)
    height_match = torch.abs(h_target - ref_height) <= height_match_tol

    robot = env.scene["robot"]
    vz = robot.data.root_link_lin_vel_w[:, 2]
    velocity_scale = torch.sqrt(h_target / ref_height)
    vz_ref = ref_vel[:, 2] * velocity_scale

    err_sq = (vz - vz_ref) ** 2

    std = torch.full_like(vz, float(std_grounded))
    std = torch.where(
        term.reference_takeoff_active(),
        torch.full_like(std, float(std_takeoff)),
        std,
    )
    std = torch.where(term.jump_stage == 1, torch.full_like(std, float(std_air)), std)
    std = torch.where(term.jump_stage == 2, torch.full_like(std, float(std_landing)), std)
    reward = torch.exp(-err_sq / (std.clamp_min(1.0e-3) ** 2))

    return reward * (jump_flag & height_match).float()


def traj_joint_pos_tracking(
    env: ManagerBasedRlEnv,
    command_name: str,
    traj_path: str | tuple[str, ...],
    sigma: float = 0.15,
    sigma_grounded: float | None = None,
    sigma_takeoff: float | None = None,
    sigma_air: float | None = None,
    sigma_landing: float | None = None,
    traj_target_height: float = 0.6,
    traj_target_heights: tuple[float, ...] | None = None,
    height_match_tol: float = 0.15,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    grounded_weight: float = 0.05,
) -> torch.Tensor:
    """关节角跟踪：腿部 4 关节的阶段相关 exp 奖励，全阶段激活。

    关节角 tracking 是动作时序的核心约束：
    - 蹲下段：膝关节应屈曲（跟 q_tuck 方向）
    - 起跳段：膝关节应迅速伸展（跟 q_extend 方向）
    - 飞行段：按收腿→保持→展腿时序变化
    - 着陆段：应屈曲吸收冲击

    起跳期允许腿部大幅蹲展，因此 sigma 放宽；空中和落地逐步收紧，用于减少
    双腿分叉和落地不对称。这个设计借鉴 Unitree variable_posture 的“运动阶段
    决定关节容忍度”思路，但仍保留当前参考轨迹的时序信息。
    只跟踪腿部关节（lf0/lf1/rf0/rf1），不约束轮子（轮子速度由其他奖励管）。
    """
    cmd = env.command_manager.get_command(command_name)
    jump_flag = cmd[:, 5] > 0.5
    h_target = cmd[:, 6]

    term = _get_jump_term(env, command_name)

    traj_paths = _as_tuple(traj_path)
    heights = traj_target_heights if traj_target_heights is not None else (traj_target_height,)
    library = JumpTrajLibrary.get(traj_paths, tuple(heights), str(env.device))
    _, _, ref_q, _, ref_height, _ = library.gather(h_target, term.traj_step)
    height_match = torch.abs(h_target - ref_height) <= height_match_tol

    robot = env.scene[asset_cfg.name]
    # 腿部关节索引（MJLab joint_pos 10 维中的列）
    leg_idx = JointGroup.LEGS  # [lf0, lf1, rf0, rf1]
    q_leg = robot.data.joint_pos[:, leg_idx]

    # ref_q 是 6 维受控关节 [lf0, lf1, lw, rf0, rf1, rw]，腿部取 0,1,3,4
    ref_q_leg = ref_q[:, [0, 1, 3, 4]]

    err = q_leg - ref_q_leg

    sigma_grounded_f = float(sigma if sigma_grounded is None else sigma_grounded)
    sigma_takeoff_f = float(sigma if sigma_takeoff is None else sigma_takeoff)
    sigma_air_f = float(sigma if sigma_air is None else sigma_air)
    sigma_landing_f = float(sigma if sigma_landing is None else sigma_landing)
    sigma_stage = torch.full((env.num_envs,), sigma_grounded_f, device=env.device)
    sigma_stage = torch.where(
        term.reference_takeoff_active(),
        torch.full_like(sigma_stage, sigma_takeoff_f),
        sigma_stage,
    )
    sigma_stage = torch.where(
        term.jump_stage == 1,
        torch.full_like(sigma_stage, sigma_air_f),
        sigma_stage,
    )
    sigma_stage = torch.where(
        term.jump_stage == 2,
        torch.full_like(sigma_stage, sigma_landing_f),
        sigma_stage,
    )

    # 逐关节 exp 形状，再取均值
    reward = torch.exp(-(err**2) / (sigma_stage.unsqueeze(1).clamp_min(1.0e-3) ** 2)).mean(dim=1)

    # 接地期大幅降权：只给微弱引导信号，防止「不动就能拿满分」
    in_air_or_landing = term.jump_stage >= 1
    weight_mask = torch.where(
        in_air_or_landing, torch.ones_like(reward), torch.full_like(reward, grounded_weight)
    )

    return reward * weight_mask * (jump_flag & height_match).float()
