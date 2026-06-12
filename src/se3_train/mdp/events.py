"""SE3 轮腿机器人的域随机化事件。

与原始 Isaac Gym 实现配置保持一致。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    euler_xyz_from_quat,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
)

from se3_shared import (
    RobotConfig as SharedRobotConfig,
)
from se3_shared import (
    output_to_policy_pos_torch,
    policy_to_output_pos_torch,
    policy_to_output_vel_torch,
)
from se3_train.mdp import recovery_state
from se3_train.mdp.height_default_cache import update_policy_default_from_height_cache
from se3_train.mdp.joint_indices import (
    is_closedchain_model,
    is_fourbar_surrogate_model,
    policy_leg_joint_ids,
    tensor_ids,
    wheel_joint_ids,
)
from se3_train.mdp.jump_commands import JumpCommandTerm
from se3_train.mdp.jump_trajectories import (
    DEFAULT_JUMP_TRAJ_HEIGHTS,
    DEFAULT_JUMP_TRAJ_PATHS,
    JumpTrajLibrary,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_SHARED_ROBOT = SharedRobotConfig()
_FULL_ANGLE_RESET_BBOX_MIN = (-0.278, -0.242, -0.323)
_FULL_ANGLE_RESET_BBOX_MAX = (0.278, 0.242, 0.111)
_FOURBAR_WHEEL_RADIUS_M = 0.060
_DEFAULT_RESET_WHEEL_CLEARANCE_M = 0.001
_GEOM_PLANE = int(mujoco.mjtGeom.mjGEOM_PLANE)
_GEOM_HFIELD = int(mujoco.mjtGeom.mjGEOM_HFIELD)
_GEOM_SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
_GEOM_CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
_GEOM_ELLIPSOID = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
_GEOM_CYLINDER = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
_GEOM_BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
_GEOM_MESH = int(mujoco.mjtGeom.mjGEOM_MESH)


def _stage_value(stage: dict, key: str, default):
    """读取 recovery stage 字段，缺省时使用默认值。"""
    return stage.get(key, default)


def _active_recovery_stage(
    env: ManagerBasedRlEnv,
    recovery_stages: list[dict] | None,
) -> dict:
    """按 common_step_counter 选择当前 recovery reset 课程阶段。"""
    if not recovery_stages:
        return {}
    step = getattr(env, "common_step_counter", 0)
    active = recovery_stages[0]
    for stage in recovery_stages:
        if step >= int(stage.get("step", 0)):
            active = stage
    return active


def _curriculum_progress(
    env: ManagerBasedRlEnv,
    *,
    use_iterations: bool,
    steps_per_policy_iter: int,
    offset_iter: int = 0,
) -> int:
    """返回 reset 课程进度；自起任务用 PPO iter，避免课程在几十 iter 内全展开。"""
    step = int(getattr(env, "common_step_counter", 0))
    if not use_iterations:
        return step
    steps_per_iter = max(1, int(steps_per_policy_iter))
    return max(0, step // steps_per_iter - int(offset_iter))


def _active_curriculum_stage(
    env: ManagerBasedRlEnv,
    stages: list[dict] | None,
    *,
    use_iterations: bool,
    steps_per_policy_iter: int,
    offset_iter: int = 0,
) -> tuple[dict, int]:
    """按课程进度选择当前 stage。"""
    if not stages:
        return {}, _curriculum_progress(
            env,
            use_iterations=use_iterations,
            steps_per_policy_iter=steps_per_policy_iter,
            offset_iter=offset_iter,
        )

    progress = _curriculum_progress(
        env,
        use_iterations=use_iterations,
        steps_per_policy_iter=steps_per_policy_iter,
        offset_iter=offset_iter,
    )
    key = "iteration" if use_iterations else "step"
    active = stages[0]
    for stage in stages:
        if progress >= int(stage.get(key, stage.get("step", 0))):
            active = stage
    return active, progress


def _ensure_recovery_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
    """创建并返回每个 env 的 recovery reset 标记。"""
    return recovery_state.ensure_bool_buffer(env, "_recovery_reset_mask")


def _ensure_recovery_float_buffer(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
    """创建 recovery reset 诊断浮点缓存。"""
    values = getattr(env, name, None)
    if not isinstance(values, torch.Tensor) or values.shape[0] != env.num_envs:
        values = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        setattr(env, name, values)
    return values


def _load_recovery_state_cache(
    env: ManagerBasedRlEnv, path: str | None
) -> dict[str, torch.Tensor] | None:
    """懒加载离线生成的倒地稳定状态缓存。"""
    if path is None or path == "":
        return None

    cache_store = getattr(env, "_recovery_state_cache_store", None)
    if not isinstance(cache_store, dict):
        cache_store = {}
        env._recovery_state_cache_store = cache_store
    cached = cache_store.get(path)
    if cached is not None:
        return cached

    cache_path = Path(path)
    if not cache_path.is_absolute():
        cache_path = Path.cwd() / cache_path
    if not cache_path.exists():
        if hasattr(env, "extras"):
            env.extras.setdefault("log", {})["Recovery/cache_missing"] = 1.0
        return None

    import numpy as np

    data = np.load(cache_path)
    required = ("root_pos", "root_quat", "root_lin_vel", "root_ang_vel", "joint_pos", "joint_vel")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"recovery state cache 缺少字段: {missing}")

    cache = {
        name: torch.as_tensor(data[name], device=env.device, dtype=torch.float32)
        for name in required
    }
    if "pose_type" in data:
        cache["pose_type"] = torch.as_tensor(data["pose_type"], device=env.device, dtype=torch.long)
    else:
        cache["pose_type"] = torch.zeros(
            cache["root_pos"].shape[0], device=env.device, dtype=torch.long
        )

    cache_store[path] = cache
    if hasattr(env, "extras"):
        env.extras.setdefault("log", {}).update(
            {
                "Recovery/cache_missing": 0.0,
                "Recovery/cache_size": float(cache["root_pos"].shape[0]),
            }
        )
    return cache


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


def _quat_z_row(quat: torch.Tensor) -> torch.Tensor:
    """返回四元数对应旋转矩阵的世界 z 行。"""
    w, x, y, z = quat.unbind(dim=-1)
    return torch.stack(
        (
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    )


def _full_angle_safe_base_height(
    z_row: torch.Tensor,
    clearance: torch.Tensor,
) -> torch.Tensor:
    """按完整机器人包络估算任意姿态 reset 时不穿地的 base 高度。"""
    bbox_min = torch.tensor(_FULL_ANGLE_RESET_BBOX_MIN, device=z_row.device, dtype=z_row.dtype)
    bbox_max = torch.tensor(_FULL_ANGLE_RESET_BBOX_MAX, device=z_row.device, dtype=z_row.dtype)

    min_z = torch.minimum(z_row * bbox_min, z_row * bbox_max).sum(dim=-1)
    return -min_z + clearance


def reset_root_state_robotlab_full_random(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    pos_xy_range: tuple[float, float] = (-0.5, 0.5),
    pos_xy_offset: tuple[float, float] = (0.0, 0.0),
    height_offset_range: tuple[float, float] = (0.0, 0.2),
    roll_range: tuple[float, float] = (-3.141592653589793, 3.141592653589793),
    pitch_range: tuple[float, float] = (-3.141592653589793, 3.141592653589793),
    yaw_range: tuple[float, float] = (-3.141592653589793, 3.141592653589793),
    lin_vel_range: tuple[float, float] = (-0.5, 0.5),
    ang_vel_range: tuple[float, float] = (-0.5, 0.5),
    clearance_range: tuple[float, float] = (0.0, 0.05),
    pitch_inverted_prob: float = 0.0,
    roll_side_prob: float = 0.0,
    pitch_inverted_jitter_range: tuple[float, float] = (-0.2617993877991494, 0.2617993877991494),
    pitch_inverted_roll_jitter_range: tuple[float, float] = (
        -0.17453292519943295,
        0.17453292519943295,
    ),
    roll_side_jitter_range: tuple[float, float] = (-0.2617993877991494, 0.2617993877991494),
    roll_side_pitch_jitter_range: tuple[float, float] = (-0.17453292519943295, 0.17453292519943295),
    curriculum_stages: list[dict] | None = None,
    use_iterations: bool = False,
    steps_per_policy_iter: int = 64,
    offset_iter: int = 0,
    mark_recovery_episode: bool = False,
    recovery_command_height: float = _SHARED_ROBOT.default_base_height,
) -> None:
    """按 RobotLab Go2W 语义随机化 root：默认状态叠加 xyz/rpy 和速度扰动。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    _pre_resample_jump_command_for_reset(env, env_ids)
    stage, curriculum_progress = _active_curriculum_stage(
        env,
        curriculum_stages,
        use_iterations=use_iterations,
        steps_per_policy_iter=steps_per_policy_iter,
        offset_iter=offset_iter,
    )
    roll_range = _stage_value(stage, "roll_range", roll_range)
    pitch_range = _stage_value(stage, "pitch_range", pitch_range)
    yaw_range = _stage_value(stage, "yaw_range", yaw_range)
    pos_xy_range = _stage_value(stage, "pos_xy_range", pos_xy_range)
    pos_xy_offset = _stage_value(stage, "pos_xy_offset", pos_xy_offset)
    height_offset_range = _stage_value(stage, "height_offset_range", height_offset_range)
    lin_vel_range = _stage_value(stage, "lin_vel_range", lin_vel_range)
    ang_vel_range = _stage_value(stage, "ang_vel_range", ang_vel_range)
    clearance_range = _stage_value(stage, "clearance_range", clearance_range)
    pitch_inverted_prob = float(_stage_value(stage, "pitch_inverted_prob", pitch_inverted_prob))
    roll_side_prob = float(_stage_value(stage, "roll_side_prob", roll_side_prob))
    pitch_inverted_jitter_range = _stage_value(
        stage, "pitch_inverted_jitter_range", pitch_inverted_jitter_range
    )
    pitch_inverted_roll_jitter_range = _stage_value(
        stage,
        "pitch_inverted_roll_jitter_range",
        pitch_inverted_roll_jitter_range,
    )
    roll_side_jitter_range = _stage_value(stage, "roll_side_jitter_range", roll_side_jitter_range)
    roll_side_pitch_jitter_range = _stage_value(
        stage, "roll_side_pitch_jitter_range", roll_side_pitch_jitter_range
    )

    asset: Entity = env.scene[asset_cfg.name]
    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    root_states = default_root_state[env_ids].clone()

    def _sample_range(value_range: tuple[float, float], shape: tuple[int, ...]) -> torch.Tensor:
        return sample_uniform(
            torch.tensor(float(value_range[0]), device=env.device),
            torch.tensor(float(value_range[1]), device=env.device),
            shape,
            env.device,
        )

    n = len(env_ids)
    pos = root_states[:, 0:3].clone()
    pos[:, 0] += _sample_range(pos_xy_range, (n,))
    pos[:, 1] += _sample_range(pos_xy_range, (n,))
    pos[:, 0] += float(pos_xy_offset[0])
    pos[:, 1] += float(pos_xy_offset[1])

    roll = _sample_range(roll_range, (n,))
    pitch = _sample_range(pitch_range, (n,))
    yaw = _sample_range(yaw_range, (n,))
    pitch_inverted_prob = max(0.0, pitch_inverted_prob)
    roll_side_prob = max(0.0, roll_side_prob)
    pose_prob_sum = pitch_inverted_prob + roll_side_prob
    if pose_prob_sum > 1.0:
        pitch_inverted_prob /= pose_prob_sum
        roll_side_prob /= pose_prob_sum
    pose_rand = torch.rand(n, device=env.device)
    pitch_inverted_mask = pose_rand < pitch_inverted_prob
    roll_side_mask = (pose_rand >= pitch_inverted_prob) & (
        pose_rand < pitch_inverted_prob + roll_side_prob
    )
    reset_pose_bins = torch.zeros(n, device=env.device, dtype=torch.long)
    reset_pose_bins[pitch_inverted_mask] = 1
    reset_pose_bins[roll_side_mask] = 2
    if torch.any(pitch_inverted_mask):
        count = int(pitch_inverted_mask.sum().item())
        pitch_sign = torch.where(
            torch.rand(count, device=env.device) < 0.5,
            torch.full((count,), -1.0, device=env.device),
            torch.ones(count, device=env.device),
        )
        roll[pitch_inverted_mask] = _sample_range(pitch_inverted_roll_jitter_range, (count,))
        pitch[pitch_inverted_mask] = pitch_sign * torch.pi + _sample_range(
            pitch_inverted_jitter_range, (count,)
        )
    if torch.any(roll_side_mask):
        count = int(roll_side_mask.sum().item())
        roll_sign = torch.where(
            torch.rand(count, device=env.device) < 0.5,
            torch.full((count,), -1.0, device=env.device),
            torch.ones(count, device=env.device),
        )
        roll[roll_side_mask] = roll_sign * (0.5 * torch.pi) + _sample_range(
            roll_side_jitter_range, (count,)
        )
        pitch[roll_side_mask] = _sample_range(roll_side_pitch_jitter_range, (count,))
    quat_delta = quat_from_euler_xyz(roll, pitch, yaw)
    new_quat = quat_mul(root_states[:, 3:7], quat_delta)
    z_row = _quat_z_row(new_quat)

    sampled_height = root_states[:, 2] + _sample_range(height_offset_range, (n,))
    safe_height = _full_angle_safe_base_height(
        z_row,
        _sample_range(clearance_range, (n,)),
    )
    base_height = torch.maximum(sampled_height, safe_height)

    pos[:, 0:3] += env.scene.env_origins[env_ids]
    pos[:, 2] = base_height + env.scene.env_origins[env_ids, 2]

    vel = root_states[:, 7:13].clone()
    vel[:, 0:3] += _sample_range(lin_vel_range, (n, 3))
    vel[:, 3:6] += _sample_range(ang_vel_range, (n, 3))

    pitch_flip_reset = recovery_state.ensure_bool_buffer(env, "_recovery_pitch_flip_reset_mask")
    pitch_flip_reset[env_ids] = pitch_inverted_mask

    init_tilt = torch.acos(torch.clamp(z_row[:, 2], -1.0, 1.0))
    init_roll, init_pitch, init_yaw = euler_xyz_from_quat(new_quat)
    if mark_recovery_episode:
        recovery_mask = torch.ones(n, device=env.device, dtype=torch.bool)
        recovery_state.set_recovery_episode(env, env_ids, recovery_mask)
        env._recovery_command_height = float(recovery_command_height)
        init_tilt_buf = _ensure_recovery_float_buffer(env, "_recovery_init_tilt")
        init_yaw_buf = _ensure_recovery_float_buffer(env, "_recovery_init_yaw")
        init_roll_buf = _ensure_recovery_float_buffer(env, "_recovery_init_roll")
        init_pitch_buf = _ensure_recovery_float_buffer(env, "_recovery_init_pitch")
        init_tilt_buf[env_ids] = init_tilt
        init_yaw_buf[env_ids] = init_yaw
        init_roll_buf[env_ids] = init_roll
        init_pitch_buf[env_ids] = init_pitch
    else:
        for name in (
            "_recovery_reset_mask",
            "_recovery_episode_mask",
            "_recovery_cache_reset_mask",
        ):
            values = getattr(env, name, None)
            if isinstance(values, torch.Tensor) and values.shape[0] == env.num_envs:
                values[env_ids] = False

    asset.write_root_link_pose_to_sim(torch.cat([pos, new_quat], dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(vel, env_ids=env_ids)

    if not hasattr(env, "_jump_pose_ref_pos_w"):
        env._jump_pose_ref_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
    if not hasattr(env, "_jump_pose_ref_yaw"):
        env._jump_pose_ref_yaw = torch.zeros(env.num_envs, device=env.device)
    env._jump_pose_ref_pos_w[env_ids] = pos
    env._jump_pose_ref_yaw[env_ids] = init_yaw

    init_tilt_deg = torch.rad2deg(init_tilt)
    sampled_roll_deg = torch.rad2deg(roll)
    sampled_pitch_deg = torch.rad2deg(pitch)
    sampled_yaw_deg = torch.rad2deg(yaw)
    init_roll_deg = torch.rad2deg(init_roll)
    init_pitch_deg = torch.rad2deg(init_pitch)
    init_yaw_deg = torch.rad2deg(init_yaw)
    root_lin_vel_norm = torch.linalg.norm(vel[:, 0:3], dim=1)
    root_ang_vel_norm = torch.linalg.norm(vel[:, 3:6], dim=1)
    bins = torch.bucketize(
        init_tilt_deg,
        torch.tensor((30.0, 75.0, 130.0), device=env.device),
    )
    tilt_bins = getattr(env, "_reset_init_tilt_bin", None)
    if not isinstance(tilt_bins, torch.Tensor) or tilt_bins.shape[0] != env.num_envs:
        tilt_bins = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env._reset_init_tilt_bin = tilt_bins
    tilt_bins[env_ids] = bins

    pose_bins = getattr(env, "_reset_pose_bin", None)
    if not isinstance(pose_bins, torch.Tensor) or pose_bins.shape[0] != env.num_envs:
        pose_bins = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env._reset_pose_bin = pose_bins
    pose_bins[env_ids] = reset_pose_bins

    log_values = {
        "Reset/robotlab_full_random_ratio": 1.0,
        "Reset/full_angle_random_ratio": 1.0,
        "Reset/pitch_flip_ratio": pitch_inverted_mask.float().mean().item(),
        "Reset/pose_bin_mixed_full_ratio": (reset_pose_bins == 0).float().mean().item(),
        "Reset/pose_bin_pitch_inverted_ratio": (reset_pose_bins == 1).float().mean().item(),
        "Reset/pose_bin_roll_side_ratio": (reset_pose_bins == 2).float().mean().item(),
        "Reset/pitch_inverted_prob": float(pitch_inverted_prob),
        "Reset/roll_side_prob": float(roll_side_prob),
        "Reset/curriculum_progress": float(curriculum_progress),
        "Reset/curriculum_tilt_max_deg": max(
            abs(float(roll_range[0])),
            abs(float(roll_range[1])),
            abs(float(pitch_range[0])),
            abs(float(pitch_range[1])),
        )
        * 180.0
        / 3.141592653589793,
        "Reset/curriculum_roll_max_deg": max(abs(float(roll_range[0])), abs(float(roll_range[1])))
        * 180.0
        / 3.141592653589793,
        "Reset/curriculum_pitch_max_deg": max(
            abs(float(pitch_range[0])),
            abs(float(pitch_range[1])),
        )
        * 180.0
        / 3.141592653589793,
        "Reset/mean_init_tilt_deg": init_tilt_deg.mean().item(),
        "Reset/max_init_tilt_deg": init_tilt_deg.max().item(),
        "Reset/mean_abs_sampled_roll_deg": sampled_roll_deg.abs().mean().item(),
        "Reset/mean_abs_sampled_pitch_deg": sampled_pitch_deg.abs().mean().item(),
        "Reset/mean_abs_sampled_yaw_deg": sampled_yaw_deg.abs().mean().item(),
        "Reset/mean_abs_roll_deg": init_roll_deg.abs().mean().item(),
        "Reset/mean_abs_pitch_deg": init_pitch_deg.abs().mean().item(),
        "Reset/mean_abs_yaw_deg": init_yaw_deg.abs().mean().item(),
        "Reset/mean_base_height_m": base_height.mean().item(),
        "Reset/min_base_height_m": base_height.min().item(),
        "Reset/max_base_height_m": base_height.max().item(),
        "Reset/root_lin_vel_norm": root_lin_vel_norm.mean().item(),
        "Reset/root_ang_vel_norm": root_ang_vel_norm.mean().item(),
        "Reset/safe_height_clamp_ratio": (base_height > sampled_height).float().mean().item(),
        "Reset/init_tilt_bin_upright_noise_ratio": (bins == 0).float().mean().item(),
        "Reset/init_tilt_bin_near_fall_ratio": (bins == 1).float().mean().item(),
        "Reset/init_tilt_bin_hard_tilt_ratio": (bins == 2).float().mean().item(),
        "Reset/init_tilt_bin_inverted_ratio": (bins == 3).float().mean().item(),
    }
    env._reset_robotlab_full_random_log_values = log_values

    if hasattr(env, "extras"):
        log = env.extras.setdefault("log", {})
        log.update(log_values)


def reset_root_state_full(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    recovery_prob: float = 0.0,
    recovery_stages: list[dict] | None = None,
    recovery_roll_range: tuple[float, float] = (-0.35, 0.35),
    recovery_pitch_range: tuple[float, float] = (-0.35, 0.35),
    recovery_height_range: tuple[float, float] = (0.24, 0.34),
    recovery_lin_vel_range: tuple[float, float] = (-0.15, 0.15),
    recovery_ang_vel_range: tuple[float, float] = (-0.8, 0.8),
    recovery_side_roll_prob: float = 0.0,
    recovery_side_roll_min_abs: float = 0.75,
    recovery_side_pitch_range: tuple[float, float] = (-0.35, 0.35),
    recovery_fallen_pose_prob: float = 0.0,
    recovery_fallen_roll_pose_prob: float = 0.5,
    recovery_fallen_roll_abs_range: tuple[float, float] = (1.35, 1.75),
    recovery_fallen_pitch_abs_range: tuple[float, float] = (1.35, 1.75),
    recovery_fallen_coupled_range: tuple[float, float] = (-0.35, 0.35),
    recovery_fallen_height_range: tuple[float, float] = (0.12, 0.24),
    recovery_state_cache_path: str | None = None,
    recovery_state_cache_prob: float = 0.0,
    recovery_grace_steps: int = 400,
    recovery_command_height: float = _SHARED_ROBOT.default_base_height,
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

    stage = _active_recovery_stage(env, recovery_stages)
    recovery_prob = float(_stage_value(stage, "prob", recovery_prob))
    recovery_roll_range = _stage_value(stage, "roll_range", recovery_roll_range)
    recovery_pitch_range = _stage_value(stage, "pitch_range", recovery_pitch_range)
    recovery_height_range = _stage_value(stage, "height_range", recovery_height_range)
    recovery_side_roll_prob = float(_stage_value(stage, "side_roll_prob", recovery_side_roll_prob))
    recovery_side_roll_min_abs = float(
        _stage_value(stage, "side_roll_min_abs", recovery_side_roll_min_abs)
    )
    recovery_side_pitch_range = _stage_value(stage, "side_pitch_range", recovery_side_pitch_range)
    recovery_fallen_pose_prob = float(
        _stage_value(stage, "fallen_pose_prob", recovery_fallen_pose_prob)
    )
    recovery_fallen_roll_pose_prob = float(
        _stage_value(stage, "fallen_roll_pose_prob", recovery_fallen_roll_pose_prob)
    )
    recovery_fallen_roll_abs_range = _stage_value(
        stage, "fallen_roll_abs_range", recovery_fallen_roll_abs_range
    )
    recovery_fallen_pitch_abs_range = _stage_value(
        stage, "fallen_pitch_abs_range", recovery_fallen_pitch_abs_range
    )
    recovery_fallen_coupled_range = _stage_value(
        stage, "fallen_coupled_range", recovery_fallen_coupled_range
    )
    recovery_fallen_height_range = _stage_value(
        stage, "fallen_height_range", recovery_fallen_height_range
    )
    recovery_state_cache_path = _stage_value(stage, "state_cache_path", recovery_state_cache_path)
    recovery_state_cache_prob = float(
        _stage_value(stage, "state_cache_prob", recovery_state_cache_prob)
    )
    env._recovery_stage_step = int(stage.get("step", 0))
    env._recovery_stage_prob = float(recovery_prob)
    env._recovery_stage_fallen_pose_prob = float(recovery_fallen_pose_prob)
    env._recovery_stage_cache_prob = float(recovery_state_cache_prob)

    recovery_mask = torch.rand(n, device=env.device) < recovery_prob
    recovery_state.set_recovery_episode(env, env_ids, recovery_mask)
    init_roll = _ensure_recovery_float_buffer(env, "_recovery_init_roll")
    init_pitch = _ensure_recovery_float_buffer(env, "_recovery_init_pitch")
    init_yaw = _ensure_recovery_float_buffer(env, "_recovery_init_yaw")
    init_tilt = _ensure_recovery_float_buffer(env, "_recovery_init_tilt")
    init_roll[env_ids] = 0.0
    init_pitch[env_ids] = 0.0
    init_yaw[env_ids] = 0.0
    init_tilt[env_ids] = 0.0
    env._recovery_grace_steps = int(recovery_grace_steps)
    env._recovery_command_height = float(recovery_command_height)

    # 默认仅随机化 yaw,保持直立；recovery env 额外随机 roll/pitch。
    yaw = sample_uniform(
        torch.tensor(-torch.pi, device=env.device),
        torch.tensor(torch.pi, device=env.device),
        (n,),
        env.device,
    )
    roll = torch.zeros(n, device=env.device)
    pitch = torch.zeros(n, device=env.device)
    cache_mask = torch.zeros(n, device=env.device, dtype=torch.bool)
    cached_root_quat = torch.zeros(n, 4, device=env.device)
    cached_root_vel = torch.zeros(n, 6, device=env.device)
    if recovery_mask.any():
        n_recovery = int(recovery_mask.sum().item())
        recovery_indices = recovery_mask.nonzero().flatten()
        cache = _load_recovery_state_cache(env, recovery_state_cache_path)
        if cache is not None and recovery_state_cache_prob > 0.0:
            cache_mask[recovery_indices] = (
                torch.rand(n_recovery, device=env.device) < recovery_state_cache_prob
            )
            if cache_mask.any():
                n_cache = int(cache_mask.sum().item())
                cache_size = int(cache["root_pos"].shape[0])
                cache_ids = torch.randint(cache_size, (n_cache,), device=env.device)
                cache_indices = cache_mask.nonzero().flatten()
                pos[cache_indices, 0:3] = (
                    cache["root_pos"][cache_ids] + env.scene.env_origins[env_ids][cache_indices]
                )
                cached_root_quat[cache_indices] = cache["root_quat"][cache_ids]
                cached_root_vel[cache_indices] = torch.cat(
                    [cache["root_lin_vel"][cache_ids], cache["root_ang_vel"][cache_ids]], dim=1
                )

                if not hasattr(env, "_recovery_cached_joint_pos"):
                    env._recovery_cached_joint_pos = torch.zeros_like(asset.data.default_joint_pos)
                    env._recovery_cached_joint_vel = torch.zeros_like(asset.data.default_joint_pos)
                env._recovery_cached_joint_pos[env_ids[cache_indices]] = cache["joint_pos"][
                    cache_ids
                ]
                env._recovery_cached_joint_vel[env_ids[cache_indices]] = cache["joint_vel"][
                    cache_ids
                ]
                recovery_state.mark_cache_reset(
                    env,
                    env_ids,
                    cache_mask,
                    cache_type_values=cache["pose_type"][cache_ids],
                )

                cache_roll, cache_pitch, cache_yaw = euler_xyz_from_quat(
                    cache["root_quat"][cache_ids]
                )
                roll[cache_indices] = cache_roll
                pitch[cache_indices] = cache_pitch
                yaw[cache_indices] = cache_yaw
                init_tilt[env_ids[cache_indices]] = torch.acos(
                    torch.clamp(torch.cos(cache_roll) * torch.cos(cache_pitch), -1.0, 1.0)
                )
        procedural_recovery_mask = recovery_mask & ~cache_mask
        procedural_recovery_indices = procedural_recovery_mask.nonzero().flatten()
        n_procedural = int(procedural_recovery_mask.sum().item())
        if n_procedural == 0:
            fallen_pose_mask = torch.zeros(0, device=env.device, dtype=torch.bool)
        else:
            fallen_pose_mask = (
                torch.rand(n_procedural, device=env.device) < recovery_fallen_pose_prob
            )
        if n_procedural > 0:
            roll[procedural_recovery_mask] = sample_uniform(
                torch.tensor(float(recovery_roll_range[0]), device=env.device),
                torch.tensor(float(recovery_roll_range[1]), device=env.device),
                (n_procedural,),
                env.device,
            )
            pitch[procedural_recovery_mask] = sample_uniform(
                torch.tensor(float(recovery_pitch_range[0]), device=env.device),
                torch.tensor(float(recovery_pitch_range[1]), device=env.device),
                (n_procedural,),
                env.device,
            )
        old_recovery_indices = recovery_indices
        recovery_indices = procedural_recovery_indices
        n_recovery = n_procedural
        if fallen_pose_mask.any():
            n_fallen = int(fallen_pose_mask.sum().item())
            fallen_indices = recovery_indices[fallen_pose_mask]
            fallen_roll_pose = (
                torch.rand(n_fallen, device=env.device) < recovery_fallen_roll_pose_prob
            )
            fallen_sign = torch.where(
                torch.rand(n_fallen, device=env.device) < 0.5,
                torch.tensor(-1.0, device=env.device),
                torch.tensor(1.0, device=env.device),
            )
            if fallen_roll_pose.any():
                n_roll = int(fallen_roll_pose.sum().item())
                roll_indices = fallen_indices[fallen_roll_pose]
                roll_abs = sample_uniform(
                    torch.tensor(float(recovery_fallen_roll_abs_range[0]), device=env.device),
                    torch.tensor(float(recovery_fallen_roll_abs_range[1]), device=env.device),
                    (n_roll,),
                    env.device,
                )
                roll[roll_indices] = roll_abs * fallen_sign[fallen_roll_pose]
                pitch[roll_indices] = sample_uniform(
                    torch.tensor(float(recovery_fallen_coupled_range[0]), device=env.device),
                    torch.tensor(float(recovery_fallen_coupled_range[1]), device=env.device),
                    (n_roll,),
                    env.device,
                )
            fallen_pitch_pose = ~fallen_roll_pose
            if fallen_pitch_pose.any():
                n_pitch = int(fallen_pitch_pose.sum().item())
                pitch_indices = fallen_indices[fallen_pitch_pose]
                pitch_abs = sample_uniform(
                    torch.tensor(float(recovery_fallen_pitch_abs_range[0]), device=env.device),
                    torch.tensor(float(recovery_fallen_pitch_abs_range[1]), device=env.device),
                    (n_pitch,),
                    env.device,
                )
                pitch[pitch_indices] = pitch_abs * fallen_sign[fallen_pitch_pose]
                roll[pitch_indices] = sample_uniform(
                    torch.tensor(float(recovery_fallen_coupled_range[0]), device=env.device),
                    torch.tensor(float(recovery_fallen_coupled_range[1]), device=env.device),
                    (n_pitch,),
                    env.device,
                )
            pos[fallen_indices, 2] = (
                sample_uniform(
                    torch.tensor(float(recovery_fallen_height_range[0]), device=env.device),
                    torch.tensor(float(recovery_fallen_height_range[1]), device=env.device),
                    (n_fallen,),
                    env.device,
                )
                + env.scene.env_origins[env_ids][fallen_indices, 2]
            )
        side_roll_mask = (
            torch.rand(n_recovery, device=env.device) < recovery_side_roll_prob
        ) & ~fallen_pose_mask
        if side_roll_mask.any():
            n_side = int(side_roll_mask.sum().item())
            max_abs_roll = max(
                abs(float(recovery_roll_range[0])),
                abs(float(recovery_roll_range[1])),
                float(recovery_side_roll_min_abs),
            )
            side_roll_abs = sample_uniform(
                torch.tensor(float(recovery_side_roll_min_abs), device=env.device),
                torch.tensor(float(max_abs_roll), device=env.device),
                (n_side,),
                env.device,
            )
            side_sign = torch.where(
                torch.rand(n_side, device=env.device) < 0.5,
                torch.tensor(-1.0, device=env.device),
                torch.tensor(1.0, device=env.device),
            )
            side_indices = recovery_indices[side_roll_mask]
            roll[side_indices] = side_roll_abs * side_sign
            pitch[side_indices] = sample_uniform(
                torch.tensor(float(recovery_side_pitch_range[0]), device=env.device),
                torch.tensor(float(recovery_side_pitch_range[1]), device=env.device),
                (n_side,),
                env.device,
            )
        standard_recovery_mask = procedural_recovery_mask.clone()
        if fallen_pose_mask.any():
            standard_recovery_mask[recovery_indices[fallen_pose_mask]] = False
        if standard_recovery_mask.any():
            n_standard = int(standard_recovery_mask.sum().item())
            pos[standard_recovery_mask, 2] = (
                sample_uniform(
                    torch.tensor(float(recovery_height_range[0]), device=env.device),
                    torch.tensor(float(recovery_height_range[1]), device=env.device),
                    (n_standard,),
                    env.device,
                )
                + env.scene.env_origins[env_ids][standard_recovery_mask, 2]
            )
        init_roll[env_ids] = roll
        init_pitch[env_ids] = pitch
        init_yaw[env_ids] = yaw
        init_tilt[env_ids] = torch.acos(torch.clamp(torch.cos(roll) * torch.cos(pitch), -1.0, 1.0))
        recovery_indices = old_recovery_indices
    quat_delta = quat_from_euler_xyz(roll, pitch, yaw)
    default_quat = root_states[:, 3:7]
    new_quat = quat_mul(default_quat, quat_delta)
    if cache_mask.any():
        new_quat[cache_mask] = cached_root_quat[cache_mask]

    vel = torch.zeros(n, 6, device=env.device)
    if recovery_mask.any():
        n_recovery = int(recovery_mask.sum().item())
        vel[recovery_mask, 0:3] = sample_uniform(
            torch.tensor(float(recovery_lin_vel_range[0]), device=env.device),
            torch.tensor(float(recovery_lin_vel_range[1]), device=env.device),
            (n_recovery, 3),
            env.device,
        )
        vel[recovery_mask, 3:6] = sample_uniform(
            torch.tensor(float(recovery_ang_vel_range[0]), device=env.device),
            torch.tensor(float(recovery_ang_vel_range[1]), device=env.device),
            (n_recovery, 3),
            env.device,
        )
        if hasattr(env, "command_manager"):
            try:
                cmd = env.command_manager.get_command("velocity_height")
                recovery_env_ids = env_ids[recovery_mask]
                cmd[recovery_env_ids, 0:4] = 0.0
                cmd[recovery_env_ids, 4] = float(recovery_command_height)
                if cmd.shape[1] >= 8:
                    cmd[recovery_env_ids, 5] = 0.0
                    cmd[recovery_env_ids, 7] = 0.0
                update_policy_default_from_height_cache(
                    env,
                    "velocity_height",
                    env_ids=recovery_env_ids,
                    command=cmd,
                )
            except Exception:
                pass
        cache_reset_mask = recovery_state.ensure_bool_buffer(env, "_recovery_cache_reset_mask")
        local_cache_mask = cache_reset_mask[env_ids]
        if local_cache_mask.any():
            vel[local_cache_mask] = cached_root_vel[local_cache_mask]

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


def _symmetric_or_explicit_range(value: float | tuple[float, float]) -> tuple[float, float]:
    """把标量扰动转成对称区间，显式二元组保持原样。"""
    if isinstance(value, tuple):
        return float(value[0]), float(value[1])
    magnitude = float(value)
    return -magnitude, magnitude


def _sample_joint_offset(
    env: ManagerBasedRlEnv,
    value_range: float | tuple[float, float],
    shape: tuple[int, int],
) -> torch.Tensor:
    """采样关节角 offset。"""
    low, high = _symmetric_or_explicit_range(value_range)
    return sample_uniform(
        torch.tensor(low, device=env.device),
        torch.tensor(high, device=env.device),
        shape,
        env.device,
    )


def _leg_values(
    values: torch.Tensor,
    leg_ids: torch.Tensor,
    rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """读取局部 reset 张量中的腿部 4 维值。"""
    if rows is None:
        return values[:, leg_ids]
    return values[rows[:, None], leg_ids]


def _write_leg_values(
    values: torch.Tensor,
    leg_ids: torch.Tensor,
    new_values: torch.Tensor,
    rows: torch.Tensor | None = None,
) -> None:
    """写回局部 reset 张量中的腿部 4 维值。"""
    if rows is None:
        values[:, leg_ids] = new_values
    else:
        values[rows[:, None], leg_ids] = new_values


def _model_leg_pos_to_policy(asset: Entity, leg_pos: torch.Tensor) -> torch.Tensor:
    """把模型可写的腿部位置转成 policy 主动杆语义。"""
    if is_fourbar_surrogate_model(asset):
        return output_to_policy_pos_torch(leg_pos)
    return leg_pos.clone()


def _policy_leg_pos_to_model(asset: Entity, policy_pos: torch.Tensor) -> torch.Tensor:
    """把 policy 主动杆语义腿部位置转成模型可写关节。"""
    if is_fourbar_surrogate_model(asset):
        return policy_to_output_pos_torch(policy_pos)
    return policy_pos


def _policy_leg_vel_to_model(
    asset: Entity,
    policy_pos: torch.Tensor,
    policy_vel: torch.Tensor,
) -> torch.Tensor:
    """把 policy 主动杆语义腿部速度转成模型可写关节速度。"""
    if is_fourbar_surrogate_model(asset):
        return policy_to_output_vel_torch(policy_pos, policy_vel)
    return policy_vel


def _clamp_active_rod_policy_pose(asset: Entity, policy_pos: torch.Tensor) -> None:
    """在 policy 主动杆语义下裁剪同侧两杆夹角。"""
    if not (is_closedchain_model(asset) or is_fourbar_surrogate_model(asset)):
        return
    lower, upper = _SHARED_ROBOT.active_rod_angle_limits
    for side_idx, (front_idx, back_idx) in enumerate(((0, 1), (2, 3))):
        front_coef, back_coef = _SHARED_ROBOT.active_rod_angle_coeffs[side_idx]
        angle = torch.clamp(
            front_coef * policy_pos[:, front_idx] + back_coef * policy_pos[:, back_idx],
            min=float(lower),
            max=float(upper),
        )
        policy_pos[:, back_idx] = (angle - front_coef * policy_pos[:, front_idx]) / back_coef


def _sample_full_random_policy_leg_pose(
    env: ManagerBasedRlEnv,
    n: int,
    front_joint_offset_range: float | tuple[float, float],
    active_rod_angle_range: tuple[float, float] | None,
) -> torch.Tensor:
    """直接采样 lf0/rf0 与主动杆夹角，并构造 policy 语义下的四个腿部关节。"""
    default = torch.tensor(
        _SHARED_ROBOT.default_dof_pos[:4],
        device=env.device,
        dtype=torch.float32,
    )
    if isinstance(front_joint_offset_range, tuple):
        front_lo, front_hi = front_joint_offset_range
    else:
        front_lo, front_hi = -float(front_joint_offset_range), float(front_joint_offset_range)
    lower, upper = (
        _SHARED_ROBOT.active_rod_angle_limits
        if active_rod_angle_range is None
        else active_rod_angle_range
    )

    policy_pos = torch.empty(n, 4, device=env.device)
    policy_pos[:, 0] = default[0] + sample_uniform(
        torch.tensor(float(front_lo), device=env.device),
        torch.tensor(float(front_hi), device=env.device),
        (n,),
        env.device,
    )
    policy_pos[:, 2] = default[2] + sample_uniform(
        torch.tensor(float(front_lo), device=env.device),
        torch.tensor(float(front_hi), device=env.device),
        (n,),
        env.device,
    )
    active = sample_uniform(
        torch.tensor(float(lower), device=env.device),
        torch.tensor(float(upper), device=env.device),
        (n, 2),
        env.device,
    )
    policy_pos[:, 1] = policy_pos[:, 0] - active[:, 0]
    policy_pos[:, 3] = policy_pos[:, 2] + active[:, 1]
    return policy_pos


def _apply_policy_leg_reset(
    asset: Entity,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    leg_ids: torch.Tensor,
    policy_pos: torch.Tensor,
    policy_vel: torch.Tensor,
    *,
    rows: torch.Tensor | None = None,
) -> None:
    """把 policy 主动杆语义 reset 姿态写回模型关节。"""
    _clamp_active_rod_policy_pose(asset, policy_pos)
    _write_leg_values(joint_pos, leg_ids, _policy_leg_pos_to_model(asset, policy_pos), rows)
    _write_leg_values(
        joint_vel,
        leg_ids,
        _policy_leg_vel_to_model(asset, policy_pos, policy_vel),
        rows,
    )


def _reset_wheel_body_ids(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> list[int]:
    """缓存 reset 高度校正用的左右轮 body 索引。"""
    attr_name = f"_reset_wheel_body_ids_{asset_cfg.name}"
    cached = getattr(env, attr_name, None)
    if isinstance(cached, list) and len(cached) == 2:
        return cached
    robot = env.scene[asset_cfg.name]
    body_ids, body_names = robot.find_bodies(("l_wheel_Link", "r_wheel_Link"), preserve_order=True)
    if len(body_ids) != 2:
        raise RuntimeError(f"必须找到左右轮 body，实际找到: {body_names}")
    setattr(env, attr_name, body_ids)
    return body_ids


def _non_jump_reset_mask(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
) -> torch.Tensor:
    """返回允许按轮底高度校正的 reset 样本，jump_flag=1 时跳过。"""
    mask = torch.ones(len(env_ids), device=env.device, dtype=torch.bool)
    if not hasattr(env, "command_manager"):
        return mask
    try:
        cmd = env.command_manager.get_command(command_name)
    except Exception:
        return mask
    if cmd.shape[-1] > 5:
        mask &= cmd[env_ids, 5] <= 0.5
    return mask


def _lift_root_to_wheel_clearance(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset: Entity,
    asset_cfg: SceneEntityCfg,
    min_clearance: float,
    command_name: str,
    terrain_height_sensor_names: tuple[str, ...] | None = None,
    allow_lowering: bool = False,
    max_adjustment: float = 0.3,
) -> None:
    """在 reset 后把轮底最小高度校正到指定离地间隙。"""
    if min_clearance < 0.0 or len(env_ids) == 0:
        return

    active_local = _non_jump_reset_mask(env, env_ids, command_name)
    if not active_local.any():
        return
    active_env_ids = env_ids[active_local]

    env.sim.forward()
    target_wheel_center_height = _FOURBAR_WHEEL_RADIUS_M + float(min_clearance)
    terrain_center_clearance = _wheel_center_clearance_from_terrain_sensors(
        env,
        active_env_ids,
        terrain_height_sensor_names,
    )
    if terrain_center_clearance is None:
        body_ids = _reset_wheel_body_ids(env, asset_cfg)
        wheel_pos_w = asset.data.body_link_pos_w[active_env_ids][:, body_ids, :]
        ground_z = env.scene.env_origins[active_env_ids, 2].unsqueeze(1)
        wheel_bottom = wheel_pos_w[:, :, 2] - ground_z - _FOURBAR_WHEEL_RADIUS_M
        min_wheel_bottom = wheel_bottom.min(dim=1).values
        adjustment = torch.clamp(float(min_clearance) - min_wheel_bottom, min=0.0)
        before_wheel_bottom = min_wheel_bottom
        mode = "origin"
    else:
        before_wheel_bottom = terrain_center_clearance - _FOURBAR_WHEEL_RADIUS_M
        adjustment = target_wheel_center_height - terrain_center_clearance
        if not allow_lowering:
            adjustment = torch.clamp(adjustment, min=0.0)
        adjustment = torch.clamp(
            adjustment,
            min=-float(max_adjustment),
            max=float(max_adjustment),
        )
        mode = "terrain"

    if adjustment.any():
        pos = asset.data.root_link_pos_w[active_env_ids].clone()
        quat = asset.data.root_link_quat_w[active_env_ids].clone()
        pos[:, 2] += adjustment
        asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=active_env_ids)
        env.sim.forward()

    if hasattr(env, "extras"):
        log = env.extras.setdefault("log", {})
        after_wheel_bottom = before_wheel_bottom + adjustment
        log["Reset/wheel_clearance_before_min_m"] = float(before_wheel_bottom.min().item())
        log["Reset/wheel_clearance_after_min_m"] = float(after_wheel_bottom.min().item())
        log["Reset/wheel_clearance_adjustment_max_m"] = float(adjustment.max().item())
        log["Reset/wheel_clearance_adjustment_min_m"] = float(adjustment.min().item())
        log["Reset/wheel_clearance_adjustment_mean_m"] = float(adjustment.mean().item())
        log["Reset/wheel_clearance_adjustment_ratio"] = float(
            (torch.abs(adjustment) > 1.0e-6).float().mean().item()
        )
        log["Reset/wheel_clearance_mode_terrain"] = float(mode == "terrain")


def _wheel_center_clearance_from_terrain_sensors(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    sensor_names: tuple[str, ...] | None,
) -> torch.Tensor | None:
    """读取左右轮 raycast 高度，返回轮心到真实地形的最小距离。"""
    if not sensor_names:
        return None

    try:
        env.sim.sense()
    except Exception:
        return None

    clearances: list[torch.Tensor] = []
    for sensor_name in sensor_names:
        try:
            sensor = env.scene[sensor_name]
            invalidate = getattr(sensor, "_invalidate_cache", None)
            if callable(invalidate):
                invalidate()
            heights = sensor.data.heights
        except Exception:
            continue
        if not isinstance(heights, torch.Tensor) or heights.shape[0] != env.num_envs:
            continue
        clearances.append(heights.reshape(env.num_envs, -1)[env_ids].min(dim=1).values)

    if not clearances:
        return None
    return torch.stack(clearances, dim=1).min(dim=1).values


def _entity_model_field(
    env: ManagerBasedRlEnv,
    asset: Entity,
    field_name: str,
    env_ids: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """读取 entity geom 对应的 MuJoCo model 字段，兼容逐 env 和全局字段。"""
    geom_ids = asset.indexing.geom_ids.to(device=env.device, dtype=torch.long)
    value = torch.as_tensor(getattr(env.sim.model, field_name), device=env.device)
    if dtype is not None:
        value = value.to(dtype=dtype)
    if value.ndim >= 2 and value.shape[0] == env.num_envs:
        return value[env_ids][:, geom_ids]
    selected = value[geom_ids]
    return selected.unsqueeze(0).expand(len(env_ids), *selected.shape)


def _entity_model_field_one_env(
    env: ManagerBasedRlEnv,
    asset: Entity,
    field_name: str,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """读取单个 env 的 entity geom model 字段，用于类型和碰撞属性。"""
    geom_ids = asset.indexing.geom_ids.to(device=env.device, dtype=torch.long)
    value = torch.as_tensor(getattr(env.sim.model, field_name), device=env.device)
    if dtype is not None:
        value = value.to(dtype=dtype)
    if value.ndim >= 2 and value.shape[0] == env.num_envs:
        return value[0, geom_ids]
    return value[geom_ids]


def _entity_data_geom_field(
    env: ManagerBasedRlEnv,
    asset: Entity,
    field_name: str,
    env_ids: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """读取 entity geom 对应的 MuJoCo data 字段。"""
    geom_ids = asset.indexing.geom_ids.to(device=env.device, dtype=torch.long)
    value = torch.as_tensor(getattr(asset.data.data, field_name), device=env.device)
    if dtype is not None:
        value = value.to(dtype=dtype)
    return value[env_ids][:, geom_ids]


def _entity_collision_geom_local_mask(
    env: ManagerBasedRlEnv,
    asset: Entity,
) -> torch.Tensor:
    """返回 entity 内可碰撞、非地形 geom 的 local mask。"""
    geom_type = _entity_model_field_one_env(env, asset, "geom_type", dtype=torch.long)
    contype = _entity_model_field_one_env(env, asset, "geom_contype", dtype=torch.long)
    conaffinity = _entity_model_field_one_env(env, asset, "geom_conaffinity", dtype=torch.long)
    finite_geom = (geom_type != _GEOM_PLANE) & (geom_type != _GEOM_HFIELD)
    collidable = (contype != 0) | (conaffinity != 0)
    return finite_geom & collidable


def _mesh_geom_min_z(
    env: ManagerBasedRlEnv,
    geom_pos: torch.Tensor,
    geom_xmat: torch.Tensor,
    geom_dataid: torch.Tensor,
    local_geom_index: int,
) -> torch.Tensor:
    """计算 mesh geom 的最低世界 z；默认训练模型主要使用 primitive collision。"""
    mesh_id = int(geom_dataid[local_geom_index].item())
    mesh_vertadr = torch.as_tensor(env.sim.model.mesh_vertadr, device=env.device)
    mesh_vertnum = torch.as_tensor(env.sim.model.mesh_vertnum, device=env.device)
    vert_adr = int(mesh_vertadr[mesh_id].item())
    vert_num = int(mesh_vertnum[mesh_id].item())
    vertices = torch.as_tensor(
        env.sim.model.mesh_vert,
        device=env.device,
        dtype=geom_pos.dtype,
    )[vert_adr : vert_adr + vert_num]
    row_z = geom_xmat[:, local_geom_index, 2, :]
    return geom_pos[:, local_geom_index, 2] + torch.matmul(vertices, row_z.T).amin(dim=0)


def _collision_geom_min_z(
    env: ManagerBasedRlEnv,
    asset: Entity,
    env_ids: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """计算 entity 所有可碰撞 geom 的最低世界 z。"""
    env.sim.forward()
    mask = _entity_collision_geom_local_mask(env, asset)
    if not torch.any(mask):
        return torch.zeros(len(env_ids), device=env.device), 0

    geom_pos = _entity_data_geom_field(env, asset, "geom_xpos", env_ids, dtype=torch.float32)
    geom_xmat = _entity_data_geom_field(env, asset, "geom_xmat", env_ids, dtype=torch.float32)
    if geom_xmat.shape[-1] == 9:
        geom_xmat = geom_xmat.reshape(*geom_xmat.shape[:-1], 3, 3)
    geom_size = _entity_model_field(
        env,
        asset,
        "geom_size",
        env_ids,
        dtype=geom_pos.dtype,
    )
    geom_type = _entity_model_field_one_env(env, asset, "geom_type", dtype=torch.long)
    geom_dataid = _entity_model_field_one_env(env, asset, "geom_dataid", dtype=torch.long)

    geom_pos = geom_pos[:, mask]
    geom_xmat = geom_xmat[:, mask]
    geom_size = geom_size[:, mask]
    geom_type = geom_type[mask]
    geom_dataid = geom_dataid[mask]

    min_z = geom_pos[:, :, 2].clone()

    is_sphere = geom_type == _GEOM_SPHERE
    min_z = torch.where(is_sphere.unsqueeze(0), geom_pos[:, :, 2] - geom_size[:, :, 0], min_z)

    is_capsule = geom_type == _GEOM_CAPSULE
    is_cylinder = geom_type == _GEOM_CYLINDER
    is_axial_round = is_capsule | is_cylinder
    if torch.any(is_axial_round):
        axis_z = torch.abs(geom_xmat[:, :, 2, 2])
        radial_z = torch.sqrt(torch.clamp(1.0 - axis_z * axis_z, min=0.0))
        half_length = geom_size[:, :, 1]
        half_length = torch.where(
            is_capsule.unsqueeze(0), half_length + geom_size[:, :, 0], half_length
        )
        extent = half_length * axis_z + geom_size[:, :, 0] * radial_z
        min_z = torch.where(is_axial_round.unsqueeze(0), geom_pos[:, :, 2] - extent, min_z)

    is_box = geom_type == _GEOM_BOX
    if torch.any(is_box):
        extent = torch.sum(torch.abs(geom_xmat[:, :, 2, :]) * geom_size[:, :, :3], dim=-1)
        min_z = torch.where(is_box.unsqueeze(0), geom_pos[:, :, 2] - extent, min_z)

    is_ellipsoid = geom_type == _GEOM_ELLIPSOID
    if torch.any(is_ellipsoid):
        extent = torch.linalg.norm(geom_xmat[:, :, 2, :] * geom_size[:, :, :3], dim=-1)
        min_z = torch.where(is_ellipsoid.unsqueeze(0), geom_pos[:, :, 2] - extent, min_z)

    mesh_indices = torch.nonzero(geom_type == _GEOM_MESH, as_tuple=False).flatten()
    for mesh_index in mesh_indices.tolist():
        min_z[:, mesh_index] = _mesh_geom_min_z(
            env,
            geom_pos,
            geom_xmat,
            geom_dataid,
            int(mesh_index),
        )

    return min_z.amin(dim=1), int(mask.sum().item())


def snap_root_to_collision_clearance(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    clearance_range: tuple[float, float] = (0.001, 0.005),
    max_downward_adjustment: float = 0.5,
    max_upward_adjustment: float = 0.05,
    command_name: str = "velocity_height",
) -> None:
    """把 reset 后机器人整体平移到最低碰撞体接近地面的小间隙。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    if len(env_ids) == 0:
        return

    active_local = _non_jump_reset_mask(env, env_ids, command_name)
    if not active_local.any():
        return
    active_env_ids = env_ids[active_local]

    asset: Entity = env.scene[asset_cfg.name]
    before_min_z, geom_count = _collision_geom_min_z(env, asset, active_env_ids)
    if geom_count <= 0:
        if hasattr(env, "extras"):
            env.extras.setdefault("log", {})["Reset/collision_snap_missing_geom"] = 1.0
        return

    target_clearance = sample_uniform(
        torch.tensor(float(clearance_range[0]), device=env.device),
        torch.tensor(float(clearance_range[1]), device=env.device),
        (len(active_env_ids),),
        env.device,
    )
    target_min_z = env.scene.env_origins[active_env_ids, 2] + target_clearance
    adjustment = torch.clamp(
        target_min_z - before_min_z,
        min=-float(max_downward_adjustment),
        max=float(max_upward_adjustment),
    )

    if adjustment.any():
        pos = asset.data.root_link_pos_w[active_env_ids].clone()
        quat = asset.data.root_link_quat_w[active_env_ids].clone()
        pos[:, 2] += adjustment
        asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=active_env_ids)
        env.sim.forward()

    after_min_z, _ = _collision_geom_min_z(env, asset, active_env_ids)
    before_clearance = before_min_z - env.scene.env_origins[active_env_ids, 2]
    after_clearance = after_min_z - env.scene.env_origins[active_env_ids, 2]

    if hasattr(env, "extras"):
        log = env.extras.setdefault("log", {})
        log["Reset/collision_snap_geom_count"] = float(geom_count)
        log["Reset/collision_snap_target_clearance_mean_m"] = float(target_clearance.mean().item())
        log["Reset/collision_snap_before_min_m"] = float(before_clearance.min().item())
        log["Reset/collision_snap_before_mean_m"] = float(before_clearance.mean().item())
        log["Reset/collision_snap_after_min_m"] = float(after_clearance.min().item())
        log["Reset/collision_snap_after_mean_m"] = float(after_clearance.mean().item())
        log["Reset/collision_snap_adjustment_min_m"] = float(adjustment.min().item())
        log["Reset/collision_snap_adjustment_max_m"] = float(adjustment.max().item())
        log["Reset/collision_snap_adjustment_mean_m"] = float(adjustment.mean().item())
        log["Reset/collision_snap_adjustment_abs_mean_m"] = float(
            torch.abs(adjustment).mean().item()
        )
        log["Reset/collision_snap_adjustment_ratio"] = float(
            (torch.abs(adjustment) > 1.0e-6).float().mean().item()
        )


def _clamp_policy_leg_pose(
    asset: Entity,
    joint_pos: torch.Tensor,
    env_ids: torch.Tensor,
    leg_ids: torch.Tensor,
    *,
    rows: torch.Tensor | None = None,
) -> None:
    """按模型语义裁剪腿部关节：闭链裁剪主动杆夹角，开链裁剪单关节限位。"""
    if is_closedchain_model(asset) or is_fourbar_surrogate_model(asset):
        policy_pos = _model_leg_pos_to_policy(asset, _leg_values(joint_pos, leg_ids, rows))
        _clamp_active_rod_policy_pose(asset, policy_pos)
        _write_leg_values(joint_pos, leg_ids, _policy_leg_pos_to_model(asset, policy_pos), rows)
        return

    soft_limits = asset.data.soft_joint_pos_limits
    if soft_limits is None:
        return
    if rows is None:
        joint_pos[:, leg_ids] = torch.clamp(
            joint_pos[:, leg_ids],
            soft_limits[env_ids[:, None], leg_ids, 0],
            soft_limits[env_ids[:, None], leg_ids, 1],
        )
    else:
        active_env_ids = env_ids[rows]
        joint_pos[rows[:, None], leg_ids] = torch.clamp(
            joint_pos[rows[:, None], leg_ids],
            soft_limits[active_env_ids[:, None], leg_ids, 0],
            soft_limits[active_env_ids[:, None], leg_ids, 1],
        )


def reset_joints(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_offset_range: float = 0.0,
    joint_vel_range: tuple[float, float] = (0.0, 0.0),
    wheel_joint_vel_range: tuple[float, float] = (0.0, 0.0),
    wheel_joint_randomization_prob: float | None = None,
    hip_joint_offset_range: float | tuple[float, float] | None = None,
    knee_joint_offset_range: float | tuple[float, float] | None = None,
    joint_randomization_prob: float = 1.0,
    full_joint_randomization: bool = False,
    full_front_joint_offset_range: float | tuple[float, float] = 1.0,
    full_active_rod_angle_range: tuple[float, float] | None = None,
    curriculum_stages: list[dict] | None = None,
    use_iterations: bool = False,
    steps_per_policy_iter: int = 64,
    offset_iter: int = 0,
    recovery_joint_offset_range: float = 0.0,
    recovery_joint_vel_range: tuple[float, float] = (0.0, 0.0),
    align_root_height_to_wheels: bool = False,
    wheel_clearance: float = _DEFAULT_RESET_WHEEL_CLEARANCE_M,
    command_name: str = "velocity_height",
    height_conditioned_default: bool = False,
    terrain_height_sensor_names: tuple[str, ...] | None = None,
    allow_wheel_clearance_lowering: bool = False,
    max_wheel_clearance_adjustment: float = 0.3,
) -> None:
    """重置关节位置到默认站立姿态(default_joint_pos)附近小范围随机。"""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    stage, curriculum_progress = _active_curriculum_stage(
        env,
        curriculum_stages,
        use_iterations=use_iterations,
        steps_per_policy_iter=steps_per_policy_iter,
        offset_iter=offset_iter,
    )
    joint_offset_range = _stage_value(stage, "joint_offset_range", joint_offset_range)
    joint_vel_range = _stage_value(stage, "joint_vel_range", joint_vel_range)
    wheel_joint_vel_range = _stage_value(stage, "wheel_joint_vel_range", wheel_joint_vel_range)
    wheel_joint_randomization_prob = _stage_value(
        stage,
        "wheel_joint_randomization_prob",
        wheel_joint_randomization_prob,
    )
    hip_joint_offset_range = _stage_value(stage, "hip_joint_offset_range", hip_joint_offset_range)
    knee_joint_offset_range = _stage_value(
        stage, "knee_joint_offset_range", knee_joint_offset_range
    )
    full_joint_randomization = bool(
        _stage_value(stage, "full_joint_randomization", full_joint_randomization)
    )
    full_front_joint_offset_range = _stage_value(
        stage, "full_front_joint_offset_range", full_front_joint_offset_range
    )
    full_active_rod_angle_range = _stage_value(
        stage, "full_active_rod_angle_range", full_active_rod_angle_range
    )
    joint_randomization_prob = float(
        _stage_value(stage, "joint_randomization_prob", joint_randomization_prob)
    )
    joint_randomization_prob = min(max(joint_randomization_prob, 0.0), 1.0)
    if wheel_joint_randomization_prob is None:
        wheel_joint_randomization_prob = joint_randomization_prob
    else:
        wheel_joint_randomization_prob = min(max(float(wheel_joint_randomization_prob), 0.0), 1.0)

    asset: Entity = env.scene[asset_cfg.name]

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    wheel_ids = tensor_ids(wheel_joint_ids(asset), device=env.device)
    joint_pos[:, wheel_ids] = 0.0
    wheel_vel_randomization_enabled = (
        abs(float(wheel_joint_vel_range[0])) > 0.0 or abs(float(wheel_joint_vel_range[1])) > 0.0
    )
    if wheel_vel_randomization_enabled:
        wheel_randomize_mask = (
            torch.rand(len(env_ids), device=env.device) < wheel_joint_randomization_prob
        )
        if wheel_randomize_mask.any():
            wheel_rows = wheel_randomize_mask.nonzero().flatten()
            joint_vel[wheel_rows[:, None], wheel_ids] = sample_uniform(
                torch.tensor(float(wheel_joint_vel_range[0]), device=env.device),
                torch.tensor(float(wheel_joint_vel_range[1]), device=env.device),
                (int(wheel_rows.numel()), len(wheel_ids)),
                env.device,
            )
    else:
        wheel_randomize_mask = torch.zeros(len(env_ids), device=env.device, dtype=torch.bool)

    leg_ids = tensor_ids(policy_leg_joint_ids(asset), device=env.device)
    if height_conditioned_default and hasattr(env, "command_manager"):
        try:
            default_policy_pos = update_policy_default_from_height_cache(
                env,
                command_name,
                env_ids=env_ids,
            )
            local_env_ids = env_ids.to(device=env.device, dtype=torch.long)
            policy_leg_pos = default_policy_pos[local_env_ids]
            policy_leg_vel = torch.zeros_like(policy_leg_pos)
            _apply_policy_leg_reset(
                asset,
                joint_pos,
                joint_vel,
                leg_ids,
                policy_leg_pos,
                policy_leg_vel,
            )
        except Exception:
            if hasattr(env, "extras"):
                env.extras.setdefault("log", {})["Reset/height_conditioned_default_failed"] = 1.0
    randomization_enabled = (
        full_joint_randomization
        or hip_joint_offset_range is not None
        or knee_joint_offset_range is not None
        or joint_offset_range > 0.0
    )
    if randomization_enabled:
        randomize_mask = torch.rand(len(env_ids), device=env.device) < joint_randomization_prob
    else:
        randomize_mask = torch.zeros(len(env_ids), device=env.device, dtype=torch.bool)
    if full_joint_randomization:
        if randomize_mask.any():
            random_rows = randomize_mask.nonzero().flatten()
            n_random = int(random_rows.numel())
            policy_leg_pos = _sample_full_random_policy_leg_pose(
                env,
                n_random,
                full_front_joint_offset_range,
                full_active_rod_angle_range,
            )
            policy_leg_vel = sample_uniform(
                torch.tensor(float(joint_vel_range[0]), device=env.device),
                torch.tensor(float(joint_vel_range[1]), device=env.device),
                (n_random, len(leg_ids)),
                env.device,
            )

            _apply_policy_leg_reset(
                asset,
                joint_pos,
                joint_vel,
                leg_ids,
                policy_leg_pos,
                policy_leg_vel,
                rows=random_rows,
            )
    elif hip_joint_offset_range is not None or knee_joint_offset_range is not None:
        if randomize_mask.any():
            random_rows = randomize_mask.nonzero().flatten()
            n_random = int(random_rows.numel())
            policy_leg_pos = _model_leg_pos_to_policy(
                asset,
                _leg_values(joint_pos, leg_ids, random_rows),
            )
            policy_leg_vel = torch.zeros_like(policy_leg_pos)
            if hip_joint_offset_range is not None:
                policy_leg_pos[:, (0, 2)] += _sample_joint_offset(
                    env,
                    hip_joint_offset_range,
                    (n_random, 2),
                )
            if knee_joint_offset_range is not None:
                policy_leg_pos[:, (1, 3)] += _sample_joint_offset(
                    env,
                    knee_joint_offset_range,
                    (n_random, 2),
                )
            policy_leg_vel[:] = sample_uniform(
                torch.tensor(float(joint_vel_range[0]), device=env.device),
                torch.tensor(float(joint_vel_range[1]), device=env.device),
                (n_random, len(leg_ids)),
                env.device,
            )

            _apply_policy_leg_reset(
                asset,
                joint_pos,
                joint_vel,
                leg_ids,
                policy_leg_pos,
                policy_leg_vel,
                rows=random_rows,
            )
    elif joint_offset_range > 0.0 and randomize_mask.any():
        random_rows = randomize_mask.nonzero().flatten()
        n_random = int(random_rows.numel())
        policy_leg_pos = _model_leg_pos_to_policy(
            asset,
            _leg_values(joint_pos, leg_ids, random_rows),
        )
        policy_leg_vel = torch.zeros_like(policy_leg_pos)
        offset = sample_uniform(
            torch.tensor(-float(joint_offset_range), device=env.device),
            torch.tensor(float(joint_offset_range), device=env.device),
            (n_random, len(leg_ids)),
            env.device,
        )
        policy_leg_pos += offset
        policy_leg_vel[:] = sample_uniform(
            torch.tensor(float(joint_vel_range[0]), device=env.device),
            torch.tensor(float(joint_vel_range[1]), device=env.device),
            (n_random, len(leg_ids)),
            env.device,
        )

        _apply_policy_leg_reset(
            asset,
            joint_pos,
            joint_vel,
            leg_ids,
            policy_leg_pos,
            policy_leg_vel,
            rows=random_rows,
        )

    if hasattr(env, "extras"):
        log = env.extras.setdefault("log", {})
        log["Reset/joint_curriculum_progress"] = float(curriculum_progress)
        log["Reset/joint_randomization_prob"] = float(joint_randomization_prob)
        log["Reset/joint_randomization_ratio"] = randomize_mask.float().mean().item()
        log["Reset/full_joint_randomization"] = float(full_joint_randomization)
        log["Reset/wheel_joint_vel_randomization_prob"] = float(wheel_joint_randomization_prob)
        log["Reset/wheel_joint_vel_randomization_ratio"] = (
            wheel_randomize_mask.float().mean().item()
        )

    recovery_mask = getattr(env, "_recovery_reset_mask", None)
    if (
        isinstance(recovery_mask, torch.Tensor)
        and recovery_mask.shape[0] == env.num_envs
        and recovery_joint_offset_range > 0.0
    ):
        local_recovery = recovery_mask[env_ids].to(device=env.device, dtype=torch.bool)
        if local_recovery.any():
            n_recovery = int(local_recovery.sum().item())
            local_ids = local_recovery.nonzero().flatten()
            policy_leg_pos = _model_leg_pos_to_policy(
                asset, _leg_values(joint_pos, leg_ids, local_ids)
            )
            policy_leg_vel = torch.zeros_like(policy_leg_pos)
            offset = sample_uniform(
                torch.tensor(-float(recovery_joint_offset_range), device=env.device),
                torch.tensor(float(recovery_joint_offset_range), device=env.device),
                (n_recovery, len(leg_ids)),
                env.device,
            )
            policy_leg_pos += offset
            policy_leg_vel[:] = sample_uniform(
                torch.tensor(float(recovery_joint_vel_range[0]), device=env.device),
                torch.tensor(float(recovery_joint_vel_range[1]), device=env.device),
                (n_recovery, len(leg_ids)),
                env.device,
            )

            _apply_policy_leg_reset(
                asset,
                joint_pos,
                joint_vel,
                leg_ids,
                policy_leg_pos,
                policy_leg_vel,
                rows=local_ids,
            )

    cache_reset_mask = getattr(env, "_recovery_cache_reset_mask", None)
    cache_joint_pos = getattr(env, "_recovery_cached_joint_pos", None)
    cache_joint_vel = getattr(env, "_recovery_cached_joint_vel", None)
    if (
        isinstance(cache_reset_mask, torch.Tensor)
        and isinstance(cache_joint_pos, torch.Tensor)
        and isinstance(cache_joint_vel, torch.Tensor)
        and cache_reset_mask.shape[0] == env.num_envs
    ):
        local_cache = cache_reset_mask[env_ids].to(device=env.device, dtype=torch.bool)
        if local_cache.any():
            joint_pos[local_cache] = cache_joint_pos[env_ids[local_cache]]
            joint_vel[local_cache] = cache_joint_vel[env_ids[local_cache]]

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

                if is_closedchain_model(asset):
                    if hasattr(env, "extras"):
                        env.extras.setdefault("log", {})["Jump/closedchain_rsi_disabled"] = 1.0
                    rsi_done_mask = jump_mask
                elif rsi_frames is not None:
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
                        joint_pos[rsi_done_mask, leg_ids[0]] = q_ref[:, 0]
                        joint_pos[rsi_done_mask, leg_ids[1]] = q_ref[:, 1]
                        joint_pos[rsi_done_mask, leg_ids[2]] = q_ref[:, 3]
                        joint_pos[rsi_done_mask, leg_ids[3]] = q_ref[:, 4]
                        joint_vel[rsi_done_mask, leg_ids[0]] = q_vel[:, 0]
                        joint_vel[rsi_done_mask, leg_ids[1]] = q_vel[:, 1]
                        joint_vel[rsi_done_mask, leg_ids[2]] = q_vel[:, 3]
                        joint_vel[rsi_done_mask, leg_ids[3]] = q_vel[:, 4]

                # 未做 RSI 的 jump env 默认保持站立姿态。
                # 只有显式启用部分 RSI 时，未注入的样本才回退到预蹲姿态做消融。
                term = env.command_manager.get_term("velocity_height")
                use_fallback_squat = not (
                    isinstance(term, JumpCommandTerm) and term.cfg.rsi_takeoff_prob <= 0.0
                )
                fallback_mask = jump_mask & ~rsi_done_mask & use_fallback_squat
                if fallback_mask.any():
                    joint_pos[fallback_mask, leg_ids[0]] = _jump_hip_fallback
                    joint_pos[fallback_mask, leg_ids[1]] = _jump_knee_fallback
                    joint_pos[fallback_mask, leg_ids[2]] = _jump_hip_fallback
                    joint_pos[fallback_mask, leg_ids[3]] = _jump_knee_fallback
        except Exception:
            pass

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    if align_root_height_to_wheels:
        _lift_root_to_wheel_clearance(
            env,
            env_ids,
            asset,
            asset_cfg,
            float(wheel_clearance),
            command_name,
            terrain_height_sensor_names,
            bool(allow_wheel_clearance_lowering),
            float(max_wheel_clearance_adjustment),
        )


def push_robots(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """随机速度扰动机器人根节点。"""
    asset: Entity = env.scene[asset_cfg.name]
    vel_w = asset.data.root_link_vel_w[env_ids]
    active_velocity_range = getattr(env, "_push_velocity_range", velocity_range)

    range_list = [
        active_velocity_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=env.device)
    vel_w += sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=env.device)
    asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)

    push_max = max(max(abs(low), abs(high)) for low, high in range_list)
    env.extras.setdefault("log", {})
    env.extras["log"]["PushDisturbance/active_velocity_max"] = torch.tensor(
        float(push_max), device=env.device
    )


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
    _sync_action_leg_pd_gains(env, env_ids, kp_scale, kd_scale)


def _sync_action_leg_pd_gains(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    kp_scale: torch.Tensor,
    kd_scale: torch.Tensor,
) -> None:
    """把 actuator 域随机化的同一组增益同步给自定义 action term。"""
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None:
        return
    for term_name in action_manager.active_terms:
        term = action_manager.get_term(term_name)
        setter = getattr(term, "set_leg_pd_gain_scale", None)
        if setter is not None:
            setter(env_ids, kp_scale, kd_scale)


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

    default_joint_pos = asset.data.default_joint_pos.clone()

    leg_ids = tensor_ids(policy_leg_joint_ids(asset), device=env.device)
    leg_offset = sample_uniform(
        torch.tensor(offset_range[0], device=env.device),
        torch.tensor(offset_range[1], device=env.device),
        (n, len(leg_ids)),
        env.device,
    )
    selected_joint_pos = default_joint_pos[env_ids].clone()
    policy_leg_pos = _model_leg_pos_to_policy(asset, selected_joint_pos[:, leg_ids])
    policy_leg_pos += leg_offset
    _clamp_active_rod_policy_pose(asset, policy_leg_pos)
    selected_joint_pos[:, leg_ids] = _policy_leg_pos_to_model(asset, policy_leg_pos)

    if not (is_closedchain_model(asset) or is_fourbar_surrogate_model(asset)):
        soft_limits = asset.data.soft_joint_pos_limits
        if soft_limits is not None:
            selected_joint_pos[:, leg_ids] = torch.clamp(
                selected_joint_pos[:, leg_ids],
                soft_limits[env_ids[:, None], leg_ids, 0],
                soft_limits[env_ids[:, None], leg_ids, 1],
            )

    default_joint_pos[env_ids] = selected_joint_pos

    asset.data.default_joint_pos[env_ids] = default_joint_pos[env_ids]
