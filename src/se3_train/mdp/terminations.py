"""SE3 轮腿机器人的终止函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_shared import output_to_policy_pos_torch, output_to_policy_vel_torch
from se3_train.mdp import recovery_state
from se3_train.mdp.contact_utils import finite_contact_force_norm
from se3_train.mdp.joint_indices import (
    is_fourbar_surrogate_model,
    policy_leg_joint_ids,
    wheel_joint_ids,
)
from se3_train.mdp.leg_alignment import wheel_alignment_ok

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.sensor import ContactSensor

_DEFAULT_TERMINATION_LOG_INTERVAL_STEPS = 64


def _should_log_step(
    env: ManagerBasedRlEnv, interval: int = _DEFAULT_TERMINATION_LOG_INTERVAL_STEPS
) -> bool:
    """按 policy step 降频写 termination 诊断日志。"""
    step = int(getattr(env, "common_step_counter", 0))
    interval = max(1, int(getattr(env, "_se3_termination_log_interval_steps", interval)))
    return interval <= 1 or (step - 1) % interval == 0


def _recovery_reset_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
    """返回当前仍处于 recovery active 模式的 env。"""
    return recovery_state.recovery_active_mask(env)


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


def _policy_leg_state_and_default(robot) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 policy 主动杆语义下的腿部位置、默认位置和速度。"""
    leg_ids = policy_leg_joint_ids(robot)
    leg_pos = robot.data.joint_pos[:, leg_ids]
    leg_default = robot.data.default_joint_pos[:, leg_ids]
    leg_vel = robot.data.joint_vel[:, leg_ids]
    if is_fourbar_surrogate_model(robot):
        leg_pos = output_to_policy_pos_torch(leg_pos)
        leg_default = output_to_policy_pos_torch(leg_default)
        leg_vel = output_to_policy_vel_torch(robot.data.joint_pos[:, leg_ids], leg_vel)
    return leg_pos, leg_default, leg_vel


class BadOrientationDelayed:
    """倾斜超过阈值连续 max_steps 步才终止(给恢复机会)。"""

    def __init__(self) -> None:
        self._fail_count: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        limit_angle: float = 0.5236,
        max_steps: int = 100,
        recovery_grace_steps: int = 0,
        recovery_terminate: bool = True,
    ) -> torch.Tensor:
        if (
            self._fail_count is None
            or self._fail_count.shape[0] != env.num_envs
            or self._fail_count.device != env.device
        ):
            self._fail_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

        robot = env.scene["robot"]
        pg_z = robot.data.projected_gravity_b[:, 2]
        tilt_angle = torch.acos(torch.clamp(-pg_z, -1.0, 1.0))
        raw_bad = tilt_angle > limit_angle
        bad = raw_bad
        recovery_mask = _recovery_reset_mask(env)
        in_recovery_grace = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        if not recovery_terminate:
            bad = bad & ~recovery_mask
        if recovery_grace_steps > 0:
            in_recovery_grace = recovery_mask & (
                env.episode_length_buf <= int(recovery_grace_steps)
            )
            bad = bad & ~in_recovery_grace

        self._fail_count[bad] += 1
        self._fail_count[~bad] = 0
        self._fail_count[env.episode_length_buf <= 1] = 0

        terminated = self._fail_count > max_steps
        if hasattr(env, "extras"):

            def _masked_rate(values: torch.Tensor, mask: torch.Tensor) -> float:
                if not mask.any():
                    return 0.0
                return values[mask].float().mean().item()

            env.extras.setdefault("log", {}).update(
                {
                    "Recovery/bad_orientation_raw_rate": _masked_rate(raw_bad, recovery_mask),
                    "Recovery/bad_orientation_counted_rate": _masked_rate(bad, recovery_mask),
                    "Recovery/bad_orientation_grace_rate": _masked_rate(
                        in_recovery_grace, recovery_mask
                    ),
                    "Recovery/bad_orientation_termination_rate": _masked_rate(
                        terminated, recovery_mask
                    ),
                }
            )
            # reset 日志会被清空，所以先缓存逐环境诊断，由 command reset 阶段搬运到 log。
            env.extras["_bad_orientation_diag"] = {
                "raw_bad": raw_bad.detach().clone(),
                "counted_bad": bad.detach().clone(),
                "terminated": terminated.detach().clone(),
                "recovery_mask": recovery_mask.detach().clone(),
                "recovery_grace": in_recovery_grace.detach().clone(),
            }

        return terminated

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._fail_count is not None and env_ids is not None:
            self._fail_count[env_ids] = 0


bad_orientation_delayed = BadOrientationDelayed()


class RecoveryStagnation:
    """恢复样本长时间没有变好时终止，避免躺平刷低惩罚。"""

    def __init__(self) -> None:
        self._best_score: torch.Tensor | None = None
        self._stagnation_count: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        max_steps: int = 256,
        min_delta: float = 0.02,
    ) -> torch.Tensor:
        active = _recovery_reset_mask(env)
        robot = env.scene["robot"]
        pg_z = robot.data.projected_gravity_b[:, 2]
        score = torch.clamp((-pg_z + 1.0) * 0.5, 0.0, 1.0)

        if (
            self._best_score is None
            or self._best_score.shape[0] != env.num_envs
            or self._best_score.device != env.device
        ):
            self._best_score = score.detach().clone()
            self._stagnation_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

        assert self._stagnation_count is not None
        first_step = env.episode_length_buf <= 1
        reset_mask = first_step | ~active
        self._best_score[reset_mask] = score[reset_mask]
        self._stagnation_count[reset_mask] = 0

        improved = active & (score > self._best_score + float(min_delta))
        self._best_score = torch.maximum(self._best_score, score.detach())
        self._stagnation_count[active & ~improved] += 1
        self._stagnation_count[improved] = 0

        terminated = active & (self._stagnation_count >= int(max_steps))
        if hasattr(env, "extras"):
            env.extras.setdefault("log", {}).update(
                {
                    "Recovery/stagnation_steps": self._stagnation_count[active]
                    .float()
                    .mean()
                    .item()
                    if active.any()
                    else 0.0,
                    "Recovery/stagnation_termination_rate": terminated[active].float().mean().item()
                    if active.any()
                    else 0.0,
                }
            )
        return terminated

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._stagnation_count is not None and env_ids is not None:
            self._stagnation_count[env_ids] = 0


recovery_stagnation = RecoveryStagnation()


def catastrophic_state(
    env: ManagerBasedRlEnv,
    max_leg_pos_error: float | None = 3.0,
    max_leg_vel: float = 120.0,
    max_root_lin_vel: float = 80.0,
    max_root_ang_vel: float = 500.0,
    min_base_height: float = -0.5,
    max_base_height: float = 3.0,
) -> torch.Tensor:
    """物理状态已经发散时立即终止，防止 NaN 观测进入 PPO。"""
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel
    root_pos = robot.data.root_link_pos_w
    root_lin_vel = robot.data.root_link_lin_vel_w
    root_ang_vel = robot.data.root_link_ang_vel_b
    projected_gravity = robot.data.projected_gravity_b

    finite = (
        torch.isfinite(joint_pos).all(dim=1)
        & torch.isfinite(joint_vel).all(dim=1)
        & torch.isfinite(root_pos).all(dim=1)
        & torch.isfinite(root_lin_vel).all(dim=1)
        & torch.isfinite(root_ang_vel).all(dim=1)
        & torch.isfinite(projected_gravity).all(dim=1)
    )

    leg_pos, leg_default, leg_vel = _policy_leg_state_and_default(robot)
    base_height = root_pos[:, 2] - env.scene.env_origins[:, 2]

    if max_leg_pos_error is None:
        leg_pos_bad = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    else:
        leg_pos_bad = torch.any(torch.abs(leg_pos - leg_default) > float(max_leg_pos_error), dim=1)
    leg_vel_bad = torch.any(torch.abs(leg_vel) > float(max_leg_vel), dim=1)
    root_lin_bad = torch.linalg.norm(root_lin_vel, dim=1) > float(max_root_lin_vel)
    root_ang_bad = torch.linalg.norm(root_ang_vel, dim=1) > float(max_root_ang_vel)
    height_bad = (base_height < float(min_base_height)) | (base_height > float(max_base_height))

    terminated = ~finite | leg_pos_bad | leg_vel_bad | root_lin_bad | root_ang_bad | height_bad
    if hasattr(env, "extras") and _should_log_step(env):
        env.extras.setdefault("log", {}).update(
            {
                "Episode_Termination/catastrophic_state": terminated.float().mean().item(),
                "Debug/catastrophic_nonfinite": (~finite).float().mean().item(),
                "Debug/catastrophic_leg_pos": leg_pos_bad.float().mean().item(),
                "Debug/catastrophic_leg_vel": leg_vel_bad.float().mean().item(),
                "Debug/catastrophic_root_vel": (root_lin_bad | root_ang_bad).float().mean().item(),
                "Debug/catastrophic_height": height_bad.float().mean().item(),
            }
        )
    return terminated


_RECOVERY_STAND_BIN_NAMES = ("0_30", "30_60", "60_90", "90_135", "135_180")


def _bool_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float,
) -> torch.Tensor:
    """读取接触传感器，返回每个 env 是否存在超过阈值的接触。"""
    sensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    force_mag = finite_contact_force_norm(data.force)
    if force_mag.ndim == 1:
        return force_mag > float(force_threshold)
    return (force_mag > float(force_threshold)).any(dim=1)


def _recovery_stand_bins(env: ManagerBasedRlEnv) -> torch.Tensor:
    """读取 Recovery-Stand reset 时记录的初始倾角桶。"""
    bins = getattr(env, "_recovery_stand_init_tilt_bin", None)
    if isinstance(bins, torch.Tensor) and bins.shape[0] == env.num_envs:
        return bins.to(device=env.device, dtype=torch.long).clamp(0, 4)
    init_tilt = getattr(env, "_recovery_init_tilt", None)
    if isinstance(init_tilt, torch.Tensor) and init_tilt.shape[0] == env.num_envs:
        init_tilt_deg = torch.rad2deg(init_tilt.to(device=env.device))
        return torch.bucketize(
            init_tilt_deg,
            torch.tensor((30.0, 60.0, 90.0, 135.0), device=env.device),
        )
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.long)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    """计算 mask 内均值；空 mask 返回 0。"""
    if mask.any():
        return values[mask].float().mean().item()
    return 0.0


def recovery_success(
    env: ManagerBasedRlEnv,
    left_wheel_sensor_name: str,
    right_wheel_sensor_name: str,
    nonwheel_sensor_name: str,
    height_sensor_name: str,
    command_name: str,
    upright_angle_deg: float = 15.0,
    max_abs_roll_deg: float = 3.0,
    max_abs_pitch_deg: float = 5.0,
    height_tolerance: float = 0.05,
    ang_vel_threshold: float = 0.5,
    lin_vel_threshold: float = 0.05,
    wheel_speed_threshold: float = 0.05,
    wheel_radius: float = 0.059,
    force_threshold: float = 1.0,
    stable_steps_required: int = 50,
    min_episode_steps: int = 50,
    min_wheel_lateral_distance: float = 0.40,
    max_wheel_lateral_distance: float = 0.46,
    max_wheel_fore_aft_offset: float = 0.03,
) -> torch.Tensor:
    """纯自起任务成功终止：连续稳定站立到可交接窗口。"""
    active = recovery_state.recovery_active_mask(env)
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    tilt = torch.acos(torch.clamp(-pg_z, -1.0, 1.0))
    tilt_deg = torch.rad2deg(tilt)
    upright_limit = torch.deg2rad(torch.tensor(float(upright_angle_deg), device=env.device))
    upright_ok = tilt < upright_limit
    current_pitch = torch.asin(torch.clamp(robot.data.projected_gravity_b[:, 0], -1.0, 1.0))
    current_roll = torch.asin(torch.clamp(-robot.data.projected_gravity_b[:, 1], -1.0, 1.0))
    roll_limit = torch.deg2rad(torch.tensor(float(max_abs_roll_deg), device=env.device))
    pitch_limit = torch.deg2rad(torch.tensor(float(max_abs_pitch_deg), device=env.device))
    roll_ok = torch.abs(current_roll) < roll_limit
    pitch_ok = torch.abs(current_pitch) < pitch_limit

    cmd = env.command_manager.get_command(command_name)
    height_sensor = env.scene[height_sensor_name]
    height = torch.nan_to_num(height_sensor.data.heights[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
    height_error = torch.abs(height - cmd[:, 4])
    height_ok = height_error < float(height_tolerance)

    ang_vel_norm = torch.linalg.norm(robot.data.root_link_ang_vel_b, dim=1)
    lin_vel_norm = torch.linalg.norm(robot.data.root_link_lin_vel_b, dim=1)
    ang_vel_ok = ang_vel_norm < float(ang_vel_threshold)
    lin_vel_ok = lin_vel_norm < float(lin_vel_threshold)
    wheel_vel = robot.data.joint_vel[:, wheel_joint_ids(robot)]
    wheel_forward_speed = torch.stack(
        (
            wheel_vel[:, 0] * float(wheel_radius),
            -wheel_vel[:, 1] * float(wheel_radius),
        ),
        dim=1,
    )
    wheel_speed = torch.linalg.norm(wheel_forward_speed, dim=1)
    wheel_speed_ok = wheel_speed < float(wheel_speed_threshold)

    left_contact = _bool_contact(env, left_wheel_sensor_name, force_threshold)
    right_contact = _bool_contact(env, right_wheel_sensor_name, force_threshold)
    dual_wheel_contact = left_contact & right_contact
    nonwheel_contact = _bool_contact(env, nonwheel_sensor_name, force_threshold)
    nonwheel_clear = ~nonwheel_contact
    wheel_alignment, wheel_lateral_distance, wheel_fore_aft_offset = wheel_alignment_ok(
        env,
        min_lateral_distance=min_wheel_lateral_distance,
        max_lateral_distance=max_wheel_lateral_distance,
        max_fore_aft_offset=max_wheel_fore_aft_offset,
    )

    raw_success = (
        active
        & upright_ok
        & roll_ok
        & pitch_ok
        & height_ok
        & ang_vel_ok
        & lin_vel_ok
        & wheel_speed_ok
        & dual_wheel_contact
        & nonwheel_clear
        & wheel_alignment
    )
    completed = recovery_state.update_success_window(
        env,
        raw_success,
        stable_steps_required=stable_steps_required,
        min_episode_steps=min_episode_steps,
    )

    if hasattr(env, "extras"):
        episode = recovery_state.recovery_episode_mask(env)
        time_to_success = recovery_state.ensure_long_buffer(env, "_recovery_time_to_success_steps")
        timeout = (env.episode_length_buf >= env.max_episode_length) & (time_to_success < 0)
        bins = _recovery_stand_bins(env)
        log = env.extras.setdefault("log", {})
        log.update(
            {
                "RecoveryStand/success_condition/upright": _masked_mean(upright_ok.float(), active),
                "RecoveryStand/success_condition/roll": _masked_mean(roll_ok.float(), active),
                "RecoveryStand/success_condition/pitch": _masked_mean(pitch_ok.float(), active),
                "RecoveryStand/success_condition/height": _masked_mean(height_ok.float(), active),
                "RecoveryStand/success_condition/ang_vel": _masked_mean(ang_vel_ok.float(), active),
                "RecoveryStand/success_condition/lin_vel": _masked_mean(lin_vel_ok.float(), active),
                "RecoveryStand/success_condition/wheel_speed": _masked_mean(
                    wheel_speed_ok.float(), active
                ),
                "RecoveryStand/success_condition/dual_wheel": _masked_mean(
                    dual_wheel_contact.float(), active
                ),
                "RecoveryStand/success_condition/nonwheel_clear": _masked_mean(
                    nonwheel_clear.float(), active
                ),
                "RecoveryStand/success_condition/wheel_alignment": _masked_mean(
                    wheel_alignment.float(), active
                ),
                "RecoveryStand/success_raw_rate": _masked_mean(raw_success.float(), active),
                "RecoveryStand/success_completed_rate": _masked_mean(completed.float(), episode),
                "RecoveryStand/stable_steps": _masked_mean(
                    recovery_state.ensure_long_buffer(env, "_recovery_success_steps").float(),
                    episode,
                ),
                "RecoveryStand/tilt_deg": _masked_mean(tilt_deg, episode),
                "RecoveryStand/abs_roll_deg": _masked_mean(
                    torch.rad2deg(torch.abs(current_roll)), episode
                ),
                "RecoveryStand/abs_pitch_deg": _masked_mean(
                    torch.rad2deg(torch.abs(current_pitch)), episode
                ),
                "RecoveryStand/height_error": _masked_mean(height_error, episode),
                "RecoveryStand/lin_vel_norm": _masked_mean(lin_vel_norm, episode),
                "RecoveryStand/wheel_speed_norm": _masked_mean(wheel_speed, episode),
                "RecoveryStand/wheel_lateral_distance_m": _masked_mean(
                    wheel_lateral_distance, episode
                ),
                "RecoveryStand/wheel_fore_aft_offset_m": _masked_mean(
                    wheel_fore_aft_offset, episode
                ),
                "Episode_Termination/recovery_success": completed.float().mean().item(),
            }
        )
        for bin_id, bin_name in enumerate(_RECOVERY_STAND_BIN_NAMES):
            in_bin = episode & (bins == bin_id)
            success_in_bin = in_bin & (time_to_success >= 0)
            timeout_in_bin = in_bin & timeout
            log.update(
                {
                    f"RecoveryStand/success_rate_by_tilt_bin/{bin_name}": _masked_mean(
                        success_in_bin.float(), in_bin
                    ),
                    f"RecoveryStand/time_to_success_by_tilt_bin/{bin_name}": _masked_mean(
                        time_to_success.float(), success_in_bin
                    ),
                    f"RecoveryStand/timeout_rate_by_tilt_bin/{bin_name}": _masked_mean(
                        timeout.float(), in_bin
                    ),
                    f"RecoveryStand/final_tilt_deg_by_tilt_bin/{bin_name}": _masked_mean(
                        tilt_deg, timeout_in_bin
                    ),
                    f"RecoveryStand/final_height_error_by_tilt_bin/{bin_name}": _masked_mean(
                        height_error, timeout_in_bin
                    ),
                    f"RecoveryStand/final_nonwheel_contact_rate_by_tilt_bin/{bin_name}": (
                        _masked_mean(nonwheel_contact.float(), timeout_in_bin)
                    ),
                    f"RecoveryStand/dual_wheel_contact_rate_by_tilt_bin/{bin_name}": (
                        _masked_mean(dual_wheel_contact.float(), in_bin)
                    ),
                }
            )

    return completed


def leg_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 1.0,
    command_name: str | None = None,
    jump_force_threshold: float | None = None,
    jump_landing_force_threshold: float | None = None,
    jump_grace_steps: int = 0,
    recovery_grace_steps: int = 0,
    recovery_terminate: bool = True,
    terminate: bool = True,
) -> torch.Tensor:
    """腿部 link 接触地面即时终止（膝盖着地 = 非法运动模式）。

    同时将精细拆分指标写入 extras['log']，用于诊断 leg_contact 来源：
    - diag_leg_contact_jump：jump_flag=1 的 env 中腿部触地比例
    - diag_leg_contact_walk：jump_flag=0 的 env 中腿部触地比例
    - diag_leg_contact_rsi：RSI 注入期（前 debounce+2 步）触地比例
    - diag_leg_contact_landing：landing stage（stage==2）触地比例
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    force_mag = finite_contact_force_norm(data.force)
    max_force = force_mag.max(dim=1).values
    has_contact = max_force > force_threshold
    terminate_threshold = torch.full_like(max_force, force_threshold)
    terminate_contact = has_contact
    recovery_mask = _recovery_reset_mask(env)
    if not recovery_terminate:
        terminate_contact = terminate_contact & ~recovery_mask
    if recovery_grace_steps > 0:
        in_recovery_grace = recovery_mask & (env.episode_length_buf <= int(recovery_grace_steps))
        terminate_contact = terminate_contact & ~in_recovery_grace

    # 精细拆分诊断：写入 extras["_leg_contact_diag"]（不是 log，log 在 reset 时会被清空）
    # command_manager._update_metrics 在 reset 之后调用，从此处读取并搬入 extras["log"]
    if hasattr(env, "extras"):
        try:
            diag_command_name = command_name or "velocity_height"
            cmd = env.command_manager.get_command(diag_command_name)
            jump_flag = cmd[:, 5] > 0.5

            n_jump = jump_flag.sum().item()
            n_walk = (~jump_flag).sum().item()
            leg_contact_jump = has_contact[jump_flag].float().mean().item() if n_jump > 0 else 0.0
            leg_contact_walk = has_contact[~jump_flag].float().mean().item() if n_walk > 0 else 0.0

            # RSI 期门控
            from se3_train.mdp.jump_commands import JumpCommandTerm

            term = env.command_manager.get_term(diag_command_name)
            rsi_window = 5
            in_rsi = env.episode_length_buf <= rsi_window
            rsi_jump = jump_flag & in_rsi
            n_rsi = rsi_jump.sum().item()
            leg_contact_rsi = has_contact[rsi_jump].float().mean().item() if n_rsi > 0 else 0.0

            # landing stage 门控
            if isinstance(term, JumpCommandTerm):
                in_landing = term.jump_stage == 2
                landing_jump = jump_flag & in_landing
                n_landing = landing_jump.sum().item()
                leg_contact_landing = (
                    has_contact[landing_jump].float().mean().item() if n_landing > 0 else 0.0
                )

                if command_name is not None:
                    jump_threshold = (
                        force_threshold if jump_force_threshold is None else jump_force_threshold
                    )
                    landing_threshold = (
                        jump_threshold
                        if jump_landing_force_threshold is None
                        else jump_landing_force_threshold
                    )
                    in_grace = env.episode_length_buf <= jump_grace_steps
                    jump_threshold_tensor = torch.full_like(max_force, jump_threshold)
                    landing_threshold_tensor = torch.full_like(max_force, landing_threshold)
                    terminate_threshold = torch.where(
                        jump_flag, jump_threshold_tensor, terminate_threshold
                    )
                    terminate_threshold = torch.where(
                        landing_jump | (jump_flag & in_grace),
                        landing_threshold_tensor,
                        terminate_threshold,
                    )
                    terminate_contact = max_force > terminate_threshold
                    leg_contact_termination = (
                        terminate_contact[jump_flag].float().mean().item() if n_jump > 0 else 0.0
                    )
                else:
                    leg_contact_termination = leg_contact_jump
            else:
                leg_contact_landing = 0.0
                leg_contact_termination = leg_contact_jump

            # 存入 extras["_leg_contact_diag"]，reset 时不会被清空
            # _update_metrics 在 reset 之后读取并搬入 extras["log"]
            env.extras["_leg_contact_diag"] = {
                "Jump/diag_leg_contact_jump": leg_contact_jump,
                "Jump/diag_leg_contact_walk": leg_contact_walk,
                "Jump/diag_leg_contact_rsi": leg_contact_rsi,
                "Jump/diag_leg_contact_landing": leg_contact_landing,
                "Jump/diag_leg_contact_termination": leg_contact_termination,
            }
        except Exception:
            pass

    if terminate:
        return terminate_contact
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
