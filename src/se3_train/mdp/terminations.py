"""SE3 轮腿机器人的终止函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_shared import JointGroup
from se3_train.mdp import recovery_state
from se3_train.mdp.contact_utils import finite_contact_force_norm

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.sensor import ContactSensor


def _recovery_reset_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
    """返回当前仍处于 recovery active 模式的 env。"""
    return recovery_state.recovery_active_mask(env)


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


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
    max_leg_pos_error: float = 3.0,
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

    leg_pos = joint_pos[:, JointGroup.LEGS]
    leg_default = robot.data.default_joint_pos[:, JointGroup.LEGS]
    leg_vel = joint_vel[:, JointGroup.LEGS]
    base_height = root_pos[:, 2] - env.scene.env_origins[:, 2]

    leg_pos_bad = torch.any(torch.abs(leg_pos - leg_default) > float(max_leg_pos_error), dim=1)
    leg_vel_bad = torch.any(torch.abs(leg_vel) > float(max_leg_vel), dim=1)
    root_lin_bad = torch.linalg.norm(root_lin_vel, dim=1) > float(max_root_lin_vel)
    root_ang_bad = torch.linalg.norm(root_ang_vel, dim=1) > float(max_root_ang_vel)
    height_bad = (base_height < float(min_base_height)) | (base_height > float(max_base_height))

    terminated = ~finite | leg_pos_bad | leg_vel_bad | root_lin_bad | root_ang_bad | height_bad
    if hasattr(env, "extras"):
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
