"""SE3 轮腿机器人的动作项。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from se3_shared import (
    ActionDelayConfig,
    JointGroup,
    knee_gas_spring_compensation_torque_torch,
    output_leg_length_limits_torch,
    output_leg_wheel_xz_torch,
    output_to_policy_pos_torch,
    policy_leg_position_error_torch,
    policy_to_output_pos_torch,
    wheel_xz_to_output_pos_torch,
)
from se3_shared import RobotConfig as SharedRobotConfig
from se3_train.mdp.height_default_cache import get_policy_default_from_height_cache
from se3_train.mdp.joint_indices import (
    leg_actuator_ids,
)
from se3_train.mdp.recovery_teacher import RecoveryTeacherCfg, RecoveryTeacherController
from se3_train.mdp.recovery_torque_assist import (
    RECOVERY_TORQUE_ASSIST_STATE_DIM,
    RecoveryTorqueAssistCfg,
    RecoveryTorqueAssistController,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_SHARED_ROBOT = SharedRobotConfig()
_DEFAULT_DELAY = _SHARED_ROBOT.action_delay
_CTBC_SOURCE_OUTPUT_LEG_SCALE = 0.25
_CTBC_SOURCE_TO_TARGET_OUTPUT_SIGN = (-1.0, -1.0, 1.0, 1.0)


@dataclass(kw_only=True)
class SerialLegDelayedActionCfg(ActionTermCfg):
    """6D 策略动作项，支持训练端 action 延迟。"""

    leg_actuator_names: tuple[str, ...] = JointGroup.POLICY_LEG_NAMES
    wheel_actuator_names: tuple[str, ...] = JointGroup.WHEEL_NAMES
    leg_scales: tuple[float, ...] = tuple(
        _SHARED_ROBOT.action_scale[i] for i in JointGroup.LEG_ACTUATORS
    )
    wheel_scale: float = _SHARED_ROBOT.action_scale[JointGroup.WHEEL_ACTUATORS[0]]
    action_delay_enabled: bool = _DEFAULT_DELAY.enabled
    action_delay_s: float = _DEFAULT_DELAY.delay_s
    action_delay_randomize: bool = _DEFAULT_DELAY.randomize
    action_delay_min_s: float = _DEFAULT_DELAY.min_delay_s
    action_delay_max_s: float = _DEFAULT_DELAY.max_delay_s
    action_clip: float | None = _SHARED_ROBOT.action_clip
    height_conditioned_action_default: bool = False
    action_default_command_name: str = "velocity_height"
    active_rod_lower_target_overdrive: float = _SHARED_ROBOT.active_rod_lower_target_overdrive
    knee_gas_spring_force: float = _SHARED_ROBOT.knee_gas_spring_force
    knee_gas_spring_compensation_enabled: bool = _SHARED_ROBOT.knee_gas_spring_compensation_enabled
    recovery_teacher: RecoveryTeacherCfg | None = None
    recovery_torque_assist: RecoveryTorqueAssistCfg | None = None

    def build(self, env: ManagerBasedRlEnv) -> SerialLegDelayedAction:
        return SerialLegDelayedAction(self, env)

    def delay_config(self) -> ActionDelayConfig:
        """生成共享延迟配置对象。"""
        return ActionDelayConfig(
            enabled=self.action_delay_enabled,
            delay_s=self.action_delay_s,
            randomize=self.action_delay_randomize,
            min_delay_s=self.action_delay_min_s,
            max_delay_s=self.action_delay_max_s,
        )


class SerialLegDelayedAction(ActionTerm):
    """在 raw action 层做 FIFO 延迟，然后写入腿位置目标和轮速目标。"""

    cfg: SerialLegDelayedActionCfg

    def __init__(self, cfg: SerialLegDelayedActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        self._env = env
        leg_ids, leg_names = self._entity.find_joints_by_actuator_names(cfg.leg_actuator_names)
        wheel_ids, wheel_names = self._entity.find_joints_by_actuator_names(
            cfg.wheel_actuator_names
        )
        if len(leg_ids) != 4:
            raise ValueError(f"SerialLegDelayedAction expects 4 leg actuators, got {leg_names}")
        if len(wheel_ids) != 2:
            raise ValueError(f"SerialLegDelayedAction expects 2 wheel actuators, got {wheel_names}")
        if len(cfg.leg_scales) != 4:
            raise ValueError(
                f"SerialLegDelayedAction expects 4 leg action scales, got {cfg.leg_scales}"
            )
        if cfg.action_clip is not None and cfg.action_clip <= 0.0:
            raise ValueError(f"action_clip must be positive or None, got {cfg.action_clip}")
        if cfg.knee_gas_spring_force < 0.0:
            raise ValueError(f"knee_gas_spring_force 必须非负，实际为 {cfg.knee_gas_spring_force}")
        self._leg_joint_ids = torch.tensor(leg_ids, device=self.device, dtype=torch.long)
        self._wheel_joint_ids = torch.tensor(wheel_ids, device=self.device, dtype=torch.long)
        self._leg_action_scales = torch.tensor(cfg.leg_scales, device=self.device)
        self._leg_actuator_ids = torch.tensor(
            leg_actuator_ids(self._entity),
            device=self.device,
            dtype=torch.long,
        )
        self._active_rod_angle_limits = torch.tensor(
            _SHARED_ROBOT.active_rod_angle_limits,
            device=self.device,
        )
        self._active_rod_angle_mid = torch.mean(self._active_rod_angle_limits)
        self._active_rod_angle_coeffs = torch.tensor(
            _SHARED_ROBOT.active_rod_angle_coeffs,
            device=self.device,
        )
        self._action_dim = 6

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._unclipped_actions = torch.zeros_like(self._raw_actions)
        self._policy_actions = torch.zeros_like(self._raw_actions)
        self._delayed_actions = torch.zeros_like(self._raw_actions)
        self._recovery_teacher_action = torch.zeros_like(self._raw_actions)
        self._recovery_teacher_active = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )
        self._recovery_teacher_alpha = 0.0
        self._ctbc_output_bias = torch.zeros_like(self._raw_actions)
        self._ctbc_action_delta = torch.zeros_like(self._raw_actions)
        self._ctbc_wheel_delta_xz = torch.zeros(self.num_envs, 2, 2, device=self.device)
        self._policy_leg_torque = torch.zeros(self.num_envs, 4, device=self.device)
        self._policy_leg_vel = torch.zeros_like(self._policy_leg_torque)
        self._policy_leg_target = torch.zeros_like(self._policy_leg_torque)
        self._knee_gas_spring_compensation_torque = torch.zeros_like(self._policy_leg_torque)
        self._active_rod_angle_target = torch.zeros(self.num_envs, 2, device=self.device)
        self._active_rod_angle_target_clamped = torch.zeros(
            self.num_envs, 2, device=self.device, dtype=torch.bool
        )
        self._env_indices = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._delay_cfg = cfg.delay_config()
        self._sim_dt = float(env.physics_dt)
        self._min_delay_steps, self._max_delay_steps = self._delay_cfg.step_bounds(self._sim_dt)
        self._delay_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._action_fifo = torch.zeros(
            self._max_delay_steps + 1,
            self.num_envs,
            self.action_dim,
            device=self.device,
        )
        self._recovery_teacher = None
        if cfg.recovery_teacher is not None and cfg.recovery_teacher.enabled:
            self._recovery_teacher = RecoveryTeacherController(
                env,
                cfg.recovery_teacher,
                self._leg_action_scales,
            )
        self._recovery_torque_assist = None
        if cfg.recovery_torque_assist is not None and cfg.recovery_torque_assist.enabled:
            if self._recovery_teacher is not None:
                raise ValueError("Recovery 动作 teacher 与外部扭矩引导不能同时启用")
            self._recovery_torque_assist = RecoveryTorqueAssistController(
                env,
                cfg.recovery_torque_assist,
                self._entity,
            )
        self._resample_delay(self._env_indices)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def unclipped_action(self) -> torch.Tensor:
        return self._unclipped_actions

    @property
    def policy_action(self) -> torch.Tensor:
        return self._policy_actions

    @property
    def delayed_action(self) -> torch.Tensor:
        return self._delayed_actions

    @property
    def recovery_teacher_action(self) -> torch.Tensor:
        return self._recovery_teacher_action

    @property
    def recovery_teacher_active(self) -> torch.Tensor:
        return self._recovery_teacher_active

    @property
    def recovery_teacher_alpha(self) -> float:
        return self._recovery_teacher_alpha

    @property
    def recovery_torque_assist_active(self) -> torch.Tensor:
        """返回外部扭矩引导 active mask，未启用时恒为 false。"""

        if self._recovery_torque_assist is None:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        return self._recovery_torque_assist.active

    @property
    def recovery_torque_assist_nm(self) -> torch.Tensor:
        """返回每个环境当前写入的外部扭矩模长。"""

        if self._recovery_torque_assist is None:
            return torch.zeros(self.num_envs, device=self.device)
        return self._recovery_torque_assist.applied_torque_nm

    @property
    def recovery_torque_assist_state(self) -> torch.Tensor:
        """返回外部扭矩引导的 critic 特权状态，未启用时恒为零。"""

        if self._recovery_torque_assist is None:
            return torch.zeros(
                self.num_envs,
                RECOVERY_TORQUE_ASSIST_STATE_DIM,
                device=self.device,
            )
        return self._recovery_torque_assist.critic_state

    @property
    def ctbc_output_bias(self) -> torch.Tensor:
        return self._ctbc_output_bias

    @property
    def ctbc_action_delta(self) -> torch.Tensor:
        return self._ctbc_action_delta

    @property
    def ctbc_wheel_delta_xz(self) -> torch.Tensor:
        return self._ctbc_wheel_delta_xz

    @property
    def actual_wheel_xz(self) -> torch.Tensor:
        output_pos = self._entity.data.joint_pos[:, self._leg_joint_ids]
        return output_leg_wheel_xz_torch(output_pos)

    @property
    def target_wheel_xz(self) -> torch.Tensor:
        output_target = policy_to_output_pos_torch(self._policy_leg_target)
        return output_leg_wheel_xz_torch(output_target)

    @property
    def policy_leg_torque(self) -> torch.Tensor:
        return self._policy_leg_torque

    @property
    def policy_leg_vel(self) -> torch.Tensor:
        return self._policy_leg_vel

    @property
    def policy_leg_target(self) -> torch.Tensor:
        return self._policy_leg_target

    @property
    def knee_gas_spring_compensation_torque(self) -> torch.Tensor:
        return self._knee_gas_spring_compensation_torque

    @property
    def active_rod_angle_target(self) -> torch.Tensor:
        return self._active_rod_angle_target

    @property
    def active_rod_angle_target_clamped(self) -> torch.Tensor:
        return self._active_rod_angle_target_clamped

    @property
    def delay_steps(self) -> torch.Tensor:
        return self._delay_steps

    def process_actions(self, actions: torch.Tensor) -> None:
        incoming_actions = actions.to(self.device)
        self._unclipped_actions[:] = incoming_actions
        if self.cfg.action_clip is None:
            self._raw_actions[:] = incoming_actions
        else:
            clip = float(self.cfg.action_clip)
            self._raw_actions[:] = torch.clamp(incoming_actions, -clip, clip)
        self._policy_actions[:] = self._raw_actions
        self._recovery_teacher_action.zero_()
        self._recovery_teacher_active.zero_()
        self._recovery_teacher_alpha = 0.0
        if self._recovery_teacher is not None:
            transformed, teacher_action, teacher_active, alpha = self._recovery_teacher.transform(
                self._policy_actions
            )
            if self.cfg.action_clip is None:
                self._raw_actions[:] = transformed
            else:
                clip = float(self.cfg.action_clip)
                self._raw_actions[:] = torch.clamp(transformed, -clip, clip)
            self._recovery_teacher_action[:] = teacher_action
            self._recovery_teacher_active[:] = teacher_active
            self._recovery_teacher_alpha = float(alpha)
        if self._recovery_torque_assist is not None:
            self._recovery_torque_assist.log_diagnostics()
        self._ctbc_output_bias.zero_()
        self._ctbc_action_delta.zero_()
        self._ctbc_wheel_delta_xz.zero_()
        state = getattr(self._env, "stair_climb_state", None)
        if state is not None and state.kff != 0.0:
            recovery_active = getattr(self._env, "_recovery_reset_mask", None)
            bias_fn = getattr(state, "ff_bias", None)
            if callable(bias_fn):
                self._ctbc_output_bias[:] = state.ff_bias()
                action_delta = self._ctbc_output_bias_to_action_delta(
                    self._raw_actions[:, :4],
                    self._ctbc_output_bias[:, :4],
                )
                wheel_delta_fn = getattr(state, "ff_wheel_delta_xz", None)
                if callable(wheel_delta_fn):
                    retreat_delta_xz = wheel_delta_fn().to(
                        device=self.device,
                        dtype=self._ctbc_wheel_delta_xz.dtype,
                    )
                    self._ctbc_wheel_delta_xz += retreat_delta_xz
                    action_delta = self._ctbc_wheel_delta_to_action_delta(
                        self._raw_actions[:, :4],
                        self._ctbc_wheel_delta_xz,
                    )
            else:
                wheel_delta_fn = getattr(state, "ff_wheel_delta_xz", None)
                action_delta = None
                if callable(wheel_delta_fn):
                    self._ctbc_wheel_delta_xz[:] = wheel_delta_fn()
                    action_delta = self._ctbc_wheel_delta_to_action_delta(
                        self._raw_actions[:, :4],
                        self._ctbc_wheel_delta_xz,
                    )
            if action_delta is None:
                return
            if (
                isinstance(recovery_active, torch.Tensor)
                and recovery_active.shape[0] == self.num_envs
            ):
                active_recovery = recovery_active.to(device=self.device, dtype=torch.bool)
                action_delta[active_recovery] = 0.0
                self._ctbc_wheel_delta_xz[active_recovery] = 0.0
            self._apply_ctbc_action_delta(action_delta)
            wheel_action_delta_fn = getattr(state, "ff_wheel_action_delta", None)
            if callable(wheel_action_delta_fn):
                wheel_action_delta = wheel_action_delta_fn()
                if (
                    isinstance(recovery_active, torch.Tensor)
                    and recovery_active.shape[0] == self.num_envs
                ):
                    wheel_action_delta[active_recovery] = 0.0
                self._apply_ctbc_wheel_action_delta(wheel_action_delta)

    def apply_actions(self) -> None:
        if self._max_delay_steps > 0:
            self._action_fifo[1:] = self._action_fifo[:-1].clone()
        self._action_fifo[0] = self._raw_actions
        self._delayed_actions = self._action_fifo[self._delay_steps, self._env_indices]

        leg_target = self._leg_action_to_policy_target(
            self._delayed_actions[:, :4],
            self._current_leg_action_defaults(),
        )
        self._policy_leg_target[:] = leg_target
        current_leg_pos = self._entity.data.joint_pos[:, self._leg_joint_ids]
        servo_leg_target = current_leg_pos + policy_leg_position_error_torch(
            leg_target,
            current_leg_pos,
            self._active_rod_angle_coeffs,
        )
        servo_leg_target = servo_leg_target - self._entity.data.encoder_bias[:, self._leg_joint_ids]
        self._entity.set_joint_position_target(servo_leg_target, joint_ids=self._leg_joint_ids)
        if self.cfg.knee_gas_spring_compensation_enabled:
            measured_leg_pos = (
                current_leg_pos + self._entity.data.encoder_bias[:, self._leg_joint_ids]
            )
            self._knee_gas_spring_compensation_torque[:] = (
                knee_gas_spring_compensation_torque_torch(
                    measured_leg_pos,
                    self.cfg.knee_gas_spring_force,
                )
            )
        else:
            self._knee_gas_spring_compensation_torque.zero_()
        self._entity.set_joint_effort_target(
            self._knee_gas_spring_compensation_torque,
            joint_ids=self._leg_joint_ids,
        )
        self._policy_leg_torque[:] = self._entity.data.actuator_force[:, self._leg_actuator_ids]
        self._policy_leg_vel[:] = self._entity.data.joint_vel[:, self._leg_joint_ids]
        wheel_target = (
            self._delayed_actions[:, 4:6] * float(self.cfg.wheel_scale)
            + self._entity.data.default_joint_vel[:, self._wheel_joint_ids]
        )

        self._entity.set_joint_velocity_target(wheel_target, joint_ids=self._wheel_joint_ids)
        if self._recovery_torque_assist is not None:
            self._recovery_torque_assist.apply()

    def _current_leg_action_defaults(self) -> torch.Tensor:
        """返回当前 leg action 零点姿态。"""
        if self.cfg.height_conditioned_action_default:
            return get_policy_default_from_height_cache(
                self._env,
                self.cfg.action_default_command_name,
                device=self.device,
                dtype=self._leg_action_scales.dtype,
            )
        return self._entity.data.default_joint_pos[:, self._leg_joint_ids]

    def _leg_action_to_policy_target(
        self,
        leg_action: torch.Tensor,
        policy_default: torch.Tensor,
        *,
        update_active_targets: bool = True,
    ) -> torch.Tensor:
        """把腿部 action 解释为前杆角和主动杆夹角目标。"""
        lower, upper = self._active_rod_angle_limits
        target_lower = float(lower) - float(self.cfg.active_rod_lower_target_overdrive)
        target = torch.empty_like(policy_default)
        active_targets: list[torch.Tensor] = []
        active_clamped: list[torch.Tensor] = []
        for side_idx, (front_idx, back_idx) in enumerate(((0, 1), (2, 3))):
            front_coef, back_coef = self._active_rod_angle_coeffs[side_idx]
            front_target = (
                policy_default[:, front_idx]
                + leg_action[:, front_idx] * (self._leg_action_scales[front_idx])
            )
            if self.cfg.height_conditioned_action_default:
                active_default = (
                    front_coef * policy_default[:, front_idx]
                    + back_coef * policy_default[:, back_idx]
                )
            else:
                active_default = self._active_rod_angle_mid
            active_raw = (
                active_default + leg_action[:, back_idx] * self._leg_action_scales[back_idx]
            )
            active_target = torch.clamp(active_raw, target_lower, upper)
            target[:, front_idx] = front_target
            target[:, back_idx] = (active_target - front_coef * front_target) / back_coef
            active_targets.append(active_target)
            active_clamped.append(active_target != active_raw)
        if update_active_targets:
            self._active_rod_angle_target[:] = torch.stack(active_targets, dim=1)
            self._active_rod_angle_target_clamped[:] = torch.stack(active_clamped, dim=1)
        return target

    def _ctbc_output_bias_to_action_delta(
        self,
        leg_action: torch.Tensor,
        output_action_bias: torch.Tensor,
    ) -> torch.Tensor:
        """把旧输出关节 CTBC bias 等效换算成当前 action 语义。

        源 stair 任务的 CTBC 状态机输出的是输出关节语义的 4 维腿部 action bias，
        乘以旧 leg_scale=0.25 后表示 [lf0, lf1, rf0, rf1] 输出关节角增量。
        源模型左右腿关节轴均为 -Y，目标模型左腿轴改为 +Y，因此左腿两维
        需要取反，右腿保持不变。
        固定关节角增量只在默认姿态表现为轮子向后上方缩回；叠加到策略的任意
        当前姿态后可能反而向前伸。这里先在当前默认姿态上求出旧 CTBC 对应的
        轮心后上位移，再把同一物理 XZ 位移施加到策略当前姿态，最后反解回
        [front, active_angle] action。
        """
        output_action_bias = output_action_bias.to(self.device)
        source_to_target_sign = output_action_bias.new_tensor(_CTBC_SOURCE_TO_TARGET_OUTPUT_SIGN)
        output_delta = output_action_bias * source_to_target_sign * _CTBC_SOURCE_OUTPUT_LEG_SCALE
        active_side = output_action_bias.reshape(-1, 2, 2).abs().amax(dim=-1) > 0.0
        active_env_ids = active_side.any(dim=1).nonzero().flatten()
        if active_env_ids.numel() == 0:
            return torch.zeros_like(leg_action)
        active_output_delta = output_delta[active_env_ids]

        policy_default = self._current_leg_action_defaults()[active_env_ids]
        default_output = policy_to_output_pos_torch(policy_default)
        nominal_requested_output = default_output + active_output_delta
        nominal_realizable_output = policy_to_output_pos_torch(
            output_to_policy_pos_torch(nominal_requested_output)
        )
        default_wheel_xz = output_leg_wheel_xz_torch(default_output)
        nominal_wheel_xz = output_leg_wheel_xz_torch(nominal_realizable_output)
        wheel_delta_xz = nominal_wheel_xz - default_wheel_xz
        wheel_delta_xz[..., 0] = torch.clamp(wheel_delta_xz[..., 0], max=0.0)
        wheel_delta_xz[..., 1] = torch.clamp(wheel_delta_xz[..., 1], min=0.0)
        self._ctbc_wheel_delta_xz[active_env_ids] = wheel_delta_xz
        return self._ctbc_wheel_delta_to_action_delta(leg_action, self._ctbc_wheel_delta_xz)

    def _ctbc_wheel_delta_to_action_delta(
        self,
        leg_action: torch.Tensor,
        wheel_delta_xz: torch.Tensor,
    ) -> torch.Tensor:
        """把轮端后上方 Cartesian 位移反解成当前 action 语义下的增量。"""
        active_side = wheel_delta_xz.abs().amax(dim=-1) > 0.0
        active_env_ids = active_side.any(dim=1).nonzero().flatten()
        if active_env_ids.numel() == 0:
            return torch.zeros_like(leg_action)

        active_leg_action = leg_action[active_env_ids]
        active_side = active_side[active_env_ids]
        active_wheel_delta_xz = wheel_delta_xz[active_env_ids]
        policy_default = self._current_leg_action_defaults()[active_env_ids]
        current_policy = self._leg_action_to_policy_target(
            active_leg_action,
            policy_default,
            update_active_targets=False,
        )
        current_output = policy_to_output_pos_torch(current_policy)
        current_wheel_xz = output_leg_wheel_xz_torch(current_output)
        desired_wheel_xz = self._reachable_ctbc_wheel_target(
            current_wheel_xz,
            active_wheel_delta_xz,
        )
        cartesian_desired_output = wheel_xz_to_output_pos_torch(desired_wheel_xz)
        active_joint = active_side.repeat_interleave(2, dim=1)
        desired_output = torch.where(active_joint, cartesian_desired_output, current_output)
        desired_policy = output_to_policy_pos_torch(desired_output)

        desired_action = torch.zeros_like(active_leg_action)
        for side_idx, (front_idx, back_idx) in enumerate(((0, 1), (2, 3))):
            front_coef, back_coef = self._active_rod_angle_coeffs[side_idx]
            if self.cfg.height_conditioned_action_default:
                active_default = (
                    front_coef * policy_default[:, front_idx]
                    + back_coef * policy_default[:, back_idx]
                )
            else:
                active_default = self._active_rod_angle_mid
            active_desired = (
                front_coef * desired_policy[:, front_idx] + back_coef * desired_policy[:, back_idx]
            )
            desired_action[:, front_idx] = (
                desired_policy[:, front_idx] - policy_default[:, front_idx]
            ) / self._leg_action_scales[front_idx]
            desired_action[:, back_idx] = (active_desired - active_default) / (
                self._leg_action_scales[back_idx]
            )
        action_delta = torch.zeros_like(leg_action)
        action_delta[active_env_ids] = desired_action - active_leg_action
        return torch.nan_to_num(action_delta, nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_ctbc_action_delta(self, action_delta: torch.Tensor) -> None:
        """注入 CTBC 后再次裁剪，保持最终 action 不越过策略动作契约。"""
        leg_action = self._raw_actions[:, :4]
        if self.cfg.action_clip is None:
            self._ctbc_action_delta[:, :4] = action_delta
            leg_action += action_delta
            return

        clip = float(self.cfg.action_clip)
        desired = leg_action + action_delta
        applied = torch.clamp(desired, -clip, clip)
        self._ctbc_action_delta[:, :4] = applied - leg_action
        leg_action[:] = applied

    def _apply_ctbc_wheel_action_delta(self, wheel_action_delta: torch.Tensor) -> None:
        """注入 CTBC 轮速前馈后再次裁剪。"""
        wheel_action_delta = wheel_action_delta.to(self.device)
        if wheel_action_delta.numel() == 0:
            return
        wheel_action = self._raw_actions[:, 4:6]
        if self.cfg.action_clip is None:
            self._ctbc_action_delta[:, 4:6] = wheel_action_delta
            wheel_action += wheel_action_delta
            return

        clip = float(self.cfg.action_clip)
        desired = wheel_action + wheel_action_delta
        applied = torch.clamp(desired, -clip, clip)
        self._ctbc_action_delta[:, 4:6] = applied - wheel_action
        wheel_action[:] = applied

    def _reachable_ctbc_wheel_target(
        self,
        current_wheel_xz: torch.Tensor,
        wheel_delta_xz: torch.Tensor,
    ) -> torch.Tensor:
        """沿后上位移缩短到四连杆可达范围，保持位移方向不翻转。"""
        min_length, max_length = output_leg_length_limits_torch(
            current_wheel_xz.device,
            current_wheel_xz.dtype,
        )

        def reachable(scale: torch.Tensor) -> torch.Tensor:
            candidate = current_wheel_xz + wheel_delta_xz * scale.unsqueeze(-1)
            length = torch.linalg.vector_norm(candidate, dim=-1)
            return (length >= min_length) & (length <= max_length)

        lower = torch.zeros_like(current_wheel_xz[..., 0])
        upper = torch.ones_like(lower)
        full_reachable = reachable(upper)
        for _ in range(8):
            middle = 0.5 * (lower + upper)
            middle_reachable = reachable(middle)
            lower = torch.where(middle_reachable, middle, lower)
            upper = torch.where(middle_reachable, upper, middle)
        scale = torch.where(full_reachable, torch.ones_like(lower), lower)
        return current_wheel_xz + wheel_delta_xz * scale.unsqueeze(-1)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        resolved_env_ids = self._resolve_env_ids(env_ids)
        self._raw_actions[resolved_env_ids] = 0.0
        self._unclipped_actions[resolved_env_ids] = 0.0
        self._policy_actions[resolved_env_ids] = 0.0
        self._delayed_actions[resolved_env_ids] = 0.0
        self._recovery_teacher_action[resolved_env_ids] = 0.0
        self._recovery_teacher_active[resolved_env_ids] = False
        self._recovery_teacher_alpha = 0.0
        if self._recovery_teacher is not None:
            self._recovery_teacher.reset(resolved_env_ids)
        if self._recovery_torque_assist is not None:
            self._recovery_torque_assist.reset(resolved_env_ids)
        self._ctbc_output_bias[resolved_env_ids] = 0.0
        self._ctbc_action_delta[resolved_env_ids] = 0.0
        self._ctbc_wheel_delta_xz[resolved_env_ids] = 0.0
        self._policy_leg_torque[resolved_env_ids] = 0.0
        self._policy_leg_vel[resolved_env_ids] = 0.0
        self._knee_gas_spring_compensation_torque[resolved_env_ids] = 0.0
        self._action_fifo[:, resolved_env_ids] = 0.0
        self._resample_delay(resolved_env_ids)

    def _resolve_env_ids(self, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
        if env_ids is None:
            return self._env_indices
        if isinstance(env_ids, slice):
            return self._env_indices[env_ids]
        return env_ids.to(device=self.device, dtype=torch.long)

    def _resample_delay(self, env_ids: torch.Tensor) -> None:
        num_envs = int(env_ids.numel())
        if num_envs == 0:
            return
        if self._min_delay_steps == self._max_delay_steps:
            self._delay_steps[env_ids] = int(self._min_delay_steps)
            return
        self._delay_steps[env_ids] = torch.randint(
            low=int(self._min_delay_steps),
            high=int(self._max_delay_steps) + 1,
            size=(num_envs,),
            device=self.device,
            dtype=torch.long,
        )


__all__ = [
    "JointPositionActionCfg",
    "JointVelocityActionCfg",
    "SerialLegDelayedActionCfg",
]
