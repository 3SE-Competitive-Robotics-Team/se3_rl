"""SE3 轮腿机器人的动作项。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from se3_shared import (
    DM8009P,
    ActionDelayConfig,
    JointGroup,
    output_to_policy_pos_torch,
    output_to_policy_vel_torch,
    policy_to_output_torque_torch,
)
from se3_shared import RobotConfig as SharedRobotConfig
from se3_train.mdp.joint_indices import is_closedchain_model, is_fourbar_surrogate_model

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_SHARED_ROBOT = SharedRobotConfig()
_DEFAULT_DELAY = _SHARED_ROBOT.action_delay


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

        try:
            leg_ids, leg_names = self._entity.find_joints_by_actuator_names(cfg.leg_actuator_names)
        except ValueError:
            leg_ids, leg_names = self._entity.find_joints_by_actuator_names(
                JointGroup.OPENCHAIN_LEG_NAMES
            )
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

        self._leg_joint_ids = torch.tensor(leg_ids, device=self.device, dtype=torch.long)
        self._wheel_joint_ids = torch.tensor(wheel_ids, device=self.device, dtype=torch.long)
        self._leg_action_scales = torch.tensor(cfg.leg_scales, device=self.device)
        self._closedchain = is_closedchain_model(self._entity)
        self._fourbar_surrogate = is_fourbar_surrogate_model(self._entity)
        self._active_rod_angle_limits = torch.tensor(
            _SHARED_ROBOT.active_rod_angle_limits,
            device=self.device,
        )
        self._active_rod_angle_coeffs = torch.tensor(
            _SHARED_ROBOT.active_rod_angle_coeffs,
            device=self.device,
        )
        self._action_dim = 6

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._delayed_actions = torch.zeros_like(self._raw_actions)
        self._env_indices = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._leg_kp = torch.full(
            (self.num_envs, 4),
            float(_SHARED_ROBOT.leg_kp),
            device=self.device,
        )
        self._leg_kd = torch.full(
            (self.num_envs, 4),
            float(_SHARED_ROBOT.leg_kd),
            device=self.device,
        )

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
        self._resample_delay(self._env_indices)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def delayed_action(self) -> torch.Tensor:
        return self._delayed_actions

    @property
    def delay_steps(self) -> torch.Tensor:
        return self._delay_steps

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions.to(self.device)

    def apply_actions(self) -> None:
        if self._max_delay_steps > 0:
            self._action_fifo[1:] = self._action_fifo[:-1].clone()
        self._action_fifo[0] = self._raw_actions
        self._delayed_actions = self._action_fifo[self._delay_steps, self._env_indices]

        if self._fourbar_surrogate:
            policy_target = self._delayed_actions[:, :4] * self._leg_action_scales
            policy_target = policy_target + self._current_policy_leg_defaults()
            policy_target = self._clamp_active_rod_angles(policy_target)
            output_pos = self._entity.data.joint_pos[:, self._leg_joint_ids]
            output_vel = self._entity.data.joint_vel[:, self._leg_joint_ids]
            policy_pos = output_to_policy_pos_torch(output_pos)
            policy_vel = output_to_policy_vel_torch(output_pos, output_vel)
            policy_torque = self._leg_kp * (policy_target - policy_pos)
            policy_torque -= self._leg_kd * policy_vel
            policy_torque = self._clip_active_motor_torque(policy_torque, policy_vel)
            leg_torque = policy_to_output_torque_torch(policy_pos, policy_torque)
            self._entity.set_joint_effort_target(leg_torque, joint_ids=self._leg_joint_ids)
        else:
            leg_target = (
                self._delayed_actions[:, :4] * self._leg_action_scales
                + self._entity.data.default_joint_pos[:, self._leg_joint_ids]
            )
            leg_target = self._clamp_active_rod_angles(leg_target)
            leg_target = leg_target - self._entity.data.encoder_bias[:, self._leg_joint_ids]
            self._entity.set_joint_position_target(leg_target, joint_ids=self._leg_joint_ids)
        wheel_target = (
            self._delayed_actions[:, 4:6] * float(self.cfg.wheel_scale)
            + self._entity.data.default_joint_vel[:, self._wheel_joint_ids]
        )

        self._entity.set_joint_velocity_target(wheel_target, joint_ids=self._wheel_joint_ids)

    def _current_policy_leg_defaults(self) -> torch.Tensor:
        """返回当前 env 随机化后的腿部默认位姿，坐标系与 policy 动作一致。"""
        output_default = self._entity.data.default_joint_pos[:, self._leg_joint_ids]
        return output_to_policy_pos_torch(output_default)

    def set_leg_pd_gain_scale(
        self,
        env_ids: torch.Tensor | slice | None,
        kp_scale: torch.Tensor,
        kd_scale: torch.Tensor,
    ) -> None:
        """同步 startup 域随机化采样到 fourbar 手写腿部 PD 控制器。"""
        if not self._fourbar_surrogate:
            return
        resolved_env_ids = self._resolve_env_ids(env_ids)
        if resolved_env_ids.numel() == 0:
            return
        kp_scale = kp_scale.to(device=self.device).reshape(-1, 1)
        kd_scale = kd_scale.to(device=self.device).reshape(-1, 1)
        if kp_scale.shape[0] != resolved_env_ids.numel():
            raise ValueError(
                "kp_scale env 数量与 env_ids 不一致: "
                f"{kp_scale.shape[0]} != {resolved_env_ids.numel()}"
            )
        if kd_scale.shape[0] != resolved_env_ids.numel():
            raise ValueError(
                "kd_scale env 数量与 env_ids 不一致: "
                f"{kd_scale.shape[0]} != {resolved_env_ids.numel()}"
            )
        self._leg_kp[resolved_env_ids] = float(_SHARED_ROBOT.leg_kp) * kp_scale
        self._leg_kd[resolved_env_ids] = float(_SHARED_ROBOT.leg_kd) * kd_scale

    def _clamp_active_rod_angles(self, leg_target: torch.Tensor) -> torch.Tensor:
        """闭链下按同侧两主动杆夹角裁剪后杆目标。"""
        if not (self._closedchain or self._fourbar_surrogate):
            return leg_target
        target = leg_target.clone()
        lower, upper = self._active_rod_angle_limits
        for side_idx, (front_idx, back_idx) in enumerate(((0, 1), (2, 3))):
            front_coef, back_coef = self._active_rod_angle_coeffs[side_idx]
            angle = torch.clamp(
                front_coef * target[:, front_idx] + back_coef * target[:, back_idx],
                lower,
                upper,
            )
            target[:, back_idx] = (angle - front_coef * target[:, front_idx]) / back_coef
        return target

    def _clip_active_motor_torque(
        self, torque: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """按虚拟主动杆速度应用 DM8009P T-N 包络限幅。"""
        saturation = float(DM8009P.stall_torque)
        velocity_limit = float(DM8009P.no_load_speed)
        effort_limit = float(DM8009P.rated_torque)
        vel_at_effort_limit = velocity_limit * (1.0 + effort_limit / saturation)
        clipped_velocity = velocity.clamp(-vel_at_effort_limit, vel_at_effort_limit)
        top = saturation * (1.0 - clipped_velocity / velocity_limit)
        bottom = saturation * (-1.0 - clipped_velocity / velocity_limit)
        max_effort = torch.clamp(top, max=effort_limit)
        min_effort = torch.clamp(bottom, min=-effort_limit)
        return torque.clamp(min_effort, max_effort)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        resolved_env_ids = self._resolve_env_ids(env_ids)
        self._raw_actions[resolved_env_ids] = 0.0
        self._delayed_actions[resolved_env_ids] = 0.0
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
