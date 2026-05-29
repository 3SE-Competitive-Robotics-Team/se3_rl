"""SE3 轮腿机器人的动作项。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from se3_shared import ActionDelayConfig, JointGroup
from se3_shared import RobotConfig as SharedRobotConfig

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_SHARED_ROBOT = SharedRobotConfig()
_DEFAULT_DELAY = _SHARED_ROBOT.action_delay


@dataclass(kw_only=True)
class SerialLegDelayedActionCfg(ActionTermCfg):
    """6D 策略动作项，支持训练端 action 延迟。"""

    leg_actuator_names: tuple[str, ...] = ("lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint")
    wheel_actuator_names: tuple[str, ...] = ("l_wheel_Joint", "r_wheel_Joint")
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

        self._leg_joint_ids = torch.tensor(leg_ids, device=self.device, dtype=torch.long)
        self._wheel_joint_ids = torch.tensor(wheel_ids, device=self.device, dtype=torch.long)
        self._leg_action_scales = torch.tensor(cfg.leg_scales, device=self.device)
        self._action_dim = 6

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._delayed_actions = torch.zeros_like(self._raw_actions)
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

        leg_target = (
            self._delayed_actions[:, :4] * self._leg_action_scales
            + self._entity.data.default_joint_pos[:, self._leg_joint_ids]
        )
        leg_target = leg_target - self._entity.data.encoder_bias[:, self._leg_joint_ids]
        wheel_target = (
            self._delayed_actions[:, 4:6] * float(self.cfg.wheel_scale)
            + self._entity.data.default_joint_vel[:, self._wheel_joint_ids]
        )

        self._entity.set_joint_position_target(leg_target, joint_ids=self._leg_joint_ids)
        self._entity.set_joint_velocity_target(wheel_target, joint_ids=self._wheel_joint_ids)

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
