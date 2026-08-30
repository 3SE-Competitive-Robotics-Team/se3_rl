"""本任务使用的 actor/critic 观测函数。"""

from __future__ import annotations

from se3_train.mdp.observations import (
    base_ang_vel_obs,
    base_height_obs,
    base_lin_vel_obs,
    commands_obs,
    contact_force_norm_obs,
    dr_model_params_obs,
    joint_acc_obs,
    jump_commands_obs,
    knee_gas_spring_force_obs,
    last_actions_obs,
    leg_joint_pos_obs,
    leg_joint_vel_obs,
    motor_torque_obs,
    projected_gravity_obs,
    wheel_contact_force_obs,
    wheel_pos_obs,
    wheel_vel_obs,
)

__all__ = [
    "base_ang_vel_obs",
    "base_height_obs",
    "base_lin_vel_obs",
    "commands_obs",
    "contact_force_norm_obs",
    "dr_model_params_obs",
    "joint_acc_obs",
    "jump_commands_obs",
    "knee_gas_spring_force_obs",
    "last_actions_obs",
    "leg_joint_pos_obs",
    "leg_joint_vel_obs",
    "motor_torque_obs",
    "projected_gravity_obs",
    "wheel_contact_force_obs",
    "wheel_pos_obs",
    "wheel_vel_obs",
]
