"""复旦上台阶3奖励的闭链适配，逐项加权后裁到每秒 [-1, 1]。

来源：yly-true/fudan_rl_wheel_leg，8204e853，plane/logs/wheel_legged/上台阶3。
高度沿用本机测高；名义腿角由髋到轮心的实际几何计算；关节成本采用六电机坐标。
"""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from se3_train.mdp.joint_indices import leg_actuator_ids, policy_joint_ids, wheel_actuator_ids
from se3_train.mdp.terrain_height import frame_height_above_terrain

WEIGHTS = {
    "tracking_lin_vel": 1.5,
    "tracking_lin_vel_enhance": 1.5,
    "tracking_ang_vel": 1.0,
    "base_height": 1.0,
    "base_height_enhance": 1.0,
    "nominal_state": -1.0,
    "lin_vel_z": -0.1,
    "ang_vel_xy": -0.05,
    "orientation": -10.0,
    "dof_vel": -5e-5,
    "dof_acc": -2.5e-7,
    "torques": -0.0001,
    "action_rate": -0.05,
    "action_smooth": -0.05,
    "collision": -1.0,
    "dof_pos_limits": -1.0,
}


def _state(env):
    """每个环境实例保存自己的策略步历史，避免十六个奖励项重复推进。"""
    state = getattr(env, "_fudan_reward_state", None)
    if state is None:
        robot = env.scene["robot"]
        body_ids, _ = robot.find_bodies(
            ("lf0_Link", "rf0_Link", "l_wheel_Link", "r_wheel_Link"), preserve_order=True
        )
        state = {
            "joint_ids": list(policy_joint_ids(robot)),
            "actuator_ids": list(leg_actuator_ids(robot) + wheel_actuator_ids(robot)),
            "body_ids": body_ids,
            "prev_action": torch.zeros_like(env.action_manager.action),
            "prev_prev_action": torch.zeros_like(env.action_manager.action),
            "prev_velocity": torch.zeros_like(env.action_manager.action),
            "step": -1,
        }
        env._fudan_reward_state = state
    return state


def reset_history(env, env_ids) -> None:
    """回合重置时清空差分历史，保留其他环境的历史。"""
    state = _state(env)
    for key in ("prev_action", "prev_prev_action", "prev_velocity"):
        state[key][env_ids] = 0.0
    state["step"] = -1


def reward(env, term: str) -> torch.Tensor:
    """返回单项每秒奖励；RewardManager 只需再乘一次策略步长。"""
    state = _state(env)
    if state["step"] != env.common_step_counter:
        robot = env.scene["robot"]
        data = robot.data
        command = env.command_manager.get_command("velocity_height")
        action = env.action_manager.action
        q = data.joint_pos[:, state["joint_ids"]]
        velocity = data.joint_vel[:, state["joint_ids"]]
        acceleration = (velocity - state["prev_velocity"]) / env.step_dt
        ev2 = (command[:, 0] - data.root_link_lin_vel_b[:, 0]).square()
        ew2 = (command[:, 2] - data.root_link_ang_vel_b[:, 2]).square()
        eh2 = (frame_height_above_terrain(env, "base_height_sensor") - command[:, 4]).square()
        positions = data.body_link_pos_w[:, state["body_ids"], :]
        delta = positions[:, 2:] - positions[:, :2]
        quat = data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
        delta = quat_apply_inverse(quat.reshape(-1, 4), delta.reshape(-1, 3)).reshape(-1, 2, 3)
        # 相对向下轴的腿角；两腿角差不依赖参考实现的共同零位偏移。
        theta = torch.atan2(delta[:, :, 0], -delta[:, :, 2])
        limits = data.joint_pos_limits[:, state["joint_ids"][:4], :]
        center = limits.mean(dim=-1)
        half_range = (limits[..., 1] - limits[..., 0]) * 0.97 / 2
        excess = (center - half_range - q[:, :4]).clamp(min=0)
        excess += (q[:, :4] - center - half_range).clamp(min=0)
        second = action - 2 * state["prev_action"] + state["prev_prev_action"]
        raw = {
            "tracking_lin_vel": torch.exp(-ev2 / 0.25),
            "tracking_lin_vel_enhance": torch.exp(-ev2 / 2.5) - 1,
            "tracking_ang_vel": torch.exp(-ew2 / 0.25),
            "base_height": 1.5 * torch.exp(-1000 * eh2),
            "base_height_enhance": torch.exp(-eh2 / 0.01) - 1,
            "nominal_state": (theta[:, 0] - theta[:, 1]).square(),
            "lin_vel_z": data.root_link_lin_vel_b[:, 2].square(),
            "ang_vel_xy": data.root_link_ang_vel_b[:, :2].square().sum(dim=1),
            "orientation": data.projected_gravity_b[:, :2].square().sum(dim=1),
            "dof_vel": velocity[:, :4].square().sum(dim=1),
            "dof_acc": acceleration.square().sum(dim=1),
            "torques": data.actuator_force[:, state["actuator_ids"]].square().sum(dim=1),
            "action_rate": (action - state["prev_action"]).square().sum(dim=1),
            "action_smooth": second[:, :4].square().sum(dim=1),
            # 原实验 penalize_contacts_on=[]，这项实际恒为零。
            "collision": torch.zeros_like(ev2),
            "dof_pos_limits": excess.sum(dim=1),
        }
        state["values"] = {
            name: (raw[name] * scale).clamp(-1, 1) for name, scale in WEIGHTS.items()
        }
        state["prev_prev_action"].copy_(state["prev_action"])
        state["prev_action"].copy_(action)
        state["prev_velocity"].copy_(velocity)
        state["step"] = env.common_step_counter
    return state["values"][term]
