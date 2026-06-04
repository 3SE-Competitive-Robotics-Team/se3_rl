"""Task Mode 奖励包装。

这些函数复用现有奖励公式，只负责把奖励挂到明确的 Task Mode 上。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from se3_shared import DM8009P, M3508_HEXROLL, JointGroup, TaskMode
from se3_train.mdp import rewards
from se3_train.mdp.contact_utils import finite_contact_force_norm
from se3_train.mdp.jump_rewards import _landing_impulse_support_error
from se3_train.mdp.task_modes import mode_weight

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_WHEEL_RADIUS = 0.059


def _mean_on_mask(value: torch.Tensor, mask: torch.Tensor) -> float:
    """计算掩码内均值；无样本时返回 0。"""
    if mask.any():
        return float(value[mask].mean().item())
    return 0.0


def _wheel_body_ids(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> list[int]:
    """缓存左右轮 body 在 entity 内的局部索引。"""
    attr_name = f"_task_mode_wheel_body_ids_{asset_cfg.name}"
    cached = getattr(env, attr_name, None)
    if isinstance(cached, list) and len(cached) == 2:
        return cached
    robot = env.scene[asset_cfg.name]
    body_ids, body_names = robot.find_bodies(("l_wheel_Link", "r_wheel_Link"), preserve_order=True)
    if len(body_ids) != 2:
        raise RuntimeError(f"必须找到左右轮 body,实际找到: {body_names}")
    setattr(env, attr_name, body_ids)
    return body_ids


def _leg_body_ids(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> list[int]:
    """缓存四个腿杆 body 在 entity 内的局部索引。"""
    attr_name = f"_task_mode_leg_body_ids_{asset_cfg.name}"
    cached = getattr(env, attr_name, None)
    if isinstance(cached, list) and len(cached) == 4:
        return cached
    robot = env.scene[asset_cfg.name]
    body_ids, body_names = robot.find_bodies(
        ("lf0_Link", "lf1_Link", "rf0_Link", "rf1_Link"),
        preserve_order=True,
    )
    if len(body_ids) != 4:
        raise RuntimeError(f"必须找到四个腿杆 body,实际找到: {body_names}")
    setattr(env, attr_name, body_ids)
    return body_ids


def _any_body_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str | None,
    force_threshold: float,
) -> torch.Tensor:
    """返回指定接触传感器是否有任意 body 触地。"""
    if sensor_name is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    return finite_contact_force_norm(data.force).max(dim=1).values > float(force_threshold)


def _wheel_pos_body_frame(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """返回左右轮中心在机身坐标系中的位置。"""
    robot = env.scene[asset_cfg.name]
    body_ids = _wheel_body_ids(env, asset_cfg)
    delta_w = robot.data.body_link_pos_w[:, body_ids, :] - robot.data.root_link_pos_w[:, None, :]
    quat = robot.data.root_link_quat_w[:, None, :].expand(-1, len(body_ids), -1)
    return quat_apply_inverse(quat.reshape(-1, 4), delta_w.reshape(-1, 3)).reshape(
        env.num_envs, len(body_ids), 3
    )


def gait_leg_contact_force(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    force_scale: float = 50.0,
    contact_threshold: float = 1.0,
    contact_event_scale: float = 0.0,
) -> torch.Tensor:
    """GAIT 模式腿部连杆触地连续惩罚。

    leg_contact 终止只告诉策略"这一条 episode 结束了"。这里把腿部触地力
    和触地事件直接写进 reward，让策略区分"干净抬轮"和"用腿连杆蹭地换高度"。
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force_mag = finite_contact_force_norm(data.force)
    force_norm = torch.clamp(force_mag / float(force_scale), max=2.0)
    gate = mode_weight(env, command_name, TaskMode.GAIT)
    contact = force_mag > float(contact_threshold)
    contact_event = contact.any(dim=1).float() * float(contact_event_scale)
    active = gate > 0.0
    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        labels = ("lf0", "lf1", "rf0", "rf1")
        log = {
            "TaskMode/diag_gait_leg_contact_ratio": _mean_on_mask(
                contact.any(dim=1).float(),
                active,
            ),
            "TaskMode/diag_gait_leg_contact_force_max": _mean_on_mask(
                force_mag.max(dim=1).values,
                active,
            ),
        }
        for idx, label in enumerate(labels):
            log[f"TaskMode/diag_gait_leg_contact_{label}"] = _mean_on_mask(
                contact[:, idx].float(),
                active,
            )
        env.extras["log"].update(log)
    return (torch.sum(force_norm, dim=1) + contact_event) * gate


def gait_low_base_height_barrier(
    env: ManagerBasedRlEnv,
    command_name: str,
    height_sensor_name: str,
    soft_min_height: float = 0.32,
    hard_min_height: float = 0.24,
    max_penalty: float = 4.0,
) -> torch.Tensor:
    """GAIT 模式低机身高度 barrier。

    这个项只惩罚低于软下限的 base height。它不要求精确跟踪目标高度，
    只给"base 越接近贴地越差"的连续信号，避免策略只在接触终止时才收到反馈。
    """
    from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

    sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    height = sensor.data.heights[:, 0]
    band = max(float(soft_min_height) - float(hard_min_height), 1.0e-6)
    deficit = torch.clamp((float(soft_min_height) - height) / band, min=0.0)
    penalty = torch.clamp(deficit**2, max=float(max_penalty))
    gate = mode_weight(env, command_name, TaskMode.GAIT)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        active = gate > 0.0
        env.extras["log"].update(
            {
                "TaskMode/diag_gait_base_height_m": _mean_on_mask(height, active),
                "TaskMode/diag_gait_low_base_height_barrier": _mean_on_mask(
                    penalty,
                    active,
                ),
            }
        )

    return penalty * gate


def gait_natural_swing_clearance(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    leg_sensor_name: str | None = None,
    target_clearance: float = 0.04,
    balance_window_s: float = 0.6,
    contact_force_threshold: float = 1.0,
    leg_contact_force_threshold: float = 0.2,
    wheel_radius: float = _WHEEL_RADIUS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """无相位双侧摆腿高度奖励。

    不指定哪条腿在某一刻摆动，只要求一个短窗口内左右两侧都能贡献离地高度。
    奖励取左右 EMA 中较弱的一侧，堵住单侧高抬、另一侧贴地的套利。
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene[asset_cfg.name]
    body_ids = _wheel_body_ids(env, asset_cfg)
    wheel_z_w = robot.data.body_link_pos_w[:, body_ids, 2]
    clearance = torch.clamp(wheel_z_w - float(wheel_radius), min=0.0)
    normalized = torch.clamp(clearance / float(target_clearance), max=2.0)

    contact = finite_contact_force_norm(data.force) > float(contact_force_threshold)
    left_score = normalized[:, 0] * (~contact[:, 0]).float() * contact[:, 1].float()
    right_score = normalized[:, 1] * (~contact[:, 1]).float() * contact[:, 0].float()
    no_leg_contact = ~_any_body_contact(env, leg_sensor_name, leg_contact_force_threshold)

    left_attr = "_gait_swing_clearance_left_ema"
    right_attr = "_gait_swing_clearance_right_ema"
    left_ema = getattr(env, left_attr, None)
    right_ema = getattr(env, right_attr, None)
    if not isinstance(left_ema, torch.Tensor) or left_ema.shape != (env.num_envs,):
        left_ema = torch.zeros(env.num_envs, device=env.device)
    if not isinstance(right_ema, torch.Tensor) or right_ema.shape != (env.num_envs,):
        right_ema = torch.zeros(env.num_envs, device=env.device)

    new_episode = env.episode_length_buf <= 1
    if new_episode.any():
        left_ema[new_episode] = 0.0
        right_ema[new_episode] = 0.0

    alpha = min(float(env.step_dt) / max(float(balance_window_s), float(env.step_dt)), 1.0)
    left_ema = (1.0 - alpha) * left_ema + alpha * left_score
    right_ema = (1.0 - alpha) * right_ema + alpha * right_score

    raw_best = torch.maximum(left_score, right_score)
    reward = torch.clamp(2.0 * torch.minimum(left_ema, right_ema), max=2.0)
    reward = reward * no_leg_contact.float()

    setattr(env, left_attr, left_ema.detach())
    setattr(env, right_attr, right_ema.detach())
    gate = mode_weight(env, command_name, TaskMode.GAIT)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        active = gate > 0.0
        env.extras["log"].update(
            {
                "TaskMode/diag_gait_raw_swing_clearance": _mean_on_mask(
                    raw_best,
                    active,
                ),
                "TaskMode/diag_gait_swing_clearance_left_ema": _mean_on_mask(
                    left_ema,
                    active,
                ),
                "TaskMode/diag_gait_swing_clearance_right_ema": _mean_on_mask(
                    right_ema,
                    active,
                ),
                "TaskMode/diag_gait_natural_swing_clearance": _mean_on_mask(
                    reward,
                    active,
                ),
            }
        )

    return reward * gate


def gait_single_support_contact(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """无相位单腿支撑接触奖励。

    奖励左右轮恰好一个触地，避免策略用双轮贴地绕过自然抬腿阶段。
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    contact = finite_contact_force_norm(data.force) > float(contact_force_threshold)
    contact_count = contact.float().sum(dim=1)
    reward = (contact_count == 1.0).float()
    gate = mode_weight(env, command_name, TaskMode.GAIT)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        active = gate > 0.0
        env.extras["log"].update(
            {
                "TaskMode/diag_gait_single_support_ratio": _mean_on_mask(reward, active),
            }
        )

    return reward * gate


def gait_single_support_air_time(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    target_air_time: float = 0.08,
    min_command: float = 0.05,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """单支撑持续离地奖励。

    只有恰好一个轮子触地，另一侧轮子已经连续离地一段时间时给分。
    这保留自然单支撑目标，同时避免短促点地刷单支撑接触奖励。
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    attr_time = "_gait_single_support_air_time_seconds"
    air_time = getattr(env, attr_time, None)
    if not isinstance(air_time, torch.Tensor) or air_time.shape != (env.num_envs, 2):
        air_time = torch.zeros(env.num_envs, 2, device=env.device)

    contact = finite_contact_force_norm(data.force) > float(contact_force_threshold)
    new_episode = env.episode_length_buf <= 1
    if new_episode.any():
        air_time[new_episode] = 0.0

    air_time = torch.where(contact, torch.zeros_like(air_time), air_time + float(env.step_dt))

    single_support = contact[:, 0] ^ contact[:, 1]
    left_swing = (~contact[:, 0]) & contact[:, 1]
    right_swing = contact[:, 0] & (~contact[:, 1])
    swing_air_time = torch.where(
        left_swing,
        air_time[:, 0],
        torch.zeros(env.num_envs, device=env.device),
    )
    swing_air_time = torch.where(right_swing, air_time[:, 1], swing_air_time)
    score = torch.clamp(swing_air_time / float(target_air_time), max=1.0)

    cmd = env.command_manager.get_command(command_name)
    moving = (torch.abs(cmd[:, 0]) > float(min_command)) | (
        torch.abs(cmd[:, 1]) > float(min_command)
    )
    reward = score * single_support.float() * moving.float()

    setattr(env, attr_time, air_time.detach())

    gate = mode_weight(env, command_name, TaskMode.GAIT)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        active = gate > 0.0
        env.extras["log"].update(
            {
                "TaskMode/diag_gait_single_support_air_time": _mean_on_mask(
                    swing_air_time,
                    active,
                ),
                "TaskMode/diag_gait_single_support_air_time_reward": _mean_on_mask(
                    reward,
                    active,
                ),
            }
        )

    return reward * gate


def gait_stuck_stance_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    grace_time_s: float = 0.35,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """惩罚同一侧轮子长时间连续作为唯一支撑。

    单腿支撑本身是对的，但同一条腿一直支撑会退化成“固定一腿站住，另一腿抖动”。
    这里记录连续 stance 侧，超过宽限时间后线性加重惩罚，换腿或非单支撑时重置。
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    side_attr = "_gait_stuck_stance_side"
    steps_attr = "_gait_stuck_stance_steps"
    side = getattr(env, side_attr, None)
    steps = getattr(env, steps_attr, None)
    if not isinstance(side, torch.Tensor) or side.shape != (env.num_envs,):
        side = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    if not isinstance(steps, torch.Tensor) or steps.shape != (env.num_envs,):
        steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    contact = finite_contact_force_norm(data.force) > float(contact_force_threshold)
    single_support = contact[:, 0] ^ contact[:, 1]
    current_side = torch.full_like(side, -1)
    current_side[contact[:, 0] & ~contact[:, 1]] = 0
    current_side[contact[:, 1] & ~contact[:, 0]] = 1

    same_side = single_support & (current_side == side)
    steps = torch.where(same_side, steps + 1, torch.zeros_like(steps))
    steps = torch.where(single_support & ~same_side, torch.ones_like(steps), steps)
    side = torch.where(single_support, current_side, torch.full_like(side, -1))

    setattr(env, side_attr, side)
    setattr(env, steps_attr, steps)

    grace_steps = max(round(float(grace_time_s) / env.step_dt), 1)
    excess = torch.clamp(steps.float() - float(grace_steps), min=0.0)
    penalty = torch.clamp(excess / float(grace_steps), max=2.0)
    gate = mode_weight(env, command_name, TaskMode.GAIT)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        active = gate > 0.0
        env.extras["log"].update(
            {
                "TaskMode/diag_gait_stuck_stance_penalty": _mean_on_mask(penalty, active),
            }
        )

    return penalty * gate


def gait_swing_side_balance_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    window_s: float = 1.0,
    min_single_support_ratio: float = 0.2,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """惩罚左右摆腿占比长期失衡。

    该项不指定哪一时刻该抬哪条腿，只要求一个时间窗口内左右两侧都参与摆动。
    失衡直接线性计入惩罚，让单侧摆腿在 PPO 更新里有足够大的反向信号。
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    left_attr = "_gait_swing_balance_left_ema"
    right_attr = "_gait_swing_balance_right_ema"
    left_ema = getattr(env, left_attr, None)
    right_ema = getattr(env, right_attr, None)
    if not isinstance(left_ema, torch.Tensor) or left_ema.shape != (env.num_envs,):
        left_ema = torch.zeros(env.num_envs, device=env.device)
    if not isinstance(right_ema, torch.Tensor) or right_ema.shape != (env.num_envs,):
        right_ema = torch.zeros(env.num_envs, device=env.device)

    contact = finite_contact_force_norm(data.force) > float(contact_force_threshold)
    new_episode = env.episode_length_buf <= 1
    if new_episode.any():
        left_ema[new_episode] = 0.0
        right_ema[new_episode] = 0.0

    left_swing = (~contact[:, 0]) & contact[:, 1]
    right_swing = contact[:, 0] & (~contact[:, 1])
    alpha = min(float(env.step_dt) / max(float(window_s), float(env.step_dt)), 1.0)
    left_ema = (1.0 - alpha) * left_ema + alpha * left_swing.float()
    right_ema = (1.0 - alpha) * right_ema + alpha * right_swing.float()

    single_support_ratio = left_ema + right_ema
    active_enough = single_support_ratio > float(min_single_support_ratio)
    imbalance = torch.abs(left_ema - right_ema) / torch.clamp(single_support_ratio, min=1.0e-6)
    penalty = torch.where(active_enough, imbalance, torch.zeros_like(imbalance))

    setattr(env, left_attr, left_ema.detach())
    setattr(env, right_attr, right_ema.detach())

    gate = mode_weight(env, command_name, TaskMode.GAIT)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        active = gate > 0.0
        env.extras["log"].update(
            {
                "TaskMode/diag_gait_swing_left_ema": _mean_on_mask(left_ema, active),
                "TaskMode/diag_gait_swing_right_ema": _mean_on_mask(right_ema, active),
                "TaskMode/diag_gait_swing_balance_penalty": _mean_on_mask(penalty, active),
            }
        )

    return penalty * gate


def gait_air_time(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    leg_sensor_name: str | None = None,
    target_air_time: float = 0.35,
    max_reward_air_time: float = 0.8,
    min_command: float = 0.1,
    contact_force_threshold: float = 1.0,
    leg_contact_force_threshold: float = 0.2,
) -> torch.Tensor:
    """Legged Gym 风格离地时间奖励。

    奖励在摆动轮重新触地的瞬间发放，数值取决于本次离地持续时间。
    低速或零速指令下不发放，避免策略为了原地刷奖励而颤腿。
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    attr_time = "_gait_air_time_seconds"
    attr_last_contact = "_gait_air_time_last_contact"
    attr_dirty_air = "_gait_air_time_dirty_air"
    air_time = getattr(env, attr_time, None)
    last_contact = getattr(env, attr_last_contact, None)
    dirty_air = getattr(env, attr_dirty_air, None)
    if not isinstance(air_time, torch.Tensor) or air_time.shape != (env.num_envs, 2):
        air_time = torch.zeros(env.num_envs, 2, device=env.device)
    if not isinstance(last_contact, torch.Tensor) or last_contact.shape != (env.num_envs, 2):
        last_contact = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
    if not isinstance(dirty_air, torch.Tensor) or dirty_air.shape != (env.num_envs, 2):
        dirty_air = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)

    contact = finite_contact_force_norm(data.force) > float(contact_force_threshold)
    new_episode = env.episode_length_buf <= 1
    if new_episode.any():
        air_time[new_episode] = 0.0
        last_contact[new_episode] = contact[new_episode]
        dirty_air[new_episode] = False

    leg_contact = _any_body_contact(env, leg_sensor_name, leg_contact_force_threshold)
    dirty_air = dirty_air | ((~contact) & leg_contact.unsqueeze(1))
    contact_filt = contact | last_contact
    first_contact = (air_time > 0.0) & contact_filt
    air_time = air_time + float(env.step_dt)

    reward_window = max(float(max_reward_air_time) - float(target_air_time), 1.0e-6)
    reward_air_time = torch.clamp(
        (air_time - float(target_air_time)) / reward_window,
        min=0.0,
        max=1.0,
    )
    reward_air_time = 0.5 * reward_air_time + 0.5 * reward_air_time**2
    cmd = env.command_manager.get_command(command_name)
    moving = (torch.abs(cmd[:, 0]) > float(min_command)) | (
        torch.abs(cmd[:, 1]) > float(min_command)
    )
    clean_touchdown = first_contact & (~dirty_air)
    reward = torch.sum(reward_air_time * clean_touchdown.float(), dim=1) * moving.float()

    air_time = torch.where(contact_filt, torch.zeros_like(air_time), air_time)
    dirty_air = torch.where(contact_filt, torch.zeros_like(dirty_air), dirty_air)
    setattr(env, attr_time, air_time.detach())
    setattr(env, attr_last_contact, contact.detach())
    setattr(env, attr_dirty_air, dirty_air.detach())

    gate = mode_weight(env, command_name, TaskMode.GAIT)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        active = gate > 0.0
        env.extras["log"].update(
            {
                "TaskMode/diag_gait_air_time_s": _mean_on_mask(
                    air_time.max(dim=1).values,
                    active,
                ),
                "TaskMode/diag_gait_air_time_touchdown": _mean_on_mask(
                    first_contact.float().sum(dim=1),
                    active,
                ),
                "TaskMode/diag_gait_moving_command_ratio": _mean_on_mask(
                    moving.float(),
                    active,
                ),
            }
        )

    return reward * gate


def gait_alternating_air_time(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    leg_sensor_name: str | None = None,
    target_air_time: float = 0.18,
    max_reward_air_time: float = 0.6,
    min_command: float = 0.05,
    contact_force_threshold: float = 1.0,
    leg_contact_force_threshold: float = 0.2,
) -> torch.Tensor:
    """无相位交替落地奖励。

    轮子离地时间达标后重新触地才发奖励。连续同侧落地不给分，
    促使策略自然形成左右交替支撑，而不用外部相位指定哪条腿该动。
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    attr_time = "_gait_alt_air_time_seconds"
    attr_last_contact = "_gait_alt_last_contact"
    attr_last_touch_side = "_gait_alt_last_touch_side"
    attr_dirty_air = "_gait_alt_dirty_air"
    air_time = getattr(env, attr_time, None)
    last_contact = getattr(env, attr_last_contact, None)
    last_touch_side = getattr(env, attr_last_touch_side, None)
    dirty_air = getattr(env, attr_dirty_air, None)
    if not isinstance(air_time, torch.Tensor) or air_time.shape != (env.num_envs, 2):
        air_time = torch.zeros(env.num_envs, 2, device=env.device)
    if not isinstance(last_contact, torch.Tensor) or last_contact.shape != (env.num_envs, 2):
        last_contact = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
    if not isinstance(last_touch_side, torch.Tensor) or last_touch_side.shape != (env.num_envs,):
        last_touch_side = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    if not isinstance(dirty_air, torch.Tensor) or dirty_air.shape != (env.num_envs, 2):
        dirty_air = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)

    contact = finite_contact_force_norm(data.force) > float(contact_force_threshold)
    new_episode = env.episode_length_buf <= 1
    if new_episode.any():
        air_time[new_episode] = 0.0
        last_contact[new_episode] = contact[new_episode]
        last_touch_side[new_episode] = -1
        dirty_air[new_episode] = False

    leg_contact = _any_body_contact(env, leg_sensor_name, leg_contact_force_threshold)
    dirty_air = dirty_air | ((~contact) & leg_contact.unsqueeze(1))
    contact_filt = contact | last_contact
    first_contact = (air_time > 0.0) & contact_filt
    air_time = air_time + float(env.step_dt)

    reward_window = max(float(max_reward_air_time) - float(target_air_time), 1.0e-6)
    reward_air_time = torch.clamp(
        (air_time - float(target_air_time)) / reward_window,
        min=0.0,
        max=1.0,
    )
    reward_air_time = 0.5 * reward_air_time + 0.5 * reward_air_time**2
    side_id = (
        torch.arange(2, device=env.device, dtype=torch.long).unsqueeze(0).expand(env.num_envs, -1)
    )
    alternating = side_id != last_touch_side.unsqueeze(1)
    cmd = env.command_manager.get_command(command_name)
    moving = (torch.abs(cmd[:, 0]) > float(min_command)) | (
        torch.abs(cmd[:, 1]) > float(min_command)
    )
    reward = (
        torch.sum(
            reward_air_time * first_contact.float() * alternating.float() * (~dirty_air).float(),
            dim=1,
        )
        * moving.float()
    )

    touch_this_step = first_contact.any(dim=1)
    touched_side = torch.argmax(
        torch.where(first_contact, air_time, torch.full_like(air_time, -1.0)),
        dim=1,
    )
    last_touch_side = torch.where(touch_this_step, touched_side, last_touch_side)

    air_time = torch.where(contact_filt, torch.zeros_like(air_time), air_time)
    dirty_air = torch.where(contact_filt, torch.zeros_like(dirty_air), dirty_air)
    setattr(env, attr_time, air_time.detach())
    setattr(env, attr_last_contact, contact.detach())
    setattr(env, attr_last_touch_side, last_touch_side.detach())
    setattr(env, attr_dirty_air, dirty_air.detach())

    gate = mode_weight(env, command_name, TaskMode.GAIT)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        active = gate > 0.0
        env.extras["log"].update(
            {
                "TaskMode/diag_gait_alt_air_time_s": _mean_on_mask(
                    air_time.max(dim=1).values,
                    active,
                ),
                "TaskMode/diag_gait_alt_touchdown": _mean_on_mask(
                    first_contact.float().sum(dim=1),
                    active,
                ),
                "TaskMode/diag_gait_alt_reward": _mean_on_mask(reward, active),
            }
        )

    return reward * gate


def gait_short_air_time_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    min_air_time: float = 0.04,
    min_command: float = 0.05,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """惩罚过短离地后的落地。

    单支撑奖励会鼓励释放一侧接触，但如果马上落地，行为会退化成短促点地。
    该项只在 touchdown 瞬间计分，离地时间低于阈值时给惩罚。
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    attr_time = "_gait_short_air_time_seconds"
    attr_last_contact = "_gait_short_last_contact"
    air_time = getattr(env, attr_time, None)
    last_contact = getattr(env, attr_last_contact, None)
    if not isinstance(air_time, torch.Tensor) or air_time.shape != (env.num_envs, 2):
        air_time = torch.zeros(env.num_envs, 2, device=env.device)
    if not isinstance(last_contact, torch.Tensor) or last_contact.shape != (env.num_envs, 2):
        last_contact = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)

    contact = finite_contact_force_norm(data.force) > float(contact_force_threshold)
    new_episode = env.episode_length_buf <= 1
    if new_episode.any():
        air_time[new_episode] = 0.0
        last_contact[new_episode] = contact[new_episode]

    contact_filt = contact | last_contact
    first_contact = (air_time > 0.0) & contact_filt
    air_time = air_time + float(env.step_dt)

    shortfall = torch.clamp((float(min_air_time) - air_time) / float(min_air_time), min=0.0)
    cmd = env.command_manager.get_command(command_name)
    moving = (torch.abs(cmd[:, 0]) > float(min_command)) | (
        torch.abs(cmd[:, 1]) > float(min_command)
    )
    penalty = torch.sum(shortfall * first_contact.float(), dim=1) * moving.float()

    air_time = torch.where(contact_filt, torch.zeros_like(air_time), air_time)
    setattr(env, attr_time, air_time.detach())
    setattr(env, attr_last_contact, contact.detach())

    gate = mode_weight(env, command_name, TaskMode.GAIT)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        active = gate > 0.0
        env.extras["log"].update(
            {
                "TaskMode/diag_gait_short_air_time_penalty": _mean_on_mask(
                    penalty,
                    active,
                ),
            }
        )

    return penalty * gate


def gait_action_smoothness(
    env: ManagerBasedRlEnv,
    command_name: str,
    max_penalty: float = 80.0,
) -> torch.Tensor:
    """GAIT 模式二阶动作平滑惩罚。

    action_rate 惩罚相邻动作差值；这里惩罚动作曲率，压制快速来回抽动。
    该项不指定步态相位，只约束动作变化方式。
    """
    action = env.action_manager.action
    prev_attr = "_gait_action_smooth_prev_action"
    prev_prev_attr = "_gait_action_smooth_prev_prev_action"
    prev = getattr(env, prev_attr, None)
    prev_prev = getattr(env, prev_prev_attr, None)

    if not isinstance(prev, torch.Tensor) or prev.shape != action.shape:
        setattr(env, prev_attr, action.detach().clone())
        setattr(env, prev_prev_attr, action.detach().clone())
        return torch.zeros(env.num_envs, device=env.device)
    if not isinstance(prev_prev, torch.Tensor) or prev_prev.shape != action.shape:
        setattr(env, prev_prev_attr, prev.detach().clone())
        setattr(env, prev_attr, action.detach().clone())
        return torch.zeros(env.num_envs, device=env.device)

    penalty = torch.sum((action - 2.0 * prev + prev_prev) ** 2, dim=1)
    setattr(env, prev_prev_attr, prev.detach().clone())
    setattr(env, prev_attr, action.detach().clone())

    startup = env.episode_length_buf < 3
    penalty = torch.where(startup, torch.zeros_like(penalty), penalty)
    penalty = torch.clamp(penalty, max=float(max_penalty))
    gate = mode_weight(env, command_name, TaskMode.GAIT)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        active = gate > 0.0
        env.extras["log"].update(
            {
                "TaskMode/diag_gait_action_smoothness": _mean_on_mask(
                    penalty,
                    active,
                ),
            }
        )

    return penalty * gate


def gait_touchdown_softness(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    allowed_down_vel: float = 0.12,
    max_penalty: float = 4.0,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """GAIT 模式摆动轮落地缓冲惩罚。

    轮子从离地切到触地的瞬间，如果 base 仍有明显向下速度，则惩罚。
    这鼓励策略在 touchdown 前收住竖直下落速度，减少硬砸地。
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    contact = finite_contact_force_norm(data.force) > float(contact_force_threshold)
    last_attr = "_gait_touchdown_soft_last_contact"
    last_contact = getattr(env, last_attr, None)
    if not isinstance(last_contact, torch.Tensor) or last_contact.shape != (env.num_envs, 2):
        last_contact = contact.clone()

    new_episode = env.episode_length_buf <= 1
    if new_episode.any():
        last_contact[new_episode] = contact[new_episode]

    touchdown = contact & (~last_contact)
    robot = env.scene["robot"]
    down_vel = torch.clamp(-robot.data.root_link_lin_vel_b[:, 2] - float(allowed_down_vel), min=0.0)
    penalty = torch.clamp(down_vel**2, max=float(max_penalty)) * touchdown.any(dim=1).float()
    setattr(env, last_attr, contact.detach())

    gate = mode_weight(env, command_name, TaskMode.GAIT)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        active = gate > 0.0
        env.extras["log"].update(
            {
                "TaskMode/diag_gait_touchdown_ratio": _mean_on_mask(
                    touchdown.any(dim=1).float(),
                    active,
                ),
                "TaskMode/diag_gait_touchdown_down_vel": _mean_on_mask(
                    down_vel,
                    active,
                ),
                "TaskMode/diag_gait_touchdown_softness": _mean_on_mask(
                    penalty,
                    active,
                ),
            }
        )

    return penalty * gate


def gait_touchdown_support_alignment(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    wheel_radius: float = _WHEEL_RADIUS,
    contact_force_threshold: float = 1.0,
    tolerance: float = 0.06,
    max_penalty: float = 4.0,
    height_sensor_name: str = "base_height_sensor",
    landing_up_speed: float = 0.0,
    min_down_speed: float = 0.20,
    min_prediction_time: float = 0.04,
    max_prediction_time: float = 0.24,
    max_support_offset: float = 0.30,
    lateral_weight: float = 0.35,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """GAIT 和 GAIT_WHEEL 模式首次 touchdown 的支撑落点惩罚。"""
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    contact = finite_contact_force_norm(data.force) > float(contact_force_threshold)
    last_attr = "_gait_touchdown_support_last_contact"
    last_contact = getattr(env, last_attr, None)
    if not isinstance(last_contact, torch.Tensor) or last_contact.shape != (env.num_envs, 2):
        last_contact = torch.zeros_like(contact)

    new_episode = env.episode_length_buf <= 1
    if new_episode.any():
        last_contact[new_episode] = False

    touchdown = contact & (~last_contact)
    cmd = env.command_manager.get_command(command_name)
    geom = _landing_impulse_support_error(
        env,
        asset_cfg,
        wheel_radius=wheel_radius,
        target_vxy=cmd[:, :2],
        height_sensor_name=height_sensor_name,
        landing_up_speed=landing_up_speed,
        min_down_speed=min_down_speed,
        min_prediction_time=min_prediction_time,
        max_prediction_time=max_prediction_time,
        max_support_offset=max_support_offset,
        lateral_weight=lateral_weight,
    )

    tol = max(float(tolerance), 1.0e-6)
    per_wheel_excess = torch.clamp(geom["dist_by_wheel"] - tol, min=0.0)
    per_wheel_penalty = torch.clamp(per_wheel_excess / tol, max=float(max_penalty))
    touchdown_count = touchdown.sum(dim=1)
    penalty = torch.where(
        touchdown_count > 0,
        (per_wheel_penalty * touchdown.float()).sum(dim=1) / touchdown_count.clamp_min(1).float(),
        torch.zeros(env.num_envs, device=env.device),
    )
    setattr(env, last_attr, contact.detach())

    gate = mode_weight(env, command_name, TaskMode.GAIT, TaskMode.GAIT_WHEEL)
    active = gate > 0.0
    touchdown_any = touchdown.any(dim=1)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"].update(
            {
                "TaskMode/diag_gait_touchdown_support_error_m": _mean_on_mask(
                    torch.where(
                        touchdown_count > 0,
                        (geom["dist_by_wheel"] * touchdown.float()).sum(dim=1)
                        / touchdown_count.clamp_min(1).float(),
                        torch.zeros(env.num_envs, device=env.device),
                    ),
                    active & touchdown_any,
                ),
                "TaskMode/diag_gait_touchdown_support_penalty": _mean_on_mask(
                    penalty,
                    active & touchdown_any,
                ),
                "TaskMode/diag_gait_touchdown_support_ratio": _mean_on_mask(
                    touchdown_any.float(),
                    active,
                ),
            }
        )

    return penalty * gate


def gait_pose(
    env: ManagerBasedRlEnv,
    command_name: str,
    hip_std: float = 0.5,
    knee_std: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """GAIT 模式松散姿态正奖励（G1 variable_posture 风格）。

    reward = exp(-mean((q - q_default)² / σ²))

    在 default 附近给正分（最高 1.0），远离时奖励指数衰减但不惩罚。
    高 std = 宽松允许大幅运动；策略有动力回到附近但不会被锁死。
    当速度跟踪说"你得走"时，策略愿意离开（奖励减少，不是被罚）。
    """
    robot = env.scene[asset_cfg.name]
    pos = robot.data.joint_pos[:, JointGroup.LEGS]
    default = robot.data.default_joint_pos[:, JointGroup.LEGS]
    diff = pos - default
    std = torch.tensor(
        [float(hip_std), float(knee_std), float(hip_std), float(knee_std)],
        device=pos.device,
    )
    reward = torch.exp(-torch.mean((diff / std) ** 2, dim=1))
    gate = mode_weight(env, command_name, TaskMode.GAIT)
    return reward * gate


def wheel_feet_distance(
    env: ManagerBasedRlEnv,
    command_name: str,
    min_feet_distance: float,
    max_feet_distance: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """WHEEL 模式左右轮心水平距离越界惩罚。

    参考 tron1 WF 的 feet_distance：只在左右轮心距离小于下限或大于上限时产生代价。
    """
    robot = env.scene[asset_cfg.name]
    body_ids = _wheel_body_ids(env, asset_cfg)
    wheel_pos = robot.data.body_link_pos_w[:, body_ids, :2]
    feet_distance = torch.linalg.norm(wheel_pos[:, 0, :] - wheel_pos[:, 1, :], dim=1)
    penalty = torch.clamp(float(min_feet_distance) - feet_distance, min=0.0)
    penalty += torch.clamp(feet_distance - float(max_feet_distance), min=0.0)
    gate = mode_weight(env, command_name, TaskMode.WHEEL)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        active = gate > 0.0
        env.extras["log"].update(
            {
                "TaskMode/diag_wheel_feet_distance_m": _mean_on_mask(
                    feet_distance,
                    active,
                ),
                "TaskMode/diag_wheel_feet_distance_penalty": _mean_on_mask(
                    penalty,
                    active,
                ),
            }
        )

    return penalty * gate


def wheel_idle_action_rate(
    env: ManagerBasedRlEnv,
    command_name: str,
    idle_command_threshold: float = 0.08,
    max_penalty: float = 80.0,
) -> torch.Tensor:
    """WHEEL 零指令静站时额外压低动作变化。"""
    cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.linalg.norm(cmd[:, :2], dim=1)
    idle = cmd_norm < float(idle_command_threshold)
    raw = torch.sum((env.action_manager.action - env.action_manager.prev_action) ** 2, dim=1)
    penalty = torch.clamp(raw, max=float(max_penalty))
    gate = mode_weight(env, command_name, TaskMode.WHEEL) * idle.float()

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        active = gate > 0.0
        env.extras["log"].update(
            {
                "TaskMode/diag_wheel_idle_action_rate": _mean_on_mask(penalty, active),
                "TaskMode/diag_wheel_idle_command_ratio": _mean_on_mask(
                    idle.float(),
                    mode_weight(env, command_name, TaskMode.WHEEL) > 0.0,
                ),
            }
        )

    return penalty * gate


def wheel_idle_motion_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    wheel_radius: float = _WHEEL_RADIUS,
    idle_command_threshold: float = 0.08,
    contact_force_threshold: float = 1.0,
    base_speed_scale: float = 0.18,
    wheel_speed_scale: float = 0.22,
    max_penalty: float = 9.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """WHEEL 零指令静站时惩罚机身水平运动和接地轮滚动。"""
    cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.linalg.norm(cmd[:, :2], dim=1)
    idle = cmd_norm < float(idle_command_threshold)

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)
    force_mag = finite_contact_force_norm(data.force)
    in_contact = force_mag > float(contact_force_threshold)
    has_contact = in_contact.any(dim=1)

    robot = env.scene[asset_cfg.name]
    base_vel_b = robot.data.root_link_lin_vel_b
    base_vxy_sq = base_vel_b[:, 0] ** 2 + base_vel_b[:, 1] ** 2

    wheel_vel = robot.data.joint_vel[:, JointGroup.WHEELS]
    wheel_forward_speed = torch.stack(
        (
            wheel_vel[:, 0] * float(wheel_radius),
            -wheel_vel[:, 1] * float(wheel_radius),
        ),
        dim=1,
    )
    contact_count = in_contact.float().sum(dim=1).clamp_min(1.0)
    wheel_speed_sq = torch.sum((wheel_forward_speed**2) * in_contact.float(), dim=1)
    wheel_speed_sq = wheel_speed_sq / contact_count

    penalty = base_vxy_sq / (float(base_speed_scale) ** 2) + wheel_speed_sq / (
        float(wheel_speed_scale) ** 2
    )
    penalty = torch.clamp(penalty, max=float(max_penalty))
    gate = mode_weight(env, command_name, TaskMode.WHEEL) * idle.float() * has_contact.float()

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        active = gate > 0.0
        env.extras["log"].update(
            {
                "TaskMode/diag_wheel_idle_motion_penalty": _mean_on_mask(penalty, active),
                "TaskMode/diag_wheel_idle_base_vxy": _mean_on_mask(
                    torch.sqrt(base_vxy_sq),
                    active,
                ),
                "TaskMode/diag_wheel_idle_wheel_speed_abs": _mean_on_mask(
                    torch.sqrt(wheel_speed_sq),
                    active,
                ),
            }
        )

    return penalty * gate


def wheel_straight_yaw_drift(
    env: ManagerBasedRlEnv,
    command_name: str,
    min_speed: float = 0.2,
    max_yaw_command: float = 0.05,
) -> torch.Tensor:
    """WHEEL 直线跑样本的 yaw 漂移惩罚。"""
    cmd = env.command_manager.get_command(command_name)
    straight = (torch.abs(cmd[:, 0]) > float(min_speed)) & (
        torch.abs(cmd[:, 1]) < float(max_yaw_command)
    )
    robot = env.scene["robot"]
    yaw_rate = robot.data.root_link_ang_vel_b[:, 2]
    penalty = yaw_rate**2
    gate = mode_weight(env, command_name, TaskMode.WHEEL) * straight.float()

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        active = gate > 0.0
        env.extras["log"].update(
            {
                "TaskMode/diag_wheel_straight_sample_ratio": _mean_on_mask(
                    straight.float(),
                    mode_weight(env, command_name, TaskMode.WHEEL) > 0.0,
                ),
                "TaskMode/diag_wheel_straight_yaw_rate_abs": _mean_on_mask(
                    torch.abs(yaw_rate),
                    active,
                ),
            }
        )

    return penalty * gate


def wheel_straight_lateral_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    min_speed: float = 0.2,
    max_yaw_command: float = 0.05,
) -> torch.Tensor:
    """WHEEL 直线跑样本的横向速度惩罚。"""
    cmd = env.command_manager.get_command(command_name)
    straight = (torch.abs(cmd[:, 0]) > float(min_speed)) & (
        torch.abs(cmd[:, 1]) < float(max_yaw_command)
    )
    robot = env.scene["robot"]
    lateral_vel = robot.data.root_link_lin_vel_b[:, 1]
    penalty = lateral_vel**2
    gate = mode_weight(env, command_name, TaskMode.WHEEL) * straight.float()

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        active = gate > 0.0
        env.extras["log"].update(
            {
                "TaskMode/diag_wheel_straight_lateral_vel_abs": _mean_on_mask(
                    torch.abs(lateral_vel),
                    active,
                ),
            }
        )

    return penalty * gate


def wheel_in_place_linear_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    max_linear_command: float = 0.05,
    min_yaw_command: float = 0.2,
) -> torch.Tensor:
    """WHEEL 原地转样本的平移速度惩罚。"""
    cmd = env.command_manager.get_command(command_name)
    in_place_turn = (torch.abs(cmd[:, 0]) < float(max_linear_command)) & (
        torch.abs(cmd[:, 1]) > float(min_yaw_command)
    )
    robot = env.scene["robot"]
    lin_vel_xy = robot.data.root_link_lin_vel_b[:, :2]
    penalty = torch.sum(lin_vel_xy**2, dim=1)
    gate = mode_weight(env, command_name, TaskMode.WHEEL) * in_place_turn.float()

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        active = gate > 0.0
        env.extras["log"].update(
            {
                "TaskMode/diag_wheel_turn_sample_ratio": _mean_on_mask(
                    in_place_turn.float(),
                    mode_weight(env, command_name, TaskMode.WHEEL) > 0.0,
                ),
                "TaskMode/diag_wheel_turn_linear_vel_abs": _mean_on_mask(
                    torch.linalg.norm(lin_vel_xy, dim=1),
                    active,
                ),
            }
        )

    return penalty * gate


def gait_no_wheel_drive(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """GAIT 模式下禁止轮子滚动和主动出力。

    纯 GAIT 要靠腿部摆动推进。轮子在地面上连续滚动会让策略绕过步态，
    所以同时惩罚轮速和轮子执行器出力。
    """
    robot = env.scene[asset_cfg.name]
    wheel_torque = robot.data.actuator_force[:, JointGroup.WHEEL_ACTUATORS]
    wheel_vel = robot.data.joint_vel[:, JointGroup.WHEELS]
    torque_cost = torch.sum((wheel_torque / float(M3508_HEXROLL.rated_torque)) ** 2, dim=1)
    vel_cost = torch.sum((wheel_vel / 20.0) ** 2, dim=1)
    cost = torque_cost + vel_cost
    gate = mode_weight(env, command_name, TaskMode.GAIT)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        active = gate > 0.0
        env.extras["log"].update(
            {
                "TaskMode/diag_gait_wheel_vel_cost": _mean_on_mask(vel_cost, active),
                "TaskMode/diag_gait_wheel_torque_cost": _mean_on_mask(torque_cost, active),
            }
        )

    return cost * gate


def gait_leg_idle_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    motion_threshold: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """GAIT 模式下腿不动的惩罚。

    无死区——GAIT 模式始终要求踏步，髋关节速度低于 motion_threshold 就罚。
    """
    robot = env.scene[asset_cfg.name]
    leg_vel = robot.data.joint_vel[:, JointGroup.LEGS]
    leg_speed = torch.abs(leg_vel).mean(dim=1)
    idle = (leg_speed < motion_threshold).float()
    gate = mode_weight(env, command_name, TaskMode.GAIT)
    return idle * gate


def gait_wheel_velocity_assist(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """gait_wheel mode 下奖励轮速方向与前进指令一致。

    轴方向约定（见 docs/common_mistakes.md#1）：
      前进时左轮 joint_vel > 0，右轮 joint_vel < 0（左右轴方向相反）。
      因此前进方向轮速代理 = (left_vel - right_vel) / 2，与 cmd_x 同号时为正。
    """
    cmd = env.command_manager.get_command(command_name)
    robot = env.scene[asset_cfg.name]
    wheel_vel = robot.data.joint_vel[:, JointGroup.WHEELS]  # [N, 2]: [left, right]
    # 前进速度代理：左轮 vel 减右轮 vel，异号相消得到前进分量
    forward_vel_proxy = (wheel_vel[:, 0] - wheel_vel[:, 1]) * 0.5
    assist = torch.tanh(cmd[:, 0] * forward_vel_proxy * 0.05)
    gate = mode_weight(env, command_name, TaskMode.GAIT_WHEEL)
    return assist * gate


def wheel_swing_clearance(
    env: ManagerBasedRlEnv,
    command_name: str,
    target_clearance_m: float,
    sensor_name: str,
    contact_force_threshold: float = 1.0,
    wheel_radius: float = _WHEEL_RADIUS,
    height_scale: float = 0.06,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """轮足版 Foot Clearance：摆动轮要抬到目标离地高度。

    NP3O 的 foot clearance 用足端横向速度隐式门控 swing 相。SerialLeg 的足端是轮子，
    这里改成两轮中心的机身坐标系 x/z 运动：轮子沿机身前后快速摆动时，高度误差会被放大。
    """
    cmd = env.command_manager.get_command(command_name)
    wheel_pos_b = _wheel_pos_body_frame(env, asset_cfg)
    wheel_z = wheel_pos_b[:, :, 2] - float(wheel_radius)

    robot = env.scene[asset_cfg.name]
    wheel_vel = robot.data.joint_vel[:, JointGroup.WHEELS] * float(wheel_radius)
    swing_speed = torch.abs(wheel_vel)

    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        no_contact = torch.ones(env.num_envs, 2, dtype=torch.bool, device=env.device)
    else:
        no_contact = finite_contact_force_norm(data.force) <= float(contact_force_threshold)

    target_z = -torch.clamp(cmd[:, 4] - float(target_clearance_m), min=0.02).unsqueeze(1)
    error = ((wheel_z - target_z) / float(height_scale)) ** 2
    penalty = torch.sum(error * swing_speed * no_contact.float(), dim=1)
    gate = mode_weight(env, command_name, TaskMode.GAIT, TaskMode.WHEEL_LEG, TaskMode.GAIT_WHEEL)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        active = gate > 0.0
        env.extras["log"].update(
            {
                "TaskMode/diag_wheel_swing_clearance_raw": _mean_on_mask(penalty, active),
                "TaskMode/diag_wheel_swing_no_contact_ratio": _mean_on_mask(
                    no_contact.float().mean(dim=1),
                    active,
                ),
            }
        )

    return penalty * gate


def wheel_stumble(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    horizontal_to_vertical_ratio: float = 5.0,
) -> torch.Tensor:
    """轮足版 Stumble：水平冲击远大于竖直支撑时视为踢到立面。"""
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force = torch.nan_to_num(data.force, nan=0.0, posinf=5000.0, neginf=-5000.0)
    horizontal = torch.linalg.norm(force[:, :, :2], dim=2)
    vertical = torch.abs(force[:, :, 2])
    stumble = torch.any(horizontal > float(horizontal_to_vertical_ratio) * vertical, dim=1)
    gate = mode_weight(env, command_name, TaskMode.WHEEL_LEG, TaskMode.GAIT_WHEEL)

    if hasattr(env, "extras") and isinstance(env.extras.get("log"), dict):
        env.extras["log"]["TaskMode/diag_wheel_stumble_ratio"] = _mean_on_mask(
            stumble.float(),
            gate > 0.0,
        )

    return stumble.float() * gate


def leg_obstacle_collision(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """腿杆碰撞惩罚：大腿/小腿接触地形表示越障动作失败。"""
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device)
    force_mag = finite_contact_force_norm(data.force)
    contact = force_mag > float(force_threshold)
    collision = contact.float().sum(dim=1)
    gate = mode_weight(env, command_name, TaskMode.WHEEL_LEG, TaskMode.GAIT_WHEEL)
    return collision * gate


def loco_base_height(
    env: ManagerBasedRlEnv,
    command_name: str,
    height_sensor_name: str,
    target_height: float = 0.32,
    tolerance: float = 0.04,
) -> torch.Tensor:
    """越障/步态模式机身高度约束，对应 NP3O base_height_up。"""
    from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

    sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    height = sensor.data.heights[:, 0]
    penalty = ((height - float(target_height)) / float(tolerance)) ** 2
    gate = mode_weight(env, command_name, TaskMode.GAIT, TaskMode.WHEEL_LEG, TaskMode.GAIT_WHEEL)
    return penalty * gate


def loco_orientation(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """越障模式机身姿态约束（GAIT 除外，步态迈腿必然有姿态波动）。"""
    robot = env.scene["robot"]
    penalty = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)
    gate = mode_weight(env, command_name, TaskMode.WHEEL_LEG, TaskMode.GAIT_WHEEL)
    return penalty * gate


def loco_lin_vel_z(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """越障模式竖直速度约束（GAIT 除外，迈腿有竖直分量）。"""
    robot = env.scene["robot"]
    penalty = robot.data.root_link_lin_vel_b[:, 2] ** 2
    gate = mode_weight(env, command_name, TaskMode.WHEEL_LEG, TaskMode.GAIT_WHEEL)
    return penalty * gate


def loco_ang_vel_xy(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """越障模式横滚/俯仰角速度约束（GAIT 除外，迈腿有角速度波动）。"""
    robot = env.scene["robot"]
    ang_vel = robot.data.root_link_ang_vel_b
    penalty = ang_vel[:, 0] ** 2 + ang_vel[:, 1] ** 2
    gate = mode_weight(env, command_name, TaskMode.WHEEL_LEG, TaskMode.GAIT_WHEEL)
    return penalty * gate


def loco_dof_pos_limit_cost(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """NP3O pos_limit 对应项：关节位置接近软限位时产生代价。"""
    base = rewards.dof_pos_limits(env, asset_cfg=asset_cfg)
    gate = mode_weight(env, command_name, TaskMode.GAIT, TaskMode.WHEEL_LEG, TaskMode.GAIT_WHEEL)
    return base * gate


def loco_torque_limit_cost(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """NP3O torque_limit 对应项：超过连续额定力矩的部分产生代价。"""
    robot = env.scene[asset_cfg.name]
    torques = robot.data.actuator_force
    leg_excess = torch.clamp(
        torch.abs(torques[:, JointGroup.LEG_ACTUATORS]) - float(DM8009P.rated_torque),
        min=0.0,
    )
    wheel_excess = torch.clamp(
        torch.abs(torques[:, JointGroup.WHEEL_ACTUATORS]) - float(M3508_HEXROLL.rated_torque),
        min=0.0,
    )
    cost = torch.sum(leg_excess**2, dim=1) + torch.sum(wheel_excess**2, dim=1)
    gate = mode_weight(env, command_name, TaskMode.GAIT, TaskMode.WHEEL_LEG, TaskMode.GAIT_WHEEL)
    return cost * gate


def loco_dof_vel_limit_cost(
    env: ManagerBasedRlEnv,
    command_name: str,
    leg_velocity_limit: float = DM8009P.no_load_speed,
    wheel_velocity_limit: float = M3508_HEXROLL.no_load_speed,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """NP3O dof_vel_limits 对应项：超过电机空载速度的部分产生代价。"""
    robot = env.scene[asset_cfg.name]
    leg_excess = torch.clamp(
        torch.abs(robot.data.joint_vel[:, JointGroup.LEGS]) - float(leg_velocity_limit),
        min=0.0,
    )
    wheel_excess = torch.clamp(
        torch.abs(robot.data.joint_vel[:, JointGroup.WHEELS]) - float(wheel_velocity_limit),
        min=0.0,
    )
    cost = torch.sum(leg_excess**2, dim=1) + torch.sum(wheel_excess**2, dim=1)
    gate = mode_weight(env, command_name, TaskMode.GAIT, TaskMode.WHEEL_LEG, TaskMode.GAIT_WHEEL)
    return cost * gate


def mode_tracking_lin_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    sigma_move: float,
    sigma_stand: float,
    vz_weight: float,
    modes: tuple[int, ...],
) -> torch.Tensor:
    """指定 mode 下的线速度跟踪。"""
    base = rewards.tracking_lin_vel(env, command_name, sigma_move, sigma_stand, vz_weight)
    gate = mode_weight(env, command_name, *modes)
    return base * gate


def mode_tracking_ang_vel(
    env: ManagerBasedRlEnv, command_name: str, sigma: float, modes: tuple[int, ...]
) -> torch.Tensor:
    """指定 mode 下的偏航速度跟踪。"""
    base = rewards.tracking_ang_vel(env, command_name, sigma)
    gate = mode_weight(env, command_name, *modes)
    return base * gate


def mode_stand_still(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float,
    default_height: float,
    height_tolerance: float,
    modes: tuple[int, ...],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """指定 mode 下的静站姿态惩罚。"""
    base = rewards.stand_still(
        env,
        command_name,
        command_threshold=command_threshold,
        default_height=default_height,
        height_tolerance=height_tolerance,
        asset_cfg=asset_cfg,
    )
    gate = mode_weight(env, command_name, *modes)
    return base * gate


__all__ = [
    "gait_action_smoothness",
    "gait_air_time",
    "gait_alternating_air_time",
    "gait_leg_idle_penalty",
    "gait_low_base_height_barrier",
    "gait_natural_swing_clearance",
    "gait_no_wheel_drive",
    "gait_pose",
    "gait_short_air_time_penalty",
    "gait_single_support_air_time",
    "gait_single_support_contact",
    "gait_stuck_stance_penalty",
    "gait_swing_side_balance_penalty",
    "gait_touchdown_softness",
    "gait_touchdown_support_alignment",
    "gait_wheel_velocity_assist",
    "leg_obstacle_collision",
    "loco_ang_vel_xy",
    "loco_base_height",
    "loco_dof_pos_limit_cost",
    "loco_dof_vel_limit_cost",
    "loco_lin_vel_z",
    "loco_orientation",
    "loco_torque_limit_cost",
    "mode_stand_still",
    "mode_tracking_ang_vel",
    "mode_tracking_lin_vel",
    "wheel_feet_distance",
    "wheel_idle_action_rate",
    "wheel_idle_motion_penalty",
    "wheel_in_place_linear_vel",
    "wheel_straight_lateral_vel",
    "wheel_straight_yaw_drift",
    "wheel_stumble",
    "wheel_swing_clearance",
]
