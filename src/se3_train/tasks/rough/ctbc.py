"""崎岖地形线的 CTBC 接线：接触触发的单侧轮端"后缩-抬升"前馈，用来把轮子送上台阶沿。

来源是本仓库 stair 线（tasks/stair）的 CTBC teacher-forcing，状态机本体在
`se3_train.mdp.ctbc_state.StairClimbState`，这里只放 rough 需要的三个事件和一个观测：

- startup `init_ctbc_state`：在 env 上挂 `stair_climb_state`（动作项按这个属性名读前馈，见
  mdp/actions.py `process_actions`）。
- interval `step_ctbc_state`：每个控制步用轮子对地形**立面**的水平接触力更新触发窗口和相位。
  立面用接触法向 |n_z| ≤ riser_normal_z_max 判定，上表面支撑不算顶住台阶。
- reset `reset_ctbc_state`：清状态。
- `ctbc_obs`：3 维 [左相位, 右相位, 触发位×退火权重]，占 Flat 契约里 jump_commands 那 3 个扩展槽
  （行走任务 jump_prob=0，那 3 维恒为 0）。

为什么需要它：R3 回放（model_500）显示策略在 4 cm 台阶前停在坑底平台边缘反复撞退，
6 cm 轮子靠滚动翻不过三分之二个半径高的竖直沿，地形感知高度下限只抬机身高度、不抬轮子。
CTBC 在轮子顶住立面时替策略把那一侧轮子向后上方缩回，再按轮次退火让策略自己接管。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from se3_train.mdp.ctbc_state import StairClimbState
from se3_train.mdp.observations import _finite_clamp

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

# 动作项按这个属性名找状态机（mdp/actions.py），与 stair 线保持同名。
CTBC_STATE_ATTR = "stair_climb_state"

# 从 state.diag() 里挑出常驻到 W&B 的键；Stair/ 命名空间被 log_filter 裁掉，这里改挂到 Rough/。
_DIAG_KEYS = {
    "Stair/diag_ctbc_trigger_rate": "Rough/ctbc_trigger_rate",
    "Stair/diag_ctbc_complete_ff_cycles": "Rough/ctbc_complete_ff_cycles",
    "Stair/diag_riser_stall_rate": "Rough/ctbc_riser_stall_rate",
    "Stair/diag_ctbc_kff": "Rough/ctbc_kff",
}


def init_ctbc_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    contact_window: int = 3,
    force_threshold_n: float = 10.0,
    ff_amplitude_rad: float = 1.70,
    ff_x_m: float = 0.02,
    ff_lift_m: float = 0.02,
    ff_period_s: float = 0.6,
    ff_rise_ratio: float = 0.35,
    ff_hold_ratio: float = 0.0,
    ff_wheel_action: float = 0.0,
    ff_start_iter: int = 0,
    ann_start_iter: int = 500,
    ann_end_iter: int = 1500,
    phantom_trigger_iter: int = 0,
    allow_bilateral_trigger: bool = False,
    profile_path: Path | str | None = None,
) -> None:
    """startup 事件：在 env 上挂载 CTBC 状态机。参数语义与 stair 线 `init_stair_climb_state` 相同。"""
    del env_ids
    if getattr(env, CTBC_STATE_ATTR, None) is not None:
        return
    control_dt = float(env.physics_dt) * int(env.cfg.decimation)
    state = StairClimbState(
        num_envs=env.num_envs,
        device=env.device,
        contact_window=contact_window,
        force_threshold_n=force_threshold_n,
        ff_amplitude_rad=ff_amplitude_rad,
        ff_x_m=ff_x_m,
        ff_lift_m=ff_lift_m,
        ff_period_s=ff_period_s,
        ff_rise_ratio=ff_rise_ratio,
        ff_hold_ratio=ff_hold_ratio,
        ff_wheel_action=ff_wheel_action,
        control_dt=control_dt,
        ff_start_iter=ff_start_iter,
        ann_start_iter=ann_start_iter,
        ann_end_iter=ann_end_iter,
        phantom_trigger_iter=phantom_trigger_iter,
        allow_bilateral_trigger=allow_bilateral_trigger,
        profile_path=profile_path,
    )
    setattr(env, CTBC_STATE_ATTR, state)


def reset_ctbc_state(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
    """reset 事件：清空指定 env 的 CTBC 状态。"""
    state = getattr(env, CTBC_STATE_ATTR, None)
    if state is None:
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    state.reset(env_ids)


def _riser_contact_force_xy(
    env: ManagerBasedRlEnv,
    wheel_sensor_name: str,
    riser_sensor_name: str | None,
    riser_normal_z_max: float,
) -> torch.Tensor:
    """返回左右轮顶在地形立面上的水平接触力 [N, 2]。

    有立面传感器（带 normal 字段、逐槽位）时只累计法向接近水平的接触；没有时退化为
    轮子净接触力的水平分量，与 stair 线 `step_stair_climb_state` 的两级回退一致。
    """
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[wheel_sensor_name]
    data = sensor.data
    if data.force is None:
        wheel_xy = torch.zeros(env.num_envs, 2, device=env.device)
    else:
        wheel_xy = torch.norm(data.force[..., :2], dim=-1).reshape(env.num_envs, -1)[:, :2]

    if riser_sensor_name:
        riser: ContactSensor = env.scene[riser_sensor_name]
        riser_data = riser.data
        if riser_data.force is not None and riser_data.normal is not None:
            force = riser_data.force.reshape(env.num_envs, 2, -1, 3)
            normal = riser_data.normal.reshape(env.num_envs, 2, -1, 3)
            force_xy = torch.norm(force[..., :2], dim=-1)
            valid = torch.abs(normal[..., 2]) <= float(riser_normal_z_max)
            if riser_data.found is not None:
                valid = valid & (riser_data.found.reshape(env.num_envs, 2, -1) > 0)
            wheel_xy = torch.where(valid, force_xy, torch.zeros_like(force_xy)).sum(dim=-1)
    return torch.nan_to_num(wheel_xy, nan=0.0, posinf=0.0, neginf=0.0)


def _terrain_inactive_mask(
    env: ManagerBasedRlEnv, terrain_type_names: tuple[str, ...] | None
) -> torch.Tensor | None:
    """返回“不在允许触发 CTBC 的子地形列上”的 env 掩码；不限列或非课程地形时返回 None。"""
    if not terrain_type_names:
        return None
    terrain = getattr(env.scene, "terrain", None)
    generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    if generator is None or terrain_types is None:
        return None
    names = list(generator.sub_terrains.keys())
    allowed = [names.index(n) for n in terrain_type_names if n in names]
    types = terrain_types.to(device=env.device, dtype=torch.long)
    active = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for col in allowed:
        active |= types == col
    return ~active


def step_ctbc_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    wheel_sensor_name: str = "wheel_sensor",
    riser_sensor_name: str | None = "wheel_riser_sensor",
    riser_normal_z_max: float = 0.5,
    num_steps_per_env: int = 24,
    terrain_type_names: tuple[str, ...] | None = ("stairs_up",),
) -> None:
    """interval 事件（每控制步）：更新触发窗口、前馈相位与退火权重，并写诊断日志。

    只有 `terrain_type_names` 列（默认只有上台阶列）允许触发；其余列每步清零接触输入并复位状态，
    下台阶、斜坡、起伏上偶发的近水平接触不会引发前馈。
    """
    del env_ids
    state = getattr(env, CTBC_STATE_ATTR, None)
    if state is None:
        return
    wheel_xy = _riser_contact_force_xy(env, wheel_sensor_name, riser_sensor_name, riser_normal_z_max)
    inactive = _terrain_inactive_mask(env, terrain_type_names)
    if inactive is not None and bool(inactive.any()):
        inactive_ids = inactive.nonzero(as_tuple=False).flatten()
        wheel_xy[inactive_ids] = 0.0
        state.reset(inactive_ids)
    state.step(wheel_xy)
    iteration = int(env.common_step_counter) // max(1, int(num_steps_per_env))
    state.update_iter(iteration)

    log = env.extras.get("log") if hasattr(env, "extras") else None
    if isinstance(log, dict):
        diag = state.diag()
        for src, dst in _DIAG_KEYS.items():
            if src in diag:
                log[dst] = diag[src]


def ctbc_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """3 维 CTBC 观测：[左相位 0–1, 右相位 0–1, 触发位×退火权重]。

    未挂状态机或前馈已退火到 0（kff=0）时全 0：部署端没有状态机，这 3 维恒为 0，
    退火结束后训练侧也必须看到同样的 0，否则相位会变成只在仿真里存在的隐藏输入。
    """
    state = getattr(env, CTBC_STATE_ATTR, None)
    if state is None or float(state.kff) <= 0.0:
        return torch.zeros(env.num_envs, 3, device=env.device)
    phase = state.ff_phase.to(dtype=torch.float32)
    period = float(max(1, state.ff_period_steps))
    phase = torch.where(phase >= 0.0, phase / period, torch.zeros_like(phase))
    trigger = state.ctbc_trigger_weight().to(dtype=torch.float32).unsqueeze(-1)
    return _finite_clamp(torch.cat([phase, trigger], dim=-1))


__all__ = [
    "CTBC_STATE_ATTR",
    "ctbc_obs",
    "init_ctbc_state",
    "reset_ctbc_state",
    "step_ctbc_state",
]
