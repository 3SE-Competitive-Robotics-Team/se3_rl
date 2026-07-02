"""SE3 轮腿机器人的课程学习函数。"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import TYPE_CHECKING

import torch

from se3_train.mdp.commands import VelocityHeightCommandCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


_DEFAULT_STEPS_PER_POLICY_ITER = 64


def _curriculum_progress(
    env: ManagerBasedRlEnv,
    *,
    use_iterations: bool,
    steps_per_policy_iter: int,
    offset_iter: int = 0,
    fixed_iteration: int | None = None,
) -> int:
    """返回课程进度；recovery 任务使用 PPO iter，普通任务沿用 policy step。"""
    if fixed_iteration is not None:
        return max(0, int(fixed_iteration) - int(offset_iter))

    step = int(getattr(env, "common_step_counter", 0))
    if not use_iterations:
        return step

    steps_per_iter = max(1, int(steps_per_policy_iter))
    return max(0, step // steps_per_iter - int(offset_iter))


def commands_vel(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_stages: list[dict],
    use_iterations: bool = False,
    steps_per_policy_iter: int = _DEFAULT_STEPS_PER_POLICY_ITER,
    offset_iter: int = 0,
    fixed_iteration: int | None = None,
) -> dict[str, torch.Tensor]:
    """按课程进度阶梯式扩大速度指令范围。"""
    del env_ids
    term = env.command_manager.get_term(command_name)
    cfg: VelocityHeightCommandCfg = term.cfg  # type: ignore[assignment]
    progress = _curriculum_progress(
        env,
        use_iterations=use_iterations,
        steps_per_policy_iter=steps_per_policy_iter,
        offset_iter=offset_iter,
        fixed_iteration=fixed_iteration,
    )
    threshold_key = "iteration" if use_iterations else "step"
    for stage in velocity_stages:
        threshold = int(stage.get(threshold_key, stage.get("step", 0)))
        if progress >= threshold:
            if "lin_vel_x_range" in stage:
                cfg.lin_vel_x_range = stage["lin_vel_x_range"]
            if "ang_vel_yaw_range" in stage:
                cfg.ang_vel_yaw_range = stage["ang_vel_yaw_range"]
    return {
        "step_counter": torch.tensor(float(getattr(env, "common_step_counter", 0))),
        "progress": torch.tensor(float(progress)),
        "lin_vel_x_max": torch.tensor(cfg.lin_vel_x_range[1]),
        "ang_vel_yaw_max": torch.tensor(cfg.ang_vel_yaw_range[1]),
    }


def commands_height(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    height_stages: list[dict],
    use_iterations: bool = False,
    steps_per_policy_iter: int = _DEFAULT_STEPS_PER_POLICY_ITER,
    offset_iter: int = 0,
    interpolate: bool = False,
    fixed_iteration: int | None = None,
) -> dict[str, torch.Tensor]:
    """按课程进度逐步放开高度指令范围。"""
    del env_ids
    term = env.command_manager.get_term(command_name)
    cfg: VelocityHeightCommandCfg = term.cfg  # type: ignore[assignment]
    progress = _curriculum_progress(
        env,
        use_iterations=use_iterations,
        steps_per_policy_iter=steps_per_policy_iter,
        offset_iter=offset_iter,
        fixed_iteration=fixed_iteration,
    )
    threshold_key = "iteration" if use_iterations else "step"
    if interpolate:
        _apply_interpolated_height_stage(cfg, height_stages, progress, threshold_key)
    else:
        for stage in height_stages:
            threshold = int(stage.get(threshold_key, stage.get("step", 0)))
            if progress >= threshold:
                if "height_range" in stage:
                    cfg.height_range = stage["height_range"]
                if "standing_height_range" in stage:
                    cfg.standing_height_range = stage["standing_height_range"]
                elif "height_range" in stage:
                    cfg.standing_height_range = stage["height_range"]
    return {
        "step_counter": torch.tensor(float(getattr(env, "common_step_counter", 0))),
        "progress": torch.tensor(float(progress)),
        "height_min": torch.tensor(cfg.height_range[0]),
        "height_max": torch.tensor(cfg.height_range[1]),
        "standing_height_min": torch.tensor(cfg.standing_height_range[0]),
        "standing_height_max": torch.tensor(cfg.standing_height_range[1]),
    }


def _apply_interpolated_height_stage(
    cfg: VelocityHeightCommandCfg,
    height_stages: list[dict],
    progress: int,
    threshold_key: str,
) -> None:
    stages = sorted(
        height_stages,
        key=lambda stage: int(stage.get(threshold_key, stage.get("step", 0))),
    )
    if not stages:
        return

    if progress <= int(stages[0].get(threshold_key, stages[0].get("step", 0))):
        _apply_height_stage(cfg, stages[0])
        return

    for lower, upper in pairwise(stages):
        lower_threshold = int(lower.get(threshold_key, lower.get("step", 0)))
        upper_threshold = int(upper.get(threshold_key, upper.get("step", 0)))
        if progress > upper_threshold:
            continue
        span = max(1, upper_threshold - lower_threshold)
        ratio = min(max((progress - lower_threshold) / span, 0.0), 1.0)
        cfg.height_range = _lerp_range(
            _height_range_for_stage(lower),
            _height_range_for_stage(upper),
            ratio,
        )
        cfg.standing_height_range = _lerp_range(
            _standing_height_range_for_stage(lower),
            _standing_height_range_for_stage(upper),
            ratio,
        )
        return

    _apply_height_stage(cfg, stages[-1])


def _apply_height_stage(cfg: VelocityHeightCommandCfg, stage: dict) -> None:
    if "height_range" in stage:
        cfg.height_range = stage["height_range"]
    if "standing_height_range" in stage:
        cfg.standing_height_range = stage["standing_height_range"]
    elif "height_range" in stage:
        cfg.standing_height_range = stage["height_range"]


def _height_range_for_stage(stage: dict) -> tuple[float, float]:
    return tuple(float(value) for value in stage["height_range"])  # type: ignore[return-value]


def _standing_height_range_for_stage(stage: dict) -> tuple[float, float]:
    source = stage.get("standing_height_range", stage["height_range"])
    return tuple(float(value) for value in source)  # type: ignore[return-value]


def _lerp_range(
    lower: tuple[float, float],
    upper: tuple[float, float],
    ratio: float,
) -> tuple[float, float]:
    return (
        lower[0] + (upper[0] - lower[0]) * ratio,
        lower[1] + (upper[1] - lower[1]) * ratio,
    )


def push_disturbance(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    push_stages: list[dict],
    use_iterations: bool = False,
    steps_per_policy_iter: int = _DEFAULT_STEPS_PER_POLICY_ITER,
    offset_iter: int = 0,
    fixed_iteration: int | None = None,
) -> dict[str, torch.Tensor]:
    """按训练进度逐步增大推扰动强度。

    修改 env 上存储的 push velocity_range 配置。
    push_stages 格式:
    [{"step": 0, "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}, ...]
    """
    del env_ids
    step = int(getattr(env, "common_step_counter", 0))
    progress = _curriculum_progress(
        env,
        use_iterations=use_iterations,
        steps_per_policy_iter=steps_per_policy_iter,
        offset_iter=offset_iter,
        fixed_iteration=fixed_iteration,
    )
    threshold_key = "iteration" if use_iterations else "step"
    current_max = 0.0

    for stage in push_stages:
        threshold = int(stage.get(threshold_key, stage.get("step", 0)))
        if progress >= threshold:
            velocity_range = stage["velocity_range"]
            current_max = max(max(abs(low), abs(high)) for low, high in velocity_range.values())
            env._push_velocity_range = velocity_range

    return {
        "step_counter": torch.tensor(float(step)),
        "progress": torch.tensor(float(progress)),
        "push_vel_max": torch.tensor(current_max),
    }


def terrain_levels(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    level_stages: list[dict],
    use_iterations: bool = False,
    steps_per_policy_iter: int = _DEFAULT_STEPS_PER_POLICY_ITER,
    offset_iter: int = 0,
    terrain_type_names: tuple[str, ...] | list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """按训练进度放开地形 row 难度，不依赖策略当前表现。"""
    env_ids = _as_env_id_tensor(env, env_ids)
    device = env.device
    zero = torch.tensor(0.0, device=device)
    if env_ids.numel() == 0:
        return {
            "level_mean": zero,
            "scheduled_max_level": zero,
            "scheduled_max_difficulty": zero,
            "active_rate": zero,
            "progress": zero,
        }

    terrain = getattr(env.scene, "terrain", None)
    terrain_levels_buf = getattr(terrain, "terrain_levels", None)
    terrain_origins = getattr(terrain, "terrain_origins", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    env_origins = getattr(terrain, "env_origins", None)
    if (
        not isinstance(terrain_levels_buf, torch.Tensor)
        or not isinstance(terrain_origins, torch.Tensor)
        or not isinstance(terrain_types, torch.Tensor)
        or not isinstance(env_origins, torch.Tensor)
    ):
        return {
            "level_mean": zero,
            "scheduled_max_level": zero,
            "scheduled_max_difficulty": zero,
            "active_rate": zero,
            "progress": zero,
        }

    progress = _curriculum_progress(
        env,
        use_iterations=use_iterations,
        steps_per_policy_iter=steps_per_policy_iter,
        offset_iter=offset_iter,
    )
    threshold_key = "iteration" if use_iterations else "step"
    max_level = 0
    max_difficulty = 0.0
    for stage in level_stages:
        threshold = int(stage.get(threshold_key, stage.get("step", 0)))
        if progress >= threshold:
            if "max_difficulty" in stage:
                max_difficulty = min(max(float(stage["max_difficulty"]), 0.0), 1.0)
                max_level = math.ceil(max_difficulty * int(terrain_origins.shape[0])) - 1
            else:
                max_level = int(stage["max_level"])
                max_difficulty = max_level / max(1, int(terrain_origins.shape[0]) - 1)

    max_level = max(0, min(max_level, int(terrain_origins.shape[0]) - 1))
    active = _terrain_type_mask(env, terrain_type_names, env_ids)
    active_env_ids = env_ids[active]
    if active_env_ids.numel() > 0:
        sampled_levels = torch.randint(
            0,
            max_level + 1,
            (active_env_ids.numel(),),
            device=device,
            dtype=terrain_levels_buf.dtype,
        )
        terrain_levels_buf[active_env_ids] = sampled_levels
        env_origins[active_env_ids] = terrain_origins[
            terrain_levels_buf[active_env_ids], terrain_types[active_env_ids]
        ]

    terrain_levels = getattr(terrain, "terrain_levels", terrain_levels_buf)
    level_mean = terrain_levels[env_ids].float().mean()
    return {
        "level_mean": level_mean,
        "scheduled_max_level": torch.tensor(float(max_level), device=device),
        "scheduled_max_difficulty": torch.tensor(float(max_difficulty), device=device),
        "active_rate": active.float().mean(),
        "progress": torch.tensor(float(progress), device=device),
    }


def _as_env_id_tensor(env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    """把 manager 传入的 env_ids 统一成一维 long tensor。"""
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)


def _terrain_type_mask(
    env: ManagerBasedRlEnv,
    terrain_type_names: tuple[str, ...] | list[str] | None,
    env_ids: torch.Tensor,
) -> torch.Tensor:
    """根据 terrain type 名称筛选需要参与课程推进的 env。"""
    if not terrain_type_names:
        return torch.ones(env_ids.shape, device=env.device, dtype=torch.bool)

    terrain = getattr(env.scene, "terrain", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    if not isinstance(terrain_types, torch.Tensor):
        return torch.zeros(env_ids.shape, device=env.device, dtype=torch.bool)

    cfg = getattr(terrain, "cfg", None)
    generator_cfg = getattr(cfg, "terrain_generator", None)
    sub_terrains = getattr(generator_cfg, "sub_terrains", {}) or {}
    selected = {str(name) for name in terrain_type_names}
    mask = torch.zeros(env_ids.shape, device=env.device, dtype=torch.bool)
    env_terrain_types = terrain_types.to(device=env.device)[env_ids]
    for terrain_index, terrain_name in enumerate(sub_terrains):
        if str(terrain_name) in selected:
            mask = mask | (env_terrain_types == terrain_index)
    return mask


_VEL_ADAPTIVE_LIN_X_MAX_ATTR = "_vel_adaptive_lin_x_max"
_VEL_ADAPTIVE_YAW_MAX_ATTR = "_vel_adaptive_yaw_max"
_VEL_ADAPTIVE_EMA_ATTR = "_vel_adaptive_ema"


def commands_vel_adaptive(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    *,
    lin_vel_x_step: float = 0.2,
    ang_vel_yaw_step: float = 1.0,
    max_lin_vel_x: float = 2.4,
    max_ang_vel_yaw: float = 12.0,
    init_lin_vel_x: float = 0.0,
    init_ang_vel_yaw: float = 0.0,
    advance_threshold: float = 0.5,
    ema_alpha: float = 0.05,
) -> dict[str, torch.Tensor]:
    """ETH 风格的自适应速度指令课程：用小步长丝滑推进速度范围。

    从 vx=0（纯静站）起步，用 Locomotion/tracking_lin_vel_reward_all 的 EMA
    评估策略是否适应了当前速度。EMA > advance_threshold 时小步扩大速度范围，
    只扩不缩。

    Locomotion/tracking_lin_vel_reward_all 是未乘权重的 exp 核均值，值域 (0, 1]，
    按 locomotion mask（排除跳跃）取全体均值，不按 moving（|cmd_x|>=0.2）过滤。
    必须用这个不依赖 moving 的版本：初始 lin_vel_x_range=(0,0) 时所有 env 的
    cmd_mag 恒为 0，如果按 moving 过滤会导致该指标恒为 0、EMA 永远追不上
    threshold，课程永久卡死在初始范围（2026-07 曾复现，807 iteration 内
    vel_lin_x_max 无变化）。用全体均值后，cmd=0 阶段该值反映 sigma_stand
    打分下的站立稳定度，天然给出连续、非零的推进信号。

    第二个独立死锁（同一次排查发现）：`env.extras["log"]` 在 ManagerBasedRlEnv.step()
    开头逐 step 清空（`self.extras["log"] = dict()`），而 tracking_lin_vel 只在
    `_should_log_step` 命中时（默认每 64 step 一次）才写入这个 key；本函数在
    `_reset_idx` 里对几乎每个 step 都会被调用（2048 个并行 env 几乎每 step 都有
    env 到期重置）。如果对缺失 key 用 `0.0` 兜底，63/64 次读到的都是假的 0，
    EMA 会被稀释到大约 `真实值/64`——σ_move=0.08、reward 已经接近满分 1.0 时，
    稀释后的稳态 EMA 也只有约 0.015，永远追不上任何有意义的 threshold（哪怕
    threshold 定得再低）。这不是"及格线"或"moving mask"问题，是把一个为降低
    TensorBoard 写盘开销而设的节流开关，误用成了课程判据的采样开关。
    修复方式：区分"key 缺失（这个 step 没有新样本）"和"key 存在但值为 0（真的
    跟踪失败）"，缺失时跳过本次 EMA 更新，只用真正刷新过的样本推进 EMA。

    当前 σ_move=0.08 时，threshold=0.5 约对应跟踪误差 0.24 m/s，
    threshold=0.6 约对应 0.19 m/s（该换算只在 moving 段严格成立，
    cmd=0 段走的是更松的 sigma_stand）。
    """
    del env_ids
    term = env.command_manager.get_term(command_name)
    cfg: VelocityHeightCommandCfg = term.cfg  # type: ignore[assignment]

    if not hasattr(env, _VEL_ADAPTIVE_LIN_X_MAX_ATTR):
        setattr(env, _VEL_ADAPTIVE_LIN_X_MAX_ATTR, float(init_lin_vel_x))
        setattr(env, _VEL_ADAPTIVE_YAW_MAX_ATTR, float(init_ang_vel_yaw))
        setattr(env, _VEL_ADAPTIVE_EMA_ATTR, 0.0)

    lin_x_max = float(getattr(env, _VEL_ADAPTIVE_LIN_X_MAX_ATTR))
    yaw_max = float(getattr(env, _VEL_ADAPTIVE_YAW_MAX_ATTR))
    ema = float(getattr(env, _VEL_ADAPTIVE_EMA_ATTR))

    # 从 step 级日志读取跟踪奖励（Episode_Reward 在 curriculum 之后才写入）。
    # log 每 step 清空，tracking_lin_vel 每 _should_log_step 个 step 才写一次，
    # 所以必须用 None 兜底区分“没有新样本”和“样本值真的是 0”，否则会被
    # 未刷新的 step 稀释，EMA 永远追不上 threshold。
    log = getattr(env, "extras", {}).get("log", {})
    tracking_lin_vel = log.get("Locomotion/tracking_lin_vel_reward_all", None)

    # advance 判定必须和 EMA 刷新绑在一起，不能每次 compute() 调用都判一次。
    # commands_vel_adaptive 由 _reset_idx 触发，2048 个并行 env 几乎每个
    # global step 都有到期重置，也就几乎每 step 都会调用一次本函数；但
    # tracking_lin_vel 只在 _should_log_step 命中时（默认每 64 step 一次）
    # 才会刷新。如果 advance 判定不绑定"这次是不是刚刷新",同一个未变化的
    # ema 会在刷新间隔内的每一次 reset 调用上都重新判一次"> threshold",
    # 在一个 64-step 窗口内连续触发几十次 +lin_vel_x_step,一步顶原设计的
    # 十几步（2026-07 复测复现：iteration 21 单窗口内从 0 直接跳到 2.18,
    # 下一次刷新就顶到 max_lin_vel_x=2.4,和 commit message 里"小步丝滑
    # 推进,约需 10 步"的设计意图完全不符,课程等于形同虚设）。
    if tracking_lin_vel is not None:
        ema = (1.0 - ema_alpha) * ema + ema_alpha * float(tracking_lin_vel)
        setattr(env, _VEL_ADAPTIVE_EMA_ATTR, ema)

        if ema > float(advance_threshold):
            lin_x_max = min(lin_x_max + float(lin_vel_x_step), float(max_lin_vel_x))
            yaw_max = min(yaw_max + float(ang_vel_yaw_step), float(max_ang_vel_yaw))
            setattr(env, _VEL_ADAPTIVE_LIN_X_MAX_ATTR, lin_x_max)
            setattr(env, _VEL_ADAPTIVE_YAW_MAX_ATTR, yaw_max)

    cfg.lin_vel_x_range = (-lin_x_max, lin_x_max)
    cfg.ang_vel_yaw_range = (-yaw_max, yaw_max)

    return {
        "vel_lin_x_max": torch.tensor(lin_x_max),
        "vel_yaw_max": torch.tensor(yaw_max),
        "vel_ema": torch.tensor(ema),
        "vel_tracking_lin_vel": torch.tensor(
            float(tracking_lin_vel) if tracking_lin_vel is not None else float("nan")
        ),
    }
