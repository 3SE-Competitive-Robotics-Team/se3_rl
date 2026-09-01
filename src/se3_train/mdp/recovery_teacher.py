"""Recovery 倒置起身的动作层 scripted teacher。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


_TEACHER_CONTRACT_VERSION = "se3.recovery-scripted-teacher.v1"


@dataclass(kw_only=True)
class RecoveryTeacherCfg:
    """倒置起身 teacher 的显式动作契约。"""

    enabled: bool = True
    enter_pg_z: float = 0.80
    exit_pg_z: float = -0.80
    exit_stable_steps: int = 20
    max_steps: int = 450
    hip_sweep_speed_rad_s: float = math.pi
    hip_sweep_max_rad: float = 2.0 * math.pi
    max_abs_action: float = 32.0
    alpha_start: float = 0.85
    hold_iters: int = 200
    end_iter: int = 500
    alpha_end: float = 0.0
    steps_per_policy_iter: int = 24
    fixed_iteration: int | None = None
    direction_deadband: float = 1.0e-5
    sweep_action_indices: tuple[int, int] = (0, 2)
    hard_zero_wheel_indices: tuple[int, int] = (4, 5)

    def validate(self) -> None:
        """拒绝会让 teacher 时序或动作语义静默漂移的配置。"""

        if not -1.0 <= self.exit_pg_z < self.enter_pg_z <= 1.0:
            raise ValueError(
                "Recovery teacher projected_gravity 阈值非法: "
                f"exit={self.exit_pg_z}, enter={self.enter_pg_z}"
            )
        if self.exit_stable_steps <= 0:
            raise ValueError("Recovery teacher exit_stable_steps 必须为正数")
        if self.max_steps <= 0:
            raise ValueError("Recovery teacher max_steps 必须为正数")
        if self.hip_sweep_speed_rad_s <= 0.0 or self.hip_sweep_max_rad <= 0.0:
            raise ValueError("Recovery teacher 扫动速度与幅度必须为正数")
        if self.max_abs_action <= 0.0:
            raise ValueError("Recovery teacher max_abs_action 必须为正数")
        if not 0.0 <= self.alpha_end <= self.alpha_start <= 1.0:
            raise ValueError("Recovery teacher alpha 必须满足 0 <= alpha_end <= alpha_start <= 1")
        if self.hold_iters < 0 or self.end_iter <= self.hold_iters:
            raise ValueError("Recovery teacher 退火区间必须满足 0 <= hold_iters < end_iter")
        if self.steps_per_policy_iter <= 0:
            raise ValueError("Recovery teacher steps_per_policy_iter 必须为正数")
        if tuple(self.sweep_action_indices) != (0, 2):
            raise ValueError("Recovery teacher 只允许扫动 lf0/rf0，即 action dim 0/2")
        if tuple(self.hard_zero_wheel_indices) != (4, 5):
            raise ValueError("Recovery teacher 只允许硬置零左右轮，即 action dim 4/5")


def recovery_teacher_alpha(current_iter: int, cfg: RecoveryTeacherCfg) -> float:
    """按用户定稿的 hold 后单段直线计算 teacher 权重。"""

    if not cfg.enabled:
        return 0.0
    if current_iter < cfg.hold_iters:
        return float(cfg.alpha_start)
    if current_iter >= cfg.end_iter:
        return float(cfg.alpha_end)
    progress = (current_iter - cfg.hold_iters) / max(cfg.end_iter - cfg.hold_iters, 1)
    return float(cfg.alpha_start + progress * (cfg.alpha_end - cfg.alpha_start))


def recovery_teacher_contract_info(env_cfg: object) -> tuple[str, str] | None:
    """返回 opt-in teacher 的模式名与动作/观测契约指纹。"""

    actions = getattr(env_cfg, "actions", None)
    if not isinstance(actions, dict):
        return None
    action_cfg = actions.get("delayed_action")
    teacher_cfg = getattr(action_cfg, "recovery_teacher", None)
    if not isinstance(teacher_cfg, RecoveryTeacherCfg):
        return None

    observations = getattr(env_cfg, "observations", None)
    last_action_functions: dict[str, str | None] = {}
    if isinstance(observations, dict):
        for group_name in ("actor", "critic"):
            group = observations.get(group_name)
            terms = getattr(group, "terms", None)
            func_name = None
            if isinstance(terms, dict) and "last_actions" in terms:
                func = terms["last_actions"].func
                func_name = f"{func.__module__}.{func.__qualname__}"
            last_action_functions[group_name] = func_name

    mode = "train" if teacher_cfg.enabled else "teacher-off"
    payload = {
        "version": _TEACHER_CONTRACT_VERSION,
        "mode": mode,
        "teacher": asdict(teacher_cfg),
        "last_action_functions": last_action_functions,
        "transform_order": "clip->teacher->ctbc->delay",
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"recovery-scripted-teacher-v1-{mode}", hashlib.sha256(encoded).hexdigest()


class RecoveryTeacherController:
    """维护 per-env teacher 触发、方向、退出与超时状态。"""

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        cfg: RecoveryTeacherCfg,
        leg_action_scales: torch.Tensor,
    ) -> None:
        cfg.validate()
        self._env = env
        self.cfg = cfg
        self._leg_action_scales = leg_action_scales.to(device=env.device, dtype=torch.float32)
        if self._leg_action_scales.shape != (4,):
            raise ValueError(
                "Recovery teacher 要求 4 维腿 action scale，"
                f"实际为 {tuple(self._leg_action_scales.shape)}"
            )
        self._active = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self._exit_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self._timed_out = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._direction = torch.ones(env.num_envs, device=env.device)
        self._sweep_indices = torch.tensor(
            cfg.sweep_action_indices,
            device=env.device,
            dtype=torch.long,
        )
        self._wheel_indices = torch.tensor(
            cfg.hard_zero_wheel_indices,
            device=env.device,
            dtype=torch.long,
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        """按 episode reset 清空 teacher 锁存状态。"""

        self._active[env_ids] = False
        self._steps[env_ids] = 0
        self._exit_count[env_ids] = 0
        self._timed_out[env_ids] = False
        self._direction[env_ids] = 1.0

    def transform(
        self,
        policy_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """把策略动作转换为 teacher-assisted 动作，返回完整诊断状态。"""

        current_iter = self._current_iteration()
        alpha = recovery_teacher_alpha(current_iter, self.cfg)
        teacher_action = torch.zeros_like(policy_action)
        if alpha <= 0.0:
            self._active.zero_()
            self._steps.zero_()
            self._exit_count.zero_()
            self._log_diagnostics(policy_action, teacher_action, policy_action, alpha)
            return policy_action, teacher_action, self._active.clone(), alpha

        robot = self._env.scene["robot"]
        projected_gravity = robot.data.projected_gravity_b
        pg_x = projected_gravity[:, 0]
        pg_z = projected_gravity[:, 2]
        eligible = self._recovery_eligible_mask()
        enter = eligible & (pg_z > float(self.cfg.enter_pg_z)) & ~self._timed_out
        newly_entered = enter & ~self._active
        if newly_entered.any():
            direction = torch.where(
                pg_x >= -float(self.cfg.direction_deadband),
                torch.ones_like(pg_x),
                -torch.ones_like(pg_x),
            )
            self._direction[newly_entered] = direction[newly_entered]

        candidate_active = (self._active | enter) & eligible
        self._exit_count[:] = torch.where(
            candidate_active & (pg_z < float(self.cfg.exit_pg_z)),
            self._exit_count + 1,
            torch.zeros_like(self._exit_count),
        )
        exit_pose = self._exit_count >= int(self.cfg.exit_stable_steps)
        timed_out_now = candidate_active & (self._steps >= int(self.cfg.max_steps))
        self._timed_out |= timed_out_now
        self._active[:] = candidate_active & ~exit_pose & ~timed_out_now
        self._steps[:] = torch.where(
            self._active,
            self._steps + 1,
            torch.zeros_like(self._steps),
        )
        self._exit_count[:] = torch.where(
            self._active,
            self._exit_count,
            torch.zeros_like(self._exit_count),
        )

        hip_phase = (
            self._steps.to(dtype=torch.float32)
            * float(self._env.step_dt)
            * float(self.cfg.hip_sweep_speed_rad_s)
        ).clamp(max=float(self.cfg.hip_sweep_max_rad))
        for action_idx in self.cfg.sweep_action_indices:
            teacher_action[:, action_idx] = (
                self._direction * hip_phase / self._leg_action_scales[action_idx]
            )
        teacher_action.clamp_(-float(self.cfg.max_abs_action), float(self.cfg.max_abs_action))

        transformed = policy_action.clone()
        blend = self._active.to(dtype=policy_action.dtype).unsqueeze(1) * float(alpha)
        transformed[:, self._sweep_indices] = torch.lerp(
            policy_action[:, self._sweep_indices],
            teacher_action[:, self._sweep_indices],
            blend,
        )
        transformed[:, self._wheel_indices] = torch.where(
            self._active.unsqueeze(1),
            torch.zeros_like(transformed[:, self._wheel_indices]),
            transformed[:, self._wheel_indices],
        )
        self._log_diagnostics(policy_action, teacher_action, transformed, alpha)
        return transformed, teacher_action, self._active.clone(), alpha

    def _current_iteration(self) -> int:
        """按 policy step 计数换算当前 PPO iteration，不乘 control decimation。"""

        if self.cfg.fixed_iteration is not None:
            return max(0, int(self.cfg.fixed_iteration))
        step = int(getattr(self._env, "common_step_counter", 0))
        return step // int(self.cfg.steps_per_policy_iter)

    def _recovery_eligible_mask(self) -> torch.Tensor:
        """仅允许 recovery active env 进入 teacher。"""

        mask = getattr(self._env, "_recovery_reset_mask", None)
        if isinstance(mask, torch.Tensor) and mask.shape == self._active.shape:
            return mask.to(device=self._env.device, dtype=torch.bool)
        return torch.zeros_like(self._active)

    def _log_diagnostics(
        self,
        policy_action: torch.Tensor,
        teacher_action: torch.Tensor,
        processed_action: torch.Tensor,
        alpha: float,
    ) -> None:
        """记录 teacher 接管率、真实 active 时长与策略接棒差距。"""

        extras = getattr(self._env, "extras", None)
        if not isinstance(extras, dict):
            return
        log = extras.setdefault("log", {})
        if not isinstance(log, dict):
            return

        active = self._active
        if active.any():
            step_mean = self._steps[active].float().mean().item()
            sweep_error = (
                torch.abs(
                    policy_action[active][:, self._sweep_indices]
                    - teacher_action[active][:, self._sweep_indices]
                )
                .mean()
                .item()
            )
            wheel_abs = torch.abs(policy_action[active][:, self._wheel_indices]).mean().item()
            processed_sweep_abs = (
                torch.abs(processed_action[active][:, self._sweep_indices]).mean().item()
            )
            processed_wheel_abs = (
                torch.abs(processed_action[active][:, self._wheel_indices]).mean().item()
            )
            positive_direction = (self._direction[active] > 0.0).float().mean().item()
        else:
            step_mean = 0.0
            sweep_error = 0.0
            wheel_abs = 0.0
            processed_sweep_abs = 0.0
            processed_wheel_abs = 0.0
            positive_direction = 0.0
        log.update(
            {
                "Recovery/diag_teacher_active_rate": active.float().mean().item(),
                "Recovery/diag_teacher_alpha": float(alpha),
                "Recovery/diag_teacher_step_mean": step_mean,
                "Recovery/diag_teacher_policy_sweep_error": sweep_error,
                "Recovery/diag_teacher_policy_wheel_abs": wheel_abs,
                "Recovery/diag_teacher_processed_sweep_abs": processed_sweep_abs,
                "Recovery/diag_teacher_processed_wheel_abs": processed_wheel_abs,
                "Recovery/diag_teacher_direction_positive_rate": positive_direction,
                "Recovery/diag_teacher_timeout_latched_rate": self._timed_out.float().mean().item(),
            }
        )


__all__ = [
    "RecoveryTeacherCfg",
    "RecoveryTeacherController",
    "recovery_teacher_alpha",
    "recovery_teacher_contract_info",
]
