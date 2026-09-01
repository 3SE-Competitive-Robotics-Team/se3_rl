"""Recovery 起身的外部扭矩引导。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
    from mjlab.entity.entity import Entity
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


_TORQUE_ASSIST_CONTRACT_VERSION = "se3.recovery-torque-assist.v3"

RECOVERY_TORQUE_ASSIST_STATE_DIM = 3
"""critic 特权观测维度：[selected, active, remaining_budget]。"""


@dataclass(kw_only=True)
class RecoveryTorqueAssistCfg:
    """按当前机身姿态触发的外部扭矩引导契约。"""

    enabled: bool = True
    torque_nm: float = 20.0
    min_effective_torque_nm: float = 19.0
    body_name: str = "base_link"
    body_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    exit_upright_angle_deg: float = 30.0
    max_assist_time_s: float = 3.0
    probability_start: float = 1.0
    hold_iters: int = 200
    end_iter: int = 500
    probability_end: float = 0.0
    steps_per_policy_iter: int = 24
    fixed_iteration: int | None = None

    def validate(self) -> None:
        """拒绝会改变方向、姿态门或课程语义的非法配置。"""

        if self.torque_nm <= 0.0:
            raise ValueError("Recovery torque assist 的 torque_nm 必须为正数")
        if self.min_effective_torque_nm <= 0.0:
            raise ValueError("Recovery torque assist 的 min_effective_torque_nm 必须为正数")
        if self.torque_nm < self.min_effective_torque_nm:
            raise ValueError(
                "Recovery torque assist 的 torque_nm 低于扶正阈值，施力不会有物理效果："
                f"torque_nm={self.torque_nm}, "
                f"min_effective_torque_nm={self.min_effective_torque_nm}。"
                "原生扫频实测倒置姿态下 <=18 N·m 完全翻不起来、19 N·m 需 1.99 s、"
                "20 N·m 需 1.66 s，因此退火只能降采样概率，不能降力矩幅值。"
            )
        if not self.body_name:
            raise ValueError("Recovery torque assist 的 body_name 不能为空")
        axis_norm = math.sqrt(sum(float(value) ** 2 for value in self.body_axis))
        if axis_norm <= 1.0e-9:
            raise ValueError("Recovery torque assist 的 body_axis 不能为零向量")
        if not 0.0 < self.exit_upright_angle_deg < 90.0:
            raise ValueError("Recovery torque assist 的撤力角必须在 (0, 90) 度内")
        if self.max_assist_time_s <= 0.0:
            raise ValueError("Recovery torque assist 的单次辅助时限必须为正数")
        if not 0.0 <= self.probability_end <= self.probability_start <= 1.0:
            raise ValueError(
                "Recovery torque assist 概率必须满足 0 <= probability_end <= probability_start <= 1"
            )
        if self.hold_iters < 0 or self.end_iter <= self.hold_iters:
            raise ValueError("Recovery torque assist 退火区间必须满足 0 <= hold_iters < end_iter")
        if self.steps_per_policy_iter <= 0:
            raise ValueError("Recovery torque assist steps_per_policy_iter 必须为正数")
        if self.fixed_iteration is not None and self.fixed_iteration < 0:
            raise ValueError("Recovery torque assist fixed_iteration 必须非负或为 None")


def recovery_torque_assist_probability(
    current_iter: int,
    cfg: RecoveryTorqueAssistCfg,
) -> float:
    """计算 episode 获得完整力矩引导的采样概率。

    退火只作用在概率上：被采样到的 episode 始终拿到 ``torque_nm`` 满幅力矩。
    低于扶正阈值的档位在物理上完全无效，阶梯降幅方案实测在首次降档即失效。
    """

    if not cfg.enabled:
        return 0.0
    if current_iter < cfg.hold_iters:
        return float(cfg.probability_start)
    if current_iter >= cfg.end_iter:
        return float(cfg.probability_end)
    progress = (current_iter - cfg.hold_iters) / max(cfg.end_iter - cfg.hold_iters, 1)
    return float(cfg.probability_start + progress * (cfg.probability_end - cfg.probability_start))


def recovery_torque_assist_contract_info(env_cfg: object) -> tuple[str, str] | None:
    """返回 opt-in 外部扭矩引导的稳定配置指纹。"""

    actions = getattr(env_cfg, "actions", None)
    if not isinstance(actions, dict):
        return None
    action_cfg = actions.get("delayed_action")
    assist_cfg = getattr(action_cfg, "recovery_torque_assist", None)
    if not isinstance(assist_cfg, RecoveryTorqueAssistCfg):
        return None

    mode = "train" if assist_cfg.enabled else "assist-off"
    payload = {
        "version": _TORQUE_ASSIST_CONTRACT_VERSION,
        "mode": mode,
        "assist": asdict(assist_cfg),
        "policy_action_transform": "identity",
        "application_order": "policy-action->external-wrench->physics-step",
        "assist_budget_scope": "per-fall",
        "critic_state_dim": RECOVERY_TORQUE_ASSIST_STATE_DIM,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"recovery-torque-assist-v3-{mode}", hashlib.sha256(encoded).hexdigest()


class RecoveryTorqueAssistController:
    """倾角超过直立带时，逐 physics substep 写入 body-local +Y 扭矩。"""

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        cfg: RecoveryTorqueAssistCfg,
        entity: Entity,
    ) -> None:
        cfg.validate()
        self._env = env
        self.cfg = cfg
        self._entity = entity

        body_ids, body_names = entity.find_bodies(cfg.body_name, preserve_order=True)
        if len(body_ids) != 1:
            raise ValueError(
                "Recovery torque assist 必须精确匹配一个施力刚体，"
                f"pattern={cfg.body_name!r}, matches={body_names}"
            )
        self._body_ids = body_ids

        axis_b = torch.tensor(cfg.body_axis, device=env.device, dtype=torch.float32)
        self._axis_b = axis_b / torch.linalg.vector_norm(axis_b)
        self._axis_b_batch = self._axis_b.repeat(env.num_envs, 1)
        self._torques_w = torch.zeros(env.num_envs, 1, 3, device=env.device)

        self._selected = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._active = torch.zeros_like(self._selected)
        self._near_upright = torch.zeros_like(self._selected)
        self._timed_out = torch.zeros_like(self._selected)
        self._invalid_orientation = torch.zeros_like(self._selected)
        self._elapsed_substeps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self._rearm_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self._applied_torque_nm = torch.zeros(env.num_envs, device=env.device)
        self._critic_state = torch.zeros(
            env.num_envs,
            RECOVERY_TORQUE_ASSIST_STATE_DIM,
            device=env.device,
        )

        self._exit_pg_z = -math.cos(math.radians(float(cfg.exit_upright_angle_deg)))
        self._max_substeps = max(
            1,
            math.ceil(float(cfg.max_assist_time_s) / float(env.physics_dt)),
        )

    @property
    def selected(self) -> torch.Tensor:
        """本 episode 是否被课程采样为满幅扭矩辅助样本。"""

        return self._selected

    @property
    def active(self) -> torch.Tensor:
        """当前 physics substep 是否正在施加扭矩。"""

        return self._active

    @property
    def applied_torque_nm(self) -> torch.Tensor:
        """当前实际写入的世界系扭矩模长。"""

        return self._applied_torque_nm

    @property
    def torques_w(self) -> torch.Tensor:
        """当前写入 base body 的世界系扭矩向量。"""

        return self._torques_w[:, 0, :]

    @property
    def critic_state(self) -> torch.Tensor:
        """返回形状 ``(num_envs, 3)`` 的 critic 特权辅助状态。

        依次为本 episode 是否被采样、当前是否在施力、本次跌倒剩余辅助预算比例。
        辅助对 actor 不可见，但它同时改变动力学与回报，所以必须让 critic 看见，
        否则同一观测会在「有外力」和「无外力」两套动力学下共用一个 value 目标。
        """

        return self._critic_state

    def reset(self, env_ids: torch.Tensor) -> None:
        """按当前课程概率采样 episode，姿态门与计时在 substep 实时推进。"""

        env_ids = env_ids.to(device=self._env.device, dtype=torch.long)
        probability = self._current_probability()
        selected = torch.rand(env_ids.numel(), device=self._env.device) < float(probability)

        self._selected[env_ids] = selected
        self._active[env_ids] = False
        self._near_upright[env_ids] = False
        self._timed_out[env_ids] = False
        self._invalid_orientation[env_ids] = False
        self._elapsed_substeps[env_ids] = 0
        self._rearm_count[env_ids] = 0
        self._applied_torque_nm[env_ids] = 0.0
        self._critic_state[env_ids] = 0.0
        self._critic_state[env_ids, 0] = selected.to(dtype=self._critic_state.dtype)
        self._torques_w[env_ids] = 0.0
        self._entity.write_external_wrench_to_sim(
            forces=None,
            torques=self._torques_w[env_ids],
            env_ids=env_ids,
            body_ids=self._body_ids,
        )

    def apply(self) -> None:
        """在 sim.step 前刷新 body-local 扭矩方向并写入 xfrc_applied。"""

        probability = self._current_probability()
        curriculum_enabled = probability > 0.0
        scheduled_torque_nm = float(self.cfg.torque_nm) if curriculum_enabled else 0.0
        projected_gravity = self._entity.data.projected_gravity_b
        pg_z = projected_gravity[:, 2]
        orientation_finite = torch.isfinite(pg_z) & torch.isfinite(
            self._entity.data.root_link_quat_w
        ).all(dim=1)
        self._near_upright[:] = orientation_finite & (pg_z <= self._exit_pg_z)

        # 辅助时限按「本次跌倒」计，而不是整个 episode 累计后锁死：回到直立带即
        # 结束本次辅助并清零计时与超时闩，下一次跌出直立带重新获得完整预算。
        rearm = self._near_upright & ((self._elapsed_substeps > 0) | self._timed_out)
        self._rearm_count += rearm.to(dtype=torch.long)
        self._elapsed_substeps[self._near_upright] = 0
        self._timed_out &= ~self._near_upright

        wants_assist = self._selected & ~self._near_upright & curriculum_enabled
        timed_out = wants_assist & (self._elapsed_substeps >= self._max_substeps)
        invalid_orientation = self._selected & ~orientation_finite

        self._timed_out |= timed_out
        self._invalid_orientation |= invalid_orientation
        self._active[:] = (
            wants_assist & orientation_finite & ~self._timed_out & ~self._invalid_orientation
        )

        axis_w = quat_apply(self._entity.data.root_link_quat_w, self._axis_b_batch)
        axis_w = torch.nan_to_num(axis_w, nan=0.0, posinf=0.0, neginf=0.0)
        torque_w = axis_w * scheduled_torque_nm
        self._torques_w[:, 0, :] = torch.where(
            self._active.unsqueeze(1),
            torque_w,
            torch.zeros_like(torque_w),
        )
        self._entity.write_external_wrench_to_sim(
            forces=None,
            torques=self._torques_w,
            body_ids=self._body_ids,
        )
        self._applied_torque_nm[:] = torch.linalg.vector_norm(
            self._torques_w[:, 0, :],
            dim=1,
        )
        self._elapsed_substeps += self._active.to(dtype=torch.long)
        self._refresh_critic_state(curriculum_enabled)

    def log_diagnostics(self) -> None:
        """每个 policy step 记录采样、施力、撤力与泄漏诊断。"""

        extras = getattr(self._env, "extras", None)
        if not isinstance(extras, dict):
            return
        log = extras.setdefault("log", {})
        if not isinstance(log, dict):
            return

        selected_count = self._selected.float().sum()
        active_count = self._active.float().sum()
        selected_denominator = torch.clamp(selected_count, min=1.0)
        active_denominator = torch.clamp(active_count, min=1.0)
        probability = self._current_probability()
        curriculum_enabled = probability > 0.0
        scheduled_torque_nm = float(self.cfg.torque_nm) if curriculum_enabled else 0.0
        active_torque_nm = self._active.float() * scheduled_torque_nm
        upright_leakage = self._active & self._near_upright
        selected_non_upright = self._selected & ~self._near_upright
        non_upright_unassisted = selected_non_upright & ~self._active

        log.update(
            {
                "Recovery/diag_torque_assist_probability": probability,
                "Recovery/diag_torque_assist_selected_rate": self._selected.float().mean(),
                "Recovery/diag_torque_assist_scheduled_nm": scheduled_torque_nm,
                "Recovery/diag_torque_assist_curriculum_enabled": float(curriculum_enabled),
                "Recovery/diag_torque_assist_near_upright_rate": (
                    self._near_upright.float().mean()
                ),
                "Recovery/diag_torque_assist_active_rate": self._active.float().mean(),
                "Recovery/diag_torque_assist_active_selected_ratio": (
                    active_count / selected_denominator
                ),
                "Recovery/diag_torque_assist_mean_nm": (
                    active_torque_nm.sum() / active_denominator
                ),
                "Recovery/diag_torque_assist_max_nm": active_torque_nm.max(),
                "Recovery/diag_torque_assist_elapsed_s": (
                    (self._elapsed_substeps.float() * self._active.float()).sum()
                    * float(self._env.physics_dt)
                    / active_denominator
                ),
                "Recovery/diag_torque_assist_remaining_budget": (
                    self._critic_state[:, 2].sum() / selected_denominator
                ),
                "Recovery/diag_torque_assist_rearm_count_mean": (
                    self._rearm_count.float().sum() / selected_denominator
                ),
                "Recovery/diag_torque_assist_withdrawn_upright_rate": (
                    (self._selected & self._near_upright).float().mean()
                ),
                "Recovery/diag_torque_assist_non_upright_unassisted_rate": (
                    non_upright_unassisted.float().mean()
                ),
                "Recovery/diag_torque_assist_timeout_rate": self._timed_out.float().mean(),
                "Recovery/diag_torque_assist_invalid_orientation_rate": (
                    self._invalid_orientation.float().mean()
                ),
                "Recovery/diag_torque_assist_upright_leakage_rate": (
                    upright_leakage.float().mean()
                ),
            }
        )

    def _refresh_critic_state(self, curriculum_enabled: bool) -> None:
        """刷新 critic 特权状态：采样标志、施力标志与本次跌倒剩余预算。"""

        available = self._selected & curriculum_enabled
        remaining = 1.0 - self._elapsed_substeps.float() / float(self._max_substeps)
        self._critic_state[:, 0] = self._selected.to(dtype=self._critic_state.dtype)
        self._critic_state[:, 1] = self._active.to(dtype=self._critic_state.dtype)
        self._critic_state[:, 2] = remaining.clamp(0.0, 1.0) * available.to(
            dtype=self._critic_state.dtype
        )

    def _current_iteration(self) -> int:
        """按 policy step 计数换算 PPO iteration。"""

        if self.cfg.fixed_iteration is not None:
            return int(self.cfg.fixed_iteration)
        step = int(getattr(self._env, "common_step_counter", 0))
        return step // int(self.cfg.steps_per_policy_iter)

    def _current_probability(self) -> float:
        """返回当前 PPO iteration 的 episode 采样概率。"""

        return recovery_torque_assist_probability(self._current_iteration(), self.cfg)


__all__ = [
    "RECOVERY_TORQUE_ASSIST_STATE_DIM",
    "RecoveryTorqueAssistCfg",
    "RecoveryTorqueAssistController",
    "recovery_torque_assist_contract_info",
    "recovery_torque_assist_probability",
]
