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


_TORQUE_ASSIST_CONTRACT_VERSION = "se3.recovery-torque-assist.v2"


@dataclass(kw_only=True)
class RecoveryTorqueAssistCfg:
    """按当前机身姿态触发的外部扭矩引导契约。"""

    enabled: bool = True
    torque_nm: float = 20.0
    body_name: str = "base_link"
    body_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    exit_upright_angle_deg: float = 30.0
    max_assist_time_s: float = 3.0
    decay_start_iter: int = 100
    decay_interval_iters: int = 100
    decay_step_nm: float = 5.0
    steps_per_policy_iter: int = 24
    fixed_iteration: int | None = None

    def validate(self) -> None:
        """拒绝会改变方向、姿态门或课程语义的非法配置。"""

        if self.torque_nm <= 0.0:
            raise ValueError("Recovery torque assist 的 torque_nm 必须为正数")
        if not self.body_name:
            raise ValueError("Recovery torque assist 的 body_name 不能为空")
        axis_norm = math.sqrt(sum(float(value) ** 2 for value in self.body_axis))
        if axis_norm <= 1.0e-9:
            raise ValueError("Recovery torque assist 的 body_axis 不能为零向量")
        if not 0.0 < self.exit_upright_angle_deg < 90.0:
            raise ValueError("Recovery torque assist 的撤力角必须在 (0, 90) 度内")
        if self.max_assist_time_s <= 0.0:
            raise ValueError("Recovery torque assist 的最长辅助时间必须为正数")
        if self.decay_start_iter < 0:
            raise ValueError("Recovery torque assist 首次降力矩轮次必须非负")
        if self.decay_interval_iters <= 0:
            raise ValueError("Recovery torque assist 降力矩间隔必须为正数")
        if self.decay_step_nm <= 0.0:
            raise ValueError("Recovery torque assist 每档降幅必须为正数")
        if self.steps_per_policy_iter <= 0:
            raise ValueError("Recovery torque assist steps_per_policy_iter 必须为正数")
        if self.fixed_iteration is not None and self.fixed_iteration < 0:
            raise ValueError("Recovery torque assist fixed_iteration 必须非负或为 None")


def recovery_torque_assist_nm(
    current_iter: int,
    cfg: RecoveryTorqueAssistCfg,
) -> float:
    """计算当前 PPO iteration 对所有样本施加的阶梯辅助力矩。"""

    if not cfg.enabled:
        return 0.0
    if current_iter < cfg.decay_start_iter:
        return float(cfg.torque_nm)
    decay_count = 1 + (current_iter - cfg.decay_start_iter) // cfg.decay_interval_iters
    return max(0.0, float(cfg.torque_nm) - decay_count * float(cfg.decay_step_nm))


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
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"recovery-torque-assist-v2-{mode}", hashlib.sha256(encoded).hexdigest()


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

        self._active = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._near_upright = torch.zeros_like(self._active)
        self._timed_out = torch.zeros_like(self._active)
        self._invalid_orientation = torch.zeros_like(self._active)
        self._elapsed_substeps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self._applied_torque_nm = torch.zeros(env.num_envs, device=env.device)

        self._exit_pg_z = -math.cos(math.radians(float(cfg.exit_upright_angle_deg)))
        self._max_substeps = max(
            1,
            math.ceil(float(cfg.max_assist_time_s) / float(env.physics_dt)),
        )

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

    def reset(self, env_ids: torch.Tensor) -> None:
        """重置 episode 累计状态，姿态门与当前力矩档位在 substep 实时计算。"""

        env_ids = env_ids.to(device=self._env.device, dtype=torch.long)
        self._active[env_ids] = False
        self._near_upright[env_ids] = False
        self._timed_out[env_ids] = False
        self._invalid_orientation[env_ids] = False
        self._elapsed_substeps[env_ids] = 0
        self._applied_torque_nm[env_ids] = 0.0
        self._torques_w[env_ids] = 0.0
        self._entity.write_external_wrench_to_sim(
            forces=None,
            torques=self._torques_w[env_ids],
            env_ids=env_ids,
            body_ids=self._body_ids,
        )

    def apply(self) -> None:
        """在 sim.step 前刷新 body-local 扭矩方向并写入 xfrc_applied。"""

        scheduled_torque_nm = self._current_torque_nm()
        curriculum_enabled = scheduled_torque_nm > 0.0
        projected_gravity = self._entity.data.projected_gravity_b
        pg_z = projected_gravity[:, 2]
        orientation_finite = torch.isfinite(pg_z) & torch.isfinite(
            self._entity.data.root_link_quat_w
        ).all(dim=1)
        self._near_upright[:] = orientation_finite & (pg_z <= self._exit_pg_z)
        wants_assist = ~self._near_upright & curriculum_enabled
        timed_out = wants_assist & (self._elapsed_substeps >= self._max_substeps)
        invalid_orientation = ~orientation_finite & curriculum_enabled

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

    def log_diagnostics(self) -> None:
        """每个 policy step 记录采样、施力、撤力与泄漏诊断。"""

        extras = getattr(self._env, "extras", None)
        if not isinstance(extras, dict):
            return
        log = extras.setdefault("log", {})
        if not isinstance(log, dict):
            return

        active_count = self._active.float().sum()
        active_denominator = torch.clamp(active_count, min=1.0)
        scheduled_torque_nm = self._current_torque_nm()
        curriculum_enabled = scheduled_torque_nm > 0.0
        active_torque_nm = self._active.float() * scheduled_torque_nm
        upright_leakage = self._active & self._near_upright
        non_upright_unassisted = ~self._near_upright & ~self._active

        log.update(
            {
                "Recovery/diag_torque_assist_scheduled_nm": scheduled_torque_nm,
                "Recovery/diag_torque_assist_curriculum_enabled": float(curriculum_enabled),
                "Recovery/diag_torque_assist_near_upright_rate": (
                    self._near_upright.float().mean()
                ),
                "Recovery/diag_torque_assist_active_rate": self._active.float().mean(),
                "Recovery/diag_torque_assist_mean_nm": (
                    active_torque_nm.sum() / active_denominator
                ),
                "Recovery/diag_torque_assist_max_nm": active_torque_nm.max(),
                "Recovery/diag_torque_assist_elapsed_s": (
                    (self._elapsed_substeps.float() * self._active.float()).sum()
                    * float(self._env.physics_dt)
                    / active_denominator
                ),
                "Recovery/diag_torque_assist_withdrawn_upright_rate": (
                    self._near_upright.float().mean() * float(curriculum_enabled)
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

    def _current_iteration(self) -> int:
        """按 policy step 计数换算 PPO iteration。"""

        if self.cfg.fixed_iteration is not None:
            return int(self.cfg.fixed_iteration)
        step = int(getattr(self._env, "common_step_counter", 0))
        return step // int(self.cfg.steps_per_policy_iter)

    def _current_torque_nm(self) -> float:
        """返回当前 PPO iteration 的阶梯辅助力矩。"""

        return recovery_torque_assist_nm(self._current_iteration(), self.cfg)


__all__ = [
    "RecoveryTorqueAssistCfg",
    "RecoveryTorqueAssistController",
    "recovery_torque_assist_contract_info",
    "recovery_torque_assist_nm",
]
