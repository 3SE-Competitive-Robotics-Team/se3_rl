"""倒地自启任务使用的课程函数。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.managers import ManagerTermBase

from se3_train.mdp.commands import VelocityHeightCommandTerm
from se3_train.tasks.flat.curriculums import (
    commands_height,
    commands_vel,
    commands_vel_adaptive,
    push_disturbance,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.managers.manager_base import ManagerTermBaseCfg


_CURRICULUM_METRIC_NAMES = (
    "lin_score",
    "yaw_score",
    "lin_score_all",
    "yaw_score_all",
    "ready_score",
    "action_saturation",
    "leg_torque_saturation",
    "wheel_torque_saturation",
    "leg_contact",
    "base_contact",
)


@dataclass
class _MetricWindow:
    """按 episode 等权汇总一个逐步指标。"""

    episode_score_sum: torch.Tensor
    episode_count: torch.Tensor
    sample_count: torch.Tensor

    @classmethod
    def create(cls, device: str) -> _MetricWindow:
        """在训练设备上创建无同步开销的标量累计器。"""
        return cls(
            episode_score_sum=torch.zeros((), device=device),
            episode_count=torch.zeros((), device=device, dtype=torch.long),
            sample_count=torch.zeros((), device=device, dtype=torch.long),
        )

    def add(self, sums: torch.Tensor, counts: torch.Tensor) -> None:
        """把逐环境的 episode 累计值加入当前窗口。"""
        valid = counts > 0.0
        safe_counts = torch.clamp(counts, min=1.0)
        scores = torch.where(valid, sums / safe_counts, torch.zeros_like(sums))
        self.episode_score_sum += scores.sum()
        self.episode_count += valid.sum()
        self.sample_count += torch.where(valid, counts, torch.zeros_like(counts)).sum().long()

    @property
    def mean(self) -> float:
        """返回 episode 等权均值；无样本时返回 0。"""
        episode_count = int(self.episode_count.item())
        if episode_count <= 0:
            return 0.0
        return float(self.episode_score_sum.item()) / episode_count

    @property
    def samples(self) -> int:
        """返回窗口内有效的逐步样本数。"""
        return int(self.sample_count.item())


@dataclass
class _GroupWindow:
    """一个环境组在当前评估窗口内的完整 episode 统计。"""

    episode_count: torch.Tensor
    survival_sum: torch.Tensor
    base_contact_terminations: torch.Tensor
    metrics: dict[str, _MetricWindow]

    @classmethod
    def create(cls, device: str) -> _GroupWindow:
        """在训练设备上创建一个空统计窗口。"""
        return cls(
            episode_count=torch.zeros((), device=device, dtype=torch.long),
            survival_sum=torch.zeros((), device=device),
            base_contact_terminations=torch.zeros((), device=device, dtype=torch.long),
            metrics={name: _MetricWindow.create(device) for name in _CURRICULUM_METRIC_NAMES},
        )


@dataclass
class _GroupState:
    """一个环境组的课程级别与跨窗口状态。"""

    level: float
    next_evaluation_step: int
    window: _GroupWindow
    pass_streak: int = 0
    last_metrics: dict[str, float] = field(default_factory=dict)


class GroupedRewardVelocityCurriculum(ManagerTermBase):
    """按 loco/recover 奖励分别推进速度包络。

    loco 只使用真实移动命令的跟踪分数，并要求完整存活且无 base contact；
    recover 使用包含起身阶段的整段 episode 跟踪分数和 ready 比例。两组均受
    动作与力矩饱和安全门约束，课程只升不降，且 recover 永不超过 loco。
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        params = dict(cfg.params)

        self._command_name = str(params.get("command_name", "velocity_height"))
        self._loco_group_name = str(params.get("loco_group_name", "loco"))
        self._recover_group_name = str(params.get("recover_group_name", "recover"))
        self._max_lin_vel_x = float(params.get("max_lin_vel_x", 1.89))
        self._max_ang_vel_yaw = float(params.get("max_ang_vel_yaw", 9.41))
        self._level_step = float(params.get("level_step", 0.05))
        self._evaluation_window_steps = int(params.get("evaluation_window_steps", 1000))
        self._required_passes = int(params.get("required_consecutive_windows", 2))
        self._configured_min_episodes = int(params.get("min_episodes_per_window", 32))
        self._min_tracking_samples = int(params.get("min_tracking_samples", 256))

        self._lin_score_threshold = float(params.get("lin_score_threshold", 0.80))
        self._yaw_score_threshold = float(params.get("yaw_score_threshold", 0.70))
        self._recover_ready_threshold = float(params.get("recover_ready_threshold", 0.60))
        self._loco_survival_threshold = float(params.get("loco_survival_threshold", 0.98))
        self._loco_base_contact_max = float(params.get("loco_base_contact_max", 0.02))
        self._action_saturation_max = float(params.get("action_saturation_max", 0.35))
        self._leg_torque_saturation_max = float(params.get("leg_torque_saturation_max", 0.20))
        self._wheel_torque_saturation_max = float(params.get("wheel_torque_saturation_max", 0.20))
        self._base_contact_termination_name = str(
            params.get("base_contact_termination_name", "loco_base_contact")
        )

        loco_level = float(params.get("loco_init_level", 0.15))
        recover_level = float(params.get("recover_init_level", 0.10))
        self._validate_config(loco_level=loco_level, recover_level=recover_level)

        name_to_id = getattr(env, "env_group_name_to_id", None)
        group_ids = getattr(env, "env_group_ids", None)
        if not isinstance(name_to_id, dict) or not isinstance(group_ids, torch.Tensor):
            raise RuntimeError(
                "GroupedRewardVelocityCurriculum 需要先执行 AssignEnvGroups startup event。"
            )
        missing_groups = {
            self._loco_group_name,
            self._recover_group_name,
        } - set(name_to_id)
        if missing_groups:
            raise ValueError(f"速度课程缺少环境组：{sorted(missing_groups)}")
        if self._loco_group_name == self._recover_group_name:
            raise ValueError("loco_group_name 与 recover_group_name 不能相同。")

        self._group_ids = {
            self._loco_group_name: int(name_to_id[self._loco_group_name]),
            self._recover_group_name: int(name_to_id[self._recover_group_name]),
        }
        self._group_env_counts = {
            name: int((group_ids == group_id).sum().item())
            for name, group_id in self._group_ids.items()
        }

        self._states = {
            self._loco_group_name: _GroupState(
                level=loco_level,
                next_evaluation_step=self._evaluation_window_steps,
                window=_GroupWindow.create(self.device),
            ),
            self._recover_group_name: _GroupState(
                level=recover_level,
                next_evaluation_step=self._evaluation_window_steps,
                window=_GroupWindow.create(self.device),
            ),
        }
        self._command_term = env.command_manager.get_term(self._command_name)
        if not isinstance(self._command_term, VelocityHeightCommandTerm):
            raise TypeError(
                f"{self._command_name} 必须是 VelocityHeightCommandTerm，"
                f"实际为 {type(self._command_term).__name__}。"
            )
        try:
            env.termination_manager.get_term(self._base_contact_termination_name)
        except ValueError as error:
            raise ValueError(
                f"速度课程找不到 loco base contact 终止项：{self._base_contact_termination_name}"
            ) from error

        # runner 会随机化初始 episode_length；每个 env 的首个 reset 因而不是完整 episode。
        self._has_completed_initial_fragment = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )
        env._se3_enable_command_curriculum_metrics = True
        self._apply_all_group_ranges()
        cfg.params = {}

    def _validate_config(self, *, loco_level: float, recover_level: float) -> None:
        """在训练启动时拒绝会破坏课程单调性的配置。"""
        if not (0.0 < recover_level <= loco_level <= 1.0):
            raise ValueError(
                "初始课程级别必须满足 0 < recover_init_level <= "
                f"loco_init_level <= 1，实际为 {recover_level}/{loco_level}。"
            )
        if self._max_lin_vel_x <= 0.0 or self._max_ang_vel_yaw <= 0.0:
            raise ValueError("完整速度包络上限必须为正数。")
        if not (0.0 < self._level_step <= 1.0):
            raise ValueError("level_step 必须位于 (0, 1]。")
        if self._evaluation_window_steps <= 0:
            raise ValueError("evaluation_window_steps 必须为正整数。")
        if self._required_passes <= 0:
            raise ValueError("required_consecutive_windows 必须为正整数。")
        if self._configured_min_episodes <= 0 or self._min_tracking_samples <= 0:
            raise ValueError("课程最小 episode 数和跟踪样本数必须为正整数。")

        thresholds = {
            "lin_score_threshold": self._lin_score_threshold,
            "yaw_score_threshold": self._yaw_score_threshold,
            "recover_ready_threshold": self._recover_ready_threshold,
            "loco_survival_threshold": self._loco_survival_threshold,
            "loco_base_contact_max": self._loco_base_contact_max,
            "action_saturation_max": self._action_saturation_max,
            "leg_torque_saturation_max": self._leg_torque_saturation_max,
            "wheel_torque_saturation_max": self._wheel_torque_saturation_max,
        }
        invalid = {name: value for name, value in thresholds.items() if not 0.0 <= value <= 1.0}
        if invalid:
            raise ValueError(f"课程阈值必须位于 [0, 1]：{invalid}")

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | slice | None,
    ) -> dict[str, float]:
        """读取刚结束的 episode，并在统计窗口达标时小步升级。"""
        if env is not self._env:
            raise RuntimeError("速度课程被绑定到了不同的环境实例。")

        ids = self._resolve_env_ids(env_ids)
        completed_ids = ids[env.episode_length_buf[ids] > 0]
        if completed_ids.numel() > 0:
            complete_episode_ids = completed_ids[
                self._has_completed_initial_fragment[completed_ids]
            ]
            if complete_episode_ids.numel() > 0:
                self._collect_completed_episodes(complete_episode_ids)
            self._has_completed_initial_fragment[completed_ids] = True
        self._clear_episode_metrics(ids)

        evaluation_events = {
            self._loco_group_name: False,
            self._recover_group_name: False,
        }
        upgrade_events = dict(evaluation_events)
        safety_blocks = dict(evaluation_events)
        step = int(getattr(env, "common_step_counter", 0))

        for group_name in (self._loco_group_name, self._recover_group_name):
            state = self._states[group_name]
            if self._group_env_counts[group_name] <= 0:
                continue
            min_episodes = min(
                self._configured_min_episodes,
                self._group_env_counts[group_name],
            )
            if step < state.next_evaluation_step:
                continue
            if int(state.window.episode_count.item()) < min_episodes:
                continue

            evaluation_events[group_name] = True
            passed, performance_ok, safety_ok = self._evaluate_group(group_name)
            if passed:
                state.pass_streak = min(state.pass_streak + 1, self._required_passes)
            else:
                state.pass_streak = 0
            safety_blocks[group_name] = performance_ok and not safety_ok

            if state.pass_streak >= self._required_passes:
                upper_level = (
                    1.0
                    if group_name == self._loco_group_name
                    else self._states[self._loco_group_name].level
                )
                next_level = min(state.level + self._level_step, upper_level, 1.0)
                if next_level > state.level + 1.0e-9:
                    state.level = next_level
                    state.pass_streak = 0
                    upgrade_events[group_name] = True

            state.window = _GroupWindow.create(self.device)
            while state.next_evaluation_step <= step:
                state.next_evaluation_step += self._evaluation_window_steps

        recover_state = self._states[self._recover_group_name]
        loco_state = self._states[self._loco_group_name]
        recover_state.level = min(recover_state.level, loco_state.level)
        if any(upgrade_events.values()):
            self._apply_all_group_ranges()

        return self._log_state(
            evaluation_events=evaluation_events,
            upgrade_events=upgrade_events,
            safety_blocks=safety_blocks,
        )

    def _resolve_env_ids(
        self,
        env_ids: torch.Tensor | slice | None,
    ) -> torch.Tensor:
        """把 curriculum manager 的 env_ids 转为一维 long tensor。"""
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        return env_ids.to(device=self.device, dtype=torch.long).reshape(-1)

    def _collect_completed_episodes(self, env_ids: torch.Tensor) -> None:
        """按环境组消费完整 episode 指标。"""
        all_group_ids = self._env.env_group_ids
        max_episode_length = max(1, int(self._env.max_episode_length))
        base_contact_termination = self._env.termination_manager.get_term(
            self._base_contact_termination_name
        )

        for group_name, group_id in self._group_ids.items():
            group_env_ids = env_ids[all_group_ids[env_ids] == group_id]
            if group_env_ids.numel() == 0:
                continue

            window = self._states[group_name].window
            window.episode_count += group_env_ids.numel()
            survival = torch.clamp(
                self._env.episode_length_buf[group_env_ids].float() / max_episode_length,
                0.0,
                1.0,
            )
            window.survival_sum += survival.sum()
            window.base_contact_terminations += base_contact_termination[group_env_ids].sum()

            for metric_name in _CURRICULUM_METRIC_NAMES:
                sums = getattr(self._env, f"_command_curriculum_{metric_name}_sum", None)
                counts = getattr(self._env, f"_command_curriculum_{metric_name}_count", None)
                if not isinstance(sums, torch.Tensor) or not isinstance(counts, torch.Tensor):
                    continue
                window.metrics[metric_name].add(
                    sums[group_env_ids],
                    counts[group_env_ids],
                )

    def _clear_episode_metrics(self, env_ids: torch.Tensor) -> None:
        """清空刚 reset 环境的课程原始累计器。"""
        for metric_name in _CURRICULUM_METRIC_NAMES:
            sums = getattr(self._env, f"_command_curriculum_{metric_name}_sum", None)
            counts = getattr(self._env, f"_command_curriculum_{metric_name}_count", None)
            if isinstance(sums, torch.Tensor):
                sums[env_ids] = 0.0
            if isinstance(counts, torch.Tensor):
                counts[env_ids] = 0.0

    def _evaluate_group(self, group_name: str) -> tuple[bool, bool, bool]:
        """计算一个窗口是否满足任务表现与安全门。"""
        state = self._states[group_name]
        window = state.window
        is_loco = group_name == self._loco_group_name
        lin_name = "lin_score" if is_loco else "lin_score_all"
        yaw_name = "yaw_score" if is_loco else "yaw_score_all"
        lin_metric = window.metrics[lin_name]
        yaw_metric = window.metrics[yaw_name]

        tracking_samples_ok = (
            lin_metric.samples >= self._min_tracking_samples
            and yaw_metric.samples >= self._min_tracking_samples
        )
        tracking_ok = (
            tracking_samples_ok
            and lin_metric.mean >= self._lin_score_threshold
            and yaw_metric.mean >= self._yaw_score_threshold
        )
        episode_count = max(1, int(window.episode_count.item()))
        survival_score = float(window.survival_sum.item()) / episode_count
        base_contact_rate = float(window.base_contact_terminations.item()) / episode_count
        ready_metric = window.metrics["ready_score"]

        if is_loco:
            task_ok = (
                survival_score >= self._loco_survival_threshold
                and base_contact_rate <= self._loco_base_contact_max
            )
        else:
            task_ok = (
                ready_metric.samples > 0 and ready_metric.mean >= self._recover_ready_threshold
            )

        action_metric = window.metrics["action_saturation"]
        leg_torque_metric = window.metrics["leg_torque_saturation"]
        wheel_torque_metric = window.metrics["wheel_torque_saturation"]
        safety_samples_ok = all(
            metric.samples > 0 for metric in (action_metric, leg_torque_metric, wheel_torque_metric)
        )
        safety_ok = (
            safety_samples_ok
            and action_metric.mean <= self._action_saturation_max
            and leg_torque_metric.mean <= self._leg_torque_saturation_max
            and wheel_torque_metric.mean <= self._wheel_torque_saturation_max
        )
        performance_ok = tracking_ok and task_ok

        state.last_metrics = {
            "episode_count": float(episode_count),
            "lin_sample_count": float(lin_metric.samples),
            "yaw_sample_count": float(yaw_metric.samples),
            "lin_score": lin_metric.mean,
            "yaw_score": yaw_metric.mean,
            "ready_score": ready_metric.mean,
            "survival_score": survival_score,
            "base_contact_termination_rate": base_contact_rate,
            "leg_contact_rate": window.metrics["leg_contact"].mean,
            "base_contact_step_rate": window.metrics["base_contact"].mean,
            "action_saturation_rate": action_metric.mean,
            "leg_torque_saturation_rate": leg_torque_metric.mean,
            "wheel_torque_saturation_rate": wheel_torque_metric.mean,
            "tracking_samples_ok": float(tracking_samples_ok),
            "performance_ok": float(performance_ok),
            "safety_ok": float(safety_ok),
            "passed": float(performance_ok and safety_ok),
        }
        return performance_ok and safety_ok, performance_ok, safety_ok

    def _apply_all_group_ranges(self) -> None:
        """把两个组当前级别写入命令生成器的逐环境范围。"""
        group_ids = self._env.env_group_ids
        for group_name, group_id in self._group_ids.items():
            level = self._states[group_name].level
            env_ids = torch.nonzero(group_ids == group_id, as_tuple=False).squeeze(-1)
            lin_max = self._max_lin_vel_x * level
            yaw_max = self._max_ang_vel_yaw * level
            self._command_term.set_velocity_ranges(
                env_ids,
                lin_vel_x_range=(-lin_max, lin_max),
                ang_vel_yaw_range=(-yaw_max, yaw_max),
            )

    def _log_state(
        self,
        *,
        evaluation_events: dict[str, bool],
        upgrade_events: dict[str, bool],
        safety_blocks: dict[str, bool],
    ) -> dict[str, float]:
        """生成 CurriculumManager 可直接写入 W&B 的标量状态。"""
        result: dict[str, float] = {}
        for group_name in (self._loco_group_name, self._recover_group_name):
            state = self._states[group_name]
            prefix = "loco" if group_name == self._loco_group_name else "recover"
            values = {
                "level": state.level,
                "lin_vel_x_max": self._max_lin_vel_x * state.level,
                "ang_vel_yaw_max": self._max_ang_vel_yaw * state.level,
                "pass_streak": float(state.pass_streak),
                "evaluation_event": float(evaluation_events[group_name]),
                "upgrade_event": float(upgrade_events[group_name]),
                "safety_block": float(safety_blocks[group_name]),
                **state.last_metrics,
            }
            for key, value in values.items():
                result[f"{prefix}_{key}"] = float(value)
        return result


__all__ = [
    "GroupedRewardVelocityCurriculum",
    "commands_height",
    "commands_vel",
    "commands_vel_adaptive",
    "push_disturbance",
]
