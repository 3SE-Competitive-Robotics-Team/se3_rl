"""PPO + stair-only masked teacher loss 的最小实现。"""

from __future__ import annotations

import copy
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from tensordict import TensorDict

from .config import (
    MaskConfig,
    StairTeacherStudentConfig,
    TeacherCheckpointConfig,
    TeacherLossConfig,
)
from .masks import TerrainMetadata, build_teacher_masks
from .teachers import TeacherBank, load_teacher_bank


@dataclass
class TeacherTargetBatch:
    """一次 rollout step 或 mini-batch 对应的 teacher 目标。"""

    actions: dict[str, torch.Tensor]
    masks: dict[str, torch.Tensor]


class TeacherRolloutStorage(RolloutStorage):
    """在 RSL-RL rollout storage 上附加 teacher action/mask。"""

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
        teacher_names: tuple[str, ...] = ("stair",),
    ) -> None:
        """初始化 PPO 原始 buffer 和 teacher 监督 buffer。"""
        super().__init__(
            training_type, num_envs, num_transitions_per_env, obs, actions_shape, device
        )
        self.teacher_names = teacher_names
        self.teacher_actions = {
            name: torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            for name in teacher_names
        }
        self.teacher_masks = {
            name: torch.zeros(
                num_transitions_per_env, num_envs, device=self.device, dtype=torch.bool
            )
            for name in teacher_names
        }

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        """保存 PPO transition，同时保存 teacher 目标。"""
        step = self.step
        super().add_transition(transition)
        transition_actions = getattr(transition, "teacher_actions", {})
        transition_masks = getattr(transition, "teacher_masks", {})
        for name in self.teacher_names:
            action = transition_actions.get(name)
            mask = transition_masks.get(name)
            if action is not None:
                self.teacher_actions[name][step].copy_(action)
            if mask is not None:
                self.teacher_masks[name][step].copy_(mask.to(dtype=torch.bool).view(-1))

    def mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[RolloutStorage.Batch, None, None]:
        """生成 feedforward batch，并附加 teacher 字段。"""
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size,
            requires_grad=False,
            device=self.device,
        )

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)
        teacher_actions = {
            name: value.flatten(0, 1) for name, value in self.teacher_actions.items()
        }
        teacher_masks = {name: value.flatten(0, 1) for name, value in self.teacher_masks.items()}

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]
                batch = RolloutStorage.Batch(
                    observations=observations[batch_idx],
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                )
                batch.teacher_actions = {
                    name: value[batch_idx] for name, value in teacher_actions.items()
                }
                batch.teacher_masks = {
                    name: value[batch_idx] for name, value in teacher_masks.items()
                }
                yield batch

    def recurrent_mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[RolloutStorage.Batch, None, None]:
        """生成 recurrent batch，并附加未 padding 的 teacher 字段。"""
        batch_iter = super().recurrent_mini_batch_generator(num_mini_batches, num_epochs)
        mini_batch_size = self.num_envs // num_mini_batches
        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch = next(batch_iter)
                batch.teacher_actions = {
                    name: value[:, start:stop] for name, value in self.teacher_actions.items()
                }
                batch.teacher_masks = {
                    name: value[:, start:stop] for name, value in self.teacher_masks.items()
                }
                yield batch


class StairTeacherPPO(PPO):
    """在 RSL-RL PPO 上增加 stair-only masked teacher loss。"""

    def __init__(
        self,
        *args: Any,
        teacher_bank: TeacherBank | None = None,
        mask_config: MaskConfig | None = None,
        teacher_loss_config: TeacherLossConfig | None = None,
        terrain_metadata: TerrainMetadata | None = None,
        **kwargs: Any,
    ) -> None:
        """保留 PPO 初始化路径，同时挂载 stair teacher 组件。"""
        super().__init__(*args, **kwargs)
        if self.rnd is not None:
            raise ValueError("StairTeacherPPO 暂不支持 RND。")
        if self.symmetry is not None:
            raise ValueError("StairTeacherPPO 暂不支持 symmetry augmentation。")
        self.teacher_bank = teacher_bank
        self.mask_config = mask_config or MaskConfig()
        self.teacher_loss_config = teacher_loss_config or TeacherLossConfig()
        self.terrain_metadata = terrain_metadata
        self.teacher_update_iter = 0

    def act(self, obs: Any) -> torch.Tensor:
        """采样 student action，并计算 stair teacher target 供 storage 保存。"""
        actions = super().act(obs)
        if self.teacher_bank is not None and "stair" in self.teacher_bank:
            masks = build_teacher_masks(obs, self.terrain_metadata, self.mask_config)
            self.transition.teacher_actions = {"stair": self.teacher_bank["stair"].act(obs)}
            self.transition.teacher_masks = {"stair": masks.stair}
        return actions

    def process_env_step(
        self,
        obs: Any,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """沿用 PPO transition 处理，并同步重置 teacher hidden state。"""
        super().process_env_step(obs, rewards, dones, extras)
        if self.teacher_bank is not None:
            self.teacher_bank.reset(dones)

    def update(self) -> dict[str, float]:
        """执行 PPO update，并在 loss 上加入 masked stair teacher imitation。"""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_teacher_total = 0.0
        mean_teacher_stair = 0.0
        mean_teacher_stair_mask = 0.0

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
            )

        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1e-8
                    )

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(
                batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
            )
            distribution_params = self.actor.output_distribution_params
            entropy = self.actor.output_entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(
                        batch.old_distribution_params, distribution_params
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
            )
            teacher_loss, teacher_logs = self.masked_teacher_loss(
                student_actions=self.actor.output_mean,
                teacher_targets=TeacherTargetBatch(
                    actions=getattr(batch, "teacher_actions", {}),
                    masks=getattr(batch, "teacher_masks", {}),
                ),
                iteration=self.teacher_update_iter,
            )
            loss = loss + teacher_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_teacher_total += teacher_logs.get("teacher_total", 0.0)
            mean_teacher_stair += teacher_logs.get("teacher_stair", 0.0)
            mean_teacher_stair_mask += teacher_logs.get("teacher_stair_mask_rate", 0.0)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        teacher_coef = self.teacher_loss_config.coef_at(self.teacher_update_iter)
        self.storage.clear()
        self.teacher_update_iter += 1
        return {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "teacher_total": mean_teacher_total / num_updates,
            "teacher_stair": mean_teacher_stair / num_updates,
            "teacher_stair_mask": mean_teacher_stair_mask / num_updates,
            "teacher_coef": teacher_coef,
        }

    def masked_teacher_loss(
        self,
        student_actions: torch.Tensor,
        teacher_targets: TeacherTargetBatch,
        iteration: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """计算可加到 PPO loss 上的 masked teacher loss。"""
        coef = self.teacher_loss_config.coef_at(iteration)
        zero = student_actions.sum() * 0.0
        if coef <= 0.0 or not teacher_targets.actions:
            return zero, {"teacher_coef": coef, "teacher_total": 0.0}

        total = zero
        logs: dict[str, float] = {"teacher_coef": coef}
        active_terms = 0
        for name, teacher_actions in teacher_targets.actions.items():
            mask = teacher_targets.masks.get(name)
            if mask is None:
                continue
            term = _masked_action_loss(
                student_actions=student_actions,
                teacher_actions=teacher_actions,
                mask=mask,
                loss_type=self.teacher_loss_config.loss_type,
            )
            total = total + term
            logs[f"teacher_{name}"] = float(term.detach().cpu().item())
            logs[f"teacher_{name}_mask_rate"] = float(mask.float().mean().detach().cpu().item())
            active_terms += 1

        if active_terms == 0:
            return zero, {"teacher_coef": coef, "teacher_total": 0.0}
        scaled = total * coef
        logs["teacher_total"] = float(scaled.detach().cpu().item())
        return scaled, logs

    @staticmethod
    def construct_algorithm(
        obs: TensorDict, env: VecEnv, cfg: dict, device: str
    ) -> StairTeacherPPO:
        """按 RSL-RL 约定构造 PPO student、storage 和 frozen stair teacher。"""
        alg_class: type[StairTeacherPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))
        _strip_none_model_options(cfg["actor"])
        _strip_none_model_options(cfg["critic"])
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))

        teacher_student_cfg = _teacher_student_config_from_dict(
            cfg["algorithm"].pop("teacher_student", {})
        )
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor_kwargs = copy.deepcopy(cfg["actor"])
        critic_kwargs = copy.deepcopy(cfg["critic"])
        actor = actor_class(
            obs,
            cfg["obs_groups"],
            "actor",
            env.num_actions,
            **copy.deepcopy(actor_kwargs),
        ).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            critic_kwargs["cnns"] = actor.cnns
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_kwargs).to(device)
        print(f"Critic Model: {critic}")

        storage = TeacherRolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            device,
            teacher_names=tuple(teacher.name for teacher in teacher_student_cfg.teachers),
        )
        teacher_bank = load_teacher_bank(
            teacher_student_cfg,
            obs=obs,
            obs_groups=cfg["obs_groups"],
            actor_class=actor_class,
            actor_kwargs=actor_kwargs,
            action_dim=env.num_actions,
            device=device,
        )
        terrain_metadata = TerrainMetadata.from_env(getattr(env, "unwrapped", env))
        alg = alg_class(
            actor,
            critic,
            storage,
            device=device,
            teacher_bank=teacher_bank,
            mask_config=teacher_student_cfg.mask,
            teacher_loss_config=teacher_student_cfg.teacher_loss,
            terrain_metadata=terrain_metadata,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        alg.compile(cfg.get("torch_compile_mode"))
        return alg


def _masked_action_loss(
    student_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    """按样本 mask 对齐 student deterministic action 和 teacher action。"""
    if student_actions.shape != teacher_actions.shape:
        raise ValueError(
            f"student/teacher action shape 不一致: {student_actions.shape} vs {teacher_actions.shape}"
        )
    mask_f = mask.to(device=student_actions.device, dtype=student_actions.dtype)
    while mask_f.ndim < student_actions.ndim:
        mask_f = mask_f.unsqueeze(-1)
    denominator = mask_f.sum().clamp_min(1.0)
    if loss_type == "mse":
        per_action = F.mse_loss(student_actions, teacher_actions.detach(), reduction="none")
    elif loss_type == "huber":
        per_action = F.huber_loss(student_actions, teacher_actions.detach(), reduction="none")
    else:
        raise ValueError(f"未知 teacher loss 类型: {loss_type}")
    return (per_action * mask_f).sum() / denominator


def _teacher_student_config_from_dict(data: dict[str, Any]) -> StairTeacherStudentConfig:
    """从算法配置 dict 构造 teacher/student dataclass。"""
    default = StairTeacherStudentConfig()
    if not data:
        return default

    mask = data.get("mask", {})
    teacher_loss = data.get("teacher_loss", {})
    stair_teacher = data.get("stair_teacher", {})
    return StairTeacherStudentConfig(
        enabled=bool(data.get("enabled", default.enabled)),
        mask=MaskConfig(**{**default.mask.__dict__, **mask}),
        teacher_loss=TeacherLossConfig(**{**default.teacher_loss.__dict__, **teacher_loss}),
        stair_teacher=_teacher_checkpoint_config(default.stair_teacher, stair_teacher),
    )


def _teacher_checkpoint_config(
    default: TeacherCheckpointConfig,
    overrides: dict[str, Any],
) -> TeacherCheckpointConfig:
    """构造 checkpoint 配置，并把字符串路径恢复成 Path。"""
    data = {**default.__dict__, **overrides}
    data["path"] = Path(data["path"])
    return TeacherCheckpointConfig(**data)


def _strip_none_model_options(model_cfg: dict[str, Any]) -> None:
    """剥掉 MJLab model cfg 中值为 None 的可选字段。"""
    for opt in ("cnn_cfg", "distribution_cfg"):
        if model_cfg.get(opt) is None:
            model_cfg.pop(opt, None)
    if model_cfg.get("rnn_type") is None:
        for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            model_cfg.pop(opt, None)
