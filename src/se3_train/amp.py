"""AMP（Adversarial Motion Priors）扩展，移植自 BioInnov/rsl_rl_bioin 的 `rsl_rl/extensions/amp.py`。

我们装的是 PyPI 的 rsl_rl 5.4.0，没有 fork 的 ExtManager / 多 RL 分组，所以：
- 保留原实现的全部数学：MLP 判别器 + 输入经验归一化（用专家数据热启动）、LSGAN 损失
  （专家目标 +1、策略目标 −1）、专家样本上的 R1 梯度惩罚（系数 0.5×w）、
  风格奖励 r = clamp(1 − 0.25 (D − 1)^2, 0)，按 reward_weight × step_dt × 热身因子加进任务奖励；
  策略侧的 transition 窗口每步经逐 env 的帧环形缓冲拼出，判别器更新时直接从 rollout storage 采窗口，
  跨 done 的窗口剔除。
- 去掉多数据集 / group_ids 路由与 DDP 同步（本仓库多卡走 torchrunx 逐卡独立训练）。
- 归一化按 docs/amp_input.md：统计量在 19 维单帧上、专家与策略共用，前后两帧各自归一化。
- PPO 侧钩子在 se3_train.ppo.Se3PPO：process_env_step 里在写 storage 之前调用
  `AMP.process_env_step(obs, transition, extras)`；update 里在 storage 清空之前调用
  `AMP.individual_update(storage)`；存档键沿用 fork 的 `ext_state_dict["amp"]`。

专家数据：契约 `se3.amp.motion.v1`（se3_shared.amp，19 维单帧、20 ms），由
scripts/export_fudan_amp_features.py / package_fudan_amp_dataset.py 产出，
`build_amp_dataset` 读成 fork 约定的 `{"sequences": [Tensor[T, 19]], "lengths": [T]}`。
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rsl_rl.modules import MLP
from rsl_rl.modules.normalization import EmpiricalNormalization
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict
from torch import nn, optim

from se3_shared.amp import AMP_CONTROL_DT_S, AMP_FRAME_DIM

# ---------------------------------------------------------------------------
# 专家数据集（se3.amp.motion.v1）
# ---------------------------------------------------------------------------
# 19 维单帧的左右镜像（关于机身 xz 平面）：
#   gravity y 取反；omega x、z 取反；velocity y 取反；
#   左右轮 xz 位置/速度互换；左右轮自转互换。
_MIRROR_PERM = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 9, 10, 15, 16, 13, 14, 18, 17]
_MIRROR_SIGN = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0] + [1.0] * 10


def mirror_amp_frames(frames: torch.Tensor) -> torch.Tensor:
    """[T, 19] → 左右镜像后的 [T, 19]。"""
    if frames.shape[-1] != AMP_FRAME_DIM:
        raise ValueError(f"AMP 帧必须是 {AMP_FRAME_DIM} 维，实际 {frames.shape}")
    perm = torch.tensor(_MIRROR_PERM, device=frames.device)
    sign = torch.tensor(_MIRROR_SIGN, device=frames.device, dtype=frames.dtype)
    return frames[..., perm] * sign


def _sequences_from_frames(frames: np.ndarray, offsets: np.ndarray | None) -> list[np.ndarray]:
    if frames.ndim != 2 or frames.shape[1] != AMP_FRAME_DIM:
        raise ValueError(f"frames 形状应为 [N, {AMP_FRAME_DIM}]，实际 {frames.shape}")
    if not np.isfinite(frames).all():
        raise ValueError("frames 含非有限值")
    if offsets is None:
        return [frames]
    offsets = np.asarray(offsets, dtype=np.int64).reshape(-1)
    if offsets[0] != 0 or offsets[-1] != frames.shape[0] or np.any(np.diff(offsets) <= 0):
        raise ValueError(f"frame_offsets 必须从 0 单调递增到 {frames.shape[0]}，实际 {offsets.tolist()}")
    return [frames[a:b] for a, b in pairwise(offsets)]


def _check_control_dt(path: Path, simulation_dt: float) -> None:
    """契约固定 20 ms；训练控制周期必须一致，不做重采样。"""
    if abs(float(simulation_dt) - AMP_CONTROL_DT_S) > 1e-8:
        raise ValueError(f"AMP 契约要求相邻帧 20 ms，训练 step_dt={simulation_dt}（{path}）")


def load_amp_sequences(dataset_root: str | Path, simulation_dt: float) -> list[np.ndarray]:
    """读专家序列，返回若干 [T, 19]。

    `dataset_root` 可以是：
    - 含 `source_dataset.npz`（frames + frame_offsets，package 脚本产物）的目录；
    - 含若干 `<sample>/amp/source_features.npz`（frames，export 脚本产物）的目录；
    - 单个上述 .npz 文件。
    """
    root = Path(dataset_root)
    _check_control_dt(root, simulation_dt)
    if root.is_file():
        files = [root]
    elif (root / "source_dataset.npz").is_file():
        files = [root / "source_dataset.npz"]
    else:
        files = sorted(root.glob("*/amp/source_features.npz")) + sorted(root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"AMP 数据集为空：{root}")
    sequences: list[np.ndarray] = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            if "frames" not in data:
                raise ValueError(f"{path} 缺少 frames")
            frames = np.asarray(data["frames"], dtype=np.float32)
            offsets = np.asarray(data["frame_offsets"]) if "frame_offsets" in data else None
        for seq in _sequences_from_frames(frames, offsets):
            if seq.shape[0] >= 2:
                sequences.append(seq)
    if not sequences:
        raise ValueError(f"AMP 数据集 {root} 没有长度 ≥ 2 帧的序列")
    return sequences


def _dataset_readiness_note(dataset_root: str | Path) -> str | None:
    """读导出元数据里的就绪标记，未重定向到 SerialLeg 的数据给出提示（不阻断）。"""
    root = Path(dataset_root)
    for meta in [root / "amp_training.json", root / "metadata.json", *sorted(root.glob("*/amp/metadata.json"))[:1]]:
        if meta.is_file():
            try:
                payload = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if payload.get("ready_for_discriminator_training") is False:
                return f"{meta}: ready_for_discriminator_training=false（{payload.get('status', '')}）"
            return None
    return None


def build_amp_dataset(
    device: str = "cpu",
    *,
    dataset_root: str,
    simulation_dt: float,
    mirror_augmentation: bool = True,
) -> dict[str, Any]:
    """AMP 的 dataset_callable：返回 fork 约定的 {"sequences": [Tensor[T, 19]], "lengths": [T]}。"""
    sequences = [
        torch.as_tensor(seq, dtype=torch.float32, device=device)
        for seq in load_amp_sequences(dataset_root, float(simulation_dt))
    ]
    if mirror_augmentation:
        sequences = sequences + [mirror_amp_frames(seq) for seq in sequences]
    note = _dataset_readiness_note(dataset_root)
    if note:
        print(f"[AMP] 提示：{note}")
    print(f"[AMP] 数据集 {dataset_root}：{len(sequences)} 段序列（含镜像），{sum(int(s.shape[0]) for s in sequences)} 帧")
    return {"sequences": sequences, "lengths": [int(seq.shape[0]) for seq in sequences]}


def default_dataset_root(fallback: str) -> str:
    """数据集目录：环境变量 SE3_AMP_DATASET_ROOT 优先。"""
    return os.environ.get("SE3_AMP_DATASET_ROOT", fallback)


# ---------------------------------------------------------------------------
# 判别器（照抄 fork 的 MotionDiscriminator，只保留 MLP 主干；归一化按 19 维单帧共享）
# ---------------------------------------------------------------------------
class MotionDiscriminator(nn.Module):
    """AMP 判别器：输入 [N, frames×19]，可选按单帧共享统计量归一化 + MLP，输出标量分数。"""

    def __init__(self, transition_frames: int, frame_dim: int, model_cfg: Mapping[str, Any], device: str) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.transition_frames = int(transition_frames)
        self.frame_dim = int(frame_dim)
        self.input_dim = self.transition_frames * self.frame_dim
        self.state_normalization = bool(model_cfg.get("state_normalization", True))
        self.normalizer = (
            EmpiricalNormalization(shape=[self.frame_dim], until=int(1.0e8)).to(self.device)
            if self.state_normalization
            else nn.Identity()
        )
        self.backbone = MLP(
            input_dim=self.input_dim,
            output_dim=1,
            hidden_dims=list(model_cfg.get("hidden_dims", [512, 256])),
            activation=model_cfg.get("activation", "elu"),
        )
        self.to(self.device)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.state_normalization:
            return x
        frames = x.view(x.shape[0], self.transition_frames, self.frame_dim)
        return self.normalizer(frames).reshape(x.shape[0], self.input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self._normalize(x)).squeeze(-1)

    def update_normalizer(self, x: torch.Tensor) -> None:
        """用 transition 里的每一帧更新单帧统计量（专家与策略共用同一份）。"""
        if self.state_normalization:
            self.normalizer.update(x.reshape(-1, self.frame_dim))  # type: ignore[attr-defined]

    def compute_loss(
        self, *, expert_sequences: torch.Tensor, policy_sequences: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """LSGAN：专家 → +1，策略 → −1。"""
        expert_scores = self.forward(expert_sequences)
        policy_scores = self.forward(policy_sequences)
        expert_loss = torch.square(expert_scores - 1.0).mean()
        policy_loss = torch.square(policy_scores + 1.0).mean()
        loss = expert_loss + policy_loss
        return loss, {
            "discriminator_loss": loss.detach(),
            "expert_loss": expert_loss.detach(),
            "policy_loss": policy_loss.detach(),
            "expert_score": expert_scores.detach().mean(),
            "policy_score": policy_scores.detach().mean(),
        }

    def gradient_penalty(self, expert_sequences: torch.Tensor) -> torch.Tensor:
        """R1：专家样本上判别器输入梯度的平方范数。"""
        expert_in = expert_sequences.detach().requires_grad_(True)
        expert_scores = self.forward(expert_in)
        expert_grad = torch.autograd.grad(outputs=expert_scores.sum(), inputs=expert_in, create_graph=True)[0]
        return expert_grad.square().sum(dim=-1).mean()

    def predict_reward(self, sequences: torch.Tensor) -> torch.Tensor:
        scores = self.forward(sequences)
        return torch.clamp(1.0 - 0.25 * torch.square(scores - 1.0), min=0.0)


# ---------------------------------------------------------------------------
# AMP 扩展
# ---------------------------------------------------------------------------
class AMP(nn.Module):
    """单数据集 AMP：逐 env 帧缓冲出风格奖励，判别器从 rollout storage 采策略窗口更新。

    cfg 键与 kyber_rl_lab 的 RslRlAmpCfg 一致：obs_group、transition_frames、reward_weight、
    reward_warmup_updates、discriminator_updates、discriminator_batch_size、
    discriminator_grad_penalty_weight、learning_rate、max_grad_norm、model_cfg{hidden_dims, activation,
    state_normalization}、resume_checkpoint、resume_optimizer；数据集用 dataset_kwargs 交给
    `build_amp_dataset`（dataset_root、mirror_augmentation）。
    """

    amp_update_counter: torch.Tensor

    def __init__(self, num_envs: int, step_dt: float, obs: TensorDict, cfg: Mapping[str, Any], device: str) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.obs_group = str(cfg.get("obs_group", "amp"))
        if self.obs_group not in obs:
            raise ValueError(f"AMP 观测组 '{self.obs_group}' 不存在，可用：{list(obs.keys())}")
        self.step_dt = float(step_dt)
        self.transition_frames = int(cfg.get("transition_frames", 2))
        if self.transition_frames < 2:
            raise ValueError("transition_frames 必须 ≥ 2")
        self.reward_weight = float(cfg.get("reward_weight", 1.0))
        self.reward_warmup_updates = int(cfg.get("reward_warmup_updates", 50))
        self.discriminator_updates = int(cfg.get("discriminator_updates", 4))
        self.discriminator_batch_size = int(cfg.get("discriminator_batch_size", 256))
        self.discriminator_grad_penalty_weight = float(cfg.get("discriminator_grad_penalty_weight", 5.0))
        self.max_grad_norm = cfg.get("max_grad_norm")
        self.resume_checkpoint = bool(cfg.get("resume_checkpoint", True))
        self.resume_optimizer = bool(cfg.get("resume_optimizer", False))
        learning_rate = cfg.get("learning_rate", cfg.get("discriminator_lr", 5.0e-4))
        self.learning_rate = float(learning_rate) if learning_rate is not None else 5.0e-4

        self.amp_obs_dim = int(obs[self.obs_group].shape[-1])
        if self.amp_obs_dim != AMP_FRAME_DIM:
            raise ValueError(f"AMP 观测组 '{self.obs_group}' 应为 {AMP_FRAME_DIM} 维，实际 {self.amp_obs_dim}")
        self.sequence_dim = self.transition_frames * self.amp_obs_dim

        self._register_dataset(self._load_dataset(cfg))

        self.discriminator = MotionDiscriminator(
            transition_frames=self.transition_frames,
            frame_dim=self.amp_obs_dim,
            model_cfg=dict(cfg.get("model_cfg", {})),
            device=str(self.device),
        )
        self._warmup_normalization_from_offline_data()
        self.optimizer = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)

        self.frame_buffer = torch.zeros(num_envs, self.transition_frames, self.amp_obs_dim, device=self.device)
        self.frame_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.write_idx = 0
        order = torch.arange(self.transition_frames, device=self.device)
        self.order_lut = torch.stack(
            [torch.roll(order, shifts=-(idx + 1), dims=0) for idx in range(self.transition_frames)], dim=0
        )
        self.window_offsets = torch.arange(self.transition_frames, device=self.device).unsqueeze(0)
        self.register_buffer("amp_update_counter", torch.zeros((), dtype=torch.long, device=self.device))
        self._style_sum = 0.0
        self._style_count = 0

    # ---- 数据集 ----
    def _load_dataset(self, cfg: Mapping[str, Any]) -> dict[str, Any]:
        dataset_kwargs = dict(cfg.get("dataset_kwargs", {}))
        if "dataset_root" not in dataset_kwargs:
            raise ValueError("AMP cfg.dataset_kwargs 必须包含 dataset_root")
        dataset_kwargs.setdefault("simulation_dt", self.step_dt)
        return build_amp_dataset(device=str(self.device), **dataset_kwargs)

    def _register_dataset(self, dataset: Mapping[str, Any]) -> None:
        sequences = [seq.to(self.device) for seq in dataset["sequences"]]
        if len(sequences) == 0:
            raise ValueError("AMP 数据集为空")
        lengths = torch.as_tensor(dataset["lengths"], dtype=torch.long, device=self.device)
        dataset_obs_dim = int(sequences[0].shape[-1])
        if dataset_obs_dim != self.amp_obs_dim:
            raise ValueError(f"AMP obs dim mismatch: env '{self.obs_group}' 是 {self.amp_obs_dim}，数据集是 {dataset_obs_dim}")
        eligible_idx = torch.nonzero(lengths >= self.transition_frames, as_tuple=False).squeeze(-1)
        if eligible_idx.numel() == 0:
            raise ValueError(f"没有专家序列长到 transition_frames={self.transition_frames}")
        sequence_offsets = torch.empty_like(lengths)
        sequence_offsets[0] = 0
        if len(sequences) > 1:
            sequence_offsets[1:] = torch.cumsum(lengths[:-1], dim=0)
        eligible_windows = (lengths[eligible_idx] - self.transition_frames + 1).float()
        self.flat_sequences = torch.cat(sequences, dim=0).contiguous()
        self.sequence_offsets = sequence_offsets
        self.lengths = lengths
        self.eligible_idx = eligible_idx
        self.eligible_probs = eligible_windows / eligible_windows.sum()
        self.num_expert_sequences = len(sequences)
        self.num_expert_frames = int(self.flat_sequences.shape[0])

    def _warmup_normalization_from_offline_data(self) -> None:
        if not self.discriminator.state_normalization:
            return
        with torch.no_grad():
            self.discriminator.update_normalizer(self.flat_sequences)

    # ---- 奖励 ----
    def reward_scale(self) -> float:
        warmup = 1.0
        if self.reward_warmup_updates > 0:
            warmup = min(1.0, float(self.amp_update_counter.item()) / self.reward_warmup_updates)
        return self.reward_weight * self.step_dt * warmup

    def process_env_step(self, obs: TensorDict, transition: RolloutStorage.Transition, extras: dict[str, Any]) -> None:
        """把风格奖励加到 transition.rewards 上；done 的 env 清帧缓冲，本步不给风格奖励。"""
        rewards = transition.rewards
        dones = transition.dones
        assert rewards is not None and dones is not None
        frames = obs[self.obs_group]
        dones_bool = dones.view(-1).bool()

        self.frame_count[dones_bool] = 0
        write_mask = ~dones_bool
        self.frame_buffer[write_mask, self.write_idx] = frames[write_mask]
        self.frame_count[write_mask] = torch.clamp(self.frame_count[write_mask] + 1, max=self.transition_frames)

        amp_rewards = frames.new_zeros(frames.shape[0])
        valid_idx = (write_mask & (self.frame_count >= self.transition_frames)).nonzero(as_tuple=False).flatten()
        if valid_idx.numel() > 0:
            order = self.order_lut[self.write_idx]
            sequences = self.frame_buffer[valid_idx][:, order, :].reshape(-1, self.sequence_dim)
            with torch.no_grad():
                amp_rewards[valid_idx] = self.discriminator.predict_reward(sequences)
        self.write_idx = (self.write_idx + 1) % self.transition_frames

        self._style_sum += float(amp_rewards[valid_idx].mean()) if valid_idx.numel() > 0 else 0.0
        self._style_count += 1
        amp_rewards = self.reward_scale() * amp_rewards
        rewards.add_(amp_rewards)
        extras.setdefault("ext_reward", {})["amp"] = amp_rewards.detach()

    # ---- 更新 ----
    def individual_update(self, storage: RolloutStorage) -> dict[str, float]:
        """从 rollout storage 采策略窗口，做 discriminator_updates 步 LSGAN 更新。"""
        metrics: dict[str, float] = {
            "amp/style_reward": (self._style_sum / self._style_count) if self._style_count else 0.0,
            "amp/reward_scale": self.reward_scale(),
        }
        self._style_sum = 0.0
        self._style_count = 0

        valid_idx = self._rollout_valid_end_indices(storage)
        batch = min(int(valid_idx.numel()), self.discriminator_batch_size)
        if batch <= 0 or self.discriminator_updates <= 0:
            return metrics
        rollout_frames = storage.observations[self.obs_group][: storage.step]
        totals: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            policy = self._sample_policy_batch(rollout_frames, valid_idx, batch)
            expert = self._sample_expert_sequences(batch)
            self.discriminator.update_normalizer(policy)
            self.discriminator.update_normalizer(expert)
        for _ in range(self.discriminator_updates):
            policy = self._sample_policy_batch(rollout_frames, valid_idx, batch)
            expert = self._sample_expert_sequences(batch)
            loss, step_metrics = self.discriminator.compute_loss(expert_sequences=expert, policy_sequences=policy)
            grad_penalty = torch.tensor(0.0, device=self.device)
            if self.discriminator_grad_penalty_weight > 0.0:
                grad_penalty = self.discriminator.gradient_penalty(expert_sequences=expert)
                loss = loss + 0.5 * self.discriminator_grad_penalty_weight * grad_penalty
            step_metrics["grad_penalty"] = grad_penalty.detach()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), float(self.max_grad_norm))
            self.optimizer.step()
            for key, value in step_metrics.items():
                totals[key] = totals.get(key, value.new_zeros(())) + value
        self.amp_update_counter += 1
        for key, value in totals.items():
            metrics[f"amp/{key}"] = float(value.item()) / float(self.discriminator_updates)
        return metrics

    def _sample_expert_sequences(self, batch_size: int) -> torch.Tensor:
        pick = torch.multinomial(self.eligible_probs, batch_size, replacement=True)
        seq_ids = self.eligible_idx[pick]
        max_starts = self.lengths[seq_ids] - self.transition_frames
        starts = torch.floor(torch.rand(batch_size, device=self.device) * (max_starts + 1).float()).long()
        time_idx = self.sequence_offsets[seq_ids].unsqueeze(1) + starts.unsqueeze(1) + self.window_offsets
        return self.flat_sequences[time_idx].reshape(batch_size, self.sequence_dim)

    def _sample_policy_batch(self, rollout_frames: torch.Tensor, valid_idx: torch.Tensor, batch_size: int) -> torch.Tensor:
        pick = valid_idx[torch.randint(0, valid_idx.numel(), (batch_size,), device=self.device)]
        num_envs = rollout_frames.shape[1]
        end_t = torch.div(pick, num_envs, rounding_mode="floor")
        env_ids = pick % num_envs
        time_idx = end_t.unsqueeze(1) - (self.transition_frames - 1) + self.window_offsets
        return rollout_frames[time_idx, env_ids.unsqueeze(1)].reshape(batch_size, self.sequence_dim)

    def _rollout_valid_end_indices(self, storage: RolloutStorage) -> torch.Tensor:
        """窗口末帧的展平索引：窗口内任一步 done 即无效（跨 episode）。"""
        if storage.step < self.transition_frames:
            return torch.empty(0, dtype=torch.long, device=self.device)
        step_valid = ~storage.dones[: storage.step].squeeze(-1).bool()
        window_valid = step_valid.clone()
        for offset in range(1, self.transition_frames):
            shifted = torch.zeros_like(step_valid)
            shifted[offset:] = step_valid[:-offset]
            window_valid &= shifted
        return window_valid.flatten().nonzero(as_tuple=False).flatten()

    # ---- 持久化 ----
    def save(self) -> dict[str, Any]:
        return {
            "module_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load(self, state: Mapping[str, Any], strict: bool = True) -> None:
        if not self.resume_checkpoint:
            return
        self.load_state_dict(state["module_state_dict"], strict=strict)
        if self.resume_optimizer and state.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])


__all__ = [
    "AMP",
    "MotionDiscriminator",
    "build_amp_dataset",
    "default_dataset_root",
    "load_amp_sequences",
    "mirror_amp_frames",
]
