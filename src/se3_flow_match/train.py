"""Flow Matching 学生离线训练。"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import time

import torch

from se3_shared import TaskMode

from .checkpoint import save_flow_checkpoint
from .config import FlowPolicyConfig
from .dataset import load_teacher_dataset
from .losses import flow_matching_loss
from .model import FlowVelocityField
from .task_mode import mode_transition_obs


def train_flow_student(
    *,
    dataset_path: Path,
    output: Path,
    device: str,
    batch_size: int,
    max_steps: int,
    learning_rate: float,
    transition_ratio: float,
    burn_in_steps: int,
    loss_steps: int,
) -> None:
    """训练 Flow Matching 学生模型。"""
    if burn_in_steps < 0:
        raise ValueError(f"burn_in_steps 不能为负数，实际为 {burn_in_steps}")
    if loss_steps <= 0:
        raise ValueError(f"loss_steps 必须为正数，实际为 {loss_steps}")
    config = FlowPolicyConfig()
    dataset = load_teacher_dataset(dataset_path, config=config)
    _validate_window_length(dataset.obs, burn_in_steps=burn_in_steps, loss_steps=loss_steps)
    obs, actions, dones, modes = _augment_transitions(
        dataset.obs,
        dataset.actions,
        dataset.dones,
        dataset.modes,
        transition_ratio=transition_ratio,
    )
    model = FlowVelocityField(config).to(device)
    _init_obs_normalizer(model, dataset.obs)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    step = 0
    start = time()
    last_loss = float("nan")
    while step < max_steps:
        batch_obs, batch_actions, batch_dones, batch_modes = _sample_training_window(
            obs,
            actions,
            dones,
            modes,
            batch_size=batch_size,
            burn_in_steps=burn_in_steps,
            loss_steps=loss_steps,
        )
        batch_weights = _action_weights_for_modes(batch_modes).to(dtype=torch.float32)
        if burn_in_steps > 0:
            batch_weights[:, :burn_in_steps, :] = 0.0
        batch_obs = batch_obs.to(device)
        batch_actions = batch_actions.to(device)
        batch_dones = batch_dones.to(device)
        batch_weights = batch_weights.to(device)
        loss_info = flow_matching_loss(
            model,
            batch_obs,
            batch_actions,
            dones=batch_dones,
            action_weights=batch_weights,
        )
        optimizer.zero_grad(set_to_none=True)
        loss_info.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1
        last_loss = float(loss_info.loss.detach().cpu())
        if step % 50 == 0 or step == 1:
            print(f"[flow-train] step={step} loss={last_loss:.6f}")

    save_flow_checkpoint(
        output,
        model,
        config,
        metadata={
            "dataset": str(dataset_path),
            "max_steps": max_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "transition_ratio": transition_ratio,
            "burn_in_steps": burn_in_steps,
            "loss_steps": loss_steps,
            "final_loss": last_loss,
            "train_seconds": time() - start,
            "dataset_metadata": dataset.metadata,
        },
    )
    print(f"[flow-train] saved {output} final_loss={last_loss:.6f}")


def _augment_transitions(
    obs: torch.Tensor,
    actions: torch.Tensor,
    dones: torch.Tensor,
    modes: torch.Tensor,
    *,
    transition_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """复制目标模式样本，生成最简单 prev/current/blend 过渡样本。"""
    if transition_ratio <= 0.0:
        return obs, actions, dones, modes
    parts_obs = [obs]
    parts_actions = [actions]
    parts_dones = [dones]
    parts_modes = [modes]

    max_count = round(float(obs.shape[0]) * float(transition_ratio))
    pairs = ((TaskMode.GAIT, TaskMode.WHEEL), (TaskMode.WHEEL, TaskMode.GAIT))
    per_pair_count = max(1, max_count // len(pairs)) if max_count > 0 else 0
    added = 0
    for current, prev in pairs:
        idx = (modes[:, 0] == int(current)).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        remaining = max_count - added if max_count > 0 else idx.numel()
        take = min(idx.numel(), per_pair_count, remaining)
        if take <= 0:
            break
        idx = idx[:take]
        parts_obs.append(mode_transition_obs(obs[idx], current=current, prev=prev))
        parts_actions.append(actions[idx])
        parts_dones.append(dones[idx])
        parts_modes.append(modes[idx])
        added += take
        if max_count > 0 and added >= max_count:
            break
    return (
        torch.cat(parts_obs, dim=0),
        torch.cat(parts_actions, dim=0),
        torch.cat(parts_dones, dim=0),
        torch.cat(parts_modes, dim=0),
    )


def _validate_window_length(
    obs: torch.Tensor,
    *,
    burn_in_steps: int,
    loss_steps: int,
) -> None:
    """确认长轨迹足够切出训练窗口。"""
    window = int(burn_in_steps) + int(loss_steps)
    if obs.shape[1] < window:
        raise ValueError(
            f"数据序列长度 {obs.shape[1]} 小于 burn_in+loss 窗口 {window}，"
            "请增加采集 command_hold_s 或减小训练窗口。"
        )


def _sample_training_window(
    obs: torch.Tensor,
    actions: torch.Tensor,
    dones: torch.Tensor,
    modes: torch.Tensor,
    *,
    batch_size: int,
    burn_in_steps: int,
    loss_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """从长轨迹中随机采样 [burn_in + loss] 训练窗口。"""
    window = int(burn_in_steps) + int(loss_steps)
    _validate_window_length(obs, burn_in_steps=burn_in_steps, loss_steps=loss_steps)
    traj_ids = torch.randint(obs.shape[0], (batch_size,))
    max_start = obs.shape[1] - window
    if max_start == 0:
        starts = torch.zeros(batch_size, dtype=torch.long)
    else:
        starts = torch.randint(max_start + 1, (batch_size,))
    offsets = torch.arange(window).view(1, -1)
    time_ids = starts.view(-1, 1) + offsets
    return (
        obs[traj_ids[:, None], time_ids],
        actions[traj_ids[:, None], time_ids],
        dones[traj_ids[:, None], time_ids],
        modes[traj_ids[:, None], time_ids],
    )


def _action_weights_for_modes(modes: torch.Tensor) -> torch.Tensor:
    """按 mode 生成 action loss 权重。"""
    weights = torch.ones(*modes.shape, 6, dtype=torch.float32)
    gait = modes == int(TaskMode.GAIT)
    weights[gait, 4:6] = 1.0
    return weights


def _init_obs_normalizer(model: FlowVelocityField, obs: torch.Tensor) -> None:
    """用离线数据集统计量初始化观测归一化。"""
    flat = obs.reshape(-1, obs.shape[-1]).to(dtype=torch.float32)
    mean = flat.mean(dim=0)
    std = flat.std(dim=0).clamp_min(1.0e-4)
    model.set_obs_statistics(mean, std)


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI parser。"""
    parser = argparse.ArgumentParser(description="Train Flow Matching student")
    parser.add_argument("--dataset", type=Path, default=Path("data/flow_match/wheel_gait.pt"))
    parser.add_argument("--output", type=Path, default=Path("logs/flow_match/wheel_gait/flow.pt"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--transition-ratio", type=float, default=0.25)
    parser.add_argument("--burn-in-steps", type=int, default=64)
    parser.add_argument("--loss-steps", type=int, default=128)
    return parser


def main() -> None:
    """CLI 入口。"""
    args = build_parser().parse_args()
    train_flow_student(
        dataset_path=args.dataset,
        output=args.output,
        device=str(args.device),
        batch_size=int(args.batch_size),
        max_steps=int(args.max_steps),
        learning_rate=float(args.learning_rate),
        transition_ratio=float(args.transition_ratio),
        burn_in_steps=int(args.burn_in_steps),
        loss_steps=int(args.loss_steps),
    )


if __name__ == "__main__":
    main()
