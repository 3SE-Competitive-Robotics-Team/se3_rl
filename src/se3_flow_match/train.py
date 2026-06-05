"""Flow Matching 学生离线训练。"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import time

import torch
from torch.utils.data import DataLoader, TensorDataset

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
) -> None:
    """训练 Flow Matching 学生模型。"""
    config = FlowPolicyConfig()
    dataset = load_teacher_dataset(dataset_path, config=config)
    obs, actions, dones, modes = _augment_transitions(
        dataset.obs,
        dataset.actions,
        dataset.dones,
        dataset.modes,
        transition_ratio=transition_ratio,
    )
    action_weights = _action_weights_for_modes(modes).to(dtype=torch.float32)

    tensor_dataset = TensorDataset(obs, actions, dones, action_weights)
    loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model = FlowVelocityField(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    step = 0
    start = time()
    last_loss = float("nan")
    while step < max_steps:
        for batch_obs, batch_actions, batch_dones, batch_weights in loader:
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
            if step >= max_steps:
                break

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
    added = 0
    for current, prev in ((TaskMode.GAIT, TaskMode.WHEEL), (TaskMode.WHEEL, TaskMode.GAIT)):
        idx = (modes[:, 0] == int(current)).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        take = min(idx.numel(), max_count - added if max_count > 0 else idx.numel())
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


def _action_weights_for_modes(modes: torch.Tensor) -> torch.Tensor:
    """按 mode 生成 action loss 权重。"""
    weights = torch.ones(*modes.shape, 6, dtype=torch.float32)
    gait = modes == int(TaskMode.GAIT)
    weights[gait, 4:6] = 1.0
    return weights


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI parser。"""
    parser = argparse.ArgumentParser(description="Train Flow Matching student")
    parser.add_argument("--dataset", type=Path, default=Path("data/flow_match/wheel_gait.pt"))
    parser.add_argument("--output", type=Path, default=Path("logs/flow_match/wheel_gait/flow.pt"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--transition-ratio", type=float, default=1.0)
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
    )


if __name__ == "__main__":
    main()
