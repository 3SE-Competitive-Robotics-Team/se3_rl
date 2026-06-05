"""se3_flow_match smoke 实验。"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import DataLoader

from se3_shared import TaskMode

from .checkpoint import load_flow_checkpoint, save_flow_checkpoint
from .collect import collect_teacher_rollouts
from .config import FlowPolicyConfig
from .dataset import TeacherFlowDataset
from .losses import flow_matching_loss
from .model import FlowVelocityField
from .play import run_flow_play
from .sampler import sample_actions, sample_actions_from_context
from .train import train_flow_student


class _AnalyticVelocityField(torch.nn.Module):
    """用于验证反向 Euler 采样方向的解析速度场。"""

    def __init__(self, config: FlowPolicyConfig, target: torch.Tensor) -> None:
        """记录目标动作，令 velocity = (x_t - target) / t。"""
        super().__init__()
        self.config = config
        self.target = target

    def velocity_from_context(
        self, context: torch.Tensor, noisy_action: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """返回能一步步从噪声动作积分到目标动作的速度。"""
        del context
        return (noisy_action - self.target) / t.clamp_min(1.0e-6)


def main() -> None:
    """运行 synthetic 和 tiny real smoke。"""
    torch.manual_seed(42)
    _run_synthetic_smoke()
    _run_real_pipeline_smoke()


def _run_synthetic_smoke() -> None:
    """合成数据 smoke，确认训练、采样和 checkpoint 可用。"""
    config = FlowPolicyConfig(rnn_hidden_dim=64)
    obs, actions = _make_synthetic_teacher(config)
    dataset = TeacherFlowDataset(obs, actions, config=config)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = FlowVelocityField(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-3)

    eval_noise = torch.randn_like(actions[:8])
    eval_times = torch.rand(actions[:8].shape[0], actions[:8].shape[1], 1)
    initial = flow_matching_loss(
        model, obs[:8], actions[:8], noise=eval_noise, times=eval_times
    ).loss.item()

    for _ in range(80):
        for batch in loader:
            loss_info = flow_matching_loss(
                model,
                batch["obs"],
                batch["actions"],
                dones=batch["dones"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss_info.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    final = flow_matching_loss(
        model, obs[:8], actions[:8], noise=eval_noise, times=eval_times
    ).loss.item()
    if final >= initial:
        raise RuntimeError(
            f"synthetic smoke 失败：loss 未下降 initial={initial:.6f}, final={final:.6f}"
        )

    sampled = sample_actions(model, obs[:8, -1], steps=5)
    if sampled.shape != (8, config.action_dim):
        raise RuntimeError(f"synthetic smoke 失败：采样形状错误 {tuple(sampled.shape)}")
    _assert_sampler_direction(config)

    with TemporaryDirectory() as tmp:
        checkpoint_path = Path(tmp) / "flow.pt"
        save_flow_checkpoint(
            checkpoint_path,
            model,
            config,
            metadata={"initial_loss": initial, "final_loss": final},
        )
        loaded, loaded_config, metadata = load_flow_checkpoint(checkpoint_path)
        if loaded_config != config:
            raise RuntimeError("synthetic smoke 失败：checkpoint 配置未正确恢复")
        loaded_sample = sample_actions(loaded, obs[:8, -1], steps=3)
        if loaded_sample.shape != (8, config.action_dim):
            raise RuntimeError(
                f"synthetic smoke 失败：加载后采样形状错误 {tuple(loaded_sample.shape)}"
            )
        if "final_loss" not in metadata:
            raise RuntimeError("synthetic smoke 失败：checkpoint metadata 未正确恢复")

    print(f"[flow-smoke] synthetic passed initial={initial:.6f} final={final:.6f}")


def _run_real_pipeline_smoke() -> None:
    """真实 teacher tiny pipeline smoke。"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset_path = root / "teacher.pt"
        checkpoint_path = root / "flow.pt"
        collect_teacher_rollouts(
            tasks=["wheel", "gait"],
            output=dataset_path,
            num_envs=1,
            steps=64,
            sequence_length=16,
            device=device,
        )
        train_flow_student(
            dataset_path=dataset_path,
            output=checkpoint_path,
            device=device,
            batch_size=2,
            max_steps=5,
            learning_rate=3.0e-4,
            transition_ratio=1.0,
        )
        run_flow_play(
            checkpoint=checkpoint_path,
            task_mode=TaskMode.WHEEL,
            task_mode_script=[],
            blend_s=0.5,
            num_envs=1,
            device=device,
            viewer="none",
            sample_steps=2,
            max_steps=2,
        )
    print("[flow-smoke] real pipeline passed")


def _make_synthetic_teacher(config: FlowPolicyConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """生成可学习的合成 teacher 序列。"""
    batch = 32
    steps = 12
    obs = torch.randn(batch, steps, config.obs_dim)
    weights = torch.randn(config.obs_dim, config.action_dim) * 0.18
    temporal = torch.linspace(-0.3, 0.3, steps).view(1, steps, 1)
    actions = torch.tanh(obs @ weights + temporal)
    return obs, actions


def _assert_sampler_direction(config: FlowPolicyConfig) -> None:
    """验证反向 Euler 能把噪声动作积分回目标动作。"""
    context = torch.zeros(4, config.rnn_hidden_dim)
    target = torch.linspace(-0.3, 0.3, config.action_dim).expand(context.shape[0], -1)
    noise = torch.full_like(target, 0.75)
    model = _AnalyticVelocityField(config, target)
    sampled = sample_actions_from_context(model, context, steps=16, noise=noise)
    if not torch.allclose(sampled, target, atol=1.0e-6):
        raise RuntimeError("synthetic smoke 失败：采样积分方向错误")


if __name__ == "__main__":
    main()
