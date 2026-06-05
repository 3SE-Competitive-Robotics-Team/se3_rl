"""se3_flow_match smoke 实验。"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import DataLoader

from se3_shared import ObservationConfig, TaskMode

from .checkpoint import load_flow_checkpoint, save_flow_checkpoint
from .collect import collect_teacher_rollouts
from .config import FlowPolicyConfig
from .dataset import TeacherFlowDataset
from .losses import flow_matching_loss
from .model import FlowVelocityField
from .play import run_flow_play
from .sampler import sample_actions, sample_actions_from_context
from .train import _sample_training_window, _training_loss_weights, train_flow_student


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
    _assert_reset_window_is_supervised()
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
            num_envs=2,
            command_hold_s=1.0,
            command_batches=1,
            coverage="stratified",
            device=device,
        )
        _assert_real_dataset_shape(dataset_path)
        train_flow_student(
            dataset_path=dataset_path,
            output=checkpoint_path,
            device=device,
            batch_size=2,
            max_steps=5,
            learning_rate=3.0e-4,
            transition_ratio=0.25,
            burn_in_steps=4,
            loss_steps=8,
            reset_window_ratio=0.5,
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


def _assert_real_dataset_shape(dataset_path: Path) -> None:
    """检查 tiny 长轨迹数据集格式和固定 command 观测契约。"""
    raw = torch.load(dataset_path, map_location="cpu", weights_only=False)
    obs = raw["obs"]
    actions = raw["actions"]
    commands = raw.get("commands")
    metadata = raw.get("metadata", {})
    if commands is None:
        raise RuntimeError("real smoke 失败：dataset 缺少 commands 字段")
    if tuple(obs.shape[:2]) != tuple(actions.shape[:2]):
        raise RuntimeError(
            f"real smoke 失败：obs/actions [N,T] 不一致 obs={tuple(obs.shape)} "
            f"actions={tuple(actions.shape)}"
        )
    if obs.shape[0] != 4 or commands.shape != (4, 5):
        raise RuntimeError(
            f"real smoke 失败：tiny dataset shape 错误 obs={tuple(obs.shape)} "
            f"commands={tuple(commands.shape)}"
        )
    hold_steps = int(metadata["task_rollouts"][0]["command_hold_steps"])
    if obs.shape[1] != hold_steps or hold_steps < 12:
        raise RuntimeError(
            f"real smoke 失败：hold_steps 错误 obs_T={obs.shape[1]} metadata={hold_steps}"
        )
    scale = torch.tensor(ObservationConfig().command_scale, dtype=obs.dtype)
    expected = commands[:, None, :] * scale
    max_diff = (obs[:, :, 6:11] - expected).abs().max().item()
    if max_diff > 1.0e-5:
        raise RuntimeError(f"real smoke 失败：trajectory 内 command 观测漂移 max_diff={max_diff}")


def _make_synthetic_teacher(config: FlowPolicyConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """生成可学习的合成 teacher 序列。"""
    batch = 32
    steps = 12
    obs = torch.randn(batch, steps, config.obs_dim)
    weights = torch.randn(config.obs_dim, config.action_dim) * 0.18
    temporal = torch.linspace(-0.3, 0.3, steps).view(1, steps, 1)
    actions = torch.tanh(obs @ weights + temporal)
    return obs, actions


def _assert_reset_window_is_supervised() -> None:
    """确认 reset 起点窗口不会被 burn-in mask 清掉。"""
    modes = torch.zeros(4, 12, dtype=torch.long)
    starts = torch.tensor([0, 3, 0, 5], dtype=torch.long)
    weights = _training_loss_weights(modes, starts, burn_in_steps=4)
    if not torch.all(weights[starts == 0, :4] > 0.0):
        raise RuntimeError("synthetic smoke 失败：reset 窗口冷启动动作未参与监督")
    if not torch.all(weights[starts != 0, :4] == 0.0):
        raise RuntimeError("synthetic smoke 失败：非 reset 窗口 burn-in mask 未生效")

    obs = torch.randn(8, 24, 42)
    actions = torch.randn(8, 24, 6)
    dones = torch.zeros(8, 24, dtype=torch.bool)
    sampled_modes = torch.zeros(8, 24, dtype=torch.long)
    *_batch, sampled_starts = _sample_training_window(
        obs,
        actions,
        dones,
        sampled_modes,
        batch_size=8,
        burn_in_steps=4,
        loss_steps=8,
        reset_window_ratio=0.5,
    )
    if not (sampled_starts == 0).any():
        raise RuntimeError("synthetic smoke 失败：训练 batch 未混入 reset 起点窗口")


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
