"""将 GRU actor checkpoint 导出为 ONNX 格式。

用法: uv run python .agents/skills/export-onnx/scripts/export.py [--checkpoint <path>] [--output <path>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn


class EmpiricalNormalizer(nn.Module):
    """观测归一化层，权重从 checkpoint 加载。"""

    def __init__(self, num_obs: int, eps: float = 1e-2) -> None:
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("_mean", torch.zeros(1, int(num_obs)))
        self.register_buffer("_std", torch.ones(1, int(num_obs)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + self.eps)


class DeterministicGRUActor(nn.Module):
    """GRU 推理模型：obs_normalizer → GRU → MLP head。

    与 sim2sim 中 _DeterministicGRUActor 结构完全一致。
    """

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        rnn_hidden_dim: int = 512,
        rnn_num_layers: int = 1,
        actor_hidden_dims: tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.obs_normalizer = EmpiricalNormalizer(num_obs)
        self.rnn = nn.GRU(
            input_size=num_obs,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
        )
        self.mlp = self._build_mlp(rnn_hidden_dim, num_actions, actor_hidden_dims, activation)
        self._hidden_dim = rnn_hidden_dim
        self._num_layers = rnn_num_layers

    def forward(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """单步推理，支持任意 batch 维度。

        Args:
            obs: (batch, num_obs) 观测
            hidden: (num_layers, batch, hidden_dim) GRU 隐状态

        Returns:
            action: (batch, num_actions) 动作输出
            new_hidden: (num_layers, batch, hidden_dim) 新隐状态
        """
        normalized = self.obs_normalizer(obs)
        rnn_input = normalized.unsqueeze(1)  # (1, 1, num_obs)
        rnn_out, new_hidden = self.rnn(rnn_input, hidden)
        action = self.mlp(rnn_out.squeeze(1))
        return action, new_hidden

    @staticmethod
    def _build_mlp(
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str,
    ) -> nn.Sequential:
        dims = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(_make_activation(activation))
        return nn.Sequential(*layers)


def _make_activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "sigmoid":
        return nn.Sigmoid()
    if normalized == "leaky_relu":
        return nn.LeakyReLU()
    if normalized == "selu":
        return nn.SELU()
    if normalized == "mish":
        return nn.Mish()
    raise ValueError(f"unsupported activation: {name}")


def extract_actor_state_dict(payload: dict) -> dict[str, torch.Tensor]:
    """从 checkpoint payload 中提取 actor state_dict。"""
    actor = payload.get("actor_state_dict")
    if isinstance(actor, dict):
        return _normalize_keys({k: v for k, v in actor.items() if isinstance(v, torch.Tensor)})

    model = payload.get("model_state_dict")
    if isinstance(model, dict):
        model_state = {k: v for k, v in model.items() if isinstance(v, torch.Tensor)}
        prefix = "actor."
        actor_state = {
            k.removeprefix(prefix): v for k, v in model_state.items() if k.startswith(prefix)
        }
        if actor_state:
            return _normalize_keys(actor_state)

    raise KeyError("checkpoint 中找不到 actor_state_dict")


def _normalize_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """将 rsl_rl 的扁平数字键改为 mlp.0.weight 形式。"""
    if any(k.startswith("mlp.") for k in state_dict):
        return state_dict
    result: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        parts = key.split(".", 1)
        if parts[0].isdigit():
            result[f"mlp.{key}"] = value
        else:
            result[key] = value
    return result


def infer_spec(actor_state: dict[str, torch.Tensor]) -> dict:
    """从 state_dict 推断模型规格。"""
    # 检测 GRU 参数（兼容 rnn.weight_ih_l0 和 rnn.rnn.weight_ih_l0）
    ih_candidates = ["rnn.rnn.weight_ih_l0", "rnn.weight_ih_l0"]
    ih_key = next((k for k in ih_candidates if k in actor_state), None)
    if ih_key is not None:
        prefix = ih_key.removesuffix("weight_ih_l0")  # "rnn.rnn." 或 "rnn."
        gate_hidden = actor_state[ih_key].shape[0]
        hidden_dim = gate_hidden // 3  # GRU: 3 gates
        num_layers = 0
        while f"{prefix}weight_ih_l{num_layers}" in actor_state:
            num_layers += 1
        rnn_type = "gru"
    else:
        hidden_dim = 0
        num_layers = 0
        rnn_type = None

    # num_obs 从 normalizer mean 推断
    num_obs = int(actor_state["obs_normalizer._mean"].shape[1])

    # MLP 各层从 mlp.{i}.weight 推断
    mlp_layers: list[tuple[int, int]] = []
    for key, tensor in sorted(actor_state.items()):
        if not key.startswith("mlp.") or not key.endswith(".weight") or tensor.ndim != 2:
            continue
        parts = key.split(".")
        if len(parts) != 3 or not parts[1].isdigit():
            continue
        idx = int(parts[1])
        mlp_layers.append((idx, int(tensor.shape[0]), int(tensor.shape[1])))
    mlp_layers.sort()

    num_actions = mlp_layers[-1][1]
    hidden_dims = tuple(out for _, out, _ in mlp_layers[:-1])

    return {
        "num_obs": num_obs,
        "num_actions": num_actions,
        "rnn_type": rnn_type,
        "rnn_hidden_dim": hidden_dim,
        "rnn_num_layers": num_layers,
        "actor_hidden_dims": hidden_dims,
    }


def export_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    opset_version: int = 18,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"加载 checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor_state = extract_actor_state_dict(payload)
    spec = infer_spec(actor_state)

    iteration = payload.get("iter", "unknown")
    print(f"iter: {iteration}")
    print(f"num_obs: {spec['num_obs']}, num_actions: {spec['num_actions']}")
    print(
        f"rnn_type: {spec['rnn_type']}, hidden_dim: {spec['rnn_hidden_dim']}, layers: {spec['rnn_num_layers']}"
    )
    print(f"actor_hidden_dims: {spec['actor_hidden_dims']}")

    if spec["rnn_type"] is None:
        raise ValueError("仅支持 GRU 模型导出，此 checkpoint 不含 RNN")

    model = DeterministicGRUActor(
        num_obs=spec["num_obs"],
        num_actions=spec["num_actions"],
        rnn_hidden_dim=spec["rnn_hidden_dim"],
        rnn_num_layers=spec["rnn_num_layers"],
        actor_hidden_dims=spec["actor_hidden_dims"],
    )
    model.eval()

    # 加载权重: checkpoint 中 rnn.rnn.* 映射到模型的 rnn.* (nn.GRU)
    remapped: dict[str, torch.Tensor] = {}
    for key, value in actor_state.items():
        if key.startswith("rnn.rnn."):
            remapped[key.replace("rnn.rnn.", "rnn.")] = value
        else:
            remapped[key] = value

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    # 检查关键权重是否全部加载
    required_missing = [k for k in missing if k.startswith("mlp.") or k.startswith("rnn.")]
    if required_missing:
        raise ValueError(f"关键权重缺失: {required_missing}")
    print(f"权重加载完成 (missing non-critical: {len(missing)}, unexpected: {len(unexpected)})")

    # 准备导出输入
    num_obs = spec["num_obs"]
    hidden_dim = spec["rnn_hidden_dim"]
    num_layers = spec["rnn_num_layers"]

    dummy_obs = torch.randn(1, num_obs)
    dummy_hidden = torch.zeros(num_layers, 1, hidden_dim)

    print(f"导出 ONNX: {output_path}")
    tmp_path = output_path.with_suffix(".tmp.onnx")
    torch.onnx.export(
        model,
        (dummy_obs, dummy_hidden),
        tmp_path,
        input_names=["obs", "hidden_in"],
        output_names=["action", "hidden_out"],
        dynamic_axes={
            "obs": {0: "batch"},
            "hidden_in": {1: "batch"},
            "action": {0: "batch"},
            "hidden_out": {1: "batch"},
        },
        opset_version=max(opset_version, 18),  # torch 2.x 最低有效 opset 为 18
        verbose=False,
    )

    # 合并外部数据为单文件 ONNX
    import onnx
    from onnx.external_data_helper import load_external_data_for_model

    model_onnx = onnx.load(tmp_path)
    load_external_data_for_model(model_onnx, tmp_path.parent)
    onnx.save(model_onnx, output_path, save_as_external_data=False)

    # 清理临时文件
    tmp_path.unlink(missing_ok=True)
    for f in tmp_path.parent.glob(tmp_path.name + ".data"):
        f.unlink(missing_ok=True)
    print("导出完成")

    # 验证：跑一次推理对比
    with torch.no_grad():
        action_pt, hidden_pt = model(dummy_obs, dummy_hidden)
    print(f"PyTorch 推理: action={action_pt.numpy().round(4)}, hidden_norm={hidden_pt.norm():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 GRU actor 为 ONNX")
    parser.add_argument(
        "--checkpoint",
        default="assets/base_model/model_2999_flat_20260702.pt",
        help="checkpoint 路径",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出 ONNX 路径 (默认与 checkpoint 同名 .onnx)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset 版本",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output = Path(args.output) if args.output else checkpoint.with_suffix(".onnx")

    export_onnx(checkpoint, output, opset_version=args.opset)


if __name__ == "__main__":
    main()
