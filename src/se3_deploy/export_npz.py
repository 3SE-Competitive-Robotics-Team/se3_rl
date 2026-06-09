"""把 PyTorch recovery checkpoint 导出为 NX 轻量 NumPy 推理权重。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from se3_sim2sim.policy import PolicyRuntime
from se3_sim2sim.runtime_spec import RuntimeSpec

DEFAULT_RECOVERY_CHECKPOINT = Path("assets/base_model/model_5999_gru.pt")
DEFAULT_OUTPUT = Path("logs/deploy/model_5999_recovery_gru.npz")


def export_npz(checkpoint: Path, output: Path) -> None:
    runtime = PolicyRuntime(checkpoint=checkpoint, device="cpu", runtime=RuntimeSpec())
    if not runtime.spec.is_recurrent or runtime.spec.rnn_type != "gru":
        raise ValueError(f"expected GRU actor checkpoint, got {runtime.policy_type}")
    actor = runtime.actor_state_dict
    linear_layers = sorted(
        int(key.split(".")[1])
        for key in actor
        if key.startswith("mlp.") and key.endswith(".weight")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "iteration": np.asarray(int(runtime.iteration), dtype=np.int64),
        "num_obs": np.asarray(runtime.spec.num_obs, dtype=np.int64),
        "num_actions": np.asarray(runtime.spec.num_actions, dtype=np.int64),
        "activation": np.asarray(runtime.spec.activation),
        "normalizer_eps": np.asarray(1.0e-2, dtype=np.float32),
        "obs_mean": tensor_to_numpy(actor["obs_normalizer._mean"]).reshape(runtime.spec.num_obs),
        "obs_std": tensor_to_numpy(actor["obs_normalizer._std"]).reshape(runtime.spec.num_obs),
        "rnn_type": np.asarray(runtime.spec.rnn_type),
        "rnn_hidden_dim": np.asarray(runtime.spec.rnn_hidden_dim, dtype=np.int64),
        "rnn_num_layers": np.asarray(runtime.spec.rnn_num_layers, dtype=np.int64),
        "rnn_weight_ih_l0": tensor_to_numpy(actor["rnn.rnn.weight_ih_l0"]),
        "rnn_weight_hh_l0": tensor_to_numpy(actor["rnn.rnn.weight_hh_l0"]),
        "rnn_bias_ih_l0": tensor_to_numpy(actor["rnn.rnn.bias_ih_l0"]),
        "rnn_bias_hh_l0": tensor_to_numpy(actor["rnn.rnn.bias_hh_l0"]),
        "mlp_num_linear": np.asarray(len(linear_layers), dtype=np.int64),
    }
    for out_idx, layer_idx in enumerate(linear_layers):
        arrays[f"mlp_{out_idx}_weight"] = tensor_to_numpy(actor[f"mlp.{layer_idx}.weight"])
        arrays[f"mlp_{out_idx}_bias"] = tensor_to_numpy(actor[f"mlp.{layer_idx}.bias"])
    np.savez_compressed(output, **arrays)
    print(f"exported {checkpoint} -> {output}")
    print(f"policy={runtime.policy_type} iter={runtime.iteration}")


def tensor_to_numpy(tensor: object) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=True)  # type: ignore[attr-defined]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export recovery checkpoint to NumPy NPZ.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RECOVERY_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    export_npz(args.checkpoint, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
