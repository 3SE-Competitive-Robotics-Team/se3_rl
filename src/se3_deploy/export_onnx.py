"""把 PyTorch recovery checkpoint 导出为 ONNX Runtime 推理图。"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch
from torch import nn

from se3_sim2sim.policy import PolicyRuntime
from se3_sim2sim.runtime_spec import RuntimeSpec

DEFAULT_RECOVERY_CHECKPOINT = Path("logs/rsl_rl/se3_wheel_leg/2026-06-13_21-35-38/model_4999.pt")
DEFAULT_OUTPUT = Path("logs/deploy/model_4999_recovery_obs34_gru.onnx")


class _OnnxGRUActor(nn.Module):
    """导出用 wrapper：显式传入和返回 GRU hidden。"""

    def __init__(self, runtime: PolicyRuntime) -> None:
        super().__init__()
        if runtime.model is None or not runtime.spec.is_recurrent:
            raise ValueError(f"expected recurrent actor checkpoint, got {runtime.policy_type}")
        self.obs_normalizer = runtime.model.obs_normalizer
        self.rnn = runtime.model.rnn.rnn
        self.mlp = runtime.model.mlp

    def forward(
        self,
        obs: torch.Tensor,
        hidden_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.obs_normalizer(obs)
        rnn_out, hidden_out = self.rnn(normalized.unsqueeze(1), hidden_in)
        action = self.mlp(rnn_out.squeeze(1))
        return action, hidden_out


def export_onnx(checkpoint: Path, output: Path) -> None:
    runtime = PolicyRuntime(checkpoint=checkpoint, device="cpu", runtime=RuntimeSpec())
    if not runtime.spec.is_recurrent or runtime.spec.rnn_type != "gru":
        raise ValueError(f"expected GRU actor checkpoint, got {runtime.policy_type}")

    model = _OnnxGRUActor(runtime).eval()
    obs = torch.zeros(1, runtime.spec.num_obs, dtype=torch.float32)
    hidden = torch.zeros(
        runtime.spec.rnn_num_layers,
        1,
        runtime.spec.rnn_hidden_dim,
        dtype=torch.float32,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (obs, hidden),
        output,
        input_names=["obs", "hidden_in"],
        output_names=["action", "hidden_out"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    _write_metadata(checkpoint, output, runtime)
    print(f"exported {checkpoint} -> {output}")
    print(f"policy={runtime.policy_type} iter={runtime.iteration}")


def _write_metadata(checkpoint: Path, output: Path, runtime: PolicyRuntime) -> None:
    model = onnx.load(output)
    model.doc_string = f"SE3 recovery actor exported from {checkpoint.as_posix()}"
    metadata = {
        "source_checkpoint": checkpoint.as_posix(),
        "iteration": str(runtime.iteration),
        "num_obs": str(runtime.spec.num_obs),
        "num_actions": str(runtime.spec.num_actions),
        "activation": str(runtime.spec.activation),
        "rnn_type": str(runtime.spec.rnn_type),
        "rnn_hidden_dim": str(runtime.spec.rnn_hidden_dim),
        "rnn_num_layers": str(runtime.spec.rnn_num_layers),
    }
    del model.metadata_props[:]
    for key, value in metadata.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value
    onnx.save(model, output)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export recovery checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RECOVERY_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    export_onnx(args.checkpoint, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
