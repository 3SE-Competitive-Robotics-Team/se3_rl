"""从训练 checkpoint（model_N.pt）离线导出带部署元数据的 ONNX，复用 runner.save() 的同一条导出路径。

用途：nulltask1 上的训练把 .pt 传到 W&B、不传 ONNX；4 卡机上的镜像转换器按 run 的 commit
开 worktree 后调用本脚本，在 CPU 上以 num_envs=1 构造同一任务的 env 与 runner，加载 checkpoint
再调用 Se3ProfiledOnPolicyRunner.export_policy_to_onnx，产物与训练端 save() 导出的 model_N.onnx
同构（权重、图、元数据一致，见 scripts/compare_onnx.py）。

用法：
    uv run python scripts/export_onnx_from_checkpoint.py --task <task_id> \
        --output-dir <run>/onnx --checkpoint <run>/model_100.pt [--checkpoint ...]

一次进程只构造一次 env（MuJoCo-Warp CPU 编译约 1 分钟），可批量传入同一 run 的多个 checkpoint。
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

# 必须在 import torch / mjlab 之前屏蔽 GPU：导出只需要 CPU，避免和同机训练抢显存。
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("WANDB_MODE", "disabled")


def _build_runner(task_id: str, log_dir: Path):
    """按 mjlab.scripts.train.run_train 的方式构造 CPU 单环境与任务 runner。"""
    from dataclasses import asdict

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    import se3_train  # noqa: F401  注册全部任务

    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = 1
    # 离线导出不写训练日志，也绝不触发 W&B。
    agent_cfg.logger = "tensorboard"
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    return runner_cls(env, asdict(agent_cfg), str(log_dir), "cpu")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--task",
        required=True,
        help="注册的任务 ID，例如 SE3-WheelLegged-Flat-MLP",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="ONNX 输出目录（通常是 <run>/onnx）"
    )
    parser.add_argument(
        "--checkpoint", action="append", required=True, type=Path, help="model_N.pt，可重复"
    )
    parser.add_argument("--overwrite", action="store_true", help="已存在同名 ONNX 时覆盖")
    args = parser.parse_args()

    for checkpoint in args.checkpoint:
        if not checkpoint.is_file():
            print(f"错误：checkpoint 不存在 {checkpoint}", file=sys.stderr)
            return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="se3-export-") as tmp:
        runner = _build_runner(args.task, Path(tmp))
        exported = 0
        for checkpoint in args.checkpoint:
            target = args.output_dir / f"{checkpoint.stem}.onnx"
            if target.exists() and not args.overwrite:
                print(f"[跳过] 已存在 {target}")
                continue
            runner.load(str(checkpoint), map_location="cpu")
            runner.export_policy_to_onnx(str(args.output_dir), filename=target.name)
            print(
                f"[导出] {checkpoint.name} (iter {runner.current_learning_iteration}) -> {target}"
                f" {target.stat().st_size} bytes"
            )
            exported += 1
    print(f"[完成] 任务 {args.task}，导出 {exported} 个 ONNX 到 {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
