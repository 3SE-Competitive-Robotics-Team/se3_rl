"""用指定端口启动 Stair CTBC 的 MJLab Viser play。

MJLab 的 se3-play CLI 当前没有暴露 Viser 端口，这里只绕过端口创建逻辑，
环境、runner、policy 加载流程保持和 se3-play 一致。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import viser
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserPlayViewer

TASK_ID = "SE3-WheelLegged-Stair-CTBC-GRU"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=12012)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    configure_torch_backends()
    import se3_train  # noqa: F401  # 注册 se3 任务

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = int(args.num_envs)
    agent_cfg = load_rl_cfg(TASK_ID)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped_env, asdict(agent_cfg), device=args.device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    server = viser.ViserServer(host=args.host, port=int(args.port), label="mjlab")
    print(f"[INFO] Viser listening on http://127.0.0.1:{args.port}/", flush=True)
    ViserPlayViewer(wrapped_env, policy, viser_server=server).run()
    wrapped_env.close()


if __name__ == "__main__":
    main()
