"""启动 6 个固定场景的 Stair CTBC Viser。

场景顺序:
0: 17 度上坡
1: 43 度上坡
2: 上金字塔台阶
3: 下金字塔台阶
4: 上二级台阶
5: 下二级台阶
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import viser
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.terrains import (
    BoxInvertedPyramidStairsTerrainCfg,
    TerrainEntityCfg,
    TerrainGeneratorCfg,
)
from mjlab.utils.lab_api.math import quat_from_euler_xyz
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserPlayViewer

from se3_train.mdp import events
from se3_train.tasks.stair_ctbc.terrains import BoxRampTerrainCfg, BoxStageStairsTerrainCfg

TASK_ID = "SE3-WheelLegged-Stair-CTBC-GRU"
SCENE_NAMES = (
    "ramp_17deg_up",
    "ramp_43deg_up",
    "pyramid_up",
    "pyramid_down",
    "stage_up",
    "stage_down",
)


def _latest_model(path: Path) -> Path:
    models = list(path.glob("model_*.pt"))
    if not models:
        raise FileNotFoundError(f"no model_*.pt under {path}")
    return max(models, key=lambda item: int(item.stem.removeprefix("model_")))


def _set_fixed_scene_origins(env: ManagerBasedRlEnv) -> None:
    terrain = env.scene["terrain"]
    assert terrain.terrain_origins is not None
    origins = terrain.terrain_origins[0].clone()

    fixed = origins.clone()
    yaws = torch.zeros(6, device=env.device)

    # 金字塔上行：从底部中心后退一点，避免出生时直接顶住第一圈 20 cm 竖直台阶。
    fixed[2] = origins[2] + torch.tensor((-0.55, 0.0, 1.0), device=env.device)

    # 金字塔下行：从右侧高台出发，command 用负 vx 朝中心下台阶。
    fixed[3] = origins[3] + torch.tensor((2.75, 0.0, 1.0), device=env.device)

    # 二级台阶下行：从第二级高台出发，command 用负 vx 朝第一级/地面走。
    fixed[5] = origins[5] + torch.tensor((0.98, 0.0, 0.35), device=env.device)

    terrain.env_origins[:] = fixed
    terrain.terrain_levels[:] = 0
    terrain.terrain_types[:] = torch.arange(6, device=env.device)
    env.scene.env_origins[:] = fixed

    _set_fixed_commands(env)
    _reset_robot_pose(env, fixed, yaws)
    _reset_fixed_joints(env)
    env._se3_six_env_scene_names = SCENE_NAMES


def _reset_robot_pose(env: ManagerBasedRlEnv, origins: torch.Tensor, yaws: torch.Tensor) -> None:
    robot = env.scene["robot"]
    root = robot.data.default_root_state[: env.num_envs].clone()
    root[:, 0:3] += origins
    root[:, 2] = origins[:, 2] + 0.39
    quat = quat_from_euler_xyz(
        torch.zeros(env.num_envs, device=env.device),
        torch.zeros(env.num_envs, device=env.device),
        yaws,
    )
    root[:, 3:7] = quat
    root[:, 7:13] = 0.0
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    robot.write_root_link_pose_to_sim(root[:, 0:7], env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(root[:, 7:13], env_ids=env_ids)


def _set_fixed_commands(env: ManagerBasedRlEnv) -> None:
    term = env.command_manager.get_term("velocity_height")
    cmd = term.command
    cmd[:, :] = 0.0
    cmd[:, 0] = 0.25
    cmd[(3, 5), 0] = -0.25
    cmd[:, 4] = 0.39
    if cmd.shape[1] > 5:
        cmd[:, 5:] = 0.0


def _reset_fixed_joints(env: ManagerBasedRlEnv) -> None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    events.reset_joints(
        env,
        env_ids,
        asset_cfg=SceneEntityCfg("robot"),
        joint_offset_range=0.0,
        joint_vel_range=(0.0, 0.0),
        joint_randomization_prob=0.0,
        full_joint_randomization=False,
        align_root_height_to_wheels=True,
        height_conditioned_default=True,
        command_name="velocity_height",
        terrain_height_sensor_names=(
            "left_wheel_center_height_sensor",
            "right_wheel_center_height_sensor",
        ),
        allow_wheel_clearance_lowering=True,
        max_wheel_clearance_adjustment=0.25,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--port", type=int, default=12012)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    configure_torch_backends()
    import se3_train  # noqa: F401  # 注册 se3 任务

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 6
    env_cfg.terminations = {}
    env_cfg.viewer.env_idx = 0
    env_cfg.viewer.max_extra_envs = 5
    env_cfg.viewer.distance = 35.0
    env_cfg.viewer.elevation = -65.0
    env_cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            curriculum=True,
            size=(8.0, 8.0),
            border_width=20.0,
            border_height=1.0,
            num_rows=1,
            num_cols=6,
            difficulty_range=(1.0, 1.0),
            add_lights=True,
            sub_terrains={
                "ramp_17deg_up": BoxRampTerrainCfg(
                    proportion=1.0,
                    size=(8.0, 8.0),
                    slope_deg=17.0,
                    height=0.35,
                    top_platform_length=1.5,
                ),
                "ramp_43deg_up": BoxRampTerrainCfg(
                    proportion=1.0,
                    size=(8.0, 8.0),
                    slope_deg=43.0,
                    height=0.40,
                ),
                "pyramid_up": BoxInvertedPyramidStairsTerrainCfg(
                    proportion=1.0,
                    size=(8.0, 8.0),
                    step_height_range=(0.05, 0.20),
                    step_width=0.5,
                    platform_width=2.0,
                    border_width=1.0,
                ),
                "pyramid_down": BoxInvertedPyramidStairsTerrainCfg(
                    proportion=1.0,
                    size=(8.0, 8.0),
                    step_height_range=(0.05, 0.20),
                    step_width=0.5,
                    platform_width=2.0,
                    border_width=1.0,
                ),
                "stage_up": BoxStageStairsTerrainCfg(proportion=1.0, size=(8.0, 8.0)),
                "stage_down": BoxStageStairsTerrainCfg(proportion=1.0, size=(8.0, 8.0)),
            },
        ),
        max_init_terrain_level=0,
        debug_vis=True,
    )

    agent_cfg = load_rl_cfg(TASK_ID)
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
    if checkpoint is None:
        checkpoint = _latest_model(
            Path("logs/rsl_rl/se3_wheel_leg_stair_ctbc/2026-06-08_01-09-10")
        ).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    _set_fixed_scene_origins(wrapped_env.unwrapped)

    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped_env, asdict(agent_cfg), device=args.device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    server = viser.ViserServer(host=args.host, port=int(args.port), label="mjlab")
    print("[INFO] Six-env scene order:", ", ".join(SCENE_NAMES), flush=True)
    print(f"[INFO] Viser listening on http://127.0.0.1:{args.port}/", flush=True)
    ViserPlayViewer(wrapped_env, policy, viser_server=server).run()
    wrapped_env.close()


if __name__ == "__main__":
    main()
