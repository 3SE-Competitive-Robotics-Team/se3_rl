"""诊断六场景 Viser 中 env2 金字塔上行出生和早期接触。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.terrains import (
    BoxInvertedPyramidStairsTerrainCfg,
    TerrainEntityCfg,
    TerrainGeneratorCfg,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tmp.run_stair_ctbc_viser_six_envs import (
    TASK_ID,
    _set_fixed_scene_origins,
)


def _load_env():
    from se3_train.tasks.stair_ctbc.terrains import BoxRampTerrainCfg, BoxStageStairsTerrainCfg

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 6
    env_cfg.terminations = {}
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
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode=None)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASK_ID).clip_actions)
    _set_fixed_scene_origins(wrapped.unwrapped)
    return wrapped


def _wheel_pos(env):
    robot = env.unwrapped.scene["robot"]
    body_ids, _ = robot.find_bodies(("l_wheel_Link", "r_wheel_Link"), preserve_order=True)
    return robot.data.body_link_pos_w[:, body_ids, :].detach().cpu()


def _sensor_xy(env, name: str):
    sensor = env.unwrapped.scene[name]
    if sensor.data.force is None:
        return torch.zeros(env.unwrapped.num_envs, 2)
    force_xy = sensor.data.force[..., :2]
    return (
        torch.linalg.norm(force_xy, dim=-1)
        .reshape(env.unwrapped.num_envs, -1)[:, :2]
        .detach()
        .cpu()
    )


def main() -> None:
    import se3_train  # noqa: F401

    env = _load_env()
    raw_env = env.unwrapped
    robot = raw_env.scene["robot"]
    env2 = 2
    print("origin", raw_env.scene.env_origins[env2].detach().cpu().tolist())
    print("root0", robot.data.root_link_pos_w[env2].detach().cpu().tolist())
    print("wheel0", _wheel_pos(env)[env2].tolist())
    print(
        "terrain_type_level",
        int(raw_env.scene["terrain"].terrain_types[env2]),
        int(raw_env.scene["terrain"].terrain_levels[env2]),
    )

    action = torch.zeros((raw_env.num_envs, 6), device=raw_env.device)
    for step in range(40):
        env.step(action)
        if step in (0, 1, 2, 5, 10, 20, 39):
            state = getattr(raw_env, "stair_climb_state", None)
            diag = state.diag() if state is not None else {}
            root = robot.data.root_link_pos_w[env2].detach().cpu()
            vel = robot.data.root_link_vel_w[env2].detach().cpu()
            wheel = _wheel_pos(env)[env2]
            print(
                "step",
                step,
                "root",
                [round(float(x), 4) for x in root],
                "vz",
                round(float(vel[2]), 4),
                "wheel_z",
                [round(float(x), 4) for x in wheel[:, 2]],
                "wheel_xy",
                [round(float(x), 3) for x in _sensor_xy(env, "wheel_sensor")[env2]],
                "riser_xy",
                [round(float(x), 3) for x in _sensor_xy(env, "wheel_riser_sensor")[env2]],
                "active",
                bool(state.active_mask()[env2]) if state is not None else False,
                "bias_mean",
                round(float(diag.get("Stair/ctbc_bias_abs_mean", 0.0)), 4),
            )
    env.close()


if __name__ == "__main__":
    main()
