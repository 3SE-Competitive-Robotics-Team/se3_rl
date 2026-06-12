"""用 mjviser 查看左坑右坡固定训练场地和 WheelDog 初始姿态。

用法：
    uv run python scripts/view_gap_ramp_facility.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import viser
from mjviser import Viewer

from se3_train.tasks.wheel_dog.robot_cfg import (
    DOG_BASE_HEIGHT,
    DOG_DEFAULT_JOINT_POS,
    DOG_JOINT_NAMES,
)
from se3_train.terrains.gap_ramp_facility import (
    GapRampFacilitySpec,
    GapRampFacilityTerrainCfg,
)

_MODEL_PATH = Path("assets/robots/minidog/mjcf/minidog_16dof_20kg.xml")


def _build_scene_spec(model_path: Path) -> tuple[mujoco.MjSpec, np.ndarray]:
    """构造只用于可视化的场地+机器人 MuJoCo spec。"""
    spec = mujoco.MjSpec.from_file(str(model_path))

    facility = GapRampFacilitySpec()
    terrain_body = spec.worldbody.add_body(name="terrain")
    terrain_cfg = GapRampFacilityTerrainCfg(
        size=facility.terrain_size,
        facility=facility,
    )
    output = terrain_cfg.function(
        difficulty=0.0,
        spec=spec,
        rng=np.random.default_rng(0),
    )

    terrain_body.pos[:] = -output.origin
    terrain_body.pos[2] = 0.0

    base = spec.body("base_link")
    base.pos[:] = (0.0, 0.0, facility.left_platform_height + DOG_BASE_HEIGHT)

    return spec, np.asarray(output.origin, dtype=np.float64)


def _set_default_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """把机器人重置到 WheelDog 默认站姿。"""
    mujoco.mj_resetData(model, data)
    for joint_name, joint_pos in zip(DOG_JOINT_NAMES, DOG_DEFAULT_JOINT_POS, strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"找不到关节: {joint_name}")
        qpos_adr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_adr] = joint_pos
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="查看左坑右坡训练场地")
    parser.add_argument("--model", type=Path, default=_MODEL_PATH)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--save-xml", type=Path, default=None)
    args = parser.parse_args()

    spec, origin = _build_scene_spec(args.model)
    if args.save_xml is not None:
        args.save_xml.parent.mkdir(parents=True, exist_ok=True)
        args.save_xml.write_text(spec.to_xml(), encoding="utf-8")

    model = spec.compile()
    data = mujoco.MjData(model)
    _set_default_pose(model, data)

    server = viser.ViserServer(port=int(args.port), label="WheelDog Gap Ramp Facility")
    viewer = Viewer(
        model,
        data,
        step_fn=lambda _model, _data: None,
        reset_fn=_set_default_pose,
        server=server,
    )
    print(f"[gap-ramp] viewer: http://localhost:{args.port}/")
    print(f"[gap-ramp] robot origin on left platform: {origin.tolist()}")
    time.sleep(0.2)
    viewer.run()


if __name__ == "__main__":
    main()
