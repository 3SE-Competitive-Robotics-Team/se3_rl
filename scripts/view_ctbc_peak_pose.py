"""Open a local MuJoCo viewer at the CTBC peak pose.

This is a visualization-only helper.  It does not run policy inference or
modify the training task.  The pose calculation reuses the same CTBC
equivalence path as ``plot_ctbc_lift_linkage.py``:

    old output-joint CTBC bias -> target active-rod action -> output joints

Examples:
    uv run python scripts/view_ctbc_peak_pose.py --mode both
    uv run python scripts/view_ctbc_peak_pose.py --mode left --height 0.27
    uv run python scripts/view_ctbc_peak_pose.py --mode cycle
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
from plot_ctbc_lift_linkage import (
    DEFAULT_FF_AMPLITUDE,
    DEFAULT_PHASE,
    MJCF_PATH,
    _ctbc_pose,
)

from se3_shared import RobotConfig

_LEG_JOINT_NAMES = ("lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint")
_WHEEL_JOINT_NAMES = ("l_wheel_Joint", "r_wheel_Joint")
_CYCLE_MODES = ("default", "left", "right", "both")


def _load_model(mjcf_path: Path, *, hide_floor: bool) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(mjcf_path))
    if hide_floor:
        for geom in list(spec.worldbody.geoms):
            if geom.name == "floor":
                spec.delete(geom)
                break
    return spec.compile()


def _joint_qpos_address(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise KeyError(f"MJCF joint not found: {name}")
    return int(model.jnt_qposadr[joint_id])


def _output_for_mode(default_output: np.ndarray, peak_output: np.ndarray, mode: str) -> np.ndarray:
    output = np.asarray(default_output, dtype=np.float64).reshape(4).copy()
    peak = np.asarray(peak_output, dtype=np.float64).reshape(4)
    if mode == "default":
        return output
    if mode == "left":
        output[0:2] = peak[0:2]
        return output
    if mode == "right":
        output[2:4] = peak[2:4]
        return output
    if mode == "both":
        return peak.copy()
    raise ValueError(f"unsupported mode: {mode}")


def _set_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    output_pos: np.ndarray,
    base_height: float,
) -> None:
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    if model.nq < 7:
        raise RuntimeError("expected a floating-base robot with nq >= 7")
    data.qpos[:7] = np.array([0.0, 0.0, float(base_height), 1.0, 0.0, 0.0, 0.0])
    joint_values = dict(zip(_LEG_JOINT_NAMES, output_pos, strict=True))
    joint_values.update({name: 0.0 for name in _WHEEL_JOINT_NAMES})
    for name, value in joint_values.items():
        data.qpos[_joint_qpos_address(model, name)] = float(value)
    mujoco.mj_forward(model, data)


def _print_pose_summary(args: argparse.Namespace, pose, mode: str, output: np.ndarray) -> None:
    print(f"mode={mode}")
    print(f"height={args.height:.3f} phase={args.phase:.3f} ff_amplitude={args.ff_amplitude:.3f}")
    print("old_action_bias=", np.array2string(pose.old_action_bias, precision=6))
    print("output_delta=", np.array2string(pose.output_delta, precision=6))
    print("default_output=", np.array2string(pose.default_output, precision=6))
    print("peak_output=", np.array2string(pose.realizable_output, precision=6))
    print("shown_output=", np.array2string(output, precision=6))
    print("action_delta=", np.array2string(pose.action_delta, precision=6))
    print("active_target=", np.array2string(pose.active_target, precision=6))
    print("active_clamped=", pose.active_clamped)


def _configure_camera(viewer) -> None:
    viewer.cam.lookat[:] = np.array([-0.04, 0.0, 0.18], dtype=np.float64)
    viewer.cam.distance = 1.25
    viewer.cam.azimuth = 90.0
    viewer.cam.elevation = -18.0


def main() -> None:
    parser = argparse.ArgumentParser(description="View the SerialLeg CTBC peak pose locally.")
    parser.add_argument("--mode", choices=(*_CYCLE_MODES, "cycle"), default="both")
    parser.add_argument("--height", type=float, default=0.30, help="Command/base height in meters.")
    parser.add_argument("--phase", type=float, default=DEFAULT_PHASE, help="0.5 is CTBC peak.")
    parser.add_argument("--ff-amplitude", type=float, default=DEFAULT_FF_AMPLITUDE)
    parser.add_argument("--kff", type=float, default=1.0)
    parser.add_argument("--cycle-seconds", type=float, default=2.5)
    parser.add_argument("--hide-floor", action="store_true")
    parser.add_argument("--mjcf", type=Path, default=MJCF_PATH)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    pose = _ctbc_pose(
        phase=args.phase,
        command_height=args.height,
        ff_amplitude=args.ff_amplitude,
        kff=args.kff,
        robot_cfg=RobotConfig(),
    )
    mode = "both" if args.mode == "cycle" else args.mode
    output = _output_for_mode(pose.default_output, pose.realizable_output, mode)
    _print_pose_summary(args, pose, mode, output)
    if args.print_only:
        return

    import mujoco.viewer

    model = _load_model(args.mjcf, hide_floor=args.hide_floor)
    data = mujoco.MjData(model)
    _set_pose(model, data, output_pos=output, base_height=args.height)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        _configure_camera(viewer)
        last_mode = mode
        next_switch = time.monotonic() + max(0.1, float(args.cycle_seconds))
        while viewer.is_running():
            if args.mode == "cycle" and time.monotonic() >= next_switch:
                current_idx = _CYCLE_MODES.index(last_mode)
                last_mode = _CYCLE_MODES[(current_idx + 1) % len(_CYCLE_MODES)]
                output = _output_for_mode(pose.default_output, pose.realizable_output, last_mode)
                print(f"showing mode={last_mode}", flush=True)
                _set_pose(model, data, output_pos=output, base_height=args.height)
                next_switch = time.monotonic() + max(0.1, float(args.cycle_seconds))
            viewer.sync()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
