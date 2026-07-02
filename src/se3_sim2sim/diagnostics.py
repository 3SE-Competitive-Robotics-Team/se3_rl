"""Model and runtime diagnostics used by the sim2sim workflow."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True, slots=True)
class DiagnosticIssue:
    severity: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "code": self.code, "message": self.message}


def model_diagnostics(model: mujoco.MjModel) -> dict[str, object]:
    joint_names = _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    actuator_names = _names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    wheel = _wheel_collision_diagnostics(model)
    issues = _model_issues(wheel)
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "timestep": float(model.opt.timestep),
        "joint_names": joint_names,
        "actuator_names": actuator_names,
        "wheel_collision": wheel,
        "issues": [issue.to_dict() for issue in issues],
    }


def rollout_diagnostics(samples: list[dict[str, float]]) -> dict[str, object]:
    if not samples:
        return {"samples": 0}
    heights = np.asarray([s["height"] for s in samples], dtype=np.float64)
    tilts = np.asarray([s["tilt_deg"] for s in samples], dtype=np.float64)
    rewards = np.asarray([s["reward"] for s in samples], dtype=np.float64)
    result: dict[str, object] = {
        "samples": len(samples),
        "height": _stats(heights),
        "tilt_deg": _stats(tilts),
        "reward": _stats(rewards),
    }
    if "base_lin_vel_x" in samples[0]:
        base_vx = np.asarray([s["base_lin_vel_x"] for s in samples], dtype=np.float64)
        result["base_lin_vel_x"] = _stats(base_vx)
    if "wheel_lin_vel" in samples[0]:
        wheel_vx = np.asarray([s["wheel_lin_vel"] for s in samples], dtype=np.float64)
        result["wheel_lin_vel"] = _stats(wheel_vx)
    if "wheel_clearance" in samples[0]:
        wheel_clearance = np.asarray([s["wheel_clearance"] for s in samples], dtype=np.float64)
        result["wheel_clearance"] = _stats(wheel_clearance)
    if "wheel_clearance_left" in samples[0]:
        left_clearance = np.asarray([s["wheel_clearance_left"] for s in samples], dtype=np.float64)
        result["wheel_clearance_left"] = _stats(left_clearance)
    if "wheel_clearance_right" in samples[0]:
        right_clearance = np.asarray(
            [s["wheel_clearance_right"] for s in samples], dtype=np.float64
        )
        result["wheel_clearance_right"] = _stats(right_clearance)
        if "wheel_clearance_left" in samples[0]:
            left_clearance = np.asarray(
                [s["wheel_clearance_left"] for s in samples], dtype=np.float64
            )
            result["wheel_clearance_abs_diff"] = _stats(np.abs(left_clearance - right_clearance))
    for key in (
        "base_x",
        "base_y",
        "wheel_x_left",
        "wheel_x_right",
        "wheel_z_left",
        "wheel_z_right",
        "wheel_bottom_z_left",
        "wheel_bottom_z_right",
        "stair_half_width_m",
        "leg_clearance",
        "base_clearance",
        "wheel_lateral_distance",
        "wheel_fore_aft_offset",
        "leg_mirror_error",
    ):
        if key in samples[0]:
            values = np.asarray([s[key] for s in samples], dtype=np.float64)
            result[key] = _stats(values)
    for key in (
        "wheel_contact",
        "wheel_full_contact",
        "wheel_contact_left",
        "wheel_contact_right",
        "wheel_stair_contact_left",
        "wheel_stair_contact_right",
        "wheel_floor_contact_left",
        "wheel_floor_contact_right",
        "leg_contact",
        "leg_contact_left",
        "leg_contact_right",
        "base_contact",
        "nonwheel_contact",
    ):
        if key in samples[0]:
            values = np.asarray([s[key] for s in samples], dtype=np.float64)
            result[key] = _stats(values)
            result[f"{key}_rate"] = float(np.mean(values > 0.5))
    for key in (
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "roll_rate_rad_s",
        "pitch_rate_rad_s",
        "yaw_rate_rad_s",
        "reset_floor_lift_m",
        "command_lin_vel_x",
        "command_yaw_rate",
        "action_delta_l2",
        "action_delta_max_abs",
        "action_delta_sq_sum",
        "applied_action_delta_l2",
        "applied_action_delta_max_abs",
        "applied_action_delta_sq_sum",
        "ctbc_trigger",
        "ctbc_left_active",
        "ctbc_right_active",
        "ctbc_phase_left",
        "ctbc_phase_right",
        "ctbc_contact_left",
        "ctbc_contact_right",
        "ctbc_complete_ff_cycles",
        "rc_switch_r",
        "output_enabled",
        "rc_switch_event",
        "rc_policy_reset",
        "gru_hidden_reset",
        "gru_hidden_norm_before_reset",
        "gru_hidden_norm_after_reset",
        "rc_off_mode_no_torque",
        "stair_hold_active",
    ):
        if key in samples[0]:
            values = np.asarray([s[key] for s in samples], dtype=np.float64)
            result[key] = _stats(values)
    stair = _stair_climb_diagnostics(samples)
    if stair:
        result["stair_climb"] = stair
    result["final"] = samples[-1]
    return result


def _stair_climb_diagnostics(samples: list[dict[str, float]]) -> dict[str, object]:
    required = (
        "wheel_x_left",
        "wheel_x_right",
        "wheel_bottom_z_left",
        "wheel_bottom_z_right",
        "stair_step_height_m",
        "stair_step_depth_m",
        "stair_start_x_m",
        "stair_step_count",
        "wheel_contact_left",
        "wheel_contact_right",
        "wheel_stair_contact_left",
        "wheel_stair_contact_right",
        "nonwheel_contact",
        "base_lin_vel_x",
        "wheel_lin_vel",
    )
    if not all(key in samples[0] for key in required):
        return {}

    time = np.asarray([s["time"] for s in samples], dtype=np.float64)
    left_x = np.asarray([s["wheel_x_left"] for s in samples], dtype=np.float64)
    right_x = np.asarray([s["wheel_x_right"] for s in samples], dtype=np.float64)
    left_z = np.asarray([s["wheel_bottom_z_left"] for s in samples], dtype=np.float64)
    right_z = np.asarray([s["wheel_bottom_z_right"] for s in samples], dtype=np.float64)
    left_contact = (
        np.asarray([s["wheel_stair_contact_left"] for s in samples], dtype=np.float64) > 0.5
    )
    right_contact = (
        np.asarray([s["wheel_stair_contact_right"] for s in samples], dtype=np.float64) > 0.5
    )
    nonwheel_contact = np.asarray([s["nonwheel_contact"] for s in samples], dtype=np.float64) > 0.5
    base_y = np.asarray([s.get("base_y", 0.0) for s in samples], dtype=np.float64)
    base_vx = np.asarray([s["base_lin_vel_x"] for s in samples], dtype=np.float64)
    wheel_vx = np.asarray([s["wheel_lin_vel"] for s in samples], dtype=np.float64)

    step_height = float(samples[0]["stair_step_height_m"])
    step_depth = float(samples[0]["stair_step_depth_m"])
    start_x = float(samples[0]["stair_start_x_m"])
    half_width = float(samples[0].get("stair_half_width_m", 0.0))
    step_count = round(float(samples[0]["stair_step_count"]))
    if step_height <= 0.0 or step_depth <= 0.0 or step_count <= 0:
        return {}
    lateral_in_stair = np.ones_like(base_y, dtype=bool)
    if half_width > 0.0:
        lateral_in_stair = np.abs(base_y) <= half_width

    left_step = _wheel_step_index(left_x, start_x, step_depth, step_count)
    right_step = _wheel_step_index(right_x, start_x, step_depth, step_count)
    left_support_height = left_step.astype(np.float64) * step_height
    right_support_height = right_step.astype(np.float64) * step_height
    z_tol = max(0.035, 0.35 * step_height)
    left_on_step = (left_step >= 1) & left_contact & (np.abs(left_z - left_support_height) <= z_tol)
    right_on_step = (
        (right_step >= 1) & right_contact & (np.abs(right_z - right_support_height) <= z_tol)
    )
    both_on_step = left_on_step & right_on_step
    same_step = both_on_step & (left_step == right_step)
    completed_step = np.minimum(left_step, right_step)
    completed_step = np.where(same_step, completed_step, 0)
    stable = same_step & (~nonwheel_contact) & (np.abs(base_vx) < 0.35) & (np.abs(wheel_vx) < 0.8)

    dt = _median_dt(time)
    tail_count = max(1, round(1.0 / dt)) if dt > 0.0 else min(len(samples), 50)
    tail = slice(max(0, len(samples) - tail_count), len(samples))
    first_support_idx = _first_true_index(same_step)
    first_stable_idx = _first_true_index(stable)

    max_completed_step = int(np.max(completed_step)) if completed_step.size else 0
    final_completed_step = int(completed_step[-1]) if completed_step.size else 0
    tail_completed_step = int(np.max(completed_step[tail])) if completed_step.size else 0
    tail_stable_rate = float(np.mean(stable[tail]))
    stable_duration_s = _longest_true_duration(stable, dt)
    return {
        "step_height_m": float(step_height),
        "step_depth_m": float(step_depth),
        "max_completed_step": max_completed_step,
        "final_completed_step": final_completed_step,
        "tail_completed_step": tail_completed_step,
        "task_success": bool(final_completed_step >= 1 and tail_stable_rate >= 0.5),
        "climbed_first_step": bool(max_completed_step >= 1),
        "final_on_step": bool(final_completed_step >= 1),
        "tail_on_step_rate": float(np.mean(both_on_step[tail])),
        "tail_same_step_rate": float(np.mean(same_step[tail])),
        "tail_lateral_in_stair_rate": float(np.mean(lateral_in_stair[tail])),
        "support_rate": float(np.mean(same_step)),
        "support_duration_s": _longest_true_duration(same_step, dt),
        "stable_rate": float(np.mean(stable)),
        "stable_duration_s": stable_duration_s,
        "tail_stable_rate": tail_stable_rate,
        "nonwheel_contact_rate_after_support": _rate_after(nonwheel_contact, first_support_idx),
        "first_support_time_s": _time_at(time, first_support_idx),
        "first_stable_time_s": _time_at(time, first_stable_idx),
        "final_left_step": int(left_step[-1]),
        "final_right_step": int(right_step[-1]),
        "final_base_y_m": float(base_y[-1]),
        "final_lateral_in_stair": bool(lateral_in_stair[-1]),
        "final_left_height_error_m": float(left_z[-1] - left_support_height[-1]),
        "final_right_height_error_m": float(right_z[-1] - right_support_height[-1]),
    }


def _wheel_step_index(
    x: np.ndarray,
    start_x: float,
    step_depth: float,
    step_count: int,
) -> np.ndarray:
    raw = np.floor((x - start_x) / step_depth).astype(np.int64) + 1
    return np.clip(raw, 0, int(step_count))


def _median_dt(time: np.ndarray) -> float:
    if time.size < 2:
        return 0.0
    diff = np.diff(time)
    diff = diff[np.isfinite(diff) & (diff > 0.0)]
    if diff.size == 0:
        return 0.0
    return float(np.median(diff))


def _longest_true_duration(mask: np.ndarray, dt: float) -> float:
    if mask.size == 0 or dt <= 0.0:
        return 0.0
    longest = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return float(longest * dt)


def _first_true_index(mask: np.ndarray) -> int | None:
    indices = np.nonzero(mask)[0]
    if indices.size == 0:
        return None
    return int(indices[0])


def _time_at(time: np.ndarray, idx: int | None) -> float | None:
    if idx is None:
        return None
    return float(time[idx])


def _rate_after(mask: np.ndarray, idx: int | None) -> float:
    if idx is None or idx >= mask.size:
        return 0.0
    return float(np.mean(mask[idx:]))


def _names(model: mujoco.MjModel, obj_type: mujoco.mjtObj, count: int) -> list[str]:
    out: list[str] = []
    for idx in range(count):
        out.append(mujoco.mj_id2name(model, obj_type, idx) or f"{obj_type.name}_{idx}")
    return out


def _stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def _geom_type_name(code: int) -> str:
    mapping = {
        int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
        int(mujoco.mjtGeom.mjGEOM_HFIELD): "hfield",
        int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
        int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
        int(mujoco.mjtGeom.mjGEOM_BOX): "box",
        int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
    }
    return mapping.get(int(code), f"unknown_{int(code)}")


def _wheel_collision_diagnostics(model: mujoco.MjModel) -> dict[str, object]:
    geoms: list[dict[str, object]] = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
        body_id = int(model.geom_bodyid[gid])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if "wheel" not in name.lower() and "wheel" not in body_name.lower():
            continue
        if int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0:
            continue
        geom_type = _geom_type_name(int(model.geom_type[gid]))
        mesh_name = None
        mesh_source = None
        if geom_type == "mesh":
            mesh_id = int(model.geom_dataid[gid])
            mesh_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or f"mesh_{mesh_id}"
            )
            mesh_source = (
                "visual_stl" if mesh_name.startswith("visual_") else "generated_or_collision"
            )
        geoms.append(
            {
                "name": name,
                "body": body_name,
                "type": geom_type,
                "mesh": mesh_name,
                "mesh_source": mesh_source,
                "friction": np.asarray(model.geom_friction[gid], dtype=np.float64).tolist(),
            }
        )
    types = [str(g["type"]) for g in geoms]
    mode = "none"
    if types and all(t == "cylinder" for t in types):
        mode = "cylinder"
    elif types and all(t == "mesh" for t in types):
        mode = "mesh"
    elif types:
        mode = "mixed"
    return {"mode": mode, "geoms": geoms}


def _model_issues(wheel: dict[str, object]) -> list[DiagnosticIssue]:
    issues: list[DiagnosticIssue] = []
    if wheel.get("mode") != "cylinder":
        wheel_geoms = [geom for geom in wheel.get("geoms", []) if isinstance(geom, dict)]
        has_visual_mesh = any(geom.get("mesh_source") == "visual_stl" for geom in wheel_geoms)
        mesh_description = (
            "Visual STL wheel meshes" if has_visual_mesh else "Mesh wheel collision geoms"
        )
        issues.append(
            DiagnosticIssue(
                severity="warning",
                code="wheel_collision_not_cylinder",
                message=(
                    "Wheel collision geoms are not pure cylinders. "
                    f"{mesh_description} can create unstable MuJoCo contacts and poor standing behavior."
                ),
            )
        )
    for geom in wheel.get("geoms", []):
        if isinstance(geom, dict) and geom.get("mesh_source") == "visual_stl":
            issues.append(
                DiagnosticIssue(
                    severity="warning",
                    code="wheel_collision_uses_visual_mesh",
                    message=f"{geom.get('name')} uses visual mesh {geom.get('mesh')} as a collision geom.",
                )
            )
    return issues
