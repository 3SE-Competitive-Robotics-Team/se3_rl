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
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "roll_rate_rad_s",
        "pitch_rate_rad_s",
        "yaw_rate_rad_s",
        "action_delta_l2",
        "action_delta_max_abs",
        "action_delta_sq_sum",
        "applied_action_delta_l2",
        "applied_action_delta_max_abs",
        "applied_action_delta_sq_sum",
    ):
        if key in samples[0]:
            values = np.asarray([s[key] for s in samples], dtype=np.float64)
            result[key] = _stats(values)
    result["final"] = samples[-1]
    return result


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
        issues.append(
            DiagnosticIssue(
                severity="warning",
                code="wheel_collision_not_cylinder",
                message=(
                    "Wheel collision geoms are not pure cylinders. "
                    "Visual STL wheel meshes can create unstable MuJoCo contacts and poor standing behavior."
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
