from __future__ import annotations

import json
import math

import numpy as np

from se3_shared import output_leg_wheel_xz_np, wheel_xz_to_output_pos_np

_HIP_OFFSETS_BODY = np.asarray(
    ((0.0, 0.16885, 0.0), (0.0, -0.16885, 0.0)),
    dtype=np.float64,
)


def _rot_zyx(*, roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.asarray(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    ry = np.asarray(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))
    rx = np.asarray(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)))
    return rz @ ry @ rx


def _polar_to_xz(length_m: float, swing_rad: float) -> np.ndarray:
    return np.asarray(
        (length_m * math.sin(swing_rad), -length_m * math.cos(swing_rad)),
        dtype=np.float64,
    )


def _xz_to_polar(xz: np.ndarray) -> tuple[float, float]:
    x, z = float(xz[0]), float(xz[1])
    return math.hypot(x, z), math.atan2(x, -z)


def main() -> None:
    length_m = 0.18
    swing_deg = -35.0
    swing_rad = math.radians(swing_deg)
    target_xz = _polar_to_xz(length_m, swing_rad)
    target_for_ik = np.stack((target_xz, target_xz), axis=0)
    output = wheel_xz_to_output_pos_np(target_for_ik)
    roundtrip_xz = output_leg_wheel_xz_np(output)
    roundtrip_error_m = float(np.max(np.abs(roundtrip_xz - target_for_ik)))

    base_pos = np.asarray((1.2, -0.4, 0.55), dtype=np.float64)
    poses = [
        {"name": "level", "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0},
        {"name": "rotated_body", "roll_deg": -8.0, "pitch_deg": 12.0, "yaw_deg": 37.0},
    ]
    rows = []
    for pose in poses:
        rot = _rot_zyx(
            roll_deg=pose["roll_deg"],
            pitch_deg=pose["pitch_deg"],
            yaw_deg=pose["yaw_deg"],
        )
        wheel_body = _HIP_OFFSETS_BODY[0] + np.asarray((target_xz[0], 0.0, target_xz[1]))
        wheel_world = base_pos + rot @ wheel_body
        recovered_body = rot.T @ (wheel_world - base_pos) - _HIP_OFFSETS_BODY[0]
        recovered_xz = recovered_body[[0, 2]]
        recovered_length, recovered_swing = _xz_to_polar(recovered_xz)
        rows.append(
            {
                **pose,
                "world_xyz": [round(float(v), 9) for v in wheel_world],
                "recovered_body_xz": [round(float(v), 9) for v in recovered_xz],
                "recovered_length_m": round(recovered_length, 9),
                "recovered_swing_deg": round(math.degrees(recovered_swing), 9),
            }
        )

    result = {
        "coordinate_mode": "body_polar",
        "definition": "x = length * sin(swing), z = -length * cos(swing), swing=0 means body -Z/down",
        "target": {
            "leg_length_m": length_m,
            "swing_angle_deg": swing_deg,
            "derived_body_xz": [round(float(v), 9) for v in target_xz],
        },
        "ik_roundtrip_max_abs_error_m": round(roundtrip_error_m, 12),
        "body_frame_invariance_rows": rows,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
