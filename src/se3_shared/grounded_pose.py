"""轮子触地姿态 IK 工具。

给定 base_link 目标高度，使用 MuJoCo FK 求解四个腿部关节角，使左右轮子刚好触地。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize

from .robot import RobotConfig

_MJCF_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "robots"
    / "serialleg"
    / "mjcf"
    / "serialleg_fidelity_cylinder_wheels.xml"
)

_CTRL_JOINT_NAMES = (
    "lf0_Joint",
    "lf1_Joint",
    "l_wheel_Joint",
    "rf0_Joint",
    "rf1_Joint",
    "r_wheel_Joint",
)


@dataclass(frozen=True)
class GroundedPoseResult:
    """轮子触地 IK 求解结果。"""

    base_height: float
    ground_height: float
    wheel_radius: float
    q_legs: tuple[float, float, float, float]
    q6: tuple[float, float, float, float, float, float]
    left_wheel_center: tuple[float, float, float]
    right_wheel_center: tuple[float, float, float]
    left_wheel_bottom_error: float
    right_wheel_bottom_error: float
    left_wheel_x_error: float
    right_wheel_x_error: float
    com_x_error: float
    objective: float
    success: bool
    message: str

    def to_dict(self) -> dict[str, object]:
        """转换为可 JSON 序列化的字典。"""
        return asdict(self)


@dataclass(frozen=True)
class SwingPoseResult:
    """单腿抬起 IK 求解结果。"""

    base_height: float
    target_clearance: float
    wheel_radius: float
    q_swing: tuple[float, float]
    wheel_center: tuple[float, float, float]
    clearance_error: float
    wheel_x_error: float
    objective: float
    success: bool
    message: str

    def to_dict(self) -> dict[str, object]:
        """转换为可 JSON 序列化的字典。"""
        return asdict(self)


def find_wheel_collision_radius(
    model: mujoco.MjModel,
    wheel_body_name: str = "l_wheel_Link",
) -> float:
    """从 wheel collision geom 读取真实轮子半径。"""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wheel_body_name)
    if body_id < 0:
        raise ValueError(f"找不到 wheel body: {wheel_body_name}")

    candidates: list[float] = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != body_id:
            continue
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if "collision" not in geom_name:
            continue
        geom_type = int(model.geom_type[geom_id])
        if geom_type in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            candidates.append(float(model.geom_size[geom_id, 0]))

    if not candidates:
        raise ValueError(f"找不到 {wheel_body_name} 的 wheel collision geom 半径")
    return max(candidates)


class GroundedPoseSolver:
    """使用 MuJoCo FK 求解指定 base 高度下的触地关节姿态。"""

    def __init__(self, mjcf_path: Path = _MJCF_PATH) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)

        self.left_wheel_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_Link"
        )
        self.right_wheel_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_Link"
        )
        if self.left_wheel_body_id < 0 or self.right_wheel_body_id < 0:
            raise ValueError("MJCF 中找不到左右轮 body")

        self.ctrl_qpos_idx = []
        self.leg_bounds: list[tuple[float, float]] = []
        for name in _CTRL_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"MJCF 中找不到受控关节: {name}")
            self.ctrl_qpos_idx.append(int(self.model.jnt_qposadr[joint_id]))
            if (
                name.endswith("_Joint")
                and name[1] == "f"
                and bool(self.model.jnt_limited[joint_id])
            ):
                lo, hi = self.model.jnt_range[joint_id]
                self.leg_bounds.append((float(lo), float(hi)))

        self.robot_cfg = RobotConfig()
        self.default_q6 = np.asarray(self.robot_cfg.default_dof_pos, dtype=np.float64)
        self.default_base_height = float(self.robot_cfg.default_base_height)
        self.wheel_radius = find_wheel_collision_radius(self.model, "l_wheel_Link")

        if len(self.leg_bounds) < 4:
            self.leg_bounds = [(-1.5, 1.5), (-0.55, 0.75), (-1.5, 1.5), (-0.55, 0.75)]
        self.symmetric_bounds = [self.leg_bounds[0], self.leg_bounds[1]]

    def fk(self, base_height: float, q6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回左右轮子中心世界坐标。"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, float(base_height)]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for i, qpos_idx in enumerate(self.ctrl_qpos_idx):
            self.data.qpos[qpos_idx] = float(q6[i])
        mujoco.mj_forward(self.model, self.data)
        return (
            self.data.xpos[self.left_wheel_body_id].copy(),
            self.data.xpos[self.right_wheel_body_id].copy(),
        )

    def _center_of_mass(self) -> np.ndarray:
        """返回当前 MuJoCo data 下整机 COM 世界坐标。"""
        total_mass = float(np.sum(self.model.body_mass))
        weighted = np.zeros(3, dtype=np.float64)
        for body_id in range(self.model.nbody):
            weighted += float(self.model.body_mass[body_id]) * self.data.xipos[body_id]
        return weighted / total_mass

    def solve(
        self,
        base_height: float,
        ground_height: float = 0.0,
        keep_wheel_x: bool = True,
        align_com_x: bool = False,
    ) -> GroundedPoseResult:
        """求解左右对称触地姿态。"""
        default_left, default_right = self.fk(self.default_base_height, self.default_q6)
        target_left_x = float(default_left[0])
        target_right_x = float(default_right[0])
        target_wheel_z = float(ground_height + self.wheel_radius)
        q_init = np.asarray([self.default_q6[0], self.default_q6[1]], dtype=np.float64)

        def objective(q: np.ndarray) -> float:
            q6 = np.asarray([q[0], q[1], 0.0, q[0], q[1], 0.0], dtype=np.float64)
            left, right = self.fk(base_height, q6)
            err = (float(left[2]) - target_wheel_z) ** 2
            err += (float(right[2]) - target_wheel_z) ** 2
            if keep_wheel_x:
                err += (float(left[0]) - target_left_x) ** 2
                err += (float(right[0]) - target_right_x) ** 2
            if align_com_x:
                wheel_mid_x = 0.5 * (float(left[0]) + float(right[0]))
                com_x = float(self._center_of_mass()[0])
                err += 0.02 * (com_x - wheel_mid_x) ** 2
            smooth = 0.001 * float(np.sum((q - q_init) ** 2))
            return 1000.0 * err + smooth

        starts = (
            q_init,
            np.asarray([0.65, 0.25], dtype=np.float64),
            np.asarray([0.80, 0.00], dtype=np.float64),
            np.asarray([1.00, -0.30], dtype=np.float64),
            np.asarray([1.20, -0.50], dtype=np.float64),
            np.asarray([0.40, 0.70], dtype=np.float64),
        )
        best = None
        for start in starts:
            res = minimize(
                objective,
                x0=start,
                bounds=self.symmetric_bounds,
                method="L-BFGS-B",
                options={"maxiter": 300, "ftol": 1e-14, "gtol": 1e-10},
            )
            if best is None or float(res.fun) < float(best.fun):
                best = res

        if best is None:
            raise RuntimeError("IK 求解没有返回结果")

        qh, qk = float(best.x[0]), float(best.x[1])
        q6 = np.asarray([qh, qk, 0.0, qh, qk, 0.0], dtype=np.float64)
        left, right = self.fk(base_height, q6)
        left_bottom_error = float(left[2] - target_wheel_z)
        right_bottom_error = float(right[2] - target_wheel_z)
        left_x_error = float(left[0] - target_left_x) if keep_wheel_x else 0.0
        right_x_error = float(right[0] - target_right_x) if keep_wheel_x else 0.0
        wheel_mid_x = 0.5 * (float(left[0]) + float(right[0]))
        com_x_error = float(self._center_of_mass()[0] - wheel_mid_x)
        residual_ok = max(abs(left_bottom_error), abs(right_bottom_error)) < 1e-4
        if keep_wheel_x:
            residual_ok = residual_ok and max(abs(left_x_error), abs(right_x_error)) < 1e-4
        if align_com_x:
            residual_ok = residual_ok and abs(com_x_error) < 1e-4

        return GroundedPoseResult(
            base_height=float(base_height),
            ground_height=float(ground_height),
            wheel_radius=float(self.wheel_radius),
            q_legs=(qh, qk, qh, qk),
            q6=(qh, qk, 0.0, qh, qk, 0.0),
            left_wheel_center=tuple(float(v) for v in left),
            right_wheel_center=tuple(float(v) for v in right),
            left_wheel_bottom_error=left_bottom_error,
            right_wheel_bottom_error=right_bottom_error,
            left_wheel_x_error=left_x_error,
            right_wheel_x_error=right_x_error,
            com_x_error=com_x_error,
            objective=float(best.fun),
            success=bool(residual_ok),
            message=str(best.message),
        )

    def solve_swing(
        self,
        base_height: float,
        target_clearance: float,
        ground_height: float = 0.0,
        keep_wheel_x: bool = True,
    ) -> SwingPoseResult:
        """求解单腿 swing 姿态，让轮子抬到目标离地高度。"""
        stance = self.solve(base_height, ground_height=ground_height, keep_wheel_x=keep_wheel_x)
        target_wheel_z = float(ground_height + self.wheel_radius + target_clearance)
        target_wheel_x = float(stance.left_wheel_center[0])
        q_init = np.asarray(stance.q_legs[:2], dtype=np.float64)

        def objective(q: np.ndarray) -> float:
            q6 = np.asarray([q[0], q[1], 0.0, q[0], q[1], 0.0], dtype=np.float64)
            left, _ = self.fk(base_height, q6)
            err = (float(left[2]) - target_wheel_z) ** 2
            if keep_wheel_x:
                err += 0.1 * (float(left[0]) - target_wheel_x) ** 2
            smooth = 0.001 * float(np.sum((q - q_init) ** 2))
            return 1000.0 * err + smooth

        starts = (
            q_init,
            np.asarray([0.65, 0.15], dtype=np.float64),
            np.asarray([0.55, 0.35], dtype=np.float64),
            np.asarray([0.40, 0.70], dtype=np.float64),
            np.asarray([0.80, 0.00], dtype=np.float64),
        )
        best = None
        for start in starts:
            res = minimize(
                objective,
                x0=start,
                bounds=self.symmetric_bounds,
                method="L-BFGS-B",
                options={"maxiter": 300, "ftol": 1e-14, "gtol": 1e-10},
            )
            if best is None or float(res.fun) < float(best.fun):
                best = res

        if best is None:
            raise RuntimeError("单腿 swing IK 求解没有返回结果")

        qh, qk = float(best.x[0]), float(best.x[1])
        q6 = np.asarray([qh, qk, 0.0, qh, qk, 0.0], dtype=np.float64)
        left, _ = self.fk(base_height, q6)
        clearance_error = float(left[2] - target_wheel_z)
        wheel_x_error = float(left[0] - target_wheel_x) if keep_wheel_x else 0.0
        residual_ok = abs(clearance_error) < 1e-4
        if keep_wheel_x:
            residual_ok = residual_ok and abs(wheel_x_error) < 1e-4

        return SwingPoseResult(
            base_height=float(base_height),
            target_clearance=float(target_clearance),
            wheel_radius=float(self.wheel_radius),
            q_swing=(qh, qk),
            wheel_center=tuple(float(v) for v in left),
            clearance_error=clearance_error,
            wheel_x_error=wheel_x_error,
            objective=float(best.fun),
            success=bool(residual_ok),
            message=str(best.message),
        )


def solve_grounded_pose(
    base_height: float,
    ground_height: float = 0.0,
    keep_wheel_x: bool = True,
    align_com_x: bool = False,
) -> GroundedPoseResult:
    """便捷函数：求解指定 base 高度下的轮子触地姿态。"""
    return GroundedPoseSolver().solve(
        base_height=base_height,
        ground_height=ground_height,
        keep_wheel_x=keep_wheel_x,
        align_com_x=align_com_x,
    )


def solve_swing_pose(
    base_height: float,
    target_clearance: float,
    ground_height: float = 0.0,
    keep_wheel_x: bool = True,
) -> SwingPoseResult:
    """便捷函数：求解指定 base 高度下的单腿 swing 姿态。"""
    return GroundedPoseSolver().solve_swing(
        base_height=base_height,
        target_clearance=target_clearance,
        ground_height=ground_height,
        keep_wheel_x=keep_wheel_x,
    )
