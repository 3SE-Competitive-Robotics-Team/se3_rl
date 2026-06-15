"""SerialLeg MJCF 闭链位置和速度投影。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

_CLOSURE_TOLERANCE_M = 1.0e-7
_CLOSURE_SOLVER_EPS = 1.0e-5
_CLOSURE_SOLVER_ITERS = 10
_CLOSURE_STEP_LIMIT_RAD = 0.6


@dataclass(frozen=True, slots=True)
class _LegClosureSpec:
    front_joint: str
    drive_joint: str
    knee_joint: str
    coupler_joint: str
    coupler_end_site: str
    calf_closure_site: str


_LEG_CLOSURE_SPECS = (
    _LegClosureSpec(
        front_joint="lf0_Joint",
        drive_joint="l_drive_bar_Joint",
        knee_joint="lf1_Joint",
        coupler_joint="l_coupler_Joint",
        coupler_end_site="l_coupler_end",
        calf_closure_site="lf_coupler_closure",
    ),
    _LegClosureSpec(
        front_joint="rf0_Joint",
        drive_joint="r_drive_bar_Joint",
        knee_joint="rf1_Joint",
        coupler_joint="r_coupler_Joint",
        coupler_end_site="r_coupler_end",
        calf_closure_site="rf_coupler_closure",
    ),
)


class ClosedChainClosureSolver:
    """投影被动关节，使日志中的主动关节满足 MJCF 闭链约束。"""

    def __init__(self, legs: tuple[_LegClosureSolver, ...]) -> None:
        self.legs = legs

    @classmethod
    def try_create(
        cls,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> ClosedChainClosureSolver | None:
        joint_qpos_by_name: dict[str, int] = {}
        joint_qvel_by_name: dict[str, int] = {}
        for joint_id in range(model.njnt):
            if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if not name:
                continue
            joint_qpos_by_name[name] = int(model.jnt_qposadr[joint_id])
            joint_qvel_by_name[name] = int(model.jnt_dofadr[joint_id])

        legs: list[_LegClosureSolver] = []
        for spec in _LEG_CLOSURE_SPECS:
            leg = _LegClosureSolver.try_create(
                model=model,
                data=data,
                joint_qpos_by_name=joint_qpos_by_name,
                joint_qvel_by_name=joint_qvel_by_name,
                spec=spec,
            )
            if leg is not None:
                legs.append(leg)
        if len(legs) != len(_LEG_CLOSURE_SPECS):
            return None
        return cls(tuple(legs))

    def solve_positions(self) -> float:
        residuals = [leg.solve_position() for leg in self.legs]
        return float(max(residuals, default=0.0))

    def seed_passive_position(self, leg_index: int, knee_angle: float) -> None:
        """用解析膝角给指定腿的闭链求解器播种，避免跳到另一装配分支。"""

        if not (0 <= int(leg_index) < len(self.legs)):
            return
        self.legs[int(leg_index)].seed_position(float(knee_angle))

    def solve_velocities(self) -> float:
        residuals = [leg.solve_velocity() for leg in self.legs]
        return float(max(residuals, default=0.0))


class _LegClosureSolver:
    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        spec: _LegClosureSpec,
        front_qpos: int,
        drive_qpos: int,
        knee_qpos: int,
        coupler_qpos: int,
        front_qvel: int,
        drive_qvel: int,
        knee_qvel: int,
        coupler_qvel: int,
        coupler_end_site: int,
        calf_closure_site: int,
    ) -> None:
        self.model = model
        self.data = data
        self.spec = spec
        self.front_qpos = int(front_qpos)
        self.drive_qpos = int(drive_qpos)
        self.knee_qpos = int(knee_qpos)
        self.coupler_qpos = int(coupler_qpos)
        self.front_qvel = int(front_qvel)
        self.drive_qvel = int(drive_qvel)
        self.knee_qvel = int(knee_qvel)
        self.coupler_qvel = int(coupler_qvel)
        self.coupler_end_site = int(coupler_end_site)
        self.calf_closure_site = int(calf_closure_site)
        self.passive_guess = np.asarray(
            [self.data.qpos[self.knee_qpos], self.data.qpos[self.coupler_qpos]],
            dtype=np.float64,
        )
        self.neutral_passive = self.passive_guess.copy()

    @classmethod
    def try_create(
        cls,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_qpos_by_name: dict[str, int],
        joint_qvel_by_name: dict[str, int],
        spec: _LegClosureSpec,
    ) -> _LegClosureSolver | None:
        joint_names = (
            spec.front_joint,
            spec.drive_joint,
            spec.knee_joint,
            spec.coupler_joint,
        )
        if any(name not in joint_qpos_by_name for name in joint_names):
            return None
        if any(name not in joint_qvel_by_name for name in joint_names):
            return None
        coupler_end_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, spec.coupler_end_site)
        calf_closure_site = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, spec.calf_closure_site
        )
        if coupler_end_site < 0 or calf_closure_site < 0:
            return None
        return cls(
            model=model,
            data=data,
            spec=spec,
            front_qpos=joint_qpos_by_name[spec.front_joint],
            drive_qpos=joint_qpos_by_name[spec.drive_joint],
            knee_qpos=joint_qpos_by_name[spec.knee_joint],
            coupler_qpos=joint_qpos_by_name[spec.coupler_joint],
            front_qvel=joint_qvel_by_name[spec.front_joint],
            drive_qvel=joint_qvel_by_name[spec.drive_joint],
            knee_qvel=joint_qvel_by_name[spec.knee_joint],
            coupler_qvel=joint_qvel_by_name[spec.coupler_joint],
            coupler_end_site=coupler_end_site,
            calf_closure_site=calf_closure_site,
        )

    def solve_position(self) -> float:
        base_qpos = self.data.qpos.copy()
        best_error = math.inf
        best_passive = (
            float(self.data.qpos[self.knee_qpos]),
            float(self.data.qpos[self.coupler_qpos]),
        )
        for seed in self._solver_seeds(best_passive):
            self.data.qpos[:] = base_qpos
            self.data.qpos[self.knee_qpos] = float(seed[0])
            self.data.qpos[self.coupler_qpos] = float(seed[1])
            error = self._refine_passive_joints()
            if error < best_error:
                best_error = error
                best_passive = (
                    _wrap_angle_scalar(float(self.data.qpos[self.knee_qpos])),
                    _wrap_angle_scalar(float(self.data.qpos[self.coupler_qpos])),
                )
            if error <= _CLOSURE_TOLERANCE_M:
                break

        self.data.qpos[:] = base_qpos
        self.data.qpos[self.knee_qpos] = best_passive[0]
        self.data.qpos[self.coupler_qpos] = best_passive[1]
        self.passive_guess = np.asarray(best_passive, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)
        return self._residual_norm()

    def seed_position(self, knee_angle: float) -> None:
        """设置下一次闭链位置求解优先使用的被动膝角。"""

        self.data.qpos[self.knee_qpos] = float(knee_angle)
        self.passive_guess = np.asarray(
            [float(knee_angle), float(self.data.qpos[self.coupler_qpos])],
            dtype=np.float64,
        )

    def solve_velocity(self) -> float:
        jac_active = self._residual_jacobian((self.front_qpos, self.drive_qpos))
        jac_passive = self._residual_jacobian((self.knee_qpos, self.coupler_qpos))
        active_qvel = np.asarray(
            [self.data.qvel[self.front_qvel], self.data.qvel[self.drive_qvel]],
            dtype=np.float64,
        )
        rhs = -(jac_active @ active_qvel)
        passive_qvel = np.linalg.lstsq(jac_passive, rhs, rcond=None)[0]
        if not np.isfinite(passive_qvel).all():
            return math.inf
        self.data.qvel[self.knee_qvel] = float(passive_qvel[0])
        self.data.qvel[self.coupler_qvel] = float(passive_qvel[1])
        residual_vel = jac_active @ active_qvel + jac_passive @ passive_qvel
        return float(np.linalg.norm(residual_vel))

    def _solver_seeds(self, current: tuple[float, float]) -> tuple[tuple[float, float], ...]:
        previous = (float(self.passive_guess[0]), float(self.passive_guess[1]))
        neutral = (float(self.neutral_passive[0]), float(self.neutral_passive[1]))
        return (
            previous,
            current,
            neutral,
            (0.0, 0.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (-2.0, 2.0),
            (2.0, -2.0),
            (neutral[0] + math.pi, neutral[1] + math.pi),
            (neutral[0] - math.pi, neutral[1] - math.pi),
        )

    def _refine_passive_joints(self) -> float:
        passive_qpos = (self.knee_qpos, self.coupler_qpos)
        for _ in range(_CLOSURE_SOLVER_ITERS):
            residual = self._residual()
            error = float(np.linalg.norm(residual))
            if error <= _CLOSURE_TOLERANCE_M:
                return error

            jacobian = self._residual_jacobian(passive_qpos)
            delta = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
            if not np.isfinite(delta).all():
                return math.inf
            delta = np.clip(delta, -_CLOSURE_STEP_LIMIT_RAD, _CLOSURE_STEP_LIMIT_RAD)
            for col, qpos_addr in enumerate(passive_qpos):
                self.data.qpos[qpos_addr] += float(delta[col])
        return self._residual_norm()

    def _residual_jacobian(self, qpos_addrs: tuple[int, int]) -> np.ndarray:
        residual = self._residual()
        jacobian = np.zeros((3, 2), dtype=np.float64)
        for col, qpos_addr in enumerate(qpos_addrs):
            old_value = float(self.data.qpos[qpos_addr])
            self.data.qpos[qpos_addr] = old_value + _CLOSURE_SOLVER_EPS
            residual_plus = self._residual()
            self.data.qpos[qpos_addr] = old_value
            jacobian[:, col] = (residual_plus - residual) / _CLOSURE_SOLVER_EPS
        mujoco.mj_forward(self.model, self.data)
        return jacobian

    def _residual(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        return np.asarray(
            self.data.site_xpos[self.coupler_end_site]
            - self.data.site_xpos[self.calf_closure_site],
            dtype=np.float64,
        )

    def _residual_norm(self) -> float:
        return float(np.linalg.norm(self._residual()))


def _wrap_angle_scalar(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)
