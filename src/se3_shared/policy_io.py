"""训练、sim2sim 和真机部署共享的 policy I/O 语义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .fourbar import output_to_policy_pos_np, output_to_policy_vel_np
from .height_default import policy_default_from_height_np
from .leg_policy import policy_leg_phase_active_obs_np
from .observation import ObservationConfig
from .robot import JointGroup, RobotConfig

_ROBOT_CFG = RobotConfig()
_OBS_CFG = ObservationConfig()


@dataclass(frozen=True, slots=True)
class DecodedPolicyAction:
    """policy action 解码后的物理目标。"""

    clipped_action: np.ndarray
    leg_target: np.ndarray
    wheel_vel_target: np.ndarray
    policy_default: np.ndarray


@dataclass(frozen=True, slots=True)
class PolicyObservationResult:
    """一次 policy observation 拼装结果。"""

    obs: np.ndarray
    had_nonfinite_input: bool


class PolicyActionDecoder:
    """把 6D policy action 解码为腿部位置目标和轮速目标。"""

    def __init__(
        self,
        *,
        robot_cfg: RobotConfig | None = None,
        action_scale: tuple[float, ...] | np.ndarray | None = None,
        height_conditioned_action_default: bool = False,
        active_rod_semantics: bool = True,
        leg_action_reference: Literal["default", "front_current"] = "default",
        active_rod_target_lower_preload_margin: float | None = None,
        active_rod_target_upper_preload_margin: float = 0.0,
        dtype: np.dtype | type = np.float64,
    ) -> None:
        self.robot_cfg = _ROBOT_CFG if robot_cfg is None else robot_cfg
        self.dtype = np.dtype(dtype)
        self.action_scale = np.asarray(
            self.robot_cfg.action_scale if action_scale is None else action_scale,
            dtype=self.dtype,
        ).reshape(6)
        self.action_clip = self.robot_cfg.action_clip
        self.height_conditioned_action_default = bool(height_conditioned_action_default)
        self.active_rod_semantics = bool(active_rod_semantics)
        if leg_action_reference not in ("default", "front_current"):
            raise ValueError(
                "leg_action_reference must be 'default' or 'front_current', "
                f"got {leg_action_reference}"
            )
        self.leg_action_reference = leg_action_reference
        lower, upper = self.robot_cfg.active_rod_angle_limits
        lower_margin = (
            self.robot_cfg.active_rod_lower_target_overdrive
            if active_rod_target_lower_preload_margin is None
            else active_rod_target_lower_preload_margin
        )
        self.active_rod_target_lower = float(lower) - float(lower_margin)
        self.active_rod_target_upper = float(upper) + float(active_rod_target_upper_preload_margin)
        self.active_rod_angle_mid = 0.5 * (float(lower) + float(upper))
        self.active_rod_angle_coeffs = np.asarray(
            self.robot_cfg.active_rod_angle_coeffs,
            dtype=self.dtype,
        ).reshape(2, 2)

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """按共享 action clip 限制 raw policy action。"""
        raw = np.asarray(action, dtype=self.dtype).reshape(6)
        if self.action_clip is None:
            return raw.astype(self.dtype, copy=True)
        clip = float(self.action_clip)
        return np.clip(raw, -clip, clip).astype(self.dtype, copy=False)

    def policy_default(
        self,
        *,
        command_height: float | None = None,
        fallback_default: np.ndarray | None = None,
    ) -> np.ndarray:
        """返回当前 action 零点姿态，语义为 [LF, LB, RF, RB]。"""
        if self.height_conditioned_action_default:
            if command_height is None:
                raise ValueError("height-conditioned action default 需要 command_height")
            return (
                policy_default_from_height_np(command_height, self.robot_cfg)
                .astype(
                    self.dtype,
                    copy=False,
                )
                .reshape(4)
            )
        if fallback_default is None:
            fallback_default = np.asarray(self.robot_cfg.default_dof_pos, dtype=self.dtype)[
                JointGroup.CTRL_LEGS
            ]
        return np.asarray(fallback_default, dtype=self.dtype).reshape(4)

    def decode(
        self,
        action: np.ndarray,
        *,
        command_height: float | None = None,
        policy_default: np.ndarray | None = None,
        current_policy_pos: np.ndarray | None = None,
        fallback_default: np.ndarray | None = None,
    ) -> DecodedPolicyAction:
        """解码 policy action；输入输出均使用 policy 顺序。"""
        clipped = self.clip_action(action)
        default = (
            self.policy_default(
                command_height=command_height,
                fallback_default=fallback_default,
            )
            if policy_default is None
            else np.asarray(policy_default, dtype=self.dtype).reshape(4)
        )
        leg_target = self.leg_target(
            clipped[:4],
            default,
            current_policy_pos=current_policy_pos,
        )
        wheel_vel_target = (
            clipped[JointGroup.CTRL_WHEELS] * self.action_scale[JointGroup.WHEEL_ACTUATORS]
        )
        return DecodedPolicyAction(
            clipped_action=clipped,
            leg_target=leg_target,
            wheel_vel_target=wheel_vel_target.astype(self.dtype, copy=False),
            policy_default=default,
        )

    def leg_target(
        self,
        leg_action: np.ndarray,
        policy_default: np.ndarray,
        *,
        current_policy_pos: np.ndarray | None = None,
    ) -> np.ndarray:
        """解码 4D 腿部 action。"""
        leg_action = np.asarray(leg_action, dtype=self.dtype).reshape(4)
        default = np.asarray(policy_default, dtype=self.dtype).reshape(4)
        if self.leg_action_reference == "front_current":
            if current_policy_pos is None:
                raise ValueError("front_current action reference needs current_policy_pos")
            current = np.asarray(current_policy_pos, dtype=self.dtype).reshape(4)
            default = default.copy()
            default[[0, 2]] = current[[0, 2]]
        leg_scale = self.action_scale[JointGroup.LEG_ACTUATORS]
        if not self.active_rod_semantics:
            return (default + leg_action * leg_scale).astype(self.dtype, copy=False)

        target = np.empty(4, dtype=self.dtype)
        for side_idx, (front_idx, back_idx) in enumerate(((0, 1), (2, 3))):
            front_coef, back_coef = self.active_rod_angle_coeffs[side_idx]
            front_target = default[front_idx] + leg_action[front_idx] * leg_scale[front_idx]
            if self.height_conditioned_action_default:
                active_default = front_coef * default[front_idx] + back_coef * default[back_idx]
            else:
                active_default = self.active_rod_angle_mid
            active_raw = active_default + leg_action[back_idx] * leg_scale[back_idx]
            active_target = np.clip(
                active_raw,
                self.active_rod_target_lower,
                self.active_rod_target_upper,
            )
            target[front_idx] = front_target
            target[back_idx] = (active_target - front_coef * front_target) / back_coef
        return target


def build_policy_observation_np(
    *,
    base_ang_vel_body: np.ndarray,
    projected_gravity: np.ndarray,
    dof_pos: np.ndarray,
    dof_vel: np.ndarray,
    command: np.ndarray,
    action_obs: np.ndarray,
    default_dof_pos: np.ndarray,
    command_scale: tuple[float, ...] | np.ndarray | None = None,
    expected_num_obs: int | None = None,
    clip_value: float | None = None,
    fourbar_surrogate: bool = False,
    normalize_projected_gravity: bool = False,
) -> PolicyObservationResult:
    """按 34D actor contract 拼装 policy observation。"""
    had_nonfinite = False
    base_ang_vel_body, bad = _finite_array(base_ang_vel_body, 3)
    had_nonfinite = had_nonfinite or bad
    projected_gravity, bad = _projected_gravity(
        projected_gravity,
        normalize=normalize_projected_gravity,
    )
    had_nonfinite = had_nonfinite or bad
    dof_pos, bad = _finite_array(dof_pos, 6)
    had_nonfinite = had_nonfinite or bad
    dof_vel, bad = _finite_array(dof_vel, 6)
    had_nonfinite = had_nonfinite or bad
    action_obs, bad = _finite_array(action_obs, 6)
    had_nonfinite = had_nonfinite or bad

    command_arr = np.asarray(command, dtype=np.float64).reshape(-1)
    if command_arr.shape[0] not in (7, 8):
        raise ValueError(
            "command 必须为 7 或 8 维 "
            "[vx, yaw_rate, pitch, roll, height, jump_flag, jump_target_height, (jump_phase)]"
        )
    if not np.isfinite(command_arr).all():
        command_arr = np.nan_to_num(command_arr, nan=0.0, posinf=0.0, neginf=0.0)
        had_nonfinite = True

    scale = np.asarray(
        _OBS_CFG.command_scale if command_scale is None else command_scale,
        dtype=np.float64,
    ).reshape(5)
    default = np.asarray(default_dof_pos, dtype=np.float64).reshape(6)

    obs: list[float] = []
    obs.extend((base_ang_vel_body * _OBS_CFG.ang_vel_scale).tolist())
    obs.extend(projected_gravity.tolist())
    obs.extend((command_arr[:5] * scale).tolist())

    leg_pos = dof_pos[JointGroup.CTRL_LEGS]
    default_leg_pos = default[JointGroup.CTRL_LEGS]
    leg_vel = dof_vel[JointGroup.CTRL_LEGS]
    if fourbar_surrogate:
        leg_pos = output_to_policy_pos_np(leg_pos)
        default_leg_pos = output_to_policy_pos_np(default_leg_pos)
        leg_vel = output_to_policy_vel_np(dof_pos[JointGroup.CTRL_LEGS], leg_vel)
    obs.extend(policy_leg_phase_active_obs_np(leg_pos, default_leg_pos).reshape(-1).tolist())
    obs.extend((leg_vel * _OBS_CFG.leg_vel_scale).tolist())
    # 轮子是连续关节，累计位置会无界增长；保留 2D 槽位以兼容策略输入维度。
    obs.extend((0.0, 0.0))
    obs.extend((dof_vel[JointGroup.CTRL_WHEELS] * _OBS_CFG.wheel_vel_scale).tolist())
    obs.extend(action_obs.tolist())
    if command_arr.shape[0] >= 8:
        obs.extend(command_arr[5:8].tolist())
    else:
        obs.extend(command_arr[5:7].tolist())
        obs.append(0.0)

    arr = np.asarray(obs, dtype=np.float32)
    expected = _OBS_CFG.num_obs if expected_num_obs is None else int(expected_num_obs)
    if arr.shape != (expected,):
        raise RuntimeError(f"observation shape mismatch: expected {(expected,)}, got {arr.shape}")
    limit = _OBS_CFG.clip_value if clip_value is None else float(clip_value)
    arr = np.nan_to_num(arr, nan=0.0, posinf=limit, neginf=-limit)
    return PolicyObservationResult(
        obs=np.clip(arr, -limit, limit),
        had_nonfinite_input=had_nonfinite,
    )


def _finite_array(values: object, size: int) -> tuple[np.ndarray, bool]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape != (int(size),):
        raise ValueError(f"array shape mismatch: expected {(size,)}, got {arr.shape}")
    finite = np.isfinite(arr)
    if finite.all():
        return arr, False
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), True


def _projected_gravity(values: object, *, normalize: bool) -> tuple[np.ndarray, bool]:
    arr, bad = _finite_array(values, 3)
    if not normalize:
        return arr, bad
    norm = float(np.linalg.norm(arr))
    if norm < 1.0e-6:
        return np.asarray([0.0, 0.0, -1.0], dtype=np.float64), True
    return np.clip(arr / norm, -1.0, 1.0), bad
