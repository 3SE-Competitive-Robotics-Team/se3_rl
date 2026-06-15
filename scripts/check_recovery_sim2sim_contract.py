"""检查 recovery 训练与 sim2sim 共享的 policy I/O 合同。"""

from __future__ import annotations

import math

import numpy as np

from se3_shared import (
    JointGroup,
    ObservationConfig,
    PolicyActionDecoder,
    RobotConfig,
    build_policy_observation_np,
    policy_leg_position_error_np,
)
from se3_sim2sim.deploy_telemetry import _decode_state_fields
from se3_sim2sim.runtime_spec import RuntimeSpec


def main() -> int:
    """运行无需 GPU 的 recovery obs/action 合同检查。"""
    _check_runtime_slices()
    _check_observation_shape_and_leg_phase()
    _check_deploy_telemetry_last_actions_slice()
    _check_periodic_pd_error()
    _check_height_conditioned_decoder()
    print("recovery sim2sim contract ok")
    return 0


def _check_runtime_slices() -> None:
    obs_cfg = ObservationConfig()
    runtime = RuntimeSpec()
    expected = {
        "ang_vel": (0, 3),
        "gravity": (3, 6),
        "commands": (6, 11),
        "leg_joint_pos": (11, 17),
        "leg_joint_vel": (17, 21),
        "wheel_pos_zero": (21, 23),
        "wheel_vel": (23, 25),
        "last_actions": (25, 31),
        "jump_commands": (31, 34),
    }
    if runtime.policy.num_obs != obs_cfg.num_obs:
        raise AssertionError(
            f"runtime obs dim {runtime.policy.num_obs} != shared obs dim {obs_cfg.num_obs}"
        )
    actual = {name: (sl.start, sl.stop) for name, sl in runtime.observation_slices.items()}
    if actual != expected:
        raise AssertionError(f"unexpected observation slices: {actual}")


def _check_observation_shape_and_leg_phase() -> None:
    cfg = RobotConfig()
    obs_cfg = ObservationConfig()
    default = np.asarray(cfg.default_dof_pos, dtype=np.float64)
    dof_pos = default.copy()
    dof_pos[0] += 2.0 * math.pi + 0.25
    dof_pos[2] -= 2.0 * math.pi + 0.40
    command = np.asarray([0.0, 0.0, 0.0, 0.0, 0.26, 0.0, 0.2, 0.0], dtype=np.float64)
    result = build_policy_observation_np(
        base_ang_vel_body=np.zeros(3),
        projected_gravity=np.asarray([0.0, 0.0, -1.0]),
        dof_pos=dof_pos,
        dof_vel=np.zeros(6),
        command=command,
        action_obs=np.zeros(6),
        default_dof_pos=default,
    )
    if result.obs.shape != (obs_cfg.num_obs,):
        raise AssertionError(f"obs shape mismatch: {result.obs.shape}")
    leg_obs = result.obs[11:17]
    expected_front = np.asarray(
        [
            math.sin(0.25),
            math.cos(0.25),
            math.sin(-0.40),
            math.cos(-0.40),
        ],
        dtype=np.float32,
    )
    got_front = leg_obs[[0, 1, 3, 4]]
    if not np.allclose(got_front, expected_front, atol=1.0e-6, rtol=0.0):
        raise AssertionError(
            f"front phase obs mismatch: got={got_front}, expected={expected_front}"
        )


def _check_deploy_telemetry_last_actions_slice() -> None:
    obs = np.arange(ObservationConfig().num_obs, dtype=np.float64)
    row: dict[str, object] = {
        "obs": obs.tolist(),
        "command": [0.0, 0.0, 0.0, 0.0, 0.26, 0.0, 0.0, 0.0],
        "projected_gravity": [0.0, 0.0, -1.0],
        "joint_pos": [0.0, 0.0, 0.0, 0.0],
        "wheel_pos": [0.0, 0.0],
        "joint_vel": [0.0, 0.0, 0.0, 0.0],
        "wheel_vel": [0.0, 0.0],
        "base_ang_vel_body": [0.0, 0.0, 0.0],
    }
    state = _decode_state_fields(row, base_height_override=None)
    expected = tuple(float(v) for v in obs[25:31].tolist())
    if state["initial_last_action"] != expected:
        raise AssertionError(
            f"deploy telemetry last_action slice mismatch: {state['initial_last_action']}"
        )


def _check_periodic_pd_error() -> None:
    cfg = RobotConfig()
    pos = np.asarray(cfg.default_dof_pos, dtype=np.float64)[JointGroup.CTRL_LEGS]
    target = pos.copy()
    target[0] += 2.0 * math.pi - 0.10
    target[2] -= 2.0 * math.pi - 0.20
    error = policy_leg_position_error_np(target, pos)
    if not np.isclose(error[0], -0.10, atol=1.0e-9):
        raise AssertionError(f"LF front error should choose shortest angle, got {error[0]}")
    if not np.isclose(error[2], 0.20, atol=1.0e-9):
        raise AssertionError(f"RF front error should choose shortest angle, got {error[2]}")

    left_active_error = error[0] - error[1]
    right_active_error = -error[2] + error[3]
    expected_left_active = target[0] - target[1] - (pos[0] - pos[1])
    expected_right_active = -target[2] + target[3] - (-pos[2] + pos[3])
    if not np.isclose(left_active_error, expected_left_active, atol=1.0e-9):
        raise AssertionError("left active rod error was unexpectedly wrapped")
    if not np.isclose(right_active_error, expected_right_active, atol=1.0e-9):
        raise AssertionError("right active rod error was unexpectedly wrapped")


def _check_height_conditioned_decoder() -> None:
    cfg = RobotConfig()
    decoder = PolicyActionDecoder(
        robot_cfg=cfg,
        height_conditioned_action_default=True,
        active_rod_semantics=True,
    )
    decoded = decoder.decode(np.zeros(6), command_height=0.26)
    if decoded.leg_target.shape != (4,):
        raise AssertionError(f"decoded leg target shape mismatch: {decoded.leg_target.shape}")
    if decoded.wheel_vel_target.shape != (2,):
        raise AssertionError(
            f"decoded wheel target shape mismatch: {decoded.wheel_vel_target.shape}"
        )
    lower, upper = cfg.active_rod_angle_limits
    active_left = decoded.leg_target[0] - decoded.leg_target[1]
    active_right = -decoded.leg_target[2] + decoded.leg_target[3]
    if active_left < lower - cfg.active_rod_lower_target_overdrive or active_left > upper:
        raise AssertionError(f"left active target out of bounds: {active_left}")
    if active_right < lower - cfg.active_rod_lower_target_overdrive or active_right > upper:
        raise AssertionError(f"right active target out of bounds: {active_right}")


if __name__ == "__main__":
    raise SystemExit(main())
