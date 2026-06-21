"""检查 recovery deploy runtime 与闭链 sim2sim 的 policy I/O 合同。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from se3_deploy.numpy_policy import NumpyPolicyRuntime
from se3_deploy.observation import RecoveryObservationBuilder, synthetic_recovery_state
from se3_deploy.recovery_runtime import RecoveryActionTargetDecoder

from se3_shared import RECOVERY_COMMAND_HEIGHT_M, JointGroup, PolicyActionDecoder
from se3_shared import RobotConfig as SharedRobotConfig
from se3_sim2sim.cli import build_parser as build_sim2sim_parser
from se3_sim2sim.cli import config_from_args as sim2sim_config_from_args
from se3_sim2sim.config import RobotConfig as SimRobotConfig
from se3_sim2sim.observation import ObservationBuilder
from se3_sim2sim.policy import PolicyRuntime
from se3_sim2sim.robot import WheelLeggedRobot
from se3_sim2sim.runtime_spec import RuntimeSpec

DEFAULT_CHECKPOINT = Path("logs/deploy/model_4999_recovery_obs34_gru.npz")
DEFAULT_MODEL_VARIANT = "closedchain"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that recovery deploy and closed-chain sim2sim share policy I/O."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-variant", default=DEFAULT_MODEL_VARIANT)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    shared_cfg = SharedRobotConfig()
    runtime = RuntimeSpec()

    _check_cli_contract(checkpoint, str(args.model_variant))
    _check_closedchain_robot_contract(str(args.model_variant), runtime)

    deploy_obs = _deploy_observation()
    sim_obs = _sim2sim_observation(runtime)
    _assert_close("observation", deploy_obs, sim_obs, atol=float(args.atol))

    deploy_actions, sim_actions = _policy_actions(checkpoint, deploy_obs)
    _assert_close("policy actions", deploy_actions, sim_actions, atol=float(args.atol))

    deploy_target, sim_target = _decoded_targets(shared_cfg, deploy_actions[-1])
    _assert_close("decoded leg target", deploy_target[:4], sim_target[:4], atol=float(args.atol))
    _assert_close("decoded wheel target", deploy_target[4:], sim_target[4:], atol=float(args.atol))

    print("deploy/sim2sim contract ok")
    print(f"checkpoint={checkpoint}")
    print(f"obs_shape={deploy_obs.shape} max_abs_obs_diff={_max_abs_diff(deploy_obs, sim_obs):.3e}")
    print("last_action=" + np.array2string(deploy_actions[-1], precision=6, suppress_small=False))
    print("leg_target=" + np.array2string(deploy_target[:4], precision=6))
    print("wheel_target=" + np.array2string(deploy_target[4:], precision=6))
    return 0


def _check_cli_contract(checkpoint: Path, model_variant: str) -> None:
    """确认通用 sim2sim CLI 对 recovery deploy npz 自动打开部署合同。"""
    parser = build_sim2sim_parser()
    args = parser.parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--model-variant",
            model_variant,
            "--viewer",
            "none",
            "--max-steps",
            "1",
        ]
    )
    cfg = sim2sim_config_from_args(args)
    if not cfg.robot.height_conditioned_action_default:
        raise AssertionError("sim2sim CLI did not enable height-conditioned recovery contract")


def _check_closedchain_robot_contract(model_variant: str, runtime: RuntimeSpec) -> None:
    """确认闭链 sim2sim 使用主动杆 policy 关节语义。"""
    parser = build_sim2sim_parser()
    args = parser.parse_args(
        [
            "--model-variant",
            model_variant,
            "--height-conditioned-action-default",
            "--viewer",
            "none",
            "--max-steps",
            "1",
        ]
    )
    cfg = sim2sim_config_from_args(args).resolved()
    robot = WheelLeggedRobot(cfg=cfg.robot, runtime=runtime)
    if tuple(robot.policy_joint_names) != JointGroup.POLICY_JOINT_NAMES:
        raise AssertionError(f"unexpected policy joints: {robot.policy_joint_names}")
    if not robot.active_rod_action_semantics:
        raise AssertionError("closed-chain sim2sim did not use active-rod action semantics")
    if not robot.action_decoder.height_conditioned_action_default:
        raise AssertionError("closed-chain sim2sim action decoder is not height-conditioned")


def _deploy_observation() -> np.ndarray:
    """用 deploy builder 从同一个 STM32 synthetic state 拼装观测。"""
    builder = RecoveryObservationBuilder()
    state = synthetic_recovery_state(seq=0)
    last_action = np.zeros(6, dtype=np.float32)
    return builder.build(state, last_action).obs


def _sim2sim_observation(runtime: RuntimeSpec) -> np.ndarray:
    """用 sim2sim builder 拼装与 deploy synthetic state 等价的观测。"""
    shared_cfg = SharedRobotConfig()
    state = synthetic_recovery_state(seq=0)
    sim_robot_cfg = SimRobotConfig(height_conditioned_action_default=True)
    builder = ObservationBuilder(
        robot_cfg=sim_robot_cfg,
        runtime=runtime,
        default_dof_pos=np.asarray(shared_cfg.default_dof_pos, dtype=np.float64),
    )
    return builder.build(
        base_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        base_ang_vel_world=np.asarray(state.base_ang_vel_body, dtype=np.float64),
        dof_pos=np.asarray(state.dof_pos, dtype=np.float64),
        dof_vel=np.asarray(state.dof_vel, dtype=np.float64),
        command=np.asarray(RecoveryObservationBuilder().command, dtype=np.float64),
        action_obs=np.zeros(6, dtype=np.float64),
    )


def _policy_actions(checkpoint: Path, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """比较 deploy NumPy runtime 与 sim2sim PolicyRuntime 的 GRU 输出序列。"""
    deploy_policy = NumpyPolicyRuntime(checkpoint)
    sim_policy = PolicyRuntime(checkpoint=checkpoint, device="cpu", runtime=RuntimeSpec())
    deploy_policy.reset()
    sim_policy.reset()
    deploy_actions: list[np.ndarray] = []
    sim_actions: list[np.ndarray] = []
    for scale in (1.0, 0.95, 1.05):
        step_obs = np.asarray(obs * scale, dtype=np.float32)
        deploy_actions.append(deploy_policy.act(step_obs))
        sim_actions.append(sim_policy.act(step_obs))
    return np.stack(deploy_actions), np.stack(sim_actions)


def _decoded_targets(
    shared_cfg: SharedRobotConfig, action: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """比较 deploy target decoder 与 sim2sim 共享 action decoder 的物理目标。"""
    command_height = float(RECOVERY_COMMAND_HEIGHT_M)
    deploy = RecoveryActionTargetDecoder(command_height=command_height, robot_cfg=shared_cfg)
    deploy_target = deploy.decode(action)
    sim = PolicyActionDecoder(
        robot_cfg=shared_cfg,
        height_conditioned_action_default=True,
        active_rod_semantics=True,
        dtype=np.float64,
    )
    sim_decoded = sim.decode(action, command_height=command_height)
    return (
        np.asarray([*deploy_target.joint_pos, *deploy_target.wheel_vel], dtype=np.float64),
        np.asarray([*sim_decoded.leg_target, *sim_decoded.wheel_vel_target], dtype=np.float64),
    )


def _assert_close(name: str, lhs: np.ndarray, rhs: np.ndarray, *, atol: float) -> None:
    if not np.allclose(lhs, rhs, atol=atol, rtol=0.0):
        diff = _max_abs_diff(lhs, rhs)
        raise AssertionError(f"{name} mismatch: max_abs_diff={diff:.6g}, atol={atol}")


def _max_abs_diff(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(lhs) - np.asarray(rhs))))


if __name__ == "__main__":
    raise SystemExit(main())
