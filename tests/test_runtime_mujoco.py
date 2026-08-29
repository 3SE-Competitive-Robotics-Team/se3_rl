"""共用 control loop 与原生 MuJoCo adapter 的闭环测试。"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from test_onnx_metadata import _fake_env, _policy_input, _write_runtime_model

from se3_runtime import (
    DecodedPolicyAction,
    PolicyActionDecoder,
    PolicyBundle,
    PolicyControlLoop,
    PolicyInput,
    PolicyRuntime,
    SerialLegActuatorController,
)
from se3_runtime._serialleg_v1 import knee_gas_spring_compensation_torque
from se3_runtime_mujoco import (
    MujocoPolicyAdapter,
    MujocoViserViewer,
)
from se3_shared.fourbar import (
    knee_gas_spring_compensation_torque_np,
    knee_gas_spring_compensation_torque_torch,
)
from se3_train.onnx_metadata import (
    build_deployment_onnx_metadata,
    embed_onnx_metadata,
)

_MODEL_PATH = Path(
    "assets/robots/serialleg/mjcf/serialleg_closed_chain_v3_train_obb_trim.xml"
).resolve()


class _RecordingAdapter:
    """记录统一 loop 的调用顺序，不实现任何仿真器语义。"""

    def __init__(self, policy_input: PolicyInput, physics_dt_s: float) -> None:
        self._policy_input = policy_input
        self._physics_dt_s = physics_dt_s
        self.reset_count = 0
        self.read_count = 0
        self.tick_count = 0
        self.applied_actions: list[DecodedPolicyAction] = []

    @property
    def physics_dt_s(self) -> float:
        return self._physics_dt_s

    def reset(self) -> None:
        self.reset_count += 1
        self.read_count = 0
        self.tick_count = 0
        self.applied_actions.clear()

    def read_policy_input(self) -> PolicyInput:
        self.read_count += 1
        return self._policy_input

    def apply_decoded_action(self, action: DecodedPolicyAction) -> None:
        self.applied_actions.append(action)

    def advance_physics_tick(self) -> None:
        self.tick_count += 1


def _runtime_artifact(path: Path, action_values: list[float]) -> PolicyRuntime:
    metadata = build_deployment_onnx_metadata(
        _fake_env(),
        observation_group_names=("actor",),
    )
    _write_runtime_model(path, 34, action_values, recurrent=False)
    embed_onnx_metadata(
        path,
        metadata,
        policy_iteration=20,
        is_rnn=False,
    )
    return PolicyRuntime.load(path, action_delay_random_seed=1)


class PolicyControlLoopTests(unittest.TestCase):
    def test_one_policy_step_runs_exact_metadata_decimation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runtime = _runtime_artifact(
                Path(temp_dir) / "loop.onnx",
                [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
            )
            adapter = _RecordingAdapter(
                _policy_input(),
                runtime.contract.timing.physics_dt_s,
            )
            loop = PolicyControlLoop(runtime, adapter)
            loop.reset()
            result = loop.policy_step()

        self.assertEqual(adapter.reset_count, 1)
        self.assertEqual(adapter.read_count, 1)
        self.assertEqual(adapter.tick_count, 4)
        self.assertEqual(len(adapter.applied_actions), 4)
        self.assertEqual(result.policy_step, 1)
        self.assertEqual(result.physics_ticks, 4)
        np.testing.assert_array_equal(
            adapter.applied_actions[0].clipped_action,
            np.zeros(6),
        )
        np.testing.assert_allclose(
            adapter.applied_actions[-1].clipped_action,
            (0.1, -0.2, 0.3, -0.4, 0.5, -0.6),
        )


class MujocoPolicyAdapterTests(unittest.TestCase):
    def test_mlp_artifact_runs_one_headless_closed_loop_step(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runtime = _runtime_artifact(
                Path(temp_dir) / "mujoco.onnx",
                [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
            )
            adapter = MujocoPolicyAdapter(
                runtime.contract,
                model_path=_MODEL_PATH,
                artifact_path=runtime.bundle.path,
            )
            policy_input = adapter.read_policy_input()
            expected_leg_default = PolicyActionDecoder(runtime.contract).policy_default(
                adapter.commands
            )
            np.testing.assert_allclose(
                policy_input.robot.policy_joint_position,
                np.concatenate((expected_leg_default, np.zeros(2))),
                rtol=0.0,
                atol=1.0e-9,
            )
            np.testing.assert_allclose(
                policy_input.robot.projected_gravity_body,
                (0.0, 0.0, -1.0),
                rtol=0.0,
                atol=1.0e-12,
            )

            loop = PolicyControlLoop(runtime, adapter)
            loop.reset()
            result = loop.policy_step()
            control = adapter.last_control

        self.assertEqual(result.physics_ticks, 4)
        self.assertEqual(adapter.physics_ticks, 4)
        self.assertAlmostEqual(float(adapter.data.time), 0.02, places=12)
        self.assertIsNotNone(control)
        assert control is not None
        self.assertTrue(np.isfinite(control.effort).all())
        self.assertTrue((control.effort >= control.lower_effort_limit).all())
        self.assertTrue((control.effort <= control.upper_effort_limit).all())

    def test_v2_rejects_missing_model_override(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            runtime = _runtime_artifact(
                temp_path / "asset.onnx",
                [0.0] * 6,
            )
            missing_model = temp_path / "missing.xml"

            with self.assertRaisesRegex(FileNotFoundError, "MJCF 不存在"):
                MujocoPolicyAdapter(
                    runtime.contract,
                    model_path=missing_model,
                    artifact_path=runtime.bundle.path,
                )

    def test_viser_server_accepts_mujoco_snapshot_and_closes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runtime = _runtime_artifact(
                Path(temp_dir) / "viser.onnx",
                [0.0] * 6,
            )
            adapter = MujocoPolicyAdapter(
                runtime.contract,
                model_path=_MODEL_PATH,
                artifact_path=runtime.bundle.path,
            )
            viewer = MujocoViserViewer(
                model=adapter.model,
                contract=runtime.contract,
                initial_commands=adapter.commands,
                artifact_path=runtime.bundle.path,
                host="127.0.0.1",
                port=0,
                verbose=False,
            )
            try:
                viewer.update(
                    adapter.data,
                    policy_step=0,
                    telemetry=adapter.telemetry(),
                    force=True,
                )
                events = viewer.poll_events()
                self.assertEqual(events.command_updates, ())
                self.assertFalse(events.reset_requested)
                self.assertFalse(events.stop_requested)
                self.assertFalse(events.paused)
                self.assertRegex(viewer.url, r"^http://127\.0\.0\.1:\d+/$")
            finally:
                viewer.close()

            self.assertTrue(viewer.closed)
            viewer.close()


class KneeGasSpringCompensationTests(unittest.TestCase):
    """校验 sim2sim 与训练端使用同一份气弹簧前馈力矩。"""

    def test_runtime_matches_training_compensation_torque(self) -> None:
        rng = np.random.default_rng(20260826)
        samples = rng.uniform(-1.5, 1.5, size=(512, 4))
        runtime = np.stack([knee_gas_spring_compensation_torque(row, 300.0) for row in samples])

        np.testing.assert_allclose(
            runtime,
            knee_gas_spring_compensation_torque_np(samples, 300.0),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            runtime,
            knee_gas_spring_compensation_torque_torch(
                torch.as_tensor(samples, dtype=torch.float64),
                300.0,
            ).numpy(),
            atol=1.0e-6,
        )

    def test_controller_adds_compensation_before_torque_limit(self) -> None:
        position = np.asarray((0.35, -0.45, -0.35, 0.45, 0.0, 0.0))
        velocity = np.zeros(6)
        action = _decoded_action(position[:4])
        with TemporaryDirectory() as temp_dir:
            enabled = _contract_with_gas_spring(
                Path(temp_dir) / "spring.onnx",
                declared=True,
            )
            legacy = _contract_with_gas_spring(
                Path(temp_dir) / "legacy.onnx",
                declared=False,
            )

        self.assertTrue(enabled.robot.knee_gas_spring.compensation_enabled)
        self.assertEqual(enabled.robot.knee_gas_spring.force, 300.0)
        self.assertFalse(legacy.robot.knee_gas_spring.compensation_enabled)

        with_spring = SerialLegActuatorController(enabled).compute(
            action,
            policy_joint_position=position,
            policy_joint_velocity=velocity,
        )
        without_spring = SerialLegActuatorController(legacy).compute(
            action,
            policy_joint_position=position,
            policy_joint_velocity=velocity,
        )
        expected = knee_gas_spring_compensation_torque(position[:4], 300.0)

        np.testing.assert_allclose(with_spring.knee_gas_spring_compensation, expected)
        np.testing.assert_allclose(without_spring.knee_gas_spring_compensation, np.zeros(4))
        np.testing.assert_allclose(
            with_spring.unclipped_effort[:4] - without_spring.unclipped_effort[:4],
            expected,
        )
        np.testing.assert_allclose(
            with_spring.unclipped_effort[4:6],
            without_spring.unclipped_effort[4:6],
        )
        np.testing.assert_array_less(
            np.abs(with_spring.effort),
            with_spring.upper_effort_limit + 1.0e-9,
        )


def _decoded_action(leg_position_target: np.ndarray) -> DecodedPolicyAction:
    """构造一个只用于 actuator 力矩计算的最小动作。"""
    return DecodedPolicyAction(
        clipped_action=np.zeros(6),
        leg_position_target=np.asarray(leg_position_target, dtype=np.float64),
        wheel_velocity_target=np.zeros(2),
        policy_leg_default=np.zeros(4),
        active_rod_angle_target=np.zeros(2),
        active_rod_target_clamped=np.zeros(2, dtype=bool),
    )


def _contract_with_gas_spring(path: Path, *, declared: bool):
    """导出一个带或不带 `robot.knee_gas_spring` 声明的 artifact 契约。"""
    metadata = build_deployment_onnx_metadata(
        _fake_env(),
        observation_group_names=("actor",),
    )
    if not declared:
        del metadata["robot"]["knee_gas_spring"]
    _write_runtime_model(path, 34, [0.0] * 6, recurrent=False)
    embed_onnx_metadata(path, metadata, policy_iteration=20, is_rnn=False)
    return PolicyBundle.load(path).contract


if __name__ == "__main__":
    unittest.main()
