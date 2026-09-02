"""ONNX 部署 metadata 外部契约测试。"""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

import numpy as np
import onnx
from mjlab.sim import MujocoCfg
from onnx import TensorProto, helper

from se3_runtime import (
    PolicyActionDecoder,
    PolicyBundle,
    PolicyBundleError,
    PolicyContractError,
    PolicyInput,
    PolicyObservationBuilder,
    PolicyRuntime,
    RobotState,
)
from se3_shared import (
    DM8009P,
    M3508_C620_14,
    RobotConfig,
    build_policy_observation_np,
)
from se3_shared import (
    PolicyActionDecoder as SharedPolicyActionDecoder,
)
from se3_train.onnx_metadata import (
    ONNX_METADATA_KEY,
    SCHEMA_NAME,
    build_deployment_onnx_metadata,
    embed_onnx_metadata,
)

_TERM_WIDTHS = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "commands": 5,
    "leg_joint_pos": 6,
    "leg_joint_vel": 4,
    "wheel_pos_zero": 2,
    "wheel_vel": 2,
    "last_actions": 6,
    "jump_commands": 3,
}


class _FakeObservationManager:
    """只实现 metadata builder 所依赖的 MJLab observation manager 接口。"""

    def __init__(self, history_length: int) -> None:
        term_names = list(_TERM_WIDTHS)
        self.active_terms = {"actor": term_names}
        self.group_obs_concatenate = {"actor": True}
        self.group_obs_term_dim = {
            "actor": [(width * history_length,) for width in _TERM_WIDTHS.values()]
        }
        self.cfg = {"actor": SimpleNamespace(concatenate_dim=-1)}
        self._term_cfgs = {
            name: SimpleNamespace(
                history_length=0 if history_length == 1 else history_length,
                flatten_history_dim=True,
                scale=None,
                clip=None,
                delay_max_lag=0,
            )
            for name in term_names
        }

    def get_term_cfg(self, group_name: str, term_name: str) -> SimpleNamespace:
        self._require_actor(group_name)
        return self._term_cfgs[term_name]

    @staticmethod
    def _require_actor(group_name: str) -> None:
        if group_name != "actor":
            raise KeyError(group_name)


class _FakeCommandManager:
    """提供可在构建间动态修改的 live command cfg。"""

    def __init__(self) -> None:
        self.active_terms = ["velocity_height"]
        self.cfg = SimpleNamespace(
            lin_vel_x_range=(-1.0, 1.0),
            ang_vel_yaw_range=(-2.0, 2.0),
            pitch_range=(-0.2, 0.2),
            roll_range=(-0.1, 0.1),
            height_range=(0.2, 0.32),
            standing_height_range=(0.2, 0.32),
            resampling_time_range=(5.0, 5.0),
            standing_ratio=0.1,
            lin_vel_deadband=0.1,
            yaw_deadband=0.1,
            constrain_diff_drive_commands=True,
            diff_drive_wheel_radius=0.06,
            diff_drive_half_track=0.2,
            diff_drive_max_wheel_speed=45.0,
            diff_drive_wheel_speed_fraction=0.9,
            jump_prob=0.05,
            jump_cool_down_steps=100,
            jump_height_range=(0.1, 0.6),
        )
        self._term = SimpleNamespace(
            cfg=self.cfg,
            command=SimpleNamespace(shape=(1, 8)),
        )

    def get_term(self, name: str) -> SimpleNamespace:
        if name != "velocity_height":
            raise KeyError(name)
        return self._term


class _FakeActionManager:
    """提供 SerialLeg 6D delayed_action 的 live cfg。"""

    def __init__(self) -> None:
        self.active_terms = ["delayed_action"]
        self.total_action_dim = 6
        self.cfg = SimpleNamespace(
            leg_scales=(0.25, 0.25, 0.25, 0.25),
            wheel_scale=45.0,
            action_clip=100.0,
            action_delay_enabled=True,
            action_delay_s=0.005,
            action_delay_randomize=True,
            action_delay_min_s=0.004,
            action_delay_max_s=0.006,
            height_conditioned_action_default=True,
            action_default_command_name="velocity_height",
            active_rod_lower_target_overdrive=0.2,
            knee_gas_spring_force=300.0,
            knee_gas_spring_compensation_enabled=True,
        )
        self._term = SimpleNamespace(cfg=self.cfg)

    def get_term(self, name: str) -> SimpleNamespace:
        if name != "delayed_action":
            raise KeyError(name)
        return self._term


def _fake_env(history_length: int = 1) -> SimpleNamespace:
    robot = RobotConfig()
    policy_joint_names = (
        "lf0_Joint",
        "l_drive_bar_Joint",
        "rf0_Joint",
        "r_drive_bar_Joint",
        "l_wheel_Joint",
        "r_wheel_Joint",
    )
    compiled_armature = np.asarray([0.01] * 4 + [0.005] * 2, dtype=np.float64)
    leg_actuator = SimpleNamespace(
        target_names=[
            "lf0_Joint",
            "l_drive_bar_Joint",
            "rf0_Joint",
            "r_drive_bar_Joint",
        ],
        cfg=SimpleNamespace(
            stiffness=robot.leg_kp,
            damping=robot.leg_kd,
            saturation_effort=DM8009P.stall_torque,
            effort_limit=DM8009P.rated_torque,
            velocity_limit=DM8009P.no_load_speed,
        ),
    )
    wheel_actuator = SimpleNamespace(
        target_names=["l_wheel_Joint", "r_wheel_Joint"],
        cfg=SimpleNamespace(
            stiffness=0.0,
            damping=robot.wheel_kd,
            saturation_effort=M3508_C620_14.stall_torque,
            effort_limit=M3508_C620_14.rated_torque,
            velocity_limit=M3508_C620_14.no_load_speed,
        ),
    )
    robot_entity = SimpleNamespace(
        actuators=[leg_actuator, wheel_actuator],
        joint_names=policy_joint_names,
        indexing=SimpleNamespace(joint_v_adr=np.arange(6, dtype=np.int64)),
    )
    return SimpleNamespace(
        physics_dt=0.005,
        step_dt=0.02,
        num_envs=1,
        cfg=SimpleNamespace(
            decimation=4,
            sim=SimpleNamespace(mujoco=MujocoCfg(timestep=0.005)),
        ),
        sim=SimpleNamespace(
            mj_model=SimpleNamespace(dof_armature=compiled_armature),
        ),
        scene={"robot": robot_entity},
        observation_manager=_FakeObservationManager(history_length),
        command_manager=_FakeCommandManager(),
        action_manager=_FakeActionManager(),
    )


def _write_test_model(path: Path, obs_dim: int, *, recurrent: bool) -> None:
    inputs = [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_dim])]
    outputs = [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 6])]
    nodes = [
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["actions"],
            value=helper.make_tensor("action_value", TensorProto.FLOAT, [1, 6], [0.0] * 6),
        )
    ]
    if recurrent:
        inputs.append(helper.make_tensor_value_info("h_in", TensorProto.FLOAT, [1, 1, 512]))
        outputs.append(helper.make_tensor_value_info("h_out", TensorProto.FLOAT, [1, 1, 512]))
        nodes.append(helper.make_node("Identity", inputs=["h_in"], outputs=["h_out"]))

    graph = helper.make_graph(nodes, "metadata_contract", inputs, outputs)
    model = helper.make_model(
        graph,
        producer_name="se3-test",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    prop = model.metadata_props.add()
    prop.key = "existing.key"
    prop.value = "preserved"
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _write_runtime_model(
    path: Path,
    obs_dim: int,
    action_values: list[float],
    *,
    recurrent: bool,
) -> None:
    """写入固定 action、GRU hidden 每步加一的 runtime 测试模型。"""
    inputs = [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_dim])]
    outputs = [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 6])]
    nodes = [
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["actions"],
            value=helper.make_tensor(
                "action_value",
                TensorProto.FLOAT,
                [1, 6],
                action_values,
            ),
        )
    ]
    if recurrent:
        inputs.append(helper.make_tensor_value_info("h_in", TensorProto.FLOAT, [1, 1, 512]))
        outputs.append(helper.make_tensor_value_info("h_out", TensorProto.FLOAT, [1, 1, 512]))
        nodes.extend(
            (
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["hidden_increment"],
                    value=helper.make_tensor(
                        "hidden_increment_value",
                        TensorProto.FLOAT,
                        [1, 1, 512],
                        [1.0] * 512,
                    ),
                ),
                helper.make_node(
                    "Add",
                    inputs=["h_in", "hidden_increment"],
                    outputs=["h_out"],
                ),
            )
        )

    graph = helper.make_graph(nodes, "runtime_state", inputs, outputs)
    model = helper.make_model(
        graph,
        producer_name="se3-runtime-test",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _policy_input(
    *,
    base_ang_vel: tuple[float, float, float] = (0.4, -0.8, 1.2),
) -> PolicyInput:
    robot = RobotConfig()
    joint_position = np.asarray(robot.default_dof_pos, dtype=np.float64) + np.asarray(
        (0.12, -0.04, -0.08, 0.03, 7.0, -9.0),
        dtype=np.float64,
    )
    return PolicyInput(
        robot=RobotState(
            base_angular_velocity_body=np.asarray(base_ang_vel, dtype=np.float64),
            projected_gravity_body=np.asarray((0.1, -0.2, -0.97), dtype=np.float64),
            policy_joint_position=joint_position,
            policy_joint_velocity=np.asarray(
                (0.4, -0.8, 1.2, -1.6, 5.0, -7.0),
                dtype=np.float64,
            ),
        ),
        commands={
            "velocity_height": np.asarray(
                (0.6, -0.4, 0.03, -0.05, 0.26, 1.0, 0.42, 0.7),
                dtype=np.float64,
            )
        },
    )


def _rewrite_descriptor(path: Path, mutate: Any) -> None:
    model = onnx.load(path)
    prop = next(item for item in model.metadata_props if item.key == ONNX_METADATA_KEY)
    payload = json.loads(prop.value)
    mutate(payload)
    prop.value = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    onnx.save(model, path)


class OnnxMetadataTests(unittest.TestCase):
    def test_single_frame_descriptor_matches_kyber_shape_and_live_cfg(self) -> None:
        env = _fake_env()
        metadata = build_deployment_onnx_metadata(
            env,
            observation_group_names=("actor",),
        )

        self.assertEqual(metadata["meta"]["schema_name"], SCHEMA_NAME)
        self.assertEqual(
            set(metadata),
            {"meta", "robot", "commands", "policy_io"},
        )
        self.assertEqual(
            set(metadata["policy_io"]),
            {"groups", "references", "action"},
        )
        self.assertNotIn("contract_hash", metadata["meta"])
        self.assertNotIn("onnx", metadata["policy_io"])
        self.assertNotIn("policy", metadata["policy_io"])
        terms = metadata["policy_io"]["groups"]["actor"]["terms"]
        self.assertEqual([term["name"] for term in terms], list(_TERM_WIDTHS))
        self.assertTrue(all(term["history_length"] == 1 for term in terms))
        self.assertEqual(metadata["policy_io"]["action"]["scale"], [0.25] * 4 + [45.0] * 2)
        self.assertEqual(metadata["meta"]["sim"]["inference_hz"], 50.0)
        self.assertEqual(
            metadata["commands"],
            {"velocity_height": {"dimension": 8}},
        )
        self.assertNotIn("ranges", metadata["commands"]["velocity_height"])
        self.assertEqual(metadata["robot"]["KD"][4:6], [RobotConfig().wheel_kd] * 2)
        self.assertEqual(metadata["robot"]["armature"], [0.01] * 4 + [0.005] * 2)
        self.assertEqual(metadata["robot"]["saturation_effort"][:4], [40.0] * 4)
        self.assertEqual(
            metadata["robot"]["saturation_effort"][4:6],
            [M3508_C620_14.stall_torque] * 2,
        )
        self.assertEqual(metadata["robot"]["effort_limit"][:4], [20.0] * 4)
        self.assertEqual(metadata["robot"]["torque_speed_curve"], [None] * 6)
        self.assertEqual(
            metadata["robot"]["knee_gas_spring"],
            {"force": 300.0, "compensation_enabled": True},
        )
        self.assertAlmostEqual(
            metadata["robot"]["velocity_limit"][4],
            482.0 * 19.0 / 14.0 * 2.0 * np.pi / 60.0,
        )
        self.assertFalse(metadata["meta"]["assets"]["robot_config_overridden"])

        env.command_manager.cfg.lin_vel_x_range = (-2.5, 3.0)
        env.scene["robot"].actuators[1].cfg.damping = 0.31
        updated = build_deployment_onnx_metadata(
            env,
            observation_group_names=("actor",),
        )
        self.assertEqual(updated["commands"], metadata["commands"])
        self.assertEqual(updated["robot"]["KD"][4:6], [0.31, 0.31])
        self.assertTrue(updated["meta"]["assets"]["robot_config_overridden"])

    def test_history_mlp_descriptor_records_only_per_term_history(self) -> None:
        metadata = build_deployment_onnx_metadata(
            _fake_env(history_length=5),
            observation_group_names=("actor",),
        )

        group = metadata["policy_io"]["groups"]["actor"]
        self.assertEqual(set(group), {"terms"})
        self.assertEqual(
            sum(_TERM_WIDTHS[term["name"]] * term["history_length"] for term in group["terms"]),
            170,
        )
        self.assertTrue(all(term["history_length"] == 5 for term in group["terms"]))

    def test_explicit_final_curriculum_ranges_reach_runtime_contract(self) -> None:
        env = _fake_env()
        expected_ranges = {
            "lin_vel_x": (-1.89, 1.89),
            "ang_vel_yaw": (-9.41, 9.41),
            "pitch": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "height": (0.195, 0.390),
            "jump_flag": (0.0, 0.0),
            "jump_target_height": (0.0, 0.0),
            "jump_phase": (0.0, 0.0),
        }
        env.command_manager.cfg.deployment_ranges = expected_ranges
        metadata = build_deployment_onnx_metadata(
            env,
            observation_group_names=("actor",),
        )
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "command_ranges.onnx"
            _write_test_model(model_path, 34, recurrent=False)
            embed_onnx_metadata(
                model_path,
                metadata,
                policy_iteration=4999,
                is_rnn=False,
            )

            command = PolicyBundle.load(model_path).contract.command("velocity_height")

        self.assertEqual(
            {field.name: field.value_range for field in command.fields},
            expected_ranges,
        )

    def test_action_default_semantics_reach_runtime_contract(self) -> None:
        """B18：hcad 必须写入 metadata 并驱动 runtime 的 default strategy。"""
        for flag, expected_mode in (
            (True, "serialleg_height_conditioned_policy_default.v1"),
            (False, "entity_default_joint_position"),
        ):
            env = _fake_env()
            env.action_manager.cfg.height_conditioned_action_default = flag
            metadata = build_deployment_onnx_metadata(
                env,
                observation_group_names=("actor",),
            )
            self.assertIs(
                metadata["policy_io"]["action"]["height_conditioned_action_default"],
                flag,
            )
            with TemporaryDirectory() as temp_dir:
                model_path = Path(temp_dir) / f"hcad_{flag}.onnx"
                _write_test_model(model_path, 34, recurrent=False)
                embed_onnx_metadata(
                    model_path,
                    metadata,
                    policy_iteration=4999,
                    is_rnn=False,
                )
                strategy = PolicyBundle.load(model_path).contract.action.default_strategy

            self.assertEqual(strategy.mode, expected_mode)
            if not flag:
                self.assertEqual(
                    list(strategy.policy_leg_position),
                    metadata["policy_io"]["action"]["offset"][:4],
                )

    def test_legacy_v2_without_action_default_flag_keeps_height_conditioned(self) -> None:
        """旧 v2 artifact 不带 hcad 字段时，runtime 必须保持高度条件语义。"""
        env = _fake_env()
        metadata = build_deployment_onnx_metadata(
            env,
            observation_group_names=("actor",),
        )
        del metadata["policy_io"]["action"]["height_conditioned_action_default"]
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "legacy_no_flag.onnx"
            _write_test_model(model_path, 34, recurrent=False)
            embed_onnx_metadata(
                model_path,
                metadata,
                policy_iteration=4999,
                is_rnn=False,
            )
            strategy = PolicyBundle.load(model_path).contract.action.default_strategy

        self.assertEqual(strategy.mode, "serialleg_height_conditioned_policy_default.v1")

    def test_legacy_v2_without_command_ranges_uses_compat_bounds(self) -> None:
        """旧 v2 artifact 不带 ranges 时，runtime 继续使用兼容边界。"""
        env = _fake_env()
        metadata = build_deployment_onnx_metadata(
            env,
            observation_group_names=("actor",),
        )
        metadata["commands"]["velocity_height"].pop("ranges", None)
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "legacy_no_ranges.onnx"
            _write_test_model(model_path, 34, recurrent=False)
            embed_onnx_metadata(
                model_path,
                metadata,
                policy_iteration=4999,
                is_rnn=False,
            )
            command = PolicyBundle.load(model_path).contract.command("velocity_height")

        by_name = {field.name: field for field in command.fields}
        self.assertEqual(by_name["height"].value_range, (0.20, 0.32))
        self.assertEqual(by_name["height"].default_value, 0.22)

    def test_embed_mlp_preserves_properties_and_does_not_mutate_builder_output(self) -> None:
        metadata = build_deployment_onnx_metadata(
            _fake_env(),
            observation_group_names=("actor",),
        )
        original = deepcopy(metadata)
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model_7.onnx"
            _write_test_model(model_path, 34, recurrent=False)
            embedded = embed_onnx_metadata(
                model_path,
                metadata,
                policy_iteration=7,
                is_rnn=False,
            )
            model = onnx.load(model_path)

        props = {prop.key: prop.value for prop in model.metadata_props}
        stored = json.loads(props[ONNX_METADATA_KEY])
        self.assertEqual(metadata, original)
        self.assertEqual(props["existing.key"], "preserved")
        self.assertEqual(stored, embedded)
        self.assertEqual(stored["meta"]["training"]["policy_iteration"], 7)
        self.assertFalse(stored["meta"]["training"]["is_rnn"])
        self.assertIsInstance(stored["meta"]["training"]["export_timestamp"], str)
        self.assertIsInstance(stored["meta"]["training"]["export_tag"], str)
        self.assertNotIn("source_checkpoint", stored["meta"]["training"])
        self.assertEqual(set(stored["policy_io"]), {"groups", "references", "action"})

    def test_embed_gru_records_only_is_rnn_and_uses_graph_as_io_source(self) -> None:
        metadata = build_deployment_onnx_metadata(
            _fake_env(),
            observation_group_names=("actor",),
        )
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model_8.onnx"
            _write_test_model(model_path, 34, recurrent=True)
            embedded = embed_onnx_metadata(
                model_path,
                metadata,
                policy_iteration=8,
                is_rnn=True,
            )
            model = onnx.load(model_path)

        self.assertTrue(embedded["meta"]["training"]["is_rnn"])
        self.assertEqual([value.name for value in model.graph.input], ["obs", "h_in"])
        self.assertEqual([value.name for value in model.graph.output], ["actions", "h_out"])
        self.assertNotIn("onnx", embedded["policy_io"])

    def test_embed_rejects_history_descriptor_with_single_frame_graph(self) -> None:
        metadata = build_deployment_onnx_metadata(
            _fake_env(history_length=5),
            observation_group_names=("actor",),
        )
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "bad.onnx"
            _write_test_model(model_path, 34, recurrent=False)
            with self.assertRaisesRegex(ValueError, "170D"):
                embed_onnx_metadata(
                    model_path,
                    metadata,
                    policy_iteration=1,
                    is_rnn=False,
                )


class PolicyBundleTests(unittest.TestCase):
    def test_loads_34d_mlp_and_contract_is_deeply_immutable(self) -> None:
        metadata = build_deployment_onnx_metadata(
            _fake_env(),
            observation_group_names=("actor",),
        )
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "mlp.onnx"
            _write_test_model(model_path, 34, recurrent=False)
            embed_onnx_metadata(
                model_path,
                metadata,
                policy_iteration=3,
                is_rnn=False,
            )

            bundle = PolicyBundle.load(model_path)
            actions = bundle.session.run(
                ["actions"],
                {"obs": np.zeros((1, 34), dtype=np.float32)},
            )[0]

        self.assertEqual(bundle.contract.policy_type, "mlp")
        self.assertEqual(bundle.contract.frame_dim, 34)
        self.assertEqual(bundle.contract.input_dim, 34)
        self.assertEqual(bundle.contract.action.dimension, 6)
        self.assertEqual(bundle.contract.schema_version, 2)
        self.assertIsNone(bundle.contract.robot.asset.sha256)
        self.assertEqual(
            bundle.contract.robot.actuators.peak_effort_limit,
            (DM8009P.stall_torque,) * 4 + (M3508_C620_14.stall_torque,) * 2,
        )
        self.assertEqual(
            bundle.contract.robot.actuators.rated_effort_limit,
            (DM8009P.rated_torque,) * 4 + (M3508_C620_14.rated_torque,) * 2,
        )
        self.assertEqual(actions.shape, (1, 6))
        with self.assertRaises(FrozenInstanceError):
            bundle.contract.input_dim = 1
        with self.assertRaises(TypeError):
            bundle.contract.metadata["meta"]["schema_name"] = "changed"

    def test_legacy_v2_uses_canonical_tn_limits_when_actuator_is_not_overridden(self) -> None:
        metadata = build_deployment_onnx_metadata(
            _fake_env(),
            observation_group_names=("actor",),
        )
        metadata["robot"].pop("saturation_effort")
        metadata["robot"].pop("torque_speed_curve")
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "legacy_v2.onnx"
            _write_test_model(model_path, 34, recurrent=False)
            embed_onnx_metadata(
                model_path,
                metadata,
                policy_iteration=3,
                is_rnn=False,
            )

            actuators = PolicyBundle.load(model_path).contract.robot.actuators

        self.assertEqual(
            actuators.peak_effort_limit,
            (DM8009P.stall_torque,) * 4 + (M3508_C620_14.stall_torque,) * 2,
        )
        self.assertEqual(actuators.torque_speed_curve, (None,) * 6)

    def test_loads_170d_history_mlp_contract(self) -> None:
        metadata = build_deployment_onnx_metadata(
            _fake_env(history_length=5),
            observation_group_names=("actor",),
        )
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "history_mlp.onnx"
            _write_test_model(model_path, 170, recurrent=False)
            embed_onnx_metadata(
                model_path,
                metadata,
                policy_iteration=4,
                is_rnn=False,
            )
            contract = PolicyBundle.load(model_path).contract

        actor = contract.group("actor")
        self.assertEqual(contract.frame_dim, 34)
        self.assertEqual(contract.input_dim, 170)
        self.assertEqual(actor.history_mode, "per_term_circular_buffer")
        self.assertEqual(actor.history_length, 5)
        self.assertTrue(all(term.history_length == 5 for term in actor.terms))

    def test_loads_gru_with_explicit_hidden_state(self) -> None:
        metadata = build_deployment_onnx_metadata(
            _fake_env(),
            observation_group_names=("actor",),
        )
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "gru.onnx"
            _write_test_model(model_path, 34, recurrent=True)
            embed_onnx_metadata(
                model_path,
                metadata,
                policy_iteration=5,
                is_rnn=True,
            )
            bundle = PolicyBundle.load(model_path)
            outputs = bundle.session.run(
                ["actions", "h_out"],
                {
                    "obs": np.zeros((1, 34), dtype=np.float32),
                    "h_in": np.zeros((1, 1, 512), dtype=np.float32),
                },
            )

        self.assertEqual(bundle.contract.policy_type, "gru")
        self.assertTrue(bundle.contract.is_recurrent)
        self.assertEqual([item.name for item in bundle.contract.inputs], ["obs", "h_in"])
        self.assertEqual(outputs[0].shape, (1, 6))
        self.assertEqual(outputs[1].shape, (1, 1, 512))

    def test_rejects_missing_versioned_metadata(self) -> None:
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "plain.onnx"
            _write_test_model(model_path, 34, recurrent=False)
            with self.assertRaisesRegex(PolicyBundleError, "缺少 metadata key"):
                PolicyBundle.load(model_path)

    def test_rejects_wrong_descriptor_schema(self) -> None:
        metadata = build_deployment_onnx_metadata(
            _fake_env(),
            observation_group_names=("actor",),
        )
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "schema.onnx"
            _write_test_model(model_path, 34, recurrent=False)
            embed_onnx_metadata(
                model_path,
                metadata,
                policy_iteration=6,
                is_rnn=False,
            )
            _rewrite_descriptor(
                model_path,
                lambda payload: payload["meta"].__setitem__("schema_name", "other"),
            )
            with self.assertRaisesRegex(PolicyContractError, "schema_name"):
                PolicyBundle.load(model_path)

    def test_rejects_is_rnn_different_from_onnx_graph(self) -> None:
        metadata = build_deployment_onnx_metadata(
            _fake_env(),
            observation_group_names=("actor",),
        )
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "stale_hash.onnx"
            _write_test_model(model_path, 34, recurrent=False)
            embed_onnx_metadata(
                model_path,
                metadata,
                policy_iteration=7,
                is_rnn=False,
            )
            _rewrite_descriptor(
                model_path,
                lambda payload: payload["meta"]["training"].__setitem__("is_rnn", True),
            )
            with self.assertRaisesRegex(PolicyContractError, "is_rnn"):
                PolicyBundle.load(model_path)

    def test_rejects_mixed_history_lengths(self) -> None:
        metadata = build_deployment_onnx_metadata(
            _fake_env(history_length=5),
            observation_group_names=("actor",),
        )
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "history_layout.onnx"
            _write_test_model(model_path, 170, recurrent=False)
            embed_onnx_metadata(
                model_path,
                metadata,
                policy_iteration=8,
                is_rnn=False,
            )
            _rewrite_descriptor(
                model_path,
                lambda payload: payload["policy_io"]["groups"]["actor"]["terms"][0].__setitem__(
                    "history_length", 4
                ),
            )
            with self.assertRaisesRegex(PolicyContractError, "混合 history_length"):
                PolicyBundle.load(model_path)

    def test_rejects_runtime_graph_shape_different_from_metadata(self) -> None:
        metadata = build_deployment_onnx_metadata(
            _fake_env(),
            observation_group_names=("actor",),
        )
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "runtime_shape.onnx"
            _write_test_model(model_path, 34, recurrent=False)
            embed_onnx_metadata(
                model_path,
                metadata,
                policy_iteration=9,
                is_rnn=False,
            )
            model = onnx.load(model_path)
            model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 35
            onnx.checker.check_model(model)
            onnx.save(model, model_path)

            with self.assertRaisesRegex(PolicyContractError, "ONNX obs"):
                PolicyBundle.load(model_path)


class PolicyRuntimeTests(unittest.TestCase):
    def test_metadata_observation_matches_shared_single_frame_math(self) -> None:
        metadata = build_deployment_onnx_metadata(
            _fake_env(),
            observation_group_names=("actor",),
        )
        policy_input = _policy_input()
        previous_action = np.asarray((0.2, -0.3, 0.4, -0.5, 0.6, -0.7), dtype=np.float32)
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "observation.onnx"
            _write_test_model(model_path, 34, recurrent=False)
            embed_onnx_metadata(
                model_path,
                metadata,
                policy_iteration=10,
                is_rnn=False,
            )
            contract = PolicyBundle.load(model_path).contract

        actual = PolicyObservationBuilder(contract).build(policy_input, previous_action)
        expected = build_policy_observation_np(
            base_ang_vel_body=np.asarray(policy_input.robot.base_angular_velocity_body),
            projected_gravity=np.asarray(policy_input.robot.projected_gravity_body),
            dof_pos=np.asarray(policy_input.robot.policy_joint_position),
            dof_vel=np.asarray(policy_input.robot.policy_joint_velocity),
            command=np.asarray(policy_input.commands["velocity_height"]),
            action_obs=previous_action,
            default_dof_pos=np.asarray(RobotConfig().default_dof_pos),
        )

        self.assertEqual(actual.tensor.shape, (1, 34))
        self.assertFalse(actual.had_nonfinite_input)
        np.testing.assert_allclose(actual.tensor[0], expected.obs, rtol=0.0, atol=1.0e-7)

    def test_history_mlp_is_term_major_and_reset_repeats_first_sample(self) -> None:
        metadata = build_deployment_onnx_metadata(
            _fake_env(history_length=5),
            observation_group_names=("actor",),
        )
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "history_runtime.onnx"
            _write_runtime_model(
                model_path,
                170,
                [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
                recurrent=False,
            )
            embed_onnx_metadata(
                model_path,
                metadata,
                policy_iteration=11,
                is_rnn=False,
            )
            runtime = PolicyRuntime.load(model_path)

            first_frame = _policy_input(base_ang_vel=(4.0, 8.0, 12.0))
            second_frame = _policy_input(base_ang_vel=(8.0, 12.0, 16.0))
            first = runtime.infer(first_frame)
            second = runtime.infer(second_frame)

            np.testing.assert_allclose(
                first.observation[0, :15],
                np.tile((1.0, 2.0, 3.0), 5),
            )
            np.testing.assert_allclose(
                second.observation[0, :15],
                np.asarray((1.0, 2.0, 3.0) * 4 + (2.0, 3.0, 4.0)),
            )
            self.assertAlmostEqual(second.observation[0, 15], 0.1)

            runtime.reset()
            after_reset = runtime.infer(second_frame)

        np.testing.assert_allclose(
            after_reset.observation[0, :15],
            np.tile((2.0, 3.0, 4.0), 5),
        )

    def test_raw_previous_action_and_physics_tick_delay_are_separate(self) -> None:
        metadata = build_deployment_onnx_metadata(
            _fake_env(),
            observation_group_names=("actor",),
        )
        raw_action = np.asarray((101.0, -101.0, 2.0, -2.0, 0.5, -0.5), dtype=np.float32)
        clipped_action = np.asarray((100.0, -100.0, 2.0, -2.0, 0.5, -0.5))
        policy_input = _policy_input()
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "delay_runtime.onnx"
            _write_runtime_model(
                model_path,
                34,
                raw_action.tolist(),
                recurrent=False,
            )
            embed_onnx_metadata(
                model_path,
                metadata,
                policy_iteration=12,
                is_rnn=False,
            )
            runtime = PolicyRuntime.load(model_path, action_delay_random_seed=1)
            first = runtime.infer(policy_input)
            first_tick = runtime.advance_physics_tick(commands=policy_input.commands)
            second_tick = runtime.advance_physics_tick(commands=policy_input.commands)
            second = runtime.infer(policy_input)

            self.assertEqual(runtime.action_pipeline.delay_steps, 1)
            np.testing.assert_array_equal(first.observation[0, 25:31], np.zeros(6))
            np.testing.assert_array_equal(runtime.previous_action, raw_action)
            np.testing.assert_array_equal(first_tick.clipped_action, np.zeros(6))
            np.testing.assert_array_equal(second_tick.clipped_action, clipped_action)
            np.testing.assert_array_equal(second.observation[0, 25:31], clipped_action)

            runtime.reset()
            after_reset = runtime.infer(policy_input)

        np.testing.assert_array_equal(after_reset.observation[0, 25:31], np.zeros(6))

    def test_metadata_action_decoder_matches_shared_height_conditioned_math(self) -> None:
        metadata = build_deployment_onnx_metadata(
            _fake_env(),
            observation_group_names=("actor",),
        )
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "action_decoder.onnx"
            _write_test_model(model_path, 34, recurrent=False)
            embed_onnx_metadata(
                model_path,
                metadata,
                policy_iteration=13,
                is_rnn=False,
            )
            contract = PolicyBundle.load(model_path).contract

        action = np.asarray((0.4, -0.8, -0.3, 0.7, 0.25, -0.35))
        commands = _policy_input().commands
        actual = PolicyActionDecoder(contract).decode(action, commands=commands)
        expected = SharedPolicyActionDecoder(
            action_scale=np.asarray(contract.action.scale),
            height_conditioned_action_default=True,
            active_rod_target_lower_preload_margin=contract.action.lower_target_overdrive,
        ).decode(
            action,
            command_height=float(np.asarray(commands["velocity_height"])[4]),
        )

        np.testing.assert_allclose(actual.clipped_action, expected.clipped_action, atol=1.0e-12)
        np.testing.assert_allclose(actual.policy_leg_default, expected.policy_default, atol=1.0e-12)
        np.testing.assert_allclose(
            actual.leg_position_target,
            expected.leg_target,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            actual.wheel_velocity_target,
            expected.wheel_vel_target,
            atol=1.0e-12,
        )

    def test_gru_hidden_state_advances_and_resets_to_zero(self) -> None:
        metadata = build_deployment_onnx_metadata(
            _fake_env(),
            observation_group_names=("actor",),
        )
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "gru_runtime.onnx"
            _write_runtime_model(
                model_path,
                34,
                [0.0] * 6,
                recurrent=True,
            )
            embed_onnx_metadata(
                model_path,
                metadata,
                policy_iteration=14,
                is_rnn=True,
            )
            runtime = PolicyRuntime.load(model_path)

            np.testing.assert_array_equal(runtime.hidden_state, np.zeros((1, 1, 512)))
            first = runtime.infer(_policy_input())
            np.testing.assert_array_equal(runtime.hidden_state, np.ones((1, 1, 512)))
            runtime.infer(_policy_input())
            np.testing.assert_array_equal(runtime.hidden_state, np.full((1, 1, 512), 2.0))

            runtime.reset()
            np.testing.assert_array_equal(runtime.hidden_state, np.zeros((1, 1, 512)))
            np.testing.assert_array_equal(runtime.previous_action, np.zeros(6))
            after_reset = runtime.infer(_policy_input())

        self.assertEqual(first.observation.shape, (1, 34))
        np.testing.assert_array_equal(after_reset.observation[0, 25:31], np.zeros(6))

    def test_nonfinite_observation_is_reported_and_sanitized_from_metadata(self) -> None:
        metadata = build_deployment_onnx_metadata(
            _fake_env(),
            observation_group_names=("actor",),
        )
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "nonfinite.onnx"
            _write_test_model(model_path, 34, recurrent=False)
            embed_onnx_metadata(
                model_path,
                metadata,
                policy_iteration=15,
                is_rnn=False,
            )
            contract = PolicyBundle.load(model_path).contract

        result = PolicyObservationBuilder(contract).build(
            _policy_input(base_ang_vel=(float("inf"), float("nan"), float("-inf"))),
            np.zeros(6),
        )

        self.assertTrue(result.had_nonfinite_input)
        np.testing.assert_array_equal(result.tensor[0, :3], np.asarray((100.0, 0.0, -100.0)))
        self.assertTrue(np.isfinite(result.tensor).all())


if __name__ == "__main__":
    unittest.main()
