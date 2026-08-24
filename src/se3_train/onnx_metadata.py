"""构建并嵌入 SE3 部署用 ONNX metadata。"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from copy import deepcopy
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import onnx

from se3_runtime.policy_contract import (
    CONTRACT_HASH_ALGORITHM,
    ONNX_METADATA_KEY,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    deployment_contract_hash,
)
from se3_shared import (
    JointGroup,
    ObservationConfig,
    RobotConfig,
    delay_seconds_to_steps,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROBOT_ASSET_PATH = Path(
    "assets/robots/serialleg/mjcf/serialleg_closed_chain_v3_train_obb_trim.xml"
)
_COMMAND_FIELDS = (
    ("lin_vel_x", "m/s"),
    ("ang_vel_yaw", "rad/s"),
    ("pitch", "rad"),
    ("roll", "rad"),
    ("height", "m"),
    ("jump_flag", "bool"),
    ("jump_target_height", "m"),
    ("jump_phase", "normalized"),
)
_ACTION_CHANNELS = (
    ("left_front_rod", "leg_front_angle_delta", "rad"),
    ("left_active_rod", "leg_active_angle_delta", "rad"),
    ("right_front_rod", "leg_front_angle_delta", "rad"),
    ("right_active_rod", "leg_active_angle_delta", "rad"),
    ("left_wheel", "wheel_velocity", "rad/s"),
    ("right_wheel", "wheel_velocity", "rad/s"),
)


def build_deployment_onnx_metadata(
    env: Any,
    *,
    observation_group_names: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """从 live env 构建部署契约，确保课程修改后的配置会进入导出物。"""
    runtime_env = getattr(env, "unwrapped", env)
    actor_group_order = [str(name) for name in observation_group_names]
    if not actor_group_order:
        raise ValueError("ONNX metadata 至少需要一个 actor observation group")

    physics_dt_s = float(runtime_env.physics_dt)
    policy_dt_s = float(runtime_env.step_dt)
    if physics_dt_s <= 0.0 or policy_dt_s <= 0.0:
        raise ValueError(f"仿真步长必须为正数，physics_dt={physics_dt_s}, policy_dt={policy_dt_s}")

    metadata: dict[str, Any] = {
        "meta": {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "metadata_key": ONNX_METADATA_KEY,
            "contract_hash_algorithm": CONTRACT_HASH_ALGORITHM,
            "producer": {
                "name": "se3_train",
                "package_version": _package_version(),
            },
            "training": {
                **_git_metadata(),
                "scene_num_envs": int(getattr(runtime_env, "num_envs", 0)),
            },
            "sim": {
                "physics_dt_s": physics_dt_s,
                "control_decimation": int(runtime_env.cfg.decimation),
                "policy_dt_s": policy_dt_s,
                "inference_hz": 1.0 / policy_dt_s,
                "mujoco": _build_mujoco_sim_metadata(runtime_env),
            },
        },
        "robot": _build_robot_metadata(runtime_env),
        "commands": _build_command_metadata(runtime_env.command_manager),
        "policy_io": {
            **_build_observation_metadata(
                runtime_env.observation_manager,
                actor_group_order,
            ),
            "action": _build_action_metadata(runtime_env),
        },
    }
    _refresh_contract_hash(metadata)
    return metadata


def embed_onnx_metadata(
    onnx_path: str | Path,
    metadata: dict[str, Any],
    *,
    policy_iteration: int,
    policy_type: str,
    source_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """把完整部署契约写入 ONNX，同时保留模型已有的 metadata 属性。"""
    path = Path(onnx_path)
    model = onnx.load(path)
    payload = deepcopy(metadata)
    normalized_policy_type = policy_type.lower()
    is_recurrent = normalized_policy_type in {"gru", "lstm", "rnn"}

    exported_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    training = payload.setdefault("meta", {}).setdefault("training", {})
    training.update(
        {
            "policy_iteration": int(policy_iteration),
            "policy_type": normalized_policy_type,
            "is_recurrent": is_recurrent,
            "exported_at": exported_at,
        }
    )
    if source_checkpoint is not None:
        training["source_checkpoint"] = Path(source_checkpoint).name

    policy_io = payload.setdefault("policy_io", {})
    policy_io["policy"] = _build_policy_metadata(model, normalized_policy_type)
    policy_io["onnx"] = _build_onnx_graph_metadata(model)
    _validate_graph_contract(payload)
    _refresh_contract_hash(payload)

    metadata_map = {prop.key: prop.value for prop in model.metadata_props}
    metadata_map[ONNX_METADATA_KEY] = _canonical_json(payload)
    del model.metadata_props[:]
    for key, value in metadata_map.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value

    onnx.checker.check_model(model)
    onnx.save(model, path)
    return payload


def _build_robot_metadata(runtime_env: Any) -> dict[str, Any]:
    robot = RobotConfig()
    asset_path = _REPO_ROOT / _ROBOT_ASSET_PATH
    return {
        "name": "SerialLeg",
        "topology": "closed_chain_four_bar",
        "asset": {
            "format": "mjcf",
            "path": _ROBOT_ASSET_PATH.as_posix(),
            "sha256": _file_sha256(asset_path),
        },
        "base_link": "base_link",
        "policy_joint_names": list(JointGroup.POLICY_JOINT_NAMES),
        "policy_joint_order": "left_leg_then_right_leg_then_wheels",
        "default_joint_position": list(robot.default_dof_pos),
        "frames": {
            "quaternion_order": "wxyz",
            "base_angular_velocity": "body",
            "projected_gravity": "body",
            "world_up_axis": "+z",
        },
        "actuators": _build_actuator_metadata(runtime_env),
    }


def _build_mujoco_sim_metadata(runtime_env: Any) -> dict[str, Any]:
    try:
        cfg = runtime_env.cfg.sim.mujoco
    except AttributeError as error:
        raise TypeError("live env 缺少 cfg.sim.mujoco，无法导出求解器契约") from error
    return {
        "integrator": str(cfg.integrator).lower(),
        "impratio": float(cfg.impratio),
        "cone": str(cfg.cone).lower(),
        "jacobian": str(cfg.jacobian).lower(),
        "solver": str(cfg.solver).lower(),
        "iterations": int(cfg.iterations),
        "tolerance": float(cfg.tolerance),
        "ls_iterations": int(cfg.ls_iterations),
        "ls_tolerance": float(cfg.ls_tolerance),
        "ccd_iterations": int(cfg.ccd_iterations),
        "gravity": [float(value) for value in cfg.gravity],
        "disableflags": [str(value).lower() for value in cfg.disableflags],
        "enableflags": [str(value).lower() for value in cfg.enableflags],
    }


def _build_actuator_metadata(runtime_env: Any) -> dict[str, Any]:
    try:
        robot_entity = runtime_env.scene["robot"]
        live_actuators = robot_entity.actuators
    except (AttributeError, KeyError, TypeError) as error:
        raise TypeError("live env 缺少 scene['robot'].actuators，无法导出 actuator 契约") from error

    by_joint: dict[str, dict[str, Any]] = {}
    policy_joint_names = set(JointGroup.POLICY_JOINT_NAMES)
    for actuator in live_actuators:
        cfg = actuator.cfg
        for joint_name in actuator.target_names:
            if joint_name not in policy_joint_names:
                continue
            if joint_name in by_joint:
                raise ValueError(f"policy joint {joint_name!r} 被多个 actuator 控制")
            curve_value = getattr(cfg, "torque_speed_curve", None)
            if curve_value:
                curve = [[float(speed), float(effort)] for speed, effort in curve_value]
                peak_effort = max(point[1] for point in curve)
                velocity_limit = curve[-1][0]
            else:
                if not hasattr(cfg, "saturation_effort") or not hasattr(cfg, "velocity_limit"):
                    raise TypeError(f"policy joint {joint_name!r} 的 actuator 缺少 T-N 包络参数")
                curve = None
                peak_effort = float(cfg.saturation_effort)
                velocity_limit = float(cfg.velocity_limit)
            by_joint[joint_name] = {
                "stiffness": float(cfg.stiffness),
                "damping": float(cfg.damping),
                "peak_effort_limit": peak_effort,
                "rated_effort_limit": float(cfg.effort_limit),
                "velocity_limit": velocity_limit,
                "torque_speed_curve": curve,
            }

    missing = [name for name in JointGroup.POLICY_JOINT_NAMES if name not in by_joint]
    if missing:
        raise ValueError(f"live robot actuator 未覆盖全部 policy joints：{missing}")
    fields = (
        "stiffness",
        "damping",
        "peak_effort_limit",
        "rated_effort_limit",
        "velocity_limit",
        "torque_speed_curve",
    )
    return {
        field: [by_joint[joint_name][field] for joint_name in JointGroup.POLICY_JOINT_NAMES]
        for field in fields
    }


def _build_command_metadata(command_manager: Any) -> dict[str, Any]:
    commands: dict[str, Any] = {}
    for command_name in command_manager.active_terms:
        term = command_manager.get_term(command_name)
        cfg = term.cfg
        command_dim = int(term.command.shape[-1])
        if command_name != "velocity_height" or command_dim not in {5, 8}:
            raise ValueError(f"暂不支持 command {command_name!r} 的 {command_dim}D metadata 契约")

        ranges = _command_ranges(cfg, command_dim)
        fields = [
            {
                "index": index,
                "name": name,
                "unit": unit,
                "range": ranges[name],
            }
            for index, (name, unit) in enumerate(_COMMAND_FIELDS[:command_dim])
        ]
        commands[command_name] = {
            "type": type(term).__name__,
            "dimension": command_dim,
            "fields": fields,
            "ranges": ranges,
            "sampling": {
                "resampling_time_s": _float_pair(cfg.resampling_time_range),
                "standing_ratio": float(getattr(cfg, "standing_ratio", 0.0)),
                "standing_height_range": _float_pair(
                    getattr(cfg, "standing_height_range", cfg.height_range)
                ),
                "lin_vel_deadband": float(getattr(cfg, "lin_vel_deadband", 0.0)),
                "yaw_deadband": float(getattr(cfg, "yaw_deadband", 0.0)),
            },
            "differential_drive_constraint": {
                "enabled": bool(getattr(cfg, "constrain_diff_drive_commands", False)),
                "wheel_radius_m": float(getattr(cfg, "diff_drive_wheel_radius", 0.0)),
                "half_track_m": float(getattr(cfg, "diff_drive_half_track", 0.0)),
                "max_wheel_speed_rad_s": float(getattr(cfg, "diff_drive_max_wheel_speed", 0.0)),
                "wheel_speed_fraction": float(getattr(cfg, "diff_drive_wheel_speed_fraction", 1.0)),
            },
        }
        if command_dim == 8:
            commands[command_name]["jump"] = {
                "start_probability_per_policy_step": float(getattr(cfg, "jump_prob", 0.0)),
                "cool_down_policy_steps": int(getattr(cfg, "jump_cool_down_steps", 0)),
                "height_range": _float_pair(cfg.jump_height_range),
            }
    return commands


def _command_ranges(cfg: Any, command_dim: int) -> dict[str, list[float]]:
    ranges = {
        "lin_vel_x": _float_pair(cfg.lin_vel_x_range),
        "ang_vel_yaw": _float_pair(cfg.ang_vel_yaw_range),
        "pitch": _float_pair(cfg.pitch_range),
        "roll": _float_pair(cfg.roll_range),
        "height": _float_pair(cfg.height_range),
    }
    if command_dim == 8:
        ranges.update(
            {
                "jump_flag": [0.0, 1.0],
                "jump_target_height": _float_pair(cfg.jump_height_range),
                "jump_phase": [0.0, 1.0],
            }
        )
    return ranges


def _build_observation_metadata(
    observation_manager: Any,
    actor_group_order: list[str],
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    total_frame_dim = 0
    total_input_dim = 0
    for group_name in actor_group_order:
        if group_name not in observation_manager.active_terms:
            raise ValueError(f"actor observation group {group_name!r} 不存在")
        if not observation_manager.group_obs_concatenate[group_name]:
            raise ValueError(f"actor observation group {group_name!r} 必须 concatenate_terms=True")

        term_names = observation_manager.active_terms[group_name]
        flattened_dims = observation_manager.group_obs_term_dim[group_name]
        terms: list[dict[str, Any]] = []
        history_lengths: list[int] = []
        group_frame_dim = 0
        group_input_dim = 0
        for term_name, flattened_shape in zip(term_names, flattened_dims, strict=True):
            term_cfg = observation_manager.get_term_cfg(group_name, term_name)
            history_length = max(1, int(getattr(term_cfg, "history_length", 0)))
            flattened_width = int(math.prod(flattened_shape))
            if flattened_width % history_length != 0:
                raise ValueError(
                    f"{group_name}.{term_name} 的展平宽度 {flattened_width} "
                    f"不能被 history_length={history_length} 整除"
                )
            if history_length > 1 and not bool(term_cfg.flatten_history_dim):
                raise ValueError(f"{group_name}.{term_name} 的 history 必须展平后才能导出")
            base_width = flattened_width // history_length
            term = _build_observation_term_metadata(
                term_name,
                term_cfg,
                base_width=base_width,
                history_length=history_length,
                flattened_width=flattened_width,
            )
            terms.append(term)
            history_lengths.append(history_length)
            group_frame_dim += base_width
            group_input_dim += flattened_width

        common_history_length = (
            history_lengths[0] if history_lengths and len(set(history_lengths)) == 1 else None
        )
        groups[group_name] = {
            "frame_dim": group_frame_dim,
            "input_dim": group_input_dim,
            "concatenate_terms": True,
            "concatenate_dim": int(observation_manager.cfg[group_name].concatenate_dim),
            "history": {
                "mode": (
                    "per_term_circular_buffer"
                    if any(length > 1 for length in history_lengths)
                    else "none"
                ),
                "length": common_history_length,
                "order": "oldest_to_newest",
                "flatten_layout": "term_major",
                "reset_fill": "repeat_first_sample",
                "update_period_policy_steps": 1,
            },
            "terms": terms,
        }
        total_frame_dim += group_frame_dim
        total_input_dim += group_input_dim

    return {
        "observation": {
            "contract": "serialleg_actor_observation.v1",
            "frame_dim": total_frame_dim,
            "input_dim": total_input_dim,
            "actor_group_order": actor_group_order,
            "group_flatten_layout": "group_major",
            "nonfinite_policy": "nan_to_zero_and_infinity_to_clip",
            "deployment_corruption": "disabled",
        },
        "groups": groups,
    }


def _build_observation_term_metadata(
    term_name: str,
    term_cfg: Any,
    *,
    base_width: int,
    history_length: int,
    flattened_width: int,
) -> dict[str, Any]:
    semantics = _observation_term_semantics(term_name)
    expected_width = int(semantics["width"])
    if base_width != expected_width:
        raise ValueError(f"观测项 {term_name!r} 宽度应为 {expected_width}，实际为 {base_width}")

    scale = _multiply_scale(
        semantics["scale"],
        getattr(term_cfg, "scale", None),
        base_width,
    )
    clip = _merge_clip(semantics["clip"], getattr(term_cfg, "clip", None))
    term: dict[str, Any] = {
        "name": term_name,
        "base_width": base_width,
        "history_length": history_length,
        "flattened_width": flattened_width,
        "source": semantics["source"],
        "transform": semantics["transform"],
        "scale": scale,
        "clip": clip,
    }
    for key in ("fields", "command_ref", "action_stage", "temporal_offset_policy_steps"):
        if key in semantics:
            term[key] = semantics[key]

    delay_max_lag = int(getattr(term_cfg, "delay_max_lag", 0))
    if delay_max_lag > 0:
        term["delay"] = {
            "min_lag_policy_steps": int(getattr(term_cfg, "delay_min_lag", 0)),
            "max_lag_policy_steps": delay_max_lag,
            "per_environment": bool(getattr(term_cfg, "delay_per_env", True)),
            "hold_probability": float(getattr(term_cfg, "delay_hold_prob", 0.0)),
            "resample_period_policy_steps": int(getattr(term_cfg, "delay_update_period", 0)),
        }
    return term


def _observation_term_semantics(term_name: str) -> dict[str, Any]:
    cfg = ObservationConfig()
    limit = float(cfg.clip_value)
    finite_clip = [-limit, limit]
    semantics: dict[str, dict[str, Any]] = {
        "base_ang_vel": {
            "width": 3,
            "source": "robot.base_angular_velocity_body",
            "transform": "identity",
            "scale": [cfg.ang_vel_scale] * 3,
            "clip": finite_clip,
            "fields": ["x", "y", "z"],
        },
        "projected_gravity": {
            "width": 3,
            "source": "robot.projected_gravity_body",
            "transform": "identity",
            "scale": [1.0] * 3,
            "clip": [-1.0, 1.0],
            "fields": ["x", "y", "z"],
        },
        "commands": {
            "width": 5,
            "source": "commands.velocity_height[0:5]",
            "transform": "identity",
            "scale": list(cfg.command_scale),
            "clip": finite_clip,
            "fields": [name for name, _ in _COMMAND_FIELDS[:5]],
            "command_ref": "commands.velocity_height",
        },
        "leg_joint_pos": {
            "width": 6,
            "source": "robot.policy_leg_joint_position",
            "transform": "serialleg_leg_phase_active.v1",
            "scale": [1.0] * 6,
            "clip": finite_clip,
            "fields": [
                "sin_left_front",
                "cos_left_front",
                "left_active_rod_angle",
                "sin_right_front",
                "cos_right_front",
                "right_active_rod_angle",
            ],
        },
        "leg_joint_vel": {
            "width": 4,
            "source": "robot.policy_leg_joint_velocity",
            "transform": "identity",
            "scale": [cfg.leg_vel_scale] * 4,
            "clip": finite_clip,
            "fields": list(JointGroup.POLICY_LEG_NAMES),
        },
        "wheel_pos_zero": {
            "width": 2,
            "source": "constant.zero",
            "transform": "constant_zero",
            "scale": [1.0] * 2,
            "clip": None,
            "fields": list(JointGroup.WHEEL_NAMES),
        },
        "wheel_vel": {
            "width": 2,
            "source": "robot.policy_wheel_joint_velocity",
            "transform": "identity",
            "scale": [cfg.wheel_vel_scale] * 2,
            "clip": finite_clip,
            "fields": list(JointGroup.WHEEL_NAMES),
        },
        "last_actions": {
            "width": 6,
            "source": "policy.previous_action",
            "transform": "identity",
            "scale": [1.0] * 6,
            "clip": finite_clip,
            "fields": [name for name, _, _ in _ACTION_CHANNELS],
            "action_stage": "raw_policy_output_before_action_term_processing",
            "temporal_offset_policy_steps": -1,
        },
        "jump_commands": {
            "width": 3,
            "source": "commands.velocity_height[5:8]",
            "transform": "identity",
            "scale": [1.0] * 3,
            "clip": finite_clip,
            "fields": [name for name, _ in _COMMAND_FIELDS[5:8]],
            "command_ref": "commands.velocity_height",
        },
    }
    try:
        return semantics[term_name]
    except KeyError as error:
        raise ValueError(f"暂不支持观测项 {term_name!r} 的 ONNX metadata 契约") from error


def _build_action_metadata(runtime_env: Any) -> dict[str, Any]:
    manager = runtime_env.action_manager
    if manager.active_terms != ["delayed_action"] or int(manager.total_action_dim) != 6:
        raise ValueError(
            "SerialLeg ONNX metadata 要求唯一的 6D delayed_action，"
            f"实际为 {manager.active_terms}, dim={manager.total_action_dim}"
        )
    cfg = manager.get_term("delayed_action").cfg
    required = (
        "leg_scales",
        "wheel_scale",
        "action_delay_enabled",
        "action_delay_s",
        "action_delay_randomize",
        "action_delay_min_s",
        "action_delay_max_s",
        "active_rod_lower_target_overdrive",
    )
    missing = [name for name in required if not hasattr(cfg, name)]
    if missing:
        raise TypeError(f"delayed_action 不是 SerialLeg 动作项，缺少字段: {missing}")

    robot = RobotConfig()
    scale = [float(value) for value in cfg.leg_scales] + [float(cfg.wheel_scale)] * 2
    if len(scale) != 6:
        raise ValueError(f"SerialLeg action scale 必须为 6D，实际为 {scale}")
    delay_min_steps = delay_seconds_to_steps(float(cfg.action_delay_min_s), runtime_env.physics_dt)
    delay_max_steps = delay_seconds_to_steps(float(cfg.action_delay_max_s), runtime_env.physics_dt)
    if not bool(cfg.action_delay_enabled):
        delay_min_steps = delay_max_steps = 0
    elif not bool(cfg.action_delay_randomize):
        nominal_steps = delay_seconds_to_steps(float(cfg.action_delay_s), runtime_env.physics_dt)
        delay_min_steps = delay_max_steps = nominal_steps

    height_conditioned = bool(getattr(cfg, "height_conditioned_action_default", False))
    default_strategy: dict[str, Any]
    if height_conditioned:
        command_name = str(getattr(cfg, "action_default_command_name", "velocity_height"))
        default_strategy = {
            "mode": "serialleg_height_conditioned_policy_default.v1",
            "command_ref": f"commands.{command_name}",
            "height_field": "height",
            "fallback_policy_leg_position": list(robot.default_dof_pos[:4]),
        }
    else:
        default_strategy = {
            "mode": "entity_default_joint_position",
            "policy_leg_position": list(robot.default_dof_pos[:4]),
        }

    return {
        "name": "delayed_action",
        "decoder": "serialleg_active_rod.v1",
        "dimension": 6,
        "channels": [
            {
                "index": index,
                "name": name,
                "semantic": semantic,
                "target_unit": unit,
                "scale": scale[index],
            }
            for index, (name, semantic, unit) in enumerate(_ACTION_CHANNELS)
        ],
        "scale": scale,
        "raw_clip": (None if getattr(cfg, "action_clip", None) is None else float(cfg.action_clip)),
        "default_strategy": default_strategy,
        "active_rod_angle_coeffs": [list(values) for values in robot.active_rod_angle_coeffs],
        "active_rod_angle_limits": list(robot.active_rod_angle_limits),
        "active_rod_angle_midpoint": sum(robot.active_rod_angle_limits) / 2.0,
        "lower_target_overdrive": float(cfg.active_rod_lower_target_overdrive),
        "active_rod_target_limits": [
            float(robot.active_rod_angle_limits[0]) - float(cfg.active_rod_lower_target_overdrive),
            float(robot.active_rod_angle_limits[1]),
        ],
        "wheel_default_velocity": [0.0, 0.0],
        "delay": {
            "enabled": bool(cfg.action_delay_enabled),
            "stage": "raw_action_before_decode",
            "nominal_s": float(cfg.action_delay_s),
            "randomized": bool(cfg.action_delay_randomize),
            "range_s": [float(cfg.action_delay_min_s), float(cfg.action_delay_max_s)],
            "step_bounds": [delay_min_steps, delay_max_steps],
            "resample": str(getattr(cfg, "action_delay_resample", "reset")),
        },
    }


def _build_policy_metadata(model: onnx.ModelProto, policy_type: str) -> dict[str, Any]:
    recurrent = policy_type in {"gru", "lstm", "rnn"}
    metadata: dict[str, Any] = {
        "type": policy_type,
        "is_recurrent": recurrent,
    }
    if recurrent:
        input_names = [value.name for value in model.graph.input]
        output_names = [value.name for value in model.graph.output]
        metadata["state"] = {
            "input_names": [name for name in input_names if name != "obs"],
            "output_names": [name for name in output_names if name != "actions"],
            "reset": "zeros_on_episode_reset",
        }
    return metadata


def _build_onnx_graph_metadata(model: onnx.ModelProto) -> dict[str, Any]:
    return {
        "ir_version": int(model.ir_version),
        "opsets": [
            {
                "domain": opset.domain or "ai.onnx",
                "version": int(opset.version),
            }
            for opset in model.opset_import
        ],
        "inputs": [_value_info_metadata(value) for value in model.graph.input],
        "outputs": [_value_info_metadata(value) for value in model.graph.output],
    }


def _value_info_metadata(value: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor_type = value.type.tensor_type
    shape: list[int | str | None] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        elif dim.dim_param:
            shape.append(str(dim.dim_param))
        else:
            shape.append(None)
    return {
        "name": value.name,
        "dtype": onnx.TensorProto.DataType.Name(tensor_type.elem_type).lower(),
        "shape": shape,
    }


def _validate_graph_contract(metadata: dict[str, Any]) -> None:
    policy_io = metadata["policy_io"]
    graph = policy_io["onnx"]
    inputs = {entry["name"]: entry for entry in graph["inputs"]}
    outputs = {entry["name"]: entry for entry in graph["outputs"]}
    if "obs" not in inputs or "actions" not in outputs:
        raise ValueError(
            "ONNX 图必须包含 obs 输入和 actions 输出，"
            f"实际 inputs={list(inputs)}, outputs={list(outputs)}"
        )

    expected_obs_dim = int(policy_io["observation"]["input_dim"])
    graph_obs_dim = _static_last_dim(inputs["obs"], "obs")
    if graph_obs_dim != expected_obs_dim:
        raise ValueError(
            f"ONNX obs 输入为 {graph_obs_dim}D，但 metadata observation 为 {expected_obs_dim}D"
        )

    expected_action_dim = int(policy_io["action"]["dimension"])
    graph_action_dim = _static_last_dim(outputs["actions"], "actions")
    if graph_action_dim != expected_action_dim:
        raise ValueError(
            f"ONNX actions 输出为 {graph_action_dim}D，但 metadata action 为 {expected_action_dim}D"
        )

    policy_type = str(policy_io["policy"]["type"])
    if policy_type == "gru" and not ({"h_in"} <= inputs.keys() and {"h_out"} <= outputs.keys()):
        raise ValueError("GRU ONNX 图必须包含 h_in 输入和 h_out 输出")
    if policy_type == "lstm" and not (
        {"h_in", "c_in"} <= inputs.keys() and {"h_out", "c_out"} <= outputs.keys()
    ):
        raise ValueError("LSTM ONNX 图必须包含 h/c 输入和输出")
    if policy_type == "mlp" and (set(inputs) != {"obs"} or set(outputs) != {"actions"}):
        raise ValueError("MLP ONNX 图只应包含 obs 输入和 actions 输出")


def _static_last_dim(entry: dict[str, Any], tensor_name: str) -> int:
    shape = entry.get("shape") or []
    if not shape or not isinstance(shape[-1], int):
        raise ValueError(f"ONNX tensor {tensor_name!r} 的最后一维必须是静态整数，实际为 {shape}")
    return int(shape[-1])


def _multiply_scale(
    internal_scale: list[float],
    manager_scale: Any,
    width: int,
) -> list[float]:
    if manager_scale is None:
        external = [1.0] * width
    elif isinstance(manager_scale, (int, float)):
        external = [float(manager_scale)] * width
    else:
        external = [float(value) for value in manager_scale]
        if len(external) != width:
            raise ValueError(f"观测 scale 应为 {width}D，实际为 {external}")
    return [
        float(base) * multiplier for base, multiplier in zip(internal_scale, external, strict=True)
    ]


def _merge_clip(internal_clip: list[float] | None, manager_clip: Any) -> list[float] | None:
    if manager_clip is None:
        return None if internal_clip is None else list(internal_clip)
    external = [float(value) for value in manager_clip]
    if len(external) != 2:
        raise ValueError(f"观测 clip 必须为 [min, max]，实际为 {external}")
    if internal_clip is None:
        return external
    return [max(float(internal_clip[0]), external[0]), min(float(internal_clip[1]), external[1])]


def _refresh_contract_hash(metadata: dict[str, Any]) -> None:
    metadata.setdefault("meta", {})["contract_hash"] = deployment_contract_hash(metadata)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _float_pair(values: Any) -> list[float]:
    resolved = [float(value) for value in values]
    if len(resolved) != 2:
        raise ValueError(f"范围必须包含两个值，实际为 {resolved}")
    return resolved


def _package_version() -> str | None:
    try:
        return version("se3-wheel-leg")
    except PackageNotFoundError:
        return None


def _git_metadata() -> dict[str, Any]:
    commit = _git_output("rev-parse", "HEAD")
    status = _git_output("status", "--porcelain")
    return {
        "repo_commit": commit,
        "repo_commit_short": commit[:12] if commit else None,
        "repo_dirty": bool(status),
    }


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CONTRACT_HASH_ALGORITHM",
    "ONNX_METADATA_KEY",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "build_deployment_onnx_metadata",
    "deployment_contract_hash",
    "embed_onnx_metadata",
]
