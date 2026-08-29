"""按 Kyber descriptor 结构构建并嵌入 SE3 ONNX metadata。"""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import onnx

from se3_shared import (
    DM8009P,
    M3508_C620_14,
    JointGroup,
    ObservationConfig,
    RobotConfig,
)

ONNX_METADATA_KEY = "se3.meta.v2"
SCHEMA_NAME = "se3.deployment"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ASSET_NAME = "serialleg_closed_chain_v3_train_obb_trim"
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
_COMMAND_FIELD_NAMES = (
    "lin_vel_x",
    "ang_vel_yaw",
    "pitch",
    "roll",
    "height",
    "jump_flag",
    "jump_target_height",
    "jump_phase",
)


def build_deployment_onnx_metadata(
    env: Any,
    *,
    observation_group_names: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """从 live env 构建与 Kyber 同构的部署 descriptor。"""
    runtime_env = getattr(env, "unwrapped", env)
    source_groups = _resolve_observation_groups(
        runtime_env.observation_manager,
        observation_group_names,
    )
    actuator_metadata = _build_actuator_metadata(runtime_env)
    robot = RobotConfig()
    sim_dt = float(runtime_env.physics_dt)
    step_dt = float(runtime_env.step_dt)
    decimation = int(runtime_env.cfg.decimation)
    if sim_dt <= 0.0 or step_dt <= 0.0:
        raise ValueError(f"仿真步长必须为正数，sim_dt={sim_dt}, step_dt={step_dt}")
    if not math.isclose(sim_dt * decimation, step_dt, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise ValueError("step_dt 必须等于 sim_dt × decimation")

    return {
        "meta": {
            "schema_name": SCHEMA_NAME,
            "assets": _build_assets_metadata(actuator_metadata),
            "training": _build_training_metadata(runtime_env),
            "env_groups": _build_env_group_metadata(runtime_env),
            "sim": {
                "sim_dt": sim_dt,
                "decimation": decimation,
                "step_dt": step_dt,
                "inference_hz": 1.0 / step_dt,
            },
        },
        "robot": {
            "name": "SerialLeg",
            "control_type": "closed_chain_four_bar",
            "policy_joint_names": list(JointGroup.POLICY_JOINT_NAMES),
            "KP": actuator_metadata["KP"],
            "KD": actuator_metadata["KD"],
            "armature": actuator_metadata["armature"],
            "saturation_effort": actuator_metadata["saturation_effort"],
            "effort_limit": actuator_metadata["effort_limit"],
            "velocity_limit": actuator_metadata["velocity_limit"],
            "torque_speed_curve": actuator_metadata["torque_speed_curve"],
            "knee_gas_spring": _build_knee_gas_spring_metadata(runtime_env),
            "init_state": _build_initial_state(runtime_env, robot),
            "root_link": "base_link",
            "imu_link": "base_link",
            "foot_link": ["l_wheel_Link", "r_wheel_Link"],
            "hand_link": [],
        },
        "commands": _build_command_metadata(runtime_env.command_manager),
        "policy_io": {
            "groups": _build_observation_groups(
                runtime_env.observation_manager,
                source_groups,
            ),
            "references": {
                "joint_pos_ref": list(robot.default_dof_pos),
                "joint_vel_ref": [0.0] * len(JointGroup.POLICY_JOINT_NAMES),
            },
            "action": _build_action_metadata(runtime_env),
        },
    }


def embed_onnx_metadata(
    onnx_path: str | Path,
    metadata: dict[str, Any] | None,
    *,
    policy_iteration: int | None = None,
    is_rnn: bool | None = None,
    metadata_key: str = ONNX_METADATA_KEY,
) -> dict[str, Any] | None:
    """复制 Kyber 导出层：补充训练信息并写入一个 JSON metadata 属性。"""
    if not metadata:
        return None

    path = Path(onnx_path)
    payload = deepcopy(metadata)
    if policy_iteration is not None and is_rnn is not None:
        training = payload.setdefault("meta", {}).setdefault("training", {})
        now = datetime.now().astimezone()
        export_stamp = now.strftime("%y-%m-%d-%H-%M-%S")
        training["policy_iteration"] = int(policy_iteration)
        training["is_rnn"] = bool(is_rnn)
        training["export_timestamp"] = now.replace(microsecond=0).isoformat()
        training["export_tag"] = f"{training.get('repo_commit_short') or 'unknown'}+{export_stamp}"

    model = onnx.load(path)
    _validate_onnx_graph(model, payload)
    metadata_map = {prop.key: prop.value for prop in model.metadata_props}
    metadata_map[metadata_key] = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
    )
    del model.metadata_props[:]
    for key, value in metadata_map.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value

    onnx.checker.check_model(model)
    onnx.save(model, path)
    return payload


def _build_training_metadata(runtime_env: Any) -> dict[str, Any]:
    commit = _git_output("rev-parse", "HEAD")
    return {
        "repo_commit": commit,
        "repo_commit_short": commit[:12] if commit else None,
        "scene_num_envs": int(getattr(runtime_env, "num_envs", 0) or 0),
        "policy_iteration": None,
        "is_rnn": None,
        "export_timestamp": None,
        "export_tag": None,
    }


def _build_assets_metadata(actuator_metadata: dict[str, Any]) -> dict[str, Any]:
    commit = _git_output("rev-parse", "HEAD")
    short_commit = commit[:12] if commit else None
    return {
        "repo": "se3_wheel_leg",
        "version": commit,
        "lookup_key": f"{_ASSET_NAME}@{short_commit}" if short_commit else _ASSET_NAME,
        "asset_name": _ASSET_NAME,
        "dirty": bool(_git_output("status", "--porcelain")),
        "robot_config_overridden": bool(actuator_metadata["robot_config_overridden"]),
    }


def _build_env_group_metadata(runtime_env: Any) -> dict[str, str]:
    group_names = getattr(runtime_env, "env_group_names", None)
    if not group_names:
        return {"0": "default"}
    return {str(index): str(name) for index, name in enumerate(group_names)}


def _build_initial_state(runtime_env: Any, robot: RobotConfig) -> dict[str, list[float]]:
    try:
        scene_cfg = runtime_env.cfg.scene
        robot_cfg = getattr(scene_cfg, "robot", None)
        if robot_cfg is None:
            robot_cfg = scene_cfg.entities["robot"]
        init_state = robot_cfg.init_state
    except (AttributeError, KeyError, TypeError):
        return {
            "base_pos": [0.0, 0.0, float(robot.default_base_height)],
            "base_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
    return {
        "base_pos": [float(value) for value in init_state.pos],
        "base_quat_wxyz": [float(value) for value in init_state.rot],
    }


def _build_actuator_metadata(runtime_env: Any) -> dict[str, Any]:
    try:
        robot_entity = runtime_env.scene["robot"]
        live_actuators = robot_entity.actuators
    except (AttributeError, KeyError, TypeError) as error:
        raise TypeError("live env 缺少 scene['robot'].actuators") from error
    compiled_armature = _build_compiled_policy_armature(runtime_env, robot_entity)

    by_joint: dict[str, dict[str, Any]] = {}
    armature_config_overridden = False
    policy_joint_names = set(JointGroup.POLICY_JOINT_NAMES)
    for actuator in live_actuators:
        cfg = actuator.cfg
        armature_config_overridden |= getattr(cfg, "armature", None) is not None
        for joint_name in actuator.target_names:
            if joint_name not in policy_joint_names:
                continue
            if joint_name in by_joint:
                raise ValueError(f"policy joint {joint_name!r} 被多个 actuator 控制")
            raw_curve = getattr(cfg, "torque_speed_curve", None)
            curve = (
                [[float(speed), float(effort)] for speed, effort in raw_curve]
                if raw_curve
                else None
            )
            velocity_limit = float(curve[-1][0]) if curve else float(cfg.velocity_limit)
            by_joint[joint_name] = {
                "KP": float(getattr(cfg, "stiffness", 0.0)),
                "KD": float(getattr(cfg, "damping", 0.0)),
                "armature": compiled_armature[joint_name],
                "saturation_effort": float(cfg.saturation_effort),
                "effort_limit": float(cfg.effort_limit),
                "velocity_limit": velocity_limit,
                "torque_speed_curve": curve,
            }

    missing = [name for name in JointGroup.POLICY_JOINT_NAMES if name not in by_joint]
    if missing:
        raise ValueError(f"live robot actuator 未覆盖全部 policy joints：{missing}")
    metadata = {
        key: [by_joint[name][key] for name in JointGroup.POLICY_JOINT_NAMES]
        for key in (
            "KP",
            "KD",
            "armature",
            "saturation_effort",
            "effort_limit",
            "velocity_limit",
            "torque_speed_curve",
        )
    }
    robot = RobotConfig()
    leg_curve = [list(point) for point in DM8009P.torque_speed_curve] or None
    wheel_curve = [list(point) for point in M3508_C620_14.torque_speed_curve] or None
    expected = {
        "KP": [robot.leg_kp] * 4 + [0.0, 0.0],
        "KD": [robot.leg_kd] * 4 + [robot.wheel_kd] * 2,
        "saturation_effort": [DM8009P.stall_torque] * 4 + [M3508_C620_14.stall_torque] * 2,
        "effort_limit": [DM8009P.rated_torque] * 4 + [M3508_C620_14.rated_torque] * 2,
        "velocity_limit": [DM8009P.no_load_speed] * 4 + [M3508_C620_14.no_load_speed] * 2,
        "torque_speed_curve": [leg_curve] * 4 + [wheel_curve] * 2,
    }
    metadata["robot_config_overridden"] = armature_config_overridden or any(
        metadata[key] != values for key, values in expected.items()
    )
    return metadata


def _build_compiled_policy_armature(
    runtime_env: Any,
    robot_entity: Any,
) -> dict[str, float]:
    """读取最终编译模型中的 policy joint armature。"""
    try:
        joint_names = list(robot_entity.joint_names)
        joint_v_addresses = robot_entity.indexing.joint_v_adr
        dof_armature = runtime_env.sim.mj_model.dof_armature
    except (AttributeError, TypeError) as error:
        raise TypeError("live env 缺少编译后的 robot joint armature") from error
    if len(joint_names) != len(joint_v_addresses):
        raise ValueError(
            "robot joint name 与 velocity address 数量不一致："
            f"names={len(joint_names)}, addresses={len(joint_v_addresses)}"
        )

    joint_name_to_address = {
        name: int(address.item() if hasattr(address, "item") else address)
        for name, address in zip(joint_names, joint_v_addresses, strict=True)
    }
    missing = [name for name in JointGroup.POLICY_JOINT_NAMES if name not in joint_name_to_address]
    if missing:
        raise ValueError(f"编译模型缺少 policy joint velocity address：{missing}")

    armature: dict[str, float] = {}
    for joint_name in JointGroup.POLICY_JOINT_NAMES:
        dof_address = joint_name_to_address[joint_name]
        try:
            value = float(dof_armature[dof_address])
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError(
                f"policy joint {joint_name!r} 的 armature 地址无效：{dof_address}"
            ) from error
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"policy joint {joint_name!r} 的 armature 必须为有限非负数，实际为 {value}"
            )
        armature[joint_name] = value
    return armature


def _build_knee_gas_spring_metadata(runtime_env: Any) -> dict[str, Any]:
    """导出膝关节恒力气弹簧与电机前馈补偿契约，供 sim2sim/sim2real 复现同一力矩。"""
    try:
        cfg = runtime_env.action_manager.get_term("delayed_action").cfg
    except (AttributeError, KeyError, TypeError) as error:
        raise TypeError("live env 缺少 delayed_action term，无法导出气弹簧契约") from error

    force = float(getattr(cfg, "knee_gas_spring_force", 0.0))
    compensation_enabled = bool(getattr(cfg, "knee_gas_spring_compensation_enabled", False))
    if force < 0.0:
        raise ValueError(f"knee_gas_spring_force 必须非负，实际为 {force}")
    if compensation_enabled and force <= 0.0:
        raise ValueError("启用气弹簧补偿时 knee_gas_spring_force 必须为正数")
    return {"force": force, "compensation_enabled": compensation_enabled}


def _build_command_metadata(command_manager: Any) -> dict[str, Any]:
    if list(command_manager.active_terms) != ["velocity_height"]:
        raise ValueError(
            "SE3 部署只支持唯一的 velocity_height command，"
            f"实际为 {list(command_manager.active_terms)!r}"
        )
    term = command_manager.get_term("velocity_height")
    command_dim = int(term.command.shape[-1])
    if command_dim not in {5, 8}:
        raise ValueError(f"velocity_height command 仅支持 5D/8D，实际为 {command_dim}D")
    command_metadata: dict[str, Any] = {"dimension": command_dim}
    deployment_ranges = getattr(term.cfg, "deployment_ranges", None)
    if deployment_ranges is not None:
        command_metadata["ranges"] = _normalize_deployment_command_ranges(
            deployment_ranges,
            command_dim=command_dim,
        )
    return {"velocity_height": command_metadata}


def _normalize_deployment_command_ranges(
    value: Any,
    *,
    command_dim: int,
) -> dict[str, list[float]]:
    """校验并序列化任务显式声明的最终课程 command 包络。"""
    if not isinstance(value, Mapping):
        raise TypeError("deployment_ranges 必须为 mapping")
    expected = set(_COMMAND_FIELD_NAMES[:command_dim])
    actual = set(value)
    if actual != expected:
        raise ValueError(
            "deployment_ranges 字段不匹配："
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )

    normalized: dict[str, list[float]] = {}
    for field_name in _COMMAND_FIELD_NAMES[:command_dim]:
        field_range = value[field_name]
        if not isinstance(field_range, (list, tuple)) or len(field_range) != 2:
            raise TypeError(f"deployment_ranges.{field_name} 必须为二元数组")
        if any(
            isinstance(item, bool) or not isinstance(item, (int, float)) for item in field_range
        ):
            raise TypeError(f"deployment_ranges.{field_name} 必须包含数值")
        lower, upper = (float(item) for item in field_range)
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError(f"deployment_ranges.{field_name} 必须为有限数值")
        if lower > upper:
            raise ValueError(f"deployment_ranges.{field_name} 下界不得大于上界")
        normalized[field_name] = [lower, upper]
    return normalized


def _resolve_observation_groups(
    observation_manager: Any,
    observation_group_names: list[str] | tuple[str, ...],
) -> list[str]:
    available = list(observation_manager.active_terms)
    if not observation_group_names:
        if "actor" in observation_manager.active_terms:
            return ["actor"]
        raise ValueError(f"无法推断 actor observation group，可用组为 {available!r}")
    groups = [str(name) for name in observation_group_names]
    missing = [name for name in groups if name not in observation_manager.active_terms]
    if missing:
        raise ValueError(f"actor observation groups 不存在：{missing!r}，可用组为 {available!r}")
    return groups


def _build_observation_groups(
    observation_manager: Any,
    group_names: list[str],
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for group_name in group_names:
        if not observation_manager.group_obs_concatenate[group_name]:
            raise ValueError(f"actor observation group {group_name!r} 必须 concatenate_terms=True")
        terms: list[dict[str, Any]] = []
        for term_name, flattened_shape in zip(
            observation_manager.active_terms[group_name],
            observation_manager.group_obs_term_dim[group_name],
            strict=True,
        ):
            if term_name not in _TERM_WIDTHS:
                raise ValueError(f"暂不支持 observation term {term_name!r}")
            term_cfg = observation_manager.get_term_cfg(group_name, term_name)
            history_length = max(1, int(getattr(term_cfg, "history_length", 0)))
            flattened_width = int(math.prod(flattened_shape))
            expected_width = _TERM_WIDTHS[term_name] * history_length
            if flattened_width != expected_width:
                raise ValueError(
                    f"{group_name}.{term_name} 展平宽度应为 {expected_width}，"
                    f"实际为 {flattened_width}"
                )
            if history_length > 1 and not bool(term_cfg.flatten_history_dim):
                raise ValueError(f"{group_name}.{term_name} 的 history 必须展平")
            entry: dict[str, Any] = {
                "name": term_name,
                "scale": _observation_scale(term_name, term_cfg),
                "history_length": history_length,
                "clip": _observation_clip(term_name, term_cfg),
            }
            if term_name in {"commands", "jump_commands"}:
                entry["params"] = {"command_name": "velocity_height"}
            terms.append(entry)
        groups[group_name] = {"terms": terms}
    return groups


def _observation_scale(term_name: str, term_cfg: Any) -> list[float]:
    cfg = ObservationConfig()
    internal = {
        "base_ang_vel": [cfg.ang_vel_scale] * 3,
        "projected_gravity": [1.0] * 3,
        "commands": list(cfg.command_scale),
        "leg_joint_pos": [1.0] * 6,
        "leg_joint_vel": [cfg.leg_vel_scale] * 4,
        "wheel_pos_zero": [1.0] * 2,
        "wheel_vel": [cfg.wheel_vel_scale] * 2,
        "last_actions": [1.0] * 6,
        "jump_commands": [1.0] * 3,
    }[term_name]
    manager_scale = getattr(term_cfg, "scale", None)
    if manager_scale is None:
        external = [1.0] * len(internal)
    elif isinstance(manager_scale, (int, float)):
        external = [float(manager_scale)] * len(internal)
    else:
        external = [float(value) for value in manager_scale]
        if len(external) != len(internal):
            raise ValueError(f"{term_name} observation scale 维度错误：{external!r}")
    return [left * right for left, right in zip(internal, external, strict=True)]


def _observation_clip(term_name: str, term_cfg: Any) -> list[float] | None:
    cfg = ObservationConfig()
    internal = (
        [-1.0, 1.0]
        if term_name == "projected_gravity"
        else None
        if term_name == "wheel_pos_zero"
        else [-float(cfg.clip_value), float(cfg.clip_value)]
    )
    manager_clip = getattr(term_cfg, "clip", None)
    if manager_clip is None:
        return internal
    external = [float(value) for value in manager_clip]
    if len(external) != 2:
        raise ValueError(f"{term_name} observation clip 必须为 [min, max]")
    if internal is None:
        return external
    return [max(internal[0], external[0]), min(internal[1], external[1])]


def _build_action_metadata(runtime_env: Any) -> dict[str, Any]:
    manager = runtime_env.action_manager
    if list(manager.active_terms) != ["delayed_action"] or int(manager.total_action_dim) != 6:
        raise ValueError(
            "SerialLeg ONNX metadata 要求唯一的 6D delayed_action，"
            f"实际为 {manager.active_terms}, dim={manager.total_action_dim}"
        )
    term = manager.get_term("delayed_action")
    cfg = term.cfg
    scale = [float(value) for value in cfg.leg_scales] + [float(cfg.wheel_scale)] * 2
    clip = getattr(cfg, "action_clip", None)
    height_conditioned = getattr(cfg, "height_conditioned_action_default", None)
    if height_conditioned is None:
        raise ValueError("delayed_action cfg 缺少 height_conditioned_action_default")
    return {
        "name": type(term).__name__,
        "scale": scale,
        "offset": list(RobotConfig().default_dof_pos),
        "clip": None if clip is None else [abs(float(clip))] * 6,
        # B18：动作零点语义。True = 高度条件默认姿态（随 height 指令变化），
        # False = 固定 entity 默认姿态。部署端必须按此选择 decode 策略。
        "height_conditioned_action_default": bool(height_conditioned),
    }


def _validate_onnx_graph(model: onnx.ModelProto, metadata: dict[str, Any]) -> None:
    inputs = {value.name: value for value in model.graph.input}
    outputs = {value.name: value for value in model.graph.output}
    is_rnn = metadata.get("meta", {}).get("training", {}).get("is_rnn")
    expected_inputs = {"obs", "h_in"} if is_rnn else {"obs"}
    expected_outputs = {"actions", "h_out"} if is_rnn else {"actions"}
    if set(inputs) != expected_inputs or set(outputs) != expected_outputs:
        raise ValueError(
            "ONNX I/O 与 policy 类型不一致："
            f"inputs={list(inputs)!r}, outputs={list(outputs)!r}, is_rnn={is_rnn!r}"
        )
    observation_dim = sum(
        _TERM_WIDTHS[term["name"]] * int(term["history_length"])
        for group in metadata["policy_io"]["groups"].values()
        for term in group["terms"]
    )
    action_dim = len(metadata["policy_io"]["action"]["scale"])
    if _static_last_dim(inputs["obs"]) != observation_dim:
        raise ValueError(
            f"ONNX obs 为 {_static_last_dim(inputs['obs'])}D，descriptor 为 {observation_dim}D"
        )
    if _static_last_dim(outputs["actions"]) != action_dim:
        raise ValueError(
            f"ONNX actions 为 {_static_last_dim(outputs['actions'])}D，descriptor 为 {action_dim}D"
        )


def _static_last_dim(value: onnx.ValueInfoProto) -> int:
    dimensions = value.type.tensor_type.shape.dim
    if not dimensions or not dimensions[-1].HasField("dim_value"):
        raise ValueError(f"ONNX tensor {value.name!r} 的最后一维必须为静态整数")
    return int(dimensions[-1].dim_value)


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


__all__ = [
    "ONNX_METADATA_KEY",
    "SCHEMA_NAME",
    "build_deployment_onnx_metadata",
    "embed_onnx_metadata",
]
