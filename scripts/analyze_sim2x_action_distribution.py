"""批量运行 sim2x 闭环并绘制 policy action 与相邻帧跳变分布。"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

from se3_runtime import PolicyControlLoop, PolicyRuntime
from se3_runtime_mujoco import MujocoPolicyAdapter

_SCENARIOS = ("standing", "locomotion", "side_recovery", "inverted_recovery")
_SCENARIO_LABELS = {
    "standing": "站立",
    "locomotion": "随机行驶",
    "side_recovery": "侧倒恢复",
    "inverted_recovery": "倒置恢复",
}
_ACTION_LABELS = (
    "LF common",
    "L active",
    "RF common",
    "R active",
    "L wheel",
    "R wheel",
)
_LEG_TARGET_LABELS = ("LF target", "LB target", "RF target", "RB target")
_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
_WHEEL_VELOCITY_LIMIT = 68.50167981327459


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    """描述一个可复现的闭环采样 episode。"""

    index: int
    scenario: str
    seed: int


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """传给独立采样进程的不可变配置。"""

    onnx_path: str
    policy_steps: int
    solver_hz: float
    episodes: tuple[EpisodeSpec, ...]
    worker_seed: int


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True, help="待分析的 ONNX policy")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/analysis/action_distribution"),
        help="数据、统计和图片输出目录",
    )
    parser.add_argument("--episodes", type=int, default=80, help="总 episode 数，按四类场景均分")
    parser.add_argument(
        "--policy-steps", type=int, default=160, help="每个 episode 的 policy tick 数"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并行采样进程数；正式 MJCF mesh 占用较大，Windows 默认单进程",
    )
    parser.add_argument("--physics-hz", type=float, default=500.0, help="MuJoCo solver 频率")
    parser.add_argument("--seed", type=int, default=20260830, help="采样随机种子")
    parser.add_argument(
        "--reuse-dataset",
        action="store_true",
        help="复用 output-dir 中已有 action_samples.npz，仅重绘和汇总",
    )
    return parser


def _episode_specs(count: int, seed: int) -> tuple[EpisodeSpec, ...]:
    return tuple(
        EpisodeSpec(
            index=index,
            scenario=_SCENARIOS[index % len(_SCENARIOS)],
            seed=seed + 104729 * index,
        )
        for index in range(count)
    )


def _partition_specs(
    specs: tuple[EpisodeSpec, ...],
    workers: int,
) -> tuple[tuple[EpisodeSpec, ...], ...]:
    return tuple(tuple(specs[index::workers]) for index in range(workers))


def _session_options() -> ort.SessionOptions:
    """限制每个 worker 的 ORT 线程，避免多进程过度抢占 CPU。"""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return options


def _set_normal_commands(
    adapter: MujocoPolicyAdapter,
    *,
    vx: float,
    yaw: float,
    height: float,
) -> None:
    adapter.set_command_field("velocity_height", "lin_vel_x", vx)
    adapter.set_command_field("velocity_height", "ang_vel_yaw", yaw)
    adapter.set_command_field("velocity_height", "height", height)


def _prepare_episode(
    adapter: MujocoPolicyAdapter,
    loop: PolicyControlLoop,
    spec: EpisodeSpec,
) -> dict[str, float | str]:
    rng = np.random.default_rng(spec.seed)
    if spec.scenario == "standing":
        _set_normal_commands(adapter, vx=0.0, yaw=0.0, height=0.22)
        metadata: dict[str, float | str] = {"vx": 0.0, "yaw": 0.0, "height": 0.22}
    elif spec.scenario == "locomotion":
        vx = float(rng.uniform(-1.5, 1.5))
        yaw = float(rng.uniform(-2.0, 2.0))
        height = float(rng.uniform(0.195, 0.30))
        _set_normal_commands(adapter, vx=vx, yaw=yaw, height=height)
        metadata = {"vx": vx, "yaw": yaw, "height": height}
    elif spec.scenario == "side_recovery":
        pose = "left_side" if (spec.index // len(_SCENARIOS)) % 2 == 0 else "right_side"
        adapter.request_recovery_reset(pose)
        metadata = {"pose": pose, "vx": 0.0, "yaw": 0.0, "height": 0.26}
    elif spec.scenario == "inverted_recovery":
        # 当前 prone/supine quaternion 等价，只采一份避免重复污染分布。
        adapter.request_recovery_reset("prone")
        metadata = {"pose": "prone", "vx": 0.0, "yaw": 0.0, "height": 0.26}
    else:
        raise ValueError(f"未知采样场景：{spec.scenario!r}")
    loop.reset()
    return metadata


def _sample_worker(config: WorkerConfig) -> dict[str, np.ndarray]:
    """在一个独立进程中采集一组真实闭环 episode。"""
    runtime = PolicyRuntime.load(
        config.onnx_path,
        session_options=_session_options(),
        action_delay_random_seed=config.worker_seed,
    )
    adapter = MujocoPolicyAdapter(
        runtime.contract,
        artifact_path=runtime.bundle.path,
        solver_dt_s=1.0 / config.solver_hz,
    )
    loop = PolicyControlLoop(runtime, adapter)

    raw_actions: list[np.ndarray] = []
    clipped_actions: list[np.ndarray] = []
    action_deltas: list[np.ndarray] = []
    leg_targets: list[np.ndarray] = []
    leg_target_deltas: list[np.ndarray] = []
    wheel_targets: list[np.ndarray] = []
    active_clamped: list[np.ndarray] = []
    scenario_ids: list[int] = []
    episode_ids: list[int] = []
    policy_steps: list[int] = []
    commands: list[np.ndarray] = []

    for spec in config.episodes:
        metadata = _prepare_episode(adapter, loop, spec)
        previous_raw: np.ndarray | None = None
        previous_leg_target: np.ndarray | None = None
        command_vector = np.asarray(
            (metadata["vx"], metadata["yaw"], metadata["height"]),
            dtype=np.float64,
        )
        scenario_id = _SCENARIOS.index(spec.scenario)
        for step in range(config.policy_steps):
            result = loop.policy_step()
            raw = np.asarray(result.inference.raw_action, dtype=np.float64).reshape(6)
            decoded = result.last_decoded_action
            clipped = np.asarray(decoded.clipped_action, dtype=np.float64).reshape(6)
            leg_target = np.asarray(decoded.leg_position_target, dtype=np.float64).reshape(4)

            raw_actions.append(raw)
            clipped_actions.append(clipped)
            action_deltas.append(
                np.full(6, np.nan, dtype=np.float64) if previous_raw is None else raw - previous_raw
            )
            leg_targets.append(leg_target)
            leg_target_deltas.append(
                np.full(4, np.nan, dtype=np.float64)
                if previous_leg_target is None
                else leg_target - previous_leg_target
            )
            wheel_targets.append(
                np.asarray(decoded.wheel_velocity_target, dtype=np.float64).reshape(2)
            )
            active_clamped.append(
                np.asarray(decoded.active_rod_target_clamped, dtype=np.bool_).reshape(2)
            )
            scenario_ids.append(scenario_id)
            episode_ids.append(spec.index)
            policy_steps.append(step)
            commands.append(command_vector)
            previous_raw = raw
            previous_leg_target = leg_target

    return {
        "raw_action": np.asarray(raw_actions, dtype=np.float32),
        "clipped_action": np.asarray(clipped_actions, dtype=np.float32),
        "action_delta": np.asarray(action_deltas, dtype=np.float32),
        "leg_target": np.asarray(leg_targets, dtype=np.float32),
        "leg_target_delta": np.asarray(leg_target_deltas, dtype=np.float32),
        "wheel_target": np.asarray(wheel_targets, dtype=np.float32),
        "active_clamped": np.asarray(active_clamped, dtype=np.bool_),
        "scenario_id": np.asarray(scenario_ids, dtype=np.int8),
        "episode_id": np.asarray(episode_ids, dtype=np.int32),
        "policy_step": np.asarray(policy_steps, dtype=np.int16),
        "command": np.asarray(commands, dtype=np.float32),
    }


def _merge_results(results: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = results[0].keys()
    merged = {key: np.concatenate([result[key] for result in results], axis=0) for key in keys}
    order = np.lexsort((merged["policy_step"], merged["episode_id"]))
    return {key: value[order] for key, value in merged.items()}


def _quantiles(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    levels = (50.0, 95.0, 99.0, 99.9, 100.0)
    return {
        name: float(value)
        for name, value in zip(
            ("p50", "p95", "p99", "p99_9", "max"),
            np.percentile(finite, levels),
            strict=True,
        )
    }


def _build_summary(data: dict[str, np.ndarray]) -> dict[str, object]:
    raw = np.asarray(data["raw_action"], dtype=np.float64)
    delta = np.asarray(data["action_delta"], dtype=np.float64)
    clipped = np.asarray(data["clipped_action"], dtype=np.float64)
    scenario_ids = data["scenario_id"]
    finite_delta = np.isfinite(delta).all(axis=1)
    summary: dict[str, object] = {
        "action_vectors": int(raw.shape[0]),
        "scalar_action_values": int(raw.size),
        "episodes": int(np.unique(data["episode_id"]).size),
        "policy_frequency_hz": 50.0,
        "physics_frequency_hz": 500.0,
        "action_labels": list(_ACTION_LABELS),
        "raw_clip_hit_ratio": float(np.mean(np.any(np.abs(raw - clipped) > 1.0e-9, axis=1))),
        "active_target_clamp_ratio_left_right": np.mean(data["active_clamped"], axis=0).tolist(),
        "global": {},
        "scenarios": {},
    }
    global_metrics: dict[str, object] = {}
    for index, label in enumerate(_ACTION_LABELS):
        global_metrics[label] = {
            "raw_signed": _quantiles(raw[:, index]),
            "raw_abs": _quantiles(np.abs(raw[:, index])),
            "delta_abs": _quantiles(np.abs(delta[finite_delta, index])),
            "raw_abs_over_1_ratio": float(np.mean(np.abs(raw[:, index]) > 1.0)),
            "raw_abs_over_3_ratio": float(np.mean(np.abs(raw[:, index]) > 3.0)),
            "delta_abs_over_1_ratio": float(np.mean(np.abs(delta[finite_delta, index]) > 1.0)),
            "delta_abs_over_3_ratio": float(np.mean(np.abs(delta[finite_delta, index]) > 3.0)),
        }
    summary["global"] = global_metrics

    scenario_metrics: dict[str, object] = {}
    for scenario_id, scenario in enumerate(_SCENARIOS):
        mask = scenario_ids == scenario_id
        delta_mask = mask & finite_delta
        dimensions: dict[str, object] = {}
        for index, label in enumerate(_ACTION_LABELS):
            dimensions[label] = {
                "raw_abs": _quantiles(np.abs(raw[mask, index])),
                "delta_abs": _quantiles(np.abs(delta[delta_mask, index])),
            }
        scenario_metrics[scenario] = {
            "samples": int(np.count_nonzero(mask)),
            "dimensions": dimensions,
        }
    summary["scenarios"] = scenario_metrics

    decoder_metrics: dict[str, object] = {}
    for scenario_id, scenario in enumerate(_SCENARIOS):
        mask = scenario_ids == scenario_id
        delta_mask = mask & finite_delta
        decoder_metrics[scenario] = {
            "active_target_clamp_ratio_left_right": np.mean(
                data["active_clamped"][mask], axis=0
            ).tolist(),
            "wheel_target_over_velocity_limit_ratio_left_right": np.mean(
                np.abs(data["wheel_target"][mask]) > _WHEEL_VELOCITY_LIMIT,
                axis=0,
            ).tolist(),
            "leg_target_delta_abs": _quantiles(
                np.abs(data["leg_target_delta"][delta_mask]).reshape(-1)
            ),
            "leg_target_delta_component_over_pi_ratio": float(
                np.mean(np.abs(data["leg_target_delta"][delta_mask]) > math.pi)
            ),
        }
    summary["decoder_by_scenario"] = decoder_metrics

    front_delta = np.abs(delta[:, (0, 2)])
    flat_index = int(np.nanargmax(front_delta))
    row, local_dimension = np.unravel_index(flat_index, front_delta.shape)
    dimension = (0, 2)[local_dimension]
    episode_id = int(data["episode_id"][row])
    step = int(data["policy_step"][row])
    previous_mask = (data["episode_id"] == episode_id) & (data["policy_step"] == step - 1)
    previous_action = raw[previous_mask][0] if bool(np.any(previous_mask)) else np.full(6, np.nan)
    summary["largest_front_action_jump"] = {
        "scenario": _SCENARIOS[int(scenario_ids[row])],
        "episode_id": episode_id,
        "policy_step": step,
        "dimension": _ACTION_LABELS[dimension],
        "delta": float(delta[row, dimension]),
        "before": previous_action.tolist(),
        "after": raw[row].tolist(),
        "command_vx_yaw_height": data["command"][row].astype(float).tolist(),
    }
    return summary


def _style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _plot_raw_distribution(data: dict[str, np.ndarray], output_path: Path) -> None:
    """绘制六维 raw action 直方图。"""
    _style_matplotlib()
    raw = np.asarray(data["raw_action"], dtype=np.float64)
    scenario_ids = data["scenario_id"]
    figure, axes = plt.subplots(2, 3, figsize=(17, 9))

    for dimension, label in enumerate(_ACTION_LABELS):
        row, column = divmod(dimension, 3)
        axis = axes[row, column]
        lower, upper = np.percentile(raw[:, dimension], (0.1, 99.9))
        padding = max((upper - lower) * 0.06, 0.05)
        bins = np.linspace(lower - padding, upper + padding, 100)
        for scenario_id, scenario in enumerate(_SCENARIOS):
            values = raw[scenario_ids == scenario_id, dimension]
            axis.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.5,
                color=_COLORS[scenario_id],
                label=_SCENARIO_LABELS[scenario],
            )
        maximum = float(np.max(np.abs(raw[:, dimension])))
        axis.set_title(f"{label} · raw action · max |a|={maximum:.2f}")
        axis.set_xlabel("raw action")
        axis.set_ylabel("density")
        axis.axvline(0.0, color="#666666", linewidth=0.8, alpha=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle(
        "model_4600 · 500 Hz sim2x 闭环 raw action 分布",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_delta_distribution(data: dict[str, np.ndarray], output_path: Path) -> None:
    """绘制六维 action 相邻 policy tick 跳变的 CCDF。"""
    _style_matplotlib()
    delta = np.asarray(data["action_delta"], dtype=np.float64)
    scenario_ids = data["scenario_id"]
    figure, axes = plt.subplots(2, 3, figsize=(17, 9))

    for dimension, label in enumerate(_ACTION_LABELS):
        row, column = divmod(dimension, 3)
        axis = axes[row, column]
        for scenario_id, scenario in enumerate(_SCENARIOS):
            values = np.abs(delta[scenario_ids == scenario_id, dimension])
            values = np.sort(values[np.isfinite(values)])
            if values.size == 0:
                continue
            indices = np.unique(
                np.linspace(0, values.size - 1, min(values.size, 700), dtype=np.int64)
            )
            x = values[indices]
            y = (values.size - indices) / values.size
            axis.plot(
                x,
                y,
                linewidth=1.5,
                color=_COLORS[scenario_id],
                label=_SCENARIO_LABELS[scenario],
            )
        axis.set_title(f"{label} · adjacent-tick jump")
        axis.set_xlabel("|Δ raw action| / 20 ms")
        axis.set_ylabel("P(|Δa| ≥ x)")
        axis.set_xscale("symlog", linthresh=0.01)
        axis.set_yscale("log")
        axis.set_ylim(1.0e-4, 1.05)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle(
        "model_4600 · 500 Hz sim2x 相邻 policy tick action 跳变 CCDF",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_worst_jump(data: dict[str, np.ndarray], output_path: Path) -> None:
    """绘制共同转角 action 最大跳变 episode 的局部时间序列。"""
    _style_matplotlib()
    delta = np.asarray(data["action_delta"], dtype=np.float64)
    front_delta = np.abs(delta[:, (0, 2)])
    flat_index = int(np.nanargmax(front_delta))
    row, local_dimension = np.unravel_index(flat_index, front_delta.shape)
    dimension = (0, 2)[local_dimension]
    episode_id = int(data["episode_id"][row])
    event_step = int(data["policy_step"][row])
    mask = (
        (data["episode_id"] == episode_id)
        & (data["policy_step"] >= event_step - 20)
        & (data["policy_step"] <= event_step + 20)
    )
    order = np.argsort(data["policy_step"][mask])
    steps = data["policy_step"][mask][order]
    raw = data["raw_action"][mask][order]
    targets = data["leg_target"][mask][order]
    scenario = _SCENARIOS[int(data["scenario_id"][row])]
    jump = float(delta[row, dimension])

    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    for index, label in enumerate(_ACTION_LABELS[:4]):
        axes[0].plot(steps, raw[:, index], linewidth=1.5, label=label)
    axes[0].axvline(event_step, color="#D55E00", linewidth=1.3, linestyle="--")
    axes[0].set_ylabel("raw action")
    axes[0].set_title(
        f"最大共同转角跳变：{_ACTION_LABELS[dimension]} Δ={jump:.2f} · "
        f"{_SCENARIO_LABELS[scenario]} · episode {episode_id}"
    )
    axes[0].legend(ncol=4, frameon=False)

    for index, label in enumerate(_LEG_TARGET_LABELS):
        axes[1].plot(steps, targets[:, index], linewidth=1.5, label=label)
    axes[1].axvline(event_step, color="#D55E00", linewidth=1.3, linestyle="--")
    axes[1].set_xlabel("policy step (20 ms)")
    axes[1].set_ylabel("decoded target / rad")
    axes[1].legend(ncol=4, frameon=False)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_decoder_limits(data: dict[str, np.ndarray], output_path: Path) -> None:
    """绘制 action decoder 限位命中率和腿目标最大跳变。"""
    _style_matplotlib()
    scenario_ids = data["scenario_id"]
    active_clamp = np.empty((len(_SCENARIOS), 2), dtype=np.float64)
    wheel_overspeed = np.empty((len(_SCENARIOS), 2), dtype=np.float64)
    maximum_leg_jump = np.empty(len(_SCENARIOS), dtype=np.float64)
    for scenario_id in range(len(_SCENARIOS)):
        mask = scenario_ids == scenario_id
        finite = mask & np.isfinite(data["leg_target_delta"]).all(axis=1)
        active_clamp[scenario_id] = np.mean(data["active_clamped"][mask], axis=0)
        wheel_overspeed[scenario_id] = np.mean(
            np.abs(data["wheel_target"][mask]) > _WHEEL_VELOCITY_LIMIT,
            axis=0,
        )
        maximum_leg_jump[scenario_id] = float(np.max(np.abs(data["leg_target_delta"][finite])))

    x = np.arange(len(_SCENARIOS), dtype=np.float64)
    width = 0.34
    labels = [_SCENARIO_LABELS[scenario] for scenario in _SCENARIOS]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.6))

    axes[0].bar(x - width / 2.0, active_clamp[:, 0] * 100.0, width, label="Left")
    axes[0].bar(x + width / 2.0, active_clamp[:, 1] * 100.0, width, label="Right")
    axes[0].set_title("active rod target clamp rate")
    axes[0].set_ylabel("samples / %")
    axes[0].set_xticks(x, labels, rotation=18)
    axes[0].set_ylim(0.0, 108.0)
    axes[0].legend(frameon=False)

    axes[1].bar(x - width / 2.0, wheel_overspeed[:, 0] * 100.0, width, label="Left")
    axes[1].bar(x + width / 2.0, wheel_overspeed[:, 1] * 100.0, width, label="Right")
    axes[1].set_title(f"wheel target > {_WHEEL_VELOCITY_LIMIT:.1f} rad/s")
    axes[1].set_ylabel("samples / %")
    axes[1].set_xticks(x, labels, rotation=18)
    axes[1].legend(frameon=False)

    axes[2].bar(x, maximum_leg_jump, width=0.58, color=_COLORS)
    axes[2].axhline(math.pi, color="#D55E00", linestyle="--", linewidth=1.2, label="π rad")
    axes[2].set_title("max decoded leg-target jump / 20 ms")
    axes[2].set_ylabel("|Δ target| / rad")
    axes[2].set_xticks(x, labels, rotation=18)
    axes[2].legend(frameon=False)

    figure.suptitle(
        "action decoder 限位命中与物理目标跳变",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.episodes < len(_SCENARIOS) or args.episodes % len(_SCENARIOS) != 0:
        parser.error(f"--episodes 必须是 {len(_SCENARIOS)} 的倍数且不少于该值")
    if args.policy_steps < 2:
        parser.error("--policy-steps 必须至少为 2")
    if args.workers < 1:
        parser.error("--workers 必须为正整数")
    if not math.isfinite(args.physics_hz) or args.physics_hz <= 0.0:
        parser.error("--physics-hz 必须为正有限数")

    onnx_path = args.onnx.expanduser().resolve()
    if not onnx_path.is_file():
        parser.error(f"ONNX 不存在：{onnx_path}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "action_samples.npz"
    if args.reuse_dataset:
        if not dataset_path.is_file():
            parser.error(f"复用数据集不存在：{dataset_path}")
        with np.load(dataset_path) as archive:
            data = {key: archive[key] for key in archive.files}
    else:
        workers = min(args.workers, args.episodes)
        partitions = _partition_specs(_episode_specs(args.episodes, args.seed), workers)
        configs = tuple(
            WorkerConfig(
                onnx_path=str(onnx_path),
                policy_steps=args.policy_steps,
                solver_hz=args.physics_hz,
                episodes=partition,
                worker_seed=args.seed + 1009 * worker,
            )
            for worker, partition in enumerate(partitions)
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_sample_worker, configs))
        data = _merge_results(results)
        np.savez_compressed(dataset_path, **data)
    summary = _build_summary(data)

    summary_path = output_dir / "action_summary.json"
    raw_distribution_path = output_dir / "action_raw_distribution.png"
    delta_distribution_path = output_dir / "action_delta_ccdf.png"
    decoder_limits_path = output_dir / "action_decoder_limits.png"
    jump_path = output_dir / "action_largest_jump.png"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot_raw_distribution(data, raw_distribution_path)
    _plot_delta_distribution(data, delta_distribution_path)
    _plot_decoder_limits(data, decoder_limits_path)
    _plot_worst_jump(data, jump_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"dataset={dataset_path}")
    print(f"summary={summary_path}")
    print(f"raw_distribution={raw_distribution_path}")
    print(f"delta_distribution={delta_distribution_path}")
    print(f"decoder_limits={decoder_limits_path}")
    print(f"largest_jump={jump_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
