"""在 1–9 级台阶场景上逐级评测爬升能力，判据与训练一致。

训练里 `terrain_levels` 升级的判据是：一个 episode 内机身到出生点的**切比雪夫距离**
max(|dx|,|dy|) ≥ 4.0 m（= 平台半宽 1.0 + 6 级 × 0.5 m 踏面，即越过最外一级台阶；1.5 m 踏面时是 1.0 + 2 × 1.5，同为 4.0），
episode 长 20 s。本脚本照搬这一套：机器人生成在该级台阶的坑底（世界原点），
给训练同款指令，跑 20 s，看能不能走出 4.0 m。

指令按训练时台阶列的分布抽（rough/env_cfg 的 ROUGH_STAIR_*）：vx ∈ [1.0, 2.4]、yaw 恒 0、
机身高度 ∈ [地形感知下限, 0.38]。**下限逐级不同**，照 commands.RoughCommandTerm 的算法算：
`台阶高 + ROUGH_TERRAIN_HEIGHT_CLEARANCE − ROUGH_BODY_COLLISION_BOTTOM_OFFSET`，再夹进
`ROUGH_STAIR_HEIGHT_RANGE`。第 1 级下限 0.20、第 9 级 0.34——训练时策略在每一级看到的就是这个
分布，全级都按 0.35–0.38 抽会把矮台阶测成它从没练过的样子。默认每 5 s 重采样一次，
与 `resampling_time_range=(5,5)` 对齐；`--hold-command` 可关掉重采样看差别。

`--action-noise` 叠训练式高斯探索噪声（训练里 Mean action std ≈ 0.21）；不加就是确定性回放。

**`--warmup-height` 是为了把「爬升能力」和「静止起步」分开量**：这条策略在高站姿 + 静止时
会死锁（高度指令 ≥ 0.34 时实速 0.03–0.07，先跑起来再拉高能到 2.19 m/s），而本脚本每次
`loop.reset()` 都是静止开局，正好踩在死锁那条路径上——不分开测，会把「上不去」和「起不来」
记成同一个 0。两种模式各跑一遍，差值就是起步死锁吃掉的成功率。

用法：
    uv run python scripts/eval_stair_climb.py --onnx <model.onnx> --trials 8
    uv run python scripts/eval_stair_climb.py --onnx <model.onnx> --warmup-height 0.24
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from se3_runtime import PolicyControlLoop, PolicyRuntime
from se3_runtime_mujoco.adapter import MujocoPolicyAdapter
from se3_train.tasks.rough.env_cfg import (
    ROUGH_BODY_COLLISION_BOTTOM_OFFSET,
    ROUGH_STAIR_HEIGHT_RANGE,
    ROUGH_STAIR_LIN_VEL_X_RANGE,
    ROUGH_TERRAIN_HEIGHT_CLEARANCE,
)

DEFAULT_SCENE = "assets/robots/serialleg/mjcf/serialleg_stairs_1to9.xml"
CLEAR_DISTANCE_M = 4.0  # 与 terrains.ROUGH_TERRAIN_CLEARED_DISTANCE_M 一致
EPISODE_S = 20.0  # 与 env_cfg.episode_length_s 一致
STAIR_VX_RANGE = ROUGH_STAIR_LIN_VEL_X_RANGE
RESAMPLE_S = 5.0  # resampling_time_range
PARKED_Y = 1000.0
FALL_COS = 0.5  # projected gravity z 高于 -0.5（倾角 > 60°）算摔


def _height_range(level: int, base: tuple[float, float]) -> tuple[float, float]:
    """该级台阶上训练时实际抽到的机身高度区间（地形感知下限，见 commands.RoughCommandTerm）。"""
    low, high = base
    required = (
        _step_height(level) + ROUGH_TERRAIN_HEIGHT_CLEARANCE - ROUGH_BODY_COLLISION_BOTTOM_OFFSET
    )
    return min(max(low, required), high), high


def _place_level(model: mujoco.MjModel, data: mujoco.MjData, level: int, levels: list[int]) -> None:
    """把选中的一级挪到世界原点，其余停远。mocap 与 body_pos 都写，reset 后也不会跑掉。"""
    for other in levels:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"se3_stair_level_{other}")
        y = 0.0 if other == level else PARKED_Y
        model.body_pos[body_id][:] = (0.0, y, 0.0)
        mocap_id = int(model.body_mocapid[body_id])
        if mocap_id >= 0:
            data.mocap_pos[mocap_id][:] = (0.0, y, 0.0)


def _step_height(level: int) -> float:
    """difficulty = level/9（TerrainGenerator 的行映射），台阶高 0.02 + difficulty × 0.18。"""
    return 0.02 + (level / 9.0) * 0.18


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--levels", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--trials", type=int, default=8, help="每级重复次数（指令不同）")
    parser.add_argument("--duration", type=float, default=EPISODE_S)
    parser.add_argument("--physics-hz", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hold-command", action="store_true", help="整段不重采样指令")
    parser.add_argument(
        "--height-range",
        type=float,
        nargs=2,
        default=list(ROUGH_STAIR_HEIGHT_RANGE),
        metavar=("LOW", "HIGH"),
        help="台阶列机身高度指令区间，要与被测 checkpoint 训练时的一致。"
        f"默认 {ROUGH_STAIR_HEIGHT_RANGE}（ROUGH_STAIR_HEIGHT_RANGE）；"
        "A13 系列训的是 a13_tuned.A13_STAIR_HEIGHT_RANGE，得传 0.20 0.38。"
        "地形感知下限在此基础上逐级抬高。",
    )
    parser.add_argument(
        "--warmup-height",
        type=float,
        default=None,
        help="助跑段的机身高度指令(m)；设了就先用它跑 --warmup-s 秒再切回抽样高度。"
        " 用来把「爬升能力」和「静止起步」分开测：这条线在高站姿静止起步时会死锁"
        "（实测高度指令 ≥0.34 时实速 0.03–0.07，而先跑起来再拉高能到 2.19），"
        " 不分开测就会把能力误判成 0。",
    )
    parser.add_argument("--warmup-s", type=float, default=2.5, help="助跑时长(s)")
    parser.add_argument(
        "--fixed-height",
        type=float,
        default=None,
        help="整段锁死这个机身高度指令，绕过逐级下限与重采样。用来做姿态扫描："
        "同一级台阶上扫高度，看死锁发生在哪个姿态区间。",
    )
    parser.add_argument("--dump", type=Path, default=None, help="逐次结果写 JSON")
    parser.add_argument(
        "--action-noise", default=None, help="训练式探索噪声：腿σ,轮σ，例如 0.21,0.21"
    )
    args = parser.parse_args()

    runtime = PolicyRuntime.load(args.onnx)
    if args.action_noise:
        leg_sigma, wheel_sigma = (float(v) for v in args.action_noise.split(","))
        runtime.enable_action_noise(leg_sigma, wheel_sigma)  # type: ignore[attr-defined]
    adapter = MujocoPolicyAdapter(
        runtime.contract,
        artifact_path=runtime.bundle.path,
        model_path=args.scene,
        solver_dt_s=1.0 / args.physics_hz,
    )
    loop = PolicyControlLoop(runtime, adapter)
    model, data = adapter.model, adapter.data
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, runtime.contract.robot.base_link)
    policy_dt = float(runtime.contract.timing.policy_dt_s)
    steps = int(round(args.duration / policy_dt))
    resample_every = max(1, int(round(RESAMPLE_S / policy_dt)))
    rng = np.random.default_rng(args.seed)
    warmup_steps = 0 if args.warmup_height is None else int(round(args.warmup_s / policy_dt))

    print(f"策略 {args.onnx.name}   每级 {args.trials} 次 × {args.duration:g}s"
          f"   清块判据 L∞ ≥ {CLEAR_DISTANCE_M} m")
    if warmup_steps:
        print(f"助跑：前 {args.warmup_s:g}s 高度指令锁 {args.warmup_height:.2f} m，之后按抽样值")
    print(f"{'级':>3}{'台阶高':>8}{'高度下限':>9}{'成功率':>9}{'平均最远':>10}{'最好':>8}"
          f"{'平均爬升':>10}{'摔倒率':>8}")
    records: list[dict] = []
    for level in args.levels:
        height_range = (
            (args.fixed_height, args.fixed_height)
            if args.fixed_height is not None
            else _height_range(level, tuple(args.height_range))
        )
        cleared = 0
        fell = 0
        far: list[float] = []
        climb: list[float] = []
        for trial in range(args.trials):
            loop.reset()
            _place_level(model, data, level, args.levels)
            mujoco.mj_forward(model, data)
            origin = data.xpos[base_id][:2].copy()
            vx = float(rng.uniform(*STAIR_VX_RANGE))
            height = float(rng.uniform(*height_range))
            adapter.set_command_field("velocity_height", "lin_vel_x", vx)
            adapter.set_command_field("velocity_height", "ang_vel_yaw", 0.0)
            adapter.set_command_field(
                "velocity_height",
                "height",
                args.warmup_height if warmup_steps else height,
            )
            best = 0.0
            top = 0.0
            crashed = False
            for step in range(steps):
                if warmup_steps and step == warmup_steps:
                    adapter.set_command_field("velocity_height", "height", height)
                if (
                    not args.hold_command
                    and args.fixed_height is None
                    and step > warmup_steps
                    and step % resample_every == 0
                ):
                    adapter.set_command_field(
                        "velocity_height", "lin_vel_x", float(rng.uniform(*STAIR_VX_RANGE))
                    )
                    adapter.set_command_field(
                        "velocity_height", "height", float(rng.uniform(*height_range))
                    )
                loop.policy_step()
                offset = np.abs(data.xpos[base_id][:2] - origin)
                best = max(best, float(offset.max()))
                top = max(top, float(data.xpos[base_id][2]))
                if float(data.xmat[base_id].reshape(3, 3)[2, 2]) < FALL_COS:
                    crashed = True
                    break
            far.append(best)
            climb.append(top)
            records.append(
                {
                    "level": level,
                    "trial": trial,
                    "vx_cmd": round(vx, 3),
                    "height_cmd": round(height, 3),
                    "far": round(best, 3),
                    "top_z": round(top, 3),
                    "fell": crashed,
                    "cleared": best >= CLEAR_DISTANCE_M,
                }
            )
            if crashed:
                fell += 1
            if best >= CLEAR_DISTANCE_M:
                cleared += 1
        rate = cleared / max(args.trials, 1)
        print(f"{level:>3}{_step_height(level):>8.3f}{height_range[0]:>9.3f}"
              f"{rate:>8.0%}{np.mean(far):>10.2f}"
              f"{max(far):>8.2f}{np.mean(climb):>10.3f}{fell / max(args.trials, 1):>8.0%}")
    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        args.dump.write_text(json.dumps(records, indent=1), encoding="utf-8")
        print(f"逐次结果 -> {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
