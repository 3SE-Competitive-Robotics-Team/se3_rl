"""SerialLeg 原地跳跃运动学参考轨迹生成命令行入口（se3.jump_ref.v1）。

用法（默认即训练用的四条参考：站姿 0.22 m、无起始停顿与下蹲，第 0 帧即恒加速度蹬地）：
    uv run se3-jump-to
    uv run se3-jump-to --heights 0.2 0.3 --output-dir /tmp/jump_ref

生成逻辑见 se3_jump_to.reference；旧版插值生成器在当前闭链模型上失效，已移除。
旧格式文件（assets/trajectories/jump_*.npz）仍可用 se3-jump-to-replay 回放。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from se3_jump_to.reference import JumpReferenceParams, ReferenceKinematics, generate_reference

DEFAULT_HEIGHTS = (0.20, 0.30, 0.40, 0.50)
DEFAULT_OUTPUT_DIR = Path("assets/trajectories/jump_ref_v2_nocrouch_h022")
DEFAULT_STAND_HEIGHT = 0.22


def main() -> None:
    parser = argparse.ArgumentParser(description="SerialLeg 原地跳跃运动学参考轨迹生成")
    parser.add_argument(
        "--heights",
        type=float,
        nargs="+",
        default=list(DEFAULT_HEIGHTS),
        help="目标轮底离地间隙（m），可多个",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录，默认 {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--stand-height",
        type=float,
        default=DEFAULT_STAND_HEIGHT,
        help=f"起止站姿机身高度（m），默认 {DEFAULT_STAND_HEIGHT}",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kin = ReferenceKinematics()
    for height in args.heights:
        stand = float(args.stand_height)
        # 去掉起始站立停顿与下蹲段（J7 起）：flag 一到就蹬，下蹲腿长取站姿腿长。
        overrides = {
            "stand_height": stand,
            "t_hold0": 0.0,
            "t_crouch": 0.0,
            "crouch_leg": kin.leg_len_for_base_z(stand),
        }
        ref = generate_reference(JumpReferenceParams(target_clearance=height, **overrides), kin)
        path = args.output_dir / f"jump_{height:.2f}m.npz"
        np.savez(path, **ref)
        meta = json.loads(str(ref["meta"]))
        print(
            f"[jump_ref] {path}: 离地间隙峰值 {meta['clearance_peak']:.3f} m，起跳速度 "
            f"{meta['v_takeoff']:.2f} m/s，腾空 {meta['flight_time']:.3f} s，"
            f"总时长 {len(ref['phase']) * float(ref['dt']):.2f} s"
        )


if __name__ == "__main__":
    main()
