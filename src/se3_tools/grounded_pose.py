"""命令行工具：计算指定 base_link 高度下轮子触地的腿部关节角。"""

from __future__ import annotations

import argparse
import json

from se3_shared.grounded_pose import solve_grounded_pose


def main() -> None:
    parser = argparse.ArgumentParser(
        description="计算 base_link 指定高度下，左右轮子刚好触地的四个腿部关节角。"
    )
    parser.add_argument(
        "--base-height", type=float, required=True, help="目标 base_link 高度，单位 m"
    )
    parser.add_argument("--ground-height", type=float, default=0.0, help="地面高度，默认 0")
    parser.add_argument(
        "--free-wheel-x",
        action="store_true",
        help="允许轮子 x 位置变化；默认保持默认站立姿态的 wheel x",
    )
    args = parser.parse_args()

    result = solve_grounded_pose(
        base_height=args.base_height,
        ground_height=args.ground_height,
        keep_wheel_x=not args.free_wheel_x,
    )

    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    print()
    print(f"q_legs = {result.q_legs}")
    print(f"q6 = {result.q6}")
    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
