"""从 sim2sim 的版本化几何生成 MCU 气弹簧补偿表；通过 uv run 执行。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from se3_runtime._serialleg_v1 import _ACTIVE_UPPER, knee_gas_spring_compensation_torque


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firmware-root", type=Path, required=True)
    args = parser.parse_args()
    grid = np.linspace(0.0, _ACTIVE_UPPER, 4097)
    values = [knee_gas_spring_compensation_torque([0, -a, 0, a], 1.0)[0] for a in grid]
    rows = [
        "    " + ", ".join(f"{v:.10e}f" for v in values[i : i + 8]) + ","
        for i in range(0, len(values), 8)
    ]
    text = """/* 由训练仓库 scripts/export_knee_spring_firmware.py 生成，禁止手改表值。
 * 来源：se3_runtime._serialleg_v1，c/F = -dL/dalpha，单位 m。
 * 4097 点覆盖闭链主动杆有效范围；运行期仅做线性插值。
 */
#ifndef KNEE_SPRING_COMPENSATION_H
#define KNEE_SPRING_COMPENSATION_H
#include <math.h>
#include <stdint.h>
"""
    text += f"#define KNEE_SPRING_ALPHA_MAX {_ACTIVE_UPPER:.12e}f\n"
    text += (
        "static const float knee_spring_torque_per_newton[4097] = {\n" + "\n".join(rows) + "\n};\n"
    )
    text += """
static inline float knee_spring_arm(float alpha)
{
    if (alpha <= 0.0f) return knee_spring_torque_per_newton[0];
    if (alpha >= KNEE_SPRING_ALPHA_MAX) return knee_spring_torque_per_newton[4096];
    float index = alpha * (4096.0f / KNEE_SPRING_ALPHA_MAX);
    uint32_t lower = (uint32_t)index;
    if (lower >= 4096U) return knee_spring_torque_per_newton[4096];
    return knee_spring_torque_per_newton[lower] + (index - (float)lower) *
        (knee_spring_torque_per_newton[lower + 1U] - knee_spring_torque_per_newton[lower]);
}

/* 输入输出均为 policy-order LF/LB/RF/RB；无效输入返回 0，由调用方停止输出。 */
static inline uint8_t knee_spring_compensation(const float q[4], float force,
                                              uint8_t enabled, float torque[4])
{
    for (uint8_t i = 0U; i < 4U; ++i) torque[i] = 0.0f;
    if (!isfinite(force) || force < 0.0f || enabled > 1U) return 0U;
    for (uint8_t i = 0U; i < 4U; ++i) {
        if (!isfinite(q[i])) return 0U;
    }
    if (!enabled) return 1U;
    if (force <= 0.0f) return 0U;
    float left = force * knee_spring_arm(q[0] - q[1]);
    float right = force * knee_spring_arm(q[3] - q[2]);
    if (!isfinite(left) || !isfinite(right)) return 0U;
    torque[0] = left;
    torque[1] = -left;
    torque[2] = -right;
    torque[3] = right;
    return 1U;
}
#endif
"""
    output = args.firmware_root / "Custom/Controller/knee_spring_compensation.h"
    output.write_text(text, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
