"""rough / 跳跃 mimic 的观测与部署指令布局（2026-10-02 用户定）：去掉 pitch / roll 指令与 wheel_pos_zero。

- actor 34 → 30 维：commands（5 维）换成 commands_vx_yaw_height（3 维，[vx, yaw, height]），删 wheel_pos_zero；critic 同步。
- pitch / roll 指令采样恒 0（姿态跟踪奖励按 0 计分），训练内部指令张量仍是 8 维：共享奖励 / 事件按位置读高度与
  jump_flag，Flat 等任务继续用 8 维布局。
- ONNX 指令契约导出 se3_shared.NO_ATTITUDE_COMMAND_FIELDS 六维（显式字段表），runtime / NX / 固件按字段名对应。
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from se3_shared import NO_ATTITUDE_COMMAND_FIELDS
from se3_train.mdp import observations

_DROPPED_TERMS = ("wheel_pos_zero",)


def apply_no_attitude_layout(cfg: ManagerBasedRlEnvCfg) -> None:
    """改观测项与指令包络；须在 velocity_height 指令配置定型之后调用。"""
    cfg.observations = dict(cfg.observations)
    for group in ("actor", "critic"):
        old = cfg.observations[group]
        terms = {}
        for name, term in old.terms.items():
            if name in _DROPPED_TERMS:
                continue
            if name == "commands":
                terms["commands_vx_yaw_height"] = replace(
                    term, func=observations.commands_vx_yaw_height_obs
                )
                continue
            terms[name] = term
        cfg.observations[group] = replace(old, terms=terms)

    command = cfg.commands["velocity_height"]
    command.pitch_range = (0.0, 0.0)
    command.roll_range = (0.0, 0.0)
    command.deployment_fields = NO_ATTITUDE_COMMAND_FIELDS
    ranges = dict(command.deployment_ranges or {})
    command.deployment_ranges = {name: ranges[name] for name in NO_ATTITUDE_COMMAND_FIELDS}


__all__ = ["apply_no_attitude_layout"]
