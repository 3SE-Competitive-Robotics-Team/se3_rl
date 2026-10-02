"""观测空间配置 — 训练和验证共享的缩放系数。"""

from __future__ import annotations

from pydantic import BaseModel


class ObservationConfig(BaseModel):
    """34D actor 策略输入的缩放系数，修改一处即两端同步生效。critic 额外包含特权观测（lin_vel 3D + contact 2D + height 1D）。

    观测布局（34 维）：
        [0:3]   base_ang_vel       × 0.25
        [3:6]   projected_gravity
        [6:11]  commands           × (2.0, 0.25, 5.0, 5.0, 5.0)
        [11:17] leg_joint_pos      [sin(LF), cos(LF), left_active, sin(RF), cos(RF), right_active]
        [17:21] leg_joint_vel      × 0.25
        [21:23] wheel_pos_zero     固定为 0，保留兼容槽位
        [23:25] wheel_vel          × 0.05
        [25:31] last_actions
        [31:34] jump_commands      [jump_flag, jump_target_height, jump_phase]
                jump_phase：0→1 的连续相位（grounded 时为 0，飞行/落地时随轨迹帧推进）

    rough 与跳跃 mimic（2026-10-02 起）用 30 维：commands 换成只含 [vx, yaw, height] 的 commands_vx_yaw_height
    （× (2.0, 0.25, 5.0)），删掉 wheel_pos_zero；部署指令契约为 NO_ATTITUDE_COMMAND_FIELDS 六维。
    """

    ang_vel_scale: float = 0.25
    command_scale: tuple[float, ...] = (2.0, 0.25, 5.0, 5.0, 5.0)
    leg_vel_scale: float = 0.25
    wheel_vel_scale: float = 0.05
    clip_value: float = 100.0
    num_obs: int = 34
    num_actions: int = 6


COMMAND_FIELDS: tuple[str, ...] = (
    "lin_vel_x",
    "ang_vel_yaw",
    "pitch",
    "roll",
    "height",
    "jump_flag",
    "jump_target_height",
    "jump_phase",
)
"""训练内部 velocity_height 指令张量的 8 维布局（所有任务共用）。"""
NO_ATTITUDE_COMMAND_FIELDS: tuple[str, ...] = (
    "lin_vel_x",
    "ang_vel_yaw",
    "height",
    "jump_flag",
    "jump_target_height",
    "jump_phase",
)
"""rough / 跳跃 mimic（2026-10-02 起）的部署指令布局：去掉恒 0 的 pitch / roll。训练内部仍是 8 维张量
（pitch / roll 槽恒 0，共享奖励按位置读高度与 jump_flag），只有观测与 ONNX 指令契约按这六个字段。"""
NO_ATTITUDE_COMMAND_OBS_FIELDS: tuple[str, ...] = ("lin_vel_x", "ang_vel_yaw", "height")
"""commands_vx_yaw_height 观测取的三个字段。"""
