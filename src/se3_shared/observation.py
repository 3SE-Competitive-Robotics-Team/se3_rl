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
    """

    ang_vel_scale: float = 0.25
    command_scale: tuple[float, ...] = (2.0, 0.25, 5.0, 5.0, 5.0)
    leg_vel_scale: float = 0.25
    wheel_vel_scale: float = 0.05
    clip_value: float = 100.0
    num_obs: int = 34
    num_actions: int = 6
