"""观测空间配置 — 训练和验证共享的缩放系数。"""

from __future__ import annotations

from pydantic import BaseModel


class ObservationConfig(BaseModel):
    """actor 策略输入的缩放系数，修改一处即两端同步生效。critic 额外包含特权观测（lin_vel 3D + contact 2D + height 1D）。

    旧行走/跳跃观测布局（32 维）：
        [0:3]   base_ang_vel       × 0.25
        [3:6]   projected_gravity
        [6:11]  commands           × (2.0, 0.25, 5.0, 5.0, 5.0)
        [11:15] leg_joint_pos      相对 default_dof_pos
        [15:19] leg_joint_vel      × 0.25
        [19:21] wheel_pos
        [21:23] wheel_vel          × 0.05
        [23:29] last_actions
        [29:32] jump_commands      [jump_flag, jump_target_height, jump_phase]

    TaskMode 统一任务观测为 42 维，将最后 3 维替换为 13 维 task_mode：
        [29:42] task_mode          [current_semantic(4), prev_semantic(4),
                                    mode_blend, jump_flag, jump_target_height,
                                    jump_phase, jump_stage_norm]
    """

    ang_vel_scale: float = 0.25
    command_scale: tuple[float, ...] = (2.0, 0.25, 5.0, 5.0, 5.0)
    leg_vel_scale: float = 0.25
    wheel_vel_scale: float = 0.05
    clip_value: float = 100.0
    num_obs: int = 32
    task_mode_num_obs: int = 42
    num_actions: int = 6
