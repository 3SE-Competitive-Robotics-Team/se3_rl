"""SE3 轮腿机器人的终止函数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from se3_train.mdp.contact_utils import finite_contact_force_norm

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.sensor import ContactSensor


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


class BadOrientationDelayed:
    """倾斜超过阈值连续 max_steps 步才终止(给恢复机会)。"""

    def __init__(self) -> None:
        self._fail_count: torch.Tensor | None = None

    def __call__(
        self, env: ManagerBasedRlEnv, limit_angle: float = 0.5236, max_steps: int = 100
    ) -> torch.Tensor:
        if self._fail_count is None:
            self._fail_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

        robot = env.scene["robot"]
        pg_z = robot.data.projected_gravity_b[:, 2]
        tilt_angle = torch.acos(torch.clamp(-pg_z, -1.0, 1.0))
        bad = tilt_angle > limit_angle

        self._fail_count[bad] += 1
        self._fail_count[~bad] = 0
        self._fail_count[env.episode_length_buf <= 1] = 0

        return self._fail_count > max_steps

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._fail_count is not None and env_ids is not None:
            self._fail_count[env_ids] = 0


bad_orientation_delayed = BadOrientationDelayed()


class LowBaseHeightDelayed:
    """base height 连续低于阈值后终止。"""

    def __init__(self) -> None:
        self._fail_count: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        min_height: float,
        max_steps: int = 30,
    ) -> torch.Tensor:
        if self._fail_count is None:
            self._fail_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

        sensor = env.scene[sensor_name]
        height = sensor.data.heights[:, 0]
        low = height < float(min_height)

        self._fail_count[low] += 1
        self._fail_count[~low] = 0
        self._fail_count[env.episode_length_buf <= 1] = 0

        return self._fail_count > int(max_steps)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._fail_count is not None and env_ids is not None:
            self._fail_count[env_ids] = 0


gait_low_base_height_delayed = LowBaseHeightDelayed()


def _has_jump_stage(term: object) -> bool:
    """判断指令项是否暴露跳跃阶段。"""
    return hasattr(term, "jump_stage")


def leg_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 1.0,
    command_name: str | None = None,
    jump_force_threshold: float | None = None,
    jump_landing_force_threshold: float | None = None,
    jump_grace_steps: int = 0,
    terminate: bool = True,
) -> torch.Tensor:
    """腿部 link 接触地面即时终止（膝盖着地 = 非法运动模式）。

    同时将精细拆分指标写入 extras['log']，用于诊断 leg_contact 来源：
    - diag_leg_contact_jump：jump_flag=1 的 env 中腿部触地比例
    - diag_leg_contact_walk：jump_flag=0 的 env 中腿部触地比例
    - diag_leg_contact_rsi：RSI 注入期（前 debounce+2 步）触地比例
    - diag_leg_contact_landing：landing stage（stage==2）触地比例
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    force_mag = finite_contact_force_norm(data.force)
    max_force = force_mag.max(dim=1).values
    has_contact = max_force > force_threshold
    terminate_threshold = torch.full_like(max_force, force_threshold)
    terminate_contact = has_contact

    # 精细拆分诊断：写入 extras["_leg_contact_diag"]（不是 log，log 在 reset 时会被清空）
    # command_manager._update_metrics 在 reset 之后调用，从此处读取并搬入 extras["log"]
    if hasattr(env, "extras"):
        try:
            diag_command_name = command_name or "velocity_height"
            cmd = env.command_manager.get_command(diag_command_name)
            jump_flag = cmd[:, 5] > 0.5

            n_jump = jump_flag.sum().item()
            n_walk = (~jump_flag).sum().item()
            leg_contact_jump = has_contact[jump_flag].float().mean().item() if n_jump > 0 else 0.0
            leg_contact_walk = has_contact[~jump_flag].float().mean().item() if n_walk > 0 else 0.0

            term = env.command_manager.get_term(diag_command_name)
            rsi_window = 5
            in_rsi = env.episode_length_buf <= rsi_window
            rsi_jump = jump_flag & in_rsi
            n_rsi = rsi_jump.sum().item()
            leg_contact_rsi = has_contact[rsi_jump].float().mean().item() if n_rsi > 0 else 0.0

            # landing stage 门控
            if _has_jump_stage(term):
                in_landing = term.jump_stage == 2
                landing_jump = jump_flag & in_landing
                n_landing = landing_jump.sum().item()
                leg_contact_landing = (
                    has_contact[landing_jump].float().mean().item() if n_landing > 0 else 0.0
                )

                if command_name is not None:
                    jump_threshold = (
                        force_threshold if jump_force_threshold is None else jump_force_threshold
                    )
                    landing_threshold = (
                        jump_threshold
                        if jump_landing_force_threshold is None
                        else jump_landing_force_threshold
                    )
                    in_grace = env.episode_length_buf <= jump_grace_steps
                    jump_threshold_tensor = torch.full_like(max_force, jump_threshold)
                    landing_threshold_tensor = torch.full_like(max_force, landing_threshold)
                    terminate_threshold = torch.where(
                        jump_flag, jump_threshold_tensor, terminate_threshold
                    )
                    terminate_threshold = torch.where(
                        landing_jump | (jump_flag & in_grace),
                        landing_threshold_tensor,
                        terminate_threshold,
                    )
                    terminate_contact = max_force > terminate_threshold
                    leg_contact_termination = (
                        terminate_contact[jump_flag].float().mean().item() if n_jump > 0 else 0.0
                    )
                else:
                    leg_contact_termination = leg_contact_jump
            else:
                leg_contact_landing = 0.0
                leg_contact_termination = leg_contact_jump

            # 存入 extras["_leg_contact_diag"]，reset 时不会被清空
            # _update_metrics 在 reset 之后读取并搬入 extras["log"]
            env.extras["_leg_contact_diag"] = {
                "Jump/diag_leg_contact_jump": leg_contact_jump,
                "Jump/diag_leg_contact_walk": leg_contact_walk,
                "Jump/diag_leg_contact_rsi": leg_contact_rsi,
                "Jump/diag_leg_contact_landing": leg_contact_landing,
                "Jump/diag_leg_contact_termination": leg_contact_termination,
            }
        except Exception:
            pass

    if terminate:
        return terminate_contact
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def body_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """指定 body 接触地面即时终止。"""
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    force_mag = finite_contact_force_norm(data.force)
    max_force = force_mag.max(dim=1).values
    return max_force > force_threshold


class BodyContactDelayed:
    """body 接触后连续惩罚一段时间，再终止 episode。"""

    def __init__(self) -> None:
        self._contact_count: torch.Tensor | None = None
        self._active: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        force_threshold: float = 1.0,
        delay_steps: int = 50,
    ) -> torch.Tensor:
        if self._contact_count is None:
            self._contact_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        if self._active is None:
            self._active = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

        sensor: ContactSensor = env.scene[sensor_name]
        data = sensor.data
        if data.force is None:
            contact = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        else:
            force_mag = finite_contact_force_norm(data.force)
            max_force = force_mag.max(dim=1).values
            contact = max_force > force_threshold

        self._contact_count[contact] += 1
        self._contact_count[~contact] = 0
        self._contact_count[env.episode_length_buf <= 1] = 0
        self._active[:] = self._contact_count > 0
        return self._contact_count > int(delay_steps)

    def penalty(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        """返回当前 delayed contact 窗口，供 reward 持续扣分。"""
        if self._active is None:
            return torch.zeros(env.num_envs, device=env.device)
        return self._active.float()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            return
        if self._contact_count is not None:
            self._contact_count[env_ids] = 0
        if self._active is not None:
            self._active[env_ids] = False


class BodyContactGracePenalty:
    """body 接触后的 grace 窗口持续惩罚。

    这个 reward term 独立维护接触窗口状态，不依赖 termination term 的实例。
    MJLab manager 会 deepcopy cfg，跨 manager 共享 callable 实例会导致状态脱钩。
    """

    def __init__(self) -> None:
        self._contact_count: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        force_threshold: float = 1.0,
        delay_steps: int = 50,
    ) -> torch.Tensor:
        if self._contact_count is None:
            self._contact_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

        sensor: ContactSensor = env.scene[sensor_name]
        data = sensor.data
        if data.force is None:
            contact = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        else:
            force_mag = finite_contact_force_norm(data.force)
            max_force = force_mag.max(dim=1).values
            contact = max_force > force_threshold

        self._contact_count[contact] += 1
        self._contact_count[~contact] = 0
        self._contact_count[env.episode_length_buf <= 1] = 0
        in_grace = (self._contact_count > 0) & (self._contact_count <= int(delay_steps))
        return in_grace.float()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            return
        if self._contact_count is not None:
            self._contact_count[env_ids] = 0


class BodyContactPenalty:
    """body 接触地面时每一步持续扣分。"""

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        force_threshold: float = 1.0,
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene[sensor_name]
        data = sensor.data
        if data.force is None:
            return torch.zeros(env.num_envs, device=env.device)

        force_mag = finite_contact_force_norm(data.force)
        max_force = force_mag.max(dim=1).values
        return (max_force > force_threshold).float()


base_link_contact_delayed = BodyContactDelayed()
leg_contact_delayed = BodyContactDelayed()
