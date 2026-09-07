"""崎岖地形任务的指令项：在速度/高度指令之上叠加台阶前瞻辅助（step_up 状态机）。

移植自 scutrobotlab/wheeled-legged_RL 的 `state_machines/step_up.py`（其 rough 线
`WheelbipeV14RoughEnvCfg_v1` 打开该状态机）。原理是策略本身不学“上台阶”这个新技能，
只需要会跟随高度指令；由状态机在检测到前方台阶时替它把腿伸长，把台阶问题降级成
已有能力的自动触发。

与参考实现的三处已知差异，都是 MJLab 传感器 API 的约束造成的：

1. **空间差分而非时间差分。** 参考仓库对轮前方固定偏移处每步打一次向下的射线，比较本步与
   上一步的地面高度。本实现让同一个传感器同时打“身下”和“身前”两条射线，取
   `身下净空 − 身前净空`，等于 `前方地面高度 − 身下地面高度`，与机身自身高度无关
   （两条射线共用同一个 `frame_z`，相减即抵消）。判据等价，但不需要跨步状态、不需要处理
   行进方向翻转与 reset 清理。

2. **射线挂在 base_link 而不是轮子上。** MJLab 的 `GridPatternCfg` 只在 frame 所在平面
   铺射线（局部 z 偏移恒为 0），而参考仓库是把射线起点抬高 5 m 再往下打。射线起点若落在
   几何体内部，MJLab 判为 backface 并把净空钳到 0，所以挂在轮心（离地 0.06 m）的探针
   量不出高于轮半径的台阶。base_link 是本模型最高的可用 frame。
   由此得到一条有用的性质：前方障碍高过机身时 `身前净空` 恒为 0，`rise` 读数正好等于
   机身离地高度；把 `step_up_wall_height` 卡在地形最高一级台阶之上、机身高度指令下界附近，
   墙（地形外围 border）就会稳定判为墙，而真台阶最多读到自己的高度，仍判为台阶。

3. **墙不再屏蔽同帧的失败终止。** 参考实现是 `terminate &= ~wall`；MJLab 的
   TerminationManager 按 term 独立累加 terminated/truncated 两个 buffer，没有这个钩子。
   墙是在 0.5 m 之外提前判定的，实测不会与摔倒类终止同帧触发，故只保留“墙 → time_out”。

另外，传感器在 `sim.sense()` 里更新，而 `sim.sense()` 排在 `command_manager.compute()`
之后，所以本状态机读到的地形高度恒落后一个控制步（20 ms）。2 m/s 时对应 4 cm，
远小于 0.5 m 的前瞻距离。

部署契约提醒：状态机会在训练时把高度指令顶高，部署端若不复现同一套逻辑，策略在台阶前
拿到的就是另一个高度指令。要么上层控制器复现该状态机，要么把它当成训练期的课程手段，
在评测时关掉（`step_up_enabled=False` 即逐位退化为 JumpCommandTerm）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from se3_train.mdp.commands import VelocityHeightCommandCfg, VelocityHeightCommandTerm
from se3_train.mdp.height_default_cache import update_policy_default_from_height_cache
from se3_train.mdp.jump_commands import JumpCommandCfg, JumpCommandTerm

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

# 前向探针把三条射线排成 [-offset, 0, +offset]（见 mjlab GridPatternCfg.generate_rays），
# 故索引固定为：0 后方、1 身下、2 前方。
_RAY_BACKWARD = 0
_RAY_UNDER = 1
_RAY_FORWARD = 2

# env 上挂的墙判定标志属性名；终止条件按该标志把该 env 转成 time_out。
WALL_BLOCKED_ATTR = "_se3_rough_wall_blocked"


@dataclass
class StepUpCommandCfg(JumpCommandCfg):
    """在 JumpCommand 之上增加台阶前瞻辅助的配置。

    `step_up_enabled=False` 时本类退化为 JumpCommandTerm，逐位等价。
    """

    step_up_enabled: bool = False
    """是否启用台阶前瞻辅助。"""

    step_up_sensor_name: str = "wheel_forward_sensor"
    """前向地形高度传感器名，reduction 必须为 none 且每帧 3 条射线。"""

    step_up_height_min: float = 0.06
    """判为可跨台阶的最小前方抬升(m)。低于该值视为地面起伏，不触发。"""

    step_up_height_max: float = 0.22
    """判为可跨台阶的最大前方抬升(m)。"""

    step_up_wall_height: float = 0.22
    """判为墙的前方抬升阈值(m)。墙优先于台阶，且不算策略失败。

    取值应略高于地形最高一级台阶（见 terrains._STEP_HEIGHT_RANGE），
    使真台阶永远落在台阶窗口内，只有 border 那种高过机身的障碍才判为墙。
    """

    step_up_height_bias: float = 0.10
    """检测到台阶后在当前高度指令上叠加的抬升量(m)。"""

    step_up_hold_s: float = 2.0
    """一次触发后高度指令保持抬升的时长(s)。"""

    step_up_height_max_cmd: float | None = None
    """抬升后的高度指令上限(m)；None 时取 `height_range` 上界。"""

    terrain_command_override_enabled: bool = False
    """是否按所在地形列限制速度指令。

    开启后，`terrain_command_flat_names` 以外的列（台阶、斜坡、起伏）只发前向直行指令：
    vx 在 `terrain_lin_vel_x_range` 内、yaw 在 `terrain_ang_vel_yaw_range` 内采样，且不抽静站样本；
    平地列不受影响，仍走 Flat 的速度课程。目的是让机器人正对台阶直冲，而不是在台阶前
    转圈或倒车（对称随机指令下净位移是随机游走，地形课程无法推进，见 curriculums.py）。
    """

    terrain_command_flat_names: tuple[str, ...] = ("flat",)
    """沿用 Flat 速度指令的子地形名（课程模式下列号即子地形名的序号）。"""

    terrain_lin_vel_x_range: tuple[float, float] = (0.4, 2.4)
    """非平地列的 vx 采样范围(m/s)。下界为正，保证一直朝前走。"""

    terrain_ang_vel_yaw_range: tuple[float, float] = (-0.2, 0.2)
    """非平地列的 yaw 角速度采样范围(rad/s)。"""

    terrain_lin_vel_x_follow_curriculum: bool = True
    """非平地列 vx 上限是否跟随平地速度课程的当前上限（`cfg.lin_vel_x_range[1]`）。

    开启时每次重采样取 min(terrain_lin_vel_x_range[1], 当前课程上限)，且不低于下界；
    课程起点 0 时地形列拿到的就是下界 0.4 m/s 的定速指令，随课程一起爬到 2.4。
    R3 从第 0 轮就给 0.4–2.4，500 轮的策略对 vx ≥ 1.0 的指令原地不动。
    """

    def build(self, env: ManagerBasedRlEnv) -> StepUpCommandTerm:
        return StepUpCommandTerm(self, env)


class StepUpCommandTerm(JumpCommandTerm):
    """带台阶前瞻辅助的速度/姿态/高度指令项。"""

    cfg: StepUpCommandCfg

    def __init__(self, cfg: StepUpCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        zeros = torch.zeros(self.num_envs, device=self.device)
        # 采样得到的原始高度指令。抬升是叠加在它之上的覆盖层，不写回它，
        # 否则每步都在上一次抬升的结果上再加一次 bias，两三步就顶到上限。
        self._base_height_cmd = self._command[:, 4].clone()
        self._hold_remaining_s = zeros.clone()
        self._hold_value = zeros.clone()
        self._step_detect = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._wall_blocked = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # 终止条件通过 env 上的属性读取墙标志，避免终止函数依赖 command term 的内部结构。
        setattr(env, WALL_BLOCKED_ATTR, self._wall_blocked)
        self._max_cmd = (
            float(cfg.step_up_height_max_cmd)
            if cfg.step_up_height_max_cmd is not None
            else float(cfg.height_range[1])
        )
        # 非平地列的 env 掩码；None 表示没有可用的分列地形或覆盖未启用。
        self._terrain_override_mask: torch.Tensor | None = None
        if cfg.terrain_command_override_enabled:
            self.refresh_terrain_override()

    def refresh_terrain_override(self) -> None:
        """按当前 terrain_types 重算“非平地列”掩码并刷新逐 env 的速度范围覆盖。

        env 换列（平地热身结束、课程重掷）后必须调用，否则覆盖还按旧列生效。
        """
        if not self.cfg.terrain_command_override_enabled:
            return
        self._terrain_override_mask = self._build_terrain_override_mask(self._env)
        if self._terrain_override_mask is None:
            return
        all_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        # 先把所有 env 的覆盖清掉（回到 cfg 范围），再给非平地列重新设。
        self.set_velocity_ranges(
            all_ids,
            lin_vel_x_range=tuple(self.cfg.lin_vel_x_range),
            ang_vel_yaw_range=tuple(self.cfg.ang_vel_yaw_range),
        )
        assert self._velocity_range_override_mask is not None
        self._velocity_range_override_mask[:] = False
        if bool(self._terrain_override_mask.any()):
            ids = self._terrain_override_mask.nonzero(as_tuple=False).flatten()
            self.set_velocity_ranges(
                ids,
                lin_vel_x_range=tuple(self.cfg.terrain_lin_vel_x_range),
                ang_vel_yaw_range=tuple(self.cfg.terrain_ang_vel_yaw_range),
            )

    def _build_terrain_override_mask(self, env: ManagerBasedRlEnv) -> torch.Tensor | None:
        """返回“不在平地列”的 env 掩码。

        只在课程模式（每种子地形独占一列）下有定义：`terrain_types` 即列号，列号对应
        `sub_terrains` 的键序。非课程模式或平面地形时返回 None，覆盖静默关闭。
        """
        terrain = getattr(env.scene, "terrain", None)
        generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
        terrain_types = getattr(terrain, "terrain_types", None)
        terrain_origins = getattr(terrain, "terrain_origins", None)
        if generator is None or terrain_types is None or terrain_origins is None:
            return None
        names = list(generator.sub_terrains.keys())
        if not generator.curriculum or terrain_origins.shape[1] != len(names):
            return None
        flat_cols = [i for i, name in enumerate(names) if name in self.cfg.terrain_command_flat_names]
        types = terrain_types.to(device=self.device, dtype=torch.long)
        is_flat = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        for col in flat_cols:
            is_flat |= types == col
        return ~is_flat

    def _forward_rise(self) -> torch.Tensor | None:
        """返回每个 env 前方地面相对身下地面的抬升(m)，形状 [B, F]。

        行进方向为负时改用后方射线，与参考仓库按 vx 指令取扫描方向一致。
        """
        sensor = self._env.scene.sensors.get(self.cfg.step_up_sensor_name)
        if sensor is None:
            return None
        try:
            heights = sensor.data.heights
        except (AttributeError, AssertionError, RuntimeError):
            # 第一次 sim.sense() 之前传感器缓冲区还没建立。
            return None
        if heights is None or heights.ndim != 3 or heights.shape[-1] < 3:
            return None
        under = heights[..., _RAY_UNDER]
        ahead = heights[..., _RAY_FORWARD]
        behind = heights[..., _RAY_BACKWARD]
        # heights 是净空（frame_z - hit_z），故 under - probe = probe 处地面 - 身下地面。
        rise_forward = under - ahead
        rise_backward = under - behind
        reversing = (self._command[:, 0] < 0.0).unsqueeze(-1)
        rise = torch.where(reversing, rise_backward, rise_forward)
        # 身下射线打空（max_distance 之内没有地面）时 under 会被填成 max_distance，
        # 直接相减会得到一个假的巨大抬升。这种帧一律判为无效，不触发台阶也不触发墙。
        max_distance = float(getattr(sensor.cfg, "max_distance", float("inf")))
        rise = torch.where(under >= max_distance - 1e-3, torch.zeros_like(rise), rise)
        return torch.nan_to_num(rise, nan=0.0, posinf=0.0, neginf=0.0)

    def _update_command(self) -> None:
        super()._update_command()
        if not self.cfg.step_up_enabled:
            return

        self._hold_remaining_s.sub_(self._env.step_dt).clamp_(min=0.0)
        self._step_detect.zero_()
        self._wall_blocked.zero_()

        rise = self._forward_rise()
        if rise is not None:
            wall = torch.any(rise > self.cfg.step_up_wall_height, dim=1)
            step = (
                torch.any(
                    (rise > self.cfg.step_up_height_min) & (rise < self.cfg.step_up_height_max),
                    dim=1,
                )
                & ~wall
            )
            self._wall_blocked.copy_(wall)
            self._step_detect.copy_(step)
            # 撞墙时撤销已保持的抬升，避免下一次 reset 前继续顶着高指令。
            self._hold_remaining_s[wall] = 0.0
            self._hold_value[wall] = 0.0
            if self.cfg.step_up_hold_s > 0.0 and bool(step.any()):
                self._hold_value[step] = torch.clamp(
                    self._base_height_cmd[step] + self.cfg.step_up_height_bias,
                    max=self._max_cmd,
                )
                self._hold_remaining_s[step] = self.cfg.step_up_hold_s

        # 覆盖层每步从原始指令重算：保持期内只抬高不压低，保持结束立即回到原始指令。
        holding = self._hold_remaining_s > 0.0
        effective = torch.where(
            holding,
            torch.maximum(self._base_height_cmd, self._hold_value),
            self._base_height_cmd,
        )
        changed = (effective != self._command[:, 4]).nonzero(as_tuple=False).flatten()
        self._command[:, 4] = effective
        if changed.numel() > 0:
            # 奖励侧的默认腿姿随高度指令变化，抬升生效/失效时必须同步刷新缓存。
            update_policy_default_from_height_cache(
                self._env,
                "velocity_height",
                env_ids=changed,
                command=self._command,
            )

        log = self._env.extras.setdefault("log", {})
        log["Rough/step_up_hold_rate"] = holding.float().mean()
        log["Rough/step_up_detect_rate"] = self._step_detect.float().mean()
        log["Rough/wall_blocked_rate"] = self._wall_blocked.float().mean()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if self._terrain_override_mask is None:
            super()._resample_command(env_ids)
        else:
            # 非平地列不抽静站样本：vx 下界为正的约束对静站（vx=0）没有意义。
            overridden = self._terrain_override_mask[env_ids]
            flat_ids = env_ids[~overridden]
            terrain_ids = env_ids[overridden]
            if flat_ids.numel() > 0:
                super()._resample_command(flat_ids)
            if terrain_ids.numel() > 0:
                if self.cfg.terrain_lin_vel_x_follow_curriculum:
                    lo, hi = (float(v) for v in self.cfg.terrain_lin_vel_x_range)
                    hi = max(lo, min(hi, float(self.cfg.lin_vel_x_range[1])))
                    self.set_velocity_ranges(
                        terrain_ids,
                        lin_vel_x_range=(lo, hi),
                        ang_vel_yaw_range=tuple(self.cfg.terrain_ang_vel_yaw_range),
                    )
                standing_ratio = self.cfg.standing_ratio
                self.cfg.standing_ratio = 0.0
                try:
                    super()._resample_command(terrain_ids)
                finally:
                    self.cfg.standing_ratio = standing_ratio
        self._base_height_cmd[env_ids] = self._command[env_ids, 4]
        if not self.cfg.step_up_enabled:
            return
        self._hold_remaining_s[env_ids] = 0.0
        self._hold_value[env_ids] = 0.0
        self._step_detect[env_ids] = False
        self._wall_blocked[env_ids] = False


__all__ = [
    "WALL_BLOCKED_ATTR",
    "JumpCommandCfg",
    "StepUpCommandCfg",
    "StepUpCommandTerm",
    "VelocityHeightCommandCfg",
    "VelocityHeightCommandTerm",
]
