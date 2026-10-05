"""崎岖地形任务的指令项：在 Flat 的速度/姿态/高度指令之上按地形列改采样范围。

高度这一侧只有**地形感知抬高下限**（基类 `VelocityHeightCommandCfg` 实现）：重采样时按 env 所在列
与难度行算出这一级台阶需要的最低机身高度 `step_height + terrain_height_clearance −
body_collision_bottom_offset`，把高度指令采样区间的下界顶到该值，上界仍是 `height_range[1]`。
它按列按行静态生效、不用传感器、部署端没有额外契约（2026-09-08 用户定，取代原 step_up 前瞻状态机）。

速度这一侧：非平地列只发前向直行指令（对称随机指令下 20 s 的净位移是随机游走，官方地形课程的
位移判据推不动），台阶列再单独给高速与高站姿（6 cm 轮子靠 0.8 m/s 的动量翻不过 4 cm 立面，A8），
且台阶列 yaw 指令恒 0（A10，配合奖励侧把 yaw 工资归零）。

本文件的模块常量是这些数值的唯一来源；env_cfg 只做转发。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from se3_shared.torch_constants import device_index
from se3_train.mdp.commands import VelocityHeightCommandCfg, VelocityHeightCommandTerm
from se3_train.mdp.height_default_cache import update_policy_default_from_height_cache
from se3_train.mdp.jump_commands import JumpCommandCfg, JumpCommandTerm

from .columns import column_mask, non_flat_column_mask
from .curriculums import FLAT_WARMUP_ORIGINAL_TYPES_ATTR
from .terrains import (
    ROUGH_JUMP_COLUMN,
    ROUGH_OBSTACLE_COLUMN,
    ROUGH_RANDOM_ROUGH_COLUMN,
    ROUGH_STAIR_LIKE_COLUMNS,
    ROUGH_TWO_STEP_DOWN_COLUMN,
    ROUGH_WAVE_COLUMN,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

# 地形感知高度下限的标定，沿用 stair 线（tasks/stair/env_cfg.py）：机体碰撞网格底面在 base_link 下方
# 0.12 m（COACD 网格 z 范围 [-0.1376, 0.1118]，取平底面而非最低角点），再留 0.02 m 余量。
# 台阶 0.02→0.20 m 对应的下限是 0.20（行 0-1 不生效）→0.34 m，始终在 height_range 上界 0.38 之内。
ROUGH_TERRAIN_HEIGHT_CLEARANCE = 0.02
ROUGH_BODY_COLLISION_BOTTOM_OFFSET = -0.12
# 只在上台阶列抬高：下行列的台阶在身后，抬高只是白白升高重心。名字必须是 terrains.rough_terrains_cfg()
# 里带 step_height_range 的子地形名，对不上时下限静默失效（由测试钉住）。二级台阶列（M24）用它的**第一级**
# 高度（0.05–0.20）作为下限依据——两级里更高的那一级才安全。
ROUGH_TERRAIN_STEP_HEIGHT_TYPE_NAMES = ROUGH_STAIR_LIKE_COLUMNS

# 哪些列按"平地方式"发指令：速度跟平地课程（最终 ±2.4）、yaw 用平地范围、参与静站与高姿起步转移采样。
# 2026-09-15 用户定：下台阶与上下坡都按平地发——下台阶需要偏航跟踪，新列速度要 ±2.4 而不是只前向 0.4–0.8。
# 只有上台阶类列留在"非平地覆盖"那一路（再被下面的台阶覆盖压一层，最终是 0.4–2.4 前向、yaw ±0.3）。
# M24（用户定）：二级台阶的**下行**列按平地待遇（指令与 flat 同），只有上行列算台阶。
# 2026-10-04 试过下行列改直行指令（vx ±2.4、yaw 0、无静站 / 高姿起步、出生正对台阶 ±15°，6656fb5，W&B ctw47krj）：
# 课程等级 5.0 → 6.1 到顶，但确定性回放 r0–r9 摔倒 22 → 29/300、r6–r9 18% → 21%，没有收益，2026-10-05 用户定回退。
# 副作用：新列也会被 _sample_high_stand_transition 采到（它只在这份名单的列上采样），
# 即坡上与下台阶也会练高姿起步，这是想要的；若发现下台阶因此摔得多，先把 stairs_down 移出这份名单。
ROUGH_TERRAIN_COMMAND_FLAT_NAMES = (
    "flat",
    "stairs_down",
    ROUGH_TWO_STEP_DOWN_COLUMN,
    "slope_up",
    "slope_down",
    # M51 的三列随机地形按平地发指令（不在当前地形集里的列名会被跳过，对其他入口无影响）。
    ROUGH_RANDOM_ROUGH_COLUMN,
    ROUGH_WAVE_COLUMN,
    ROUGH_OBSTACLE_COLUMN,
    # 跳跃列按平地发指令（跳跃样本高度另行固定为参考站姿，见 RoughJumpCommandTerm）。
    ROUGH_JUMP_COLUMN,
)
# A7 留下的"非平地列前向指令"，2026-09-15 起已无列使用（stairs_up 被台阶覆盖压在上面），
# 保留是为了以后再加"需要限速的列"时有现成档位：vx 0.4–0.8 与平地课程脱钩、yaw ±0.2。
ROUGH_TERRAIN_LIN_VEL_X_RANGE = (0.4, 0.8)
ROUGH_TERRAIN_ANG_VEL_YAW_RANGE = (-0.2, 0.2)

# 上台阶列独立采样前进速度 0.4–2.4 m/s；偏航角速度 M50（2026-09-29 用户定）起恒 0，转向多样性改由出生朝向提供
# （env_cfg.ROUGH_STAIR_SPAWN_YAW_HALF_RANGE_DEG，正对台阶 ±15°），此前是 −0.3–0.3 rad/s。
# 高度 0.20–0.38 与 Flat 同区间——A13 由 sim2x 定位到 0.35–0.38 会让每个 episode 都从
# "高站姿 + 够不着的高速指令"开局而训出静止策略，矮站姿开局必须保留；第 9 行由地形感知下限自动收窄到 0.34–0.38。
ROUGH_STAIR_COMMAND_TERRAIN_NAMES = ROUGH_STAIR_LIKE_COLUMNS
ROUGH_STAIR_LIN_VEL_X_RANGE = (0.4, 2.4)
ROUGH_STAIR_ANG_VEL_YAW_RANGE = (0.0, 0.0)
ROUGH_STAIR_HEIGHT_RANGE = (0.20, 0.38)

# 台阶列逐 env 速度上限（2026-09-25 用户定做对照，参考 yly-true/fudan_rl_wheel_leg 的逐 env 指令课程）。
# 每个 env 记一个 vx 上限，台阶列按 [ROUGH_STAIR_LIN_VEL_X_RANGE[0], 上限] 采样；每个 episode 结束时按
# 速度达成率 r = Σmax(vx, 0) / Σ指令 vx 调整：r < 0.4 降 0.25、r ≥ 0.7 升 0.1，夹在 [1.0, 2.4]。
# 0.4 / 0.7 沿用复旦的降级线（跟踪分 < 40%）与指令扩张线（> 70%），降幅 0.25 与下限 1.0 同复旦。
# 复旦只在"第 0 级失败 / 最高级通关"时调速度；我们的官方升降级只看 20 s 内是否走到地块边缘
# （平均 0.225 m/s 就够），管不到速度跟踪，所以改成每个 episode 按达成率连续调。
# 初值取区间上界，开局分布与关闭时相同。默认关，实验时单独翻这个开关。
ROUGH_STAIR_SPEED_CAP_ENABLED = False
ROUGH_STAIR_SPEED_CAP_MIN = 1.0
ROUGH_STAIR_SPEED_CAP_SHRINK_BELOW = 0.4
ROUGH_STAIR_SPEED_CAP_GROW_ABOVE = 0.7
ROUGH_STAIR_SPEED_CAP_SHRINK_STEP = 0.25
ROUGH_STAIR_SPEED_CAP_GROW_STEP = 0.1
# episode 内在台阶列上不足这么久（如刚迁进来就摔）不调，样本太短的达成率只是噪声。
ROUGH_STAIR_SPEED_CAP_MIN_EPISODE_S = 1.0


@dataclass
class RoughCommandCfg(JumpCommandCfg):
    """在 JumpCommand 之上增加"按地形列限制速度指令"与"高姿静站→前进"转移的配置。

    高度指令的地形感知下限走基类 `VelocityHeightCommandCfg` 的 `terrain_aware_height` /
    `terrain_height_clearance` / `body_collision_bottom_offset` / `terrain_step_height_type_names`。
    """

    terrain_command_override_enabled: bool = True
    """是否按所在地形列限制速度指令；关掉即全部列都走 Flat 的对称随机指令。"""

    terrain_command_flat_names: tuple[str, ...] = ROUGH_TERRAIN_COMMAND_FLAT_NAMES
    """沿用 Flat 速度指令与速度课程的子地形名。"""

    terrain_lin_vel_x_range: tuple[float, float] = ROUGH_TERRAIN_LIN_VEL_X_RANGE
    """非平地列的 vx 采样范围(m/s)，下界为正保证一直朝前走。"""

    terrain_ang_vel_yaw_range: tuple[float, float] = ROUGH_TERRAIN_ANG_VEL_YAW_RANGE
    """非平地列的 yaw 角速度采样范围(rad/s)。"""

    terrain_lin_vel_x_follow_curriculum: bool = False
    """非平地列 vx 上限是否跟随平地速度课程的当前上限（R4 曾开，A7 关：平地 350 轮就冲到 2.4）。"""

    stair_command_terrain_names: tuple[str, ...] = ROUGH_STAIR_COMMAND_TERRAIN_NAMES
    """单独定价的台阶列名；空元组即这些列沿用通用地形列范围。"""

    stair_lin_vel_x_range: tuple[float, float] = ROUGH_STAIR_LIN_VEL_X_RANGE
    stair_ang_vel_yaw_range: tuple[float, float] = ROUGH_STAIR_ANG_VEL_YAW_RANGE
    stair_height_range: tuple[float, float] = ROUGH_STAIR_HEIGHT_RANGE
    """台阶列的机身高度指令范围(m)，采样下界再与地形感知下限取较大者。"""

    stair_speed_cap_enabled: bool = ROUGH_STAIR_SPEED_CAP_ENABLED
    """台阶列是否按 env 自适应 vx 上限；打开时还要注册课程项 `curriculums.stair_speed_cap`。"""
    stair_speed_cap_min: float = ROUGH_STAIR_SPEED_CAP_MIN
    stair_speed_cap_shrink_below: float = ROUGH_STAIR_SPEED_CAP_SHRINK_BELOW
    stair_speed_cap_grow_above: float = ROUGH_STAIR_SPEED_CAP_GROW_ABOVE
    stair_speed_cap_shrink_step: float = ROUGH_STAIR_SPEED_CAP_SHRINK_STEP
    stair_speed_cap_grow_step: float = ROUGH_STAIR_SPEED_CAP_GROW_STEP
    stair_speed_cap_min_episode_s: float = ROUGH_STAIR_SPEED_CAP_MIN_EPISODE_S

    high_stand_transition_prob: float = 0.0
    """平地每次重采样时生成"高姿态静站→前进"序列的概率；0 关闭。

    A20/A21（从 A15 warm-start）：A15 在 0.38 m 静站后给 0.8/1.6/2.4 m/s 只能跑到 0.05 m/s，
    加入这个转移后 A21 model_999 跑到 0.80/1.55/2.14 m/s；代价是训练内 catastrophic 终止
    从 0.03–0.08/轮升到 0.12–0.15/轮。默认关，按验收表决定是否打开。
    """

    high_stand_height_range: tuple[float, float] = (0.36, 0.38)
    high_stand_duration_range_s: tuple[float, float] = (1.5, 2.5)
    high_stand_move_vx_range: tuple[float, float] = (0.8, 2.4)

    def build(self, env: ManagerBasedRlEnv) -> RoughCommandTerm:
        return RoughCommandTerm(self, env)


class RoughCommandTerm(JumpCommandTerm):
    """按地形列限制速度指令的速度/姿态/高度指令项。"""

    cfg: RoughCommandCfg

    def __init__(self, cfg: RoughCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        # 非平地列的 env 掩码；None 表示没有可用的分列地形或覆盖未启用。
        self._terrain_override_mask: torch.Tensor | None = None
        # 单独定价的台阶列掩码，是上面那个的子集。
        self._stair_mask: torch.Tensor | None = None
        self._high_stand_selected = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._high_stand_steps_left = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._high_stand_target_vx = torch.zeros(
            self.num_envs, device=self.device, dtype=self._command.dtype
        )
        # 台阶列逐 env vx 上限与本 episode 的速度累计（stair_speed_cap_enabled 时才用）。
        self._stair_speed_cap = torch.full(
            (self.num_envs,),
            float(cfg.stair_lin_vel_x_range[1]),
            device=self.device,
            dtype=self._command.dtype,
        )
        self._stair_vx_sum = torch.zeros(
            self.num_envs, device=self.device, dtype=self._command.dtype
        )
        self._stair_cmd_sum = torch.zeros_like(self._stair_vx_sum)
        self._stair_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        if cfg.terrain_command_override_enabled:
            self.refresh_terrain_override()

    def refresh_terrain_override(self) -> None:
        """按当前 terrain_types 重算列掩码并刷新逐 env 的速度范围覆盖。

        env 换列（平地热身结束、课程升降级）后必须调用，否则覆盖还按旧列生效。
        """
        if not self.cfg.terrain_command_override_enabled:
            return
        self._terrain_override_mask = non_flat_column_mask(
            self._env, self.cfg.terrain_command_flat_names
        )
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
        # 台阶列的覆盖压在通用地形覆盖之上，必须后设。
        self._stair_mask = column_mask(self._env, self.cfg.stair_command_terrain_names)
        if self._stair_mask is not None and bool(self._stair_mask.any()):
            ids = self._stair_mask.nonzero(as_tuple=False).flatten()
            self.set_velocity_ranges(
                ids,
                lin_vel_x_range=tuple(self.cfg.stair_lin_vel_x_range),
                ang_vel_yaw_range=tuple(self.cfg.stair_ang_vel_yaw_range),
            )
            if self.cfg.stair_speed_cap_enabled:
                assert self._lin_vel_x_range_override is not None
                low = float(self.cfg.stair_lin_vel_x_range[0])
                self._lin_vel_x_range_override[ids, 1] = torch.clamp(
                    self._stair_speed_cap[ids], min=low
                )

    def update_stair_speed_caps(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """按刚结束 episode 的台阶速度达成率调整这些 env 的 vx 上限，写回采样范围并清空累计。

        由课程项 `curriculums.stair_speed_cap` 在 reset 时调用：此时下一 episode 的指令还没采样
        （reset 事件里的预采样也在课程之后），新上限当场生效。只调 episode 内在台阶列上待够
        `stair_speed_cap_min_episode_s` 的 env；全程用掩码不做布尔索引，避免每步 reset 引入主机同步。
        """
        cfg = self.cfg
        cap_max = float(cfg.stair_lin_vel_x_range[1])
        cap_min = min(float(cfg.stair_speed_cap_min), cap_max)
        min_steps = max(1, math.ceil(float(cfg.stair_speed_cap_min_episode_s) / self._env.step_dt))

        ids = env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        counted = self._stair_steps[ids] >= min_steps
        ratio = self._stair_vx_sum[ids] / self._stair_cmd_sum[ids].clamp(min=1e-6)
        shrink = counted & (ratio < float(cfg.stair_speed_cap_shrink_below))
        grow = counted & (ratio >= float(cfg.stair_speed_cap_grow_above))
        cap = self._stair_speed_cap[ids]
        cap = torch.where(
            shrink, torch.clamp(cap - float(cfg.stair_speed_cap_shrink_step), min=cap_min), cap
        )
        cap = torch.where(
            grow, torch.clamp(cap + float(cfg.stair_speed_cap_grow_step), max=cap_max), cap
        )
        self._stair_speed_cap[ids] = cap
        self._stair_vx_sum[ids] = 0.0
        self._stair_cmd_sum[ids] = 0.0
        self._stair_steps[ids] = 0

        zero = torch.zeros((), device=self.device)
        if self._stair_mask is None or self._lin_vel_x_range_override is None:
            return {"cap_mean": zero, "ratio_mean": zero}
        on_stairs = self._stair_mask[ids]
        low = float(cfg.stair_lin_vel_x_range[0])
        self._lin_vel_x_range_override[ids, 1] = torch.where(
            on_stairs, torch.clamp(cap, min=low), self._lin_vel_x_range_override[ids, 1]
        )

        stairs = self._stair_mask.float()
        n_stairs = stairs.sum().clamp(min=1.0)
        n_counted = counted.float().sum().clamp(min=1.0)
        return {
            "cap_mean": (self._stair_speed_cap * stairs).sum() / n_stairs,
            "cap_at_min": ((self._stair_speed_cap <= cap_min + 1e-6).float() * stairs).sum()
            / n_stairs,
            "ratio_mean": (ratio.clamp(max=2.0) * counted.float()).sum() / n_counted,
            "shrink_rate": shrink.float().sum() / n_counted,
            "grow_rate": grow.float().sum() / n_counted,
        }

    def _update_command(self) -> None:
        super()._update_command()
        self._update_high_stand_transition()
        if self.cfg.stair_speed_cap_enabled and self._stair_mask is not None:
            # 台阶列逐步累计实速与指令，episode 结束时由 update_stair_speed_caps 结算。
            on_stairs = self._stair_mask
            base_vx = self._env.scene["robot"].data.root_link_lin_vel_b[:, 0]
            self._stair_vx_sum += torch.clamp(base_vx, min=0.0) * on_stairs
            self._stair_cmd_sum += self._command[:, 0] * on_stairs
            self._stair_steps += on_stairs.long()
        # 地形感知下限只在重采样时抬高采样下界，没有逐步状态；记一笔均值，否则 W&B 上看不出它有没有顶起来。
        if self._terrain_override_mask is None:
            return
        height_cmd = self._command[:, 4]
        terrain = self._terrain_override_mask.float()
        log = self._env.extras.setdefault("log", {})
        log["Rough/height_cmd_terrain_mean"] = (height_cmd * terrain).sum() / terrain.sum().clamp(
            min=1.0
        )

    def _update_high_stand_transition(self) -> None:
        """保持高姿态静站一段时间，再在不改变高度的情况下切换到前进指令。"""
        if self.cfg.high_stand_transition_prob <= 0.0:
            return

        # 全部用掩码运算写回（与布尔索引写入逐位相同），避免每步 nonzero 同步。
        waiting = self._high_stand_selected & (self._high_stand_steps_left > 0)
        self._high_stand_steps_left.sub_(waiting.to(self._high_stand_steps_left.dtype))
        start_moving = waiting & (self._high_stand_steps_left == 0)
        self._command[:, 0] = torch.where(
            start_moving, self._high_stand_target_vx, self._command[:, 0]
        )
        self._command[:, 1:4].masked_fill_(start_moving.unsqueeze(1), 0.0)
        self._standing_mask.masked_fill_(start_moving, False)

        waiting = self._high_stand_selected & (self._high_stand_steps_left > 0)
        moving = self._high_stand_selected & ~waiting
        log = self._env.extras.setdefault("log", {})
        log["Rough/high_stand_transition_waiting"] = waiting.float().mean()
        log["Rough/high_stand_transition_moving"] = moving.float().mean()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self._high_stand_selected.index_fill_(0, env_ids, False)
        self._high_stand_steps_left.index_fill_(0, env_ids, 0)
        self._high_stand_target_vx.index_fill_(0, env_ids, 0.0)
        if self._terrain_override_mask is None:
            super()._resample_command(env_ids)
            self._sample_high_stand_transition(env_ids)
            return
        # 非平地列不抽静站样本：vx 下界为正的约束对静站（vx=0）没有意义。
        overridden = self._terrain_override_mask[env_ids]
        flat_ids = env_ids[~overridden]
        terrain_ids = env_ids[overridden]
        if flat_ids.numel() > 0:
            super()._resample_command(flat_ids)
            self._sample_high_stand_transition(flat_ids)
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
        self._apply_stair_height(env_ids)

    def _sample_high_stand_transition(self, flat_ids: torch.Tensor) -> None:
        """在平地样本中注入部署时会遇到的高姿态冷启动指令跳变。"""
        probability = min(max(float(self.cfg.high_stand_transition_prob), 0.0), 1.0)
        if probability <= 0.0 or flat_ids.numel() == 0:
            return
        selected = torch.rand(len(flat_ids), device=self.device) < probability
        ids = flat_ids[selected]
        if ids.numel() == 0:
            return

        height_low, height_high = (float(v) for v in self.cfg.high_stand_height_range)
        duration_low, duration_high = (float(v) for v in self.cfg.high_stand_duration_range_s)
        vx_low, vx_high = (float(v) for v in self.cfg.high_stand_move_vx_range)
        self._command[:, 0:4].index_fill_(0, ids, 0.0)
        self._command[ids, 4] = (
            torch.rand(len(ids), device=self.device) * (height_high - height_low) + height_low
        )
        duration_s = (
            torch.rand(len(ids), device=self.device) * (duration_high - duration_low) + duration_low
        )
        self._high_stand_steps_left[ids] = torch.ceil(duration_s / self._env.step_dt).long()
        self._high_stand_target_vx[ids] = (
            torch.rand(len(ids), device=self.device) * (vx_high - vx_low) + vx_low
        )
        self._high_stand_selected.index_fill_(0, ids, True)
        self._standing_mask.index_fill_(0, ids, True)
        update_policy_default_from_height_cache(
            self._env,
            "velocity_height",
            env_ids=ids,
            command=self._command,
        )

    def _apply_stair_height(self, env_ids: torch.Tensor) -> None:
        """把台阶列 env 的高度指令改到 `stair_height_range` 内重新采样。

        写在基类采样之后而不是改基类：基类那一路还要管静站/运动两个区间与 jump 生命周期。
        采样下界取 `stair_height_range[0]` 与地形感知下限的较大者——本方法覆盖的是基类的采样结果，
        而地形感知下限正是基类算的，直接覆盖会把它整个盖掉（第 9 行台阶 0.20 m 时抽到 0.20 的
        高度指令，机体碰撞盒底面 0.08 m 低于台阶顶面，机身直接撞立面）。改完必须同步刷新高度条件
        默认腿姿缓存，否则奖励侧用的还是旧高度对应的默认姿态。
        """
        if self._stair_mask is None:
            return
        ids = env_ids[self._stair_mask[env_ids]]
        if ids.numel() == 0:
            return
        low, high = (float(v) for v in self.cfg.stair_height_range)
        lower = torch.full((len(ids),), low, device=self.device, dtype=self._command.dtype)
        if self.cfg.terrain_aware_height and self.cfg.terrain_height_clearance > 0.0:
            lower = torch.maximum(lower, self._terrain_aware_min_height(ids, lower))
        # 下限顶到上界时退化成定值指令，不能让区间变负。
        lower = torch.clamp(lower, max=high)
        span = high - lower
        self._command[ids, 4] = torch.rand(len(ids), device=self.device) * span + lower
        update_policy_default_from_height_cache(
            self._env,
            "velocity_height",
            env_ids=ids,
            command=self._command,
        )


# ---------------------------------------------------------------- 跳跃合入（RJ1，2026-10-02 用户定）
ROUGH_JUMP_COLUMN_NAMES: tuple[str, ...] = (ROUGH_JUMP_COLUMN,)
"""跳跃样本所在的子地形列：2026-10-03 起为专用平地跳跃列（占全部 env 的 terrains.ROUGH_JUMP_COLUMN_PROPORTION），
此前是平地列。按 env 的原始列归属判定（热身期全体在平地列时跳跃 env 照常跳）。"""
ROUGH_JUMP_ENV_FRACTION = 1.0
"""跳跃样本占跳跃列 env 的比例：专用列全部是跳跃样本（此前平地列 30%）。"""
ROUGH_JUMP_TRIGGER_RATE_HZ = 0.5
"""跳跃样本站满 min_idle_s 后每秒触发概率：2026-10-03 由 0.2 提到 J10 验证过的 0.5（每次循环约 4.5 s、约 1/3 时间在跳）。"""
ROUGH_JUMP_MAX_LIN_VEL_X: float | None = None
"""触发时 vx 指令夹到 ±该值；None 不夹。2026-10-04（用户定）由 1.5（J10 训练范围）改为不夹：跳跃样本等待期 vx 跟随平地课程到 ±2.4，
夹到 ±1.5 会把约 1/3 跳跃堆在 ±1.5 上并让高速 env 起跳时被迫减速；部署运行时（se3_runtime）本来就不夹。"""


@dataclass
class RoughJumpCommandCfg(RoughCommandCfg):
    """rough 指令 + J10 跳跃参考时钟。

    旧 JumpCommandTerm 的跳跃生命周期（原地跳、旧轨迹、jump_phase 写成 0→1 相位）必须关掉
    （enable_jump_lifecycle=False），否则它会把 jump_flag=1 的 env 当成旧跳跃推进并改写 [5:8]。
    """

    reference_paths: tuple[str, ...] = ()
    phase_time_scale_s: float | None = None
    min_idle_s: float = 1.0
    trigger_rate_hz: float = ROUGH_JUMP_TRIGGER_RATE_HZ
    jump_column_names: tuple[str, ...] = ROUGH_JUMP_COLUMN_NAMES
    jump_env_fraction: float = ROUGH_JUMP_ENV_FRACTION
    jump_max_lin_vel_x: float | None = ROUGH_JUMP_MAX_LIN_VEL_X

    def build(self, env: ManagerBasedRlEnv) -> RoughJumpCommandTerm:
        if self.enable_jump_lifecycle:
            raise ValueError(
                "RoughJumpCommandCfg 必须关掉旧跳跃生命周期（enable_jump_lifecycle=False）"
            )
        if not self.reference_paths:
            raise ValueError("RoughJumpCommandCfg 需要跳跃参考 reference_paths")
        return RoughJumpCommandTerm(self, env)


class RoughJumpCommandTerm(RoughCommandTerm):
    """rough 指令项上叠加跳跃样本与参考时钟（与 J10 同一时钟，见 jump_mimic.clock）。

    每回合 reset 时，在允许跳跃的列上按 jump_env_fraction 抽"跳跃样本"：整回合高度指令固定为参考站姿
    （与 J10 一样，用户定）、不参加高姿态起步序列，只有它们会触发跳跃。跳跃期间冻结速度 / 姿态 / 高度指令，
    触发时 yaw / pitch / roll 置 0，vx 保持当前指令（jump_max_lin_vel_x 非 None 时才夹到 ±该值）。其余 env 与 M54 完全一样。
    """

    cfg: RoughJumpCommandCfg

    def __init__(self, cfg: RoughJumpCommandCfg, env: ManagerBasedRlEnv):
        from se3_train.tasks.jump_mimic.clock import JumpReferenceClock
        from se3_train.tasks.jump_mimic.reference import JumpReferenceLibrary

        self.jump_env = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.clock = JumpReferenceClock(
            JumpReferenceLibrary(cfg.reference_paths, env.device),
            env.num_envs,
            env.device,
            min_idle_s=cfg.min_idle_s,
            trigger_rate_hz=cfg.trigger_rate_hz,
            phase_time_scale_s=cfg.phase_time_scale_s,
        )
        super().__init__(cfg, env)

    # ---- 时钟状态（jump_mimic.mdp 的模仿奖励 / 偏离终止读取） ----
    @property
    def library(self):
        return self.clock.library

    @property
    def active(self) -> torch.Tensor:
        return self.clock.active

    @property
    def ref_id(self) -> torch.Tensor:
        return self.clock.ref_id

    @property
    def ref_t(self) -> torch.Tensor:
        return self.clock.ref_t

    def reference(self, offset_s: float = 0.0):
        return self.clock.reference(offset_s)

    # ---- 生命周期 ----
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if bool(getattr(self, "_resampling_for_reset", False)):
            # reset 事件里的预采样早于 command_manager.reset：不先清时钟，下面 write_dims 会把跳跃中途被终止的
            # env 写成 jump_flag=1，reset 事件随即把它当旧跳跃线 RSI 样本，从旧参考轨迹注入机身与关节状态
            # （2026-10-03 定位，见 docs/plan/rough_iteration_time_20261003.md）。新 episode 不可能在跳，先清。
            self.clock.reset(env_ids)
            self._sample_jump_envs(env_ids)
            super()._resample_command(env_ids)
        else:
            keep = env_ids[self.active[env_ids]]
            saved = self._command[keep, 0:5].clone()
            saved_standing = self._standing_mask[keep].clone()
            super()._resample_command(env_ids)
            self._command[keep, 0:5] = saved
            self._standing_mask[keep] = saved_standing
        self._fix_jump_env_height(env_ids)
        self.clock.write_dims(self._command)

    def _jump_column_mask(self) -> torch.Tensor | None:
        """按原始列归属判定跳跃列：平地热身期全体被挪到平地列，此时跳跃列 env 仍是跳跃样本（平地上跳，参考成立）。"""
        mask = column_mask(self._env, self.cfg.jump_column_names)
        original = getattr(self._env, FLAT_WARMUP_ORIGINAL_TYPES_ATTR, None)
        if mask is None or original is None:
            return mask
        names = list(self._env.scene.terrain.cfg.terrain_generator.sub_terrains.keys())
        cols = [names.index(name) for name in self.cfg.jump_column_names if name in names]
        return torch.isin(original.to(torch.long), device_index(cols, device=original.device))

    def _sample_jump_envs(self, env_ids: torch.Tensor) -> None:
        mask = self._jump_column_mask()
        on_column = (
            torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
            if mask is None
            else mask[env_ids]
        )
        draw = torch.rand(len(env_ids), device=self.device) < float(self.cfg.jump_env_fraction)
        self.jump_env[env_ids] = on_column & draw

    def _fix_jump_env_height(self, env_ids: torch.Tensor) -> None:
        """跳跃样本：高度指令固定为参考站姿，退出高姿态起步序列。"""
        ids = env_ids[self.jump_env[env_ids]]
        if ids.numel() == 0:
            return
        self._high_stand_selected.index_fill_(0, ids, False)
        self._high_stand_steps_left.index_fill_(0, ids, 0)
        self._command[:, 4].index_fill_(0, ids, float(self.library.stand_height))
        update_policy_default_from_height_cache(
            self._env,
            "velocity_height",
            env_ids=ids,
            command=self._command,
        )

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, torch.Tensor]:
        extras = super().reset(env_ids)
        assert isinstance(env_ids, torch.Tensor)
        self.clock.reset(env_ids)
        self.clock.write_dims(self._command)
        return extras

    def _update_command(self) -> None:
        super()._update_command()
        fired = self.clock.step(float(self._env.step_dt), allowed=self.jump_env)
        if fired.numel() > 0:
            if self.cfg.jump_max_lin_vel_x is not None:
                vmax = float(self.cfg.jump_max_lin_vel_x)
                self._command[fired, 0] = self._command[fired, 0].clamp(-vmax, vmax)
            self._command[:, 1:4].index_fill_(0, fired, 0.0)
        self.clock.write_dims(self._command)

    def deployment_jump_reference(self) -> dict:
        """部署契约顶层 jump_reference（同 J10）。"""
        from se3_train.tasks.jump_mimic.mdp import DEFAULT_OFFSETS_STEPS, REFERENCE_SCALES

        return self.clock.deployment_payload(DEFAULT_OFFSETS_STEPS, REFERENCE_SCALES)

    def _update_metrics(self) -> None:
        super()._update_metrics()
        log = self._env.extras.setdefault("log", {}) if hasattr(self._env, "extras") else None
        if isinstance(log, dict):
            log["Jump/active_rate"] = self.active.float().mean()
            log["Jump/jump_env_rate"] = self.jump_env.float().mean()


__all__ = [
    "ROUGH_BODY_COLLISION_BOTTOM_OFFSET",
    "ROUGH_JUMP_COLUMN_NAMES",
    "ROUGH_JUMP_ENV_FRACTION",
    "ROUGH_JUMP_MAX_LIN_VEL_X",
    "ROUGH_JUMP_TRIGGER_RATE_HZ",
    "ROUGH_STAIR_ANG_VEL_YAW_RANGE",
    "ROUGH_STAIR_COMMAND_TERRAIN_NAMES",
    "ROUGH_STAIR_HEIGHT_RANGE",
    "ROUGH_STAIR_LIN_VEL_X_RANGE",
    "ROUGH_STAIR_SPEED_CAP_ENABLED",
    "ROUGH_STAIR_SPEED_CAP_GROW_ABOVE",
    "ROUGH_STAIR_SPEED_CAP_GROW_STEP",
    "ROUGH_STAIR_SPEED_CAP_MIN",
    "ROUGH_STAIR_SPEED_CAP_MIN_EPISODE_S",
    "ROUGH_STAIR_SPEED_CAP_SHRINK_BELOW",
    "ROUGH_STAIR_SPEED_CAP_SHRINK_STEP",
    "ROUGH_TERRAIN_ANG_VEL_YAW_RANGE",
    "ROUGH_TERRAIN_COMMAND_FLAT_NAMES",
    "ROUGH_TERRAIN_HEIGHT_CLEARANCE",
    "ROUGH_TERRAIN_LIN_VEL_X_RANGE",
    "ROUGH_TERRAIN_STEP_HEIGHT_TYPE_NAMES",
    "JumpCommandCfg",
    "RoughCommandCfg",
    "RoughCommandTerm",
    "RoughJumpCommandCfg",
    "RoughJumpCommandTerm",
    "VelocityHeightCommandCfg",
    "VelocityHeightCommandTerm",
]
