"""崎岖地形任务的指令项：按所在地形列限制速度指令。

高度指令这一侧不做任何前瞻：台阶前的抬升由 `mdp/commands.py` 的**地形感知抬高下限**
负责——重采样时按 env 当前所在的地形列与难度行算出这一级台阶需要的最低机身高度
（`step_height + terrain_height_clearance − body_collision_bottom_offset`），
把采样区间的下界顶到那个值，上界仍是 `height_range[1]`。
数值在 rough/env_cfg.py 里配，与 stair 线同一套标定。

2026-09-08 用户定：删掉原来的 step_up 前瞻状态机（身前 0.5 m 探到 0.06–0.22 m 抬升就把
高度指令 +0.10 保持 2 s），改用上面这条下限。两者的差别：

* 状态机是**事件触发**的，只在台阶前 0.5 m 内抬高，抬完 2 s 自动落回，且需要一个额外的
  三射线前向传感器；下限是**按列按行静态生效**的，整条 episode 都不会低于该值，不用传感器。
* 状态机会随行进方向翻转扫描方向、会把高过机身的障碍判成墙并转 time_out；下限没有这两件事，
  出块由 `terminations.terrain_cleared` 的距离判据接管。
* 部署契约上，状态机要求上层控制器复现同一套逻辑才能对齐训练期的高度指令；下限只是
  改变了训练期高度指令的采样分布，部署端照常自己发高度指令即可，没有额外契约。

保留下来的只有速度指令的分列覆盖：非平地列只发前向直行指令，因为对称随机指令下 20 s 的
净位移是随机游走，地形课程的位移判据推不动（见 curriculums.py）。
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


@dataclass
class RoughCommandCfg(JumpCommandCfg):
    """在 JumpCommand 之上增加“按地形列限制速度指令”的配置。

    高度指令的地形感知下限走基类 `VelocityHeightCommandCfg` 的
    `terrain_aware_height` / `terrain_height_clearance` /
    `body_collision_bottom_offset` / `terrain_step_height_type_names` 四个字段，
    本类不再额外定义。
    """

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

    stair_command_terrain_names: tuple[str, ...] = ()
    """单独定价的台阶列名；空元组即关闭，这些列沿用上面的通用地形列范围。

    2026-09-08 用户定（A8）：A7 把所有非平地列的 vx 收到 (0.4, 0.8) 之后，平地能力和地形列梯度
    都回来了，但 stairs_up 1500 轮只从 1.09 挪到 1.11——6 cm 轮子靠 0.8 m/s 的动量翻不过 4 cm 立面。
    所以把台阶列拆出来单独给高速与高站姿，其余地形列（斜坡、起伏）保持 A7 的低速档。
    """

    stair_lin_vel_x_range: tuple[float, float] = (1.0, 2.4)
    """台阶列的 vx 采样范围(m/s)。专家数据（Fudan 12–20 cm 爬升）就在 1.5–2.4 这一段。"""

    stair_height_range: tuple[float, float] = (0.35, 0.38)
    """台阶列的机身高度指令范围(m)，覆盖 `height_range`。

    顶到 Flat 上界 0.38 附近，把机身抬高换离地净空；这个区间整体高于地形感知抬高下限
    在最高难度行算出的 0.34，所以那条下限在台阶列上被完全吞掉，不再起作用。
    """

    terrain_lin_vel_x_follow_curriculum: bool = True
    """非平地列 vx 上限是否跟随平地速度课程的当前上限（`cfg.lin_vel_x_range[1]`）。

    开启时每次重采样取 min(terrain_lin_vel_x_range[1], 当前课程上限)，且不低于下界；
    课程起点 0 时地形列拿到的就是下界 0.4 m/s 的定速指令，随课程一起爬到 2.4。
    R3 从第 0 轮就给 0.4–2.4，500 轮的策略对 vx ≥ 1.0 的指令原地不动。
    """

    def build(self, env: ManagerBasedRlEnv) -> RoughCommandTerm:
        return RoughCommandTerm(self, env)


class RoughCommandTerm(JumpCommandTerm):
    """按地形列限制速度指令的速度/姿态/高度指令项。"""

    cfg: RoughCommandCfg

    def __init__(self, cfg: RoughCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        # 非平地列的 env 掩码；None 表示没有可用的分列地形或覆盖未启用。
        self._terrain_override_mask: torch.Tensor | None = None
        # 单独定价的台阶列掩码（`stair_command_terrain_names`），是上面那个的子集。
        self._stair_mask: torch.Tensor | None = None
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
        # 台阶列的覆盖压在通用地形覆盖之上，必须后设。
        self._stair_mask = self._build_column_mask(self._env, self.cfg.stair_command_terrain_names)
        if self._stair_mask is not None and bool(self._stair_mask.any()):
            ids = self._stair_mask.nonzero(as_tuple=False).flatten()
            self.set_velocity_ranges(
                ids,
                lin_vel_x_range=tuple(self.cfg.stair_lin_vel_x_range),
                ang_vel_yaw_range=tuple(self.cfg.terrain_ang_vel_yaw_range),
            )

    def _build_column_mask(
        self,
        env: ManagerBasedRlEnv,
        terrain_type_names: tuple[str, ...],
    ) -> torch.Tensor | None:
        """返回“在这些子地形列上”的 env 掩码；列名为空或非课程地形时返回 None。"""
        if not terrain_type_names:
            return None
        terrain = getattr(env.scene, "terrain", None)
        generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
        terrain_types = getattr(terrain, "terrain_types", None)
        if generator is None or terrain_types is None or not generator.curriculum:
            return None
        names = list(generator.sub_terrains.keys())
        cols = [names.index(n) for n in terrain_type_names if n in names]
        if not cols:
            return None
        types = terrain_types.to(device=self.device, dtype=torch.long)
        mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        for col in cols:
            mask |= types == col
        return mask

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

    def _update_command(self) -> None:
        super()._update_command()
        # 地形感知下限只在重采样时抬高高度指令的采样下界，没有任何逐步状态；
        # 这里只把它的效果记一笔，否则 W&B 上看不出下限有没有真的顶起来
        # （原来这个位置记的是 step_up 状态机的触发率）。
        if self._terrain_override_mask is None:
            return
        height_cmd = self._command[:, 4]
        terrain = self._terrain_override_mask.float()
        log = self._env.extras.setdefault("log", {})
        log["Rough/height_cmd_terrain_mean"] = (height_cmd * terrain).sum() / terrain.sum().clamp(
            min=1.0
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if self._terrain_override_mask is None:
            super()._resample_command(env_ids)
            return
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
        self._apply_stair_height(env_ids)

    def _apply_stair_height(self, env_ids: torch.Tensor) -> None:
        """把台阶列 env 的高度指令改到 `stair_height_range` 内重新采样。

        写在基类采样之后而不是改基类：基类那一路还要管地形感知下限、静站/运动两个区间与
        jump 生命周期，绕过去容易漏。改完必须同步刷新高度条件默认腿姿缓存，否则奖励侧
        用的还是旧高度对应的默认姿态（step_up 状态机时代踩过这个坑）。
        """
        if self._stair_mask is None:
            return
        ids = env_ids[self._stair_mask[env_ids]]
        if ids.numel() == 0:
            return
        low, high = (float(v) for v in self.cfg.stair_height_range)
        self._command[ids, 4] = torch.rand(len(ids), device=self.device) * (high - low) + low
        update_policy_default_from_height_cache(
            self._env,
            "velocity_height",
            env_ids=ids,
            command=self._command,
        )


__all__ = [
    "JumpCommandCfg",
    "RoughCommandCfg",
    "RoughCommandTerm",
    "VelocityHeightCommandCfg",
    "VelocityHeightCommandTerm",
]
