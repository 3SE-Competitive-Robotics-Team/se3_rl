"""本任务的事件：平地速度课程的信号掩码、逐项奖励的分列诊断、倒地出生与起身日志。

前两个是按列的口径。mjlab 的 MetricsManager 只会对全部 reset 的 env 求均值，做不出
"只看台阶列"的均值，所以分列诊断仍走 interval 事件直接写 `extras["log"]`。

倒地出生（2026-10-20）：`reset_from_fall_states` 从 `se3_tools.fall_state_cache` 生成的状态库里取停稳的倒地状态，
覆盖平地列一部分 env 的初始根状态与关节状态；不经过旧 recovery 机制（`mdp.recovery_state` 的门控保持关闭），
奖励按普通 env 计价。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_from_angle_axis, quat_mul

from se3_shared import JointGroup
from se3_train.mdp.rewards import _policy_leg_torque_and_vel
from se3_train.robot_cfg import _MJCF_PATH

from .columns import column_mask

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

CURRICULUM_ENV_MASK_ATTR = "_se3_curriculum_env_mask"
# 本 episode 不计入速度课程信号的 env（mdp.rewards 的跟踪项按此排除）；倒地出生的 episode 挂在这里。
CURRICULUM_EXCLUDE_MASK_ATTR = "_se3_curriculum_exclude_mask"
FALL_SPAWN_MASK_ATTR = CURRICULUM_EXCLUDE_MASK_ATTR
_FALL_CACHE_ATTR = "_se3_fall_state_cache"
_FALL_CLASS_ATTR = "_se3_fall_spawn_class"
_FALL_RECOVERED_ATTR = "_se3_fall_recovered"
_FALL_RECOVER_TIME_ATTR = "_se3_fall_recover_time_s"
_FALL_GROUND_Z_ATTR = "_se3_fall_ground_z"
_FALL_HEIGHT_MAP_ATTR = "_se3_fall_spawn_height_map"
# 起身判据：机身倾角 < 15° 且机身离出生地面 ≥ 0.17 m（蹲坐构型是 0.14 m、倾角 2°，不能算起来了）。
FALL_RECOVERED_TILT_DEG = 15.0
FALL_RECOVERED_MIN_HEIGHT_M = 0.17

# 逐项奖励分列日志的键前缀，统一使用 Rough/ 命名空间。
REWARD_SPLIT_LOG_PREFIX = "Rough/rw_"


def set_curriculum_env_mask(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    terrain_type_names: tuple[str, ...] = ("flat",),
) -> None:
    """startup 事件：把"只按这些子地形列的 env 评估速度课程"的掩码挂到 env 上。

    mdp.rewards.tracking_lin_vel 据此额外记 `Locomotion/tracking_lin_vel_reward_curriculum`，
    commands_vel_adaptive 用它推进平地速度课程。地形列强制前向指令、跟踪分天然偏低，混进全体
    均值会把平地列的课程钉死在 vx=0（R3）。非课程地形（没有分列）时不挂，课程退回全体均值。
    """
    del env_ids
    mask = column_mask(env, terrain_type_names)
    if mask is not None:
        setattr(env, CURRICULUM_ENV_MASK_ATTR, mask)


def log_reward_split_by_column(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    terrain_type_names: tuple[str, ...] = ("stairs_up",),
    suffix: str = "stairs",
) -> None:
    """每步把奖励表里每一项在指定子地形列上的均值记一份（`Rough/rw_<项名>_<suffix>`）。

    `Episode_Reward/*` 是全体均值，台阶列的定价被平地列稀释了看不出来（A8 只能读出本来就只在
    台阶列生效的 command_velocity_error）。这里直接读 RewardManager 上一次 `compute()` 算好的
    逐 env 逐项缓冲 `_step_reward`（value×weight，每秒贡献，与 Episode_Reward 同口径），
    只做一次掩码均值，不重算任何奖励；落后一个控制步对日志无所谓。
    """
    del env_ids
    reward_manager = getattr(env, "reward_manager", None)
    step_reward = getattr(reward_manager, "_step_reward", None)
    if reward_manager is None or step_reward is None:
        return
    names = list(reward_manager.active_terms)
    if step_reward.shape != (env.num_envs, len(names)):
        return
    mask = column_mask(env, terrain_type_names)
    if mask is None:
        return
    weights = mask.to(dtype=step_reward.dtype).unsqueeze(1)
    means = (step_reward * weights).sum(dim=0) / weights.sum().clamp(min=1.0)
    log = env.extras.setdefault("log", {})
    for index, name in enumerate(names):
        log[f"{REWARD_SPLIT_LOG_PREFIX}{name}_{suffix}"] = means[index]


def stair_facing_yaw(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    yaw: torch.Tensor,
    terrain_type_names: tuple[str, ...] = ("stairs_up", "stairs_two_step_up"),
    half_range_rad: float = math.radians(30.0),
) -> torch.Tensor:
    """`reset_root_state_full` 的 yaw 钩子：指定台阶列的 env 初始朝向改为「正对某一面台阶 ± half_range」（M50，2026-09-29 用户定）。

    stairs_up（反金字塔）与二级台阶都是以出生点为中心、与世界轴对齐的方形环，四个正对方向是 0/90/180/270°；
    每个 env 随机挑一面，再加 U(−half_range, +half_range)。其余列保持传入的均匀 yaw。没有分列信息（plane）时原样返回。
    """
    mask = column_mask(env, terrain_type_names)
    if mask is None:
        return yaw
    ids = env_ids.to(device=env.device, dtype=torch.long)
    on_stairs = mask[ids]
    face = torch.randint(0, 4, (ids.numel(),), device=env.device).to(yaw.dtype) * (math.pi / 2.0)
    jitter = (torch.rand(ids.numel(), device=env.device, dtype=yaw.dtype) * 2.0 - 1.0) * float(
        half_range_rad
    )
    facing = torch.remainder(face + jitter + math.pi, 2.0 * math.pi) - math.pi
    return torch.where(on_stairs, facing, yaw)


class _FallStateCache:
    """倒地状态库：一次性读进设备，校验 MJCF 哈希与关节名，关节列按实体顺序重排。"""

    def __init__(self, path: Path, joint_names: tuple[str, ...], device: str):
        if not path.is_file():
            raise FileNotFoundError(
                f"倒地状态库不存在：{path}；用 `python -m se3_tools.fall_state_cache` 生成"
            )
        data = np.load(path)
        meta = json.loads(str(data["metadata_json"]))
        mjcf_sha = hashlib.sha256(_MJCF_PATH.read_bytes()).hexdigest()
        if meta.get("mjcf_sha256") != mjcf_sha:
            raise ValueError(
                f"倒地状态库 {path.name} 的 MJCF 哈希 {str(meta.get('mjcf_sha256'))[:8]} 与当前 "
                f"{_MJCF_PATH.name} 的 {mjcf_sha[:8]} 不一致，模型改过，需重新生成状态库"
            )
        cache_names = tuple(str(name) for name in data["joint_names"])
        if set(cache_names) != set(joint_names):
            raise ValueError(f"倒地状态库关节名 {cache_names} 与实体关节 {joint_names} 不一致")
        order = [cache_names.index(name) for name in joint_names]

        def tensor(key: str, dtype=torch.float32) -> torch.Tensor:
            return torch.as_tensor(data[key], dtype=dtype, device=device)

        self.root_pos = tensor("root_pos")
        self.root_quat = tensor("root_quat")
        self.root_vel = torch.cat((tensor("root_lin_vel"), tensor("root_ang_vel")), dim=1)
        self.joint_pos = tensor("joint_pos")[:, order]
        self.joint_vel = tensor("joint_vel")[:, order]
        self.tilt_class = tensor("tilt_class", torch.long)
        self.class_names = tuple(meta["tilt_class_names"])
        self.size = int(self.root_pos.shape[0])
        self.weights: torch.Tensor | None = None

    def sample_weights(self, class_weights: tuple[float, ...]) -> torch.Tensor:
        """每条样本的抽样权重：类别权重 / 该类条数，使各类出现频率正比于 class_weights。"""
        if len(class_weights) != len(self.class_names):
            raise ValueError(f"class_weights 须对应 {self.class_names}，实际 {class_weights}")
        counts = torch.bincount(self.tilt_class, minlength=len(self.class_names)).float()
        per_class = torch.tensor(class_weights, device=counts.device) / counts.clamp(min=1.0)
        return per_class[self.tilt_class]


class _SpawnHeightMap:
    """可出生地块中心附近的地面高度图：启动时对每块地块中心 ±half_extent 的网格竖直打射线（只打地形 geom 组）。

    出生高度要以机器人脚下实际地面为准：状态库在平面上落定，波浪地形出生点处的地面与地块原点高度最多差 ±幅值，
    再叠上坡度，直接叠原点高度会穿进地里（实测难度 6–9 最深 23 cm、机身被弹到 4.5 m/s）。取出生点周围
    `footprint_radius_m` 圆内的最高地面作基准，平地与障碍平台上与原点高度相同，起伏地形上只会落下几厘米。
    """

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        column_names: tuple[str, ...],
        half_extent_m: float,
        resolution_m: float = 0.05,
        terrain_geom_group: int = 0,
    ):
        terrain = env.scene.terrain
        names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
        origins = terrain.terrain_origins.detach().cpu().numpy()
        model = env.sim.mj_model
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        group = np.zeros(6, dtype=np.uint8)
        group[terrain_geom_group] = 1
        geom_id = np.zeros(1, dtype=np.int32)
        down = np.array([0.0, 0.0, -1.0])
        grid = np.arange(-half_extent_m, half_extent_m + 1e-9, resolution_m)
        heights = np.full((origins.shape[0], origins.shape[1], grid.size, grid.size), np.nan)
        for col in (names.index(name) for name in column_names if name in names):
            for row in range(origins.shape[0]):
                ox, oy, oz = origins[row, col]
                top = oz + 3.0
                for i, dx in enumerate(grid):
                    for j, dy in enumerate(grid):
                        dist = mujoco.mj_ray(
                            model,
                            data,
                            np.array([ox + dx, oy + dy, top]),
                            down,
                            group,
                            1,
                            -1,
                            geom_id,
                        )
                        if dist >= 0.0:
                            heights[row, col, i, j] = top - dist
        self.heights = torch.as_tensor(heights, dtype=torch.float32, device=env.device)
        self.grid0 = float(grid[0])
        self.resolution = float(resolution_m)
        self.size = int(grid.size)

    def max_ground_z(
        self, rows: torch.Tensor, cols: torch.Tensor, offset_xy: torch.Tensor, radius_m: float
    ) -> torch.Tensor:
        """地块 (row, col) 中心偏 offset_xy 处、半径 radius_m 圆内的最高地面 z；越出高度图或没打中地形时为 NaN。"""
        r = math.ceil(radius_m / self.resolution)
        steps = torch.arange(-r, r + 1, device=offset_xy.device)
        di, dj = torch.meshgrid(steps, steps, indexing="ij")
        inside = (di.square() + dj.square()) * self.resolution**2 <= radius_m**2
        di, dj = di[inside], dj[inside]
        ci = torch.round((offset_xy[:, 0] - self.grid0) / self.resolution).long()
        cj = torch.round((offset_xy[:, 1] - self.grid0) / self.resolution).long()
        ii = ci.unsqueeze(1) + di
        jj = cj.unsqueeze(1) + dj
        valid = (ii >= 0) & (ii < self.size) & (jj >= 0) & (jj < self.size)
        h = self.heights[
            rows.unsqueeze(1),
            cols.unsqueeze(1),
            ii.clamp(0, self.size - 1),
            jj.clamp(0, self.size - 1),
        ]
        h = torch.where(valid, h, torch.full_like(h, float("nan")))
        # 任何一个采样点无效都返回 NaN，由调用方退回原点高度（宁可不修正也不用半张图的最大值）。
        return torch.where(
            torch.isnan(h).any(dim=1), torch.full_like(h[:, 0], float("nan")), h.amax(dim=1)
        )


def _original_column_mask(env: ManagerBasedRlEnv, names: tuple[str, ...]) -> torch.Tensor | None:
    """按原始列归属取掩码：平地热身期全体 env 被挪到平地列，二级台阶门控前也会换列。"""
    # curriculums 反过来从本模块导入 set_curriculum_env_mask，这里延迟导入避免循环。
    from .curriculums import FLAT_WARMUP_ORIGINAL_TYPES_ATTR, TWO_STEP_GATE_ORIGINAL_TYPES_ATTR

    current = column_mask(env, names)
    original = getattr(env, FLAT_WARMUP_ORIGINAL_TYPES_ATTR, None)
    if original is None:
        original = getattr(env, TWO_STEP_GATE_ORIGINAL_TYPES_ATTR, None)
    if current is None or original is None:
        return current
    generator = env.scene.terrain.cfg.terrain_generator
    cols = [list(generator.sub_terrains.keys()).index(name) for name in names]
    return torch.isin(original.to(torch.long), torch.tensor(cols, device=original.device))


def reset_from_fall_states(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    cache_path: str,
    prob: float,
    class_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
    terrain_type_names: tuple[str, ...] = ("flat",),
    command_name: str = "velocity_height",
    xy_offset_m: float = 0.1,
    footprint_radius_m: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    group_mask_attr: str | None = None,
) -> None:
    """reset 事件：指定列一部分 env 改从倒地状态库出生（排在 reset_root_state / reset_joints 之后，只覆盖选中的 env）。

    选中条件：原始列与当前列都在 `terrain_type_names`、不是跳跃 env、不在高姿态起步序列里，再按 `prob` 抽取。
    状态按倾角类别加权抽样（见 `_FallStateCache.sample_weights`），绕世界 z 随机转一个 yaw；水平位置取地块中心
    ±`xy_offset_m`（状态库里落定时滑出去的 xy 丢掉，p95 有 0.5 m，会出生到障碍平台边上）；高度 = 状态库机身离地高度 +
    出生点 `footprint_radius_m` 圆内最高地面（见 `_SpawnHeightMap`）；速度随同一 yaw 旋转。选中的 env 记入
    FALL_SPAWN_MASK_ATTR（本 episode 不计入速度课程信号），并记下倾角类别与出生地面高度供日志用。
    group_mask_attr：给定时改为"该掩码里的 env 每局都倒地出生、其余 env 一律不倒地出生"（recover 组，见 recover.py），
    忽略 prob 与跳跃 / 高姿态起步的排除（这些 env 本来就只在倒地出生的列里抽）。
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ids = env_ids.to(device=env.device, dtype=torch.long)
    asset = env.scene[asset_cfg.name]

    fall_mask = getattr(env, FALL_SPAWN_MASK_ATTR, None)
    if fall_mask is None:
        fall_mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        setattr(env, FALL_SPAWN_MASK_ATTR, fall_mask)
        setattr(env, _FALL_CLASS_ATTR, torch.full_like(fall_mask, -1, dtype=torch.long))
        setattr(env, _FALL_RECOVERED_ATTR, torch.zeros_like(fall_mask))
        setattr(env, _FALL_RECOVER_TIME_ATTR, torch.zeros(env.num_envs, device=env.device))
        setattr(env, _FALL_GROUND_Z_ATTR, torch.zeros(env.num_envs, device=env.device))
    fall_mask.index_fill_(0, ids, False)
    getattr(env, _FALL_RECOVERED_ATTR).index_fill_(0, ids, False)
    group = None if group_mask_attr is None else getattr(env, group_mask_attr, None)
    if group_mask_attr is not None and group is None:
        raise RuntimeError(
            f"倒地出生按组进行，但 env 上没有 {group_mask_attr}（分组 startup 事件没跑？）"
        )
    if group is None and float(prob) <= 0.0:
        return

    cache: _FallStateCache | None = getattr(env, _FALL_CACHE_ATTR, None)
    if cache is None:
        cache = _FallStateCache(Path(cache_path), tuple(asset.joint_names), env.device)
        setattr(env, _FALL_CACHE_ATTR, cache)
        cache.weights = cache.sample_weights(tuple(class_weights))
        half_extent = float(xy_offset_m) + float(footprint_radius_m) + 0.1
        setattr(env, _FALL_HEIGHT_MAP_ATTR, _SpawnHeightMap(env, terrain_type_names, half_extent))
    height_map: _SpawnHeightMap = getattr(env, _FALL_HEIGHT_MAP_ATTR)

    if group is not None:
        eligible = group[ids].clone()
    else:
        current = column_mask(env, terrain_type_names)
        original = _original_column_mask(env, terrain_type_names)
        if current is None or original is None:
            return
        eligible = current[ids] & original[ids]
        term = env.command_manager.get_term(command_name)
        jump_env = getattr(term, "jump_env", None)
        if jump_env is not None:
            eligible &= ~jump_env[ids]
        high_stand = getattr(term, "_high_stand_selected", None)
        if high_stand is not None:
            eligible &= ~high_stand[ids]
        eligible &= torch.rand(ids.numel(), device=env.device) < float(prob)
    sel = ids[eligible]
    n = sel.numel()
    if n == 0:
        return

    k = torch.multinomial(cache.weights, n, replacement=True)
    yaw = (torch.rand(n, device=env.device) * 2.0 - 1.0) * math.pi
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(n, 3)
    q_yaw = quat_from_angle_axis(yaw, z_axis)
    offset = (torch.rand(n, 2, device=env.device) * 2.0 - 1.0) * float(xy_offset_m)
    origin = env.scene.env_origins[sel]
    terrain = env.scene.terrain
    ground = height_map.max_ground_z(
        terrain.terrain_levels[sel].long(),
        terrain.terrain_types[sel].long(),
        offset,
        footprint_radius_m,
    )
    ground = torch.where(torch.isnan(ground), origin[:, 2], ground)
    pos = torch.cat((origin[:, :2] + offset, (ground + cache.root_pos[k, 2]).unsqueeze(1)), dim=1)
    quat = quat_mul(q_yaw, cache.root_quat[k])
    vel = cache.root_vel[k]
    vel = torch.cat((quat_apply(q_yaw, vel[:, :3]), quat_apply(q_yaw, vel[:, 3:])), dim=1)

    asset.write_root_link_pose_to_sim(torch.cat((pos, quat), dim=1), env_ids=sel)
    asset.write_root_link_velocity_to_sim(vel, env_ids=sel)
    asset.write_joint_state_to_sim(cache.joint_pos[k], cache.joint_vel[k], env_ids=sel)
    fall_mask[sel] = True
    getattr(env, _FALL_CLASS_ATTR)[sel] = cache.tilt_class[k]
    getattr(env, _FALL_GROUND_Z_ATTR)[sel] = ground


def fall_recovery_within(
    env: ManagerBasedRlEnv, horizon_s: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """倒地出生已满 horizon_s 的 env 里，horizon_s 内起身的比例（按倾角类别），以及各类有效样本数、平均起身用时。

    只看出生已满 horizon_s 的 env，刚出生还没来得及起身的不会把比例拉低；课程门控与日志共用。
    返回 (rate[类别], count[类别], mean_time)；还没有倒地出生时返回 None。
    """
    fall_mask = getattr(env, FALL_SPAWN_MASK_ATTR, None)
    cache: _FallStateCache | None = getattr(env, _FALL_CACHE_ATTR, None)
    if fall_mask is None or cache is None:
        return None
    age = env.episode_length_buf.float() * env.step_dt
    old = fall_mask & (age >= float(horizon_s))
    recovered: torch.Tensor = getattr(env, _FALL_RECOVERED_ATTR)
    recover_time: torch.Tensor = getattr(env, _FALL_RECOVER_TIME_ATTR)
    ok = old & recovered & (recover_time <= float(horizon_s))
    classes: torch.Tensor = getattr(env, _FALL_CLASS_ATTR)
    n_class = len(cache.class_names)
    index = torch.where(old, classes, torch.full_like(classes, n_class))
    count = torch.bincount(index, minlength=n_class + 1)[:n_class].float()
    hits = torch.bincount(
        torch.where(ok, classes, torch.full_like(classes, n_class)), minlength=n_class + 1
    )
    rate = hits[:n_class].float() / count.clamp(min=1.0)
    mean_time = (recover_time * ok.float()).sum() / ok.float().sum().clamp(min=1.0)
    return rate, count, mean_time


def _effort_joint_ids(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """腿电机与轮关节在实体里的索引（缓存）。"""
    cached = getattr(env, "_se3_fall_effort_joint_ids", None)
    if cached is None:
        robot = env.scene["robot"]
        legs, _ = robot.find_joints(JointGroup.POLICY_LEG_NAMES, preserve_order=True)
        wheels, _ = robot.find_joints(JointGroup.WHEEL_NAMES, preserve_order=True)
        cached = (
            torch.tensor(legs, device=env.device, dtype=torch.long),
            torch.tensor(wheels, device=env.device, dtype=torch.long),
        )
        env._se3_fall_effort_joint_ids = cached
    return cached


def log_fall_recovery(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
    """interval 事件：记倒地出生 env 的起身情况（`Recovery/*`），不改任何状态以外的东西。

    起身判据见 FALL_RECOVERED_TILT_DEG / FALL_RECOVERED_MIN_HEIGHT_M；每个 env 记第一次满足判据的时刻。
    """
    del env_ids
    fall_mask = getattr(env, FALL_SPAWN_MASK_ATTR, None)
    cache: _FallStateCache | None = getattr(env, _FALL_CACHE_ATTR, None)
    if fall_mask is None or cache is None:
        return
    robot = env.scene["robot"]
    pg_z = robot.data.projected_gravity_b[:, 2]
    tilt = torch.rad2deg(torch.acos(torch.clamp(-pg_z, -1.0, 1.0)))
    # 相对出生点地面（波浪地形上地块原点与局部地面最多差 ±幅值）；起身判据只在出生附近用得上。
    height = robot.data.root_link_pos_w[:, 2] - getattr(env, _FALL_GROUND_Z_ATTR)
    up = (tilt < FALL_RECOVERED_TILT_DEG) & (height >= FALL_RECOVERED_MIN_HEIGHT_M)
    recovered: torch.Tensor = getattr(env, _FALL_RECOVERED_ATTR)
    recover_time: torch.Tensor = getattr(env, _FALL_RECOVER_TIME_ATTR)
    first = fall_mask & up & ~recovered
    recover_time[:] = torch.where(first, env.episode_length_buf.float() * env.step_dt, recover_time)
    recovered |= first

    fall = fall_mask.float()
    n_fall = fall.sum().clamp(min=1.0)
    done = (recovered & fall_mask).float()
    log = env.extras.setdefault("log", {})
    log["Recovery/fall_env_rate"] = fall.mean()
    log["Recovery/upright_now_rate"] = (up.float() * fall).sum() / n_fall
    log["Recovery/recovered_rate"] = done.sum() / n_fall
    log["Recovery/time_to_upright_s"] = (recover_time * done).sum() / done.sum().clamp(min=1.0)
    classes: torch.Tensor = getattr(env, _FALL_CLASS_ATTR)
    for index, name in enumerate(cache.class_names):
        in_class = (fall_mask & (classes == index)).float()
        log[f"Recovery/recovered_rate_{name}"] = (done * in_class).sum() / in_class.sum().clamp(
            min=1.0
        )
    within = fall_recovery_within(env, 3.0)
    if within is not None:
        for index, name in enumerate(cache.class_names):
            log[f"Recovery/recovered_within_3s_{name}"] = within[0][index]
    # 还在起身的倒地 env 上的力度（当前步，均值与最大值）：机身角速度、腿速、轮速、腿正功率。
    recovering = (fall_mask & ~recovered).float()
    n_rec = recovering.sum().clamp(min=1.0)
    omega = torch.linalg.norm(robot.data.root_link_ang_vel_b, dim=1)
    leg_ids, wheel_ids = _effort_joint_ids(env)
    leg = robot.data.joint_vel[:, leg_ids]
    wheel = robot.data.joint_vel[:, wheel_ids]
    torque, policy_leg_vel = _policy_leg_torque_and_vel(env, robot)
    for key, value in (
        ("body_ang_vel", omega),
        ("leg_speed", leg.abs().amax(dim=1)),
        ("wheel_speed", wheel.abs().amax(dim=1)),
        ("leg_power", (torque * policy_leg_vel).abs().sum(dim=1)),
        ("leg_power_pos", torch.clamp(torque * policy_leg_vel, min=0.0).sum(dim=1)),
    ):
        log[f"Recovery/{key}_mean_recovering"] = (value * recovering).sum() / n_rec
        log[f"Recovery/{key}_max_recovering"] = (value * recovering).amax()


__all__ = [
    "CURRICULUM_ENV_MASK_ATTR",
    "CURRICULUM_EXCLUDE_MASK_ATTR",
    "FALL_SPAWN_MASK_ATTR",
    "REWARD_SPLIT_LOG_PREFIX",
    "fall_recovery_within",
    "log_fall_recovery",
    "log_reward_split_by_column",
    "reset_from_fall_states",
    "set_curriculum_env_mask",
    "stair_facing_yaw",
]
