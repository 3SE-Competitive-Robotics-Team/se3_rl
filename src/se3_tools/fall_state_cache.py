"""倒地状态库生成器：零力矩重力落定，供 Rough 自起训练的倒地出生使用（2026-10-20）。

做法同 KungfuAthleteBot（arXiv 2610.03388）的 GRSI 与本仓库 S1 时期的 recovery cache（c448de4e^）：
电机零力矩、膝气弹簧照常（MJCF 里是常量 bias 的 tendon actuator，与 ctrl 无关），让物理决定最终倒地构型。
两类起始状态各占一部分：

- drop：机身朝向在 SO(3) 上均匀随机，离地 `drop_height_range` 松手；主动杆夹角在机构可行域内均匀、
  大腿角在默认值 ±`thigh_range_rad` 内均匀，左右独立。覆盖侧躺、仰卧、俯卧与各种扭曲构型。
- collapse：默认站姿、默认站高，机身随机方向倾斜 0–`collapse_tilt_deg`，再给水平速度与角速度后零力矩松手，
  模拟断电或被撞倒后的瘫倒构型。轮腿站着松手大概率只会前后趴，所以不能只靠这一类。

机器人实体、仿真参数与 Rough 训练（`tasks.rough.env_cfg`）完全一致，只把地形换成平面、去掉传感器；
每批重新调用训练端的摩擦与气弹簧 DR 函数（同一套范围），落定 `settle_seconds_range` 后只收停稳的样本。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene.scene import Scene
from mjlab.sim import Simulation
from mjlab.terrains import TerrainEntityCfg

from se3_shared import JointGroup
from se3_shared import RobotConfig as SharedRobotConfig
from se3_shared.fourbar import (
    _ACTIVE_LOWER,
    _ACTIVE_UPPER,
    policy_to_closedchain_passive_pos_torch,
)
from se3_train.mdp.events import randomize_friction, randomize_knee_spring_force
from se3_train.robot_cfg import _MJCF_PATH
from se3_train.tasks.rough.env_cfg import env_cfg as rough_env_cfg

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "assets" / "recovery_states"
FORMAT_VERSION = 1
FAMILY_NAMES = ("drop", "collapse")
# 落定后按机身倾角分桶，只用于统计与训练端按类配比，不进 actor 观测。
TILT_CLASS_NAMES = ("upright", "tilted", "side", "inverted")
_TILT_CLASS_EDGES_DEG = (30.0, 60.0, 135.0)
_DR_FIELDS = ("geom_friction", "actuator_biasprm", "actuator_forcerange")


def _git_commit() -> str:
    """当前 checkout 的提交号；读不到时返回 unknown。"""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _uniform(low: float, high: float, shape: tuple[int, ...], gen: torch.Generator, device: str):
    return torch.empty(shape, device=device).uniform_(float(low), float(high), generator=gen)


def _random_quat_wxyz(n: int, gen: torch.Generator, device: str) -> torch.Tensor:
    """SO(3) 上均匀分布的单位四元数（Shoemake 方法），wxyz 顺序。"""
    u1, u2, u3 = (_uniform(0.0, 1.0, (n,), gen, device) for _ in range(3))
    a, b = torch.sqrt(1.0 - u1), torch.sqrt(u1)
    x = a * torch.sin(2.0 * math.pi * u2)
    y = a * torch.cos(2.0 * math.pi * u2)
    z = b * torch.sin(2.0 * math.pi * u3)
    w = b * torch.cos(2.0 * math.pi * u3)
    return torch.stack((w, x, y, z), dim=1)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _axis_angle_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    half = 0.5 * angle
    return torch.cat((torch.cos(half).unsqueeze(-1), axis * torch.sin(half).unsqueeze(-1)), dim=-1)


def _build(num_envs: int, device: str) -> tuple[Scene, Simulation, SimpleNamespace]:
    """按 Rough 训练配置搭只用于落定的场景：平面地形、无传感器、机器人与仿真参数不变。"""
    cfg = rough_env_cfg(play=False)
    scene_cfg = cfg.scene
    scene_cfg.num_envs = int(num_envs)
    scene_cfg.terrain = TerrainEntityCfg(terrain_type="plane")
    scene_cfg.sensors = ()
    scene = Scene(scene_cfg, device=device)
    sim = Simulation(
        num_envs=scene.num_envs,
        cfg=cfg.sim,
        spec=scene.spec,
        variant_info=scene.collect_variant_info(),
        device=device,
    )
    scene.initialize(mj_model=sim.mj_model, model=sim.model, data=sim.data)
    sim.expand_model_fields(_DR_FIELDS)
    # 训练端 DR 函数只用到 env.scene / env.sim / env.num_envs / env.device，这里给一个同名外壳直接复用。
    shim = SimpleNamespace(scene=scene, sim=sim, num_envs=scene.num_envs, device=device)
    dr = {
        "friction_range": tuple(cfg.events["friction"].params["friction_range"]),
        "spring_scale_range": tuple(cfg.events["knee_spring_force"].params["force_scale_range"]),
    }
    shim.dr = dr
    return scene, sim, shim


def _randomize_dr(shim: SimpleNamespace) -> torch.Tensor:
    """按训练端同一套范围重采摩擦与气弹簧力，返回 [N, 3]（摩擦, 左弹簧力, 右弹簧力）。"""
    asset_cfg = SceneEntityCfg("robot")
    asset_cfg.resolve(shim.scene)
    randomize_friction(shim, None, friction_range=shim.dr["friction_range"], asset_cfg=asset_cfg)
    randomize_knee_spring_force(
        shim, None, force_scale_range=shim.dr["spring_scale_range"], asset_cfg=asset_cfg
    )
    friction = shim.sim.model.geom_friction[:, 0, 0:1].clone()
    return torch.cat((friction, shim._knee_spring_force.clone()), dim=1)


def _sample_states(
    scene: Scene,
    n: int,
    args: argparse.Namespace,
    gen: torch.Generator,
    device: str,
) -> tuple[torch.Tensor, ...]:
    """采样一批松手前状态，返回 (family, root_pose, root_vel, joint_pos, joint_vel)。"""
    robot = scene["robot"]
    names = tuple(str(name) for name in robot.joint_names)
    shared = SharedRobotConfig()
    family = (_uniform(0.0, 1.0, (n,), gen, device) < float(args.collapse_fraction)).long()
    collapse = family == 1

    # 主动杆语义 [LF0, L 驱动杆, RF0, R 驱动杆]，右侧是左侧的镜像（见 se3_shared.fourbar）。
    default = robot.data.default_joint_pos[0]
    leg_ids = [names.index(name) for name in JointGroup.POLICY_LEG_NAMES]
    default_thigh = float(default[leg_ids[0]])
    default_alpha = float(default[leg_ids[0]] - default[leg_ids[1]])
    thigh_span = torch.where(
        collapse,
        torch.tensor(0.15, device=device),
        torch.tensor(args.thigh_range_rad, device=device),
    )
    thigh = default_thigh + thigh_span.unsqueeze(1) * _uniform(-1.0, 1.0, (n, 2), gen, device)
    alpha_free = _uniform(_ACTIVE_LOWER, _ACTIVE_UPPER, (n, 2), gen, device)
    alpha_near = (default_alpha + _uniform(-0.15, 0.15, (n, 2), gen, device)).clamp(
        _ACTIVE_LOWER, _ACTIVE_UPPER
    )
    alpha = torch.where(collapse.unsqueeze(1), alpha_near, alpha_free)
    policy_pos = torch.stack(
        (thigh[:, 0], thigh[:, 0] - alpha[:, 0], -thigh[:, 1], -thigh[:, 1] + alpha[:, 1]), dim=1
    )
    joint_pos = robot.data.default_joint_pos.clone()[:n]
    joint_pos[:, leg_ids] = policy_pos
    passive_ids = [names.index(name) for name in JointGroup.CLOSEDCHAIN_PASSIVE_JOINT_NAMES]
    joint_pos[:, passive_ids] = policy_to_closedchain_passive_pos_torch(policy_pos)
    joint_vel = torch.zeros_like(joint_pos)

    # drop：SO(3) 均匀朝向、高处松手；collapse：站姿绕随机水平轴倾斜后带速度松手。
    quat_drop = _random_quat_wxyz(n, gen, device)
    heading = _uniform(-math.pi, math.pi, (n,), gen, device)
    yaw_quat = _axis_angle_quat(torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3), heading)
    tilt_dir = _uniform(-math.pi, math.pi, (n,), gen, device)
    tilt_axis = torch.stack(
        (torch.cos(tilt_dir), torch.sin(tilt_dir), torch.zeros_like(tilt_dir)), 1
    )
    tilt = torch.deg2rad(_uniform(0.0, args.collapse_tilt_deg, (n,), gen, device))
    quat_collapse = _quat_mul(_axis_angle_quat(tilt_axis, tilt), yaw_quat)
    quat = torch.where(collapse.unsqueeze(1), quat_collapse, quat_drop)

    pos = scene.env_origins[:n].clone()
    pos[:, 2] += torch.where(
        collapse,
        torch.full((n,), float(shared.default_base_height), device=device),
        _uniform(*args.drop_height_range, (n,), gen, device),
    )

    lin = _uniform(-args.drop_lin_vel, args.drop_lin_vel, (n, 3), gen, device)
    ang = _uniform(-args.drop_ang_vel, args.drop_ang_vel, (n, 3), gen, device)
    push_dir = _uniform(-math.pi, math.pi, (n,), gen, device)
    push_speed = _uniform(*args.collapse_speed_range, (n,), gen, device)
    lin_c = torch.stack(
        (
            push_speed * torch.cos(push_dir),
            push_speed * torch.sin(push_dir),
            torch.zeros_like(push_dir),
        ),
        1,
    )
    ang_c = _uniform(-args.collapse_ang_vel, args.collapse_ang_vel, (n, 3), gen, device)
    root_vel = torch.where(
        collapse.unsqueeze(1), torch.cat((lin_c, ang_c), 1), torch.cat((lin, ang), 1)
    )
    return family, torch.cat((pos, quat), 1), root_vel, joint_pos, joint_vel


def _settle(sim: Simulation, steps: int) -> None:
    """零力矩落定：电机 ctrl 与外力每步清零；气弹簧是常量 bias，不受影响。"""
    for _ in range(int(steps)):
        sim.data.ctrl[:] = 0.0
        sim.data.qfrc_applied[:] = 0.0
        sim.step()
    sim.forward()


def _tilt_class(tilt_deg: torch.Tensor) -> torch.Tensor:
    edges = torch.tensor(_TILT_CLASS_EDGES_DEG, device=tilt_deg.device)
    return torch.bucketize(tilt_deg.contiguous(), edges)


def generate(args: argparse.Namespace) -> Path:
    device = args.device
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    scene, sim, shim = _build(args.num_envs, device)
    robot = scene["robot"]
    names = tuple(str(name) for name in robot.joint_names)
    leg_ids = [names.index(name) for name in JointGroup.POLICY_LEG_NAMES]
    dt = float(sim.mj_model.opt.timestep)
    env_ids = torch.arange(scene.num_envs, device=device)

    keys = (
        "family",
        "root_pos",
        "root_quat",
        "root_lin_vel",
        "root_ang_vel",
        "joint_pos",
        "joint_vel",
        "projected_gravity",
        "tilt_deg",
        "tilt_class",
        "dr_params",
        "settle_s",
    )
    store: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    total = attempted = settled = 0
    start = time.perf_counter()
    for batch in range(1, int(args.max_batches) + 1):
        dr = _randomize_dr(shim)
        family, pose, vel, jpos, jvel = _sample_states(scene, scene.num_envs, args, gen, device)
        sim.reset(env_ids)
        robot.reset(env_ids)
        robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        robot.write_root_link_velocity_to_sim(vel, env_ids=env_ids)
        robot.write_joint_state_to_sim(jpos, jvel, env_ids=env_ids)
        settle_s = float(_uniform(*args.settle_seconds_range, (1,), gen, device).item())
        _settle(sim, max(1, round(settle_s / dt)))

        root_pose = robot.data.root_link_pose_w
        root_vel = robot.data.root_link_vel_w
        jp, jv = robot.data.joint_pos, robot.data.joint_vel
        rel = root_pose[:, :3] - scene.env_origins
        pg = robot.data.projected_gravity_b
        tilt = torch.rad2deg(torch.acos(torch.clamp(-pg[:, 2], -1.0, 1.0)))
        alpha_l = jp[:, leg_ids[0]] - jp[:, leg_ids[1]]
        alpha_r = jp[:, leg_ids[3]] - jp[:, leg_ids[2]]
        tol = 0.02
        ok = (
            torch.isfinite(rel).all(1)
            & torch.isfinite(root_vel).all(1)
            & torch.isfinite(jp).all(1)
            & torch.isfinite(jv).all(1)
            & (rel[:, 2] >= args.min_height)
            & (rel[:, 2] <= args.max_height)
            & (torch.linalg.norm(root_vel[:, :3], dim=1) <= args.max_root_lin_speed)
            & (torch.linalg.norm(root_vel[:, 3:], dim=1) <= args.max_root_ang_speed)
            & (jv.abs().amax(1) <= args.max_joint_speed)
            & (alpha_l >= _ACTIVE_LOWER - tol)
            & (alpha_l <= _ACTIVE_UPPER + tol)
            & (alpha_r >= _ACTIVE_LOWER - tol)
            & (alpha_r <= _ACTIVE_UPPER + tol)
        )
        keep = ok.nonzero().squeeze(-1)[: max(0, int(args.num_states) - total)]
        attempted += scene.num_envs
        settled += int(ok.sum())
        if keep.numel():
            values = {
                "family": family,
                "root_pos": rel,
                "root_quat": root_pose[:, 3:7],
                "root_lin_vel": root_vel[:, :3],
                "root_ang_vel": root_vel[:, 3:],
                "joint_pos": jp,
                "joint_vel": jv,
                "projected_gravity": pg,
                "tilt_deg": tilt,
                "tilt_class": _tilt_class(tilt),
                "dr_params": dr,
                "settle_s": torch.full((scene.num_envs,), settle_s, device=device),
            }
            for key in keys:
                store[key].append(values[key][keep].detach().cpu().numpy())
            total += int(keep.numel())
        elapsed = time.perf_counter() - start
        print(
            f"batch={batch} accepted={int(ok.sum())}/{scene.num_envs} total={total}/{args.num_states} "
            f"settle={settle_s:.2f}s elapsed={elapsed:.1f}s",
            flush=True,
        )
        if total >= args.num_states:
            break
    if total < args.num_states:
        raise RuntimeError(
            f"只收到 {total}/{args.num_states} 个停稳样本，调大 --max-batches 或放宽筛选"
        )

    arrays = {key: np.concatenate(values, axis=0) for key, values in store.items()}
    counts = {
        "family": {name: int((arrays["family"] == i).sum()) for i, name in enumerate(FAMILY_NAMES)},
        "tilt_class": {
            name: int((arrays["tilt_class"] == i).sum()) for i, name in enumerate(TILT_CLASS_NAMES)
        },
    }
    metadata = {
        "format_version": FORMAT_VERSION,
        "generator": "se3_tools.fall_state_cache",
        "git_commit": _git_commit(),
        "mjcf": str(_MJCF_PATH.relative_to(_PROJECT_ROOT)),
        "mjcf_sha256": _file_sha256(_MJCF_PATH),
        "sim_dt": dt,
        "num_states": int(args.num_states),
        "attempted": int(attempted),
        "settled": int(settled),
        "dr": {key: list(value) for key, value in shim.dr.items()},
        "dr_params_columns": ["friction", "spring_force_left", "spring_force_right"],
        "family_names": list(FAMILY_NAMES),
        "tilt_class_names": list(TILT_CLASS_NAMES),
        "tilt_class_edges_deg": list(_TILT_CLASS_EDGES_DEG),
        "counts": counts,
        "args": {
            key: value for key, value in vars(args).items() if key not in ("output", "device")
        },
    }
    output = args.output or _DEFAULT_OUTPUT_DIR / (
        f"fall_states_{metadata['mjcf_sha256'][:8]}_{int(args.num_states) // 1000}k.npz"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        joint_names=np.asarray(names),
        **{key: value for key, value in arrays.items()},
    )
    print(f"已保存 {total} 个倒地状态到 {output}；停稳率 {settled / attempted:.3f}；{counts}")
    return output


def _range(raw: str) -> tuple[float, float]:
    low, high = (float(v) for v in raw.split(","))
    return low, high


def main() -> None:
    parser = argparse.ArgumentParser(description="零力矩重力落定生成 Rough 自起用的倒地状态库")
    parser.add_argument("--num-states", type=int, default=40000)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--max-batches", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--collapse-fraction", type=float, default=0.3)
    parser.add_argument("--drop-height-range", type=_range, default=(0.45, 0.60))
    parser.add_argument("--thigh-range-rad", type=float, default=1.0)
    parser.add_argument("--drop-lin-vel", type=float, default=0.3)
    parser.add_argument("--drop-ang-vel", type=float, default=1.0)
    parser.add_argument("--collapse-tilt-deg", type=float, default=25.0)
    parser.add_argument("--collapse-speed-range", type=_range, default=(0.3, 1.5))
    parser.add_argument("--collapse-ang-vel", type=float, default=3.0)
    parser.add_argument("--settle-seconds-range", type=_range, default=(2.0, 4.0))
    parser.add_argument("--min-height", type=float, default=0.03)
    parser.add_argument("--max-height", type=float, default=0.45)
    parser.add_argument("--max-root-lin-speed", type=float, default=0.2)
    parser.add_argument("--max-root-ang-speed", type=float, default=0.5)
    parser.add_argument("--max-joint-speed", type=float, default=1.0)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
