"""倒地起身参考轨迹生成器（se3.recovery_ref.v1），供起身 mimic 教师训练（2026-10-21 用户定）。

参考由开环脚本在原生 MuJoCo 里滚出来，不用策略：两腿左右镜像同步转大腿、主动杆夹角不变。
R1（倒置·腿折叠）分两段：
  ① 翻正：大腿以 `flip_speed` 匀速转，腿撑地把机身撬起，倾角 < `stop_tilt_deg` 即停；
  ② 收腿：停顿 `pause_s` 后反向（`tuck_dir`）转到状态库"蹲坐"构型的大腿角，保持 `hold_s`。
  第二段必须与第一段反向：同向继续转会把机身再撬翻成倒置（2026-10-21 实测，转到 162° 时翻回 173°）。
起点取状态库（se3_tools.fall_state_cache）对应家族的中位构型样本；终点即状态库"蹲坐"中位构型。

物理与训练一致：训练 MJCF、步长与求解器迭代取 Rough 训练配置；腿为关节空间 PD（Rough 默认 60/3）、
±32 N·m（训练执行器 effort_limit）、叠加气弹簧前馈（Rough 默认开，名义 300 N）；轮速度目标 0（kd 0.2、±2.21 N·m）。
每个物理步记一帧，整条轨迹转到起点 yaw=0、xy=0 的规范坐标系。

文件字段（[T, ...]，dt = 物理步长）：
  base_pos / base_quat(wxyz) / base_lin_vel（世界系）/ base_ang_vel_b（机身系）/ projected_gravity / tilt_deg；
  leg_pos / leg_vel（policy 主动杆语义 [LF0, L 驱动杆, RF0, R 驱动杆]）/ wheel_vel [L, R]；
  joint_pos / joint_vel（joint_names 顺序，含闭链被动关节，可直接写回仿真）；
  leg_cmd（开环位置目标）/ leg_tau（施加力矩）/ segment（1 翻正 2 停顿 3 收腿 4 保持）/ phase（t / 时长）；
  meta（JSON：format、family、起点状态库与索引、MJCF sha256、生成参数、各段时长、峰值）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from se3_shared import JointGroup
from se3_shared.fourbar import knee_gas_spring_compensation_torque_np

_ROOT = Path(__file__).resolve().parents[2]
_MJCF = (
    _ROOT
    / "assets"
    / "robots"
    / "serialleg"
    / "mjcf"
    / "serialleg_closed_chain_v3_train_obb_trim.xml"
)
_CACHE = _ROOT / "assets" / "recovery_states" / "fall_states_d1ed876b_40k.npz"
_OUT_DIR = _ROOT / "assets" / "trajectories" / "recovery_ref_v1"
FORMAT = "se3.recovery_ref.v1"
_LEG = list(JointGroup.POLICY_LEG_NAMES)
_WHEELS = list(JointGroup.WHEEL_NAMES)
# 家族 → (重力投影主轴, 符号, 腿折叠?)；目前只实现 R1，其余家族按同样口径扩展。
FAMILIES = {"r1_inverted_folded": (2, 1.0, True)}
_SIT = (2, -1.0, True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _medoid(cache, axis_sign_fold) -> int:
    """状态库里某家族（重力主轴、符号、腿是否折叠）最接近组中位构型的样本。"""
    names = [str(n) for n in cache["joint_names"]]
    cols = [names.index(n) for n in _LEG]
    jp, pg = cache["joint_pos"], cache["projected_gravity"]
    alpha = np.maximum(jp[:, cols[0]] - jp[:, cols[1]], jp[:, cols[3]] - jp[:, cols[2]])
    axis = np.argmax(np.abs(pg), 1)
    sign = np.sign(pg[np.arange(len(pg)), axis])
    ax, sg, folded = axis_sign_fold
    mask = (axis == ax) & (sign == sg) & ((alpha < 0.15) if folded else (alpha >= 0.5))
    ids = np.flatnonzero(mask)
    feat = np.c_[pg[ids] * 2.0, jp[ids][:, cols]]
    return int(ids[np.argmin(np.linalg.norm(feat - np.median(feat, 0), axis=1))])


def _yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def generate(args: argparse.Namespace) -> Path:
    from se3_train.tasks.rough.env_cfg import env_cfg as rough_env_cfg

    cache = np.load(_CACHE)
    names = [str(n) for n in cache["joint_names"]]
    i_start = _medoid(cache, FAMILIES[args.family])
    i_sit = _medoid(cache, _SIT)
    sit_thigh = float(cache["joint_pos"][i_sit, names.index(_LEG[0])])

    rough = rough_env_cfg(play=True)
    kp = float(rough.scene.entities["robot"].articulation.actuators[0].stiffness)
    kd = float(rough.scene.entities["robot"].articulation.actuators[0].damping)
    tau_max = float(rough.scene.entities["robot"].articulation.actuators[0].effort_limit)
    wheel_cfg = rough.scene.entities["robot"].articulation.actuators[1]
    wheel_kd, wheel_tau_max = float(wheel_cfg.damping), float(wheel_cfg.effort_limit)
    action = rough.actions["delayed_action"]
    spring_ff = (
        float(action.knee_gas_spring_force) if action.knee_gas_spring_compensation_enabled else None
    )

    m = mujoco.MjModel.from_xml_path(str(_MJCF))
    m.opt.timestep = float(rough.sim.mujoco.timestep)
    m.opt.iterations = int(rough.sim.mujoco.iterations)
    m.opt.ls_iterations = int(rough.sim.mujoco.ls_iterations)
    data = mujoco.MjData(m)
    qadr = {n: m.jnt_qposadr[m.joint(n).id] for n in names}
    dadr = {n: m.jnt_dofadr[m.joint(n).id] for n in names}
    data.qpos[2] = cache["root_pos"][i_start, 2]
    data.qpos[3:7] = cache["root_quat"][i_start]
    for k, n in enumerate(names):
        data.qpos[qadr[n]] = cache["joint_pos"][i_start, k]
    mujoco.mj_forward(m, data)
    base = m.body("base_link").id
    q0 = np.array([data.qpos[qadr[n]] for n in _LEG])
    dt = float(m.opt.timestep)

    def command(rot: float) -> np.ndarray:
        return q0 + np.array([rot, rot, -rot, -rot])

    rec: dict[str, list] = {
        k: []
        for k in (
            "base_pos",
            "base_quat",
            "base_lin_vel",
            "base_ang_vel_b",
            "projected_gravity",
            "tilt_deg",
            "leg_pos",
            "leg_vel",
            "wheel_vel",
            "joint_pos",
            "joint_vel",
            "leg_cmd",
            "leg_tau",
            "segment",
        )
    }
    rot, t, seg, t_mark, tuck_end = 0.0, 0.0, 1, 0.0, None
    durations: dict[str, float] = {}
    while t < args.timeout_s:
        tilt = float(np.degrees(np.arccos(np.clip(data.xmat[base][8], -1.0, 1.0))))
        if seg == 1:
            if tilt < args.stop_tilt_deg:
                seg, t_mark = 2, t
                durations["flip"] = t
            else:
                rot += args.flip_speed * dt
        elif seg == 2 and t - t_mark >= args.pause_s:
            thigh_now = command(rot)[0]
            delta = (sit_thigh - thigh_now) % (2.0 * np.pi)
            tuck_end = rot + (delta if args.tuck_dir > 0 else delta - 2.0 * np.pi)
            seg, t_mark = 3, t
        elif seg == 3:
            if (tuck_end - rot) * args.tuck_dir <= 0.0:
                seg, durations["tuck"] = 4, t - t_mark
                t_mark = t
            else:
                rot += args.tuck_dir * args.tuck_speed * dt
        elif seg == 4 and t - t_mark >= args.hold_s:
            break
        target = command(rot)
        q = np.array([data.qpos[qadr[n]] for n in _LEG])
        qd = np.array([data.qvel[dadr[n]] for n in _LEG])
        tau = kp * (target - q) - kd * qd
        if spring_ff is not None:
            tau = tau + knee_gas_spring_compensation_torque_np(q, spring_ff)
        tau = np.clip(tau, -tau_max, tau_max)
        data.qfrc_applied[:] = 0.0
        for n, v in zip(_LEG, tau, strict=True):
            data.qfrc_applied[dadr[n]] = v
        for n in _WHEELS:
            data.qfrc_applied[dadr[n]] = np.clip(
                -wheel_kd * data.qvel[dadr[n]], -wheel_tau_max, wheel_tau_max
            )
        mujoco.mj_step(m, data)
        t += dt
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(m, data, mujoco.mjtObj.mjOBJ_XBODY, base, vel, 1)
        rmat = data.xmat[base].reshape(3, 3)
        rec["base_pos"].append(data.qpos[0:3].copy())
        rec["base_quat"].append(data.qpos[3:7].copy())
        rec["base_lin_vel"].append(data.qvel[0:3].copy())
        rec["base_ang_vel_b"].append(vel[0:3].copy())
        rec["projected_gravity"].append(rmat.T @ np.array([0.0, 0.0, -1.0]))
        rec["tilt_deg"].append(float(np.degrees(np.arccos(np.clip(rmat[2, 2], -1.0, 1.0)))))
        rec["leg_pos"].append(np.array([data.qpos[qadr[n]] for n in _LEG]))
        rec["leg_vel"].append(np.array([data.qvel[dadr[n]] for n in _LEG]))
        rec["wheel_vel"].append(np.array([data.qvel[dadr[n]] for n in _WHEELS]))
        rec["joint_pos"].append(np.array([data.qpos[qadr[n]] for n in names]))
        rec["joint_vel"].append(np.array([data.qvel[dadr[n]] for n in names]))
        rec["leg_cmd"].append(target.copy())
        rec["leg_tau"].append(tau.copy())
        rec["segment"].append(seg)
    if seg != 4:
        raise RuntimeError(f"{args.timeout_s} s 内没走完四段（停在第 {seg} 段），检查起点与参数")

    arr = {k: np.asarray(v) for k, v in rec.items()}
    # 规范坐标系：起点 yaw=0、xy=0（训练时再随机 yaw）
    yaw0 = _yaw(arr["base_quat"][0])
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    q_yaw = np.array([np.cos(-yaw0 / 2), 0.0, 0.0, np.sin(-yaw0 / 2)])
    xy0 = arr["base_pos"][0, :2].copy()
    arr["base_pos"][:, :2] -= xy0
    arr["base_pos"] = arr["base_pos"] @ rz.T
    arr["base_lin_vel"] = arr["base_lin_vel"] @ rz.T
    arr["base_quat"] = np.stack([_quat_mul(q_yaw, q) for q in arr["base_quat"]])
    arr["phase"] = np.arange(1, len(arr["segment"]) + 1) * dt / (len(arr["segment"]) * dt)

    cols = [names.index(n) for n in _LEG]
    alpha_l = arr["joint_pos"][:, cols[0]] - arr["joint_pos"][:, cols[1]]
    alpha_r = arr["joint_pos"][:, cols[3]] - arr["joint_pos"][:, cols[2]]
    fd = np.diff(arr["leg_pos"], axis=0) / dt
    peaks = {
        "leg_tau_abs": float(np.abs(arr["leg_tau"]).max()),
        "leg_vel_abs": float(np.abs(arr["leg_vel"]).max()),
        "body_ang_vel": float(np.linalg.norm(arr["base_ang_vel_b"], axis=1).max()),
        "wheel_vel_abs": float(np.abs(arr["wheel_vel"]).max()),
        "leg_power_w": float(np.abs(arr["leg_tau"] * arr["leg_vel"]).sum(1).max()),
        "active_angle_range": [
            float(min(alpha_l.min(), alpha_r.min())),
            float(max(alpha_l.max(), alpha_r.max())),
        ],
        "leg_vel_finite_diff_rms": float(np.sqrt(np.mean((fd - arr["leg_vel"][1:]) ** 2))),
    }
    meta = {
        "format": FORMAT,
        "family": args.family,
        "dt": dt,
        "duration_s": float(len(arr["segment"]) * dt),
        "segments": {"1": "flip", "2": "pause", "3": "tuck", "4": "hold"},
        "segment_durations_s": {**durations, "pause": args.pause_s, "hold": args.hold_s},
        "start_cache": str(_CACHE.relative_to(_ROOT)),
        "start_cache_index": i_start,
        "end_like_cache_index": i_sit,
        "start_tilt_deg": float(arr["tilt_deg"][0]),
        "end_tilt_deg": float(arr["tilt_deg"][-1]),
        "end_base_height": float(arr["base_pos"][-1, 2]),
        "mjcf_sha256": _sha256(_MJCF),
        "git_commit": _git_commit(),
        "physics": {
            "timestep": dt,
            "iterations": int(m.opt.iterations),
            "ls_iterations": int(m.opt.ls_iterations),
        },
        "control": {
            "leg_kp": kp,
            "leg_kd": kd,
            "leg_tau_max": tau_max,
            "wheel_kd": wheel_kd,
            "wheel_tau_max": wheel_tau_max,
            "spring_ff_force": spring_ff,
        },
        "params": {k: v for k, v in vars(args).items() if k != "out"},
        "peaks": peaks,
    }
    out = args.out or _OUT_DIR / f"{args.family}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        meta=np.asarray(json.dumps(meta, ensure_ascii=False)),
        joint_names=np.asarray(names),
        dt=np.asarray(dt),
        **arr,
    )
    print(
        json.dumps(
            {
                "out": str(out),
                "T": len(arr["segment"]),
                **{
                    k: meta[k]
                    for k in (
                        "duration_s",
                        "segment_durations_s",
                        "start_tilt_deg",
                        "end_tilt_deg",
                        "end_base_height",
                    )
                },
                "peaks": peaks,
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="开环滚出倒地起身参考轨迹（se3.recovery_ref.v1）")
    parser.add_argument("--family", choices=sorted(FAMILIES), default="r1_inverted_folded")
    parser.add_argument("--flip-speed", type=float, default=1.5)
    parser.add_argument("--tuck-speed", type=float, default=1.5)
    parser.add_argument("--tuck-dir", type=float, choices=(1.0, -1.0), default=-1.0)
    parser.add_argument("--stop-tilt-deg", type=float, default=20.0)
    parser.add_argument("--pause-s", type=float, default=0.5)
    parser.add_argument("--hold-s", type=float, default=1.5)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument("--out", type=Path, default=None)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
