"""每类评测的执行函数：输入一条 Case（dict 形式，可跨进程），输出该条的标量指标。

所有函数都在 worker 进程里跑，只依赖 se3_runtime / MuJoCo，不导入 se3_train。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from se3_tools.bench import spec
from se3_tools.bench.sim import (
    WHEEL_RADIUS_M,
    Episode,
    EpisodeConfig,
    VideoRecorder,
    find_risers,
    terrain_profile,
)

BASE_FRONT_M = 0.34
"""机身碰撞体前沿在轮轴前约 0.34 m（站姿关键帧实测，沿用立面事件脚本的口径）。"""
BAND_LOW_HZ, BAND_HIGH_HZ = 5.0, 15.0


def _band_rms(values: np.ndarray, dt: float, low: float, high: float) -> float:
    """去均值后 [low, high] Hz 频带内的 RMS（矩形窗 FFT 能量折算）。"""
    x = np.asarray(values, dtype=np.float64)
    if len(x) < 8:
        return math.nan
    x = x - x.mean()
    spectrum = np.fft.rfft(x)
    freq = np.fft.rfftfreq(len(x), dt)
    mask = (freq >= low) & (freq <= high)
    power = (np.abs(spectrum[mask]) ** 2).sum() * 2.0 / len(x) ** 2
    return float(math.sqrt(power))


def _settle(ep: Episode, seconds: float, *, height: float) -> bool:
    """零速站立若干秒；返回是否摔倒。"""
    ep.set_command(vx=0.0, yaw=0.0, height=height)
    for _ in range(round(seconds / ep.policy_dt)):
        ep.step()
        if ep.fell():
            return True
    return False


def run_flat(ep: Episode, p: dict, video: VideoRecorder | None) -> dict:
    """平地阶跃到 (vx, yaw, height)，取最后 5 s 的稳态。"""
    if _settle(ep, 2.0, height=p["height"]):
        return {"fell": True}
    ep.set_command(vx=p["vx"], yaw=p["yaw"], height=p["height"])
    rows = []
    for _ in range(round(10.0 / ep.policy_dt)):
        ep.step()
        if video:
            video.capture()
        lin, ang = ep.body_velocity()
        rows.append((lin[0], ang[2], ep.measured_height(), ang[1], ep.pitch_deg()))
        if ep.fell():
            return {"fell": True}
    a = np.asarray(rows)[-round(5.0 / ep.policy_dt) :]
    vx, wz, h, pitch_rate, pitch = a.T
    out = {
        "fell": False,
        "vx_mean": float(vx.mean()),
        "vx_err": float(abs(vx.mean() - p["vx"])),
        "wz_mean": float(wz.mean()),
        "wz_err": float(abs(wz.mean() - p["yaw"])),
        "height_err": float(h.mean() - p["height"]),
        "pitch_std_deg": float(pitch.std()),
        "pitch_rate_band_rms": _band_rms(pitch_rate, ep.policy_dt, BAND_LOW_HZ, BAND_HIGH_HZ),
    }
    if abs(p["yaw"]) > 1e-6:
        out["wz_ratio"] = float(wz.mean() / p["yaw"])
    if abs(p["vx"]) > 1e-6:
        out["vx_ratio"] = float(vx.mean() / p["vx"])
    return out


def run_standing(ep: Episode, p: dict, video: VideoRecorder | None) -> dict:
    """静站 stand_s 秒测漂移，再阶跃到 vx，测跟到 90% 指令所需时间（5 s 内没到记 inf）。"""
    ep.set_command(vx=0.0, yaw=0.0, height=p["height"])
    start = ep.base_position()[:2]
    yaw0 = math.atan2(ep.rotation()[1, 0], ep.rotation()[0, 0])
    vxs = []
    stand_steps = round(p["stand_s"] / ep.policy_dt)
    for k in range(stand_steps):
        ep.step()
        if video and k >= stand_steps - round(2.0 / ep.policy_dt):
            video.capture()
        vxs.append(ep.body_velocity()[0][0])
        if ep.fell():
            return {"fell": True}
    end = ep.base_position()[:2]
    yaw1 = math.atan2(ep.rotation()[1, 0], ep.rotation()[0, 0])
    tail = np.asarray(vxs[-round(min(10.0, p["stand_s"]) / ep.policy_dt) :])
    out = {
        "fell": False,
        "drift_speed": float(abs(tail.mean())),
        "drift_m": float(np.linalg.norm(end - start)),
        "yaw_drift_deg": float(
            abs(math.degrees(math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0))))
        ),
    }
    ep.set_command(vx=p["vx"])
    reached = math.inf
    filtered = 0.0
    alpha = ep.policy_dt / 0.1
    for k in range(round(5.0 / ep.policy_dt)):
        ep.step()
        if video:
            video.capture()
        filtered += alpha * (ep.body_velocity()[0][0] - filtered)
        if ep.fell():
            out["fell"] = True
            break
        if filtered >= 0.9 * p["vx"]:
            reached = (k + 1) * ep.policy_dt
            break
    out["t90_s"] = reached
    return out


def run_standing_drift(ep: Episode, p: dict, video: VideoRecorder | None) -> dict:
    return {
        k: v
        for k, v in run_standing(ep, dict(p, stand_s=p["seconds"], vx=0.0), video).items()
        if k != "t90_s"
    }


def run_push(ep: Episode, p: dict, video: VideoRecorder | None) -> dict:
    """静站 3 s 后给机身前向速度冲量 dv，测最大倾角与是否摔倒。"""
    if _settle(ep, 3.0, height=p["height"]):
        return {"fell": True}
    ep.data.qvel[0] += float(p["dv"])
    tilt = 0.0
    for _ in range(round(4.0 / ep.policy_dt)):
        ep.step()
        tilt = max(tilt, math.degrees(math.acos(float(np.clip(ep.up_cos(), -1.0, 1.0)))))
        if ep.fell():
            return {"fell": True, "tilt_max_deg": tilt}
    return {"fell": False, "tilt_max_deg": tilt}


def _riser_events(trace: np.ndarray, risers: list[tuple[float, float]], dt: float) -> list[dict]:
    """上行立面事件。trace 列：t, 机身前进量, 双轮平均前进量, 左轮离地, 右轮离地, 左轮 z, 右轮 z。

    - jumped：轮前沿触到立面前 0.5 s 内双轮同时离地 > 2 cm（跳上去，用户不要）；
    - body_first：机身前沿越过立面早于双轮抬起 6 cm（「贴地撞面、机身先过沿」）；
    - lift_dt：左右轮各自抬起 6 cm 的时刻差（「双轮同抬」）。
    """
    events = []
    for position, rise in risers:
        if rise <= 0:
            continue
        reach = trace[:, 2] >= position - WHEEL_RADIUS_M - 0.02
        if not reach.any():
            break
        i_c = int(np.argmax(reach))
        pre = trace[max(0, i_c - round(0.5 / dt)) : i_c]
        both_air = np.minimum(pre[:, 3], pre[:, 4]) if len(pre) else np.array([])
        jumped = bool(len(both_air) and np.nanmax(both_air) > 0.02)
        post = trace[i_c : i_c + round(1.5 / dt)]
        lifted_l = post[:, 5] > post[0, 5] + 0.06
        lifted_r = post[:, 6] > post[0, 6] + 0.06
        t_l = post[int(np.argmax(lifted_l)), 0] if lifted_l.any() else math.nan
        t_r = post[int(np.argmax(lifted_r)), 0] if lifted_r.any() else math.nan
        t_lift = np.nanmax([t_l, t_r]) if np.isfinite([t_l, t_r]).any() else math.nan
        body_over = trace[:, 1] + BASE_FRONT_M >= position
        t_body = trace[int(np.argmax(body_over)), 0] if body_over.any() else math.nan
        body_first = bool(np.isfinite(t_body) and np.isfinite(t_lift) and t_body <= t_lift)
        lift_dt = float(abs(t_l - t_r)) if np.isfinite(t_l) and np.isfinite(t_r) else math.nan
        events.append(
            dict(
                riser_m=position,
                rise_m=rise,
                jumped=jumped,
                body_first=body_first,
                lift_dt_s=lift_dt,
                target_style=bool(
                    (not jumped) and body_first and np.isfinite(lift_dt) and lift_dt <= 0.1
                ),
            )
        )
    return events


def run_stairs(ep: Episode, p: dict, video: VideoRecorder | None) -> dict:
    """出生在该块出生点、朝 +x 开 20 s；L∞ ≥ 4 m 算走出。上行列另记立面事件。"""
    origin = ep.base_position()[:2]
    heading = ep.heading_xy()
    distances, ground = terrain_profile(ep, origin, heading)
    risers = find_risers(distances, ground)
    running = p["mode"] == "running"
    ep.set_command(
        vx=p["vx"], yaw=0.0, height=spec.RUNNING_START_HEIGHT_M if running else p["height"]
    )
    switch = round(spec.RUNNING_START_S / ep.policy_dt) if running else -1
    best = 0.0
    fell = False
    rows = []
    for k in range(round(spec.STAIR_EPISODE_S / ep.policy_dt)):
        if k == switch:
            ep.set_command(height=p["height"])
        ep.step()
        if video:
            video.capture()
        pos = ep.base_position()
        offset = np.abs(pos[:2] - origin)
        best = max(best, float(offset.max()))
        wheels = ep.wheel_positions()
        cl, cr = ep.wheel_clearances()
        rows.append(
            (
                ep.time_s,
                float(np.dot(pos[:2] - origin, heading)),
                float(np.dot(wheels[:, :2].mean(axis=0) - origin, heading)),
                cl,
                cr,
                wheels[0, 2],
                wheels[1, 2],
            )
        )
        if ep.fell():
            fell = True
            break
        if best >= spec.STAIR_CLEAR_DISTANCE_M:
            break
    out = {
        "fell": fell,
        "cleared": bool(best >= spec.STAIR_CLEAR_DISTANCE_M),
        "far_m": best,
        "risers_seen": len([r for r in risers if r[1] > 0]),
    }
    if p["column"] in ("stairs_up", "stairs_two_step_up") and rows:
        events = _riser_events(np.asarray(rows, dtype=np.float64), risers, ep.policy_dt)
        out["riser_events"] = events
        if events:
            out["target_style_rate"] = float(np.mean([e["target_style"] for e in events]))
            out["jump_onto_rate"] = float(np.mean([e["jumped"] for e in events]))
    return out


def run_jump(ep: Episode, p: dict, video: VideoRecorder | None) -> dict:
    """站 2 s（跳跃站高）后可选前进，再请求一次跳跃；测轮底最大离地间隙与落地恢复。"""
    if not ep.runtime.supports_jump:
        return {"skipped": "策略不支持跳跃"}
    h0 = spec.JUMP_STAND_HEIGHT_M
    if _settle(ep, 2.0, height=h0):
        return {"fell": True}
    if p["vx"]:
        ep.set_command(vx=p["vx"])
        for _ in range(round(1.5 / ep.policy_dt)):
            ep.step()
    if not ep.runtime.trigger_jump(float(p["target"])):
        return {"skipped": "trigger_jump 被拒"}
    peak = 0.0
    for _ in range(round(1.5 / ep.policy_dt)):
        ep.step()
        if video:
            video.capture()
        cl = ep.wheel_clearances()
        if np.isfinite(cl).all():
            peak = max(peak, float(min(cl)))
        if ep.fell():
            return {"fell": True, "clearance_peak_m": peak}
    ep.set_command(vx=0.0)
    for _ in range(round(2.0 / ep.policy_dt)):
        ep.step()
        if video:
            video.capture()
        if ep.fell():
            return {"fell": True, "clearance_peak_m": peak}
    recovered = abs(ep.measured_height() - h0) <= 0.03
    target = float(p["target"])
    return {
        "fell": False,
        "clearance_peak_m": peak,
        "clearance_err_m": peak - target,
        "recovered": bool(recovered),
        "success": bool(peak >= spec.JUMP_SUCCESS_RATIO * target and recovered),
    }


RUNNERS = {
    "flat": run_flat,
    "standing": run_standing,
    "standing_drift": run_standing_drift,
    "push": run_push,
    "stairs": run_stairs,
    "jump": run_jump,
}


def run_case(job: dict) -> dict:
    """worker 入口：job = {case, onnx, scene_path, spawns, solver_hz, video_dir}。"""
    case = job["case"]
    spawn = None
    if case["scene"] == "stairs":
        spawn = job["spawns"][f"{case['params']['column']}:{case['params']['row']}"]
    cfg = EpisodeConfig(
        onnx=job["onnx"],
        scene=job["scene_path"] if case["scene"] == "stairs" else None,
        solver_hz=job.get("solver_hz"),
        plant=spec.PLANTS[case["plant"]],
        action_noise_std=float(case["noise_std"]),
        seed=int(case["seed"]),
        spawn_position=tuple(spawn) if spawn is not None else None,
    )
    result = {
        "suite": case["suite"],
        "name": case["name"],
        "kind": case["kind"],
        "plant": case["plant"],
    }
    try:
        ep = Episode(cfg)
        ep.reset()
        video = None
        if case["video"] and job.get("video_dir"):
            safe = case["name"].replace("/", "_")
            video = VideoRecorder(ep, Path(job["video_dir"]) / f"{safe}.mp4")
        result.update(RUNNERS[case["kind"]](ep, case["params"], video))
        if video is not None:
            saved = video.save()
            if saved:
                result["video"] = saved
            elif video.error:
                result["video_error"] = video.error
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


__all__ = ["RUNNERS", "run_case"]
