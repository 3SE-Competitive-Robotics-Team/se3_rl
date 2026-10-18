"""固定 Student128-6998，预测实车增益 / 高度扫描各组的静站振荡与闭环裕度；仅离线仿真。

复用 `.scratch/oscillation_origin_20261012/probe_sim.py` 的工况配方（固件陀螺 Q/R 平滑、
腿速度一阶滤波、12 ms 电机一阶滞后、气弹簧 400/360 N 第六秒淡入、实测静置 IMU 噪声回放），
每组先做失能趴姿自然落地 3 s，再使能零指令静站 18 s；另做一次俯仰角速度观测注入
（0.4 rad/s 正弦，8–12 s），用撤除后的衰减刻画闭环裕度。

增益覆盖只发生在本诊断 adapter 内：用改过 stiffness/damping 的契约副本重建
`SerialLegActuatorController` 并替换 `adapter._controller`；runtime 契约、观测、动作解码、
延迟队列和 ONNX 校验全部保持原样。不改 se3_runtime、训练代码和正式资产，不连硬件。

机器人模型按 git 版本钉死（用 `git show` 抽到 scratch，meshdir 改为绝对路径），
主表用 probe_sim / old_measured 当时的 4e1aeb0 版本，另跑 HEAD 与工作区快照做敏感性对照。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.signal import butter, hilbert, sosfiltfilt, welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from record_sim2x_jitter_video import FrameAdapter  # noqa: E402

from se3_runtime import (  # noqa: E402
    PolicyBundle,
    PolicyControlLoop,
    PolicyRuntime,
    SerialLegActuatorController,
)
from se3_runtime._serialleg_v1 import policy_leg_position_error  # noqa: E402

POLICY = "D:/RoboMaster/serialleg2026/build/stm32-policy/Student128-6998/model.onnx"
OUT = ROOT / ".scratch/gain_sweep_prediction_20261013"
MJCF_RELATIVE = "assets/robots/serialleg/mjcf/serialleg_closed_chain_v3_train_obb_trim.xml"
NOISE_SOURCE = ROOT / ".scratch/real_robot_static_20261012/decoded_signals.npz"
REFERENCE_BASELINE = ROOT / ".scratch/passive_armature_ablation_20261012/comparison.json"
PROBE_SIM_REFERENCE = (
    ROOT
    / ".scratch/oscillation_origin_20261012"
    / "firmware_plus_leg_velocity_freeze_none_motor_tau_12ms_spring_400_360N_start_6s_gyro_probe_0.4.json"
)
REAL_FREQUENCY_HZ = 8.144087179189361
"""2026-10-12 实车同步采集拟合出的主振频率；probe_sim 的注入频率。"""
LEG_LABELS = ("LF", "LB", "RF", "RB")
WHEEL_LABELS = ("LW", "RW")
DISABLED_S = 3.0
STAND_S = 18.0
PROBE_INTERVAL_S = (8.0, 12.0)
STATIC_WINDOW_S = (8.0, 18.0)
"""无注入工况的静站统计窗：弹簧淡入（6–6.5 s）之后，与 old_measured 同为 10 s 长。"""
MODEL_VERSIONS = {
    "baseline_4e1aeb0": "4e1aeb0",
    "head_28412c6": "28412c6",
    "worktree_20261013": None,
}
"""None 表示运行开始时的工作区快照（未提交的腿关节阻尼 0.13 等改动）。"""
PRIMARY_MODEL = "baseline_4e1aeb0"


@dataclass(frozen=True)
class Case:
    """一组仿真工况：增益、高度与可选扰动，其他配方全部固定。"""

    name: str
    leg_kp: float
    leg_kd: float
    wheel_kd: float
    height: float
    probe_amplitude: float = 0.0
    probe_hz: float = 8.3
    motor_tau_ms: float = 12.0
    spring_force_n: tuple[float, float] | None = (400.0, 360.0)
    spring_start_s: float = 6.0
    measured_noise: bool = True
    legacy_double_lag_update: bool = False
    """True 时复刻 probe_sim 每个 tick 更新两次电机滞后状态的写法（仅用于复现校验）。"""
    model: str = PRIMARY_MODEL
    label: str = ""


GAIN_GROUPS = {
    "A": (60.0, 3.0, 0.2),
    "B": (60.0, 3.0, 0.1),
    "C": (30.0, 2.0, 0.1),
    "D": (30.0, 2.0, 0.2),
    "E": (60.0, 6.0, 0.2),
}
HEIGHT_GROUPS = {"H22": 0.22, "H30": 0.30, "H38": 0.38}


def matrix(model: str) -> list[Case]:
    """增益五组（高度 0.22）+ 高度三档（增益 A），每组各一条静站与一条注入。"""
    cases: list[Case] = []
    for group, (kp, kd, wkd) in GAIN_GROUPS.items():
        cases.append(
            Case(
                f"gain_{group}",
                kp,
                kd,
                wkd,
                0.22,
                model=model,
                label=f"{group} 腿{kp:g}/{kd:g} 轮{wkd:g} 高0.22",
            )
        )
    for group, height in HEIGHT_GROUPS.items():
        kp, kd, wkd = GAIN_GROUPS["A"]
        cases.append(
            Case(
                f"height_{group}",
                kp,
                kd,
                wkd,
                height,
                model=model,
                label=f"A 腿60/3 轮0.2 高{height:.2f}",
            )
        )
    probes = [replace(case, name=case.name + "_probe", probe_amplitude=0.4) for case in cases]
    return cases + probes


def alignment_cases() -> list[Case]:
    """逐项撤掉 probe_sim 相对 old_measured 多出的假设，解释基线差异来源。"""
    kp, kd, wkd = GAIN_GROUPS["A"]
    return [
        Case(
            "align_nolag_spring300",
            kp,
            kd,
            wkd,
            0.22,
            motor_tau_ms=0.0,
            spring_force_n=None,
            label="对齐组：无电机滞后、弹簧 300 N",
        ),
        Case(
            "align_lag12_spring300",
            kp,
            kd,
            wkd,
            0.22,
            spring_force_n=None,
            label="对齐组：12 ms 滞后、弹簧 300 N",
        ),
    ]


def probe_sim_replica_case() -> Case:
    """与已保存的 probe_sim 注入结果逐数复核的工况（无实测噪声、8.144 Hz）。"""
    kp, kd, wkd = GAIN_GROUPS["A"]
    return Case(
        "replica_probe_sim_gyro_probe_0.4",
        kp,
        kd,
        wkd,
        0.22,
        probe_amplitude=0.4,
        probe_hz=REAL_FREQUENCY_HZ,
        measured_noise=False,
        legacy_double_lag_update=True,
        label="probe_sim 复现校验",
    )


def pinned_model_path(version: str) -> Path:
    """把指定 git 版本（或工作区快照）的 MJCF 抽到 scratch，meshdir 指回正式网格目录。"""
    target = OUT / "mjcf" / f"serialleg_{version}.xml"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    revision = MODEL_VERSIONS[version]
    if revision is None:
        text = (ROOT / MJCF_RELATIVE).read_text(encoding="utf-8")
    else:
        text = subprocess.run(
            ["git", "show", f"{revision}:{MJCF_RELATIVE}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout
    meshes = (ROOT / "assets/robots/serialleg/meshes").resolve().as_posix()
    assert 'meshdir="../meshes"' in text
    target.write_text(text.replace('meshdir="../meshes"', f'meshdir="{meshes}"'), encoding="utf-8")
    return target


class GainSweepAdapter(FrameAdapter):
    """probe_sim 配方的 adapter 版本：每个 tick 的操作顺序与原脚本逐一对应。"""

    configured = False

    def configure(self, case: Case, enable_time: float) -> None:
        """在失能落地结束、使能前调用；此时才开始滤波器状态与噪声时钟。"""
        self.case = case
        self.enable_time = enable_time
        # 增益覆盖：只替换本 adapter 的低层控制器，契约其余字段（限矩、弹簧补偿）原样保留。
        actuators = self.contract.robot.actuators
        overridden = replace(
            actuators,
            stiffness=(case.leg_kp,) * 4 + (0.0, 0.0),
            damping=(case.leg_kd,) * 4 + (case.wheel_kd,) * 2,
        )
        self._controller = SerialLegActuatorController(
            replace(self.contract, robot=replace(self.contract.robot, actuators=overridden))
        )
        # 固件陀螺 Q/R 标量 Kalman 稳态增益与腿速度 4.7 ms 一阶滤波，与 probe_sim 相同。
        q_var, r_var = 0.01, 0.48
        covariance = (np.sqrt(q_var * q_var + 4 * q_var * r_var) - q_var) / 2
        self.kalman_gain = (covariance + q_var) / (covariance + q_var + r_var)
        self.velocity_alpha = 1 - np.exp(-0.001 / 0.0047)
        state = super().read_policy_input().robot
        self.gyro = np.array(state.base_angular_velocity_body)
        self.filtered_velocity = np.array(state.policy_joint_velocity)
        self.sensor_time = float(self.data.time)
        self.actual_effort = np.zeros(6)
        self.lag_time = float("-inf")
        self.motor_tau_s = np.r_[np.full(4, case.motor_tau_ms / 1000), 0.003, 0.004]
        self.noise_time = None
        self.noise_data = None
        if case.measured_noise:
            source = np.load(NOISE_SOURCE)
            valid = source["time_s"] >= 5.0
            self.noise_time = source["time_s"][valid]
            self.noise_time = self.noise_time - self.noise_time[0]
            self.noise_data = np.c_[source["policy_gyro"][valid], source["policy_gravity"][valid]]
            self.noise_data -= self.noise_data.mean(0)
        self.spring_ids = [
            self.model.actuator(name).id for name in ("l_knee_gas_spring", "r_knee_gas_spring")
        ]
        self.configured = True

    def update_sensors(self) -> None:
        """1 kHz 更新滤波后的陀螺与腿速度；同一时刻只更新一次。"""
        if self.data.time <= self.sensor_time + 1e-9:
            return
        value = super().read_policy_input().robot
        self.gyro += self.kalman_gain * (np.asarray(value.base_angular_velocity_body) - self.gyro)
        self.filtered_velocity += self.velocity_alpha * (
            np.asarray(value.policy_joint_velocity) - self.filtered_velocity
        )
        self.sensor_time = float(self.data.time)

    def read_policy_input(self):
        """策略看到的是固件滤波值 + 实测静置噪声 + 可选的俯仰角速度正弦注入。"""
        if not self.configured:
            return super().read_policy_input()
        self.update_sensors()
        value = super().read_policy_input()
        current = np.array(value.robot.base_angular_velocity_body)
        current[:2] = self.gyro[:2]
        velocity = np.array(value.robot.policy_joint_velocity)
        velocity[:4] = self.filtered_velocity[:4]
        value = replace(
            value,
            robot=replace(
                value.robot, base_angular_velocity_body=current, policy_joint_velocity=velocity
            ),
        )
        if self.noise_data is not None:
            sample = np.array(
                [
                    np.interp(self.data.time - self.enable_time, self.noise_time, column)
                    for column in self.noise_data.T
                ]
            )
            value = replace(
                value,
                robot=replace(
                    value.robot,
                    base_angular_velocity_body=np.asarray(value.robot.base_angular_velocity_body)
                    + sample[:3],
                    projected_gravity_body=np.asarray(value.robot.projected_gravity_body)
                    + sample[3:],
                ),
            )
        if self.case.probe_amplitude:
            elapsed = self.data.time - self.enable_time
            start, stop = PROBE_INTERVAL_S
            envelope = np.clip((elapsed - start) / 0.1, 0, 1) * np.clip(
                (stop - elapsed) / 0.1, 0, 1
            )
            current = np.asarray(value.robot.base_angular_velocity_body).copy()
            current[1] += (
                self.case.probe_amplitude
                * envelope
                * np.sin(2 * np.pi * self.case.probe_hz * elapsed)
            )
            value = replace(value, robot=replace(value.robot, base_angular_velocity_body=current))
        return value

    def _refresh_actuator_control(self) -> None:
        """PD 用滤波腿速度；输出经一阶电机滞后后才写入 MuJoCo。"""
        if not self.configured or not self._motors_enabled or self._last_decoded_action is None:
            super()._refresh_actuator_control()
            return
        self.update_sensors()
        velocity = self.policy_joint_velocity
        velocity[:4] = self.filtered_velocity[:4]
        control = self._controller.compute(
            self._last_decoded_action,
            policy_joint_position=self.policy_joint_position,
            policy_joint_velocity=velocity,
        )
        if self.case.motor_tau_ms > 0:
            # 单极点响应只是滞后假设，不是六个电机的标定模型（与 probe_sim 相同）。
            # 共用 loop 每个 1 ms tick 会调用本方法两次（apply_decoded_action 与 solver step），
            # probe_sim 因此每 tick 推进两次滤波、等效时间常数减半；本脚本按时间门控只推进一次。
            if self.case.legacy_double_lag_update or self.data.time > self.lag_time + 1e-9:
                alpha = 1 - np.exp(-0.001 / self.motor_tau_s)
                self.actual_effort += alpha * (control.effort - self.actual_effort)
                self.lag_time = float(self.data.time)
            control = replace(control, effort=self.actual_effort.copy())
        self.data.ctrl[self._actuator_ids] = control.effort
        self._last_control = control

    def advance_physics_tick(self) -> None:
        """弹簧淡入 → 延时使能 → 原始 tick（记录轨迹、积分）→ 传感器更新。"""
        if not self.configured:
            super().advance_physics_tick()
            return
        case = self.case
        if case.spring_force_n is not None:
            fraction = np.clip(
                (self.data.time - self.enable_time - case.spring_start_s) / 0.5, 0, 1
            )
            for actuator_id, force in zip(self.spring_ids, case.spring_force_n, strict=True):
                self.model.actuator_biasprm[actuator_id, 0] = 300 + (force - 300) * fraction
                self.model.actuator_forcerange[actuator_id] = (0, max(300, force))
        if not self.motors_enabled and self.data.time >= self.enable_time + 0.002 - 1e-9:
            self.set_motors_enabled(True)
        super().advance_physics_tick()
        self.update_sensors()


def build(case: Case) -> tuple[PolicyControlLoop, GainSweepAdapter]:
    """与 `record_sim2x_jitter_video.make_case` 相同的构造，只多了可钉死的模型路径。"""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    bundle = PolicyBundle.load(POLICY, session_options=opts)
    dt, delay_s = 0.001, 0.002
    timing = replace(
        bundle.contract.timing,
        physics_dt_s=dt,
        control_decimation=round(bundle.contract.timing.policy_dt_s / dt),
    )
    delay = replace(
        bundle.contract.action.delay,
        nominal_s=delay_s,
        randomized=False,
        range_s=(delay_s,) * 2,
        step_bounds=(round(delay_s / dt),) * 2,
    )
    bundle.contract = replace(
        bundle.contract, timing=timing, action=replace(bundle.contract.action, delay=delay)
    )
    runtime = PolicyRuntime(bundle, action_delay_random_seed=42)
    adapter = GainSweepAdapter(
        runtime.contract,
        artifact_path=POLICY,
        model_path=pinned_model_path(case.model),
        solver_dt_s=dt,
    )
    adapter.set_command_field("velocity_height", "height", case.height)
    adapter.set_command_field("velocity_height", "lin_vel_x", 0.0)
    adapter.set_command_field("velocity_height", "ang_vel_yaw", 0.0)
    loop = PolicyControlLoop(runtime, adapter)
    loop.reset()
    adapter.pd_period_s = 0.0
    adapter.last_pd_time = float("-inf")
    return loop, adapter


def band_rms(values: np.ndarray, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    """实车口径的 Welch 5–15 Hz 谱 RMS 与带内峰频（fs 1000、重叠一半、线性去趋势）。"""
    segment = min(nperseg, len(values))
    freq, power = welch(
        values, fs=1000, axis=0, nperseg=segment, noverlap=segment // 2, detrend="linear"
    )
    band = (freq >= 5) & (freq <= 15)
    rms = np.sqrt(np.trapezoid(power[band], freq[band], axis=0))
    return rms, freq[band][np.argmax(power[band], axis=0)]


def window_metrics(trace: np.ndarray, lo: float, hi: float) -> dict:
    """七列：四腿角度、两轮反馈速度、机身俯仰角速度；加站立 / 高度 / 力矩统计。"""
    selected = trace[(trace[:, 0] >= lo) & (trace[:, 0] < hi)]
    values = selected[:, [1, 2, 3, 4, 11, 12, 41]]
    rms, peak = band_rms(values, 900)
    _, peak_long = band_rms(values, 4000)
    return {
        "window_s": [lo, hi],
        "leg_rms_deg": np.rad2deg(rms[:4]).tolist(),
        "wheel_rms_rad_s": rms[4:6].tolist(),
        "pitch_gyro_rms_rad_s": float(rms[6]),
        "peak_5_15_hz": peak.tolist(),
        "peak_5_15_hz_nperseg4000": peak_long.tolist(),
        "minimum_upright_cos": float(np.min(np.cos(selected[:, 37]) * np.cos(selected[:, 38]))),
        "mean_height_m": float(selected[:, 46].mean()),
        "height_p95_p05_mm": float(np.diff(np.percentile(selected[:, 46], [5, 95]))[0] * 1000),
        "pitch_mean_deg": float(np.rad2deg(selected[:, 38].mean())),
        "pitch_min_max_deg": np.rad2deg([selected[:, 38].min(), selected[:, 38].max()]).tolist(),
        "vx_body_mean_m_s": float(selected[:, 43].mean()),
        "leg_effort_rms_nm": np.sqrt(np.mean(selected[:, 19:23] ** 2, axis=0)).tolist(),
        "wheel_effort_rms_nm": np.sqrt(np.mean(selected[:, 23:25] ** 2, axis=0)).tolist(),
        "effort_saturated_fraction": (
            np.abs(selected[:, 25:31]) >= np.array([32.0] * 4 + [2.210526315789474] * 2) - 1e-6
        )
        .mean(axis=0)
        .tolist(),
    }


def tracking_error(trace: np.ndarray, coeffs, lo: float, hi: float) -> dict:
    """腿静态跟踪误差 = 控制器自己的 policy 腿误差（前杆角 wrap），目标减实际。"""
    selected = trace[(trace[:, 0] >= lo) & (trace[:, 0] < hi)]
    errors = np.array(
        [policy_leg_position_error(row[13:17], row[1:5], coeffs) for row in selected[::10]]
    )
    return {
        "mean_deg": np.rad2deg(errors.mean(axis=0)).tolist(),
        "abs_mean_deg": np.rad2deg(np.abs(errors).mean(axis=0)).tolist(),
        "std_deg": np.rad2deg(errors.std(axis=0)).tolist(),
    }


def drift_metrics(times: np.ndarray, qpos: np.ndarray, lo: float, hi: float) -> dict:
    """基于 200 Hz 完整状态的机身水平漂移：使能瞬间为零点。"""
    origin = qpos[np.argmin(np.abs(times - 0.0)), :2]
    mask = (times >= lo) & (times <= hi)
    displacement = qpos[mask, :2] - origin
    distance = np.linalg.norm(displacement, axis=1)
    end = qpos[np.argmin(np.abs(times - hi)), :2] - origin
    return {
        "window_s": [lo, hi],
        "end_displacement_m": end.tolist(),
        "end_distance_m": float(np.linalg.norm(end)),
        "max_distance_m": float(distance.max()),
    }


def decay_metrics(trace: np.ndarray, floor_rms: float) -> dict:
    """注入撤除后的衰减：5–15 Hz 带通 Hilbert 包络，给出幅值比、时间常数和主频。"""
    t = trace[:, 0]
    sos = butter(4, [5, 15], btype="band", fs=1000, output="sos")
    gyro = sosfiltfilt(sos, trace[:, 41])
    envelope = np.abs(hilbert(gyro))
    smooth = np.convolve(envelope, np.ones(100) / 100, mode="same")

    def level(lo: float, hi: float) -> float:
        mask = (t >= lo) & (t < hi)
        return float(smooth[mask].mean())

    stop = PROBE_INTERVAL_S[1]
    forced = level(stop - 1.0, stop - 0.1)
    initial = level(stop + 0.1, stop + 0.3)
    after_1s = level(stop + 0.9, stop + 1.1)
    after_4s = level(stop + 3.8, stop + 4.0)
    threshold = max(2 * floor_rms * np.sqrt(2), 1e-6)
    mask = (t >= stop + 0.1) & (t < stop + 4.0)
    below = np.flatnonzero(smooth[mask] < threshold)
    settle = float(t[mask][below[0]] - stop) if below.size else None
    fit_end = stop + settle if settle is not None else stop + 4.0
    fit_mask = (t >= stop + 0.1) & (t < fit_end)
    tau = None
    if fit_mask.sum() > 50 and np.all(smooth[fit_mask] > 0):
        slope = np.polyfit(t[fit_mask], np.log(smooth[fit_mask]), 1)[0]
        tau = float(-1 / slope) if slope < 0 else None
    after = trace[(t >= stop) & (t < stop + 4.0)]
    freq, power = welch(
        after[:, [1, 2, 3, 4, 11, 12, 41]], fs=1000, axis=0, nperseg=2000, detrend="linear"
    )
    band = (freq >= 5) & (freq <= 15)
    grid = np.arange(stop, stop + 4.0, 0.25)
    return {
        "forced_envelope_rad_s": forced,
        "envelope_at_0p2s_rad_s": initial,
        "envelope_at_1s_rad_s": after_1s,
        "envelope_at_4s_rad_s": after_4s,
        "ratio_0p2s_over_forced": float(initial / forced) if forced > 0 else None,
        "ratio_1s_over_0p2s": float(after_1s / initial) if initial > 0 else None,
        "ratio_4s_over_0p2s": float(after_4s / initial) if initial > 0 else None,
        "ratio_4s_over_forced": float(after_4s / forced) if forced > 0 else None,
        "settle_time_s_to_2x_floor": settle,
        "decay_time_constant_s": tau,
        "floor_threshold_rad_s": float(threshold),
        "dominant_hz_0_4s_after": freq[band][np.argmax(power[band], axis=0)].tolist(),
        "envelope_trace_s": grid.tolist(),
        "envelope_trace_rad_s": [level(s, s + 0.25) for s in grid],
    }


def case_dir(case: Case) -> Path:
    """每个模型版本一个子目录，避免不同模型的同名工况互相覆盖。"""
    return OUT / case.model


def time_course(trace: np.ndarray, step_s: float = 2.0) -> dict:
    """逐 2 s 窗的 5–15 Hz RMS，用来区分稳定的底噪、阶跃后的新平台和持续增长。"""
    starts = np.arange(0.0, STAND_S, step_s)
    pitch, legs = [], []
    for lo in starts:
        selected = trace[(trace[:, 0] >= lo) & (trace[:, 0] < lo + step_s)]
        rms, _ = band_rms(selected[:, [1, 2, 3, 4, 41]], 900)
        legs.append(float(np.rad2deg(rms[:4].max())))
        pitch.append(float(rms[4]))
    return {
        "window_start_s": starts.tolist(),
        "pitch_gyro_rms_rad_s": pitch,
        "leg_rms_deg_max": legs,
    }


def analyze(case: Case, trace: np.ndarray, frame_times: np.ndarray, frame_qpos: np.ndarray) -> dict:
    """从已保存的 1 kHz 轨迹和 200 Hz 状态重算全部指标；与仿真过程解耦。"""
    coeffs = ((1.0, -1.0), (-1.0, 1.0))
    windows = {"startup": window_metrics(trace, 0, 4)}
    if case.probe_amplitude:
        windows["before"] = window_metrics(trace, 6, PROBE_INTERVAL_S[0])
        windows["during"] = window_metrics(trace, 10, PROBE_INTERVAL_S[1])
        windows["after"] = window_metrics(trace, 14, 18)
        static_window = (14.0, 18.0)
    else:
        windows["static"] = window_metrics(trace, *STATIC_WINDOW_S)
        static_window = STATIC_WINDOW_S
    result = {
        "windows": windows,
        "time_course": time_course(trace),
        "whole_run": {
            "minimum_upright_cos": float(np.min(np.cos(trace[:, 37]) * np.cos(trace[:, 38]))),
            "min_height_m": float(trace[:, 46].min()),
            "max_height_m": float(trace[:, 46].max()),
        },
        "tracking_error": tracking_error(trace, coeffs, *static_window),
        "drift": drift_metrics(frame_times, frame_qpos, *static_window),
        "drift_whole_run": drift_metrics(frame_times, frame_qpos, 0.0, STAND_S),
        "finite": bool(np.isfinite(trace).all()),
    }
    if case.probe_amplitude:
        result["decay"] = decay_metrics(trace, windows["before"]["pitch_gyro_rms_rad_s"])
    return result


def print_summary(case: Case, report: dict) -> None:
    """一行紧凑摘要，便于在日志里扫视。"""
    windows = report["windows"]
    summary = windows.get("static") or windows["after"]
    print(
        json.dumps(
            {
                "model": case.model,
                "case": case.name,
                "fell_at_s": report["fell_at_s"],
                "leg_rms_deg_max": max(summary["leg_rms_deg"]),
                "wheel_rms_rad_s_max": max(summary["wheel_rms_rad_s"]),
                "pitch_gyro_rms_rad_s": summary["pitch_gyro_rms_rad_s"],
                "mean_height_m": summary["mean_height_m"],
                "upright_min_cos": summary["minimum_upright_cos"],
                "drift_m": report["drift"]["end_distance_m"],
                "settle_s": report.get("decay", {}).get("settle_time_s_to_2x_floor"),
                "seconds": report["wall_time_s"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def run_case(case: Case) -> dict:
    """失能落地 3 s → 使能静站 18 s；保存轨迹与指标。"""
    start = time.monotonic()
    loop, adapter = build(case)
    loop.runtime.reset()
    adapter.set_motors_enabled(False)
    released = {"qpos": adapter.data.qpos.tolist(), "qvel": adapter.data.qvel.tolist()}
    for _ in range(round(DISABLED_S / 0.001)):
        adapter.advance_physics_tick()
    adapter.configure(case, float(adapter.data.time))
    prone = {"qpos": adapter.data.qpos.tolist(), "qvel": adapter.data.qvel.tolist()}
    adapter.start_trace(loop.runtime)
    adapter.start_frames()
    actions = []
    fell_at = None
    for _ in range(round(STAND_S / loop.runtime.contract.timing.policy_dt_s)):
        actions.append(loop.policy_step().inference.raw_action.copy())
        if fell_at is None and adapter.data.xmat[adapter.base_id].reshape(3, 3)[2, 2] < 0.5:
            fell_at = float(adapter.data.time - adapter.enable_time)
    trace = np.asarray(adapter.rows)
    trace[:, 0] -= adapter.enable_time
    frame_times = np.asarray(adapter.frame_times) - adapter.enable_time
    frame_qpos = np.asarray(adapter.frame_qpos)
    directory = case_dir(case)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / f"{case.name}.npz",
        trace=trace,
        action=np.asarray(actions),
        time=frame_times,
        qpos=frame_qpos,
        qvel=np.asarray(adapter.frame_qvel),
    )
    assert adapter.contract.action.active_rod_angle_coeffs == ((1.0, -1.0), (-1.0, 1.0))
    passive = ("lf1_Joint", "rf1_Joint", "l_coupler_Joint", "r_coupler_Joint")
    report = {
        "case": asdict(case),
        "recipe": {
            "policy": POLICY,
            "model_path": str(pinned_model_path(case.model)),
            "model_git_revision": MODEL_VERSIONS[case.model],
            "policy_hz": 50,
            "physics_tick_ms": 1.0,
            "solver_dt_ms": adapter.solver_dt_s * 1000,
            "action_delay_ms": 2.0,
            "gyro_kalman_gain": adapter.kalman_gain,
            "leg_velocity_filter_alpha": adapter.velocity_alpha,
            "motor_tau_ms": (adapter.motor_tau_s * 1000).tolist(),
            "spring_force_n": case.spring_force_n,
            "spring_start_s": case.spring_start_s,
            "measured_noise": case.measured_noise,
            "effective_forcerange": adapter.model.actuator_forcerange[
                adapter._actuator_ids
            ].tolist(),
            "policy_dof_armature": adapter.model.dof_armature[
                adapter._joint_qvel_addresses
            ].tolist(),
            "policy_dof_damping": adapter.model.dof_damping[adapter._joint_qvel_addresses].tolist(),
            "passive_dof_armature": [
                float(adapter.model.dof_armature[adapter.model.joint(name).dofadr[0]])
                for name in passive
            ],
            "gain_override": (
                "adapter._controller 用改过 stiffness/damping 的契约副本重建；runtime 契约不变"
            ),
            "controller_gains": {
                "stiffness": adapter._controller._stiffness.tolist(),
                "damping": adapter._controller._damping.tolist(),
            },
            "runtime_contract_gains": {
                "stiffness": list(loop.runtime.contract.robot.actuators.stiffness),
                "damping": list(loop.runtime.contract.robot.actuators.damping),
            },
        },
        "released_state": released,
        "disabled_before_enable": prone,
        "enable_time_s": adapter.enable_time,
        "fell_at_s": fell_at,
        "warnings": {str(i): int(w.number) for i, w in enumerate(adapter.data.warning) if w.number},
        "wall_time_s": time.monotonic() - start,
    }
    report.update(analyze(case, trace, frame_times, frame_qpos))
    (directory / f"{case.name}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print_summary(case, report)
    return report


def recompute_all() -> None:
    """不重跑仿真，只用已保存的 npz 重算每条工况的指标并覆写 JSON。"""
    for path in sorted(OUT.glob("*/*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        case = Case(**report["case"])
        arrays = np.load(path.with_suffix(".npz"))
        report.update(analyze(case, arrays["trace"], arrays["time"], arrays["qpos"]))
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print_summary(case, report)


def verify_probe_sim_replica() -> dict:
    """与 probe_sim 已保存结果比较 before/during/after 窗的谱 RMS，证明配方复现无误。"""
    reference = json.loads(PROBE_SIM_REFERENCE.read_text(encoding="utf-8"))
    case = probe_sim_replica_case()
    run_case(case)
    trace = np.load(case_dir(case) / f"{case.name}.npz")["trace"]
    comparison = {}
    for label, lo, hi in (("before", 6, 8), ("during", 10, 12), ("after", 14, 18)):
        tail = trace[(trace[:, 0] >= lo) & (trace[:, 0] < hi)]
        freq, power = welch(
            tail[:, np.r_[1:5, 11:13, 37:40]],
            fs=1000,
            axis=0,
            nperseg=min(4000, len(tail)),
            detrend="linear",
        )
        band = (freq >= 5) & (freq <= 15)
        rms = np.sqrt(np.trapezoid(power[band], freq[band], axis=0))
        expected = np.asarray(reference["windows"][label]["rms_5_15hz"])
        comparison[label] = {
            "replica_rms": rms.tolist(),
            "probe_sim_rms": expected.tolist(),
            "max_relative_difference": float(
                np.max(np.abs(rms - expected) / np.maximum(expected, 1e-12))
            ),
            "height_mean_replica": float(tail[:, 46].mean()),
            "height_mean_probe_sim": reference["windows"][label]["height_mean"],
        }
    (OUT / "probe_sim_replica_check.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(comparison, ensure_ascii=False), flush=True)
    return comparison


REAL_REFERENCE = {
    "source": ".scratch/passive_armature_ablation_20261012/comparison.json (burst_0003, Welch 900/450)",
    "leg_rms_deg_range": [0.55, 0.64],
    "wheel_rms_rad_s_range": [2.8, 3.3],
    "pitch_gyro_rms_rad_s": 0.35,
    "frequency_hz_range": [8.14, 8.35],
}
GROUP_ORDER = [f"gain_{g}" for g in GAIN_GROUPS] + [f"height_{h}" for h in HEIGHT_GROUPS]


def load_report(model: str, name: str) -> dict | None:
    """读取一条已完成工况；缺失时返回 None，汇总时标记为未跑。"""
    path = OUT / model / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def stand_verdict(static: dict, height: float) -> dict:
    """站住判据：没摔、静站窗倾斜余弦 ≥ 0.95、高度误差 ≤ 4 cm；漂移另按 0.5 m 单独标记。"""
    window = static["windows"]["static"]
    drift = static["drift"]["end_distance_m"]
    stood = (
        static["fell_at_s"] is None
        and window["minimum_upright_cos"] >= 0.95
        and abs(window["mean_height_m"] - height) <= 0.04
    )
    return {
        "stood": bool(stood),
        "fell_at_s": static["fell_at_s"],
        "drift_over_0p5m": bool(drift > 0.5),
        "height_error_m": float(window["mean_height_m"] - height),
    }


def group_row(model: str, name: str) -> dict | None:
    """一组 = 静站工况 + 注入工况，按表格列顺序整理。"""
    static = load_report(model, name)
    probe = load_report(model, name + "_probe")
    if static is None:
        return None
    case = static["case"]
    window = static["windows"]["static"]
    row = {
        "group": name,
        "label": case["label"],
        "model": model,
        "leg_kp": case["leg_kp"],
        "leg_kd": case["leg_kd"],
        "wheel_kd": case["wheel_kd"],
        "height_cmd_m": case["height"],
        "verdict": stand_verdict(static, case["height"]),
        "upright_min_cos_static": window["minimum_upright_cos"],
        "upright_min_cos_whole_run": static["whole_run"]["minimum_upright_cos"],
        "mean_height_m": window["mean_height_m"],
        "height_p95_p05_mm": window["height_p95_p05_mm"],
        "pitch_mean_deg": window["pitch_mean_deg"],
        "drift_end_m_static_window": static["drift"]["end_distance_m"],
        "drift_max_m_whole_run": static["drift_whole_run"]["max_distance_m"],
        "drift_end_m_whole_run": static["drift_whole_run"]["end_distance_m"],
        "vx_body_mean_m_s": window["vx_body_mean_m_s"],
        "leg_rms_deg": window["leg_rms_deg"],
        "leg_rms_deg_max": max(window["leg_rms_deg"]),
        "wheel_rms_rad_s": window["wheel_rms_rad_s"],
        "wheel_rms_rad_s_max": max(window["wheel_rms_rad_s"]),
        "pitch_gyro_rms_rad_s": window["pitch_gyro_rms_rad_s"],
        "peak_5_15_hz": window["peak_5_15_hz"],
        "peak_5_15_hz_nperseg4000": window["peak_5_15_hz_nperseg4000"],
        "leg_tracking_error_mean_deg": static["tracking_error"]["mean_deg"],
        "leg_tracking_error_abs_mean_deg": static["tracking_error"]["abs_mean_deg"],
        "leg_effort_rms_nm": window["leg_effort_rms_nm"],
        "effort_saturated_fraction": window["effort_saturated_fraction"],
        "time_course": static["time_course"],
        "warnings": static["warnings"],
        "probe": None,
    }
    if probe is not None:
        decay = probe["decay"]
        row["probe"] = {
            "fell_at_s": probe["fell_at_s"],
            "before_pitch_gyro_rms_rad_s": probe["windows"]["before"]["pitch_gyro_rms_rad_s"],
            "forced_envelope_rad_s": decay["forced_envelope_rad_s"],
            "envelope_at_0p2s_rad_s": decay["envelope_at_0p2s_rad_s"],
            "envelope_at_1s_rad_s": decay["envelope_at_1s_rad_s"],
            "envelope_at_4s_rad_s": decay["envelope_at_4s_rad_s"],
            "during_pitch_gyro_rms_rad_s": probe["windows"]["during"]["pitch_gyro_rms_rad_s"],
            "during_leg_rms_deg_max": max(probe["windows"]["during"]["leg_rms_deg"]),
            "during_wheel_rms_rad_s_max": max(probe["windows"]["during"]["wheel_rms_rad_s"]),
            "after_pitch_gyro_rms_rad_s": probe["windows"]["after"]["pitch_gyro_rms_rad_s"],
            "after_leg_rms_deg_max": max(probe["windows"]["after"]["leg_rms_deg"]),
            "after_upright_min_cos": probe["windows"]["after"]["minimum_upright_cos"],
            "after_drift_end_m": probe["drift"]["end_distance_m"],
            "ratio_0p2s_over_forced": decay["ratio_0p2s_over_forced"],
            "ratio_1s_over_0p2s": decay["ratio_1s_over_0p2s"],
            "ratio_4s_over_0p2s": decay["ratio_4s_over_0p2s"],
            "ratio_4s_over_forced": decay["ratio_4s_over_forced"],
            "settle_time_s_to_2x_floor": decay["settle_time_s_to_2x_floor"],
            "decay_time_constant_s": decay["decay_time_constant_s"],
            "dominant_hz_pitch_0_4s_after": decay["dominant_hz_0_4s_after"][6],
            "dominant_hz_legs_0_4s_after": decay["dominant_hz_0_4s_after"][:4],
            "envelope_trace_s": decay["envelope_trace_s"],
            "envelope_trace_rad_s": decay["envelope_trace_rad_s"],
        }
    return row


def baseline_check() -> dict:
    """主模型 A 组与 old_measured 逐项比值，超过 20% 的项连同对齐阶梯一起给出。"""
    reference = json.loads(REFERENCE_BASELINE.read_text(encoding="utf-8"))
    old = reference["simulation"]["old_measured"]
    ladder = {}
    for name in ("align_nolag_spring300", "align_lag12_spring300", "gain_A"):
        report = load_report(PRIMARY_MODEL, name)
        if report is None:
            continue
        window = report["windows"]["static"]
        ratios = {
            "leg_rms_deg_max": max(window["leg_rms_deg"]) / max(old["leg_rms_deg"]),
            "leg_rms_deg_mean": float(np.mean(window["leg_rms_deg"]) / np.mean(old["leg_rms_deg"])),
            "wheel_rms_rad_s_max": max(window["wheel_rms_rad_s"]) / max(old["wheel_rms_rad_s"]),
            "pitch_gyro_rms_rad_s": window["pitch_gyro_rms_rad_s"] / old["pitch_gyro_rms_rad_s"],
            "mean_height_m": window["mean_height_m"] / old["mean_height_m"],
            "minimum_upright_cos": window["minimum_upright_cos"] / old["minimum_upright_cos"],
        }
        ladder[name] = {
            "label": report["case"]["label"],
            "motor_tau_ms": report["case"]["motor_tau_ms"],
            "spring_force_n": report["case"]["spring_force_n"],
            "leg_rms_deg": window["leg_rms_deg"],
            "wheel_rms_rad_s": window["wheel_rms_rad_s"],
            "pitch_gyro_rms_rad_s": window["pitch_gyro_rms_rad_s"],
            "peak_5_15_hz": window["peak_5_15_hz"],
            "mean_height_m": window["mean_height_m"],
            "minimum_upright_cos": window["minimum_upright_cos"],
            "ratio_to_old_measured": ratios,
            "over_20_percent": [k for k, v in ratios.items() if abs(v - 1) > 0.2],
        }
    return {
        "reference": "old_measured 静站 14–24 s 窗（Welch 900/450，5–15 Hz）",
        "old_measured": {
            k: old[k]
            for k in (
                "leg_rms_deg",
                "wheel_rms_rad_s",
                "pitch_gyro_rms_rad_s",
                "peak_5_15_hz",
                "minimum_upright_cos",
                "mean_height_m",
            )
        },
        "old_measured_recipe": "4e1aeb0 模型（被动 armature 0.005）、dt 0.2 ms、腿 ±40 Nm、相位拟合滤波、无电机滞后、弹簧 300 N、实测噪声",
        "this_script_window_s": list(STATIC_WINDOW_S),
        "ladder": ladder,
        "passes": bool(ladder.get("align_nolag_spring300", {}).get("over_20_percent") == []),
    }


def fmt(value, digits=3):
    """表格数字格式；None 显示为 —。"""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    return f"{value:.{digits}f}"


def fmt_list(values, digits=3):
    return "/".join(fmt(v, digits) for v in values)


def markdown_tables(rows_by_model: dict[str, list[dict]], check: dict) -> str:
    """生成文档用的表格片段：主表、注入表、模型敏感性表、基线校验表。"""
    lines = []
    primary = rows_by_model[PRIMARY_MODEL]
    lines.append("### 表 1  静站预测（主模型 4e1aeb0，静站窗 8–18 s，Welch 900/450，5–15 Hz）\n")
    lines.append(
        "| 组 | 腿 Kp/Kd | 轮 Kd | 高度指令 | 站住 | upright_min_cos | 平均高度 m | 漂移 m（8–18 s / 全程） | vx 均值 m/s | 腿 5–15 Hz RMS ° LF/LB/RF/RB | 轮 RMS rad/s LW/RW | 俯仰角速度 RMS rad/s | 谱峰 Hz（900 / 4000 点） | 腿静态跟踪误差 ° LF/LB/RF/RB |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in primary:
        verdict = row["verdict"]
        stood = (
            "是"
            if verdict["stood"]
            else f"否（{'摔倒 ' + fmt(verdict['fell_at_s'], 1) + ' s' if verdict['fell_at_s'] is not None else '姿态/高度不达标'}）"
        )
        if verdict["drift_over_0p5m"]:
            stood += "，漂移>0.5 m"
        lines.append(
            f"| {row['group']} | {row['leg_kp']:g}/{row['leg_kd']:g} | {row['wheel_kd']:g} | {row['height_cmd_m']:.2f} | {stood} | "
            f"{fmt(row['upright_min_cos_static'], 4)} | {fmt(row['mean_height_m'], 3)} | {fmt(row['drift_end_m_static_window'], 2)} / {fmt(row['drift_end_m_whole_run'], 2)} | "
            f"{fmt(row['vx_body_mean_m_s'], 3)} | {fmt_list(row['leg_rms_deg'], 3)} | {fmt_list(row['wheel_rms_rad_s'], 3)} | {fmt(row['pitch_gyro_rms_rad_s'], 4)} | "
            f"{fmt(max(row['peak_5_15_hz'][:4]), 1)} / {fmt(float(np.median(row['peak_5_15_hz_nperseg4000'])), 2)} | {fmt_list(row['leg_tracking_error_mean_deg'], 2)} |"
        )
    real = REAL_REFERENCE
    lines.append(
        f"| 实车基线（Student128-6998，A 组，0.22 m） | 60/3 | 0.2 | 0.22 | 是（自持振荡） | — | — | — | — | "
        f"{real['leg_rms_deg_range'][0]:.2f}–{real['leg_rms_deg_range'][1]:.2f} | {real['wheel_rms_rad_s_range'][0]:.1f}–{real['wheel_rms_rad_s_range'][1]:.1f} | {real['pitch_gyro_rms_rad_s']:.2f} | {real['frequency_hz_range'][0]:.2f}–{real['frequency_hz_range'][1]:.2f} | — |"
    )
    lines.append("")
    lines.append("### 表 2  8.3 Hz 俯仰角速度观测注入（0.4 rad/s，8–12 s）撤除后的衰减（主模型）\n")
    lines.append(
        "| 组 | 注入前俯仰 RMS rad/s（6–8 s） | 注入中俯仰 RMS rad/s（10–12 s） | 受迫包络 rad/s（11–11.9 s） | "
        "包络 0.2 s / 1 s / 4 s 后 rad/s | 即时残留比（0.2 s / 受迫） | 1 s 衰减比 | 4 s 衰减比 | "
        "回到 2× 底噪用时 s | 衰减时间常数 s | 撤除后 0–4 s 主频 Hz（俯仰） | 撤除后 14–18 s 腿 RMS ° max | 注入段站住 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in primary:
        probe = row["probe"]
        if probe is None:
            continue
        stood = probe["fell_at_s"] is None and probe["after_upright_min_cos"] >= 0.95
        lines.append(
            f"| {row['group']} | {fmt(probe['before_pitch_gyro_rms_rad_s'], 4)} | "
            f"{fmt(probe['during_pitch_gyro_rms_rad_s'], 4)} | {fmt(probe['forced_envelope_rad_s'], 4)} | "
            f"{fmt(probe['envelope_at_0p2s_rad_s'], 4)} / {fmt(probe['envelope_at_1s_rad_s'], 4)} / "
            f"{fmt(probe['envelope_at_4s_rad_s'], 4)} | {fmt(probe['ratio_0p2s_over_forced'], 2)} | "
            f"{fmt(probe['ratio_1s_over_0p2s'], 2)} | {fmt(probe['ratio_4s_over_0p2s'], 2)} | "
            f"{fmt(probe['settle_time_s_to_2x_floor'], 2)} | {fmt(probe['decay_time_constant_s'], 2)} | "
            f"{fmt(probe['dominant_hz_pitch_0_4s_after'], 1)} | {fmt(probe['after_leg_rms_deg_max'], 3)} | "
            f"{'是' if stood else '否'} |"
        )
    lines.append("")
    lines.append(
        "### 表 3  模型版本敏感性（同一配方，腿 RMS 取四关节最大值，轮 RMS 取两轮最大值）\n"
    )
    header = (
        "| 组 | "
        + " | ".join(
            f"{m}：站住 / 漂移 m / 腿 RMS ° / 轮 RMS rad/s / 俯仰 RMS rad/s / 高度 m"
            for m in rows_by_model
        )
        + " |"
    )
    lines.append(header)
    lines.append("|---|" + "---|" * len(rows_by_model))
    for row in primary:
        cells = []
        for rows in rows_by_model.values():
            other = next((r for r in rows if r["group"] == row["group"]), None)
            if other is None:
                cells.append("未跑")
                continue
            verdict = other["verdict"]
            flag = "是" if verdict["stood"] else "否"
            cells.append(
                f"{flag} / {fmt(other['drift_end_m_static_window'], 2)} / {fmt(other['leg_rms_deg_max'], 3)} / {fmt(other['wheel_rms_rad_s_max'], 3)} / {fmt(other['pitch_gyro_rms_rad_s'], 4)} / {fmt(other['mean_height_m'], 3)}"
            )
        lines.append(f"| {row['group']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("### 表 4  基线校验：A 组（主模型）对 old_measured 的逐项比值\n")
    lines.append(
        "| 工况 | 电机滞后 | 弹簧 N | 腿 RMS ° LF/LB/RF/RB | 轮 RMS rad/s | 俯仰 RMS rad/s | 高度 m | 比值：腿 max / 轮 max / 俯仰 / 高度 | 超 20% 的项 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    old = check["old_measured"]
    lines.append(
        f"| old_measured（参考） | 无 | 300 | {fmt_list(old['leg_rms_deg'], 4)} | {fmt_list(old['wheel_rms_rad_s'], 4)} | {fmt(old['pitch_gyro_rms_rad_s'], 5)} | {fmt(old['mean_height_m'], 4)} | 1 / 1 / 1 / 1 | — |"
    )
    for name, item in check["ladder"].items():
        ratios = item["ratio_to_old_measured"]
        lines.append(
            f"| {name}（{item['label']}） | {item['motor_tau_ms']:g} ms | {item['spring_force_n'] or '300'} | {fmt_list(item['leg_rms_deg'], 4)} | {fmt_list(item['wheel_rms_rad_s'], 4)} | {fmt(item['pitch_gyro_rms_rad_s'], 5)} | {fmt(item['mean_height_m'], 4)} | "
            f"{fmt(ratios['leg_rms_deg_max'], 2)} / {fmt(ratios['wheel_rms_rad_s_max'], 2)} / {fmt(ratios['pitch_gyro_rms_rad_s'], 2)} / {fmt(ratios['mean_height_m'], 3)} | {', '.join(item['over_20_percent']) or '无'} |"
        )
    return "\n".join(lines) + "\n"


def summarize() -> None:
    """汇总为 prediction.json 与文档表格片段 summary_tables.md。"""
    rows_by_model = {}
    for model in MODEL_VERSIONS:
        rows = [row for name in GROUP_ORDER if (row := group_row(model, name)) is not None]
        if rows:
            rows_by_model[model] = rows
    check = baseline_check()
    replica = (
        json.loads((OUT / "probe_sim_replica_check.json").read_text(encoding="utf-8"))
        if (OUT / "probe_sim_replica_check.json").exists()
        else None
    )
    primary = rows_by_model[PRIMARY_MODEL]
    prediction = {
        "date": "2026-10-13",
        "policy": POLICY,
        "primary_model": PRIMARY_MODEL,
        "model_versions": {
            name: {"git_revision": rev, "path": str(pinned_model_path(name))}
            for name, rev in MODEL_VERSIONS.items()
        },
        "recipe": primary[0] and load_report(PRIMARY_MODEL, primary[0]["group"])["recipe"],
        "protocol": {
            "disabled_fall_s": DISABLED_S,
            "stand_s": STAND_S,
            "static_window_s": list(STATIC_WINDOW_S),
            "probe": {
                "amplitude_rad_s": 0.4,
                "hz": 8.3,
                "interval_s": list(PROBE_INTERVAL_S),
                "channel": "观测 base_angular_velocity_body[1]",
            },
            "spectral": {
                "method": "Welch",
                "fs": 1000,
                "nperseg": 900,
                "noverlap": 450,
                "band_hz": [5, 15],
            },
            "stand_criteria": "fell_at 为空且静站窗 upright_min_cos ≥ 0.95 且 |平均高度−指令| ≤ 0.04 m；漂移 > 0.5 m 单独标记",
            "decay": "5–15 Hz 四阶 Butterworth 零相位带通 + Hilbert 包络（100 ms 滑动平均）；4 s 衰减比 = 撤除后 3.8–4.0 s 包络 / 0.1–0.3 s 包络",
        },
        "real_reference": REAL_REFERENCE,
        "probe_sim_replica_check": replica,
        "baseline_check": check,
        "groups": rows_by_model,
        "unstable_groups": {
            model: [
                r["group"]
                for r in rows
                if not r["verdict"]["stood"] or r["verdict"]["drift_over_0p5m"]
            ]
            for model, rows in rows_by_model.items()
        },
    }
    (OUT / "prediction.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "summary_tables.md").write_text(markdown_tables(rows_by_model, check), encoding="utf-8")
    print(
        json.dumps(
            {
                "unstable_groups": prediction["unstable_groups"],
                "baseline_check_passes": check["passes"],
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("verify", "align", "matrix", "recompute", "summarize", "all"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_VERSIONS), choices=list(MODEL_VERSIONS)
    )
    parser.add_argument("--only", action="append", default=None, help="只跑指定工况名")
    parser.add_argument("--refresh-models", action="store_true", help="重新抽取钉死的 MJCF 快照")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.refresh_models and (OUT / "mjcf").exists():
        shutil.rmtree(OUT / "mjcf")
    for version in MODEL_VERSIONS:
        pinned_model_path(version)
    if args.stage in ("verify", "all"):
        verify_probe_sim_replica()
    cases: list[Case] = []
    if args.stage in ("align", "all"):
        cases += alignment_cases()
    if args.stage in ("matrix", "all"):
        for model in args.models:
            cases += matrix(model)
    if args.only:
        cases = [case for case in cases if case.name in args.only]
    if cases:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_case, case) for case in cases]
            for future in as_completed(futures):
                future.result()
    if args.stage == "recompute":
        recompute_all()
    if args.stage in ("recompute", "summarize", "all"):
        summarize()


if __name__ == "__main__":
    main()
