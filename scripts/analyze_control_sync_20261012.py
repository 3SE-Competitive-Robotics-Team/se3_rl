"""离线分析板内同步采集；不连接硬件、不修改策略或控制参数。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from se3_shared.fourbar import knee_gas_spring_compensation_torque_np

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = Path(
    "D:/RoboMaster/serialleg2026/build/control-sync-captures/standing_20261012_reconnected"
)
OUT = ROOT / ".scratch/control_sync_analysis_20261012"
NAMES = ["LF", "RF", "LB", "RB", "LW", "RW"]
ORDER = [0, 2, 1, 3]


def clock_diff(a, b):
    """无符号 DWT 差值，处理 32 位计数器回绕。"""
    return (np.asarray(a, dtype=np.int64) - np.asarray(b, dtype=np.int64)) & 0xFFFFFFFF


def signed_clock_diff(a, b):
    """小于半个回绕周期的有符号时差。"""
    return (
        np.asarray(a, dtype=np.int64) - np.asarray(b, dtype=np.int64) + 0x80000000
    ) % 0x100000000 - 0x80000000


def stat(a):
    """逐通道保留分位数和极值。"""
    a = np.asarray(a)
    return {
        "p01": np.quantile(a, 0.01, axis=0).tolist(),
        "p50": np.median(a, axis=0).tolist(),
        "p99": np.quantile(a, 0.99, axis=0).tolist(),
        "max": np.max(a, axis=0).tolist(),
    }


def harmonic(t, y, f, harmonics=1):
    """带线性趋势的谐波回归；复系数采用实部乘 exp(iωt) 约定。"""
    t, y = np.asarray(t), np.asarray(y)
    cols = [np.ones_like(t), t - np.mean(t)]
    for k in range(1, harmonics + 1):
        cols += [np.cos(2 * np.pi * f * k * t), np.sin(2 * np.pi * f * k * t)]
    design = np.column_stack(cols)
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    variance = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
    residual_ratio = np.divide(
        np.sum(residual**2, axis=0),
        variance,
        out=np.full_like(variance, np.nan, dtype=float),
        where=variance > 1e-20,
    )
    return (
        beta[2] - 1j * beta[3],
        1 - residual_ratio,
        beta,
        residual,
    )


def ratio(num, den, f):
    """只报告信号间相位差；等效毫秒数不等于已识别的纯延迟。"""
    value = num / den
    phase = np.angle(value)
    return {
        "gain": abs(value).tolist(),
        "phase_deg": np.rad2deg(phase).tolist(),
        "phase_equivalent_lag_ms": (-phase / (2 * np.pi * f) * 1000).tolist(),
    }


def load(path):
    """把原始 CAN、实际 PD 输入和时钟还原到同一段时间。"""
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    data = dict(np.load(path))
    mon = meta["monitor_before_export"]
    hz = mon["cpu_hz"]
    data["input_time"] = data["time_s"] + clock_diff(data["input_cycle"], data["cycle"]) / hz
    data["rx_time"] = (
        data["input_time"][:, None]
        - clock_diff(data["input_cycle"][:, None], data["rx_cycle"]) / hz
    )
    data["rx_sof_time"] = data["rx_time"] - data["rx_sof_age_us"] / 1e6
    data["q"] = np.unwrap(np.angle(np.exp(1j * (data["position"] + mon["offset"]))), axis=0)
    raw = data["rx_data"].reshape(-1, 6, 8)
    raw_q = (raw[:, :4, 1] * 256 + raw[:, :4, 2]) / 65535 * (2 * np.pi) - np.pi
    raw_v = (raw[:, :4, 3] * 16 + (raw[:, :4, 4] >> 4)) / 4095 * 90 - 45
    leg_current = ((raw[:, :4, 4] & 15) * 256 + raw[:, :4, 5]) / 4095 * 108 - 54
    wheel_current = raw[:, 4:, 4] * 256 + raw[:, 4:, 5]
    wheel_current = (wheel_current + 32768) % 65536 - 32768
    wheel_rpm = raw[:, 4:, 2] * 256 + raw[:, 4:, 3]
    wheel_rpm = (wheel_rpm + 32768) % 65536 - 32768
    data["current_torque"] = np.column_stack(
        [leg_current, -wheel_current * 20 / 16384 * 0.3 * 14 / 19.2]
    )
    data["wheel_feedback_velocity"] = -wheel_rpm * 6.28 / 60 / 14
    data["wheel_q"] = (
        -np.unwrap((raw[:, 4:, 0] * 256 + raw[:, 4:, 1]) / 8192 * 2 * np.pi, axis=0) / 14
    )
    data["raw_q_error"] = np.max(abs(raw_q - data["position"]), axis=0)
    data["raw_v_error"] = np.max(abs(raw_v - data["velocity"]), axis=0)
    data["P_target"] = mon["leg_kp"] * data["target"][:, :4]
    data["P_position"] = -mon["leg_kp"] * data["q"]
    data["D"] = -mon["leg_kd"] * data["velocity"]
    data["spring"] = knee_gas_spring_compensation_torque_np(data["q"][:, ORDER], 300)[:, ORDER]
    err = data["target"][:, :4] - data["q"]
    err[:, :2] = np.angle(np.exp(1j * err[:, :2]))
    predicted = np.column_stack(
        [
            np.clip(mon["leg_kp"] * err + data["D"] + data["spring"], -40, 40),
            np.clip(0.2 * (data["target"][:, 4:] - data["wheel_velocity"]), -2.21, 2.21),
        ]
    )
    enabled = (data["flags"] & 8) != 0
    predicted[~enabled] = 0
    data["formula_error"] = np.max(abs(predicted - data["torque"]), axis=0)
    return data, meta


def timing(data, meta, initial):
    """区分 MCU 周期、数据年龄、总线时延和累计错误计数。"""
    hz = meta["monitor_before_export"]["cpu_hz"]
    valid_tx = data["tx_sof_cycle"] != 0
    tx_stats = []
    tx_latency = []
    for j in range(5):
        ok = valid_tx[:, j]
        tx_stats.append(stat(data["tx_wait_us"][ok, j]))
        source = data["tx_sequence"][ok, j]
        tx_latency.append(
            stat(clock_diff(data["tx_sof_cycle"][ok, j], data["input_cycle"][source]) / hz * 1000)
        )
    policies = meta["policies"]
    policy_ms = [int(clock_diff(p["end_cycle"], p["begin_cycle"])) / hz * 1000 for p in policies]
    counters = [
        "rx_full",
        "rx_lost",
        "tx_fail",
        "tx_abort",
        "tx_event_lost",
        "tx_unmatched",
        "tx_invalid_time",
    ]
    mon = meta["monitor_before_export"]
    result = {
        "summary": meta["summary"],
        "flags": {
            str(k): int(v)
            for k, v in zip(*np.unique(data["flags"], return_counts=True), strict=True)
        },
        "rx_sof_to_decode_us": stat(data["rx_sof_age_us"]),
        "rx_sof_to_control_input_ms": stat(
            (data["input_time"][:, None] - data["rx_sof_time"]) * 1000
        ),
        "tx_queue_us": tx_stats,
        "control_input_to_tx_sof_ms": tx_latency,
        "new_errors_since_armed": {
            key: (np.asarray(mon[key]) - initial[key]).tolist() for key in counters
        },
        "policy_compute_ms": stat(policy_ms),
        "policy_period_ms": stat(np.diff([p["input_tick"] for p in policies]))
        if len(policies) > 1
        else None,
        "imu_age_ms": stat(clock_diff(data["input_cycle"], data["imu_cycle"]) / hz * 1000),
        "feedback_frame_delta": (data["rx_sequence"][-1] - data["rx_sequence"][0]).tolist(),
        "control_formula_max_error_nm": data["formula_error"].tolist(),
        "raw_position_decode_max_error": data["raw_q_error"].tolist(),
        "raw_velocity_decode_max_error": data["raw_v_error"].tolist(),
    }
    return result


def steady_analysis(data, meta):
    """测量稳定段的主振频率、传感器相对相位和控制力矩功率分量。"""
    t = data["input_time"]
    gyro = data["gyro"][:, 0]
    freqs = np.linspace(5, 12, 701)
    score = [np.sum(harmonic(t, gyro, f)[3] ** 2) for f in freqs]
    best = int(np.argmin(score))
    opt = minimize_scalar(
        lambda f: np.sum(harmonic(t, gyro, f)[3] ** 2),
        bounds=(freqs[max(0, best - 2)], freqs[min(len(freqs) - 1, best + 2)]),
        method="bounded",
    )
    f = float(opt.x)
    omega = 2 * np.pi * f
    hq = np.array([harmonic(data["rx_sof_time"][:, j], data["q"][:, j], f, 3)[0] for j in range(4)])
    hv = np.array(
        [harmonic(data["rx_sof_time"][:, j], data["velocity"][:, j], f, 3)[0] for j in range(4)]
    )
    hwq = np.array(
        [harmonic(data["rx_sof_time"][:, 4 + j], data["wheel_q"][:, j], f, 3)[0] for j in range(2)]
    )
    hwv = np.array(
        [
            harmonic(data["rx_sof_time"][:, 4 + j], data["wheel_feedback_velocity"][:, j], f, 3)[0]
            for j in range(2)
        ]
    )
    hcurrent = np.array(
        [
            harmonic(data["rx_sof_time"][:, j], data["current_torque"][:, j], f, 3)[0]
            for j in range(6)
        ]
    )
    hcmd_tx = []
    for j in range(6):
        slot = min(j, 4)
        unique = np.r_[True, np.diff(data["tx_sequence"][:, slot]) != 0] & (
            data["tx_sof_cycle"][:, slot] != 0
        )
        source = data["tx_sequence"][unique, slot]
        tx_t = (
            data["time_s"][unique]
            + signed_clock_diff(data["tx_sof_cycle"][unique, slot], data["cycle"][unique])
            / meta["monitor_before_export"]["cpu_hz"]
        )
        hcmd_tx.append(harmonic(tx_t, data["torque"][source, j], f, 3)[0])
    hcmd_tx = np.array(hcmd_tx)
    h = {
        key: harmonic(t, data[key], f, 3)[0]
        for key in ("P_target", "P_position", "D", "spring", "target", "torque", "gyro", "accel")
    }
    hq_control = harmonic(t, data["q"], f, 3)[0]
    hphysical_v = 1j * omega * np.r_[hq, hwq]
    power = {
        key: (0.5 * np.real(h[key] * np.conj(1j * omega * hq_control))).tolist()
        for key in ("P_target", "P_position", "D", "spring")
    }
    power["total_command_at_tx"] = (0.5 * np.real(hcmd_tx * np.conj(hphysical_v))).tolist()
    power["current_feedback_torque_proxy"] = (
        0.5 * np.real(hcurrent * np.conj(hphysical_v))
    ).tolist()
    power["wheel_target_term"] = (
        0.5 * np.real(0.2 * h["target"][4:] * np.conj(hphysical_v[4:]))
    ).tolist()
    power["wheel_speed_feedback_term"] = (
        0.5
        * np.real(-0.2 * harmonic(t, data["wheel_velocity"], f, 3)[0] * np.conj(hphysical_v[4:]))
    ).tolist()
    per_half = []
    for lower, upper in ((0.02, 0.47), (0.47, 0.92)):
        mask = (t >= lower) & (t < upper)
        qh = np.array(
            [harmonic(data["rx_sof_time"][mask, j], data["q"][mask, j], f, 2)[0] for j in range(4)]
        )
        vh = np.array(
            [
                harmonic(data["rx_sof_time"][mask, j], data["velocity"][mask, j], f, 2)[0]
                for j in range(4)
            ]
        )
        per_half.append(ratio(vh, 1j * omega * qh, f))
    p = meta["policies"]
    pt = (
        np.array([item["input_tick"] - meta["monitor_before_export"]["start_tick"] for item in p])
        / 1000
    )
    pg = np.array([item["input"]["angular_velocity"] for item in p])
    grav = np.array([item["input"]["gravity"] for item in p])
    pitch = np.arctan2(grav[:, 0], -grav[:, 2])
    hpolicy_gyro = harmonic(pt, pg[:, 1], f)[0]
    hpitch = harmonic(pt, pitch, f)[0]
    result = {
        "frequency_hz": f,
        "cycles_observed": float((t[-1] - t[0]) * f),
        "gyro_pitch_fit_r2": float(harmonic(t, gyro, f)[1]),
        "harmonic_amplitude": {
            **{key: abs(value).tolist() for key, value in h.items()},
            "position_rad": abs(hq).tolist(),
            "velocity_feedback_rad_s": abs(hv).tolist(),
            "position_derivative_rad_s": abs(1j * omega * hq).tolist(),
            "current_torque_nm": abs(hcurrent).tolist(),
        },
        "leg_velocity_vs_position_derivative": ratio(hv, 1j * omega * hq, f),
        "leg_velocity_vs_position_derivative_halves": per_half,
        "wheel_velocity_vs_position_derivative": ratio(hwv, 1j * omega * hwq, f),
        "feedback_current_vs_command_at_successful_tx": ratio(hcurrent, hcmd_tx, f),
        "policy_gyro_vs_raw_gyro": ratio(hpolicy_gyro, h["gyro"][0], f),
        "policy_pitch_derivative_vs_gyro": ratio(1j * omega * hpitch, hpolicy_gyro, f),
        "harmonic_power_w": power,
        "leg_torque_saturation_fraction": np.mean(
            abs(data["torque"][:, :4]) >= 39.999, axis=0
        ).tolist(),
        "wheel_torque_saturation_fraction": np.mean(
            abs(data["torque"][:, 4:]) >= 2.20999, axis=0
        ).tolist(),
        "note": "相位等效时差不是唯一可辨识的纯延迟；同一电机的位置和速度可能有共同延迟，电流折算力矩不是独立轴力矩。功率为该频率分量，不能当总机械功率。",
    }
    return result


def plot(data, result):
    """输出可复核的稳定段曲线；各曲线均使用本段实际时间。"""
    t = data["input_time"]
    f = result["frequency_hz"]
    fig, axes = plt.subplots(4, 2, figsize=(15, 12), constrained_layout=True)
    axes[0, 0].plot(t, data["gyro"][:, 0])
    axes[0, 0].set(title=f"Raw pitch gyro: {f:.3f} Hz", ylabel="rad/s")
    axes[0, 1].plot(t, data["target"][:, 4:], label=["LW target", "RW target"])
    axes[0, 1].plot(t, data["wheel_velocity"], alpha=0.7, label=["LW measured", "RW measured"])
    axes[0, 1].set(title="Wheel velocity targets and feedback", ylabel="rad/s")
    axes[0, 1].legend(ncol=2)
    for j in range(4):
        ax = axes[1 + j // 2, j % 2]
        _, _, beta, _ = harmonic(data["rx_sof_time"][:, j], data["q"][:, j], f, 3)
        derivative = np.full_like(t, beta[1])
        for k in range(1, 4):
            qk = beta[2 * k] - 1j * beta[2 * k + 1]
            derivative += np.real(1j * 2 * np.pi * f * k * qk * np.exp(1j * 2 * np.pi * f * k * t))
        ax.plot(t, derivative, label="d(position fit)/dt")
        ax.plot(
            data["rx_sof_time"][:, j],
            data["velocity"][:, j],
            label="motor velocity feedback",
            alpha=0.8,
        )
        ax.set(title=f"{NAMES[j]}: position / velocity consistency", ylabel="rad/s")
        ax.legend(fontsize=9)
    axes[3, 0].plot(t, data["torque"][:, 4:], label=["LW command", "RW command"])
    axes[3, 0].plot(
        t, data["current_torque"][:, 4:], alpha=0.7, label=["LW current proxy", "RW current proxy"]
    )
    axes[3, 0].set(title="Wheel torque command / current feedback", ylabel="Nm")
    axes[3, 0].legend(ncol=2, fontsize=9)
    axes[3, 1].plot(t[1:], data["tx_wait_us"][1:], label=[*NAMES[:4], "Wheels"])
    axes[3, 1].set(title="CAN command queue: enqueue to successful SOF", ylabel="us")
    axes[3, 1].legend(ncol=3, fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.set_xlabel("Time within steady burst (s)")
    fig.savefig(OUT / "steady_evidence.png", dpi=150)
    plt.close(fig)


def policy_replay(data, meta, f):
    """用实测策略快照重建推理，并离线测量各观测反馈路径；不连接实车。"""
    import onnxruntime as ort

    from se3_runtime import PolicyBundle, PolicyRuntime
    from se3_runtime.policy_observation import PolicyInput, PolicyObservationBuilder, RobotState

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    bundle = PolicyBundle.load(
        "D:/RoboMaster/serialleg2026/build/stm32-policy/Student128-6998/model.onnx",
        session_options=opts,
    )
    rows = meta["policies"]
    states = np.array(
        [
            p["input"]["angular_velocity"]
            + p["input"]["gravity"]
            + p["input"]["joint_position"]
            + p["input"]["joint_velocity"]
            + p["input"]["command"]
            for p in rows
        ]
    )
    seqs = np.array([p["sequence"] for p in rows])
    target_map = {
        int(s): row[[0, 2, 1, 3, 4, 5]]
        for s, row in zip(data["target_sequence"], data["target"], strict=True)
    }
    defaults = np.r_[bundle.contract.action.default_strategy.policy_leg_position, [0, 0]]
    scales = np.asarray(bundle.contract.action.scale)
    actual = np.array(
        [
            (target_map[s] - defaults) / scales if s in target_map else np.full(6, np.nan)
            for s in seqs
        ]
    )
    t = (
        np.array([p["input_tick"] - meta["monitor_before_export"]["start_tick"] for p in rows])
        / 1000
    )

    def make_input(state):
        """保持真实零指令与完整机器人观测布局。"""
        return PolicyInput(
            RobotState(state[:3], state[3:6], state[6:12], state[12:18]),
            {"velocity_height": np.r_[state[18:21], [0, 0, 0]]},
        )

    valid = (np.arange(len(rows)) >= 17) & np.isfinite(actual).all(axis=1)
    builder = PolicyObservationBuilder(bundle.contract)
    observations, outputs = [], []
    for i, state in enumerate(states):
        previous = actual[i - 1] if i > 0 and np.isfinite(actual[i - 1]).all() else np.zeros(6)
        obs = builder.build(make_input(state), previous).tensor
        observations.append(obs.copy())
        outputs.append(bundle.session.run(None, {"obs": obs})[0][0])
    outputs = np.asarray(outputs)
    hbase = harmonic(t[valid], outputs[valid], f)[0]
    result = {
        "scope": "离线观测回放和网络输入敏感性；不等价于实车移除某个反馈后的闭环实验。",
        "valid_replay_frames": int(valid.sum()),
        "max_action_replay_error": np.max(abs(outputs[valid] - actual[valid]), axis=0).tolist(),
        "rms_action_replay_error": np.sqrt(
            np.mean((outputs[valid] - actual[valid]) ** 2, axis=0)
        ).tolist(),
        "baseline_target_harmonic_amplitude": (abs(hbase) * scales).tolist(),
        "observed_channel_removal": {},
    }
    # 从真实观测中移除一个通道的主振频率分量，保留其均值、趋势和其它观测。
    groups = {
        "pitch_gyro": [1],
        "all_gyro": [0, 1, 2],
        "gravity": [3, 4, 5],
        "leg_position": list(range(6, 10)),
        "leg_velocity": list(range(12, 16)),
        "wheel_velocity": [16, 17],
        "all_imu": list(range(6)),
    }
    state_h = harmonic(t, states, f)[0]
    for name, indices in groups.items():
        altered = states.copy()
        altered[:, indices] -= np.real(np.exp(1j * 2 * np.pi * f * t[:, None]) * state_h[indices])
        build = PolicyObservationBuilder(bundle.contract)
        altered_out = []
        for i, state in enumerate(altered):
            previous = actual[i - 1] if i > 0 and np.isfinite(actual[i - 1]).all() else np.zeros(6)
            obs = build.build(make_input(state), previous).tensor
            altered_out.append(bundle.session.run(None, {"obs": obs})[0][0])
        altered_out = np.asarray(altered_out)
        ha = harmonic(t[valid], altered_out[valid], f)[0]
        contribution = hbase - ha
        result["observed_channel_removal"][name] = {
            "remaining_target_amplitude": (abs(ha) * scales).tolist(),
            "fraction_projected_on_original_oscillation": np.real(contribution / hbase).tolist(),
            "contribution_target_amplitude": (abs(contribution) * scales).tolist(),
        }
    # 仅在主机上将传感器输入固定在实测均值，保留网络上一动作递推。
    runtime = PolicyRuntime(bundle, action_delay_steps_override=0)
    for state in states:
        runtime.infer(make_input(state))
    fixed = make_input(states.mean(axis=0))
    tail = np.array([runtime.infer(fixed).raw_action for _ in range(400)])
    result["offline_constant_input"] = {
        "duration_s": 8.0,
        "last_2s_target_std": (tail[-100:].std(axis=0) * scales).tolist(),
        "last_2s_target_amplitude_at_observed_frequency": (
            abs(harmonic(np.arange(100) * 0.02, tail[-100:], f)[0]) * scales
        ).tolist(),
        "first_0p5s_target_std": (tail[:25].std(axis=0) * scales).tolist(),
    }
    # 对平均观测的历史项作局部线性化，用已采到的各通道正弦分量预测反馈路径。
    observation = runtime.observation_builder.build(fixed, runtime.previous_action).tensor
    jacobians = []
    for epsilon in (0.001, 0.0005):
        jac = np.zeros((6, observation.size))
        for j in range(observation.size):
            plus, minus = observation.copy(), observation.copy()
            plus[0, j] += epsilon
            minus[0, j] -= epsilon
            jac[:, j] = (
                bundle.session.run(None, {"obs": plus})[0][0]
                - bundle.session.run(None, {"obs": minus})[0][0]
            ) / (2 * epsilon)
        jacobians.append(jac)
    terms, cursor = {}, 0
    for group in bundle.contract.groups:
        for term in group.terms:
            terms[term.name] = jacobians[-1][:, cursor : cursor + term.flattened_width].reshape(
                6, term.history_length, term.base_width
            )[:, ::-1]
            cursor += term.flattened_width
    z = np.exp(-2j * np.pi * f * 0.02)
    action_j = terms["last_actions"]
    feedback = np.einsum("olj,l->oj", action_j, z ** np.arange(1, 17))
    companion = np.zeros((96, 96))
    companion[:6] = action_j.reshape(6, 96)
    companion[6:, :-6] = np.eye(90)
    poles = np.linalg.eigvals(companion)
    obs_array = np.concatenate(observations, axis=0)
    obs_h = harmonic(t[valid], obs_array[valid], f)[0]
    contributions, total = {}, np.zeros(6, dtype=complex)
    cursor = 0
    for group in bundle.contract.groups:
        for term in group.terms:
            width = term.flattened_width
            if term.name != "last_actions":
                direct = jacobians[-1][:, cursor : cursor + width] @ obs_h[cursor : cursor + width]
                contribution = np.linalg.solve(np.eye(6) - feedback, direct)
                contributions[term.name] = {
                    "target_amplitude": (abs(contribution) * scales).tolist(),
                    "fraction_projected_on_actual": np.real(contribution / hbase).tolist(),
                }
                total += contribution
            cursor += width
    result["local_linear_with_previous_action_recurrence"] = {
        "finite_difference_relative_disagreement": float(
            np.linalg.norm(jacobians[0] - jacobians[1]) / np.linalg.norm(jacobians[-1])
        ),
        "largest_internal_pole_magnitude": float(max(abs(poles))),
        "predicted_to_measured": ratio(total, hbase, f),
        "contributions": contributions,
    }
    return result


def main():
    """只消费已导出的文件，保存数值和图表以供复核。"""
    OUT.mkdir(parents=True, exist_ok=True)
    initial = json.loads(next(CAPTURE.glob("session_*.json")).read_text(encoding="utf-8"))[
        "initial"
    ]
    report = {"capture": str(CAPTURE), "motor_order": NAMES, "bursts": {}}
    for path in sorted(CAPTURE.glob("burst_*.npz")):
        data, meta = load(path)
        report["bursts"][path.stem] = timing(data, meta, initial)
        if meta["monitor_before_export"]["generation"] == 3:
            report["steady"] = steady_analysis(data, meta)
            plot(data, report["steady"])
            report["policy_replay"] = policy_replay(data, meta, report["steady"]["frequency_hz"])
    (OUT / "analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "saved": str(OUT / "analysis.json"),
                "steady_hz": report["steady"]["frequency_hz"],
                "policy_replay": report["policy_replay"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
