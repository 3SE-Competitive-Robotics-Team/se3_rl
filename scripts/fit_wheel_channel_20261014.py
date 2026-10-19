"""被控对象结构候选扫描：找能让 sim2x 同时复现 2026-10-14 实车增益扫描 A–E 工作点的结构；仅离线仿真。

目的是回答"仿真缺的是什么"，不是把抖动调没，也不是以消除抖动为目的的参数拟合。

基线配方 = `scripts/predict_gain_sweep_20261013.py`：主模型 4e1aeb0、固件陀螺 Q/R 平滑、腿速度 4.7 ms
一阶滤波、按时间门控的电机一阶滞后（腿 12 ms、轮 3/4 ms）、气弹簧 400/360 N 第 6 s 淡入、实测静置
IMU 噪声、失能趴姿落地 3 s 后使能静站 18 s；策略固定 Student128-6998，增益按 A–E 只在诊断 adapter
层覆盖（重建 `SerialLegActuatorController`），runtime 契约、观测、动作解码与延迟队列不动。

候选结构只进诊断用 MJCF 副本（`.scratch/wheel_channel_fit_20261014/mjcf/`）或本 adapter 的执行器
滞后，不改 src/、submodules/ 与正式 assets：

- (a) 轮地接触柔性：轮 geom solref（以固有频率与阻尼比参数化）；轮毂与胎面之间加扭转柔性自由度
  （另有带间隙的变体）
- (b) 轮电机力矩一阶滞后与纯延迟（另有实测的轮速测量滞后）
- (c) 轮转子惯量 armature
- (d) 腿电机转子与大腿 / 驱动杆之间的链传动柔性（PD 与观测接电机侧）
- (e) 机身俯仰惯量与质心高度

另有胎面柔性 × 第二因素的组合，以及清单外的腿执行器滞后诊断（family `x_leg_lag`）。

谱口径与 `scripts/analyze_gain_sweep_20261014.py` 相同：谐波回归主频（3–14 Hz）、Welch fs 1000 /
nperseg 900 / noverlap 450 的 5–15 Hz 带内 RMS、同频指令层功率 0.5·Re(F·conj(v))；稳态窗与实车每段
一样长 0.94 s，使能沿同样按 0.2 s 窗、0.1 s 步长统计。`--modes probe` 另做 8.3 Hz 观测注入，看撤除后
的振铃频率与阻尼比。

阶段：`simulate`（批量仿真，放 Pod 跑）、`balance`（不经仿真、用实车同步数据检验轮 / 腿传动扭转能否
让刚体机构闭合）、`assess`（汇总为 assessment.json 与 summary_tables.md；后两者需要本机的实车原始采集）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_control_sync_20261012 import harmonic  # noqa: E402
from analyze_gain_sweep_20261014 import dominant_frequency, ratio, welch_band_rms  # noqa: E402
from predict_gain_sweep_20261013 import (  # noqa: E402
    DISABLED_S,
    GAIN_GROUPS,
    MJCF_RELATIVE,
    MODEL_VERSIONS,
    POLICY,
    PRIMARY_MODEL,
    STAND_S,
    Case,
    GainSweepAdapter,
    band_rms,
    decay_metrics,
)

from se3_runtime import PolicyBundle, PolicyControlLoop, PolicyRuntime  # noqa: E402

OUT = ROOT / ".scratch/wheel_channel_fit_20261014"
REAL_ANALYSIS = ROOT / ".scratch/gain_sweep_analysis_20261014/analysis.json"
CAPTURE_SWEEP = Path("D:/RoboMaster/serialleg2026/build/control-sync-captures/gain_sweep_20261013")
BASE_MJCF = ROOT / ".scratch/gain_sweep_prediction_20261013/mjcf/serialleg_baseline_4e1aeb0.xml"
MESH_DIR = ROOT / "assets/robots/serialleg/meshes"
GROUPS = tuple(GAIN_GROUPS)
WINDOW_S = 0.94
"""实车每段同步采集 940 点（1 kHz），稳态窗与使能沿窗都取同长。"""
STEADY_STARTS = tuple(float(s) for s in np.arange(8.0, 17.01, 1.0))
"""弹簧淡入（6–6.5 s）之后的 10 个稳态窗起点。"""
BASE_CONTACT_SOLREF = (0.015, 1.25)
"""基线轮地接触实际混合值：轮 geom 默认 (0.010, 1.5) 与对齐后的 floor (0.02, 1.0) 按 solmix 1:1 平均。"""
BASE_CONTACT_SOLIMP = (0.875, 0.97, 0.003, 0.5, 2.0)
WHEEL_GEOMS = ("l_wheel_Link_collision_0", "r_wheel_Link_collision_0")
LEG_DRIVES = (
    ("lf0_Link", "lf0_Joint"),
    ("l_drive_bar_Link", "l_drive_bar_Joint"),
    ("rf0_Link", "rf0_Joint"),
    ("r_drive_bar_Link", "r_drive_bar_Joint"),
)
WHEELS = (("l", "l_wheel_Link", "l_wheel_Joint"), ("r", "r_wheel_Link", "r_wheel_Joint"))
LEG_ARMATURE = 0.01
"""契约写入腿电机关节的 armature（输出轴侧），链传动柔性的阻尼比按它换算。"""
WHEEL_ARMATURE = 0.005
"""契约写入轮关节的 armature（输出轴侧）。"""
WHEEL_RATED_NM = 2.210526315789474
REAL_RECORDER_TO_POLICY = [0, 2, 1, 3]
"""实车记录器腿顺序 LF/RF/LB/RB → policy 顺序 LF/LB/RF/RB。"""


@dataclass(frozen=True)
class Plant:
    """一组被控对象结构候选；字段取默认值即保持基线。"""

    name: str
    family: str
    label: str = ""
    contact_hz: float | None = None
    """轮地接触法向固有频率（Hz）；与 contact_zeta 一起换算 solref = (1/(2π f ζ), ζ)。"""
    contact_zeta: float | None = None
    tire_stiffness: float | None = None
    """轮毂与胎面之间的扭转刚度（N·m/rad，输出轴侧）；None 表示刚性轮。"""
    tire_zeta: float = 0.1
    """按内侧（转子 + 轮毂）惯量换算的扭转阻尼比。"""
    tire_outer_fraction: float = 0.5
    """轮体（不含转子）转动惯量与质量分到胎面侧的比例。"""
    tire_backlash_deg: float = 0.0
    """轮通道间隙（单侧半宽，度）：|扭转角| 小于它时弹簧与阻尼都不传力（减速器齿隙）。"""
    wheel_tau_ms: tuple[float, float] = (3.0, 4.0)
    wheel_delay_ms: int = 0
    wheel_velocity_tau_ms: float = 0.0
    """轮速测量一阶滞后（轮速环反馈与策略轮速观测同用）；实车 C620 转速相对编码器角导数落后约 7°（8 Hz）。"""
    wheel_armature: float | None = None
    chain_stiffness: float | None = None
    """腿电机转子到大腿 / 驱动杆的链传动扭转刚度（N·m/rad）；None 表示刚性传动。"""
    chain_zeta: float = 0.1
    """按电机侧 armature 换算的链传动阻尼比。"""
    pitch_inertia_scale: float = 1.0
    com_dz_m: float = 0.0
    leg_tau_ms: float = 12.0
    """腿电机力矩一阶滞后（配方值 12 ms）；只在清单外的腿执行器诊断里改。"""
    leg_delay_ms: int = 0


@dataclass(frozen=True)
class Run:
    """一个候选在一个增益组上的一次仿真。"""

    plant: Plant
    group: str
    policy: str = POLICY
    save_trace: bool = True
    probe: bool = False
    """True 时按 predict_gain_sweep 的注入工况：观测俯仰角速度叠加 0.4 rad/s、8.3 Hz 正弦（8–12 s），
    用撤除后 5–15 Hz 包络的衰减刻画 8 Hz 附近闭环模态的阻尼。"""

    @property
    def tag(self) -> str:
        return self.group + ("_probe" if self.probe else "")


def contact_solref(plant: Plant) -> tuple[float, float]:
    """(固有频率, 阻尼比) → MuJoCo solref；MuJoCo 软约束 ω_n ≈ 1/(timeconst·dampratio)、ζ = dampratio。"""
    zeta = plant.contact_zeta
    return 1.0 / (2 * math.pi * plant.contact_hz * zeta), zeta


def tire_damping(plant: Plant) -> float:
    """胎面扭转阻尼 = 2ζ√(k·J_内)，J_内 = 转子 armature + 轮毂侧轮体惯量。"""
    armature = plant.wheel_armature or WHEEL_ARMATURE
    inner = armature + 0.00048080 * (1 - plant.tire_outer_fraction)
    return 2 * plant.tire_zeta * math.sqrt(plant.tire_stiffness * inner)


def chain_damping(plant: Plant) -> float:
    """链传动阻尼 = 2ζ√(k·J_电机)，J_电机 = 契约腿 armature。"""
    return 2 * plant.chain_zeta * math.sqrt(plant.chain_stiffness * LEG_ARMATURE)


def single_factor_plants() -> list[Plant]:
    """五类候选的单因素扫描；每类都包含物理上合理范围的两端。"""
    plants = [Plant("baseline", "baseline", "基线（predict_gain_sweep 配方，4e1aeb0）")]
    for hz in (4.0, 8.5, 17.0, 34.0):
        for zeta in (0.1, 0.3, 1.25):
            if (hz, zeta) == (8.5, 1.25):
                continue  # 等于基线混合值 (0.015, 1.25)
            plants.append(
                Plant(
                    f"contact_f{hz:g}_z{zeta:g}",
                    "a_contact",
                    f"轮地接触 {hz:g} Hz、ζ {zeta:g}",
                    contact_hz=hz,
                    contact_zeta=zeta,
                )
            )
    for stiffness in (10.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 160.0, 320.0):
        plants.append(
            Plant(
                f"tire_k{stiffness:g}_z0.1",
                "a_tire",
                f"轮毂-胎面扭转 {stiffness:g} N·m/rad、ζ 0.1",
                tire_stiffness=stiffness,
            )
        )
    for stiffness in (20.0, 40.0, 80.0):
        plants.append(
            Plant(
                f"tire_k{stiffness:g}_z0.4",
                "a_tire",
                f"轮毂-胎面扭转 {stiffness:g} N·m/rad、ζ 0.4",
                tire_stiffness=stiffness,
                tire_zeta=0.4,
            )
        )
    for zeta in (0.05, 0.2):
        plants.append(
            Plant(
                f"tire_k40_z{zeta:g}",
                "a_tire",
                f"轮毂-胎面扭转 40 N·m/rad、ζ {zeta:g}",
                tire_stiffness=40.0,
                tire_zeta=zeta,
            )
        )
    for stiffness in (40.0, 60.0, 100.0, 200.0):
        for backlash in (0.25, 0.5, 1.0):
            plants.append(
                Plant(
                    f"tire_k{stiffness:g}_gap{backlash:g}deg",
                    "a_tire",
                    f"轮通道扭转 {stiffness:g} N·m/rad + 单侧间隙 {backlash:g}°（ζ 0.1）",
                    tire_stiffness=stiffness,
                    tire_backlash_deg=backlash,
                )
            )
    plants.append(
        Plant(
            "wheel_speed_lag_2.5ms",
            "b_wheel_lag",
            "轮速测量一阶滞后 2.5 ms（实测 C620 转速相对编码器角导数 −7°@8 Hz）",
            wheel_velocity_tau_ms=2.5,
        )
    )
    for stiffness, extra_key, extra in (
        (30.0, "", {}),
        (40.0, "", {}),
        (60.0, "", {}),
        (40.0, "_delay2", {"wheel_delay_ms": 2}),
        (40.0, "_lag7.5", {"wheel_tau_ms": (7.5, 7.5)}),
    ):
        plants.append(
            Plant(
                f"combo_tire{stiffness:g}_speedlag2.5{extra_key}",
                "combo",
                f"胎面扭转 {stiffness:g} N·m/rad + 轮速测量滞后 2.5 ms"
                + (
                    " + 轮力矩纯延迟 2 ms"
                    if extra_key == "_delay2"
                    else " + 轮力矩一阶滞后 7.5 ms"
                    if extra_key
                    else ""
                ),
                tire_stiffness=stiffness,
                wheel_velocity_tau_ms=2.5,
                **extra,
            )
        )
    plants.append(
        Plant(
            "tire_k40_z0.1_outer0.9",
            "a_tire",
            "转子-轮体扭转 40 N·m/rad（轮体 90% 惯量在柔性外侧，等效减速器柔性）",
            tire_stiffness=40.0,
            tire_outer_fraction=0.9,
        )
    )
    for tau in (5.0, 7.5, 10.0, 15.0, 20.0):
        plants.append(
            Plant(
                f"wheel_lag_{tau:g}ms",
                "b_wheel_lag",
                f"轮力矩一阶滞后 {tau:g} ms",
                wheel_tau_ms=(tau, tau),
            )
        )
    for delay in (2, 3, 5, 10):
        plants.append(
            Plant(
                f"wheel_delay_{delay}ms",
                "b_wheel_lag",
                f"轮力矩纯延迟 {delay} ms（叠加基线 3/4 ms 滞后）",
                wheel_delay_ms=delay,
            )
        )
    for armature in (0.0035, 0.0075):
        plants.append(
            Plant(
                f"wheel_armature_{armature:g}",
                "c_wheel_armature",
                f"轮转子惯量 {armature:g} kg·m²",
                wheel_armature=armature,
            )
        )
    for stiffness in (15.0, 30.0, 60.0, 120.0, 300.0, 500.0):
        plants.append(
            Plant(
                f"chain_k{stiffness:g}_z0.1",
                "d_leg_chain",
                f"腿链传动 {stiffness:g} N·m/rad、ζ 0.1",
                chain_stiffness=stiffness,
            )
        )
    for stiffness in (30.0, 60.0):
        plants.append(
            Plant(
                f"chain_k{stiffness:g}_z0.4",
                "d_leg_chain",
                f"腿链传动 {stiffness:g} N·m/rad、ζ 0.4",
                chain_stiffness=stiffness,
                chain_zeta=0.4,
            )
        )
    for scale in (0.5, 0.75, 1.5, 2.0):
        plants.append(
            Plant(
                f"body_pitch_inertia_x{scale:g}",
                "e_body",
                f"机身俯仰惯量 ×{scale:g}",
                pitch_inertia_scale=scale,
            )
        )
    for dz in (-0.04, -0.02, 0.02, 0.04):
        plants.append(
            Plant(
                f"body_com_dz{dz * 100:+g}cm",
                "e_body",
                f"机身质心 {dz * 100:+g} cm",
                com_dz_m=dz,
            )
        )
    return plants


COMBO_SECOND_FACTORS = {
    "delay2": ("轮力矩纯延迟 2 ms", {"wheel_delay_ms": 2}),
    "delay3": ("轮力矩纯延迟 3 ms", {"wheel_delay_ms": 3}),
    "lag5": ("轮力矩一阶滞后 5 ms", {"wheel_tau_ms": (5.0, 5.0)}),
    "lag7.5": ("轮力矩一阶滞后 7.5 ms", {"wheel_tau_ms": (7.5, 7.5)}),
    "arm0.0075": ("轮转子惯量 0.0075", {"wheel_armature": 0.0075}),
    "arm0.0035": ("轮转子惯量 0.0035", {"wheel_armature": 0.0035}),
    "pitch0.75": ("机身俯仰惯量 ×0.75", {"pitch_inertia_scale": 0.75}),
}
"""单因素里只有轮通道相位滞后类（胎面扭转柔性、轮力矩额外滞后 / 延迟）把 A 的 8 Hz 模态阻尼降到
ζ≈0.05–0.07，同时让 D / E 在 6–7 Hz 失稳、C 在 6 Hz 轻阻尼；组合以胎面柔性为主因素，
叠加第二个能继续压低 8 Hz 阻尼或抬高模态频率的因素（链传动已被姿态否定，不进组合）。"""


LEG_LAG_VARIANTS = {
    "legtau15": ("腿力矩一阶滞后 15 ms", {"leg_tau_ms": 15.0}),
    "legtau18": ("腿力矩一阶滞后 18 ms", {"leg_tau_ms": 18.0}),
    "legdelay8_tau5": ("腿力矩纯延迟 8 ms + 一阶 5 ms", {"leg_tau_ms": 5.0, "leg_delay_ms": 8}),
}
"""清单外诊断：实车腿电流反馈落后指令 31–38°（8.14 Hz），一阶滞后折 11.8–15.3 ms、纯延迟折 10.6–13 ms；
配方取 12 ms 一阶。这里看腿执行器滞后取测量范围上端时 A 的 8 Hz 模态是否变化，不属于五类候选。"""


def leg_lag_plants() -> list[Plant]:
    """腿执行器滞后单独、以及与轮通道柔性 / 滞后组合。"""
    plants = []
    seconds = {
        "none": ("", {}),
        "tire40": ("胎面扭转 40 N·m/rad", {"tire_stiffness": 40.0}),
        "wheellag7.5": ("轮力矩一阶滞后 7.5 ms", {"wheel_tau_ms": (7.5, 7.5)}),
    }
    for key, (label, extra) in LEG_LAG_VARIANTS.items():
        for wheel_key, (wheel_label, wheel_extra) in seconds.items():
            name = f"leglag_{key}" + ("" if wheel_key == "none" else f"_{wheel_key}")
            plants.append(
                Plant(
                    name,
                    "x_leg_lag",
                    label + ("" if wheel_key == "none" else f" + {wheel_label}"),
                    **extra,
                    **wheel_extra,
                )
            )
    return plants


def combo_plants() -> list[Plant]:
    """胎面扭转柔性 25–60 N·m/rad × 第二因素。"""
    plants = []
    for stiffness in (25.0, 30.0, 40.0, 50.0, 60.0):
        for key, (label, extra) in COMBO_SECOND_FACTORS.items():
            plants.append(
                Plant(
                    f"combo_tire{stiffness:g}_{key}",
                    "combo",
                    f"胎面扭转 {stiffness:g} N·m/rad + {label}",
                    tire_stiffness=stiffness,
                    **extra,
                )
            )
    return plants


def all_plants() -> dict[str, Plant]:
    """单因素与组合候选按名字索引。"""
    return {
        plant.name: plant for plant in single_factor_plants() + combo_plants() + leg_lag_plants()
    }


# ---------------------------------------------------------------- MJCF 副本


def _fmt(value: float) -> str:
    return f"{value:.10g}"


def _find_body(root: ET.Element, name: str) -> tuple[ET.Element, ET.Element]:
    """返回 (父节点, body)。"""
    for parent in root.iter():
        for child in parent:
            if child.tag == "body" and child.get("name") == name:
                return parent, child
    raise KeyError(name)


def _inertia_matrix(full: str) -> np.ndarray:
    xx, yy, zz, xy, xz, yz = (float(v) for v in full.split())
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def _full_inertia(matrix: np.ndarray) -> str:
    return " ".join(_fmt(matrix[i, j]) for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)))


def apply_contact(root: ET.Element, plant: Plant) -> None:
    """轮 geom 的 solref 改为候选值，solimp 保持基线混合值；solmix 取极大使接触参数由轮 geom 决定。"""
    if plant.contact_hz is None:
        return
    solref = contact_solref(plant)
    for name in WHEEL_GEOMS:
        geom = next(g for g in root.iter("geom") if g.get("name") == name)
        geom.set("solref", " ".join(_fmt(v) for v in solref))
        geom.set("solimp", " ".join(_fmt(v) for v in BASE_CONTACT_SOLIMP))
        geom.set("solmix", "1e6")


def apply_tire(root: ET.Element, plant: Plant) -> None:
    """在轮体内加一个与轮轴同轴的"胎面" body：扭转弹簧 + 阻尼，碰撞 geom 移到胎面上。

    轮电机（policy 轮关节）仍驱动轮毂、编码器仍读轮毂；轮体质量与质心不变，转动惯量按比例拆分。
    """
    if plant.tire_stiffness is None:
        return
    fraction = plant.tire_outer_fraction
    for side, body_name, joint_name in WHEELS:
        _, body = _find_body(root, body_name)
        joint = body.find(f"joint[@name='{joint_name}']")
        inertial = body.find("inertial")
        collision = body.find(f"geom[@name='{body_name}_collision_0']")
        mass = float(inertial.get("mass"))
        inertia = _inertia_matrix(inertial.get("fullinertia"))
        tread = ET.SubElement(body, "body", name=f"{side}_tread_Link", pos="0 0 0")
        ET.SubElement(
            tread,
            "joint",
            name=f"{side}_tread_Joint",
            type="hinge",
            axis=joint.get("axis"),
            stiffness=_fmt(0.0 if plant.tire_backlash_deg else plant.tire_stiffness),
            damping=_fmt(0.0 if plant.tire_backlash_deg else tire_damping(plant)),
        )
        ET.SubElement(
            tread,
            "inertial",
            pos=inertial.get("pos"),
            mass=_fmt(mass * fraction),
            fullinertia=_full_inertia(inertia * fraction),
        )
        inertial.set("mass", _fmt(mass * (1 - fraction)))
        inertial.set("fullinertia", _full_inertia(inertia * (1 - fraction)))
        body.remove(collision)
        tread.append(collision)


def apply_chain(root: ET.Element, plant: Plant) -> None:
    """腿电机关节移到新的电机侧 body，原连杆经链传动扭转弹簧挂在其下。

    policy 关节名（PD、限矩、观测、armature/damping/frictionloss）留在电机侧；被动闭链、弹簧挂点、
    碰撞与质量都留在原连杆上。主动杆夹角 tendon 仍按电机侧角度计算。
    """
    if plant.chain_stiffness is None:
        return
    for link_name, joint_name in LEG_DRIVES:
        parent, link = _find_body(root, link_name)
        joint = link.find(f"joint[@name='{joint_name}']")
        index = list(parent).index(link)
        motor = ET.Element("body", name=f"{link_name}_motor_side", pos=link.get("pos"))
        link.remove(joint)
        motor.append(joint)
        ET.SubElement(motor, "inertial", pos="0 0 0", mass="0.001", diaginertia="1e-7 1e-7 1e-7")
        link.set("pos", "0 0 0")
        link.insert(
            0,
            ET.Element(
                "joint",
                name=f"{joint_name}_chain",
                type="hinge",
                axis=joint.get("axis"),
                stiffness=_fmt(plant.chain_stiffness),
                damping=_fmt(chain_damping(plant)),
            ),
        )
        parent.remove(link)
        motor.append(link)
        parent.insert(index, motor)


def apply_body(root: ET.Element, plant: Plant) -> None:
    """机身俯仰惯量按二阶矩沿 x、z 方向拉伸（保持物理可实现），质心沿机身 z 平移。"""
    if plant.pitch_inertia_scale == 1.0 and plant.com_dz_m == 0.0:
        return
    _, base = _find_body(root, "base_link")
    inertial = base.find("inertial")
    inertia = _inertia_matrix(inertial.get("fullinertia"))
    second_moment = np.trace(inertia) / 2 * np.eye(3) - inertia
    stretch = np.diag(
        [math.sqrt(plant.pitch_inertia_scale), 1.0, math.sqrt(plant.pitch_inertia_scale)]
    )
    second_moment = stretch @ second_moment @ stretch
    inertial.set("fullinertia", _full_inertia(np.trace(second_moment) * np.eye(3) - second_moment))
    position = np.array([float(v) for v in inertial.get("pos").split()])
    position[2] += plant.com_dz_m
    inertial.set("pos", " ".join(_fmt(v) for v in position))


def _qpos_layout(root: ET.Element) -> list[tuple[str, int]]:
    """按 MuJoCo 的 body 深度优先顺序列出 (关节名, qpos 维数)。"""
    layout = []
    for body in root.find("worldbody").iter("body"):
        for child in body:
            if child.tag == "freejoint":
                layout.append(("__free__", 7))
            elif child.tag == "joint":
                layout.append((child.get("name"), 7 if child.get("type") == "free" else 1))
    return layout


def plant_mjcf(plant: Plant) -> Path:
    """由钉死的 4e1aeb0 MJCF 生成该候选的诊断副本；新增自由度在 standing keyframe 里取 0。"""
    target = OUT / "mjcf" / f"{plant.name}.xml"
    target.parent.mkdir(parents=True, exist_ok=True)
    root = ET.fromstring(BASE_MJCF.read_text(encoding="utf-8"))
    root.find("compiler").set("meshdir", MESH_DIR.resolve().as_posix())
    original = _qpos_layout(root)
    key = root.find("keyframe/key[@name='standing']")
    values = [float(v) for v in key.get("qpos").split()]
    by_name, offset = {}, 0
    for name, width in original:
        by_name[name] = values[offset : offset + width]
        offset += width
    assert offset == len(values)
    apply_contact(root, plant)
    apply_tire(root, plant)
    apply_chain(root, plant)
    apply_body(root, plant)
    qpos = []
    for name, width in _qpos_layout(root):
        qpos += by_name.get(name, [0.0] * width)
    key.set("qpos", " ".join(_fmt(v) for v in qpos))
    text = ET.tostring(root, encoding="unicode")
    if not target.exists() or target.read_text(encoding="utf-8") != text:
        target.write_text(text, encoding="utf-8")
    return target


# ---------------------------------------------------------------- 仿真


class WheelChannelAdapter(GainSweepAdapter):
    """GainSweepAdapter 配方 + 轮力矩纯延迟与滞后覆盖 + 额外记录列（指令力矩、弹簧前馈、柔性形变）。"""

    plant: Plant | None = None
    backlash: Plant | None = None

    def attach_backlash(self, plant: Plant) -> None:
        """失能落地前调用：带间隙的轮通道扭转弹簧在整段仿真里都由本 adapter 施加。"""
        if not plant.tire_backlash_deg:
            return
        self.backlash = plant
        self.tread_dofs = np.array(
            [
                self.model.jnt_dofadr[self.model.joint(f"{side}_tread_Joint").id]
                for side, _, _ in WHEELS
            ]
        )
        self.tread_qpos = np.array(
            [
                self.model.jnt_qposadr[self.model.joint(f"{side}_tread_Joint").id]
                for side, _, _ in WHEELS
            ]
        )

    def _apply_backlash(self) -> None:
        """死区弹簧：|φ| ≤ b 不传力；接触后 τ = −k(φ ∓ b) − cφ̇，且只压不拉。"""
        plant = self.backlash
        gap = np.deg2rad(plant.tire_backlash_deg)
        angle = self.data.qpos[self.tread_qpos]
        rate = self.data.qvel[self.tread_dofs]
        engaged = np.abs(angle) > gap
        penetration = np.where(engaged, angle - gap * np.sign(angle), 0.0)
        torque = -plant.tire_stiffness * penetration - tire_damping(plant) * rate * engaged
        torque = np.where(torque * penetration > 0, 0.0, torque)
        self.data.qfrc_applied[self.tread_dofs] = torque

    def configure_plant(self, plant: Plant) -> None:
        """在 `configure` 之后调用；只改轮通道执行器滞后，腿 12 ms 等其余配方不动。"""
        self.plant = plant
        self.motor_tau_s[:4] = plant.leg_tau_ms / 1000
        self.motor_tau_s[4:] = np.asarray(plant.wheel_tau_ms, dtype=float) / 1000
        self.wheel_queue = deque(np.zeros(2) for _ in range(plant.wheel_delay_ms))
        self.leg_queue = deque(np.zeros(4) for _ in range(plant.leg_delay_ms))
        self.command_effort = np.zeros(6)
        self.spring_compensation = np.zeros(4)
        self.wheel_velocity = self.policy_joint_velocity[4:].copy()
        self.wheel_velocity_time = float(self.data.time)
        names = [f"{side}_tread_Joint" for side, _, _ in WHEELS] + [
            f"{joint}_chain" for _, joint in LEG_DRIVES
        ]
        self.compliance_qpos = np.array(
            [
                self.model.jnt_qposadr[self.model.joint(name).id]
                if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0
                else -1
                for name in names
            ]
        )
        self.floor_geom = self.model.geom("floor").id
        self.wheel_geoms = [self.model.geom(name).id for name in WHEEL_GEOMS]
        self.extra_rows: list[np.ndarray] = []

    def update_sensors(self) -> None:
        """在父类陀螺 / 腿速度滤波之外，按需给轮速加一阶测量滞后（同一时刻只更新一次）。"""
        super().update_sensors()
        if self.plant is None or not self.plant.wheel_velocity_tau_ms:
            return
        if self.data.time <= self.wheel_velocity_time + 1e-9:
            return
        alpha = 1 - np.exp(-0.001 / (self.plant.wheel_velocity_tau_ms / 1000))
        raw = self.data.qvel[self._joint_qvel_addresses[4:]]
        self.wheel_velocity += alpha * (raw - self.wheel_velocity)
        self.wheel_velocity_time = float(self.data.time)

    def read_policy_input(self):
        """策略轮速观测与轮速环用同一个带滞后的轮速。"""
        value = super().read_policy_input()
        if self.plant is None or not self.plant.wheel_velocity_tau_ms:
            return value
        velocity = np.array(value.robot.policy_joint_velocity)
        velocity[4:] = self.wheel_velocity
        return replace(value, robot=replace(value.robot, policy_joint_velocity=velocity))

    def _refresh_actuator_control(self) -> None:
        """与 GainSweepAdapter 相同的 PD → 按时间门控一阶滞后；轮（及诊断用的腿）通道在滞后前再过纯延迟队列。"""
        if (
            self.plant is None
            or not self.configured
            or not self._motors_enabled
            or self._last_decoded_action is None
        ):
            super()._refresh_actuator_control()
            return
        self.update_sensors()
        velocity = self.policy_joint_velocity
        velocity[:4] = self.filtered_velocity[:4]
        if self.plant.wheel_velocity_tau_ms:
            velocity[4:] = self.wheel_velocity
        control = self._controller.compute(
            self._last_decoded_action,
            policy_joint_position=self.policy_joint_position,
            policy_joint_velocity=velocity,
        )
        if self.data.time > self.lag_time + 1e-9:
            command = control.effort.copy()
            self.command_effort = command.copy()
            if self.wheel_queue:
                self.wheel_queue.append(command[4:].copy())
                command[4:] = self.wheel_queue.popleft()
            if self.leg_queue:
                self.leg_queue.append(command[:4].copy())
                command[:4] = self.leg_queue.popleft()
            alpha = 1 - np.exp(-0.001 / self.motor_tau_s)
            self.actual_effort += alpha * (command - self.actual_effort)
            self.lag_time = float(self.data.time)
        control = replace(control, effort=self.actual_effort.copy())
        self.spring_compensation = control.knee_gas_spring_compensation.copy()
        self.data.ctrl[self._actuator_ids] = control.effort
        self._last_control = control

    def _wheel_contact_gap(self) -> np.ndarray:
        """两轮与地面接触的最小间隙（m，负值为穿透）；无接触记为 +1。"""
        gaps = np.ones(2)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if self.floor_geom not in contact.geom:
                continue
            for side, geom in enumerate(self.wheel_geoms):
                if geom in contact.geom:
                    gaps[side] = min(gaps[side], float(contact.dist))
        return gaps

    def advance_physics_tick(self) -> None:
        """带间隙时先按积分前状态施加死区弹簧；与 TraceAdapter 同一时刻记录额外列，行数与主轨迹严格对齐。

        形变与接触间隙取积分前的状态（与主轨迹该行同一时刻）；指令力矩与弹簧前馈在本 tick 内
        积分前就已定下（使能那一 tick 在父类里才算出），所以在父类返回后再取。
        """
        if self.backlash is not None:
            self._apply_backlash()
        if self.plant is None or not hasattr(self, "rows"):
            super().advance_physics_tick()
            return
        before = len(self.rows)
        compliance = np.where(
            self.compliance_qpos >= 0, self.data.qpos[np.maximum(self.compliance_qpos, 0)], 0.0
        )
        gap = self._wheel_contact_gap()
        super().advance_physics_tick()
        if len(self.rows) > before:
            self.extra_rows.append(
                np.r_[self.command_effort, self.spring_compensation, compliance, gap]
            )


EXTRA_COLUMNS = {
    "command_effort": slice(0, 6),
    "spring_compensation": slice(6, 10),
    "tire_deflection": slice(10, 12),
    "chain_deflection": slice(12, 16),
    "wheel_contact_gap": slice(16, 18),
}


def build(run: Run, model_path: Path) -> tuple[PolicyControlLoop, WheelChannelAdapter]:
    """与 predict_gain_sweep 的 build 相同（1 ms tick、2 ms 动作延迟、固定种子），换成候选 MJCF。"""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    bundle = PolicyBundle.load(run.policy, session_options=opts)
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
    adapter = WheelChannelAdapter(
        runtime.contract, artifact_path=run.policy, model_path=model_path, solver_dt_s=dt
    )
    adapter.set_command_field("velocity_height", "height", 0.22)
    adapter.set_command_field("velocity_height", "lin_vel_x", 0.0)
    adapter.set_command_field("velocity_height", "ang_vel_yaw", 0.0)
    loop = PolicyControlLoop(runtime, adapter)
    loop.reset()
    adapter.pd_period_s = 0.0
    adapter.last_pd_time = float("-inf")
    return loop, adapter


def run_dir(plant: Plant) -> Path:
    return OUT / "runs" / plant.name


def simulate(run: Run) -> dict:
    """失能落地 3 s → 使能静站 18 s；保存精简轨迹与本次运行的全部指标。"""
    start = time.monotonic()
    kp, kd, wkd = GAIN_GROUPS[run.group]
    case = Case(
        run.group,
        kp,
        kd,
        wkd,
        0.22,
        probe_amplitude=0.4 if run.probe else 0.0,
        model=PRIMARY_MODEL,
        label=run.plant.label,
    )
    model_path = OUT / "mjcf" / f"{run.plant.name}.xml"
    loop, adapter = build(run, model_path)
    adapter.attach_backlash(run.plant)
    if run.plant.wheel_armature is not None:
        # 契约在编译后把轮 armature 写成 0.005；候选值在失能落地前覆盖，整段仿真一致。
        adapter.model.dof_armature[adapter._joint_qvel_addresses[4:]] = run.plant.wheel_armature
    loop.runtime.reset()
    adapter.set_motors_enabled(False)
    for _ in range(round(DISABLED_S / 0.001)):
        adapter.advance_physics_tick()
    adapter.configure(case, float(adapter.data.time))
    adapter.configure_plant(run.plant)
    prone = {"qpos": adapter.data.qpos.tolist()}
    adapter.start_trace(loop.runtime)
    fell_at = None
    for _ in range(round(STAND_S / loop.runtime.contract.timing.policy_dt_s)):
        loop.policy_step()
        if adapter.data.xmat[adapter.base_id].reshape(3, 3)[2, 2] < 0.5:
            fell_at = float(adapter.data.time - adapter.enable_time)
            break
    trace = np.asarray(adapter.rows)
    trace[:, 0] -= adapter.enable_time
    extra = np.asarray(adapter.extra_rows)
    assert len(extra) == len(trace)
    report = {
        "plant": asdict(run.plant),
        "group": run.group,
        "probe": run.probe,
        "gains": {"leg_kp": kp, "leg_kd": kd, "wheel_gain": wkd},
        "recipe": {
            "policy": run.policy,
            "base_model": f"{MODEL_VERSIONS[PRIMARY_MODEL]}:{MJCF_RELATIVE}",
            "model_path": str(model_path),
            "motor_tau_ms": (adapter.motor_tau_s * 1000).tolist(),
            "wheel_delay_ms": run.plant.wheel_delay_ms,
            "policy_dof_armature": adapter.model.dof_armature[
                adapter._joint_qvel_addresses
            ].tolist(),
            "contact_solref": contact_solref(run.plant)
            if run.plant.contact_hz is not None
            else list(BASE_CONTACT_SOLREF),
            "tire_damping": tire_damping(run.plant)
            if run.plant.tire_stiffness is not None
            else None,
            "chain_damping": chain_damping(run.plant)
            if run.plant.chain_stiffness is not None
            else None,
            "nq": int(adapter.model.nq),
            "spring_force_n": case.spring_force_n,
        },
        "prone_qpos": prone["qpos"],
        "fell_at_s": fell_at,
        "warnings": {str(i): int(w.number) for i, w in enumerate(adapter.data.warning) if w.number},
        "wall_time_s": None,
    }
    report.update(analyze_run(trace, extra, (kp, kd, wkd), fell_at, steady=not run.probe))
    if run.probe and fell_at is None:
        before = trace[(trace[:, 0] >= 6) & (trace[:, 0] < 8)]
        floor = float(band_rms(before[:, [41]].astype(float), 900)[0][0])
        report["probe_decay"] = decay_metrics(trace.astype(float), floor)
    report["wall_time_s"] = time.monotonic() - start
    directory = run_dir(run.plant)
    directory.mkdir(parents=True, exist_ok=True)
    if run.save_trace:
        np.savez_compressed(
            directory / f"{run.tag}.npz",
            trace=trace.astype(np.float32),
            extra=extra.astype(np.float32),
        )
    (directory / f"{run.tag}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "plant": run.plant.name,
                "group": run.tag,
                "fell_at_s": fell_at,
                "freq_hz": report["summary"].get("frequency_hz", {}).get("median"),
                "pitch_band_rms": report["summary"].get("pitch_rate_band_rms", {}).get("mean"),
                "edge_late_pitch": report["edge"]["late_pitch_5_15_rms"],
                "probe_tau_s": report.get("probe_decay", {}).get("decay_time_constant_s"),
                "seconds": round(report["wall_time_s"], 1),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return report


# ---------------------------------------------------------------- 分析（与实车脚本同口径）


def _window(trace: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (trace[:, 0] >= lo - 1e-9) & (trace[:, 0] < hi - 1e-9)


def steady_window(trace: np.ndarray, extra: np.ndarray, lo: float, gains) -> dict:
    """一个 0.94 s 稳态窗：主频、复幅值、策略传递、指令层功率；与实车 steady_metrics 逐项对应。"""
    kp, kd, wheel_gain = gains
    mask = _window(trace, lo, lo + WINDOW_S)
    t = trace[mask, 0].astype(float)
    pitch = trace[mask, 41].astype(float)
    f = dominant_frequency(t, pitch)
    omega = 2 * np.pi * f
    hg, r2, _, _ = harmonic(t, pitch, f)
    q = trace[mask, 1:5].astype(float)
    velocity = trace[mask, 7:11].astype(float)
    wheel_q = trace[mask, 5:7].astype(float)
    wheel_velocity = trace[mask, 11:13].astype(float)
    target = trace[mask, 13:19].astype(float)
    command = extra[mask, EXTRA_COLUMNS["command_effort"]].astype(float)
    spring = extra[mask, EXTRA_COLUMNS["spring_compensation"]].astype(float)
    hq = harmonic(t, q, f, 3)[0]
    hv = harmonic(t, velocity, f, 3)[0]
    hwq = harmonic(t, wheel_q, f, 3)[0]
    hwv = harmonic(t, wheel_velocity, f, 3)[0]
    htarget = harmonic(t, target, f, 3)[0]
    hcommand = harmonic(t, command, f, 3)[0]
    hspring = harmonic(t, spring, f, 3)[0]
    v_leg = 1j * omega * hq
    v_wheel = 1j * omega * hwq
    power = {
        "P_target": (0.5 * np.real(kp * htarget[:4] * np.conj(v_leg))).tolist(),
        "D": (0.5 * np.real(-kd * hv * np.conj(v_leg))).tolist(),
        "spring": (0.5 * np.real(hspring * np.conj(v_leg))).tolist(),
        "wheel_target_term": (0.5 * np.real(wheel_gain * htarget[4:] * np.conj(v_wheel))).tolist(),
        "wheel_feedback_term": (0.5 * np.real(-wheel_gain * hwv * np.conj(v_wheel))).tolist(),
    }
    ods = operating_shape(
        hg,
        leg_q=hq,
        wheel_velocity=1j * omega * hwq,
        wheel_torque=hcommand[4:],
        wheel_target=htarget[4:],
        leg_torque=hcommand[:4],
        leg_target=htarget[:4],
    )
    compliance = {}
    for key in ("tire_deflection", "chain_deflection"):
        values = extra[mask, EXTRA_COLUMNS[key]].astype(float)
        if np.any(values):
            compliance[key] = {
                "mean_deg": np.rad2deg(values.mean(0)).tolist(),
                "amplitude_deg": np.rad2deg(np.abs(harmonic(t, values, f, 3)[0])).tolist(),
            }
    return {
        "t0_s": lo,
        "frequency_hz": f,
        "pitch_fit_r2": float(r2),
        "amplitude": {
            "pitch_rate_rad_s": float(abs(hg)),
            "leg_deg": np.rad2deg(np.abs(hq)).tolist(),
            "wheel_velocity_rad_s": np.abs(hwv).tolist(),
            "leg_target_deg": np.rad2deg(np.abs(htarget[:4])).tolist(),
            "wheel_target_rad_s": np.abs(htarget[4:]).tolist(),
            "leg_torque_nm": np.abs(hcommand[:4]).tolist(),
            "wheel_torque_nm": np.abs(hcommand[4:]).tolist(),
        },
        "band_rms_5_15": {
            "pitch_rate_rad_s": float(welch_band_rms(t, pitch, 5, 15)),
            "leg_deg": np.rad2deg(welch_band_rms(t, q, 5, 15)).tolist(),
            "wheel_velocity_rad_s": welch_band_rms(t, wheel_velocity, 5, 15).tolist(),
        },
        "band_rms_15_100": {
            "leg_torque_nm": welch_band_rms(t, command[:, :4], 15, 100).tolist(),
            "leg_velocity_rad_s": welch_band_rms(t, velocity, 15, 100).tolist(),
            "pitch_rate_rad_s": float(welch_band_rms(t, pitch, 15, 100)),
        },
        "policy_pitch_to_wheel_target": ratio(htarget[4:], hg),
        "policy_pitch_to_leg_target": ratio(htarget[:4], hg),
        "wheel_torque_vs_wheel_target": ratio(hcommand[4:], htarget[4:]),
        "leg_velocity_feedback_vs_position_derivative": ratio(hv, 1j * omega * hq),
        "command_power_w": power,
        "leg_torque_saturation_fraction": np.mean(
            np.abs(command[:, :4]) >= 32.0 - 1e-6, axis=0
        ).tolist(),
        "wheel_torque_saturation_fraction": np.mean(
            np.abs(command[:, 4:]) >= WHEEL_RATED_NM - 1e-6, axis=0
        ).tolist(),
        "mean_leg_q_deg": np.rad2deg(q.mean(0)).tolist(),
        "mean_height_m": float(trace[mask, 46].mean()),
        "mean_pitch_deg": float(np.rad2deg(trace[mask, 38].mean())),
        "wheel_contact_gap_max_mm": (
            extra[mask, EXTRA_COLUMNS["wheel_contact_gap"]].max(0) * 1000
        ).tolist(),
        "ods_vs_pitch": ods,
        **compliance,
    }


ODS_CHANNELS = (
    "leg_q",
    "wheel_velocity",
    "wheel_torque",
    "wheel_target",
    "leg_torque",
    "leg_target",
)
"""运行变形形态：各通道同频复幅值除以俯仰角速度复幅值（腿 rad、轮速取编码器角度导数）。"""


def operating_shape(pitch: complex, **channels: np.ndarray) -> dict:
    """保存复比值的实部 / 虚部，便于跨窗做向量平均后再取增益与相位。"""
    return {
        name: {"re": np.real(value / pitch).tolist(), "im": np.imag(value / pitch).tolist()}
        for name, value in channels.items()
    }


def mean_shape(shapes: list[dict]) -> dict:
    """跨窗向量平均的运行变形形态，给出增益与相位（度）。"""
    result = {}
    for name in ODS_CHANNELS:
        values = np.mean(
            [np.asarray(s[name]["re"]) + 1j * np.asarray(s[name]["im"]) for s in shapes], axis=0
        )
        result[name] = {
            "re": values.real.tolist(),
            "im": values.imag.tolist(),
            "gain": np.abs(values).tolist(),
            "phase_deg": np.rad2deg(np.angle(values)).tolist(),
        }
    return result


def edge_window(trace: np.ndarray, extra: np.ndarray) -> dict:
    """使能沿 0.94 s：与实车 edge_metrics 相同的 0.2 s 窗 / 0.1 s 步长带通 RMS 与末 0.4 s 主频。"""
    mask = _window(trace, -1.0, WINDOW_S)
    t = trace[mask, 0].astype(float)
    pitch_raw = trace[mask, 41].astype(float)
    low = butter(4, [5, 15], btype="band", fs=1000, output="sos")
    high = butter(4, [15, 100], btype="band", fs=1000, output="sos")
    pitch = sosfiltfilt(low, pitch_raw)
    torque = sosfiltfilt(high, extra[mask, 0:4].astype(float), axis=0)
    velocity = sosfiltfilt(high, trace[mask, 7:11].astype(float), axis=0)
    rows = []
    for start in np.arange(t[0], t[-1] - 0.2 + 1e-9, 0.1):
        window = (t >= start) & (t < start + 0.2)
        rows.append(
            {
                "t0_s": float(start - t[0]),
                "pitch_5_15_rms_rad_s": float(np.sqrt(np.mean(pitch[window] ** 2))),
                "leg_torque_15_100_rms_nm_max": float(
                    np.max(np.sqrt(np.mean(torque[window] ** 2, axis=0)))
                ),
                "leg_velocity_15_100_rms_rad_s_max": float(
                    np.max(np.sqrt(np.mean(velocity[window] ** 2, axis=0)))
                ),
            }
        )
    tail = t >= t[-1] - 0.4
    tail_frequency = dominant_frequency(t[tail], pitch_raw[tail], 3.0, 40.0)
    hg, r2, _, _ = harmonic(t[tail], pitch_raw[tail], tail_frequency)
    late = [row for row in rows if row["t0_s"] >= 0.5]
    return {
        "windows": rows,
        "late_pitch_5_15_rms": float(np.mean([row["pitch_5_15_rms_rad_s"] for row in late])),
        "late_leg_torque_hf_rms": float(
            np.mean([row["leg_torque_15_100_rms_nm_max"] for row in late])
        ),
        "peak_leg_torque_hf_rms": float(max(row["leg_torque_15_100_rms_nm_max"] for row in rows)),
        "tail_frequency_hz": tail_frequency,
        "tail_pitch_amplitude_rad_s": float(abs(hg)),
        "tail_fit_r2": float(r2),
        "leg_torque_saturation_fraction": np.mean(
            np.abs(extra[mask, 0:4]) >= 32.0 - 1e-6, axis=0
        ).tolist(),
    }


def summarize_windows(windows: list[dict]) -> dict:
    """与实车 summarize 同名同义的稳态汇总（均值 / 最小 / 最大），另给中位数与 r² 统计。"""

    def collect(getter):
        values = np.array([getter(w) for w in windows], dtype=float)
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    return {
        "steady_segments": len(windows),
        "frequency_hz": collect(lambda w: w["frequency_hz"]),
        "pitch_fit_r2": collect(lambda w: w["pitch_fit_r2"]),
        "pitch_rate_amp": collect(lambda w: w["amplitude"]["pitch_rate_rad_s"]),
        "pitch_rate_band_rms": collect(lambda w: w["band_rms_5_15"]["pitch_rate_rad_s"]),
        "leg_amp_deg_max": collect(lambda w: max(w["amplitude"]["leg_deg"])),
        "leg_band_rms_deg_max": collect(lambda w: max(w["band_rms_5_15"]["leg_deg"])),
        "wheel_band_rms_max": collect(lambda w: max(w["band_rms_5_15"]["wheel_velocity_rad_s"])),
        "wheel_target_amp_max": collect(lambda w: max(w["amplitude"]["wheel_target_rad_s"])),
        "wheel_torque_amp_max": collect(lambda w: max(w["amplitude"]["wheel_torque_nm"])),
        "policy_wheel_gain_mean": collect(
            lambda w: np.mean(w["policy_pitch_to_wheel_target"]["gain"])
        ),
        "policy_wheel_phase_left_deg": collect(
            lambda w: w["policy_pitch_to_wheel_target"]["phase_deg"][0]
        ),
        "leg_D_power_sum_w": collect(lambda w: sum(w["command_power_w"]["D"])),
        "leg_policy_power_sum_w": collect(lambda w: sum(w["command_power_w"]["P_target"])),
        "wheel_target_power_sum_w": collect(
            lambda w: sum(w["command_power_w"]["wheel_target_term"])
        ),
        "wheel_feedback_power_sum_w": collect(
            lambda w: sum(w["command_power_w"]["wheel_feedback_term"])
        ),
        "leg_torque_hf_rms_max": collect(lambda w: max(w["band_rms_15_100"]["leg_torque_nm"])),
        "wheel_saturation_fraction_max": collect(
            lambda w: max(w["wheel_torque_saturation_fraction"])
        ),
        "mean_height_m": collect(lambda w: w["mean_height_m"]),
        "ods_vs_pitch": mean_shape(
            [w["ods_vs_pitch"] for w in windows if w["pitch_fit_r2"] >= 0.5]
            or [w["ods_vs_pitch"] for w in windows]
        ),
    }


def analyze_run(
    trace: np.ndarray, extra: np.ndarray, gains, fell_at: float | None, steady: bool = True
) -> dict:
    """稳态 10 窗 + 使能沿 + 逐窗时间过程；摔倒后、或注入工况（稳态窗含受迫响应）不给稳态统计。"""
    end = trace[-1, 0]
    windows = [
        steady_window(trace, extra, lo, gains)
        for lo in STEADY_STARTS
        if lo + WINDOW_S <= end + 1e-6 and fell_at is None and steady
    ]
    course = []
    for lo in np.arange(0.0, STAND_S - WINDOW_S + 1e-9, 1.0):
        mask = _window(trace, lo, lo + WINDOW_S)
        if mask.sum() < 900:
            break
        course.append(
            float(
                welch_band_rms(trace[mask, 0].astype(float), trace[mask, 41].astype(float), 5, 15)
            )
        )
    return {
        "windows": windows,
        "summary": summarize_windows(windows) if windows else {},
        "edge": edge_window(trace, extra),
        "pitch_band_rms_time_course": {"t0_s": list(range(len(course))), "rad_s": course},
        "upright_min_cos_after_1s": float(
            np.min(np.cos(trace[trace[:, 0] >= 1, 37]) * np.cos(trace[trace[:, 0] >= 1, 38]))
        )
        if np.any(trace[:, 0] >= 1)
        else None,
    }


# ---------------------------------------------------------------- 判据与对照


def real_reference() -> dict:
    """实车 analysis.json 的逐组汇总（稳态段均值 / 范围与使能沿指标）。"""
    data = json.loads(REAL_ANALYSIS.read_text(encoding="utf-8"))
    return {row["variant"]: row for row in data["summary"]}


def real_operating_shapes() -> dict:
    """实车 A–D 稳态段的运行变形形态，与仿真同一组复比值（policy 腿顺序）；需要本机原始采集。

    时标：俯仰用 IMU 采样时刻，腿 / 轮位置用 CAN 帧时刻，目标与指令力矩用控制输入时刻；
    轮速取编码器角度（转子角 / 14）的同频导数，不用 C620 转速字段（它自身滞后约 7°）。
    """
    if not CAPTURE_SWEEP.exists():
        return {}
    from analyze_control_sync_20261012 import clock_diff, load

    result = {}
    for variant in "ABCD":
        runs = sorted((CAPTURE_SWEEP / variant).glob("run_*"))
        shapes, r2s = [], []
        for path in sorted(runs[-1].glob("burst_*.npz")) if runs else ():
            data, meta = load(path)
            flags = data["flags"].astype(np.uint32)
            if not (((flags & 1) != 0).all() and ((flags & 8) != 0).all()):
                continue
            hz = meta["monitor_before_export"]["cpu_hz"]
            t = data["input_time"]
            imu_t = t - clock_diff(data["input_cycle"], data["imu_cycle"]) / hz
            pitch = data["gyro"][:, 0]
            f = dominant_frequency(t, pitch)
            omega = 2 * np.pi * f
            hg, r2, _, _ = harmonic(imu_t, pitch, f)
            hq = np.array(
                [harmonic(data["rx_sof_time"][:, j], data["q"][:, j], f, 3)[0] for j in range(4)]
            )[REAL_RECORDER_TO_POLICY]
            hwq = np.array(
                [
                    harmonic(data["rx_sof_time"][:, 4 + j], data["wheel_q"][:, j], f, 3)[0]
                    for j in range(2)
                ]
            )
            torque = harmonic(t, data["torque"], f, 3)[0]
            target = harmonic(t, data["target"], f, 3)[0]
            shapes.append(
                operating_shape(
                    hg,
                    leg_q=hq,
                    wheel_velocity=1j * omega * hwq,
                    wheel_torque=torque[4:],
                    wheel_target=target[4:],
                    leg_torque=torque[:4][REAL_RECORDER_TO_POLICY],
                    leg_target=target[:4][REAL_RECORDER_TO_POLICY],
                )
            )
            r2s.append(float(r2))
        if shapes:
            result[variant] = {"pitch_fit_r2": r2s, **mean_shape(shapes)}
    return result


def compare_shapes(sim: dict, real: dict) -> dict:
    """逐通道 sim / real 复比值：增益比与相位差（度）；只对实车有稳定振荡的组有意义。"""
    result = {}
    for name in ODS_CHANNELS:
        s = np.asarray(sim[name]["re"]) + 1j * np.asarray(sim[name]["im"])
        r = np.asarray(real[name]["re"]) + 1j * np.asarray(real[name]["im"])
        result[name] = {
            "gain_ratio": np.abs(s / r).tolist(),
            "phase_error_deg": np.rad2deg(np.angle(s / r)).tolist(),
        }
    return result


CRITERIA = {
    "A_sustained": "A 持续自持振荡：10 窗俯仰谐波拟合 r² 中位数 ≥ 0.8、5–15 Hz RMS 中位数 ≥ 0.1 rad/s 且最小窗 ≥ 中位数一半、主频中位数 7.8–8.6 Hz",
    "D_larger_lower": "以 A 满足自持振荡为前提：D 俯仰 5–15 Hz RMS 均值大于 A，且两组 r² ≥ 0.5 窗主频中位数之比 f_D / f_A 在 0.85–0.95",
    "B_small_6hz": "B 俯仰 RMS 均值 < A 的一半；有可检出振荡（r² ≥ 0.5 的窗）且这些窗主频中位数 5.8–7.0 Hz",
    "C_small_6hz": "C 同 B",
    "E_edge_stronger": "E 使能后 0.5–0.9 s（0.2 s 窗 t0 ≥ 0.5）俯仰 5–15 Hz 带通 RMS 大于 A 同窗",
}


def _detected_frequency(summary_windows: list[dict]) -> float | None:
    """只用 r² ≥ 0.5 的窗求主频中位数；没有可检出振荡时返回 None。"""
    values = [w["frequency_hz"] for w in summary_windows if w["pitch_fit_r2"] >= 0.5]
    return float(np.median(values)) if values else None


def probe_summary(report: dict | None) -> dict | None:
    """8.3 Hz 注入撤除后的衰减：时间常数、振铃频率与等效阻尼比 ζ = 1/(2π f τ)。"""
    if report is None:
        return None
    if report["fell_at_s"] is not None or "probe_decay" not in report:
        return {"fell_at_s": report["fell_at_s"]}
    decay = report["probe_decay"]
    tau = decay["decay_time_constant_s"]
    ring = decay["dominant_hz_0_4s_after"][6]
    return {
        "decay_time_constant_s": tau,
        "ring_hz": ring,
        "zeta": 1 / (2 * np.pi * ring * tau) if tau else None,
        "ratio_1s_over_0p2s": decay["ratio_1s_over_0p2s"],
        "ratio_0p2s_over_forced": decay["ratio_0p2s_over_forced"],
        "settle_time_s_to_2x_floor": decay["settle_time_s_to_2x_floor"],
        "forced_envelope_rad_s": decay["forced_envelope_rad_s"],
    }


def assess_plant(reports: dict[str, dict]) -> dict:
    """五组判据逐条给出是否满足与关键数值。"""
    missing = [g for g in GROUPS if g not in reports]
    if missing:
        return {"complete": False, "missing_groups": missing}
    fell = {g: reports[g]["fell_at_s"] for g in GROUPS if reports[g]["fell_at_s"] is not None}
    result = {"complete": True, "fell": fell, "criteria": {}, "values": {}}
    summaries = {g: reports[g]["summary"] for g in GROUPS}
    values = result["values"]
    for g in GROUPS:
        s = summaries[g]
        values[g] = {
            "frequency_median_hz": s.get("frequency_hz", {}).get("median"),
            "detected_frequency_hz": _detected_frequency(reports[g]["windows"]),
            "r2_median": s.get("pitch_fit_r2", {}).get("median"),
            "pitch_band_rms_mean": s.get("pitch_rate_band_rms", {}).get("mean"),
            "pitch_band_rms_median": s.get("pitch_rate_band_rms", {}).get("median"),
            "pitch_band_rms_min": s.get("pitch_rate_band_rms", {}).get("min"),
            "leg_band_rms_deg_max_mean": s.get("leg_band_rms_deg_max", {}).get("mean"),
            "wheel_band_rms_max_mean": s.get("wheel_band_rms_max", {}).get("mean"),
            "edge_late_pitch_5_15_rms": reports[g]["edge"]["late_pitch_5_15_rms"],
            "edge_tail_frequency_hz": reports[g]["edge"]["tail_frequency_hz"],
            "mean_height_m": s.get("mean_height_m", {}).get("mean"),
            "probe": probe_summary(reports.get(f"{g}_probe")),
        }
    criteria = result["criteria"]
    a, d = values["A"], values["D"]
    if "A" in fell or not summaries["A"]:
        criteria["A_sustained"] = False
    else:
        criteria["A_sustained"] = bool(
            a["r2_median"] >= 0.8
            and a["pitch_band_rms_median"] >= 0.1
            and a["pitch_band_rms_min"] >= 0.5 * a["pitch_band_rms_median"]
            and 7.8 <= a["frequency_median_hz"] <= 8.6
        )
    if fell.keys() & {"A", "D"} or not summaries["D"]:
        criteria["D_larger_lower"] = False
    else:
        if a["detected_frequency_hz"] and d["detected_frequency_hz"]:
            values["D"]["frequency_ratio_to_A"] = (
                d["detected_frequency_hz"] / a["detected_frequency_hz"]
            )
        ratio_fd = values["D"].get("frequency_ratio_to_A")
        # 判据比较的是 D 与 A 的振荡；A 没有自持振荡时没有可比的频率和幅值。
        criteria["D_larger_lower"] = bool(
            criteria["A_sustained"]
            and ratio_fd is not None
            and d["pitch_band_rms_mean"] > a["pitch_band_rms_mean"]
            and 0.85 <= ratio_fd <= 0.95
        )
    for g in ("B", "C"):
        key = f"{g}_small_6hz"
        if g in fell or "A" in fell or not summaries[g] or not summaries["A"]:
            criteria[key] = False
            continue
        small = values[g]["pitch_band_rms_mean"] < 0.5 * a["pitch_band_rms_mean"]
        detected = values[g]["detected_frequency_hz"]
        values[g]["amplitude_ratio_to_A"] = (
            values[g]["pitch_band_rms_mean"] / a["pitch_band_rms_mean"]
        )
        criteria[key] = bool(small and detected is not None and 5.8 <= detected <= 7.0)
        criteria[f"{g}_small_only"] = bool(small)
    criteria["E_edge_stronger"] = bool(
        values["E"]["edge_late_pitch_5_15_rms"] > a["edge_late_pitch_5_15_rms"]
    )
    result["all_satisfied"] = all(criteria[k] for k in CRITERIA)
    result["satisfied_count"] = int(sum(bool(criteria[k]) for k in CRITERIA))
    return result


REAL_KEYS = (
    ("frequency_hz", "主频 Hz"),
    ("pitch_rate_band_rms", "俯仰 5–15 Hz RMS rad/s"),
    ("leg_band_rms_deg_max", "最差腿 5–15 Hz RMS °"),
    ("wheel_band_rms_max", "最差轮速 5–15 Hz RMS rad/s"),
    ("wheel_target_amp_max", "轮速目标同频幅值 rad/s"),
    ("policy_wheel_gain_mean", "策略 俯仰→轮速目标 增益"),
    ("leg_policy_power_sum_w", "腿 P_target 功率 W"),
    ("leg_D_power_sum_w", "腿 D 功率 W"),
    ("wheel_target_power_sum_w", "轮目标项功率 W"),
    ("wheel_feedback_power_sum_w", "轮反馈项功率 W"),
)


def compare_with_real(reports: dict[str, dict], real: dict, real_shapes: dict) -> dict:
    """逐组逐项列出仿真与实车（稳态段均值）；E 只有使能沿可比；A / D 另给运行变形形态对照。"""
    table = {}
    for g in GROUPS:
        sim = reports[g]["summary"]
        row = {}
        if g in ("A", "D") and g in real_shapes and sim:
            row["ods_sim_over_real"] = compare_shapes(sim["ods_vs_pitch"], real_shapes[g])
        for key, _ in REAL_KEYS:
            real_value = real[g].get(key, {}).get("mean") if g in real else None
            sim_value = sim.get(key, {}).get("mean")
            row[key] = {
                "sim": sim_value,
                "real": real_value,
                "ratio": sim_value / real_value
                if sim_value is not None and real_value not in (None, 0)
                else None,
            }
        row["edge_late_pitch_5_15_rms"] = {
            "sim": reports[g]["edge"]["late_pitch_5_15_rms"],
            "real": real[g].get("edge_late_pitch_5_15_rms") if g in real else None,
        }
        row["edge_tail_frequency_hz"] = {
            "sim": reports[g]["edge"]["tail_frequency_hz"],
            "real": real[g].get("edge_tail_frequency_hz") if g in real else None,
        }
        table[g] = row
    return table


def load_reports() -> dict[str, dict[str, dict]]:
    """读取 runs/<plant>/<group>.json。"""
    reports: dict[str, dict[str, dict]] = {}
    for path in sorted((OUT / "runs").glob("*/*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        tag = report["group"] + ("_probe" if report.get("probe") else "")
        reports.setdefault(path.parent.name, {})[tag] = report
    return reports


FAMILY_CONCLUSIONS = {
    "baseline": "基线复现 predict_gain_sweep：A–D 没有 5–15 Hz 振荡（A 俯仰 RMS 0.010 rad/s，实车 0.385），只有 E 在 7.5 Hz 持续振荡；8.3 Hz 注入后 A 的振铃阻尼比约 0.17。",
    "a_contact": "否定。轮地接触固有频率 4–34 Hz、阻尼比 0.1–1.25 对 6–9 Hz 闭环模态几乎没有作用：A 始终无振荡（俯仰 RMS ≤ 0.011 rad/s），D 不失稳；ζ 0.1 的欠阻尼接触反而让 E 静下来。",
    "a_tire": "部分支持，必要但不充分。转子编码器到接地点之间的扭转柔性 k 30–60 N·m/rad 是唯一同时被实车数据和闭环仿真支持的结构：实车整机俯仰平衡要求的俯仰角速度由刚性的 0.70×/+31° 变为 0.95–1.04×/+14–24°、左右轮纯滚动不一致由 65% 降到 29–41%；闭环里 k 30–50 时 D 在 6.5–7.1 Hz 失稳（k 60 时临界）、E 在 6.3–7.0 Hz 失稳、C 的 6 Hz 模态阻尼比降到约 0.01–0.03、B 静默、E 使能沿最强，D 的运行变形形态与实车同量级。但 A 始终不自激（8–9.5 Hz 振铃阻尼比 0.04–0.07），D 主频比实车低 0.2–1.5 Hz、俯仰幅值只有实车的 1/6–1/2；k ≤ 20 或间隙 ≥ 1° 时 A/B/C 出现实车没有的 3.5–4.6 Hz 极限环。",
    "b_wheel_lag": "纯电气滞后否定，作为 (a) 的等效描述保留。轮力矩额外一阶滞后 10 ms 或纯延迟 5 ms 的效果与胎面柔性相同（D/E 在 6.5–6.9 Hz 失稳、A 不失稳；7.5 ms / 3 ms 时 D 临界），且 D 的运行变形形态与实车最吻合（腿 0.8–1.1 倍、轮速 1.03–1.12 倍、相位差 ≤ 23°）；但实车电流反馈只落后指令 8–11°（约 3 ms，基线已含），更大的有效滞后只能来自机械；滞后 ≥ 15 ms 或延迟 ≥ 10 ms 时 A/B 在 3–4 Hz 失稳，与实车矛盾。实测的轮速测量滞后 2.5 ms 只轻微加重 D。",
    "c_wheel_armature": "否定。0.0035 / 0.0075 只小幅移动模态频率，A–D 均无振荡；与胎面柔性组合时 0.0075 让 A/B 出现 3 Hz 左右的低频振荡（胎面 25 时最明显），0.0035 与 0.005 几乎无差别。",
    "d_leg_chain": "否定。15–30 N·m/rad 使能后 0.4 s 内摔倒，60 N·m/rad 时 A/B/E 在 1.4–11 s 内摔倒；120–500 N·m/rad 时策略落到另一种站姿（站高 0.25–0.30 m，电机侧腿角偏离实车 8–10°），A/B 出现 3–5 Hz 不规则运动、E 摔倒；用实车数据扣除链扭转后整机平衡相位误差反而变大（+31° → +33–40°）。≥ 1000 N·m/rad 时驱动杆自身惯量太小、1 ms 步长下数值发散，未评价。",
    "e_body": "否定。俯仰惯量 ×0.5–×2 让模态频率反比移动（×0.5 时 A 振铃约 9 Hz、D 约 8 Hz），但不让 A 自激，×0.5 还让 B 在 3.9 Hz 剧烈振荡；质心 ±4 cm 不让 A–D 振荡（+4 cm 反而让 E 静下来）。",
    "combo": "否定。胎面柔性 25–60 N·m/rad 叠加纯延迟 2/3 ms、一阶滞后 5/7.5 ms、转子惯量 0.0035/0.0075、俯仰惯量 ×0.75 或实测轮速测量滞后 2.5 ms：轮通道相位滞后加大只会让 D/E 振得更大、频率更低（5.3–7.3 Hz），A 要么仍是 7–9.5 Hz 振铃阻尼比 0.03–0.08 的阻尼模态，要么先在 3–4 Hz 失稳；没有组合让 A 在 7.8–8.6 Hz 自激。",
    "x_leg_lag": "清单外诊断，否定。腿力矩滞后取电流反馈实测上限（15 / 18 ms 一阶，或 8 ms 纯延迟 + 5 ms）不让 A 失稳，与胎面柔性 / 轮滞后组合时反而让 D 稳定下来；8 ms 纯延迟与胎面柔性组合时 A/B 在 3 Hz 大幅失稳。",
}
OVERALL_CONCLUSION = (
    "没有候选（117 个，单因素、组合与两项清单外诊断）同时满足五条判据，A 在任何候选下都不在 7.8–8.6 Hz 自激。"
    "仿真缺的结构里能确认的一项是轮通道（转子编码器到接地点之间）约 30–60 N·m/rad 的扭转柔性（或等效的 5–10 ms 轮力矩有效滞后）："
    "它解释了实车同步数据刚体平衡不闭合的大部分，并在闭环里复现 D/E 失稳、C 的轻阻尼 6 Hz 模态和 B 静默，"
    "但不足以让高腿增益的 A 在 8.2 Hz 失去阻尼，且 D 的主频与幅值仍偏低；"
    "说明还缺一项让 Kp 60 / Kd 3 工况在约 8 Hz 失去阻尼、并整体抬高模态频率的结构，它不在 (a)–(e) 中。"
)
NEXT_MEASUREMENTS = [
    "轮通道静态扭转标定：机身架空、轮胎压住地面或夹住胎面，轮电机给 ±0.5 / 1 / 2 N·m 慢变力矩，读编码器角，得到刚度与间隙；若 30–60 N·m/rad，2 N·m 时转角应为 2–4°。",
    "轮通道动态：同一夹持下 2–20 Hz 小幅力矩扫频，转子编码器加胎面外部测量（第二编码器、高速相机或轮上 IMU），得到转子–胎面传递函数与阻尼。",
    "腿传动与执行器：机身固定、轮离地，单腿 1–20 Hz 小幅力矩扫频，电机编码器加大腿外部角度（IMU 或相机），得到链 / 减速器柔性与各腿真实力矩滞后与增益（10-12 电流反馈增益 0.64–1.20 不一致）。",
    "气弹簧实际力与摩擦：核对 400 / 360 N 假设与迟滞，滞回直接改变腿模态阻尼。",
    "若用户批准在实车闭环中注入：0.2 N·m 已知正弦轮力矩，测 5–12 Hz 的被控对象响应（2026-10-12 会话明确禁止向实车发激励，需重新授权）。",
]


def a_mode_zeta(probe: dict | None) -> float:
    """A 组 8.3 Hz 注入撤除后、振铃落在 7–9.5 Hz（实车 A 模态附近）时的阻尼比；否则记 9 排在后面。

    不衰减、摔倒或振铃落到 5–6 Hz 的多是另一个模态（常见的是 3–5 Hz 低频失稳或 6 Hz 的 C 型模态）。
    """
    if not probe or probe.get("zeta") is None or not 7.0 <= probe["ring_hz"] <= 9.5:
        return 9.0
    return probe["zeta"]


def family_summary(rows: dict, names: list[str]) -> dict:
    """一类候选的判据通过计数与"最接近"的一个（满足条数多、A 注入后阻尼比小者优先）。"""
    complete = [n for n in names if rows[n]["verdict"].get("complete")]
    passes = {
        key: [n for n in complete if rows[n]["verdict"]["criteria"].get(key)] for key in CRITERIA
    }

    def a_zeta(name):
        return a_mode_zeta(rows[name]["verdict"]["values"]["A"].get("probe"))

    best = min(complete, key=lambda n: (-rows[n]["verdict"]["satisfied_count"], a_zeta(n)))
    return {
        "plants": sorted(names),
        "complete": len(complete),
        "all_satisfied": [n for n in complete if rows[n]["verdict"]["all_satisfied"]],
        "criterion_pass_counts": {key: len(value) for key, value in passes.items()},
        "criterion_passes": passes,
        "closest": best,
        "conclusion": FAMILY_CONCLUSIONS.get(rows[names[0]]["family"], ""),
    }


def balance_summary() -> dict | None:
    """tread_twist_balance.json 的摘要：刚性一行与平衡最接近闭合的几行。"""
    path = OUT / "tread_twist_balance.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]

    def distance(row):
        return (
            abs(row["required_over_measured_gain"] - 1)
            + abs(row["required_over_measured_phase_deg"]) / 60
        )

    return {
        "source": data["source"],
        "rigid": rows[0],
        "closest_rows": sorted(rows, key=distance)[:6],
        "file": str(path),
    }


def assess() -> dict:
    """汇总全部已完成候选，写 assessment.json。"""
    real = real_reference()
    real_shapes = real_operating_shapes()
    plants = all_plants()
    reports = load_reports()
    rows = {}
    for name, by_group in reports.items():
        stored = next(iter(by_group.values()))["plant"]
        stored["wheel_tau_ms"] = tuple(stored["wheel_tau_ms"])
        plant = plants.get(name) or Plant(**stored)
        verdict = assess_plant(by_group)
        rows[name] = {
            "family": plant.family,
            "label": plant.label,
            "plant": asdict(plant),
            "verdict": verdict,
            "comparison": compare_with_real(by_group, real, real_shapes)
            if verdict.get("complete")
            else None,
        }
    family_names: dict[str, list[str]] = {}
    for name, row in rows.items():
        family_names.setdefault(row["family"], []).append(name)
    families = {key: family_summary(rows, names) for key, names in family_names.items()}

    ranking = sorted(
        (name for name in rows if rows[name]["verdict"].get("complete")),
        key=lambda n: (
            -rows[n]["verdict"]["satisfied_count"],
            a_mode_zeta(rows[n]["verdict"]["values"]["A"].get("probe")),
        ),
    )
    assessment = {
        "date": "2026-10-14",
        "policy": POLICY,
        "base_recipe": "scripts/predict_gain_sweep_20261013.py（4e1aeb0、陀螺 Q/R、腿速度 4.7 ms、腿 12 ms / 轮 3、4 ms 一阶滞后、弹簧 400/360 N 第 6 s 淡入、实测静置噪声、失能落地 3 s 再使能静站 18 s）",
        "spectral_method": "与 scripts/analyze_gain_sweep_20261014.py 相同：谐波回归主频 3–14 Hz；Welch fs 1000、nperseg 900、noverlap 450、5–15 Hz；同频指令层功率 0.5·Re(F·conj(v))；稳态 10 个 0.94 s 窗（8–17.94 s）；使能沿 0.94 s 内 0.2 s 窗 / 0.1 s 步",
        "criteria": CRITERIA,
        "real_reference": real,
        "real_operating_shapes": real_shapes,
        "plants": rows,
        "families": families,
        "ranking": ranking,
        "all_satisfied": [n for n in ranking if rows[n]["verdict"]["all_satisfied"]],
        "tread_twist_balance": balance_summary(),
        "conclusion": OVERALL_CONCLUSION,
        "next_measurements": NEXT_MEASUREMENTS,
    }
    (OUT / "assessment.json").write_text(
        json.dumps(assessment, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (OUT / "summary_tables.md").write_text(markdown_tables(assessment), encoding="utf-8")
    return assessment


TABLE_PLANTS = (
    "baseline",
    "contact_f17_z0.3",
    "tire_k30_z0.1",
    "tire_k40_z0.1",
    "tire_k60_z0.1",
    "tire_k40_gap1deg",
    "wheel_lag_10ms",
    "wheel_delay_5ms",
    "wheel_speed_lag_2.5ms",
    "wheel_armature_0.0075",
    "chain_k300_z0.1",
    "chain_k500_z0.1",
    "body_pitch_inertia_x0.5",
    "body_com_dz+4cm",
    "combo_tire40_speedlag2.5",
    "combo_tire25_delay3",
    "combo_tire40_pitch0.75",
    "leglag_legtau15_tire40",
)
"""文档里逐组展开的代表性候选：每类的基线 / 最接近 / 边界各取一两个。"""
COMPARISON_PLANTS = (
    "baseline",
    "tire_k40_z0.1",
    "wheel_lag_10ms",
    "combo_tire40_speedlag2.5",
    "combo_tire25_delay3",
)
COMPARISON_KEYS = (
    "frequency_hz",
    "pitch_rate_band_rms",
    "leg_band_rms_deg_max",
    "wheel_band_rms_max",
    "wheel_target_amp_max",
    "policy_wheel_gain_mean",
    "leg_policy_power_sum_w",
    "leg_D_power_sum_w",
    "wheel_target_power_sum_w",
    "wheel_feedback_power_sum_w",
)


def _f(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def _probe_text(probe: dict | None) -> str:
    if not probe:
        return "—"
    if "decay_time_constant_s" not in probe:
        return "摔倒"
    if probe["zeta"] is None:
        return f"{probe['ring_hz']:.1f}·不衰减"
    return f"{probe['ring_hz']:.1f}·{probe['zeta']:.2f}"


def _group_cells(verdict: dict) -> list[str]:
    """表 2 的 A–E 单元格；摔倒的组直接写摔倒时刻。"""
    v, fell = verdict["values"], verdict["fell"]

    def cell(group, text):
        return f"摔倒 {fell[group]:.1f} s" if group in fell else text

    a, b, c, d, e = (v[g] for g in "ABCDE")
    return [
        cell(
            "A",
            f"{_f(a['detected_frequency_hz'], 1)} / {_f(a['pitch_band_rms_mean'], 3)} / "
            + _probe_text(a["probe"]),
        ),
        cell("B", _f(b["pitch_band_rms_mean"], 3)),
        cell("C", f"{_f(c['pitch_band_rms_mean'], 3)} / " + _probe_text(c["probe"])),
        cell("D", f"{_f(d['detected_frequency_hz'], 1)} / {_f(d['pitch_band_rms_mean'], 3)}"),
        f"{_f(e['detected_frequency_hz'], 1)} / "
        f"{_f(e['edge_late_pitch_5_15_rms'], 2)}:{_f(a['edge_late_pitch_5_15_rms'], 2)}",
        _f(a["mean_height_m"], 3),
    ]


def markdown_tables(assessment: dict) -> str:
    """文档用表格片段：类别判据计数、代表性候选逐组结果、与实车逐项对照、实车一致性检验。"""
    rows = assessment["plants"]
    real = assessment["real_reference"]
    lines = ["### 表 1  五类候选与组合的判据通过数", ""]
    lines.append(
        "| 类别 | 候选数 | A 自持 7.8–8.6 Hz | D 大且低约 10% | B 小且 6–7 Hz | C 小且 6–7 Hz | "
        "E 使能沿强于 A | 最接近 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for key, family in sorted(assessment["families"].items()):
        counts = family["criterion_pass_counts"]
        lines.append(
            f"| {key} | {family['complete']} | {counts['A_sustained']} | "
            f"{counts['D_larger_lower']} | {counts['B_small_6hz']} | {counts['C_small_6hz']} | "
            f"{counts['E_edge_stronger']} | {family['closest']} |"
        )
    lines += [
        "",
        "### 表 2  代表性候选逐组结果（稳态 10 窗；主频取 r² ≥ 0.5 窗的中位数，— 为无可检出振荡；"
        "振铃 = 8.3 Hz 注入撤除后的频率与阻尼比）",
        "",
        "| 候选 | A 主频 Hz / 俯仰 RMS / 振铃 Hz·ζ | B 俯仰 RMS | C 俯仰 RMS / 振铃 Hz·ζ | "
        "D 主频 Hz / 俯仰 RMS | E 主频 Hz / 使能沿 E:A | 站高 m |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in TABLE_PLANTS:
        if name in rows and rows[name]["verdict"].get("complete"):
            lines.append(f"| {name} | " + " | ".join(_group_cells(rows[name]["verdict"])) + " |")
    lines.append(
        f"| **实车** | {real['A']['frequency_hz']['mean']:.2f} / "
        f"{real['A']['pitch_rate_band_rms']['mean']:.3f} / 自持 | "
        f"{real['B']['pitch_rate_band_rms']['mean']:.3f}（6.4 Hz 间歇） | "
        f"{real['C']['pitch_rate_band_rms']['mean']:.3f}（6.2 Hz 间歇） | "
        f"{real['D']['frequency_hz']['mean']:.2f} / {real['D']['pitch_rate_band_rms']['mean']:.3f} | "
        f"{real['E']['edge_tail_frequency_hz']:.1f} / {real['E']['edge_late_pitch_5_15_rms']:.2f}:"
        f"{real['A']['edge_late_pitch_5_15_rms']:.2f} | — |"
    )
    lines += [
        "",
        "### 表 3  与实车逐项对照（稳态段均值；功率为同频指令层功率之和，W）",
        "",
        "| 候选 | 组 | 主频 Hz | 俯仰 RMS | 最差腿 RMS ° | 最差轮速 RMS | 轮速目标幅值 | "
        "策略 俯仰→轮速目标 | 腿 P_target | 腿 D | 轮目标项 | 轮反馈项 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for group in "ABCD":
        lines.append(
            f"| **实车** | {group} | "
            + " | ".join(_f(real[group][key]["mean"], 3) for key in COMPARISON_KEYS)
            + " |"
        )
    for name in COMPARISON_PLANTS:
        comparison = rows[name]["comparison"]
        for group in "ABCDE":
            lines.append(
                f"| {name} | {group} | "
                + " | ".join(_f(comparison[group][key]["sim"], 3) for key in COMPARISON_KEYS)
                + " |"
            )
    balance = assessment.get("tread_twist_balance")
    if balance:
        lines += [
            "",
            "### 表 4  实车同步数据的刚体一致性检验（2026-10-12 burst_0003，A 组增益，8.20 Hz）",
            "",
            "| 轮通道扭转 k N·m/rad | 转子惯量 | 腿链 k | 轮扭转幅值 ° L/R | 左右纯滚动不一致 | "
            "IMU 残差比 | 平衡要求俯仰 / 实测 |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in [balance["rigid"], *balance["closest_rows"]]:
            twist = "/".join(f"{x:.2f}" for x in row["twist_amplitude_deg"])
            lines.append(
                f"| {row['tread_stiffness_nm_per_rad'] or '刚性'} | {row['rotor_armature']:g} | "
                f"{row['chain_stiffness_nm_per_rad'] or '刚性'} | {twist} | "
                f"{row['rolling_left_right_disagreement']:.2f} | {row['imu_residual_fraction']:.2f} | "
                f"{row['required_over_measured_gain']:.2f}×、{row['required_over_measured_phase_deg']:+.0f}° |"
            )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 实车数据的独立检验


TWIST_STIFFNESS = (None, 1000.0, 300.0, 160.0, 100.0, 80.0, 60.0, 50.0, 40.0, 30.0, 25.0, 20.0)
TWIST_ROTOR_ARMATURE = (0.0, 0.0035, 0.005)
CHAIN_TWIST = ((1000.0, 0.01), (500.0, 0.01), (300.0, 0.01), (200.0, 0.01), (100.0, 0.01))
"""腿链传动扭转 (k, 电机侧 armature)：腿输出角 = 电机编码器角 − (τ_m + J_m ω² θ_m) / k。"""
TWIST_WITH_CHAIN = ((None, 0.0), (40.0, 0.005), (50.0, 0.0035), (30.0, 0.005), (60.0, 0.0035))


def tread_twist_balance() -> dict:
    """不经仿真闭环、只用实车同步采集检验候选 (a)/(d)：编码器角减去传动扭转后，刚体机构是否闭合。

    沿用 scripts/diagnose_sync_mechanics_20261012.py（2026-10-12 burst_0003，A 组增益，8.20 Hz）：
    纯滚动下左右轮各自给出的髋轴加速度应一致；绕两轮接触连线的整机角动量平衡要求的俯仰角速度
    应等于实测；IMU 线加速度扣掉一个拟合的安装偏置后应与滚动预测一致。刚性时三者分别差 65%、
    0.70×/31° 与 15%。轮扭转 φ = (τ_m + J_r ω² θ_r) / k，τ_m 用电流折算的实际轮电机力矩、
    θ_r 是编码器角（转子角 / 14）；腿链扭转同理用腿电机电流力矩与电机编码器角。
    """
    import diagnose_sync_mechanics_20261012 as sync

    real = sync.real_motion()
    geometry = sync.Geometry(real["mean_q"], real["rotation"])
    omega = 2 * np.pi * sync.FREQUENCY
    measured_pitch = real["hgyro"][1]
    alpha = 1j * omega * real["hgyro"]
    offset_map = np.column_stack([np.cross(alpha, axis) for axis in np.eye(3)])
    grid = [
        (k, j, None, 0.0) for k in TWIST_STIFFNESS for j in (TWIST_ROTOR_ARMATURE if k else (0.0,))
    ]
    grid += [(k, j, c, cj) for k, j in TWIST_WITH_CHAIN for c, cj in CHAIN_TWIST]
    rows = []
    for stiffness, rotor, chain, chain_armature in grid:
        hq = real["hq"].copy()
        twist = np.zeros(2, dtype=complex)
        if stiffness:
            twist = (real["hcurrent"][4:] + rotor * omega**2 * hq[4:]) / stiffness
            hq[4:] = hq[4:] - twist
        chain_twist = np.zeros(4, dtype=complex)
        if chain:
            chain_twist = (real["hcurrent"][:4] + chain_armature * omega**2 * hq[:4]) / chain
            hq[:4] = hq[:4] - chain_twist
        predictions = geometry.rolling_acceleration(hq, real["hgyro"])
        residual_target = real["haccel_imu"] - predictions.mean(0)
        offset = np.linalg.lstsq(
            np.vstack([offset_map.real, offset_map.imag]),
            np.r_[residual_target.real, residual_target.imag],
            rcond=None,
        )[0]
        imu_residual = residual_target - offset_map @ offset
        zero = real["hgyro"].copy()
        zero[1] = 0
        r0 = geometry.pitch_balance_residual(hq, zero)[0][0]
        probe = zero.copy()
        probe[1] = 0.01
        coefficient = (geometry.pitch_balance_residual(hq, probe)[0][0] - r0) / 0.01
        required = -r0 / coefficient
        rows.append(
            {
                "tread_stiffness_nm_per_rad": stiffness,
                "rotor_armature": rotor,
                "chain_stiffness_nm_per_rad": chain,
                "chain_motor_armature": chain_armature,
                "twist_amplitude_deg": np.rad2deg(np.abs(twist)).tolist(),
                "chain_twist_amplitude_deg": np.rad2deg(np.abs(chain_twist)).tolist(),
                "encoder_amplitude_deg": np.rad2deg(np.abs(real["hq"])).tolist(),
                "rolling_left_right_disagreement": float(
                    np.linalg.norm(predictions[0] - predictions[1])
                    / np.linalg.norm(predictions.mean(0))
                ),
                "rolling_base_acceleration_amplitude": np.abs(predictions).tolist(),
                "imu_offset_fit_m": offset.tolist(),
                "imu_residual_fraction": float(
                    np.linalg.norm(imu_residual) / np.linalg.norm(real["haccel_imu"])
                ),
                "required_pitch_gyro_amplitude": float(abs(required)),
                "required_over_measured_gain": float(abs(required / measured_pitch)),
                "required_over_measured_phase_deg": float(
                    np.rad2deg(np.angle(required / measured_pitch))
                ),
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    result = {
        "source": "scripts/diagnose_sync_mechanics_20261012.py real_motion()（2026-10-12 burst_0003，60/3/0.2）",
        "frequency_hz": sync.FREQUENCY,
        "measured_pitch_gyro_amplitude": float(abs(measured_pitch)),
        "imu_acceleration_amplitude": np.abs(real["haccel_imu"]).tolist(),
        "rows": rows,
    }
    (OUT / "tread_twist_balance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return result


# ---------------------------------------------------------------- 入口


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("simulate", "assess", "mjcf", "balance"), default="simulate"
    )
    parser.add_argument("--plants", nargs="*", default=None, help="只跑指定候选名")
    parser.add_argument("--families", nargs="*", default=None, help="只跑指定类别")
    parser.add_argument("--groups", nargs="*", default=list(GROUPS), choices=list(GROUPS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--policy", default=POLICY, help="ONNX 路径（Pod 上与本机不同）")
    parser.add_argument("--no-trace", action="store_true", help="不保存 npz 轨迹")
    parser.add_argument("--skip-done", action="store_true", help="已有 JSON 的运行跳过")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["static"],
        choices=("static", "probe"),
        help="static：零指令静站；probe：8.3 Hz 观测注入后看衰减",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plants = all_plants()
    selected = [
        p
        for p in plants.values()
        if (args.plants is None or p.name in args.plants)
        and (args.families is None or p.family in args.families)
    ]
    if args.stage == "balance":
        tread_twist_balance()
        return
    if args.stage == "assess":
        assessment = assess()
        print(
            json.dumps(
                {
                    "all_satisfied": assessment["all_satisfied"],
                    "ranking": assessment["ranking"][:10],
                },
                ensure_ascii=False,
            )
        )
        return
    for plant in selected:
        plant_mjcf(plant)
        mujoco.MjModel.from_xml_path(str(OUT / "mjcf" / f"{plant.name}.xml"))
    if args.stage == "mjcf":
        return
    runs = [
        run
        for plant in selected
        for group in args.groups
        for mode in args.modes
        if (
            run := Run(
                plant,
                group,
                policy=args.policy,
                save_trace=not args.no_trace,
                probe=mode == "probe",
            )
        )
        and not (args.skip_done and (run_dir(plant) / f"{run.tag}.json").exists())
    ]
    print(json.dumps({"runs": len(runs), "workers": args.workers}), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(simulate, run): run for run in runs}
        for future in as_completed(futures):
            run = futures[future]
            try:
                future.result()
            except Exception as error:
                print(
                    json.dumps(
                        {"plant": run.plant.name, "group": run.group, "error": repr(error)},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
