"""从默认站姿被动释放（不跑策略）的对照录像：验证默认站姿的静平衡与零动作时的 plant 行为。

四个画面同步播放（2×2）：
  左上  新默认 · 四根主动杆锁死在默认角 · 轮自由（纯倒立摆静平衡测试，轮子去掉摩擦损耗）
  右上  旧默认（2026-09-05 之前）· 同样锁腿 · 轮自由
  左下  新默认 · 电机零动作：腿 PD（kp/kd 取 RobotConfig）保默认角、轮速度环目标 0，气弹簧 300 N 在
  右下  新默认 · 电机断电：只剩重力、接触与 300 N 气弹簧

用法：
    uv run python scripts/record_passive_default_pose.py --output passive_default_pose.mp4 [--seconds 6] [--fps 50]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from se3_shared import RobotConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MJCF = (
    _REPO_ROOT
    / "assets"
    / "robots"
    / "serialleg"
    / "mjcf"
    / "serialleg_closed_chain_v3_train_obb_trim.xml"
)
_ACTIVE_JOINTS = ("lf0_Joint", "l_drive_bar_Joint", "rf0_Joint", "r_drive_bar_Joint")
_WHEEL_JOINTS = ("l_wheel_Joint", "r_wheel_Joint")
_OLD_DEFAULT = {
    "lf0_Joint": -0.275422946189,
    "l_drive_bar_Joint": -1.592100148957,
    "rf0_Joint": 0.275422946189,
    "r_drive_bar_Joint": 1.592100148957,
    "l_wheel_Joint": 0.0,
    "r_wheel_Joint": 0.0,
    "lf1_Joint": -1.242259649307,
    "rf1_Joint": 1.242259649307,
    "l_coupler_Joint": 1.401266340000,
    "r_coupler_Joint": -1.401269410000,
}
_FONT_CANDIDATES = (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf")


@dataclass(frozen=True)
class Scenario:
    title: str
    joint_pos: dict[str, float]
    mode: str  # locked | pd_zero_action | motors_off


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _build_model(scenario: Scenario) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(_MJCF))
    if scenario.mode == "locked":
        for name in _ACTIVE_JOINTS:
            equality = spec.add_equality()
            equality.name = f"lock_{name}"
            equality.type = mujoco.mjtEq.mjEQ_JOINT
            equality.name1 = name
            data = np.zeros(mujoco.mjNEQDATA)
            data[0] = scenario.joint_pos[name]
            equality.data = data
            equality.solref = np.asarray([0.005, 1.0])
        for name in _WHEEL_JOINTS:
            joint = spec.joint(name)
            joint.frictionloss = 0.0
    model = spec.compile()
    # 与 sim2x 一致：把 MJCF 自带 floor 对齐到训练 terrain plane。
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor >= 0:
        model.geom_friction[floor] = (1.0, 0.005, 1.0e-4)
        model.geom_solref[floor] = (0.02, 1.0)
        model.geom_solimp[floor] = (0.9, 0.95, 0.001, 0.5, 2.0)
        model.geom_condim[floor] = 3
        model.geom_contype[floor] = 1
        model.geom_conaffinity[floor] = 1
    return model


def _set_pose(
    model: mujoco.MjModel, data: mujoco.MjData, joint_pos: dict[str, float], height: float
) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = (0.0, 0.0, height)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    for name, value in joint_pos.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MJCF 缺少关节 {name}")
        data.qpos[model.jnt_qposadr[joint_id]] = float(value)
    mujoco.mj_forward(model, data)


def _com_x_offset_mm(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    total = float(np.sum(model.body_mass))
    com = np.zeros(3)
    for body_id in range(model.nbody):
        com += float(model.body_mass[body_id]) * data.xipos[body_id]
    com /= total
    left = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_Link")]
    right = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_Link")]
    return float((com[0] - 0.5 * (left[0] + right[0])) * 1000.0)


def _pitch_deg(data: mujoco.MjData) -> float:
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, data.qpos[3:7])
    gravity_body = rot.reshape(3, 3).T @ np.asarray((0.0, 0.0, -1.0))
    return math.degrees(math.asin(float(np.clip(gravity_body[0], -1.0, 1.0))))


class _Panel:
    def __init__(self, scenario: Scenario, cfg: RobotConfig, width: int, height: int) -> None:
        self.scenario = scenario
        self.cfg = cfg
        self.model = _build_model(scenario)
        self.data = mujoco.MjData(self.model)
        _set_pose(self.model, self.data, scenario.joint_pos, cfg.default_base_height)
        self.initial_com_offset_mm = _com_x_offset_mm(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 1.35
        self.camera.azimuth = 90.0
        self.camera.elevation = -12.0
        self.active_dofs = [
            int(
                self.model.jnt_dofadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
            )
            for name in _ACTIVE_JOINTS
        ]
        self.active_qpos = [
            int(
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
            )
            for name in _ACTIVE_JOINTS
        ]
        self.wheel_dofs = [
            int(
                self.model.jnt_dofadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
            )
            for name in _WHEEL_JOINTS
        ]
        self.leg_targets = np.asarray([scenario.joint_pos[name] for name in _ACTIVE_JOINTS])
        self.initial_x = float(self.data.qpos[0])
        self.tip_time: float | None = None
        self.first_frame = self.render(0.0)

    def _apply_zero_action_actuators(self) -> None:
        cfg = self.cfg
        q = self.data.qpos[self.active_qpos]
        qd = self.data.qvel[self.active_dofs]
        torque = cfg.leg_kp * (self.leg_targets - q) - cfg.leg_kd * qd
        limits = np.asarray(cfg.torque_limits[:4])
        self.data.qfrc_applied[self.active_dofs] = np.clip(torque, -limits, limits)
        wheel_speed = self.data.qvel[self.wheel_dofs]
        wheel_torque = cfg.wheel_kd * (0.0 - wheel_speed)
        wheel_limits = np.asarray(cfg.torque_limits[4:6])
        self.data.qfrc_applied[self.wheel_dofs] = np.clip(wheel_torque, -wheel_limits, wheel_limits)

    def step_to(self, t_target: float) -> None:
        while self.data.time < t_target - 1.0e-9:
            if self.scenario.mode == "pd_zero_action":
                self._apply_zero_action_actuators()
            mujoco.mj_step(self.model, self.data)
            if self.tip_time is None and abs(_pitch_deg(self.data)) > 10.0:
                self.tip_time = float(self.data.time)

    def render(self, t: float) -> np.ndarray:
        self.camera.lookat[:] = (float(self.data.qpos[0]), float(self.data.qpos[1]), 0.22)
        self.renderer.update_scene(self.data, camera=self.camera)
        frame = self.renderer.render().copy()
        pitch = _pitch_deg(self.data)
        base_z = float(self.data.qpos[2]) * 1000.0
        drift = (float(self.data.qpos[0]) - self.initial_x) * 1000.0
        leg_dev = math.degrees(
            float(np.max(np.abs(self.data.qpos[self.active_qpos] - self.leg_targets)))
        )
        lines = [
            self.scenario.title,
            f"t = {t:4.2f} s   俯仰 {pitch:+6.2f}°（前倾为正）   base 高 {base_z:5.0f} mm   前后位移 {drift:+7.1f} mm",
            f"释放瞬间整机质心-轮轴 x 偏差 {self.initial_com_offset_mm:+.1f} mm   腿最大偏离默认角 {leg_dev:5.2f}°",
        ]
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle((0, 0, image.width, 78), fill=(0, 0, 0, 150))
        for index, text in enumerate(lines):
            draw.text(
                (10, 6 + 24 * index),
                text,
                font=_FONT_LARGE if index == 0 else _FONT_SMALL,
                fill=(255, 255, 255, 255),
            )
        return np.asarray(image)


_FONT_LARGE = _load_font(20)
_FONT_SMALL = _load_font(17)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    cfg = RobotConfig()
    new_default = dict(cfg.default_model_joint_pos)
    assert set(_OLD_DEFAULT) == set(new_default), "旧默认姿态关节集合与当前配置不一致"
    scenarios = (
        Scenario("新默认 · 主动杆锁死 · 轮自由（纯静平衡测试）", new_default, "locked"),
        Scenario("旧默认（09-05 前）· 主动杆锁死 · 轮自由", _OLD_DEFAULT, "locked"),
        Scenario(
            "新默认 · 电机零动作（腿 PD 保默认，轮速度环目标 0）", new_default, "pd_zero_action"
        ),
        Scenario("新默认 · 电机断电（重力 + 300 N 气弹簧）", new_default, "motors_off"),
    )
    panels = [_Panel(scenario, cfg, args.width, args.height) for scenario in scenarios]
    for panel in panels:
        print(
            f"[init] {panel.scenario.title}: 质心-轮轴 x 偏差 {panel.initial_com_offset_mm:+.2f} mm, base z {panel.data.qpos[2]:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(args.output), fps=args.fps, codec="libx264", quality=8, macro_block_size=1
    )
    frames = round(args.seconds * args.fps)
    try:
        for index in range(frames + 1):
            t = index / args.fps
            for panel in panels:
                panel.step_to(t)
            images = [panel.render(t) for panel in panels]
            top = np.concatenate((images[0], images[1]), axis=1)
            bottom = np.concatenate((images[2], images[3]), axis=1)
            grid = np.concatenate((top, bottom), axis=0)
            writer.append_data(grid)
    finally:
        writer.close()
        for panel in panels:
            panel.renderer.close()

    for panel in panels:
        tip = "未超过 10°" if panel.tip_time is None else f"{panel.tip_time:.2f} s 时超过 10°"
        print(
            f"[end] {panel.scenario.title}: 俯仰 {_pitch_deg(panel.data):+.2f}°，base z {panel.data.qpos[2] * 1000:.0f} mm，"
            f"位移 {(panel.data.qpos[0] - panel.initial_x) * 1000:+.1f} mm，{tip}"
        )
    print(f"视频: {args.output}  帧数 {frames + 1}  画面 {2 * args.width}x{2 * args.height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
