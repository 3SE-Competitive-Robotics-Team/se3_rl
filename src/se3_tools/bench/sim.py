"""单个评测 episode：按 ONNX 契约建 runtime + MuJoCo adapter，注入扰动 plant / 动作噪声，提供测量量。

全部走 se3_runtime 的 `PolicyRuntime` / `PolicyControlLoop`，与 sim2x、真机共用观测与动作数学；
本模块只做评测侧的 adapter 包装，不改 runtime core。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import mujoco
import numpy as np
import onnxruntime as ort

from se3_runtime import PolicyControlLoop, PolicyRuntime
from se3_runtime_mujoco import MujocoPolicyAdapter

WHEEL_BODIES = ("l_wheel_Link", "r_wheel_Link")
WHEEL_RADIUS_M = 0.06
FALL_UP_COS = 0.5
"""机身 z 轴与世界 z 夹角超过 60°（R[2,2] < 0.5）记为摔倒，与以往评测脚本一致。"""
KNEE_SPRING_ACTUATORS = ("l_knee_gas_spring", "r_knee_gas_spring")


@dataclass(frozen=True)
class Plant:
    """评测用 plant 偏差，口径与训练 DR 一致（见 scratch 的 WideDR 扰动评测）。

    com：base_link 质心偏移 (m)；mass：base_link 加质量 (kg)；spring：膝气弹簧力倍数（前馈仍按契约）；
    kp / kd：runtime 执行器控制器增益倍数；friction：全部 geom 滑动摩擦系数；
    delay_steps：physics tick 计的动作延迟（None = 契约名义延迟）。
    """

    com: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mass: float = 0.0
    spring: float = 1.0
    kp: float = 1.0
    kd: float = 1.0
    friction: float | None = None
    delay_steps: int | None = None


NOMINAL_PLANT = Plant()


@dataclass
class EpisodeConfig:
    onnx: str
    scene: str | None = None
    solver_hz: float | None = None
    """None = 与契约 physics_dt 相同（训练口径）。"""
    plant: Plant = field(default_factory=Plant)
    action_noise_std: float = 0.0
    """叠在 ONNX 均值输出上的高斯噪声 σ（训练探索噪声口径，后期约 0.21）；0 = 确定性回放。"""
    seed: int = 0
    spawn_position: tuple[float, float, float] | None = None
    spawn_yaw_rad: float = 0.0


class _NoisySession:
    """包住 onnxruntime session，在动作输出上加高斯噪声；runtime 照常把加噪后的动作当 previous action。"""

    def __init__(self, session: ort.InferenceSession, std: float, seed: int) -> None:
        self._session = session
        self._std = float(std)
        self._rng = np.random.default_rng(seed)

    def run(self, output_names, feeds, *args, **kwargs):
        outputs = self._session.run(output_names, feeds, *args, **kwargs)
        actions = np.asarray(outputs[0])
        noisy = actions + self._rng.normal(0.0, self._std, size=actions.shape).astype(actions.dtype)
        return [noisy, *outputs[1:]]

    def __getattr__(self, name: str):
        return getattr(self._session, name)


class Episode:
    """一个可反复 reset 的评测 episode。"""

    def __init__(self, cfg: EpisodeConfig) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        plant = cfg.plant
        # 延迟偏差改在 adapter 侧按 physics tick 排队：各契约允许的 runtime 覆盖范围不同。
        self.runtime = PolicyRuntime.load(
            cfg.onnx,
            session_options=options,
            action_delay_random_seed=cfg.seed,
            action_delay_steps_override=0 if plant.delay_steps is not None else None,
        )
        if cfg.action_noise_std > 0.0:
            noisy = _NoisySession(self.runtime.bundle.session, cfg.action_noise_std, cfg.seed)
            object.__setattr__(self.runtime.bundle, "session", noisy)
        contract = self.runtime.contract
        solver_dt = None if cfg.solver_hz is None else 1.0 / float(cfg.solver_hz)
        self.adapter = MujocoPolicyAdapter(
            contract,
            artifact_path=cfg.onnx,
            scene_path=cfg.scene,
            solver_dt_s=solver_dt,
            spawn_position=cfg.spawn_position,
            spawn_yaw_rad=cfg.spawn_yaw_rad,
        )
        if plant.delay_steps is not None:
            self._install_delay(int(plant.delay_steps))
        self.loop = PolicyControlLoop(self.runtime, self.adapter)
        self.model = self.adapter.model
        self.data = self.adapter.data
        self._apply_plant(plant)
        self.policy_dt = float(contract.timing.policy_dt_s)
        self.base_id = self.model.body(contract.robot.base_link).id
        self.wheel_ids = tuple(self.model.body(name).id for name in WHEEL_BODIES)
        self.robot_root = int(self.model.body_rootid[self.base_id])
        self.step_count = 0

    # ---- 生命周期 ----

    def _install_delay(self, delay_steps: int) -> None:
        queue: deque = deque()
        apply = self.adapter.apply_decoded_action

        def delayed_apply(action):
            queue.append(action)
            while len(queue) > delay_steps + 1:
                queue.popleft()
            apply(queue[0])

        self.adapter.apply_decoded_action = delayed_apply  # type: ignore[method-assign]
        self._delay_queue = queue

    def _apply_plant(self, plant: Plant) -> None:
        m = self.model
        base = m.body("base_link").id
        m.body_ipos[base] += np.asarray(plant.com, dtype=np.float64)
        m.body_mass[base] += float(plant.mass)
        if plant.spring != 1.0:
            for name in KNEE_SPRING_ACTUATORS:
                actuator = m.actuator(name).id
                m.actuator_biasprm[actuator, 0] *= plant.spring
                m.actuator_forcerange[actuator, 1] = max(
                    m.actuator_forcerange[actuator, 1], m.actuator_biasprm[actuator, 0]
                )
        if plant.friction is not None:
            m.geom_friction[:, 0] = float(plant.friction)
        controller = self.adapter._controller
        if plant.kp != 1.0:
            controller._stiffness = controller._stiffness * float(plant.kp)
        if plant.kd != 1.0:
            controller._damping = controller._damping * float(plant.kd)
        mujoco.mj_setConst(m, self.data)

    def reset(self) -> None:
        if hasattr(self, "_delay_queue"):
            self._delay_queue.clear()
        self.loop.reset()
        self.step_count = 0

    def set_command(
        self,
        *,
        vx: float | None = None,
        yaw: float | None = None,
        height: float | None = None,
    ) -> None:
        for name, value in (("lin_vel_x", vx), ("ang_vel_yaw", yaw), ("height", height)):
            if value is not None:
                self.adapter.set_command_field("velocity_height", name, float(value))

    def step(self) -> None:
        self.loop.policy_step()
        self.step_count += 1

    @property
    def time_s(self) -> float:
        return self.step_count * self.policy_dt

    # ---- 测量 ----

    def rotation(self) -> np.ndarray:
        return self.data.xmat[self.base_id].reshape(3, 3).copy()

    def base_position(self) -> np.ndarray:
        return self.data.xpos[self.base_id].copy()

    def body_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """机身系 (线速度, 角速度)。"""
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, self.base_id, vel, 1
        )
        return vel[3:].copy(), vel[:3].copy()

    def up_cos(self) -> float:
        return float(self.data.xmat[self.base_id][8])

    def fell(self) -> bool:
        return (not np.isfinite(self.data.qpos).all()) or self.up_cos() < FALL_UP_COS

    def pitch_deg(self) -> float:
        rot = self.rotation()
        return float(np.degrees(np.arcsin(np.clip(-rot[2, 0], -1.0, 1.0))))

    def heading_xy(self) -> np.ndarray:
        rot = self.rotation()
        h = rot[:2, 0]
        return h / (np.linalg.norm(h) + 1e-9)

    def measured_height(self) -> float:
        return float(self.adapter.measured_height()[0])

    def ground_z_below(self, point: np.ndarray) -> float:
        """point 正下方第一个非机器人表面的 z；没命中返回 nan。"""
        origin = np.asarray(point, dtype=np.float64).copy()
        direction = np.array([0.0, 0.0, -1.0])
        geom_id = np.zeros(1, dtype=np.int32)
        for _ in range(64):
            distance = mujoco.mj_ray(self.model, self.data, origin, direction, None, 1, -1, geom_id)
            if distance < 0 or distance > 5.0:
                return math.nan
            hit_z = origin[2] - distance
            body = self.model.geom_bodyid[geom_id[0]]
            if self.model.body_rootid[body] != self.robot_root:
                return float(hit_z)
            origin[2] = hit_z - 1e-6
        return math.nan

    def wheel_clearances(self) -> tuple[float, float]:
        """左右轮底到正下方地面的距离（m）。"""
        out = []
        for wid in self.wheel_ids:
            center = self.data.xpos[wid]
            ground = self.ground_z_below(center + np.array([0.0, 0.0, 0.01]))
            out.append(
                float(center[2] - WHEEL_RADIUS_M - ground) if np.isfinite(ground) else math.nan
            )
        return out[0], out[1]

    def wheel_positions(self) -> np.ndarray:
        return np.stack([self.data.xpos[w] for w in self.wheel_ids]).copy()


def terrain_profile(
    episode: Episode,
    origin_xy: np.ndarray,
    heading_xy: np.ndarray,
    length_m: float = 4.5,
    resolution_m: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """沿航向从 origin 起每 resolution 取一次地面高度，返回 (前进距离, 地面 z)。机器人自身命中会跳过。"""
    distances = np.arange(0.0, length_m, resolution_m)
    heights = np.empty_like(distances)
    top = float(episode.base_position()[2]) + 1.5
    for i, s in enumerate(distances):
        point = np.array([*(origin_xy + s * heading_xy), top])
        heights[i] = episode.ground_z_below(point)
    return distances, heights


def find_risers(
    distances: np.ndarray, heights: np.ndarray, min_step_m: float = 0.015
) -> list[tuple[float, float]]:
    """地面剖面里的台阶立面：返回 [(立面位置, 高度变化)]，上行为正、下行为负。"""
    risers: list[tuple[float, float]] = []
    dz = np.diff(heights)
    i = 0
    while i < len(dz):
        if np.isfinite(dz[i]) and abs(dz[i]) >= min_step_m:
            j = i
            total = dz[i]
            while j + 1 < len(dz) and np.isfinite(dz[j + 1]) and abs(dz[j + 1]) >= min_step_m:
                j += 1
                total += dz[j]
            risers.append((float(distances[i + 1]), float(total)))
            i = j + 1
        else:
            i += 1
    return risers


__all__ = [
    "FALL_UP_COS",
    "NOMINAL_PLANT",
    "Episode",
    "EpisodeConfig",
    "Plant",
    "find_risers",
    "terrain_profile",
]
