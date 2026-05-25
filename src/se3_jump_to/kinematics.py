"""SerialLeg 正向运动学（FK）工具。

提供从关节角到轮子位置的映射，用于 TO 的运动学约束。

SerialLeg 腿部结构（单侧）：
    base_link
        └── lf0_Joint（髋）→ lf0_Link
                └── [四连杆传动]
                        └── lf1_Joint（膝，等效）→ lf1_Link
                                └── l_wheel_Link（轮子）

简化建模：
    - 忽略四连杆的精确几何，用等效单链近似
    - 轮子中心相对 base_link 的位置由 (lf0, lf1) 决定
    - 从 MJCF 测量的连杆长度参数

实测参数（来自 MuJoCo FK，default_dof_pos=[0.617, 0.207]）：
    base_link z = 0.301 m
    l_wheel z   = 0.101 m
    腿部展开长度 ≈ 0.200 m（base_link 到轮子中心）

    左腿：x_offset ≈ -0.0（左右对称），y_offset ≈ +0.131 m
    右腿：x_offset ≈ -0.0,             y_offset ≈ -0.131 m
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

_MJCF_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "robots"
    / "serialleg"
    / "mjcf"
    / "serialleg_fidelity_cylinder_wheels.xml"
)


class SerialLegFK:
    """SerialLeg 正向运动学。

    用于 TO 的运动学约束：给定关节角，计算轮子位置。
    使用 MuJoCo 作为精确 FK 计算后端。
    """

    # 受控关节名（6 DOF）：lf0, lf1, l_wheel, rf0, rf1, r_wheel
    CTRL_JOINT_NAMES = (
        "lf0_Joint",
        "lf1_Joint",
        "l_wheel_Joint",
        "rf0_Joint",
        "rf1_Joint",
        "r_wheel_Joint",
    )

    def __init__(self) -> None:
        self._model = mujoco.MjModel.from_xml_path(str(_MJCF_PATH))
        self._data = mujoco.MjData(self._model)

        # body id
        self._base_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self._lw_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_Link")
        self._rw_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_Link")

        # 受控关节在 qpos 中的索引
        self._ctrl_qpos_idx: list[int] = []
        for name in self.CTRL_JOINT_NAMES:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._ctrl_qpos_idx.append(self._model.jnt_qposadr[jid])

        # 基本物理参数
        self.total_mass: float = float(self._model.body_mass.sum())
        self.base_mass: float = float(self._model.body_mass[self._base_id])
        self.g: float = float(-self._model.opt.gravity[2])  # 9.81

        # 默认关节角和站立高度：从 se3_shared 读取，保持与训练端一致
        from se3_shared import RobotConfig as _SharedRobotCfg

        _shared = _SharedRobotCfg()
        self.default_qpos = np.array(_shared.default_dof_pos)
        self.default_base_height: float = _shared.default_base_height

        # 轮子半径：用 default_qpos + freejoint 设在默认站立高度，FK 正算 lw_z
        self._set_ctrl_qpos(self.default_qpos)
        self._data.qpos[0:3] = [0, 0, self.default_base_height]
        self._data.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(self._model, self._data)
        self.wheel_radius: float = float(self._data.xpos[self._lw_id, 2])

        # 两轮相对 base_link 的 y 偏移（左正右负）
        lw_y = float(self._data.xpos[self._lw_id, 1])
        base_y = float(self._data.xpos[self._base_id, 1])
        self.wheel_y_offset: float = abs(lw_y - base_y)  # ≈ 0.131 m

    def _set_ctrl_qpos(self, q6: np.ndarray) -> None:
        """设置受控关节角（6维），其余 qpos 保持默认。"""
        mujoco.mj_resetData(self._model, self._data)
        for i, idx in enumerate(self._ctrl_qpos_idx):
            self._data.qpos[idx] = q6[i]

    def fk(self, base_pos: np.ndarray, q6: np.ndarray) -> dict[str, np.ndarray]:
        """精确 FK：给定 base 位置和关节角，返回轮子世界坐标。

        Args:
            base_pos: base_link 世界坐标 [x, y, z]
            q6: 受控关节角 [lf0, lf1, lw, rf0, rf1, rw]

        Returns:
            {"l_wheel": [x,y,z], "r_wheel": [x,y,z], "com": [x,y,z]}
        """
        self._set_ctrl_qpos(q6)
        # 设置 base_link 的 root joint（freejoint）qpos
        self._data.qpos[0:3] = base_pos
        self._data.qpos[3:7] = [1, 0, 0, 0]  # 单位四元数（直立）
        mujoco.mj_forward(self._model, self._data)

        return {
            "l_wheel": self._data.xpos[self._lw_id].copy(),
            "r_wheel": self._data.xpos[self._rw_id].copy(),
            "com": self._data.subtree_com[0].copy(),
        }

    def leg_length(self, q_hip: float, q_knee: float) -> float:
        """近似腿长：base_link 到轮子中心的垂直距离。"""
        q6 = np.array([q_hip, q_knee, 0.0, q_hip, q_knee, 0.0])
        self._set_ctrl_qpos(q6)
        self._data.qpos[0:3] = [0, 0, 0.5]
        self._data.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(self._model, self._data)
        base_z = float(self._data.xpos[self._base_id, 2])
        lw_z = float(self._data.xpos[self._lw_id, 2])
        return base_z - lw_z

    def ik_grounded(
        self,
        base_z: float,
        wheel_x_fixed: float,
        q_init: np.ndarray,
    ) -> np.ndarray:
        """二维 IK：约束轮子 x 不动 + 轮子 z 接地。

        问题1修复：单参数 IK 只约束腿长（z方向），蹲下时髋关节角变化导致
        轮子在 x 方向滑动。二维 IK 同时约束 wheel_x = wheel_x_fixed，
        确保轮子在地面原地不动。

        Args:
            base_z: 目标 base_link z 高度
            wheel_x_fixed: 轮子 x 坐标（保持不变，由初始姿态确定）
            q_init: 初始关节角猜测（[hip, knee]）

        Returns:
            q6 = [hip, knee, 0, hip, knee, 0]
        """
        from scipy.optimize import minimize

        target_wz = self.wheel_radius  # 轮子中心离地 = 轮子半径

        def obj(q: np.ndarray) -> float:
            result = self.fk([0, 0, base_z], np.array([q[0], q[1], 0.0, q[0], q[1], 0.0]))
            wx = result["l_wheel"][0]
            wz = result["l_wheel"][2]
            # 同时约束 x 不滑 + z 接地，加平滑项
            err_x = (wx - wheel_x_fixed) ** 2
            err_z = (wz - target_wz) ** 2
            smooth = 0.001 * ((q[0] - q_init[0]) ** 2 + (q[1] - q_init[1]) ** 2)
            return err_x * 200 + err_z * 200 + smooth

        res = minimize(
            obj,
            x0=[float(q_init[0]), float(q_init[1])],
            bounds=[(-1.5, 1.5), (-0.55, 0.75)],
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8},
        )
        qh, qk = float(res.x[0]), float(res.x[1])
        return np.array([qh, qk, 0.0, qh, qk, 0.0])

    def optimal_tuck_pose(self, dx_tol: float = 0.005) -> np.ndarray:
        """计算最优空中收腿姿态：让轮子最靠近 base 且保持在 base 正下方。

        约束：
        - 轮子在 base 正下方（|x_wheel - x_base| < dx_tol，默认 5mm）
        - 关节在物理范围内（hip ∈ [-1.5, 1.5], knee ∈ [-0.55, 0.75]）

        目标：最小化腿长 = base_z - wheel_z

        为什么不允许轮子前伸：
        - 轮子前伸虽然能让腿长更短（极限 0.056m），但破坏中轴对称
        - 落地时机身前倾、轮子滑移、姿态不可控
        - 物理正确的收腿应该把轮子收到 base 正下方（垂直收回）

        Returns:
            q6 = [hip, knee, 0, hip, knee, 0]
        """
        from scipy.optimize import minimize

        def leg_length(q: np.ndarray) -> float:
            q6 = np.array([q[0], q[1], 0.0, q[0], q[1], 0.0])
            self._set_ctrl_qpos(q6)
            self._data.qpos[0:3] = [0, 0, 0.5]
            self._data.qpos[3:7] = [1, 0, 0, 0]
            mujoco.mj_forward(self._model, self._data)
            base_z = float(self._data.xpos[self._base_id, 2])
            lw_z = float(self._data.xpos[self._lw_id, 2])
            return base_z - lw_z

        def x_offset(q: np.ndarray) -> float:
            q6 = np.array([q[0], q[1], 0.0, q[0], q[1], 0.0])
            self._set_ctrl_qpos(q6)
            self._data.qpos[0:3] = [0, 0, 0.5]
            self._data.qpos[3:7] = [1, 0, 0, 0]
            mujoco.mj_forward(self._model, self._data)
            base_x = float(self._data.xpos[self._base_id, 0])
            lw_x = float(self._data.xpos[self._lw_id, 0])
            return lw_x - base_x

        # 多起点搜索，避免局部最优
        best = None
        for x0 in [(0.3, 0.7), (0.5, 0.5), (0.8, 0.4), (1.0, 0.2), (0.6, 0.6)]:
            res = minimize(
                leg_length,
                x0=x0,
                bounds=[(-1.5, 1.5), (-0.55, 0.75)],
                constraints={"type": "ineq", "fun": lambda q: dx_tol - abs(x_offset(q))},
                method="SLSQP",
                options={"maxiter": 200},
            )
            if res.success and abs(x_offset(res.x)) < dx_tol:
                leg = float(leg_length(res.x))
                if best is None or leg < best[1]:
                    best = (res.x.copy(), leg)

        if best is None:
            # 兜底：用 home 姿态
            return np.array([0.617, 0.207, 0.0, 0.617, 0.207, 0.0])

        qh, qk = float(best[0][0]), float(best[0][1])
        return np.array([qh, qk, 0.0, qh, qk, 0.0])

    def optimal_extend_pose(self, dx_tol: float = 0.005) -> np.ndarray:
        """计算最优"腿伸最长"姿态：让轮子离 base 最远且保持在 base 正下方。

        用途：起跳脱离地面瞬间的姿态。物理意义：
        - 站立段电机做完最大功（腿从短伸到最长）
        - 膝关节弹簧能量完全释放给机身
        - 起跳点 base_z 抬到最高，给后续抛物线最高的初始位置

        约束：
        - 轮子在 base 正下方（|x_wheel - x_base| < dx_tol，默认 5mm）
        - 关节在物理范围内

        目标：最大化腿长 = base_z - wheel_z

        实测结果：hip=1.039, knee=-0.55，腿长 0.333m，dx≈0
        起跳点 base_z = 0.333 + 0.059 = 0.392m

        Returns:
            q6 = [hip, knee, 0, hip, knee, 0]
        """
        from scipy.optimize import minimize

        def leg_length(q: np.ndarray) -> float:
            q6 = np.array([q[0], q[1], 0.0, q[0], q[1], 0.0])
            self._set_ctrl_qpos(q6)
            self._data.qpos[0:3] = [0, 0, 0.5]
            self._data.qpos[3:7] = [1, 0, 0, 0]
            mujoco.mj_forward(self._model, self._data)
            base_z = float(self._data.xpos[self._base_id, 2])
            lw_z = float(self._data.xpos[self._lw_id, 2])
            return base_z - lw_z

        def x_offset(q: np.ndarray) -> float:
            q6 = np.array([q[0], q[1], 0.0, q[0], q[1], 0.0])
            self._set_ctrl_qpos(q6)
            self._data.qpos[0:3] = [0, 0, 0.5]
            self._data.qpos[3:7] = [1, 0, 0, 0]
            mujoco.mj_forward(self._model, self._data)
            base_x = float(self._data.xpos[self._base_id, 0])
            lw_x = float(self._data.xpos[self._lw_id, 0])
            return lw_x - base_x

        # 多起点搜索（最大化腿长，目标取负）
        best = None
        for x0 in [(1.0, -0.3), (0.8, -0.5), (1.2, -0.55), (0.6, -0.4), (1.039, -0.55)]:
            res = minimize(
                lambda q: -leg_length(q),
                x0=x0,
                bounds=[(-1.5, 1.5), (-0.55, 0.75)],
                constraints={"type": "ineq", "fun": lambda q: dx_tol - abs(x_offset(q))},
                method="SLSQP",
                options={"maxiter": 200},
            )
            if res.success and abs(x_offset(res.x)) < dx_tol:
                leg = float(leg_length(res.x))
                if best is None or leg > best[1]:
                    best = (res.x.copy(), leg)

        if best is None:
            return np.array([1.039, -0.55, 0.0, 1.039, -0.55, 0.0])

        qh, qk = float(best[0][0]), float(best[0][1])
        return np.array([qh, qk, 0.0, qh, qk, 0.0])


# 单例，避免重复加载 MJCF
_fk_instance: SerialLegFK | None = None


def get_fk() -> SerialLegFK:
    """获取 SerialLegFK 单例。"""
    global _fk_instance
    if _fk_instance is None:
        _fk_instance = SerialLegFK()
    return _fk_instance
