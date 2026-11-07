"""SerialLeg 世界模型的状态 s、上下文 ξ 与积分参数化。

改写自 SuperMimic-Reproduced（Yuxi Liu，MIT，见 LICENSES/SuperMimic-Reproduced.txt）的
`world_model/state_spec.py`、`envs/state_io.py`、`envs/sim_context.py`，G1 专用部分按 SerialLeg 重写。

状态 s（41 维，env 原点为平移零点）：
    [qpos(17) | qvel(16) | prev_action(6) | wheel_lag(2)]
    qpos = 根位置 3、根四元数 wxyz 4、10 个铰链（含两个轮角）；qvel = 根线速度（世界系）3、根角速度（机身系）3、
    10 个铰链速度（MuJoCo free joint 约定）。
    prev_action：上一拍执行动作（已按 action_clip 裁剪）。物理步 5 ms × decimation 4、动作延迟 0–2 个物理步，
    每个控制步前几个子步执行的仍是上一拍动作（SerialLegDelayedAction._action_fifo），不带它 s 不是马尔可夫的。
    wheel_lag：轮执行器一阶滞后的输出状态（LaggedDcMotorActuator._lagged，N·m），同理。

规范化特征（网络输入，航向不变，航向取绕 z 轴的 twist 分量，见 heading_quat）：根高 1、去航向姿态 rot6d 6、航向系线速度 3、机身角速度 3、
    8 个非轮铰链 sin/cos 16（前杆可整周转，用 sin/cos 而不是原角）、铰链速度 10、prev_action 6、wheel_lag 2。
    轮角连续累计、物理上无关，不进特征。

网络输出 delta（34 维）与积分（dt = 控制周期 0.02 s）：
    dv_h(3) dw_b(3) dqd(10) dp_h(3) dth_b(3) dq(10) dlag(2)
    v' = v + R_h dv_h    w' = w + dw_b    qd' = qd + dqd    lag' = lag + dlag
    p' = p + dt v' + R_h dp_h    quat' = quat ⊗ exp(dt w' + dth_b)    q' = q + dt qd' + dq
    prev_action' = 本步执行动作（直接复制，不预测）
  输出全 0 时模型按匀速外推（物理先验）；动力学 MLP 最后一层零初始化，从这个先验开始学。

误差向量（损失与报告用，32 维，按名义误差归一化）：
    [dp 3, 姿态 rotvec 3, dv 3, dw 3, 非轮铰链角（取最短差）8, 非轮铰链速度 8, 轮速 2, wheel_lag 2]
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import torch
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    matrix_from_quat,
    normalize,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
)

from se3_shared import JointGroup
from se3_train.actuators import LaggedDcMotorActuator
from se3_train.mdp.joint_indices import policy_leg_joint_ids, tensor_ids
from se3_train.mdp.observations import dr_model_params_obs, knee_gas_spring_force_obs

NQ = 17
NV = 16
ACTION_DIM = 6
LAG_DIM = 2
STATE_DIM = NQ + NV + ACTION_DIM + LAG_DIM
FEAT_DIM = 1 + 6 + 3 + 3 + 16 + 10 + ACTION_DIM + LAG_DIM
DELTA_DIM = 3 + 3 + 10 + 3 + 3 + 10 + LAG_DIM
ERROR_DIM = 3 + 3 + 3 + 3 + 8 + 8 + 2 + LAG_DIM
XI_DIM = 28 + 2 + 4 + 4 + 1 + 2

# 状态切片
S_POS = slice(0, 3)
S_QUAT = slice(3, 7)
S_HINGE = slice(7, 17)
S_LIN = slice(17, 20)
S_ANG = slice(20, 23)
S_HINGE_VEL = slice(23, 33)
S_ACT = slice(33, 39)
S_LAG = slice(39, 41)


def _rep(values: tuple[float, ...], sizes: tuple[int, ...]) -> torch.Tensor:
    return torch.cat([torch.full((n,), float(v)) for v, n in zip(values, sizes, strict=True)])


@dataclass(frozen=True)
class HingeLayout:
    """10 个铰链在 qpos[7:17] / qvel[6:16] 内的排列：哪些是轮、哪些是 policy 腿关节。"""

    wheel: tuple[int, ...]
    """两个轮铰链在 10 个铰链里的下标（左、右）。"""
    leg: tuple[int, ...]
    """8 个非轮铰链的下标。"""
    policy_leg: tuple[int, ...]
    """4 个 policy 腿关节（LF0、LB、RF0、RB）在 10 个铰链里的下标。"""

    @classmethod
    def from_model(
        cls, mj_model: mujoco.MjModel, robot_joint_names: tuple[str, ...]
    ) -> HingeLayout:
        def hinge_index(name: str) -> int:
            for jid in range(mj_model.njnt):
                if mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid) == name:
                    adr = int(mj_model.jnt_qposadr[jid])
                    if not 7 <= adr < NQ:
                        raise ValueError(f"关节 {name} 的 qpos 地址 {adr} 不在铰链段")
                    return adr - 7
            # mjlab 给实体关节名加了前缀时按后缀匹配
            for jid in range(mj_model.njnt):
                full = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
                if full.endswith("/" + name):
                    return int(mj_model.jnt_qposadr[jid]) - 7
            raise KeyError(f"模型里找不到关节 {name}（实体关节：{robot_joint_names}）")

        wheel = tuple(hinge_index(n) for n in JointGroup.WHEEL_NAMES)
        policy_leg = tuple(hinge_index(n) for n in JointGroup.POLICY_LEG_NAMES)
        leg = tuple(i for i in range(10) if i not in wheel)
        return cls(wheel=wheel, leg=leg, policy_leg=policy_leg)


class StateSpec:
    """规范化特征、积分与误差（全部纯 torch、可微）。"""

    def __init__(self, layout: HingeLayout, dt: float, device: torch.device | str):
        self.layout = layout
        self.dt = float(dt)
        self.device = torch.device(device)
        self.wheel_idx = torch.tensor(layout.wheel, device=self.device)
        self.leg_idx = torch.tensor(layout.leg, device=self.device)
        self.policy_leg_idx = torch.tensor(layout.policy_leg, device=self.device)
        hinge_vel_scale = torch.ones(10)
        hinge_vel_scale[list(layout.leg)] = 5.0
        hinge_vel_scale[list(layout.wheel)] = 30.0
        # 特征缩放（除）：[z, rot6d, v_h, w_b, sin/cos, 铰链速度, prev_action, wheel_lag]
        self.feat_scale = torch.cat(
            [
                _rep((1.0, 1.0, 1.0, 2.0, 1.0), (1, 6, 3, 3, 16)),
                hinge_vel_scale,
                _rep((1.0, 3.0), (ACTION_DIM, LAG_DIM)),
            ]
        ).to(self.device)
        # 输出缩放（乘）：[dv_h, dw_b, dqd(10), dp_h, dth_b, dq(10), dlag]
        dqd_scale = torch.full((10,), 2.0)
        dqd_scale[list(layout.wheel)] = 5.0
        self.out_scale = torch.cat(
            [
                _rep((0.3, 0.5), (3, 3)),
                dqd_scale,
                _rep((0.01, 0.02, 0.02, 0.5), (3, 3, 10, LAG_DIM)),
            ]
        ).to(self.device)
        # 名义误差：[dp, 姿态, dv, dw, 腿角, 腿速, 轮速, wheel_lag]
        self.err_scale = _rep(
            (0.05, 0.1, 0.5, 1.0, 0.1, 2.0, 5.0, 0.5), (3, 3, 3, 3, 8, 8, 2, LAG_DIM)
        ).to(self.device)

    # ------------------------------------------------------------------ 特征

    def canonical(self, s: torch.Tensor) -> torch.Tensor:
        """航向不变、已缩放的网络输入特征 (…, FEAT_DIM)。"""
        quat = normalize(s[..., S_QUAT])
        heading = heading_quat(quat)
        local = quat_mul(quat_conjugate(heading), quat)
        rot6d = matrix_from_quat(local)[..., :2].reshape(*quat.shape[:-1], 6)
        v_h = quat_apply_inverse(heading, s[..., S_LIN])
        q_leg = s[..., S_HINGE][..., self.leg_idx]
        feat = torch.cat(
            [
                s[..., 2:3],
                rot6d,
                v_h,
                s[..., S_ANG],
                torch.sin(q_leg),
                torch.cos(q_leg),
                s[..., S_HINGE_VEL],
                s[..., S_ACT],
                s[..., S_LAG],
            ],
            dim=-1,
        )
        return feat / self.feat_scale

    # ------------------------------------------------------------------ 积分

    def integrate(self, s: torch.Tensor, delta: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """把网络输出 delta (…, DELTA_DIM) 积到状态 s 上；action 是本步执行动作（已裁剪）。"""
        d = delta * self.out_scale
        dv_h, dw_b, dqd = d[..., 0:3], d[..., 3:6], d[..., 6:16]
        dp_h, dth_b, dq, dlag = d[..., 16:19], d[..., 19:22], d[..., 22:32], d[..., 32:34]
        quat = normalize(s[..., S_QUAT])
        heading = heading_quat(quat)
        v_new = s[..., S_LIN] + quat_apply(heading, dv_h)
        w_new = s[..., S_ANG] + dw_b
        qd_new = s[..., S_HINGE_VEL] + dqd
        p_new = s[..., S_POS] + self.dt * v_new + quat_apply(heading, dp_h)
        rotvec = self.dt * w_new + dth_b
        quat_new = normalize(quat_mul(quat, _quat_from_rotvec(rotvec)))
        q_new = s[..., S_HINGE] + self.dt * qd_new + dq
        lag_new = s[..., S_LAG] + dlag
        return torch.cat([p_new, quat_new, q_new, v_new, w_new, qd_new, action, lag_new], dim=-1)

    # ------------------------------------------------------------------ 误差

    def error_vector(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """归一化误差 (…, ERROR_DIM)。"""
        q1 = normalize(s_pred[..., S_QUAT])
        q2 = normalize(s_true[..., S_QUAT])
        rot = axis_angle_from_quat(quat_mul(quat_conjugate(q2), q1))
        dq = s_pred[..., S_HINGE] - s_true[..., S_HINGE]
        dq_leg = torch.atan2(torch.sin(dq), torch.cos(dq))[..., self.leg_idx]
        dqd = s_pred[..., S_HINGE_VEL] - s_true[..., S_HINGE_VEL]
        err = torch.cat(
            [
                s_pred[..., S_POS] - s_true[..., S_POS],
                rot,
                s_pred[..., S_LIN] - s_true[..., S_LIN],
                s_pred[..., S_ANG] - s_true[..., S_ANG],
                dq_leg,
                dqd[..., self.leg_idx],
                dqd[..., self.wheel_idx],
                s_pred[..., S_LAG] - s_true[..., S_LAG],
            ],
            dim=-1,
        )
        return err / self.err_scale

    def normalized_sq_error(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """各维归一化平方误差的均值（报告用的「预测误差」）。"""
        e = self.error_vector(s_pred, s_true)
        return (e * e).mean(-1)

    # ------------------------------------------------------------------ 派生量

    @staticmethod
    def projected_gravity(s: torch.Tensor) -> torch.Tensor:
        quat = normalize(s[..., S_QUAT])
        g = torch.zeros_like(quat[..., :3])
        g[..., 2] = -1.0
        return quat_apply_inverse(quat, g)

    @staticmethod
    def lin_vel_b(s: torch.Tensor) -> torch.Tensor:
        return quat_apply_inverse(normalize(s[..., S_QUAT]), s[..., S_LIN])

    def policy_leg_pos(self, s: torch.Tensor) -> torch.Tensor:
        return s[..., S_HINGE][..., self.policy_leg_idx]

    def policy_leg_vel(self, s: torch.Tensor) -> torch.Tensor:
        return s[..., S_HINGE_VEL][..., self.policy_leg_idx]

    def wheel_vel(self, s: torch.Tensor) -> torch.Tensor:
        return s[..., S_HINGE_VEL][..., self.wheel_idx]


def heading_quat(quat: torch.Tensor) -> torch.Tensor:
    """quat 绕世界 z 轴的 twist 分量（swing-twist 分解）：normalize(w, 0, 0, z)。

    不用 mjlab 的 yaw_quat：它按机身 x 轴的水平投影取航向，俯仰 ±90° 时 1/|x_xy| 奇异，随机动作下首轮世界模型
    梯度范数到 1e8（2026-11-06 GPU 核对）。twist 分解只在绕水平轴转 180° 附近奇异（此时早已触发终止）；
    机身水平时两者相同。
    """
    w, z = quat[..., 0:1], quat[..., 3:4]
    norm = torch.sqrt(w.square() + z.square() + 1.0e-8)
    zeros = torch.zeros_like(w)
    return torch.cat([w / norm, zeros, zeros, z / norm], dim=-1)


def _quat_from_rotvec(rotvec: torch.Tensor) -> torch.Tensor:
    """旋转向量 → 四元数，角度 0 附近数值稳定（零向量给单位四元数）。"""
    # 零向量处 norm 的反传是 0/0；加 1e-12 再开方，角度偏差 ≤ 1e-6 rad
    angle = torch.sqrt(rotvec.square().sum(-1) + 1.0e-12)
    safe = torch.where(angle > 1.0e-8, angle, torch.ones_like(angle))
    axis = rotvec / safe[..., None]
    quat = quat_from_angle_axis(angle, axis)
    # 极小角时 axis 方向无意义，用一阶近似 [1, v/2]
    small = torch.cat([torch.ones_like(angle[..., None]), 0.5 * rotvec], dim=-1)
    return torch.where((angle > 1.0e-8)[..., None], quat, normalize(small))


class SimStateReader:
    """从 live env 读 s 与 ξ（采集端用；不需要可微）。"""

    def __init__(self, env, action_term_name: str = "delayed_action"):
        self.env = env
        self.sim = env.sim
        self.robot = env.scene["robot"]
        self.action_term = env.action_manager.get_term(action_term_name)
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        if qpos.shape[-1] != NQ or qvel.shape[-1] != NV:
            raise ValueError(
                f"SuperMimic 假定场景里只有 SerialLeg 一个带自由度的实体（nq={NQ}、nv={NV}），"
                f"实际 nq={qpos.shape[-1]}、nv={qvel.shape[-1]}"
            )
        self.origins = env.scene.env_origins.clone()
        self.leg_ids = tensor_ids(policy_leg_joint_ids(self.robot), device=env.device)
        self.lag_actuator = None
        for actuator in getattr(self.robot, "_actuators", []):
            if isinstance(actuator, LaggedDcMotorActuator):
                self.lag_actuator = actuator
        clip = getattr(self.action_term.cfg, "action_clip", None)
        self.action_clip = None if clip is None else float(clip)

    def clip_action(self, action: torch.Tensor) -> torch.Tensor:
        """与 SerialLegDelayedAction.process_actions 相同的裁剪。"""
        if self.action_clip is None:
            return action
        return torch.clamp(action, -self.action_clip, self.action_clip)

    def wheel_lag(self) -> torch.Tensor:
        n = self.env.num_envs
        if self.lag_actuator is None or self.lag_actuator._lagged is None:
            return torch.zeros(n, LAG_DIM, device=self.env.device)
        return self.lag_actuator._lagged[:, :LAG_DIM].clone()

    def read_state(self) -> torch.Tensor:
        """(N, STATE_DIM)：当前 qpos / qvel + 执行链隐状态。"""
        s = torch.cat(
            [
                self.sim.data.qpos[:],
                self.sim.data.qvel[:],
                self.action_term.raw_action,
                self.wheel_lag(),
            ],
            dim=-1,
        ).clone()
        s[:, 0:3] -= self.origins
        return s

    def read_xi(self) -> torch.Tensor:
        """(N, XI_DIM)：DR 回读 28 + 气弹簧力 2 + 腿编码器偏置 4 + 腿默认站姿 4 + 延迟步数 1 + 轮滞后 τ 2。"""
        env = self.env
        n = env.num_envs
        lag_tau = (
            self.lag_actuator.lag_tau[:, :LAG_DIM]
            if self.lag_actuator is not None and self.lag_actuator.lag_tau is not None
            else torch.zeros(n, LAG_DIM, device=env.device)
        )
        xi = torch.cat(
            [
                dr_model_params_obs(env),
                knee_gas_spring_force_obs(env),
                self.robot.data.encoder_bias[:, self.leg_ids],
                self.robot.data.default_joint_pos[:, self.leg_ids],
                self.action_term.delay_steps.float()[:, None],
                lag_tau,
            ],
            dim=-1,
        )
        if xi.shape != (n, XI_DIM):
            raise RuntimeError(f"ξ 维度 {tuple(xi.shape)} 与约定 {(n, XI_DIM)} 不符")
        return xi.clone()


# ξ 内各段的切片（解析头要用其中的默认站姿）
XI_DEFAULT_LEG = slice(28 + 2 + 4, 28 + 2 + 4 + 4)


__all__ = [
    "ACTION_DIM",
    "DELTA_DIM",
    "ERROR_DIM",
    "FEAT_DIM",
    "LAG_DIM",
    "NQ",
    "NV",
    "STATE_DIM",
    "S_ACT",
    "S_ANG",
    "S_HINGE",
    "S_HINGE_VEL",
    "S_LAG",
    "S_LIN",
    "S_POS",
    "S_QUAT",
    "XI_DEFAULT_LEG",
    "XI_DIM",
    "HingeLayout",
    "SimStateReader",
    "StateSpec",
    "heading_quat",
]
