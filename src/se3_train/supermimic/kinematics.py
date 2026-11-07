"""SerialLeg 的可微正运动学：从 MjModel 读出运动树，按铰链 qpos 算各 body 在机身系里的位姿。

世界模型预测的是 qpos / qvel，奖励里有几项要用 body 位置（如左右轮心前后错位），需要一条可反传的
FK。MJCF 是闭链（四连杆靠 equality 约束闭合），但 MuJoCo 的运动树本身是一棵树，按树做 FK 即可；
闭链约束由世界模型在 qpos 上学出来，FK 只负责几何。

只支持 free 根 + hinge 关节（SerialLeg 的全部关节类型），遇到其它类型直接报错。
"""

from __future__ import annotations

import mujoco
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_from_angle_axis, quat_mul


class TreeKinematics:
    """按 body 拓扑顺序（MuJoCo 保证父在子前）做批量 FK，输出机身系位置与姿态。"""

    def __init__(self, mj_model: mujoco.MjModel, body_ids: list[int], device: torch.device | str):
        body_ids = [int(b) for b in body_ids]
        if not body_ids:
            raise ValueError("机器人 body 列表为空")
        root = body_ids[0]
        root_joints = [j for j in range(mj_model.njnt) if int(mj_model.jnt_bodyid[j]) == root]
        if len(root_joints) != 1 or mj_model.jnt_type[root_joints[0]] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("机器人根 body 必须恰好挂一个 free joint")
        self.root_body = root
        self.body_ids = body_ids
        self._local = {b: i for i, b in enumerate(body_ids)}
        self.parents: list[int] = []
        self.joints: list[list[tuple[int, torch.Tensor, torch.Tensor]]] = []
        pos, quat = [], []
        for b in body_ids:
            parent = int(mj_model.body_parentid[b])
            self.parents.append(self._local.get(parent, -1) if b != root else -1)
            pos.append(torch.as_tensor(mj_model.body_pos[b], dtype=torch.float32))
            quat.append(torch.as_tensor(mj_model.body_quat[b], dtype=torch.float32))
            joints = []
            if b != root:
                for j in range(mj_model.njnt):
                    if int(mj_model.jnt_bodyid[j]) != b:
                        continue
                    if mj_model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                        raise ValueError(f"body {b} 的关节 {j} 不是 hinge，TreeKinematics 不支持")
                    joints.append(
                        (
                            int(mj_model.jnt_qposadr[j]),
                            torch.as_tensor(
                                mj_model.jnt_axis[j], dtype=torch.float32, device=device
                            ),
                            torch.as_tensor(
                                mj_model.jnt_pos[j], dtype=torch.float32, device=device
                            ),
                        )
                    )
            self.joints.append(joints)
        self.body_pos = torch.stack(pos).to(device)
        self.body_quat = torch.stack(quat).to(device)

    def body_index(self, mj_model: mujoco.MjModel, name: str) -> int:
        """按 body 名返回本 FK 内的局部下标。"""
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0 or bid not in self._local:
            raise KeyError(f"机器人里没有 body {name}")
        return self._local[bid]

    def forward(self, qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """qpos [..., nq]（全局 qpos 布局）→ 机身系 body 位置 [..., B, 3] 与姿态 [..., B, 4]（wxyz）。

        根 body 自身是单位位姿；free joint 的平移与姿态不参与（机身系）。
        """
        batch = qpos.shape[:-1]
        out_pos: list[torch.Tensor] = []
        out_quat: list[torch.Tensor] = []
        identity = torch.zeros(*batch, 4, dtype=qpos.dtype, device=qpos.device)
        identity[..., 0] = 1.0
        zeros = torch.zeros(*batch, 3, dtype=qpos.dtype, device=qpos.device)
        for i, parent in enumerate(self.parents):
            if parent < 0:
                out_pos.append(zeros)
                out_quat.append(identity)
                continue
            p_pos, p_quat = out_pos[parent], out_quat[parent]
            local_q = self.body_quat[i].expand(*batch, 4)
            pos = p_pos + quat_apply(p_quat, self.body_pos[i].expand(*batch, 3))
            quat = quat_mul(p_quat, local_q)
            for adr, axis, anchor in self.joints[i]:
                # 绕 body 系内过 anchor 的 axis 旋转 q：位置先补偿 anchor 的位移，再转姿态
                rot = quat_from_angle_axis(qpos[..., adr], axis.expand(*batch, 3))
                anchor_w = quat_apply(quat, anchor.expand(*batch, 3))
                quat_new = quat_mul(quat, rot)
                pos = pos + anchor_w - quat_apply(quat_new, anchor.expand(*batch, 3))
                quat = quat_new
            out_pos.append(pos)
            out_quat.append(quat)
        return torch.stack(out_pos, dim=-2), torch.stack(out_quat, dim=-2)


__all__ = ["TreeKinematics"]
