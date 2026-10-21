"""起身 mimic 参考库：加载 se3.recovery_ref.v1（se3_tools.recovery_reference 生成），按（参考编号, 时刻）在 GPU 上取帧。

参考第 0 帧是倒地起点；时刻超过参考时长时取最后一帧（蹲坐保持）。启动时校验参考的 MJCF 哈希与当前训练 MJCF 一致。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from se3_train.robot_cfg import _MJCF_PATH

TRAJECTORY_ROOT = Path(__file__).resolve().parents[4] / "assets" / "trajectories"
REFERENCE_DIR = TRAJECTORY_ROOT / "recovery_ref_v1"
REFERENCE_FORMAT = "se3.recovery_ref.v1"


@dataclass
class RecoveryReferenceFrame:
    """一批 env 的参考帧。"""

    projected_gravity: torch.Tensor  # [N, 3]
    base_z: torch.Tensor  # [N]
    base_ang_vel_b: torch.Tensor  # [N, 3]
    leg_pos: torch.Tensor  # [N, 4]，policy 主动杆语义
    leg_vel: torch.Tensor  # [N, 4]
    wheel_vel: torch.Tensor  # [N, 2]


class RecoveryReferenceLibrary:
    """多条参考 pad 到同一长度后放到 device 上，按索引 gather。"""

    def __init__(self, paths: tuple[str, ...], device: torch.device | str) -> None:
        if not paths:
            raise ValueError("起身参考列表不能为空")
        mjcf_sha = hashlib.sha256(_MJCF_PATH.read_bytes()).hexdigest()
        refs = []
        for path in paths:
            data = np.load(path)
            meta = json.loads(str(data["meta"]))
            if meta.get("format") != REFERENCE_FORMAT:
                raise ValueError(
                    f"{path} 不是 {REFERENCE_FORMAT} 参考（format={meta.get('format')}）"
                )
            if meta.get("mjcf_sha256") != mjcf_sha:
                raise ValueError(
                    f"{path} 的 MJCF 哈希与当前 {_MJCF_PATH.name} 不一致，模型改过，需用 se3_tools.recovery_reference 重新生成"
                )
            refs.append((data, meta))
        dts = {round(float(d["dt"]), 9) for d, _ in refs}
        if len(dts) != 1:
            raise ValueError(f"参考 dt 不一致：{sorted(dts)}")
        self.dt = float(next(iter(dts)))
        lengths = [len(d["segment"]) for d, _ in refs]
        t_max = max(lengths)

        def stack(key: str) -> torch.Tensor:
            arrays = []
            for d, _ in refs:
                a = np.asarray(d[key], dtype=np.float64)
                if len(a) < t_max:
                    a = np.concatenate([a, np.repeat(a[-1:], t_max - len(a), axis=0)], axis=0)
                arrays.append(a)
            return torch.as_tensor(np.stack(arrays), dtype=torch.float32, device=device)

        self.projected_gravity = stack("projected_gravity")
        self.base_z = stack("base_pos")[..., 2]
        self.base_ang_vel_b = stack("base_ang_vel_b")
        self.leg_pos = stack("leg_pos")
        self.leg_vel = stack("leg_vel")
        self.wheel_vel = stack("wheel_vel")
        self.length = torch.as_tensor(lengths, dtype=torch.long, device=device)
        self.duration = self.length.float() * self.dt
        self.families = tuple(str(m["family"]) for _, m in refs)
        self.num_refs = len(refs)

    def frame(self, ref_id: torch.Tensor, t: torch.Tensor) -> RecoveryReferenceFrame:
        """ref_id [N] long、t [N] 秒 → 参考帧；t 越过参考时长时夹到末帧。"""
        idx = torch.clamp(torch.round(t / self.dt).long(), min=0)
        idx = torch.minimum(idx, self.length[ref_id] - 1)
        return RecoveryReferenceFrame(
            projected_gravity=self.projected_gravity[ref_id, idx],
            base_z=self.base_z[ref_id, idx],
            base_ang_vel_b=self.base_ang_vel_b[ref_id, idx],
            leg_pos=self.leg_pos[ref_id, idx],
            leg_vel=self.leg_vel[ref_id, idx],
            wheel_vel=self.wheel_vel[ref_id, idx],
        )


__all__ = ["REFERENCE_DIR", "RecoveryReferenceFrame", "RecoveryReferenceLibrary"]
