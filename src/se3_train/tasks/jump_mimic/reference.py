"""跳跃 mimic 参考库：加载 se3.jump_ref.v1 参考，按（参考编号, 参考时刻）在 GPU 上取帧。

所有参考第 0 帧是同一个站姿（stand_height），未触发跳跃的 env 一律取第 0 帧，等于"站着"的参考；
时刻超过参考时长时取最后一帧（同样回到站姿）。腿长两条腿相同（参考左右对称），存成 [K, T, 2]。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REFERENCE_DIR = Path(__file__).resolve().parents[4] / "assets" / "trajectories" / "jump_ref_v1"
DEFAULT_REFERENCE_PATHS: tuple[str, ...] = tuple(
    str(REFERENCE_DIR / f"jump_{h:.2f}m.npz") for h in (0.20, 0.30, 0.40)
)
REFERENCE_FORMAT = "se3.jump_ref.v1"


@dataclass
class ReferenceFrame:
    """一批 env 的参考帧。"""

    leg_len: torch.Tensor  # [N, 2]
    base_z: torch.Tensor  # [N]
    base_vz: torch.Tensor  # [N]
    contact: torch.Tensor  # [N]，1 = 两轮接地
    leg_pos: torch.Tensor  # [N, 4]，policy 主动杆角
    leg_vel: torch.Tensor  # [N, 4]


class JumpReferenceLibrary:
    """把多条参考 pad 到同一长度后整体放到 device 上，按索引 gather。"""

    def __init__(self, paths: tuple[str, ...], device: torch.device | str) -> None:
        if not paths:
            raise ValueError("跳跃参考列表不能为空")
        refs = []
        for path in paths:
            data = np.load(path)
            meta = json.loads(str(data["meta"]))
            if meta.get("format") != REFERENCE_FORMAT:
                raise ValueError(
                    f"{path} 不是 {REFERENCE_FORMAT} 参考（format={meta.get('format')}）"
                )
            refs.append((data, meta))
        dts = {round(float(d["dt"]), 9) for d, _ in refs}
        if len(dts) != 1:
            raise ValueError(f"参考 dt 不一致：{sorted(dts)}")
        self.dt = float(next(iter(dts)))
        lengths = [len(d["phase"]) for d, _ in refs]
        t_max = max(lengths)

        def pad(arr: np.ndarray) -> np.ndarray:
            if len(arr) == t_max:
                return arr
            tail = np.repeat(arr[-1:], t_max - len(arr), axis=0)
            return np.concatenate([arr, tail], axis=0)

        def stack(key: str, fn=lambda a: a) -> torch.Tensor:
            return torch.as_tensor(
                np.stack([pad(fn(np.asarray(d[key], dtype=np.float64))) for d, _ in refs]),
                dtype=torch.float32,
                device=device,
            )

        self.leg_len = stack("leg_len", lambda a: np.stack([a, a], axis=-1))
        self.base_z = stack("base_pos", lambda a: a[:, 2])
        self.base_vz = stack("base_vel", lambda a: a[:, 2])
        self.contact = stack("contact")
        self.leg_pos = stack("leg_pos")
        self.leg_vel = stack("leg_vel")
        self.length = torch.as_tensor(lengths, dtype=torch.long, device=device)
        self.duration = self.length.float() * self.dt
        self.target_clearance = torch.as_tensor(
            [m["target_clearance"] for _, m in refs], dtype=torch.float32, device=device
        )
        stand = {round(float(m["stand_height"]), 6) for _, m in refs}
        if len(stand) != 1:
            raise ValueError(f"参考站姿高度不一致：{sorted(stand)}")
        self.stand_height = float(next(iter(stand)))
        self.num_refs = len(refs)

    def frame(self, ref_id: torch.Tensor, t: torch.Tensor) -> ReferenceFrame:
        """ref_id [N] long、t [N] 秒 → 参考帧；t 落在 [0, 时长) 外时夹到首/末帧。"""
        idx = torch.round(t / self.dt).long()
        idx = torch.clamp(idx, min=0)
        idx = torch.minimum(idx, self.length[ref_id] - 1)
        return ReferenceFrame(
            leg_len=self.leg_len[ref_id, idx],
            base_z=self.base_z[ref_id, idx],
            base_vz=self.base_vz[ref_id, idx],
            contact=self.contact[ref_id, idx],
            leg_pos=self.leg_pos[ref_id, idx],
            leg_vel=self.leg_vel[ref_id, idx],
        )


__all__ = ["DEFAULT_REFERENCE_PATHS", "JumpReferenceLibrary", "ReferenceFrame"]
