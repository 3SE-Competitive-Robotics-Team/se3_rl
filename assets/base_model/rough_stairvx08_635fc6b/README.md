# rough StairVx08 部署包（2026-10-09）

| 项 | 值 |
|---|---|
| 训练入口 | `SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08` |
| 训练 commit | `635fc6b7fea82da64beeba64855502fde239596d`（metadata `assets.dirty = false`） |
| W&B | `luzhongjin365-se3/SE3-WheelLegged-Rough/hvhgfeaq`（nulltask1 六卡 × 8192，5000 轮，seed 42） |
| 部署 ONNX | `rough_stairvx08_635fc6b_model_4999.onnx`（sha256 前缀 `5b087e32a15f5b53`，`se3.meta.v2`） |
| checkpoint | `model_4999.pt`（sha256 前缀 `7a96e0e28efdfa84`），重导出须在 635fc6b 的 worktree 里跑 `scripts/export_onnx_from_checkpoint.py` |

网络：CTS 学生路径 + 显式 vx。输入 480 维（30 维 × 16 帧历史，term-major），学生编码器 480→256→128→33
（L2 归一化隐向量 32 + v̂x 1），policy [最新帧 30, 隐向量 32, v̂x 1] = 63 → 128 → 64 → 32 → 6（ELU）。
无 GRU，无特权输入。结构与实验记录见 `docs/plan/rough_vx_observer_20261005.md`。

## 上机

```bash
uv run se3-sim2x-nx --onnx assets/base_model/rough_stairvx08_635fc6b/rough_stairvx08_635fc6b_model_4999.onnx --dev /dev/ttyACM0
```

`--action-delay-steps` 保持默认 0（训练的 0–10 ms 延迟模拟的就是真机通讯与执行延迟）。台架吊起可加
`--ignore-output-enabled`。使能 0→1 时 runtime 清零 16 帧历史与上一拍动作。

## 上机前必须和固件核对的契约（以 ONNX metadata 为准）

| 契约字段 | 值 | 固件侧 |
|---|---|---|
| 控制频率 | 50 Hz policy（sim_dt 0.005 × decimation 4） | TARGET 频率 |
| 腿 KP / KD | 60 / 3.0（LF0、LB、RF0、RB） | `NX_POLICY_JOINT_KP / KD` |
| 轮 KD | 0.2 | `WheelTorqueInt` 的 Kp |
| 腿力矩 | effort 32 N·m、saturation 105.9、velocity 16.76 rad/s | 固件当前 ±40 平限，不等价 |
| 轮力矩 | effort 2.21 N·m（rated_point）、saturation 81.96、velocity 68.5 rad/s | 轮力矩限幅 |
| 膝气弹簧 | `{force: 300, compensation_enabled: true}` | 弹簧必须装机；固件按每帧 TARGET 的开关做 −F·dL/dα 补偿 |
| 动作缩放 | 腿 0.25 rad + 默认姿态 offset，轮 15 rad/s | — |
| 默认姿态 | LF0 −0.1724、LB −1.4723、RF0 0.1724、RB 1.4723 | — |
| 指令包络 | vx ±2.4 m/s、yaw ±12 rad/s、高度 0.20–0.38 m、跳跃目标 0–0.5 m | `NX_COMM_CMD_HEIGHT_*` |

## 使用限制（仿真评测结论）

- 台阶前给 ≥ 0.8 m/s：训练台阶列指令下限是 0.8，0.4 m/s 上 12 cm 台阶只能爬到 0.24 m。
- 原地 / 低速大 yaw 转不满（vx = 0 打满上限只有 0.18）；0.9 倍上限可以转满；1.0 倍上限在 vx = 1.0 处约 0.46。
- 中高速轻微欠速：1.5 → 1.30 m/s。
- 本机无头 MuJoCo 静站 10 s：高度 0.219 m、倾角约 2°，有约 −2.5 cm/s 的后溜。
- 只在仿真里验证过（MJLab 训练 + 原生 MuJoCo sim2sim），单 seed，未上过真机。
