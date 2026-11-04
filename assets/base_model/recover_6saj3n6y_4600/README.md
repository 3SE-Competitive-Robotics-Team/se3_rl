# recover 蒸馏老师 6saj3n6y model_4600（2026-11-04）

| 项 | 值 |
|---|---|
| 用途 | `SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-JuneTouchdown-Distill` 的 recover 组在线蒸馏老师（`rl_cfg.ROUGH_RECOVER_JUNE_DISTILL_TEACHER_CKPT`） |
| 训练入口 | `SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-Recover-June-WheelAir-Touchdown` |
| 训练 commit | `873c017e`（代码同 `3e1faedb`） |
| W&B | 离线 run `6saj3n6y`（nulltask1 六卡 × 2048，平地，seed 42；用户要求停在 4600 轮） |
| checkpoint | `model_4600.pt`（sha256 前缀 `02009f80e6b3c218`，不进 git，换机器要拷） |
| ONNX | `model_4600.onnx`（sha256 前缀 `ad71e30aa0e41b4c`） |

起身奖励：6 月上车版（`env_cfg.ROUGH_RECOVER_JUNE_*`）+ 轮子悬空转速罚（ω²，−2e-3）+ 轮子触地速度罚
（max(0, v_落 − 0.75 m/s)²，−2）。网络与 Student128 相同（CTS 学生路径，学生编码器 128）。

model_4300 确定性回放（对照 sooaiehz-4300，倒置出生，峰值中位）：单轮接触力 244 vs 470 N、峰值那次触地速度 0.72 vs 3.86 m/s、
轮速 38 vs 69 rad/s；但机身接触力 719 vs 499 N、腿杆 251 vs 0 N（冲击部分转到机身 / 腿杆）；10 s 起身率 0.99 持平，
翻正中位 1.26 vs 0.68 s。侧躺单轮 298 vs 382 N、机身 313 vs 448 N，起身 0.96 vs 0.99。
