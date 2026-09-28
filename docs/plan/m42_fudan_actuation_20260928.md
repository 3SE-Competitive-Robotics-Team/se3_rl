# M42：执行链对齐复旦量级 + 动作罚 −0.05 / −0.05（2026-09-28，用户定）

## 背景

M39 → M40 → M41 一路减动作罚，σ 一路涨（0.41 → 0.47 → 1.1+），M41（action_rate −0.01、删 smoothness）在 700 轮 σ 1.14、
动作幅值 5.4、回报回落，1991 轮按用户指令停止。对照复旦 `fudan_rl_wheel_leg` 上台阶3 的配置发现 PPO 超参一致
（entropy_coef 0.01、lr 1e-3、adaptive KL），差别在执行链与动作罚：

| 项 | 复旦 上台阶3 | 本仓库 M38 默认 | M42 |
|---|---|---|---|
| 腿 action_scale | 0.5 rad | 0.25 | **0.5** |
| 腿 kp / kd | 15 / 1.0 | 60 / 3.0 | **20 / 1.5** |
| 轮 action_scale | 10 rad/s | 15 | **10** |
| 轮 kd | 0.1 | 0.2 | 0.2 |
| action_rate / action_smooth | −0.05 / −0.05（100 Hz） | −0.48 / −0.12 | **−0.05 / −0.05** |
| 单位动作力矩：腿 kp×scale | 7.5 N·m | 15 | 10 |
| 单位动作力矩：轮 kd×scale | 1.0 N·m | 3.0 | 2.0 |

同样的 σ 在本仓库默认执行链上物理抖动是复旦的 2–3 倍，这是 σ 0.4 就"看着很高"的原因之一。
动作罚对探索噪声的定价 k = 2·w_rate + 6·w_smooth 与 dt 无关，−0.05/−0.05 给 k = 0.4，与复旦等价；对平滑运动它随 dt² 与 scale²，
scale 翻倍后同一物理动作的归一化差分减半、平方 1/4，所以 M42 对真实动作的罚比 M38 轻得多。

## 改动（相对 M38 默认：六处，用户定的整组重定价）

- `robot_cfg.get_serialleg_closedchain_cfg` 新增 `leg_kp_override` / `leg_kd_override`；Flat `env_cfg` 新增 `leg_action_scale`；
  Rough `env_cfg` 新增 `leg_action_scale` / `wheel_action_scale` / `leg_kp` / `leg_kd`（默认 None = 原值）。
- 入口 `SE3-WheelLegged-Rough-Exp-FudanActuation`：腿 scale 0.5、轮 10、腿 kp 20 / kd 1.5、action_rate −0.05、action_smoothness −0.05。
- 契约：增益与尺度随 actuator cfg / action term 写进 ONNX metadata（KP/KD/scale），sim2x 的 `policy_actuator` 与真机按 metadata 执行，
  runtime 不改；metadata 的 `robot_config_overridden` 会标 True。

## 风险

- kp 20 对本机身（更重、300 N 气弹簧）是否撑得住：smoke 与首轮看 catastrophic / bad_orientation。
- 轮 kd 0.2 × scale 10：满幅 2 N·m，接近 M3508 额定 2.2；库仑死区 0.15 N·m 对应 0.75 rad/s 不变。
- 部署：真机 DM8009 MIT 模式 kp/kd 由 metadata 下发，20/1.5 是新值，实机验证前不要直接上。

## run 设置

`nulltask-5c45cdd89b-whtws` 七卡 × 8192、5000 轮、保存 200、seed 42，在线 W&B project `SE3-WheelLegged-Rough`。

## 判据

- σ：`Policy/mean_std` 稳在 0.5 以内；`Locomotion/max_abs_action`。
- 静站：`jitter_check.py` 俯仰 rms、腿峰峰、轮速 std 对 M39-4999；注意 scale 变了，归一化动作差 rms 不可直接比。
- 台阶：课程等级、台阶列 vx 对 M39（6.8 / 0.81）；riser_retry_diag、climb_map 对 M39-4999（卡 8 / 摔 15）。
- 终止：catastrophic / bad_orientation 尾部。

## 启动记录

代码 commit `f4f515c`，whtws 仓库由 8ea3c9e 快进到同一 commit；本地入口 CPU smoke 5 轮通过（灾难终止 0）；
smoke 导出的 ONNX metadata 核验：`robot/KP = [20,20,20,20,0,0]`、`robot/KD = [1.5,1.5,1.5,1.5,0.2,0.2]`、`policy_io/action/scale = [0.5×4, 10, 10]`、
`robot_config_overridden = True`。启动器 `--dry-run` 通过。用户指定本条只用两张卡。

- 启动时间：2026-09-28 23:18（Pod 时区），GPU 0–1，两卡 × 8192、5000 轮、保存 200、seed 42，在线 W&B。
- run：`2026-09-28_15-18-44_rough-M42-fudanactuation-seed42-2x8192-5k`，
  W&B [ir84bch8](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/ir84bch8)，PID/PGID `2296319`，
  state `/workspace/.se3-training-state/whtws/20260928T151837Z`。
- 首轮核验：第 33 轮在迭代、无 Traceback、无 nefc overflow，3.23 s/轮，两卡 11.0–11.4 GB、利用率 83–84%，其余五卡空闲；
  远端 `params/env.yaml`：腿 stiffness 20 / damping 1.5、`wheel_scale 10.0`、action_rate −0.05、action_smoothness −0.05。
- 批量是 M38–M41 七卡的 2/7，按同 iteration 对比带批量混杂。
