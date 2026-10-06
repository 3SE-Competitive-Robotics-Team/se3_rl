# yaw 包络顶端拒转：诊断与轮子 T-N 包络 / 指令预算（2026-10-06，用户定）

## 现象

actor 知道 vx 的 run（Vel `fpuf21dr`、Vx `qnd9o87r`、OracleVel `x0xavu14`、VxObserver `agqj496q`）训练端 yaw 误差停在 1.6–2.0，
基线 HeightSigma07（`jrvpex39`）后期降到 0.96、只给 DR 参数的 Oracle 0.93、NoRest 约 1.0、OldDR 约 0.8。

## 诊断（脚本与结果 `.scratch/yaw_diag/`，全部在 whtws 上跑）

1. 传递曲线（sim2x 原生 MuJoCo，按差速预算的比例取分布内指令，4 模型 × 4 vx 档 × 10 档 × 2 seed）：
   VxObserver 在 |yaw| ≤ 0.75 × 上限时跟得上，≥ 0.9 × 上限拒转（vx = 0、+12 只到 0.01 倍，轮动作接近 0，不是饱和），各 vx 档一样；
   Base / NoRest / OldDR 全包络跟得上。不是向心加速度约束（vx = 0 也拒），不是 vx 估计错（真 vx ≈ 0，Vel / Vx 用真值同样退化）。
2. checkpoint 趋势（300–4000）：顶端从没学会，1000–1500 短暂学到一点后来回振荡。
3. 账本（训练环境、平面、DR1 startup 随机化、每指令 48 env、去台阶专项）：(0, +12) 转满 11.1/s 对 VxObserver 不转 10.3/s；
   (0, −12) 转满 9.3–9.5/s 对 VxObserver 半转 10.3/s。转满多拿 tracking_ang_vel 2.0–2.7/s，被 tracking_lin_vel（vx 指令 < 0.2 用
   σ_stand 0.1，转时 vx 漂 0.08–0.19 m/s）−0.5…−2.2、flat_wheel_contact −0.4…−0.9、高度罚 −0.1…−0.35 抵掉。
   看不到 vx 的策略不转时 vx 跟踪本来就只有 5.1/6，转起来几乎不额外亏；VxObserver 不转时 5.7–5.9/6，转起来的漂移是净损失。
4. 顶端为什么没有余量修漂移：轮子 T-N 包络是"峰值 3.32 N·m 线性降到空载 68.5 rad/s、额定 2.21 截顶"，22.9 rad/s 起就低于额定，
   45 rad/s 只剩 1.14 N·m；而 M3508 手册额定点是 469 rpm（19:1）仍出 3 N·m，即 14:1 下 66.65 rad/s 仍有 2.21 N·m。
   另外指令预算用半轮距 0.20 m，MJCF 实测 0.2166 m，原地 12 rad/s 实际要 43.3 rad/s（45 的 96%，超出 0.9 预算）。

## 改动

- `se3_shared.motor`：录入手册额定转速（M3508 19:1 469 rpm → 14:1 66.65 rad/s），`rated_point_stall_torque` 为过额定点与空载点的
  反电动势线截距（约 82 N·m，不依赖 Kt / 相电阻估计）。
- `robot_cfg.get_serialleg_closedchain_cfg(wheel_torque_envelope=...)`："linear_peak"（默认，旧口径）| "rated_point"
  （saturation_effort = 82、effort_limit = 额定 2.21：额定平台延续到 66.65 rad/s、末端降到空载；低速力矩不变）。
  随 actuator cfg 写进 ONNX metadata，sim2x 按同一包络限矩。`serialleg_wheel_half_track()` 从 MJCF 量半轮距。
- rough `env_cfg(wheel_torque_envelope=..., command_wheel_budget=...)`：预算 "legacy"（45 rad/s、0.20 m）| "rated"
  （66.65 rad/s、0.2166 m，比例仍 0.9）。rated 下 yaw 上限：vx = 0 → 16.6（被课程截到 12），vx = 1 → 12.0，vx = 1.5 → 9.7，vx = 2.4 → 5.5。
- 临时入口（叠在 VxObserver 上）：`SE3-WheelLegged-Rough-Exp-VxObserver-WheelTN`（只换包络，对照 VxObserver）、
  `SE3-WheelLegged-Rough-Exp-VxObserver-WheelTN-Budget`（再换预算，对照 WheelTN）。默认配置不变。

## 判据

WheelTN：sim2x 传递曲线顶端（≥ 0.9 × 上限）是否跟得上、训练端 yaw 误差是否降到基线水平；平地 / 台阶指标不差于 VxObserver。
WheelTN-Budget：新增的 vx–yaw 组合（如 vx = 1、yaw 7–12）能否跟上，其余指标不退化。
注意 "rated_point" 改的是 plant：真机若 C620 / 电池电压达不到手册额定点，部署端应以实测 T-N 为准。
