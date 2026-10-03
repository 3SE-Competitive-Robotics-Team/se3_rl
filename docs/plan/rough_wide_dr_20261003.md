# Rough 加宽域随机化对照（Exp-WideDR，2026-10-03，用户定）

## 起因：DR 审计

实测脚本 `.scratch/dr_audit/verify_dr.py`（CPU，Rough 默认入口，8 env），结论：

- **摩擦**：`randomize_friction` 写该 env 的全部 geom（地形也在内），接触有效摩擦就等于采样值，0.2–1.5 全程生效。
- **pd_gains**：DcMotor 的 PD 在 torch 里算（腿 kp 60 / kd 3、轮 kd 0.2，各 env 全同），MuJoCo 侧是 `<motor>`（gain=1、无 bias）。
  该事件改 gainprm[0]，实际是 6 个电机输出力矩统一 ×0.9–1.1（乘在 T-N 限幅之后，再被 forcerange 截断）；
  改 biasprm[2] 是乘 0，所以 kd DR 不起作用，critic 回读的 kd 通道恒为 1。
- **restitution**：空函数，只采样不写入。
- 与复旦 stairs_v3（同尺寸轮腿）相比，推力、机身质量、质心、动作延迟都更窄。

## 改动（只在 `SE3-WheelLegged-Rough-Exp-WideDR`，默认入口不变）

| 项 | 默认 | WideDR |
|---|---|---|
| Kp / Kd | 输出力矩 ×0.9–1.1，kd 无效 | torch 侧真 Kp、Kd 各 ×0.9–1.1（每 env 一个系数，6 电机共用）；力矩缩放去掉 |
| 质心偏移 | ±5 mm | ±5 cm |
| base 附加质量 | −0.5…+1.5 kg | −1…+3 kg |
| 恢复系数 | 无 | 0–1 |
| 气弹簧力 | ×0.9–1.1 | ×0.9–1.5（前馈补偿仍按额定 300 N） |
| 动作延迟 | 4–6 ms（取整后恒为 1 步 = 5 ms） | 0–10 ms（0/1/2 步各 1/3），play 与 ONNX 契约同步 |

其余（摩擦、惯量、电机被动参数、默认关节位置、推力课程、观测噪声）不动。这一组是多项一起改，不是单变量对照。

### 实现要点

- `events.randomize_pd_gains_torch`：调 mjlab `IdealPdActuator.set_gains`，与官方 `dr.pd_gains` 对这类 actuator 走同一条路。
- `events.randomize_contact_restitution`：MuJoCo 没有恢复系数参数。默认 solref 的阻尼比同时改刚度（k ∝ 1/dampratio²），
  所以换成直接式 solref (−k, −b)：刚度固定为基线，只改阻尼。基线是轮-地形混合 solref (0.015, 1.25)（实测，
  `.scratch/dr_audit/contact_baseline.py`），k = 1/(tc²·dr²) = 2844。恢复系数到阻尼比的映射用 mjwarp 落球实测表反查
  （`.scratch/dr_audit/restitution_scan.py`，5 ms 步长、1.0 m/s 撞击；0.5 / 2.0 m/s 下偏差在 ±0.04 内）。
  ζ > 2 时 b·dt > 1，显式积分反冲，所以可达下限约 0.03（采样低于此值按 0.03 处理）；基线本身约 0.05。
- critic 回读 `dr_model_params_obs[8:10]`：kp = MuJoCo 力矩缩放 × torch stiffness 比值，kd = torch damping 比值。
  默认入口与旧公式逐位相同（`.scratch/dr_audit/verify_wide_dr.py` 检查通过）。恢复系数没有进 critic（维度不变）。

## 验证

`.scratch/dr_audit/verify_wide_dr.py`（CPU，16 env）：
- torch 侧 stiffness / damping 比值落在 0.91–1.10；MuJoCo 电机 gainprm 全为 1，actuator_force/ctrl = 1。
- 质心 ±4.5 cm、附加质量 −0.9…+2.9 kg、气弹簧 271–434 N、延迟步数 {0, 1, 2} 都有样本。
- 轮-地形接触的 solref 逐 env 等于该 env 写入的 (−k, −b)。
- smoke：见启动记录。

## 风险与要盯的

- 质心 ±5 cm 曾在 7lxhzb64 学出原地摆腿探测质心（见 `flat/env_cfg.py` com 注释）；回放时看 zero_hold 前杆摆动。
- 气弹簧上尾 +150 N 残差没有前馈；看低姿 / 静站时腿部力矩是否顶包络。
- 延迟 0–10 ms 会进 ONNX 契约，sim2x 每次 reset 也按 0–10 ms 采样。

## 启动记录

（待定）
