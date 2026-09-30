# M53：M51 + 电机侧膝气弹簧前馈补偿 + 腿部 T-N 包络 ×0.8（2026-09-30，用户定）

## 背景

M52（训练 plant 去掉 300 N 膝气弹簧）的 model_1000 在 sim2x 回放里明显好于同期带弹簧的策略，用户判断此前效果差的主因是气弹簧：
它在默认站姿每根主动杆等效约 13.6 N·m（DM8009P 额定 20 N·m 的 68%），是随姿态变化的类前馈机械力，纯 PD（kp 60 稳态偏差
约 13°，kp 20 约 39°）cover 不掉。

M52 本身不能部署（真机有弹簧），「M52 权重 + 部署端补偿 300 N」在 sim2x 静站里仍有约 6° 腿关节残差，一个结构性来源是限幅位置：
M52 训练时 T-N 限幅作用在 clip(PD)，部署时作用在 clip(PD + FF)，默认姿态下 FF ≈ 13.6 N·m 让 PD 两侧余量变成约 6 / 34 N·m。
M53 在训练端就走「PD + 前馈」，限幅口径与真机一致。

## 改动一：前馈补偿开关

`env_cfg(knee_gas_spring_compensation=True)` → 动作项 `knee_gas_spring_compensation_enabled=True`：

| 项 | M51 | M53 |
|---|---|---|
| plant 膝气弹簧（MJCF 恒力 tendon actuator） | 300 N | 300 N（不变） |
| 弹簧力 DR `knee_spring_force`（左右独立） | ×0.9–1.1（270–330 N） | 不变 |
| critic 特权观测 `knee_gas_spring_force` | 有 | 不变 |
| 电机侧前馈 −F·dL/dα | 关 | 开，F 按额定 300 N |
| ONNX 契约 `knee_gas_spring` | `{300, false}` | `{300, true}` |

前馈由 `se3_shared.fourbar.knee_gas_spring_compensation_torque_torch` 按测得的腿关节角（含编码器偏置）每个物理步计算，
在 PD 之后、T-N 限幅之前叠加；sim2x（`se3_runtime/_serialleg_v1.py` 独立复刻）与真机按契约做同一前馈。
DR 下额定之外的残差 ±30 N（约 ±1.4 N·m，kp 60 下约 1.3°）由策略与 PD 消化。

## 改动二：腿部 T-N 包络按物理含义取参并 ×0.8

M53 是两个变量叠加（用户定），与 M51 对比时无法单独归因。

mjlab DcMotor（照 IsaacLab DCMotor）限矩 min(stall·(1−ω/ω₀), effort_limit)。旧口径 stall 填峰值 40、effort_limit 填额定 20，
恒扭矩区被额定削平：100 rpm（10.47 rad/s）下只有 15 N·m，连额定工作点 20 N·m@100 rpm 都覆盖不到；达妙 2025 选型手册 p18
「24 V 定速 100 rpm 扫负载」曲线显示该转速下至少 25 N·m（测到 25 为止），旧口径与官方数据矛盾。真机电机为 DM-J8009P V1.0、24 V，
用户确认真机至少 30、35 肯定可给。

新口径同一公式、按物理含义取参后整体 ×0.8（`env_cfg(leg_torque_envelope_scale=0.8)`）：

| 参数 | 旧口径 | 物理口径 | M53（×0.8） |
|---|---|---|---|
| saturation_effort（零速截距） | 40（峰值） | 132.4（电压限 N·Kt·V/R，`MotorSpec.voltage_limited_stall_torque`） | 105.9 |
| effort_limit（平台） | 20（额定） | 40（峰值） | 32 |
| 转折速度 | 8.38 rad/s | 11.69 rad/s | 11.69 rad/s |
| 100 rpm 可用力矩 | 15 | 40 | 32 |
| 最大机械功率 | 168 W | 468 W | 374 W |
| 默认站姿抵掉弹簧 13.6 N·m 后余量 | 约 6 | 约 26 | 约 18 |

空载速度 16.76 rad/s 不变；轮子（M3508）不改。两值随 actuator cfg 写进 ONNX（`saturation_effort`、`effort_limit`，
`robot_config_overridden=true`），sim2x runtime 与真机按同一包络限矩。对比图 `scripts/plot_tn_envelope_proposal.py`。
Kt 由峰值 40 N·m / 50 A 估计，转折速度 25 N·m 以上无官方数据，是推算值。

## 物理步长

训练 200 Hz（sim_dt 5 ms、decimation 4）不变。前馈只依赖关节角、不含速度项，与 PD 同一步计算、同一步延迟；
其等效刚度 |dτ_ff/dq| 在默认姿态约 14 N·m/rad（默认 ±0.6 rad 范围内最大 17），约为 kp 60 的 1/4，且与 plant 弹簧自身的
几何刚度符号相反、额定力下基本抵消（DR 端点净约 ±1.4 N·m/rad）。PD 本身 kp 60 / 转子 armature 0.01 时 ω·dt ≈ 0.39，
kd 3 的显式阻尼稳定界约 6.7 ms（只计转子惯量），5 ms 步长有余量。

## 验证

- 配置核验（`.scratch/m53_check_cfg.py`）：M53 保留两个弹簧 actuator、DR ×0.9–1.1、critic 弹簧观测，动作项弹簧力 300、补偿开；
  腿部 saturation_effort 105.93、effort_limit 32、转折 11.69 rad/s，kp/kd 60/3.0 不变；地形仍为 M51 的十列。M51 / M52 不变。
- `tests.test_rough_port` + `tests.test_onnx_metadata` 70 项通过。
- M53 入口 CPU smoke（1 env、5 轮）通过，导出 ONNX 的 `knee_gas_spring` 为 `{force: 300.0, compensation_enabled: true}`，
  腿部 `saturation_effort` 105.93、`effort_limit` 32（轮子不变）；`PolicyBundle.load` 解析通过。

## 待核验

- 前馈几何（`se3_shared.fourbar` 的 dL/dα）与 MJCF 弹簧 tendon 实际几何的一致性，用户决定先不做、直接训练。
- sim2x launch 配置普遍用 `--physics-hz 500`，训练是 200 Hz；评测 M53 时建议补一遍 200 Hz。

## 启动记录

（未启动）
