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

| 项 | 值 |
|---|---|
| 机器 | nulltask1，GPU 0–5 六卡 × 1365 env |
| commit | `80019b8`（worktree `/workspace/se3-worktrees/rough-widedr-80019b8`） |
| run | `rough-widedr-actor128-seed42-6x1365-5k`，W&B `txcesf5r`（project SE3-WheelLegged-Rough） |
| 配置 | 5000 轮、seed 42、从头训；除 DR 外与 Orient24（whtws `3ix86na3`，`aeec423`）同基线 |
| 启动 | 2026-10-04 02:37（Pod 时区），PID/PGID 794204，state `20261003T183704Z-22210` |
| 首轮 | 第 15 轮无 Traceback，1.28 s/轮，jump_env_rate 0.0996，六卡 4.6–5.1 GB / 72–83% |

对照基线：默认 rough。可比的现成 run 是 nulltask1 Actor128-A（`qtpyzxtf`，六卡 × 1365，`635be5f`），
但它早于去同步优化与旧 RSI bug 修复（`b14fbf4` / `c0c791f`），比较时要记住这一差异。

## 结果（2026-10-04，跑满 5000 轮）

对照：Orient24 `3ix86na3`（同代码基线、默认 DR，但姿态罚 −24、七卡 × 1170）；Actor128-A `qtpyzxtf`（默认 DR、姿态罚 −12）
用来确认跟踪退化与姿态罚无关。都不是严格单变量对照，缺一条同 commit 的默认 DR 6×1365。

### 训练端（4800–4999 轮均值，WideDR / Orient24 / Actor128-A）

| 指标 | WideDR | Orient24 | Actor128-A |
|---|---|---|---|
| mean_reward | 41.0 | 85.5 | 77.2 |
| stairs_up 等级 | 4.07 | 6.15 | 6.19 |
| 二级台阶门（stairs_up 均级 5） | 从未打开 | 1500 轮前打开 | — |
| 跳跃列等级 | 0.80 | 4.57 | — |
| mimic_deviation 终止 | 0.58 | 0.03 | — |
| catastrophic 终止 | 0.45–0.58 | ≈0.003 | 0.03 |
| yaw 误差 | 2.82 | 0.73 | 0.81 |
| 平地 tracking_lin_vel | 0.57 | 0.75 | 0.83 |

catastrophic 全部来自 `catastrophic_leg_pos`（腿偏离默认 > 3 rad，腿被折过去），基线为 0。

### 评测端（sim2x 确定性回放，延迟统一 1 步；脚本 `.scratch/widedr_eval/`）

- **名义 plant**：WideDR 基本不听指令。平地实速锁在约 1.8 m/s（指令 1.0 跑 1.75–1.96，指令 2.4 跑 1.5–1.8），
  h0.30 vx 1.0 误差 0.85–0.93（Orient24 0.14）；转向只跟到 29%（wz 1/2/4 实际 0.29/0.57/1.13）；跳跃最高 0.25 m（目标 0.5，
  Orient24 0.49）。抗推更好（h0.30 +1.5 m/s 最大倾角 15.6° 对 31.3°）。
- **地形评分表**：通过率 89% 对 100%（下台阶 r6/r9、波浪 r6/r9 失败），但摔倒 1 对 33（Orient24 多在冲出 4 m 进入相邻块时摔），
  最大倾角 21.6° 对 51.5°。台阶 1–9 级拆解：出 4 m 5.19 s 对 5.52 s，卡顿相当，0 摔。
- **扰动 plant**（`dr_slice.py`，11 种 plant × 6 项）：
  - Orient24 对质心最敏感：质心 x +5 cm 时站立漂 6.1 m、俯仰 +12°，vx 误差 0.14 → 0.56；组合最坏（质心 +5 cm、+3 kg、
    气弹簧 ×1.5、延迟 10 ms）6 项摔 5 项。延迟 0/10 ms、摩擦 0.3、Kp/Kd ×0.9 对两者几乎无影响。
  - WideDR 的各项指标在 11 种 plant 上几乎不变（vx1 误差约 0.85、yaw 误差约 1.4），组合最坏只摔 1 项。

### 结论

WideDR 用名义性能和指令跟随换来了对参数偏差的不敏感：最坏情况能活，但在任何 plant 上都跟不好指令，跳跃没学会，
上台阶等级卡在 4。与"单帧 MLP 只能学一个对所有参数都凑合的动作"的预测一致，但单凭这一组还分不清是信息不足
（需要在线辨识）、容量不足（actor 128/64/32），还是部分 DR 组合物理上不可行（catastrophic_leg_pos 只在 WideDR 出现）。
下一步的判别实验是 Oracle（actor 直接看 DR 参数）：能恢复到 Orient24 的跟踪水平就说明瓶颈是信息。

## Oracle 诊断（Exp-WideDR-Oracle，2026-10-04，用户定）

相对 WideDR 唯一差异：actor 观测末尾追加真实 DR 参数 32 维（28 维 DR 回读、气弹簧力 2 维、动作延迟、恢复系数），
critic 不变。commit `1fa00ca`，nulltask1 六卡 × 1365、5000 轮、seed 42，W&B `ft3ire24`，PID/PGID 796848，
state `20261004T053833Z-29957`。不可部署（runtime 不认这些观测项），判定看训练端同口径指标。

判据（对 WideDR `txcesf5r` 与 Orient24 `3ix86na3`，4800–4999 轮均值）：
- 信息是瓶颈：mean_reward、yaw 误差、平地 tracking、stairs_up 等级、跳跃列等级明显向 Orient24 靠拢，
  catastrophic_leg_pos 下降 → 做历史隐向量在线辨识（CTS + ROA）。
- 不是信息：与 WideDR 持平 → 先查 actor 容量（128/64/32）与 DR 组合的物理可行性（哪些组合触发腿折叠）。

### Oracle 结果（第 4392 轮时取 3900–4000 轮均值；Oracle / WideDR / Orient24）

| 指标 | Oracle | WideDR | Orient24 |
|---|---|---|---|
| mean_reward | 38.5 | 34.9 | 83.9 |
| stairs_up 等级 | 3.96 | 4.01 | 5.99 |
| 跳跃列等级 | 1.02 | 0.73 | 3.82 |
| mimic_deviation 终止 | 0.57 | 0.57 | 0.04 |
| yaw 误差 | 3.09 | 2.83 | 0.72 |
| 平地 tracking_lin_vel | 0.54 | 0.57 | 0.75 |
| catastrophic 终止 | 0.40 | 0.54 | 0.002 |
| Loss/value | 4.55 | 4.39 | 1.74 |

全程（900–4000 轮各窗口）与 WideDR 重合在噪声范围内。**actor 直接知道真实 DR 参数也恢复不了**，所以 WideDR 的退化
不是信息不足；任何历史 → 隐向量的在线辨识方案都以 Oracle 为上限，在这组 DR 下拿不回性能。
剩下的解释是 DR 本身改变了任务：部分参数组合在当前奖励与执行器包络下做不到或代价极高（质心 ±5 cm 按整机折算约
±4.2 cm，默认站姿配平倾角约 ±19°，会被姿态罚持续计价；气弹簧 ×1.5 的 +150 N 残差没有前馈），或回报方差变大
（value loss 是 Orient24 的 2.5 倍）拖慢学习。下一步按 DR 项归因。

## 离线归因（2026-10-04）

脚本 `.scratch/widedr_eval/attribution.py`（nulltask1 GPU 0，3.5 min）：WideDR 训练环境 4096 env × 2000 步，
model_4999 确定性策略，课程拨到第 5000 轮、地形等级 0–9 均匀铺开；逐 env 记录 DR 参数与表现，
`attr_analyze.py` 做 OLS（地形列 one-hot + 难度为协变量），数据 `attr_4096.npz`。各 DR 参数两两近似独立（|r| < 0.03）。

| 指标 | 第一嫌疑（低→高端效应，t） | 其次 |
|---|---|---|
| 每步回报（均值 0.190） | 恢复系数 −0.132（t −39） | 摩擦 +0.040、质心 x −0.025、质量 −0.015 |
| 非超时终止 | 恢复系数 +0.55/千步（t +19） | 质心 x、质心 z、质量（各 +0.1） |
| vx 误差 | 恢复系数 +0.25（t +28） | 摩擦 −0.11 |
| yaw 误差 | 恢复系数 +0.18（t +39） | 质心 x +0.04 |
| 静站俯仰 | 质心 x +12.2°（t +89，正常配平） | 恢复系数 +1.5° |
| 腿电机顶限比例 | Kp 缩放 +0.12（t +44） | 气弹簧 +0.03 |

恢复系数是悬崖式：0–0.6 回报 0.219–0.225 基本不变，0.7–0.8 掉到 0.180，0.8–0.9 0.143，0.9–1.0 只剩 0.053
（−76%），非超时终止翻倍多。气弹簧 ×1.5、延迟 0–10 ms、真 Kp/Kd 对回报和跟踪几乎无影响；质心 ±5 cm 主要表现为
配平倾角（策略在做对的事），对回报 −13%；低摩擦有影响但默认 DR 本来就有 0.2–1.5。确定性回放里没有腿折叠
（训练期的腿折叠应来自探索噪声叠加高恢复系数的弹跳）。

**根因是我实现的恢复系数 DR 与复旦（Isaac Gym / PhysX）语义不同**：MuJoCo 软接触的阻尼对持续接触一直生效，
恢复系数接近 1 时阻尼比 ζ → 0，轮地接触变成无阻尼弹簧，滚动中持续振荡，不是只在撞击时反弹；PhysX 的
restitution 只在相对速度超过 bounce 阈值的撞击时起作用，静止 / 滚动接触不受影响。所以 e > 0.7 的那部分
env 是一个物理上不真实、也几乎不可控的 plant，约占 30% 的 env，而名义 plant（e ≈ 0.05）上的策略同样被拖坏，
说明它通过共享策略伤到了全局，而不只是那部分 env。Oracle 救不回来也与此一致：知道 e 也压不住无阻尼接触。

建议下一个单变量对照：WideDR 只把恢复系数收到 0–0.5（实测 0–0.6 无影响，也符合橡胶轮实际），其余不动。

## Rest05 对照（Exp-WideDR-Rest05，2026-10-04，用户定）

相对 WideDR 唯一差异：恢复系数 DR 0–1 → 0–0.5。commit `17c91ba`，nulltask1 六卡 × 1365、5000 轮、seed 42，
W&B `hipb4j3a`，PID/PGID 800122，state `20261004T150954Z-00883`。

判据（4800–4999 轮，对 WideDR `txcesf5r` 与 Orient24 `3ix86na3`；评测端再跑 flat_eval / terrain_scoreboard /
stair_breakdown / dr_slice）：
- 恢复到 Orient24 附近（回报、yaw 误差、stairs_up 等级、跳跃列等级、catastrophic）→ 质心 ±5 cm、气弹簧 ×1.5、
  真 Kp/Kd、延迟 0–10 ms 这组加宽 DR 可以保留，且扰动 plant 下应比 Orient24 稳。
- 仍明显低于 Orient24 → 剩余差距来自其余加宽项（归因里次要嫌疑是质心 x 与低摩擦），再逐项拆。

中期（2340–2440 轮，Rest05 / WideDR / Orient24）：stairs_up 5.88 / 3.70 / 6.07，二级台阶门约 1300 轮打开，yaw 误差
1.06 / 3.01 / 0.91，但平地 tracking 0.59 / 0.56 / 0.72、catastrophic 0.12 / 0.55 / 0.002。2026-10-05 按用户指令在第 2537 轮停止。
**Rest05 只是截掉最坏一段的临时对策**：randomize_contact_restitution 改的是持续接触阻尼，e = 0.5 时阻尼比仍只有
0.2–0.25（基线 1.25），剩余差距不能归到质心 / 摩擦。

## NoRest 对照（Exp-WideDR-NoRest，2026-10-05，用户定）

相对 WideDR 唯一差异：不做恢复系数 DR（保留 Flat 的空事件，接触参数即 MJCF 默认，e ≈ 0.05）。MuJoCo 软接触表达不了
「只在撞击时反弹」，这项要么不做、要么以后改成语义如实的接触阻尼 DR（只在充分阻尼区间随机）。commit `11f9d40`，
nulltask1 六卡 × 1365、5000 轮、seed 42，W&B `5ji1xq6w`，PID/PGID 802601，state `20261004T163444Z-28466`。
对 Rest05 看 0–0.5 欠阻尼接触是否仍有代价；对 Orient24 的剩余差距才归到质心 ±5 cm、气弹簧、真 Kp/Kd、延迟、质量这组加宽项。

### NoRest 结果（2026-10-05，跑满 5000 轮）

训练端（NoRest / Rest05 / WideDR / Orient24）：
- 同轮次 2340–2440：回报 82.2 / 47.9 / 31.3 / 59.3，跳跃列等级 4.68 / 0.27 / 0.80 / 0.13，vx 误差 0.64 / 0.86 / 0.91 / 0.63。
  → 0–0.5 的欠阻尼接触仍有明显代价，NoRest 全面好于 Rest05；跳跃在 1300–2300 轮学会（Orient24 约 2900 轮）。
- 4800–4999：回报 78.2 / — / 41.0 / 85.5，stairs_up 5.92 / — / 4.07 / 6.15，跳跃列 4.68 / — / 0.80 / 4.57，平地 tracking
  0.757 / — / 0.572 / 0.747，yaw 误差 0.97 / — / 2.82 / 0.73，catastrophic 0.065 / — / 0.47 / 0.002，value loss 4.3 / — / 4.3 / 1.9。

评测端（sim2x 确定性，延迟 1 步；`.scratch/widedr_eval/*_norest.*`、`dr_slice3.*`）：
- 平地 21 组 vx 误差均值 0.171（Orient24 0.082，WideDR 0.477）；短板在 h0.30 中速（vx 1.0/1.5 误差 0.30/0.25，俯仰 std 3–4°，
  有前后晃）与 h0.38 高速（2.4 误差 0.53）；h0.38 静站 6 s 漂约 1 m。转向 1/2/4 rad/s 实际 1.05/2.12/4.22。
  跳跃离地高度与 Orient24 相同（0.21–0.51 m），最大倾角 13–16°（Orient24 21–22°）。
- 地形评分表：通过率 96%（波浪 r6/r9 12 s 内没走完 4 m，未摔），摔倒 1（Orient24 33），最大倾角 18.4°（51.5°），vx 误差 0.285（0.404）。
- 台阶 1–9 级：出 4 m 5.51 s（Orient24 5.52 s），0 摔。
- 扰动 plant（11 plant × 6 项）：摔倒 NoRest 0 / WideDR 1 / Orient24 5；质心 x +5 cm 时 vx1 误差 0.14（Orient24 0.56）、
  站立漂 0.36 m（6.09 m）；组合最坏 plant 6 项全过（vx2 误差 0.06、yaw 误差 0.00），Orient24 摔 5 项。

结论：去掉恢复系数 DR 后，这组加宽 DR（真 Kp/Kd、质心 ±5 cm、质量 −1…+3 kg、气弹簧 ×0.9–1.5、延迟 0–10 ms）
基本不伤地形、转向和跳跃能力，换来对参数偏差的明显鲁棒性；代价是名义 plant 上平地速度跟踪变粗（误差约 2 倍）、
h0.30 中速有前后晃、h0.38 静站缓慢漂移、catastrophic 仍高于基线。注意 Orient24 的姿态罚是 −24，名义平稳性的对比偏向它。

### 对照的混杂（2026-10-05 补记）

同步代码时 bundle 取的是「上一实验 commit..分支头」，期间另一会话在同一分支提交了默认改动，没有逐个核对就带进了实验：
- WideDR（80019b8）、Oracle（1fa00ca）：相对 Orient24 基线（aeec423）干净。
- Rest05（17c91ba）：额外带入 `55bce6d`（默认姿态罚 −12 → −24）。
- NoRest（11f9d40）：在 Rest05 之上又带入 `3919db7`（跳跃触发时 vx 不再截到 ±1.5）、`b50ef1f`（high_stand_transition_prob 0.5 → 0.1）、
  `6656fb5`（二级下台阶改直行指令）。
因此「NoRest 远好于 Rest05」「跳跃提前学会」有一部分可能来自这三处；离线归因（同一 WideDR 模型内回归）不受影响。

## 决定：DR1 并入默认（2026-10-05，用户定）

NoRest 的 DR 并入 rough 默认（`env_cfg.ROUGH_DR_*`，`_apply_rough_domain_randomization`），Exp-WideDR / -Oracle / -Rest05 / -NoRest
四个临时入口删除，`randomize_contact_restitution` 与 Oracle 用的两个观测函数一并删除（复现用对应 commit）。
等价校验：新默认 `SE3-WheelLegged-Rough` 的事件、延迟、观测项、奖励与 NoRest（11f9d40）逐项一致（`.scratch/widedr_eval/dump_cfg.py`）；
smoke（Rough、Flat-MLP 各 5 轮）通过，ONNX 契约延迟为 0–10 ms。
待补：同 commit、旧 DR、6×1365 的干净基线，单独量出 DR1 对名义性能的代价。

## 干净基线 OldDR（2026-10-05，用户定）

`SE3-WheelLegged-Rough-Exp-OldDR`：DR1 默认去掉 DR1（退回 Flat 继承的旧 DR、4–6 ms 延迟），其余不动。commit `afda1c0`，
在独立分支 `xyh/1005-olddr-baseline`（基于 c21d3ea）上——主 checkout 里另一会话有未提交的 commands / env_cfg 改动，不能混进来。
等价校验：配置与 11f9d40 的默认 Rough 逐项一致（事件、延迟、观测、奖励），与 NoRest（11f9d40 + DR1）只差质量、质心、
Kp/Kd、气弹簧、延迟五项；`git log 11f9d40..afda1c0 -- src` 只有 3f8db58 与 afda1c0。nulltask1 六卡 × 1365、5000 轮、seed 42，
W&B `8nv7aaoa`，PID/PGID 805137，state `20261005T053925Z-27281`。
对照 NoRest `5ji1xq6w`：训练端同轮次指标 + 评测端四套脚本，差值即 DR1 的单独代价 / 收益。

### OldDR 结果（2026-10-05，跑满 5000 轮；DR1 = NoRest 的单独效果）

训练端 4800–4999（OldDR / NoRest）：回报 79.7 / 78.2，地形均级 5.85 / 5.62，stairs_up 6.12 / 5.92，跳跃列 5.68 / 4.68，
vx 误差 0.56 / 0.65，平地 tracking 0.76 / 0.76，yaw 误差 0.77 / 0.97，catastrophic 0.016 / 0.065，value loss 1.93 / 4.26，
回合长度 386 / 472。

评测端 model_4999（sim2x 确定性，延迟 1 步；`.scratch/widedr_eval/*_olddr.*`、`dr_slice_od.*`，NoRest 复用 `*_norest.*`、`dr_slice3.*`）：
- 名义平地：21 组 vx 误差均值 / 最大 0.087 / 0.25 对 0.171 / 0.53，俯仰 std 均值 0.30° 对 1.30°；h0.30 vx 1.0 误差 0.11 对 0.27–0.30
  且 NoRest 俯仰 std 约 4°（中速前后晃，11 种扰动 plant 里 9 种都有）；h0.38 vx 2.4 误差 0.03 对 0.53；wz 4 误差 0.02 对 0.22。
  h0.38 静站漂移两者都有（0.8 / 1.0 m），不是 DR1 引起。
- 地形评分表：通过率都是 96%，vx 误差 0.195 对 0.285，摔 0 对 1。
- 台阶 1–9 级拆解：OldDR 第 7 级（vx 1.0）与第 9 级（三档速度）卡住出不了 4 m，NoRest 全部通过。
- 扰动 plant（66 项）：摔 3 对 0；质心 +5 cm 时 h0.30 vx 1.0 误差 0.76 对 0.14、h0.38 vx 2.0 摔对 0.08；组合最坏 plant OldDR
  摔 2 项、站立漂 11 m，NoRest 全过。
- 第 2000 轮同轮次评测趋势一致（扰动 plant 摔 3 对 0）。

结论（与预期一致，用户确认）：DR1 换来对质心 / 质量 / 气弹簧等参数偏差的鲁棒性与高台阶通过，代价是名义平地 vx 误差约 2 倍、
中速前后晃、转向略粗、跳跃课程略慢、catastrophic 略高。中速前后晃疑为质心 ±5 cm 带出的「探测质心」行为
（见 flat/env_cfg.py com 注释），候选单变量对照：DR1 上质心收到 ±2 cm，或按真机实测误差定范围。
OldDR 入口留在分支 `xyh/1005-olddr-baseline`（afda1c0），不并入主线。
