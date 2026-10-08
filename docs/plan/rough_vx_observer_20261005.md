# 显式 vx 观测器（2026-10-05，用户定）

## 依据

DR1 下 actor 只给 vx 特权（`Exp-Vx`，`qnd9o87r`）就拿回了 3 维速度特权的大部分平地跟踪收益（vx 误差 0.583，基线 0.659），
DR 参数 31 维基本没用（`docs/plan/rough_dr1_oracle_20261005.md`）。部署端没有 vx，用 actor 历史估计。

## 结构

入口 `SE3-WheelLegged-Rough-Exp-VxObserver`，代码 `se3_train/vx_observer.py`，相对当前默认只改 actor：

- actor 观测组整组 16 帧历史（0.32 s @ 50 Hz），term-major、oldest→newest 展平，30 × 16 = 480 维；
  reset 后第一帧回填整段（mjlab CircularBuffer，与 runtime 一致）；噪声逐帧在入历史前施加。
- 估计器 MLP 480 → 128 → 64 → 1（ELU），输入为与 policy 共用的经验归一化历史，输出 v̂x（m/s）。
- policy MLP 输入 [最新一帧 30 维, sg(v̂x)] = 31 维，隐藏层 128/64/32（同默认）。
- critic 不变（单帧 + 特权）。

梯度（concurrent state estimator，Ji et al. 2022 RA-L / HIMLoco）：

- PPO 损失 → policy MLP 与分布参数；v̂x 在 policy 前 detach，PPO 对估计器梯度恒为 None。
- MSE(v̂x, vx_true) → 估计器，独立 Adam lr 1e-3；目标组 `estimator_target`（同一时刻真实机身系 vx，无噪声），
  不进任何网络输入，只进 rollout storage。
- 估计器在 PPO 各 epoch 之后用同一批 rollout 更新（epoch 5 × mini-batch 4），使 PPO 的 ratio 从 1 开始；多卡时估计器梯度跨卡平均。
- 日志 `Loss/estimator_vx_mse`、`Loss/estimator_vx_rmse`；checkpoint 额外保存 `estimator_optimizer_state_dict`。

部署：ONNX 输入 `obs [1, 480]`、输出 `actions [1, 6]`，metadata 按 History-MLP 契约（每项 history_length 16）导出，
sim2x runtime 无需改动。STM32 侧需另做：转换器放开 history_length≠1、policy_runtime 加 16 帧环形缓冲与首帧回填。
估计器约 7.0 万参数，policy 约 1.5 万参数。

## 对照与判据

对照默认基线与 Vx 特权（`qnd9o87r`，上限）。同轮次比平地 tracking / vx 误差、台阶航向、跳跃列等级、catastrophic，
外加 `estimator_vx_rmse`；接近 Vx 特权 → 观测器有效；与基线持平 → 查 RMSE 是否学低、16 帧是否不够。

## 启动记录

nulltask1 六卡 × 1170、5000 轮、seed 42、从头训，cdfc7ec，W&B `agqj496q`，PGID 914325，state `20261006T034350Z-23459`。
规模与 Vx 特权（`qnd9o87r`）相同，便于直接对比。第 37 轮 1.19 s/轮（与 Vel 相同），estimator_vx_rmse 0.052 m/s。

## 验证

- 本地：最新帧下标与 mjlab 历史布局一致、两路梯度互不泄漏、ONNX 与 PyTorch 前向一致（`.scratch/vx_observer_check.py`）。
- smoke：`SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Rough-Exp-VxObserver --env.scene.num-envs 1 --gpu-ids None`。
- 多卡：两进程 gloo 下各 rank 数据不同，估计器更新后权重逐位相同（`.scratch/vxobs_ddp_check.py`）；nulltask1 双卡 × 256 smoke 5 轮通过。

## SpringFF（2026-10-06，已删除）

尝试过让估计器再输出左右弹簧力、替代固定 300 N 做前馈（入口 `Exp-VxObserver-SpringFF`，6c4bd2a，W&B `pcz5opfg`，
actor 回传上一拍下发值以保证闭环可辨识）。训练端下发值 RMSE 约 22 N（固定 300 N 约 79 N），但 sim2x 静站时估计器在左右反对称方向
只照抄上一拍（开环响应斜率 1.01–1.04），真值处的小偏差无回复力，左侧漂到约 420 N、右侧压到下限 250 N。用户判断没用，
入口与代码删除，前馈维持固定 300 N。复现用 6c4bd2a。

## H5：历史 16 → 5 帧（2026-10-06，用户定）

入口 `SE3-WheelLegged-Rough-Exp-VxObserver-H5`（b425351）：只把 actor 历史改为 5 帧（0.1 s，估计器输入 480 → 150，MCU 上约 2.6 万次乘加），
其余与 VxObserver 逐项相同。nulltask1 六卡 × 1170、5000 轮、seed 42，W&B `4b2gzdny`，state `20261006T164657Z-07242`，
与 `agqj496q` 同规模直接对比；判据：estimator_vx_rmse、平地 / 台阶 vx 误差、yaw 误差与 sim2x 回放。

## Z3：估计器额外输出 3 维隐向量（2026-10-06，用户定，方案 A）

入口 `SE3-WheelLegged-Rough-Exp-VxObserver-Z3`（3879d15）：估计器输出 [v̂x, z₁, z₂, z₃]，v̂x 仍只由 MSE 监督、detach 后进 policy；
z **不 detach**、无额外监督，由 PPO 梯度端到端训练（梯度经 z 进入与 v̂x 共用的估计器主干）。policy 输入 34 维 [ô_t, sg(v̂x), z]。
曾比较过的另两种训练方式：β-VAE 预测下一帧（DreamWaQ 风格）、特权编码再回归（CTS 风格，DR1 Oracle 显示 DR 参数对跟踪帮助小）。
whtws 六卡 × 1170（GPU 0–5）、5000 轮、seed 42，W&B `byw4hj29`，state `20261006T170216Z`，与 `agqj496q` 同规模直接对比。
判据：Loss/estimator_latent_std（z 是否塌缩）、estimator_vx_rmse 是否被 PPO 梯度拖坏、平地 / 台阶 vx 误差、yaw 误差与 sim2x 回放。
结果（第 1500 轮，H5 / 16 帧 VxObserver）：estimator_vx_rmse 0.168 / 0.125（全程高约 34%），vx 误差 0.685 / 0.627，平地 tracking 0.760 / 0.787，
回报 50.3 / 57.0，台阶航向与 yaw 持平。0.1 s 历史不够估准 vx，用户判断不再继续，第 2043 轮停止；默认维持 16 帧。

## CTS + 显式 vx（2026-10-07，用户定）

入口 `SE3-WheelLegged-Rough-Exp-CTS`（05dc18f，`se3_train/cts_observer.py`）。参照 clearlab-sustech/multi_loco_isaacgym 的
`ppo_cts.py` 与 Realsbt/RL-RobotLab 的 rsl_rl MoE-CTS（均 Apache-2.0）：教师 env 用 critic 特权观测编码的 32 维 L2 归一化隐向量拼真实 vx，
学生 env 用 16 帧历史编码器（256/128）回归 [sg(z_T), vx_true]，两组共用 policy（128/64/32）；教师编码器（512/256）只由 PPO 训练、
学生编码器只由重建损失训练（隐向量 MSE + vx MSE 等权，官方代码对 33 维拼接整体取均值会让 vx 只占 1/33 权重），critic 不变。
教师 : 学生 = 3 : 1（env 编号 % 4 == 0 为学生，观测组 cts_role）。部署只走学生路径，ONNX 480 维输入。
训练端环境指标被占 75% 的教师 env 主导，另记 Loss/cts_reward_teacher、Loss/cts_reward_student；结论以 sim2x 扫描为准。
nulltask1 六卡 × 8192（用户定）、5000 轮、seed 42，W&B `n6femb3m`，state `20261007T025340Z-03848`，新默认轮子包络。
规模与此前 VxObserver 系列（六卡 × 1170）不同，对比混有规模变量。
对照：同 commit（05dc18f）、同规模的普通 VxObserver 在 whtws 六卡 × 8192（GPU 0–5）、5000 轮、seed 42，W&B `gxse68wn`，
state `20261007T033707Z`；两条只差 CTS 一个变量。


中途结果（约 3100 轮）：训练端 vx 误差 0.36（VxObs8k 0.51），但这是教师主导的数字；学生 / 教师单步奖励差从 500 轮的 0.012
拉大到 2000 轮的 0.114（3100 轮 0.092），隐向量重建 MSE 0.004 → 0.009；2700 轮起 catastrophic_state 从约 0.01 升到 0.2–0.27。
model_3700 学生路径 sim2x：上台阶动作怪异（用户观察）。归因：教师编码器读整个 critic 组，含 77 点前视高度扫描（x ±0.5 m、
y ±0.3 m），这是盲学生从本体历史原理上推不出的信息；共用 policy 学会按 z 里的前视地形提前动作，学生只能回归条件均值，
撞上立面前做"半个台阶动作"、撞上后 z 跳变。用户定停掉这条，改做 NoScan。

## CTS-NoScan：教师编码器去掉高度扫描（2026-10-07，用户定）

入口 `SE3-WheelLegged-Rough-Exp-CTS-NoScan`。与 Exp-CTS 只差一处：教师编码器改读观测组 `cts_teacher`
（critic 各项去掉 `height_scan`，160 → 83 维）；critic 价值函数仍带高度扫描（不部署，不影响学生）。
保留的 base_height 是机身下方局部高度，学生可由腿部构型与接触推断，不算前视信息。
判据：学生 / 教师单步奖励差保持在 0.03 以内、重建 MSE 不随课程上升；sim2x 学生路径上台阶动作正常；yaw 与纯 vx 扫描不差于 VxObs8k。
Exp-CTS（`n6femb3m`）于约 4198 轮停止（SIGTERM，用户定）。NoScan 在 nulltask1 六卡 × 8192、5000 轮、seed 42（ac79a46，
worktree `rough-cts-noscan-ac79a46`），W&B `le9s9gk1`，state `20261007T070820Z-28400`，约 3.07 s/轮。
model_1000 学生路径 sim2x（m51_r369 网格）：用户观察上台阶不再出现 Exp-CTS 的怪异动作。注意轮次混杂（Exp-CTS 看的是 model_3700），
训练端师生单步奖励差在 1000 轮时两条相同（0.045 vs 0.043），须在 ≥3000 轮复看确认。
同轮次对照：Exp-CTS model_1000 学生路径上台阶也正常（用户观察）——怪异动作是后期学出来的，NoScan model_1000 正常不构成证据。
待定位 Exp-CTS 怪动作出现的轮次（已拉回 1500/2000/2500/3000），NoScan 跑到该轮次后再同轮次对比。
Exp-CTS 怪动作起点：model_1500 正常、model_2000 开始出现（用户观察），与师生单步奖励差 1000 → 2000 轮从 0.043 拉到 0.114 同步。
判定点：NoScan model_2000（及之后）学生路径同场景回放。

NoScan 结果（5000 轮跑完；W&B 在 1641 轮因入口机中断 + gateway 主循环卡死断档，修复后客户端自动补传）。train.log 500 轮窗口均值：

| 窗口 | 教师 | 学生 | 差 | 重建 MSE | catastrophic |
|---|---|---|---|---|---|
| 1000 | 0.291 | 0.249 | 0.042 | 0.007 | 0.012 |
| 2000 | 0.307 | 0.268 | 0.039 | 0.005 | 0.036 |
| 3000 | 0.328 | 0.287 | 0.041 | 0.004 | 0.137 |
| 4000 | 0.294 | 0.243 | 0.051 | 0.004 | 0.065 |
| 4500 | 0.306 | 0.258 | 0.048 | 0.003 | 0.036 |

Exp-CTS 同期差 0.043 → 0.114（2000）→ 0.091（3000），重建 MSE 0.004 → 0.009；NoScan 的差在 1000 轮后不再扩大、重建 MSE 单调下降，
师生分叉由前视扫描造成。catastrophic_state 3000 轮附近的尖峰两条都有（NoScan 0.137、Exp-CTS 0.27），与扫描无关，另查。

### sim2x 扫描：NoScan vs VxObs8k（model_4999，whtws，`.scratch/yaw_diag/{yaw,vx}_sweep.py`，结果 `.scratch/sweep3/`）

锚点 VxObs1170 与 2026-10-06 扫描逐位一致，口径可比。yaw 实际/指令（全部 vx、正负号均值）：

| 模型 | 0.75 | 0.9 | 1.0 | 纯 vx nominal | low_mu | com_fwd | mass_p2 |
|---|---|---|---|---|---|---|---|
| NoScan | **1.00** | **0.70** | **0.56** | 0.093 | 0.088 | 0.081 | 0.080 |
| VxObs8k | 0.83 | 0.59 | 0.37 | 0.063 | **0.057** | **0.055** | 0.057 |
| VxObs1170 | 0.90 | 0.52 | 0.32 | **0.054** | 0.060 | 0.102 | **0.053** |
| Z3（10-06） | 1.03 | 1.03 | 0.95 | 0.093 | 0.090 | 0.115 | 0.091 |

包络顶端按 vx 拆开：NoScan 在 vx ≥ 1.0 时 0.9 / 1.0 倍上限都转满（0.98–1.00），vx = 0、0.5 的原地 / 低速大 yaw 三个模型都拒转（0.09–0.63）。
NoScan 纯 vx 误差集中在 1.5 m/s 欠速（实际 1.23）与 0.6 m/s 超速（0.75）；VxObs8k 的扩规模收益主要是鲁棒性（com_fwd 0.102 → 0.055），
名义跟踪与 yaw 不比 VxObs1170 好。结论：带隐向量的两条（Z3、CTS-NoScan）都是 yaw 好、vx 差，显式 vx 单独用 vx 最好、yaw 顶端拒转。

### 干净基线：无观测器 + rated_point + 6 × 8192（2026-10-07，用户定）

NoScan 对 Base / Default 的比较混有规模（4–6 × 1170）与轮子包络（Default 是旧包络）两个变量，补一条只差"有无观测器"的基线：
默认入口 `SE3-WheelLegged-Rough`（单帧 30 维 actor、rated_point、legacy 指令预算），ac79a46，whtws 六卡 × 8192、5000 轮、seed 42，
save 100，W&B `8c62k9l7`，state `20261007T140237Z`。跑完同口径 sim2x 扫描。

### NoScan vx 偏差归因（`.scratch/vx_diag/vx_counterfactual.py`，whtws，nominal，2 seed 均值）

改图把估计器输出接出来读 v̂x；oracle 模式把送进 policy 的 v̂x 换成 MuJoCo 机身系真实 vx（z 仍为学生估计）。
真实 vx（student / oracle）：

| ckpt | 0.3 | 0.6 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|
| 2000 | 0.28 / 0.31 | 0.60 / 0.62 | 1.03 / 1.04 | 1.51 / 1.51 | 1.97 / 1.97 |
| 3000 | 0.29 / 0.31 | 0.65 / 0.67 | 0.99 / 1.00 | 1.37 / 1.39 | 1.78 / 1.78 |
| 3700 | 0.27 / 0.30 | 0.72 / 0.72 | 1.02 / 1.02 | 1.35 / 1.36 | 1.92 / 1.93 |
| 4999 | 0.25 / 0.27 | 0.76 / 0.77 | 1.00 / 1.02 | 1.23 / 1.26 | 1.89 / 1.90 |

v̂x 偏差全程 ≤ +0.03，换成真实 vx 不改变结果——不是估计误差。偏差是 2000 轮后 policy 随训练逐步长出来的（2000 轮全速度跟准）。
待判：MJLab 内教师 / 学生 env 同指令是否也偏（训练端问题）还是只在 MuJoCo 偏（z 放大 sim2sim gap）。

MJLab 判别（`.scratch/vx_diag/cts_mjlab_eval.py`，nulltask1 GPU 0，平面、64 env/速度、无观测噪声；dr=off 无事件，dr=on 只留 startup DR）：
从 .pt 恢复完整 CTS actor，同批 env 分别喂 [z_T, vx_true] / [z_S, v̂x] / [z_T, v̂x] / [z_S, vx_true]。

| model_4999，dr=off | 0.3 | 0.6 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|
| 教师 [z_T, vx_true] | 0.27 | 0.75 | 0.99 | 1.27 | 1.97 |
| 学生 [z_S, v̂x] | 0.25 | 0.75 | 1.00 | 1.24 | 1.89 |
| MuJoCo 学生 | 0.25 | 0.76 | 1.00 | 1.23 | 1.89 |

model_2000 四种输入、两种 DR、MuJoCo 全部跟准（0.6 → 0.58–0.65，1.5 → 1.50–1.53）。z_S 与 z_T 余弦 0.97–1.00。
结论：偏差是训练端 policy 本身学出来的——教师路径（特权信息 + 真实 vx）在训练仿真里同样 0.6 超速、1.5 欠速；
学生 z 与教师一致，MJLab 与 MuJoCo 的学生结果逐位相近，sim2sim gap 可忽略。下一步看奖励账本：4999 的偏差行为在同一套奖励下是否比 2000 的跟准行为更赚。

### 奖励账本与密集传递曲线（2026-10-08）

账本（`.scratch/vx_diag/cts_reward_ledger.py`，MJLab 平面，观测指令 = 计酬指令）：同一指令下 4999 的每秒总回报比 2000 低
0.6：−1.6…−2.0，1.5：−2.6…−3.8（四种 DR × 路径组合一致），主要亏在 tracking_lin_vel_narrow（−1.2…−2.4）与宽核（−0.3…−0.8），
没有一项为偏差买单——奖励不鼓励偏差，是 policy 在自己的目标上退化。

MuJoCo 学生路径 0.2–2.4 步长 0.1 传递曲线（`vx_dense.py`），平均 / 最大 |误差|：

| | 2000 | 3000 | 3700 | 4999 |
|---|---|---|---|---|
| NoScan | **0.022 / 0.050** | 0.088 / 0.220 | 0.099 / 0.244 | 0.140 / 0.398 |
| 干净基线（8c62k9l7） | 0.183 / 0.383 | 0.164 / 0.440 | 0.107 / 0.303 | 0.129 / 0.337 |

NoScan-4999：0.4–1.0 超速（0.5 → 0.65），1.1–1.8 欠速（1.7 → 1.30），≥1.9 恢复。基线：0.4 → 0.5 跳变（0.31 → 0.84），0.5–1.4 全段超速 +0.3。
训练端：平地列 tracking_lin_vel_flat 1500–5000 轮 0.885 → 0.877 不变；台阶列同期 base_vx 0.90 → 0.99、高度罚 −0.49 → −0.27，仍在进步。
台阶列 vx 指令 U(0.4, 2.4)（均值 1.33），实速约 1.0。

假设（待证）：盲 policy 在碰到立面之前分不清平地与台阶列的引道，2000 轮后继续优化台阶列，把台阶上的速度曲线
（低指令冲速靠动量过沿、中高指令按台阶节奏放慢）泄漏到平地：偏差带（0.4–1.0 超、1.1–1.8 欠）正落在台阶列指令区间内，
≥1.9 恢复。验证：MJLab 台阶列按指令扫实速，看 4999 的台阶传递曲线是否与平地偏差同形、2000 是否不同。

### 台阶列验证（`.scratch/vx_diag/cts_stair_eval.py`，MJLab 训练地形，等级 5 / 3，学生路径，16 env/格）

平地列取最后 3 s，台阶列取前进 0.3–3.0 m 窗口（首道立面起几级）。等级 5、学生路径：

| 指令 | 0.4 | 0.6 | 0.8 | 1.2 | 1.4 | 1.6 | 1.8 |
|---|---|---|---|---|---|---|---|
| stairs_up 2000 | 0.37 | 0.61 | 0.80 | 1.03 | 1.19 | 1.13 | 1.27 |
| stairs_up 4999 | **0.59** | **0.79** | 0.92 | 1.11 | 1.19 | 1.14 | 1.27 |
| flat 2000 | 0.41 | 0.64 | 0.86 | 1.20 | 1.41 | 1.55 | 1.57 |
| flat 4999 | **0.54** | **0.73** | 0.90 | **1.07** | **1.17** | **1.28** | 1.47 |

stairs_up 0.4 指令的爬升高度 2000 为 0.36 m、4999 为 0.56 m（满程约 0.65 m）：低指令冲速是 2000 轮后为爬完台阶学出来的，
同幅度的超速原样出现在平地。中高指令台阶列两个 checkpoint 都按台阶节奏 1.1–1.3 跑；2000 在平地会跑回指令（1.4 → 1.41），
4999 在平地也按台阶节奏跑（1.4 → 1.17、1.6 → 1.28）。等级 3 与教师路径同形。假设成立：台阶列的速度曲线泄漏到平地，
policy 不再依据"一直没碰到立面"把平地与台阶引道区分开。台阶类列占地形 39%（stairs_up 25.2%、two_step_up 13.5%），
flat 6.3%；台阶列指令 U(0.4, 2.4)，非台阶列指令均值约 0.6，平地 1.1–1.8 的样本主要来自台阶列。
（平地列 ≥ 2.0 的尾段数会驶出 9 m 地块，不可信。）

## CTS-NoScan-StairVx08：台阶类列前进指令下限 0.4 → 0.8（2026-10-08，用户定）

入口 `SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08`。与 NoScan 只差 `stair_lin_vel_x_range`（0.4, 2.4）→（0.8, 2.4），
作用于 stairs_up、stairs_two_step_up；观测、奖励、网络逐项相同（本地核对 cmd cfg 只差这一字段）。针对低速段泄漏
（平地 0.4–1.0 超速），中高速段（1.1–1.8 欠速）不在本变量范围内。
判据：平地密集传递曲线 0.4–1.0 段不再超速且 4999 不劣于 NoScan-2000；台阶列 0.8 起的爬升不退；
低于 0.8 的指令在台阶列不再出现，需单独看平地 0.4–0.8 是否仍跟准。
