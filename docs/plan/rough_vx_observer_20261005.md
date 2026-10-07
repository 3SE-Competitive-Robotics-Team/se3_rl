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
