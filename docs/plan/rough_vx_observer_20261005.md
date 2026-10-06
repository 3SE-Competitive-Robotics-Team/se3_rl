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

## 验证

- 本地：最新帧下标与 mjlab 历史布局一致、两路梯度互不泄漏、ONNX 与 PyTorch 前向一致（`.scratch/vx_observer_check.py`）。
- smoke：`SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Rough-Exp-VxObserver --env.scene.num-envs 1 --gpu-ids None`。
