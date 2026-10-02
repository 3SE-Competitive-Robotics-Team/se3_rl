# J3：前进中起跳（放开 vx 指令）（2026-10-01，用户定）

## 动机

J2 三档原地跳都已成立（见 `j2_jump_mimic_tight_height_termination_20261001.md` 结果节），下一步是跑动中起跳。
按 J1 定下的口径，参考只约束局部竖直动作，前进速度交给奖励塑形，三条参考复用，不加轮速参考。

## 改动

`SE3-WheelLegged-Jump-Mimic-Exp-J3` = J2 + `env_cfg(max_lin_vel_x=1.5)`，`max_lin_vel_x` 默认 0，J1 / J2 不受影响：

- **vx 指令**：±1.5 m/s 均匀采样（Flat 部署上限 2.4 的约 60%），保留 20% 零速回合，原地跳不丢；ONNX 部署包络同为 ±1.5。
  yaw / pitch / roll 仍恒 0，跳跃仍可在任意速度下触发。
- **跳跃中冻结速度指令**：腾空时水平速度改不了，跳跃中重采样只会制造做不到的指令；reset 照常重采样。
- **跳跃期间速度跟踪只看 vx**：Flat 跟踪核 exp(−(Δvx² + 2·vz²)/σ) 在腾空 vz≈2 m/s 时整项归零，J2 跳跃期间
  前进速度没有任何塑形（原地跳无所谓，前进跳必须改）；跳跃期间竖直运动由模仿项管，跟踪核去掉 vz 项。
  非跳跃期间定价不变。
- **RSI 带速度**：从参考中途开始的回合按本回合 vx 指令给机身水平速度（按随机朝向投影）和无滑轮速
  （左轮 +v/r、右轮 −v/r，轴向见 common_mistakes #1）；站姿开始的回合从静止起步。

这几处是"放开 vx"成立的配套，没有其中任何一处，vx 指令在跳跃期间要么无塑形、要么不可达。

## 预期与观察点

- 看 `Locomotion/base_vx_error_abs` 与 `Jump/height_err_abs_active`：前者在跳跃期间也应收敛，后者不应比 J2 明显变差。
- 风险：起跳时轮子仍在地上滚动，四连杆在蹬地相同时承担前后平衡与竖直加速，高速段可能先学成"刹车再跳"；
  回放要专门看 1.0–1.5 m/s 触发时落地前后的速度损失。
- sim2x 回放：分别在 vx = 0 / 0.5 / 1.0 / 1.5 下触发三档跳跃。

## 验证

- 功能检查（`.scratch/j3_check.py`，CPU 16 env）：vx 指令在 ±1.5 内、4/16 零速；RSI 回合机身 vx 与轮速×半径都等于指令，
  一步后速度保持（无打滑）；强制重采样时跳跃中的 5 个 env vx 都不变；200 步奖励与观测全有限，actor 54 维。
- 三个入口参数核对：J1 / J2 的 vx 包络 (0, 0)、跟踪项为原 `tracking_lin_vel`；J3 为 ±1.5、`tracking_lin_vel_jump`。
- `tests.test_onnx_metadata` + `tests.test_flat_baseline` 38 项通过；CPU smoke（1 env、5 轮）通过，导出的 ONNX
  能被 se3_runtime 加载（`supports_jump=True`）。

## 启动记录

代码 commit `4a43f13`。whtws GPU 0–5 被别人的实验占用，改在 nulltask1 六卡，所以比 J2 少一张卡（总样本约为 J2 的 6/7）。
nulltask1 主 checkout 停在别人的分支，J3 在独立 worktree `/workspace/se3-worktrees/j3-4a43f13` 中运行（见机器档案）。

- 启动时间：2026-10-01 18:15（Pod 时区），六卡 × 8192、5000 轮、每 200 轮保存、seed 42，从头训，在线 W&B（项目 `SE3-WheelLegged-Jump-Mimic`）。
- run：`2026-10-01_18-15-37_jump-J3-mimic-vx15-seed42-6x8192-5k`，
  W&B [cjilu9wl](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Jump-Mimic/runs/cjilu9wl)，PID/PGID `748772`，
  state `/workspace/.se3-training-state/nulltask1/20261001T101530Z-18005`。
- 首轮核验：第 37 轮在迭代、无 Traceback、无 nefc overflow，1.37 s/轮，六卡各 6.1–6.5 GB、利用率 70–73%；
  `Jump/active_rate` 0.25，`Locomotion/cmd_vx_mean` −0.02（对称采样），`base_vx_error_abs` 0.79（随机策略），
  `mimic_deviation` 终止每轮约 54 次（与 J2 开局同量级）。
- 2026-10-01 18:47 按用户指令 SIGINT 停止（第 1396 轮，最后 checkpoint `model_1200`），六卡让给 J4。
  停止前 model_800 回放：vx 0 / 0.5 / 1.0 / 1.5 下三档全部起跳、落稳，离地间隙在目标 ±1 cm 内；跳跃期间平均速度基本保持，
  最低点比起跳前低 0.2–0.3 m/s；vx=0 时向后溜约 0.1 m/s，vx=0.5 欠速约 18%（`.scratch/j3_moving_jump.py`）。
