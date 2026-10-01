# J2：跳跃 mimic 收紧机身高度偏离终止（2026-10-01，用户定方案 1）

## 动机

J1（edy6opq0）model_2000 在 sim2x 回放里只有 0.40 m 真的起跳，0.20 / 0.30 m 原地不动。
根因在终止阈值：原地不跳时机身与参考的最大高度差，0.20 m 约 0.19 m、0.30 m 约 0.23 m，
都低于 J1 的 0.25 m 阈值，所以"不跳"不会被提前终止。只有 0.40 m 的偏差超过阈值，不起跳就会被终止。
对这两条参考来说，不起跳只损失模仿奖励，反而躲开了起跳失败的摔倒风险，策略学成"装没听见"。

## 改动（单变量）

`SE3-WheelLegged-Jump-Mimic-Exp-J2` = J1，只把 `mimic_deviation` 的机身高度阈值从 0.25 m 收紧到 0.12 m。
0.12 m 取在用户给的 0.10–0.15 m 区间里：三条参考的不跳偏差都会越过它，空中正常跟踪的误差又留有余量。
腿长阈值 0.12 m、奖励、RSI、执行链、PPO 都不动。

实现上 `env_cfg(max_height_error=...)` 新增参数，J1 默认仍是 0.25（常量 `JUMP_MIMIC_MAX_HEIGHT_ERROR`），
J2 用 `JUMP_MIMIC_J2_MAX_HEIGHT_ERROR`。

## 预期与观察点

- 开局 `mimic_deviation` 终止率会比 J1 高，之后应该下降。如果长期不降，说明 0.20 m 参考的蹬地/腾空
  在当前执行链下难跟（J1 文档里 RSI 空中初始化还有 2.6 cm 偏差待查）。
- 必看 sim2x 回放：三档都要起跳，落地稳住回站姿。

## 验证

- J1 / J2 的终止参数分别是 0.25 / 0.12；`tests.test_onnx_metadata` + `tests.test_flat_baseline` 38 项通过。
- CPU smoke（1 env、5 轮）通过，导出 ONNX。

## 启动记录

代码 commit `e0be3e4`，Pod 经 git bundle 由 `8cb1dcb` 快进（子模块 bdff426 → 446e89f）；J1 为此停止。

- 启动时间：2026-10-01 07:44（Pod 时区），七卡 × 8192、5000 轮、每 200 轮保存、seed 42，从头训，在线 W&B（项目 `SE3-WheelLegged-Jump-Mimic`）。
- run：`2026-10-01_07-44-16_jump-J2-mimic-h012-seed42-7x8192-5k`，
  W&B [naqomk10](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Jump-Mimic/runs/naqomk10)，PID/PGID `3566109`，
  state `/workspace/.se3-training-state/whtws/20261001T074409Z`。
- 首轮核验：第 33 轮在迭代、无 Traceback、无 nefc overflow，1.39–1.41 s/轮，七卡各 6.2–6.7 GB、利用率 67–71%；
  `Jump/active_rate` 0.24，`mimic_deviation` 终止每轮约 55–57 次。
