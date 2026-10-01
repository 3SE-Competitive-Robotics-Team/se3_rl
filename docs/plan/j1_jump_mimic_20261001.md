# J1：跳跃 mimic 首版（单独跳跃策略、MLP、单帧 + 参考帧、从头训）（2026-10-01，用户定）

## 决策来源

调研（华南理工 wheeled-legged_RL、复旦 fudan_rl_wheel_leg、四足/双足跳跃论文）后用户定：

- 单独训一个跳跃策略（站立 + 起跳 + 落地稳住），部署时与行走策略切换；不从行走策略热启动，从头训。
- MLP、单帧本体观测；按 mimic 思路，参考帧（当前 + 未来帧）可以进 actor，但**不把相位标量输入网络**。
- 参考轨迹第一阶段只要运动学合理，不追求动力学可行；后续用训练 rollout 当新数据集迭代。
- 参考只约束局部竖直动作（腿长、机身高度、竖直速度、接触）；前进速度、轮速等交给奖励塑形，前进跳复用同一组参考。
- 执行链与 rough M54 对齐。

## 参考轨迹（se3.jump_ref.v1）

`uv run se3-jump-to` → `assets/trajectories/jump_ref_v1/jump_{0.20,0.30,0.40}m.npz`，生成器 `src/se3_jump_to/reference.py`。
旧 `se3_jump_to.cli` 插值生成器在当前闭链模型上失效（旧输出坐标膝角边界把膝钉在上限、蹲深约 4 mm；起点 0.295 m
与 v2 默认站姿 0.22 m 不符；起跳一帧 14 g），已替换。

| 目标轮底离地间隙 | 起跳速度 | 蹬地 | 腾空 | 空中最短腿长 | 总时长 |
|---|---|---|---|---|---|
| 0.20 m | 1.47 m/s | 0.226 s | 0.310 s | 0.216 m | 1.96 s |
| 0.30 m | 1.72 m/s | 0.194 s | 0.360 s | 0.150 m | 1.96 s |
| 0.40 m | 2.22 m/s | 0.150 s | 0.460 s | 0.150 m | 1.98 s |

站姿 0.28 m → 下蹲 → 恒加速蹬地 → 腾空（质心抛体，上升收腿、下降伸腿）→ 恒减速缓冲 → 回站姿；接地段轮心固定在髋后
2.96 cm（默认站姿），dt 0.005 s。准静态力矩检查（`.scratch/jump_ref/check_feasibility.py`，只作参考）：三条都在
32 N·m 包络内（利用率峰值 0.58），蹬地时 300 N 气弹簧提供约 18 N·m。已知运动学瑕疵：离地瞬间腿伸缩速度不连续、
触地时轮子以下落速度接地。回放视频 `.scratch/jump_ref/videos/`。

## 任务 `SE3-WheelLegged-Jump-Mimic-MLP`（`src/se3_train/tasks/jump_mimic/`）

以 Flat-MLP 基线为底：

- **执行链**：腿部 T-N 包络 ×0.8（平台 32 N·m）+ 300 N 气弹簧电机侧前馈补偿，与 M54 同。
- **指令**（8 维契约不变）：vx / yaw / pitch / roll 恒 0，站姿高度恒 0.28 m；站满 1 s 后每秒 0.5 概率触发，
  三条参考均匀选一；参考时钟环境内部按时间推进，播完回站立。jump_flag = 跳跃中，jump_target_height = 目标间隙，
  jump_phase 恒 0。
- **观测**：actor 54 维 = 34 维本体 + 参考帧 20 维（偏移 0/2/5/10 policy step，即当前、+40/+100/+200 ms；每帧
  [左腿长差×5, 右腿长差×5, (参考机身高度−0.28)×5, 参考竖直速度×0.5, 参考接触]）；critic 112 维，另含参考时钟与编号。
- **模仿对象是虚拟腿长而非关节角**：参考的轮心 x 固定，逐关节跟踪会钉死轮子前后位置、与轮腿前后平衡冲突。
- **奖励**：Flat 原定价（速度跟踪、轮/腿离地罚、高度罚本就按 jump_flag 屏蔽）+ 模仿项
  腿长 exp(−ΣΔL²/0.03²) ×3.0、机身高度 exp(−Δz²/0.05²) ×3.0、竖直速度 exp(−Δvz²/0.5²) ×1.5、轮接触一致 ×1.0；
  静站罚与轮子大接触力罚在跳跃期间置零。
- **reset / 终止**：50% 回合从参考随机时刻初始化（RSI，含空中状态），其余从站姿开始；跳跃期间机身高度偏差 > 0.25 m
  或腿长偏差 > 0.12 m 提前终止。
- 去掉 Flat 速度课程（会自动放开 vx），保留推扰课程；回合 10 s；PPO 同 Flat-MLP（[512, 256, 128]，24 步/env）。

## 部署

ONNX metadata 已登记 `jump_reference` 观测项（宽 20、偏移与特征名），训练端导出可用；se3_runtime 尚未实现参考
播放器与该观测项，sim2x / 真机加载会拒绝，部署前需在 runtime 实现（第二步）。

## 验证

- 生成器与 `.scratch` v1 参考逐数值一致（差值 0）。
- 功能检查（`.scratch/jump_mimic_check.py`，CPU 8 env）：actor 54 维、critic 112 维；RSI 写入后机身高度与腿长和参考
  差 1 cm 内（空中初始化的一个 env 差 2.6 cm，待查）；300 步内触发 10 次跳跃，奖励与观测全有限。
- `tests.test_rough_port` + `tests.test_onnx_metadata` + `tests.test_flat_baseline` 79 项通过；
  CPU smoke（1 env、5 轮）通过，导出 ONNX。

## 启动记录

代码 commit `8cb1dcb`，Pod 由 `01e87e2` 经 git bundle 快进。M54 已跑满，七卡空闲。

- 启动时间：2026-10-01 06:34（Pod 时区），七卡 × 8192、5000 轮、保存 200、seed 42，从头训，在线 W&B（项目 `SE3-WheelLegged-Jump-Mimic`）。
- run：`2026-10-01_06-34-29_jump-J1-mimic-v1ref-seed42-7x8192-5k`，
  W&B [edy6opq0](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Jump-Mimic/runs/edy6opq0)，PID/PGID `3541217`，
  state `/workspace/.se3-training-state/whtws/20261001T063422Z`。
- 首轮核验：第 36 轮在迭代、无 Traceback、无 nefc overflow，1.41 s/轮（平地，约为 rough 的 40%），七卡各 6.2–6.7 GB、
  利用率 64–87%；`Jump/active_rate` 0.25，开局 `mimic_deviation` 终止多（策略随机，属预期），后续看它是否下降。
- 2026-10-01 07:43 按用户指令 SIGINT 停止（第 2928 轮，最后 checkpoint `model_2800`），七卡让给 J2。
  model_2000 回放结论：只有 0.40 m 起跳，0.20 / 0.30 m 不起跳不会被终止（见 `j2_jump_mimic_tight_height_termination_20261001.md`）。
