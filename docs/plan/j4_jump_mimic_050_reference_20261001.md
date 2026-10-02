# J4：加入 0.50 m 跳跃参考（2026-10-01，用户定）

## 改动（单变量）

`SE3-WheelLegged-Jump-Mimic-Exp-J4` = J3 + 第四条参考 `assets/trajectories/jump_ref_v1/jump_0.50m.npz`，
其余（vx ±1.5、高度偏离终止 0.12 m、奖励、RSI、执行链、PPO）与 J3 相同。触发时四条参考均匀选一。

实现：`env_cfg(reference_heights=...)` 新增参数，J1–J3 默认仍是 0.20 / 0.30 / 0.40；ONNX 部署包络
`jump_target_height` 上界取参考最大值（J4 为 0.50）。critic 的参考编号 one-hot 多一维（112 → 113），actor 仍 54 维。
sim2x 的 Jump Height 下拉框加了 0.50；runtime 按最近参考选轨，三条参考的旧策略选 0.50 时会落到 0.40。

## 0.50 m 参考

`uv run se3-jump-to --heights 0.50`，生成器参数与另外三条相同（站姿 0.28 m、下蹲腿长 0.150、起跳腿长 0.315、
空中收腿下限 0.150 m、dt 0.005 s）。

| | 0.40 m | 0.50 m |
|---|---|---|
| 起跳速度 | 2.22 m/s | 2.62 m/s |
| 蹬地加速度 / 时长 | 14.8 m/s² / 0.150 s | 20.7 m/s² / 0.127 s |
| 腾空 | 0.46 s | 0.545 s |
| 触地速度 / 缓冲减速度 | −2.29 m/s / 18.6 m/s² | −2.72 m/s / 26.2 m/s² |
| 准静态力矩峰值（平台 32 N·m） | 18.6 N·m，利用率 0.58 | 19.7 N·m，利用率 0.72 |
| 关节转速峰值（空载 16.76 rad/s） | 10.9 rad/s | 12.9 rad/s |

主动杆夹角在机械行程内，没有超出包络的帧（`.scratch/jump_ref/check_feasibility.py`）。准静态检查忽略腿部惯量，
只作参考；起跳和触地瞬间腿伸缩速度不连续的已知运动学瑕疵在 0.50 m 上更大（|a_z| 单帧峰值 138 m/s²）。

## 预期与观察点

- 0.50 m 的蹬地更短、更猛，落地冲击更大；看 `Jump/height_err_abs_active` 与 `mimic_deviation` 终止是否比 J3 收敛慢。
- 回放必测：四档 × vx 0 / 0.5 / 1.0 / 1.5（`.scratch/j3_moving_jump.py` 加 0.50）；0.50 m 是否真的跳到、落地是否摔。
- 不应拖累 0.20–0.40 三档的表现（对照 J3 同轮次回放）。

## 验证

- 四个入口参数核对：J1–J3 仍是三条参考、包络上界 0.40；J4 为四条、0.50。
- 功能检查（`.scratch/j4_check.py`，CPU 32 env）：actor 54 维、critic 113 维；四条参考的 RSI 写入后与参考的机身高度差
  ≤ 4.0 cm、腿长差 ≤ 3.2 cm（低于 0.12 m 终止阈值）；400 步触发 16 / 16 / 12 / 19 次，奖励与观测全有限。
- `tests.test_onnx_metadata` + `tests.test_flat_baseline` 通过；CPU smoke（1 env、5 轮）通过，导出的 ONNX 被 runtime
  加载后有四条参考，`trigger_jump(0.5)` 选中 0.50 m 轨（时长 2.02 s）。

## 启动记录

代码 commit `6f558da`，nulltask1 独立 worktree `/workspace/se3-worktrees/j4-6f558da`（`.venv` 软链主 checkout，uv.lock 一致）；
J3 为此在 1396 轮停止。

- 启动时间：2026-10-01 18:48（Pod 时区），六卡 × 8192、5000 轮、每 200 轮保存、seed 42，从头训，在线 W&B（项目 `SE3-WheelLegged-Jump-Mimic`）。
- run：`2026-10-01_18-48-34_jump-J4-mimic-vx15-ref050-seed42-6x8192-5k`，
  W&B [6upe0z1l](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Jump-Mimic/runs/6upe0z1l)，PID/PGID `751429`，
  state `/workspace/.se3-training-state/nulltask1/20261001T104827Z-02527`。
- 首轮核验：第 40 轮无 Traceback、无 nefc overflow，1.37 s/轮，六卡 6.1–6.5 GB / 69–80%；`Jump/active_rate` 0.24，
  `mimic_deviation` 终止每轮约 57 次（与 J3 开局同量级）。
