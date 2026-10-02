# M39：M38 默认 + action_rate 权重 −0.48 → −0.10（2026-09-28，用户定）

## 背景

M38（已定为默认，docs/plan/m38_upward_20260928.md）台阶列每秒账本里 action_rate 实付 −0.85，是台阶列第二大罚项
（第一是高度罚/orientation 合计）。M34-7800 反事实账本显示它在 0.73–0.89 m/s 之间不随速度变（−0.55…−0.58/s），
即它罚的是动作抖动而不是爬升本身；Flat 线 D 系列曾测得收敛后 action_rate 的 72–81% 是探索噪声地板。
用户决定把权重减到 −0.10 看一次。

## 改动（相对 M38 默认唯一差异）

任务入口 `SE3-WheelLegged-Rough-Exp-ActionRate010`：`env_cfg(action_rate_weight=-0.10)`，只改 `action_rate` 项的权重，
函数与 leg/wheel 分量参数不动；`action_smoothness`（−0.12）不动。配置逐项对比：只有 `action_rate.weight −0.48 → −0.1`。

## run 设置

`nulltask-5c45cdd89b-whtws` 七卡 × 8192、5000 轮、保存 200、seed 42，在线 W&B project `SE3-WheelLegged-Rough`，与 M38 同批量。

## 判据

- 动作噪声：`Policy/mean_std` 与 `Locomotion/max_abs_action`（动作罚减小后 σ 平衡点会上移，见记忆"σ 平衡点"）；
  确定性回放里腿关节的极限环幅值（`.scratch/m15_diag` 站立/行进峰峰）不显著变大。
- 台阶：`Rough/base_vx_stairs`、`Curriculum/terrain_levels/stairs_up` 不低于 M38 同期（0.72–0.75 / 5.4）；
  riser_retry_diag 的首撞俯仰、首撞到过沿时间；climb_map 通关格与摔倒格对 M38-4999（3 卡 / 20 摔）。
- 平地：`.scratch/m15_diag/sweep.py` 低速段实速与站立抖动。
- 终止：`Episode_Termination/catastrophic_state` 尾部（M38 4000 轮后升到 0.11）。

## 启动记录

代码 commit `a724b49`，whtws 仓库由 02946a4 快进到同一 commit；本地入口 CPU smoke 5 轮通过（action_rate −0.1）；启动器 `--dry-run` 通过。

- 启动时间：2026-09-28 12:42（Pod 时区），七卡 `--gpu-ids all` × 8192、5000 轮、保存 200、seed 42，在线 W&B。
- run：`2026-09-28_04-42-40_rough-M39-actionrate010-seed42-7x8192-5k`，
  W&B [uouikynn](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/uouikynn)，PID/PGID `2088093`，
  state `/workspace/.se3-training-state/whtws/20260928T044233Z`。
- 首轮核验：第 29 轮在迭代、无 Traceback、无 nefc overflow，3.13 s/轮，七卡 13.0–13.5 GB、利用率 84–87%；
  远端 `params/env.yaml`：`action_rate.weight −0.1`、`upward.weight 1.0`。

## 结论（2026-09-28，跑满 5000 轮）

训练进程正常结束（`model_4999`）。入口机 sshd 在约 3575 轮时再次拒绝连接、两条 boring 隧道被关，W&B 中途标 crashed；
网关任务自动重开后进程把剩余历史补传完（末段以 train.log 为准）。

train.log 末段（与 M38 同轮次）：

| 轮次 | stairs_up | 台阶列 vx | action_rate /s | 平滑罚 /s | 高度罚 /s | 倾角 ° | 灾难终止 | 穿块终止 |
|---|---|---|---|---|---|---|---|---|
| M39 3000 | 5.93 | 0.77 | −0.31 | −0.90 | −0.48 | 9.2 | 0.018 | 14.2 |
| M39 4000 | 6.24 | 0.78 | −0.31 | −0.88 | −0.49 | 9.2 | 0.029 | 14.9 |
| M39 4999 | **6.81** | **0.81** | −0.32 | −0.90 | −0.45 | 8.9 | 0.046 | 15.6 |
| M38 4999 | 5.38 | 0.74 | −0.85 | −0.50 | −0.59 | 8.0 | 0.119 | 15.4 |

σ 0.41 → 4999 轮 max_abs_action 3.1（M38 3.2）。课程一路涨到 6.8（M38 停在 5.4），台阶列 vx 0.81，灾难终止尾部 0.046（M38 0.119）。

`model_4999` 确定性回放（`.scratch/m35_eval/`：`riser_retry_M39-4999.log`、`climb_map_m39_4999.log`、`jitter_M39-4999.json`）对 M38-4999：

- **撞面**：首撞俯仰 −2…+3°（M38 −2…−10°）；首撞到过沿 cmd 1.0 h=0.34 为 0.14–0.20 s、cmd 1.4 为 0.12–0.18 s，回到 M37 的水平；
  h=0.38 cmd 1.0 四级全过、每道 0.2–0.26 s。错位仍在（右轮领先 13–17 cm，与 M38 同量级、比 M39-2200 的 18–27 cm 收回来了），偏航甩动 ≤13°（2200 轮时 48°）。
- **通关图 45 格**：卡 8（M38 3），集中在 h=0.30 的 18/20 cm（20 cm 五个速度全卡，M38 只卡 h=0.38 低速）；通关后落差摔倒 15（M38 20）；通关格均时 5.2 s（M38 5.4 s）。
  短板从 M38 的高姿态低速换成了低姿态高阶，h=0.34/0.38 全部通关。
- **平地静站**：俯仰 rms 1.6–2.0°、腿峰峰 ≤9°（h=0.38 时右髋 25°），动作差 rms 0.018–0.022，与 M38 相当，无抖动问题；
  行进段 action_diff_rms 0.27–0.51（M38 0.30–0.34），动作确实更"野"。

判断：action_rate −0.10 在课程、速度、撞面姿态、尾部灾难终止上都优于 M38，σ 升到 0.41 没有带来站立抖动；代价是 h=0.30 上 20 cm 台阶全卡（腿收不够）和行进动作变野。
下一条候选：平滑罚减半（−0.06）或轮分量定价 2.0 → 1.0；h=0.30 高阶卡住要在 sim2x 里看一下是机身刮台阶沿还是腿行程不够。
