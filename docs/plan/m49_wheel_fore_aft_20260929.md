# M49：对称只罚水平分量（删 joint_mirror、加轮心前后错位 Δx²），从 M48 续训（2026-09-29，用户定）

## 背景

M48 把 joint_mirror 从 −0.179 加到 −5。joint_mirror 罚前杆角与主动杆角的左右差，两个角经四连杆同时影响轮子的前后与上下，
所以它连左右腿长差也罚，和 roll 指令（±0.1 rad，需要左右腿长不等）顶牛。用户要求只罚水平分量、不罚竖直分量。

竖直分量已有别的项计价：平地上左右腿长不等会让机身侧倾，由 `tracking_orientation_l2`（−12）按 roll 指令罚；
需要 roll 时同一项反过来要求腿长不等。所以只需在水平方向加约束。

## 改动（相对 M48）

- 删 `joint_mirror`（`joint_mirror_weight=0.0` = 删项）。
- 加 `wheel_fore_aft_offset`：左右轮心在机身系里的前后错位 Δx = x左 − x右，Δx² × 直立门控，全列生效、无死区，w −50/m²。
  机身系而非世界系，俯仰不产生假误差；roll 时腿长差落在机身 z，不混进 x。
- 同式项 M18–M36 用过（M37 删除）。M18 全列无死区时策略学成"两轮齐平"、台阶上只剩双轮同抬，当时当副作用，
  M19 改成只罚平地列 + 10 cm 死区；现在的上台阶目标就是"腿一收双轮同抬"，所以这次全列生效。
- 新增日志：`Rough/wheel_dx_abs`、`Rough/wheel_dx_abs_stairs`、`Rough/wheel_dz_body_abs`（机身系左右腿长差）。

| Δx | 5 cm | 10 cm | 20 cm |
|---|---|---|---|
| 每秒扣分 | 0.13 | 0.5 | 2.0 |

22° 摆角差约对应 Δx 11 cm、每秒约 0.6，与 M48 的 joint_mirror −5 在同样不对称下（约 0.76）同量级。

## 续训

从 M48 最新 checkpoint 完整续训（用户要求不浪费时间）：checkpoint 复制到 M49 实验目录的 `m48-src/`，
启动器 `-L m48-src -K model_N.pt -i (5000−N)`；网络、优化器、轮次与 `common_step_counter` 随 checkpoint 恢复，
地形等级与速度课程从初值重新推进。critic 是按 M48 奖励学的，换项后有一段重新拟合。

## 风险

- 上台阶：全列罚 Δx 可能削弱一先一后的爬法，通关图对比 M39-4999（24/24）。
- roll 跟踪：目前没有专门评测，需加平地 roll ±0.1 回放，确认实测 roll 跟得上、Δx 没被放大。

## run 设置

`nulltask-5c45cdd89b-whtws` 七卡 × 8192，续到 5000 轮，保存 200、seed 42，在线 W&B project `SE3-WheelLegged-Rough`。

## 启动记录

代码 commit `d5db384`，whtws 仓库由 be494d1 快进到同一 commit；41 项 rough 测试通过，本地入口 CPU smoke 5 轮通过。
配置核验：相对 M48 删 `joint_mirror`、加 `wheel_fore_aft_offset`，其余奖励与终止项逐项相同。

- M48 在 1354 轮按用户指令停止，最新 checkpoint model_1200.pt（sha256 前缀 `3679ff244901647a`）复制到 `m48-src/`。
- 启动时间：2026-09-29 20:22（Pod 时区），七卡 × 8192，从 1200 续 3800 轮到 5000，保存 200、seed 42，在线 W&B。
- run：`2026-09-29_12-22-37_rough-M49-wheeldx50-from-m48-1200-seed42-7x8192`，
  W&B [fxig1m4z](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/fxig1m4z)，PID/PGID `2711263`，
  state `/workspace/.se3-training-state/whtws/20260929T122230Z`。
- 首轮核验：日志加载 `m48-src/model_1200.pt`，迭代从 1200 接着走（1214 轮 3.46 s/轮），无 Traceback、无 nefc overflow，
  七卡各 13.0–13.5 GB、利用率 77–87%；奖励表 `wheel_fore_aft_offset` −50.0、无 `joint_mirror`。
- 续训开局：`Rough/wheel_dx_abs` 2.4 cm、`Rough/wheel_dz_body_abs` 1.0 cm（M48 的 joint_mirror −5 已把两轮压齐）；
  回合长 200 步是地形等级重置与 critic 重拟合的开局期。
