# M45：M42 + 高度罚换成复旦式高度对（2026-09-29，用户定）

## 背景

M42/M43/M44 跑满 5000 轮后，用户判断 M42 效果最好，但高度罚不够。model_4999 在台阶列第 9 级的确定性回放
（`.scratch/m45_height/height_err_signed.py`，窗口口径，与训练同式）：

| | 指令 | 误差中位 | 偏低占比 | 落在 ±5 cm 死区内 | 现行每秒罚 | 复旦式每秒损失 |
|---|---|---|---|---|---|---|
| M42 | 0.34 / 1.0 | −9.5 cm | 99% | 22% | 0.62 | 1.78 |
| M42 | 0.38 / 1.0 | −9.0 cm | 100% | 25% | 0.93 | 1.84 |
| M43 | 0.38 / 1.0 | −3.3 cm | 81% | 57% | 0.18 | 1.12 |
| M44 | 0.34 / 1.0 | −4.5 cm | 82% | 52% | 0.26 | 1.31 |

平地列能正常走的几条误差约 1 cm。问题在台阶列的形状而不是权重：±5 cm 死区让工作区间免费，死区外核宽 0.10 有界，
−6 cm 附近每多蹲 1 cm 只多付约 0.08/s；只把权重翻倍价格仍远低于 upward 的 +2/s。

## 复旦原式（fudan_rl_wheel_leg 8204e85，plane/logs/wheel_legged/上台阶3/legged_robot.py）

- 高度：`base_height = mean(root_z − measured_heights)`，11×7 点（x ±0.5、y ±0.3 m，随 yaw），每点取相邻三格最低地形高。
- `_reward_base_height`（正权重分支）= 1.5·exp(−1000·e²)，w 1.0。
- `_reward_base_height_enhance` = exp(−e²/0.01) − 1，w 1.0。
- 全地形生效，无死区，不按台阶列关；复旦环境另有逐项每秒 ±1 裁剪（`clip_single_reward = 1`）。

## 改动（相对 M42 唯一差异：高度项）

`env_cfg(height_shape="fudan")`：删 `flat_base_height`，加 `base_height_fudan`（1.5·exp(−1000e²)，w 1）与
`base_height_fudan_enhance`（exp(−e²/0.01) − 1，w 1）。误差 = 机身高度传感器 z − critic 77 点窗口均值 − 高度指令，
全列统一，非跳跃、非恢复 reset 时生效。**按原式、不做逐项裁剪**（用户定），尖核峰值 1.5。

与复旦的已知口径差：机身 z 用本仓库 `base_height_sensor` frame 而非 root；窗口射线取命中上表面、没有"三格取最低"。

相对零误差的每秒损失：1 cm 0.15、2 cm 0.53、3 cm 0.98、5 cm 1.60、9 cm 2.06、15 cm 2.40，封顶 2.5。
原 flat_base_height 台阶列 5 cm 内为 0、9 cm 0.59、15 cm 2.53、封顶 4；其余列二次罚 5 cm 1.0、9 cm 3.24、15 cm 9。

## 注意

- 尖核是正奖励，站在指令高度上每秒 +1.5，相当于一份按高度发的存活奖励；早期倒地提前终止会少拿这份钱。
- 死区去掉后，过沿过渡段的窗口参考会先抬高（M34 账本：罚款 80% 落在过渡段），爬得快的速度税可能回来；看台阶列 vx 与 stairs_up。
- 平地列从二次罚（15 cm 处 9/s）换成有界罚（封顶 2.5/s），大偏差的约束变弱；看 `Rough/base_height_abs_err_window_all`。

## run 设置

`nulltask-5c45cdd89b-whtws` 七卡 × 8192、5000 轮、保存 200、seed 42，在线 W&B project `SE3-WheelLegged-Rough`。

## 判据

- 高度：`Rough/base_height_err_window_stairs_signed`（台阶列带符号）、`Rough/base_height_err_window_all`；回放复用 `height_err_signed.py` 对 M42-4999。
- 台阶：stairs_up / stairs_down 等级、`Rough/base_vx_stairs` 对 M42；riser_retry_diag、climb_map。
- σ 与抖动：`Policy/mean_std` 对 M42（0.64）。

## 启动记录

代码 commit `569aaf2`，whtws 仓库由 d79e094 快进到同一 commit；本地入口 CPU smoke 5 轮通过（无 Traceback），
`tests/test_rough_port.py` 41 项通过。配置核验：相对 M42 删 `flat_base_height`、加两项，其余逐位相同。
M42/M43/M44 已跑满 5000 轮退出，七卡全空。

- 启动时间：2026-09-29 13:06（Pod 时区），七卡 × 8192、5000 轮、保存 200、seed 42，在线 W&B。
- run：`2026-09-29_05-06-36_rough-M45-fudanheight-seed42-7x8192-5k`，
  W&B [gyoug52h](https://wandb.ai/luzhongjin365-se3/SE3-WheelLegged-Rough/runs/gyoug52h)，PID/PGID `2564245`，
  state `/workspace/.se3-training-state/whtws/20260929T050629Z`。
- 首轮核验：第 15 轮在迭代、无 Traceback、无 nefc overflow，3.20 s/轮，七卡各 13.0–13.5 GB、利用率 80–87%；
  远端 `params/env.yaml`：`flat_base_height` 不存在，`base_height_fudan` / `base_height_fudan_enhance` 各 w 1.0（窗口 `critic_height_scan`），
  共 26 项；腿 scale 0.5 / 轮 10、腿 stiffness 20 / damping 1.5、action_rate −0.05、action_smoothness −0.05、M42 的双跟踪核不变。
- 批量是 M42 两卡的 3.5 倍，按同 iteration 对比带批量混杂。
