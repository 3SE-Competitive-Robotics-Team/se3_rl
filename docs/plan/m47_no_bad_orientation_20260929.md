# M47：M39 + 删 bad_orientation 终止（2026-09-29，用户定）

## 背景

用户决定不再调奖励，从近期对照里选最好的一条。终选评测（`.scratch/final_pick/`，model_4999，sim2x 部署链路确定性回放）：

| | 金字塔上台阶通关（24 工况） | 卡住 | 平均通关用时 | 前进 1.0 + yaw 1.0 实测 yaw | 前进 2.0 + yaw 2.0 实测 yaw | 静站俯仰峰峰 |
|---|---|---|---|---|---|---|
| M39 | 24/24 | 0 | 5.3 s | 1.11 | 2.00 | 7.2° |
| M42 | 23/24 | 1 | 7.5 s | 0.26 | 0.33 | 2.1° |
| M38 | 21/24 | 3 | 5.5 s | 0.80 | 1.59 | 3.0° |
| M44 | 15/24 | 9 | 6.5 s | 0.61 | 2.03 | 0.8° |
| M43 | 12/24 | 12 | 7.9 s | 0.57 | 1.88 | ≈0 |

选定 M39（M38 默认 + action_rate −0.48 → −0.10）。M38/M39/M42 通关后从金字塔另一侧下台阶都摔约 10 次，是共同短板。
随后用户要求在 M39 上删掉 bad_orientation 终止重新训练。

## 改动（相对 M39 唯一差异）

`env_cfg(bad_orientation_termination=False)`：删除 `bad_orientation`（机身倾角 > 30° 连续 100 步即终止，终止时吃 `fall_penalty` −500）。
其余终止项不变：`time_out`、`catastrophic_state`（数值发散）、`leg_contact`（terminate=False，只记日志）、
`terrain_edge_reached`、`out_of_terrain_bounds`。奖励、课程、动作、执行链与 M39 逐项相同。

## 注意

- 删后失败终止只剩数值发散时的 `catastrophic_state`，`fall_penalty`（−500）实际上几乎不再触发；倒地 env 躺到 20 s 超时，
  只由 `tracking_orientation_l2`（−12）、`flat_leg_contact`（−25）、`upward`（+1）等按秒计价，同时丢掉跟踪与存活收益。
- 倒地的 env 会占着 batch 到超时，有效样本比例下降；看 `Episode_Termination/time_out` 与平均回合长度。
- 策略可能学会从倒地状态自己爬起来，也可能学会"倒着也行"；评测看摔倒后是否恢复、`fall` 次数与恢复时间。

## run 设置

`nulltask-5c45cdd89b-whtws` 七卡 × 8192、5000 轮、保存 200、seed 42，在线 W&B project `SE3-WheelLegged-Rough`，与 M39 同批量可直接按轮次对比。

## 判据

- 上台阶：climb_map（`.scratch/final_pick/eval_all.py`）对 M39-4999（24/24）。
- 下台阶侧摔倒次数与摔后恢复（M39 为 11）。
- 平地跟踪与静站抖动对 M39。

## 启动记录

（启动后补）
