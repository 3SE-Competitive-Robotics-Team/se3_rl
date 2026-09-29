# M46：M45 + 台阶类列高度参考改两轮支撑面，从 M45-600 续训（2026-09-29，用户定）

## 背景

M45（复旦式高度对、全列窗口口径）到 1132 轮：平地行驶高度误差从 M42 的 −2.9 cm 降到 +0.1 cm，但台阶列窗口口径
误差随等级变大（带符号均值 −4.6 → −6.3 cm），尖核奖励从 0.49 掉到 0.35。model_1000 分解回放
（`.scratch/m45_height/decompose.py`、`ref_series.py`，中位数）：

| | 窗口口径 | 两轮支撑面口径 | 机身离轮轴 |
|---|---|---|---|
| 平地 0.38 静站（基准） | −0.3 cm | −0.3 cm | 31.8 cm |
| 台阶 6 级 0.38/1.0 | −5.8 cm | −1.0 cm | 30.1 cm |
| 台阶 9 级 0.38/1.0 | −10.5 cm | −3.1 cm | 27.2 cm |

窗口前后各伸 0.5 m，轮子顶在立面上时前半截已落在上一阶，参考被抬高约半阶；机身并没有真的蹲下。
同一轨迹按复旦式高度对算每秒损失（9 级 0.38/1.0）：窗口 1.82（立面前 1.07、着陆后 0.5 s 0.66），
支撑面 0.98（立面前 0.31、着陆后 0.58）。着陆后那段是真实的"机身还没跟上新踏面"，两种口径都收。

两轮不在同一阶（回放里 6–13% 的时间）时，支撑面取两阶平均：分腿时误差中位 −4…−10 cm、每秒损失约为同面时的
1.5–2 倍，对分腿收温和的钱，方向与"腿一收双轮同抬"一致；不取较高阶（会重演 M21 的拖着不抬轮），也不取较低阶。

## 改动（相对 M45 唯一差异：台阶类列的高度参考）

`env_cfg(fudan_height_stair_reference="support")`：`base_height_fudan` / `base_height_fudan_enhance` 在
`ROUGH_BASE_HEIGHT_SUPPORT_COLUMNS`（stairs_up、stairs_two_step_up）上把地面参考换成 `stair_reward_height`
（两轮轮心正下方各 4 根竖直射线的有效命中均值），其余列仍是 77 点窗口均值。公式、权重、无死区、不裁剪都不变。
新增日志 `Rough/base_height_err_reward_stairs_signed`（计酬口径）；`Rough/base_height_err_window_stairs(_signed)`
仍按窗口口径记，与 M45 可比。

## 续训

- 用户要求从 M45 约 500 轮续、平地热身不重复。M45 每 200 轮存一次，没有 model_500；取 model_600
  （已过前 500 轮纯平地阶段，换列过渡进度 20%）。
- mjlab 完整续训恢复网络、优化器、轮次（600）以及 checkpoint `infos.env_state` 里的 `common_step_counter`（14424 = 601×24），
  `flat_warmup` 按该计数算轮次，所以热身从 20% 接着走，不会重来；其他按步数计时的调度同理。
- 不恢复的课程状态：地形等级（M45-600 时台阶等级约 1.5）与速度课程范围从初值重新推进。
- mjlab 只在本实验目录找 checkpoint：Pod 上把 M45 的 model_600.pt 复制到
  `logs/rsl_rl/SE3-WheelLegged-Rough-Exp-FudanActuationHeightSupport/m45-model600-src/`，启动器 `-L m45-model600-src -K model_600.pt`，
  `-i 4400` 跑到 5000 轮。
- 本地 CPU 续训 smoke：从 600 轮接着跑到 605，无 Traceback（本地需把 checkpoint 转成 CPU 张量，Pod 上不需要）。
- critic 是按 M45 的台阶列窗口口径学的，换参考后台阶列价值会有一段重新拟合。

## run 设置

`nulltask-5c45cdd89b-whtws` 七卡 × 8192，从 600 续到 5000 轮，保存 200、seed 42，在线 W&B project `SE3-WheelLegged-Rough`。
M45 在 1365 轮按用户指令停止。

## 判据

- 台阶列：`Rough/base_height_err_reward_stairs_signed` 回到 ±2 cm 内；窗口口径误差仅作参考；尖核奖励不再随等级下滑。
- 抬轮：riser_retry_diag 看抬轮时机（M21 支撑面口径曾出现抬轮延迟 0.3 s）与分腿比例；climb_map 对 M42-4999。
- 课程：stairs_up / stairs_down 等级对 M45 同轮次。

## 启动记录

（启动后补）
