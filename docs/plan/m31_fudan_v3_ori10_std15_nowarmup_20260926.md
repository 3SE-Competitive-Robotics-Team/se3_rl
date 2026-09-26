# M31：M30 去掉平地热身（2026-09-26，用户定）

## 背景

M30（M29 + actor 初始 std 1.5）的大探索噪声在前 300 轮就被 PPO 收掉（500 轮 std 0.32，与 M29 相同），而前 500 轮是
平地热身、全体 env 都在平地列，台阶列出现时噪声已经收完。stairs_up 仍停在 1.2–1.7。

## 改动（相对 M30 唯一差异）

任务入口 `SE3-WheelLegged-Rough-Exp-FudanRewardOri10Std15NoWarmup`：`env_cfg(..., flat_warmup=False)` 不注册平地热身
课程，env 从第 0 轮起就在各自地形列的第 0 级（`ROUGH_MAX_INIT_TERRAIN_LEVEL = 0`，上台阶列第 0 级阶高 2 cm），
init_std 1.5 的大噪声期直接落在各地形上。奖励（复旦 v3 + orientation −10 + 格点高度参考）、指令、终止、观测、
域随机化与 PPO 其余超参数都与 M30 相同。

二级台阶门控在没有热身状态时处理全部 env（门控期二级上行 env 放在 stairs_up、二级下行放在 flat）；速度课程只读
平地列的跟踪分，没有热身时平地列约占 15% 的 env，信号仍在。

注意：平地热身是 R7/A7 证据下加的（地形课程 250 轮就把还不会稳走的策略推上 12 cm 台阶、学成原地站着），
本 run 起步阶段可能更乱，前 500 轮重点看是否塌。

## run 设置

nulltask1 六卡 × 8192 envs、8000 轮、保存间隔 200、seed 42，W&B project `SE3-WheelLegged-Rough`，与 M28–M30 同配置。

## 判据

- 上台阶：stairs_up 与 M30 同轮次比（M30：1000 → 1.72、1800 → 1.32、3800 → 1.31），1500 轮前超过 2.5 才算在学爬；
  二级台阶门控（均级 ≥ 5）能否打开。
- 起步：前 500 轮 `Train/mean_episode_length`、平地跟踪 `Rough/tracking_lin_vel_flat`、速度课程能否推满 2.4、
  `Policy/mean_std` 轨迹（M30：300 轮 0.40、500 轮 0.32）。

## 启动记录

（启动后补）
