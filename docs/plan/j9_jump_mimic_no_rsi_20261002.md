# J9：J7 去掉 RSI（2026-10-02，用户定）

## 动机

J7 的 50% 回合从参考随机时刻开始（RSI）。合入 rough 时 RSI 要在地形列上写空中状态，与地形课程、台阶列、出生朝向等逻辑交织。
用户定：先看不带 RSI 的 J7 能否学会；能学会，后续训练就不带 RSI。

## 改动（单变量）

`SE3-WheelLegged-Jump-Mimic-Exp-J9` = J7 + `env_cfg(rsi_prob=0.0)`：所有回合从站姿开始，跳跃只能由触发进入（站满 1 s 后每秒 0.5 次）。
其余全部照 J7：无下蹲参考（站姿 0.22 m）、actor 54 维（含参考帧）、critic 含参考帧与参考时钟、四项模仿奖励、偏离终止 0.12 m、
vx ±1.5、执行链与 PPO。

## 判据

与 J7（qar24d7n）同轮次对比偏离终止、回合长度、跳跃中机身高度误差、模仿奖励；跑完用 `.scratch/j4_moving_jump.py` 与
`.scratch/liftoff_latency.py` 回放，看四档高度 × 四速是否与 J7 持平。

## 验证

J7 / J9 配置对比只差 rsi_prob（0.5 → 0）；CPU smoke（1 env、5 轮）通过。
