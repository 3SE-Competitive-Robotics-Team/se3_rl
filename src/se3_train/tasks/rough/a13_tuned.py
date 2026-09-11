"""A13：按 A12 十组单因素消融的结论拼出的配置——有利项全开、有害项全关。

依据是 `SE3-WheelLegged-Rough-A12Abl-*` 十组（1×8192、seed42、共同窗口 3400–3600 轮），
判据用 `Rough/stair_supported_steps`（实际踩住了几级踏面），比 `terrain_levels` 直接——
后者在 6 附近饱和，平地列也停在 5.9，已经没有分辨力。

  组  撤掉的因素          踩住踏面   等级   清块    结论
  A0  完整 A12            0.368   5.84   8.48   基准
  A9  关闭 AMP            0.528   6.12  13.57   **有害**：关掉后全面变好
  A6  恢复能耗原价          0.352   5.96   8.65   无差别
  A2  恢复竖直速度项         0.296   5.94   6.86   小帮助
  A4  恢复 yaw 奖励        0.226   5.38   4.60   有帮助
  A1  恢复窄速度核          0.202   5.13   3.32   有帮助
  A7  删前进进度奖励         0.002   1.82   1.93   **必需**
  A5  删速度违令罚          0.001   1.95   1.95   **必需**
  A3  恢复高度惩罚          0.003   1.50   1.93   **必需**
  A8  删双轮支撑奖励          ≈0    1.52   1.95   **必需**

四个「必需」组的失败形态一致：踩面归零、等级掉到 1.5–2.05、清块 1.93–1.95、
`is_alive` 0.886–0.889（episode 全跑满从不清块），但台阶跟踪分仍有 0.81——
机器人在坑底那 2 m 平台上来回跑拿跟踪分，就是不上台阶。

相对 A12 的三处改动：

1. **关闭 AMP**（A9）。风格奖励在台阶列贡献 +9.1/秒、占该列奖励信号约 90%，
   而判别器只把专家/策略分到 ±0.24（LSGAN 目标 ±1）、`style_reward` 长期贴 0.6——
   它给的是一笔近似常数的存活奖金，把任务奖励的梯度淹掉了。env 侧仍保留 amp/amp_mask
   两个观测组（`amp_enabled=True`），只把 `amp_cfg` 置空，与 A12 的 env 逐位可比。
2. **能耗退回原价**（A6，`energy_penalty_scale=1.0`）。撤掉它踩面 −4.6%、等级 +2.1%，
   全在噪声里；退回原价等于少一处偏离 Flat 基线。
3. **`stair_height_range` 0.35–0.38 → 0.20–0.38**。这一项不在十组消融里，来自 sim2x 侧
   的定位：A12 的策略在「高站姿 + 静止」下起不了步——平地上给 vx 2.4，高度指令 ≤0.30 时
   实速 2.40–2.46，≥0.34 时掉到 0.03–0.07，而轮法向力两边都是 125 N（牵引条件相同），
   轮速目标在 ±1 之间抖、没有持续正输出，是学出来的行为而非物理限制。A7 的 checkpoint
   在 0.38 静止起步仍有 2.33 m/s，说明是 A8 把台阶列钉死在 0.35–0.38 之后训出来的：
   那个区间下每个 episode 都从「静止 + 够不着的高速指令」开局，加速拿不到跟踪奖励
   （核死了）却要吃动作罚，最优解就是不动。回到全区间后每个 episode 有相当比例
   从矮站姿开局，能起步、能拿到奖励，那块观测区域就不会再被训死。
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from se3_train.rl_cfg import RslRlOnPolicyRunnerCfg

from .env_cfg import env_cfg
from .rl_cfg import amp_rl_cfg

A13_STAIR_HEIGHT_RANGE = (0.20, 0.38)
"""台阶列机身高度指令范围；与 Flat 基线的 _STANDING_HEIGHT_RANGE 相同。"""

A13_ENERGY_PENALTY_SCALE = 1.0
"""能耗三项退回 Flat 原价（A6 显示折价无差别）。"""


A13_ENV_KWARGS: dict = {
    # 观测组保留，便于与 A12 逐位对照；AMP 奖励侧由 rl_cfg 关掉。
    "amp_enabled": True,
    "terrain_vz_weight": 0.0,
    "zero_base_height_on_terrain": True,
    "zero_tracking_ang_vel_on_terrain": True,
    "command_velocity_error_weight": -2.0,
    "energy_penalty_scale": A13_ENERGY_PENALTY_SCALE,
    "stair_height_range": A13_STAIR_HEIGHT_RANGE,
}
"""A13 相对 Flat 基线的全部 env 改动。后续单因素组（A14…）在此之上只加自己那一个 key，
单变量关系由结构保证，不靠人对着抄。"""


def a13_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """有利项全开：进度/支撑奖励、速度违令罚、台阶列置零高度罚、宽速度核、关 yaw 工资、关 vz 项。"""
    return env_cfg(play=play, **A13_ENV_KWARGS)


def a13_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO 与 A12 逐项相同，只把 amp_cfg 置空（A9 证明 AMP 有害）。"""
    cfg = amp_rl_cfg()
    cfg.algorithm.amp_cfg = None
    return cfg
