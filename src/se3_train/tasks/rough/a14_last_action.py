"""A14：在 A13 之上只改一件事——reset 帧的 `last_actions` 不再恒为 0。

单变量对照的另一端就是正在跑的 `rough-A13b-floorfix-seed42-*`，env 其余逐位相同。

## 为什么动这一项

2026-09-10 在 A13b model_1600（四卡）上，平地、同一条指令 vx 2.0 / h 0.38，只换进入路径：

  进入路径                              末 2 s 实速
  reset 后直接给指令                        2.15
  先站 5 s（vx=0, h=0.38）再切              0.01
  先蹲 5 s（h=0.22）再切                   -0.03
  先给 vx=2.4 冻住再退回 2.0                -0.18
  vx 三秒内 0→2.0 平滑升                   -0.08

策略是无历史 MLP，路径只能经由当前观测起作用；观测里带状态的只有关节姿态和 `last_actions`。
把机器人先站进不动点再分别只清一半：

  只清 `last_actions`（物理状态保留）        0.01
  只清关节姿态（`last_actions` 保留）        0.04
  两个都清（= 按一下 reset）                2.15

**两把锁各自都够锁死**。静止时策略输出一个近似常数的动作（六维 std 仅 0.03–0.10，而走路时
是 0.19–0.65），该动作原样回到下一帧观测，网络再把它映回自己——自洽的静止不动点。而训练里
`ActionManager.reset()` 把 `_action` 清零，"reset 帧 last_actions 全 0"因此是个恒成立的
信号，策略可以拿它当模式开关用。本组把它打掉。

注入范围 3.5 取自实测原始动作跨度（走路 [-3.62, 3.40]）。只改策略看到的值，不碰
`_action`/`_prev_action`——那两个参与动作平滑惩罚的差分。

## 这一组预期解决不了什么

上表说明关节姿态是**另一把独立的锁**：单独清 `last_actions` 仍然只有 0.01。所以本组预期
只拆掉一半，姿态那半要靠 `reset_joints` 把"已经站定"的姿态放进起始分布（见 A15）。
先单独跑 A14 是为了量出这一把锁各值多少。
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from se3_train.rl_cfg import RslRlOnPolicyRunnerCfg

from .a13_tuned import A13_ENV_KWARGS, a13_rl_cfg
from .env_cfg import env_cfg

A14_RESET_LAST_ACTION_RANGE = 3.5
"""逐维均匀采样区间 U(-3.5, 3.5)，覆盖实测原始动作跨度。"""

A14_RESET_LAST_ACTION_PROB = 1.0
"""每个 env 都注入。留 <1 的余地是为了保住一部分"全 0"样本，本组先不用。"""


def a14_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """A13 的 env 加 reset 帧 last_actions 随机化，其余逐位不变。"""
    return env_cfg(
        play=play,
        **A13_ENV_KWARGS,
        reset_last_action_range=A14_RESET_LAST_ACTION_RANGE,
        reset_last_action_prob=A14_RESET_LAST_ACTION_PROB,
    )


def a14_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO 与 A13 逐项相同。"""
    return a13_rl_cfg()
