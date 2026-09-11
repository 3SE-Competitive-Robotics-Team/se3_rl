"""A15：把台阶列早就改对的定价，扩到被漏掉的其余五列。

对照端是 `rough-A13b-floorfix-seed42-4x8192`（A13 的 env），只差本文件列出的四个数值。

## 为什么改

A13b model_1600 在 **flat 列**上的实测奖励账本（96 env，53 个在走 / 24 个卡住，
指令 vx 2.0 / h 0.38，`RewardManager._step_reward`，单位每秒）：

  奖励项                        走 1.64 m/s   卡 0.11 m/s        差
  flat_base_height                  -4.625       -0.473     -4.152
  tracking_orientation_l2           -0.743       -0.019     -0.724
  bad_tilt                          -0.572       -0.005     -0.567
  tracking_ang_vel                  +2.711       +2.328     +0.383
  action_rate                       -0.174       -0.035     -0.139
  tracking_lin_vel                  +0.115        0.000     +0.115
  is_alive                          +1.000       +1.000          0
  合计                              -2.932       +2.407     -5.339

**站着不动净赚 2.407/秒，走路净亏 2.932/秒。** 所谓「起步死锁」不是 bug：站着是这个奖励
函数在非台阶列上真正的最优解。双稳态、随站姿升高的速度天花板、只有 3° 的起步基域、训练里
`Rough/base_vx_terrain` 长期 0.245 m/s（指令 1.0–2.4），都是同一个最优解的表现。

根因是「走路必然产生的机身起伏」被罚了三遍。实测走时机身高度误差 0.054 m、站着 0.017 m，
垂直速度 vz ≈ 0.28 m/s：

1. `flat_base_height` 直接罚：-4.0 权重、σ=0.05 的二次罚，差额 4.15/秒，**占缺口 78%**；
2. `tracking_lin_vel` 的核里再罚一次：核是 exp(-(err_x² + vz_weight·vz²)/σ)，平地列
   vz_weight=2.0。实测核衰减 0.284 中 vz 占 0.154，**比速度误差的 0.130 还多**；
3. `tracking_orientation_l2`(-12.0) 与 `bad_tilt` 再罚一次姿态，合计 1.29/秒。

而 `tracking_lin_vel` 满分只有 4.0/秒，光高度罚的差额就 4.15/秒——**即使速度跟踪做到满分，
非台阶列上的高速行走在这个奖励函数里也不可能划算。**

对照 A0（开着 AMP）跑同一份账本：走 -3.885、卡 +1.879，缺口 5.77 **比 A13b 还大**，
但它照样走（干净 reset 下 78/96 在走，速度网格全格跟踪）。两者 RewardManager 侧几乎一样，
唯一区别是 AMP 那笔发在管理器之外的 +9.1/秒。**AMP 一直是压住这个反向激励的配重**；
A12 消融按 `stair_supported_steps` 判 AMP 有害、A13 把它拿掉，等于抽掉配重，于是策略收敛到
奖励函数真正的最优解。所以修法是把价格改对，不是把配重加回来——AMP 留待价格改对后再单独评估。

## 改了什么，各收回多少

台阶列这四条早就是对的（A6–A10 一路调出来的），A15 只是扩到其余五列：

  flat_base_height  σ 0.05 → 0.10（非台阶列）               +3.11/秒
  tracking_lin_vel  σ_move 0.08 → 0.5（台阶列外的五列）      合计
  tracking_lin_vel  vz_weight 2.0 → 0.0（平地列）            +2.97/秒
  command_velocity_error 的列门控放开到全部六列             ≈ +1.2/秒（只打在"不动"那边）

合计约 +7.3，缺口 5.34 —— 走路净赚约 2/秒，符号翻过来。每一格都是从上面那份账本按各项的
解析形式算出来的，不是估的。

**为什么不是单变量**：最大的单个旋钮（高度罚 σ）只有 +3.11，关不上 5.34 的缺口；前两项一起
5.26 刚好打平，必须加上违令罚才有余量。四项是同一处错配（非台阶列的参数是为 Flat 基线
±1.0 m/s 的指令范围调的，现在被课程推到 2.4 m/s）的四个面，拆开跑没有单独可解释的中间态。
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from se3_train.rl_cfg import RslRlOnPolicyRunnerCfg

from .a13_tuned import A13_ENV_KWARGS, a13_rl_cfg
from .env_cfg import ROUGH_ALL_TERRAIN_TYPE_NAMES, env_cfg

A15_BASE_HEIGHT_SIGMA = 0.10
"""非台阶列机身高度罚的 σ。0.05 时 5.4 cm 的行走起伏就吃 -4.6/秒；0.10 降到 -1.16/秒。"""

A15_OFF_STAIR_TRACKING_SIGMA_MOVE = 0.5
"""台阶列外五列的速度跟踪运动核分母。0.08 是为 ±1.0 m/s 指令调的，2.4 m/s 下核是死的。"""

A15_FLAT_VZ_WEIGHT = 0.0
"""平地列核里的 vz 权重。其余列早已由 terrain_vz_weight=0 关掉（A7），平地列被漏下。"""

A15_COMMAND_VELOCITY_ERROR_TERRAIN_NAMES = ROUGH_ALL_TERRAIN_TYPE_NAMES
"""违令罚放开到全部六列。这是唯一不对称的一项：只打在"给了指令却不动"那边。"""


def a15_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """A13 的 env，只改非台阶列的四个定价数值。"""
    return env_cfg(
        play=play,
        **A13_ENV_KWARGS,
        base_height_sigma=A15_BASE_HEIGHT_SIGMA,
        off_stair_tracking_sigma_move=A15_OFF_STAIR_TRACKING_SIGMA_MOVE,
        flat_vz_weight=A15_FLAT_VZ_WEIGHT,
        command_velocity_error_terrain_names=A15_COMMAND_VELOCITY_ERROR_TERRAIN_NAMES,
    )


def a15_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO 与 A13 逐项相同（AMP 仍关闭）。"""
    return a13_rl_cfg()
