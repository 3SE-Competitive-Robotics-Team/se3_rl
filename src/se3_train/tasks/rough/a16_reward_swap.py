"""A16：把 stairs_up 整列的奖励换成参考实现那 10 项，一项不改；其余五列保持 A15 不变。

对照端是 A15（`rough-A15-pricing-seed42-6x8192-5k`，已跑满 5000 轮）。唯一差异是
stairs_up 列上生效的是哪一套奖励。

## 为什么整列换而不是逐项调

2026-09-11 对照 se3_rl_competiition 的 `cloud-changes` 分支：

  |            | 他们 | 我们 A15 |
  | 奖励项数    | 10   | 25       |
  | 权重跨度    | 0.01 – 1.5 | 0.0002 – 25 |
  | 最重的三项  | 无   | flat_leg_contact -25 / collision -16 / tracking_orientation_l2 -12 |

我们那份奖励在台阶列上的实测账本是「爬 +5.809 / 站 -0.423」，方向正确但由 19 项加权而成，
逐项试验的组合数不可接受。整列替换是一次性判定「这套简单奖励是不是更好」的最小实验。

## 10 项原样（权重与参数一个不改）

   1. tracking_lin_vel           +1.5   sigma=0.245
   2. tracking_ang_vel           +1.0   sigma=4.5
   3. tracking_height            +1.0   sigma=0.1        指数正奖励，用脚下 clearance
   4. climb_progress             +1.0   scale=1.0, min_vx=0.1
   5. step_height_progress       +0.5   max_step=0.05    带跨帧缓存的净爬升
   6. chassis_clearance_penalty  +1.0   sigma=0.1        函数内部取负
   7. orientation                -1.0
   8. leg_alignment_penalty      -0.5   max_fore_aft_offset=0.06, max_penalty=4.0
   9. action_rate                -0.01
  10. is_alive                   +0.1

第 8 项正对 2026-09-11 定位到的死锁构型：卡死时两条腿前后叉开（共模 1.71 弧度），而它罚的
正是左右轮在车身系的前后错位。我们原有的 `joint_mirror` 罚的是 rod1+rod2，实测卡死时 0.28、
比正常爬台阶的 0.34 还小，抓不住这个模态。

## 做法

- A15 原有 25 项全部用 `off_column` 包一层，在 stairs_up 上置零、其余五列逐位不变；
- 新增的 10 项全部用 `on_column` 包一层，只在 stairs_up 上生效；
- `step_height_progress` 的跨帧缓存加一个 reset 事件清理。

其余一切（观测、动作、终止、域随机化、课程、指令采样、PPO）与 A15 逐位相同。
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from se3_train.rl_cfg import RslRlOnPolicyRunnerCfg

from . import stair_column_rewards as ref
from .a15_pricing import a15_env_cfg, a15_rl_cfg
from .env_cfg import ROUGH_REWARD_TERRAIN_TYPE_NAMES

_CMD = "velocity_height"

A16_STAIR_COLUMN = ROUGH_REWARD_TERRAIN_TYPE_NAMES
"""被替换的列，与 A15 判定「台阶列」用的是同一个名字元组。"""

A16_REFERENCE_REWARDS: dict[str, tuple[float, dict]] = {
    "ref_tracking_lin_vel": (1.5, {"inner": ref.ref_tracking_lin_vel,
                                   "params": {"command_name": _CMD, "sigma": 0.245}}),
    "ref_tracking_ang_vel": (1.0, {"inner": ref.ref_tracking_ang_vel,
                                   "params": {"command_name": _CMD, "sigma": 4.5}}),
    "ref_tracking_height": (1.0, {"inner": ref.ref_tracking_height,
                                  "params": {"command_name": _CMD, "sigma": 0.1}}),
    "ref_climb_progress": (1.0, {"inner": ref.ref_climb_progress,
                                 "params": {"command_name": _CMD, "scale": 1.0, "min_vx": 0.1}}),
    "ref_step_height_progress": (0.5, {"inner": ref.ref_step_height_progress,
                                       "params": {"command_name": _CMD, "scale": 1.0,
                                                  "min_vx": 0.1, "max_step": 0.05}}),
    "ref_chassis_clearance_penalty": (1.0, {"inner": ref.ref_chassis_clearance_penalty,
                                            "params": {"weight": 1.0, "sigma": 0.1}}),
    "ref_orientation": (-1.0, {"inner": ref.ref_tracking_orientation_l2,
                               "params": {"command_name": _CMD}}),
    "ref_leg_alignment_penalty": (-0.5, {"inner": ref.ref_leg_alignment_penalty,
                                         "params": {"min_lateral_distance": 0.40,
                                                    "max_lateral_distance": 0.46,
                                                    "max_fore_aft_offset": 0.06,
                                                    "lateral_scale": 0.04,
                                                    "fore_aft_scale": 0.03,
                                                    "fore_aft_weight": 1.5,
                                                    "max_penalty": 4.0}}),
    "ref_action_rate": (-0.01, {"inner": ref.ref_action_rate, "params": {}}),
    "ref_is_alive": (0.1, {"inner": ref.ref_is_alive, "params": {}}),
}
"""参考实现的 10 项：名称 -> (权重, on_column 的参数)。权重与 params 与来源逐位相同。"""


def a16_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """A15 的 env，只把 stairs_up 列的奖励整列换成参考实现那 10 项。"""
    cfg = a15_env_cfg(play=play)
    column = tuple(A16_STAIR_COLUMN)

    # 1) A15 原有的每一项都在 stairs_up 上置零，其余五列逐位不变。
    swapped = {}
    for name, term in cfg.rewards.items():
        swapped[name] = RewardTermCfg(
            func=ref.off_column,
            weight=term.weight,
            params={
                "inner": term.func,
                "params": dict(term.params or {}),
                "terrain_type_names": column,
            },
        )

    # 2) 参考实现的 10 项只在 stairs_up 上生效。
    for name, (weight, gate_params) in A16_REFERENCE_REWARDS.items():
        swapped[name] = RewardTermCfg(
            func=ref.on_column,
            weight=weight,
            params={**gate_params, "terrain_type_names": column},
        )
    cfg.rewards = swapped

    # 3) step_height_progress 的跨帧缓存要在 reset 时清理，否则首帧会用上一局的脏值。
    cfg.events["reset_ref_step_height_cache"] = EventTermCfg(
        func=ref.reset_step_height_cache, mode="reset"
    )
    return cfg


def a16_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO 与 A15 逐项相同。"""
    return a15_rl_cfg()
