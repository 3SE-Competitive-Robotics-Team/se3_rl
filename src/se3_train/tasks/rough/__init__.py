"""崎岖地形行走任务（MLP / GRU 两个入口）与台阶定向评测入口。"""

from mjlab.tasks.registry import register_mjlab_task

from se3_train.rl_cfg import bind_task_name
from se3_train.tasks.common import Se3ProfiledOnPolicyRunner

from .env_cfg import ROUGH_M37_DROPPED_REWARDS, env_cfg
from .rl_cfg import gru_rl_cfg, rl_cfg
from .terrains import stair_only_terrains_cfg

TASK_ID = "SE3-WheelLegged-Rough"
# M16：同一份 env_cfg，只把 actor/critic 换成 GRU（rl_cfg.gru_rl_cfg）。
GRU_TASK_ID = "SE3-WheelLegged-Rough-GRU"
STAIR_EVAL_TASK_ID = "SE3-WheelLegged-Rough-StairEval"
# M26/M27（2026-09-25 用户定）：与 M25（TASK_ID，两个开关都关）同一 commit 并发对照，各只翻一个开关。
# 并发 run 共用 Pod 上同一份仓库，中途切 commit 会让在跑的 run 把新 commit 写进 ONNX 溯源，
# 所以用任务入口而不是逐实验 commit 区分。临时入口：对照结束、定下默认值后删除（复现用对应 commit）。
EXP_STAIR_SPEED_CAP_TASK_ID = "SE3-WheelLegged-Rough-Exp-StairSpeedCap"
EXP_HEIGHT_WINDOW_TASK_ID = "SE3-WheelLegged-Rough-Exp-HeightWindow"
EXP_NO_WHEEL_DEADZONE_TASK_ID = "SE3-WheelLegged-Rough-Exp-HeightWindowNoWheelDeadzone"
# M35（2026-09-26 用户定）：M34 + 上台阶列窗口高度罚 ±5 cm 死区，回答"上台阶为什么慢"
# （docs/plan/m35_m36_stair_speed_20260926.md）。2026-09-27 用户判定 M35 优于 M36（进度权重 ×2），
# M36 入口已删除（复现用 commit 5af0d00）。M35 是当前 rough 线的保留配置。同为临时入口。
EXP_STAIR_HEIGHT_DZ5_TASK_ID = "SE3-WheelLegged-Rough-Exp-HeightWindowDz5NoWheelDeadzone"
# M37（2026-09-27 用户定）：M35 + 整组删掉五项重叠定价（ROUGH_M37_DROPPED_REWARDS），
# 见 docs/plan/m37_reward_prune_20260927.md。同为临时入口。
EXP_STAIR_HEIGHT_DZ5_PRUNE_TASK_ID = "SE3-WheelLegged-Rough-Exp-HeightWindowDz5Prune5"
# M38（2026-09-28 用户定）：M37 + upward 权重 1.0（docs/plan/m38_upward_20260928.md）。同为临时入口。
EXP_STAIR_HEIGHT_DZ5_PRUNE_UPWARD_TASK_ID = "SE3-WheelLegged-Rough-Exp-HeightWindowDz5Prune5Upward1"
_M34 = {
    "stair_height_reference": "window",
    "wheel_offset_dead_zone_m": 0.0,
    "wheel_height_diff_dead_zone_m": 0.0,
}
_M37 = {**_M34, "stair_height_dead_zone_m": 0.05, "dropped_rewards": ROUGH_M37_DROPPED_REWARDS}
_EXP_VARIANTS = (
    (EXP_STAIR_SPEED_CAP_TASK_ID, {"stair_speed_cap": True}),
    (EXP_HEIGHT_WINDOW_TASK_ID, {"stair_height_reference": "window"}),
    (EXP_NO_WHEEL_DEADZONE_TASK_ID, _M34),
    (EXP_STAIR_HEIGHT_DZ5_TASK_ID, {**_M34, "stair_height_dead_zone_m": 0.05}),
    (EXP_STAIR_HEIGHT_DZ5_PRUNE_TASK_ID, _M37),
    (EXP_STAIR_HEIGHT_DZ5_PRUNE_UPWARD_TASK_ID, {**_M37, "upward_weight": 1.0}),
)


def register() -> None:
    """注册原始 Rough（MLP）、Rough-GRU、台阶定向评测任务与临时对照入口。"""
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=bind_task_name(rl_cfg(), TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=GRU_TASK_ID,
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=bind_task_name(gru_rl_cfg(), GRU_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=STAIR_EVAL_TASK_ID,
        env_cfg=env_cfg(terrain_generator=stair_only_terrains_cfg()),
        play_env_cfg=env_cfg(play=True, terrain_generator=stair_only_terrains_cfg()),
        rl_cfg=bind_task_name(rl_cfg(), STAIR_EVAL_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    for task_id, overrides in _EXP_VARIANTS:
        register_mjlab_task(
            task_id=task_id,
            env_cfg=env_cfg(**overrides),
            play_env_cfg=env_cfg(play=True, **overrides),
            rl_cfg=bind_task_name(rl_cfg(), task_id),
            runner_cls=Se3ProfiledOnPolicyRunner,
        )


__all__ = [
    "EXP_HEIGHT_WINDOW_TASK_ID",
    "EXP_NO_WHEEL_DEADZONE_TASK_ID",
    "EXP_STAIR_HEIGHT_DZ5_PRUNE_TASK_ID",
    "EXP_STAIR_HEIGHT_DZ5_PRUNE_UPWARD_TASK_ID",
    "EXP_STAIR_HEIGHT_DZ5_TASK_ID",
    "EXP_STAIR_SPEED_CAP_TASK_ID",
    "GRU_TASK_ID",
    "STAIR_EVAL_TASK_ID",
    "TASK_ID",
    "env_cfg",
    "gru_rl_cfg",
    "register",
    "rl_cfg",
]
