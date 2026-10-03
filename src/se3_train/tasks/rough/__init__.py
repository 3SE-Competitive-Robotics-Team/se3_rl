"""崎岖地形行走任务（MLP / GRU 两个入口）与台阶定向评测入口。"""

from mjlab.tasks.registry import register_mjlab_task

from se3_train.rl_cfg import bind_task_name
from se3_train.tasks.common import Se3ProfiledOnPolicyRunner

from .env_cfg import env_cfg
from .rl_cfg import gru_rl_cfg, rl_cfg
from .terrains import stair_only_terrains_cfg

TASK_ID = "SE3-WheelLegged-Rough"
# M16：同一份 env_cfg，只把 actor/critic 换成 GRU（rl_cfg.gru_rl_cfg）。
GRU_TASK_ID = "SE3-WheelLegged-Rough-GRU"
STAIR_EVAL_TASK_ID = "SE3-WheelLegged-Rough-StairEval"
# 对照实验用临时入口的约定：并发 run 共用 Pod 上同一份仓库，中途切 commit 会让在跑的 run 把新 commit 写进 ONNX 溯源，
# 所以对照用任务入口而不是逐实验 commit 区分；对照结束、定下默认值后删除入口，复现用对应 commit。
# 2026-10-02：M39–M54 与 RJ1 的临时入口已全部删除，默认配置 = RJ1（M54 + 合入 J10 跳跃，env_cfg 模块常量），
# 各对照的改动与依据见 env_cfg.py 模块 docstring 与 docs/plan/m39_*–m54_*.md、rj1_rough_jump_20261002.md。
# 2026-10-04：Exp-Actor128（actor 128/64/32，已并入默认，见 rl_cfg.ROUGH_ACTOR_HIDDEN_DIMS）、
# Exp-Dec2（推理 100 Hz）与 Exp-Dec2-Steps48（100 Hz + 每轮 48 步）临时入口删除；维持 50 Hz。
# 复现 Dec2 用 69fd369，Dec2-Steps48 用 07fe1c2，Actor128 用 1545d9f / 635be5f。
# 2026-10-04（用户定）：姿态约束加倍的对照——tracking_orientation_l2 权重 −12 → −24，其余与默认相同。
# 依据：基线平均倾角 10°、11% 时间超过 15°，姿态罚只有速度跟踪奖励的 8.5%（bad_tilt 已在 M37 删除）。
# 临时入口：对照结束、定下权重后删除，复现用对应 commit。
EXP_ORIENT24_TASK_ID = "SE3-WheelLegged-Rough-Exp-Orient24"
EXP_ORIENT24_WEIGHT = -24.0
# 2026-10-03（用户定）：加宽域随机化对照——真 Kp/Kd、质心 ±5 cm、质量 −1…+3 kg、恢复系数 0–1、气弹簧 ×0.9–1.5、
# 动作延迟 0–10 ms，其余与默认相同（见 env_cfg.ROUGH_WIDE_DR_* 注释）。临时入口：对照结束、定下默认后删除，复现用对应 commit。
EXP_WIDE_DR_TASK_ID = "SE3-WheelLegged-Rough-Exp-WideDR"


def register() -> None:
    """注册原始 Rough（MLP）、Rough-GRU 与台阶定向评测任务。"""
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
    register_mjlab_task(
        task_id=EXP_ORIENT24_TASK_ID,
        env_cfg=env_cfg(orientation_weight=EXP_ORIENT24_WEIGHT),
        play_env_cfg=env_cfg(play=True, orientation_weight=EXP_ORIENT24_WEIGHT),
        rl_cfg=bind_task_name(rl_cfg(), EXP_ORIENT24_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_WIDE_DR_TASK_ID,
        env_cfg=env_cfg(wide_dr=True),
        play_env_cfg=env_cfg(play=True, wide_dr=True),
        rl_cfg=bind_task_name(rl_cfg(), EXP_WIDE_DR_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )


__all__ = [
    "EXP_ORIENT24_TASK_ID",
    "EXP_ORIENT24_WEIGHT",
    "EXP_WIDE_DR_TASK_ID",
    "GRU_TASK_ID",
    "STAIR_EVAL_TASK_ID",
    "TASK_ID",
    "register",
]
