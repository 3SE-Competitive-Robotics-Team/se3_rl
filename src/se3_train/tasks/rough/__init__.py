"""崎岖地形行走任务（MLP / GRU 两个入口）与台阶定向评测入口。"""

from mjlab.tasks.registry import register_mjlab_task

from se3_train.rl_cfg import bind_task_name
from se3_train.tasks.common import Se3ProfiledOnPolicyRunner

from .env_cfg import env_cfg
from .rl_cfg import (
    ROUGH_LONG_ROLLOUT_STEPS,
    gru_rl_cfg,
    long_rollout_rl_cfg,
    rl_cfg,
    small_actor_rl_cfg,
)
from .terrains import stair_only_terrains_cfg

TASK_ID = "SE3-WheelLegged-Rough"
# M16：同一份 env_cfg，只把 actor/critic 换成 GRU（rl_cfg.gru_rl_cfg）。
GRU_TASK_ID = "SE3-WheelLegged-Rough-GRU"
STAIR_EVAL_TASK_ID = "SE3-WheelLegged-Rough-StairEval"
# 对照实验用临时入口的约定：并发 run 共用 Pod 上同一份仓库，中途切 commit 会让在跑的 run 把新 commit 写进 ONNX 溯源，
# 所以对照用任务入口而不是逐实验 commit 区分；对照结束、定下默认值后删除入口，复现用对应 commit。
# 2026-10-02：M39–M54 与 RJ1 的临时入口已全部删除，默认配置 = RJ1（M54 + 合入 J10 跳跃，env_cfg 模块常量），
# 各对照的改动与依据见 env_cfg.py 模块 docstring 与 docs/plan/m39_*–m54_*.md、rj1_rough_jump_20261002.md。
# 2026-10-03（用户定）：actor 隐藏层 128/64/32 的对照（critic 与环境不变，见 rl_cfg.ROUGH_SMALL_ACTOR_HIDDEN_DIMS）。
# 临时入口：对照结束、定下 actor 尺寸后删除，复现用对应 commit。
EXP_ACTOR_128_TASK_ID = "SE3-WheelLegged-Rough-Exp-Actor128"
# 2026-10-03（用户定）：推理频率 50 → 100 Hz 的对照——只改 decimation 4 → 2（物理 5 ms 不变），
# PPO 超参数（γ / λ / 每轮 24 步）与奖励权重都不变。临时入口：对照结束、定下推理频率后删除，复现用对应 commit。
EXP_DEC2_TASK_ID = "SE3-WheelLegged-Rough-Exp-Dec2"
EXP_DEC2_DECIMATION = 2
# 2026-10-03（用户定）：Dec2 + 每轮 48 步（每轮仍覆盖 0.48 s，同复旦 100 Hz 配方），课程/推力按 48 步换算轮次；
# γ / λ 等其余 PPO 超参数不变。临时入口，同上。
EXP_DEC2_STEPS48_TASK_ID = "SE3-WheelLegged-Rough-Exp-Dec2-Steps48"


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
        task_id=EXP_ACTOR_128_TASK_ID,
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=bind_task_name(small_actor_rl_cfg(), EXP_ACTOR_128_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_DEC2_TASK_ID,
        env_cfg=env_cfg(decimation=EXP_DEC2_DECIMATION),
        play_env_cfg=env_cfg(play=True, decimation=EXP_DEC2_DECIMATION),
        rl_cfg=bind_task_name(rl_cfg(), EXP_DEC2_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    dec2_steps48 = {
        "decimation": EXP_DEC2_DECIMATION,
        "steps_per_policy_iter": ROUGH_LONG_ROLLOUT_STEPS,
    }
    register_mjlab_task(
        task_id=EXP_DEC2_STEPS48_TASK_ID,
        env_cfg=env_cfg(**dec2_steps48),
        play_env_cfg=env_cfg(play=True, **dec2_steps48),
        rl_cfg=bind_task_name(long_rollout_rl_cfg(), EXP_DEC2_STEPS48_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )


__all__ = [
    "EXP_ACTOR_128_TASK_ID",
    "EXP_DEC2_DECIMATION",
    "EXP_DEC2_STEPS48_TASK_ID",
    "EXP_DEC2_TASK_ID",
    "GRU_TASK_ID",
    "STAIR_EVAL_TASK_ID",
    "TASK_ID",
    "register",
]
