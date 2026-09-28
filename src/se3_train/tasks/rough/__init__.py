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
# 2026-09-28：M25–M38 的临时入口已全部删除，默认配置 = M38（env_cfg 模块常量），
# 各对照 commit 见 env_cfg.py 模块 docstring 与 docs/plan/m3*_*.md。
# M39（2026-09-28 用户定）：M38 默认 + action_rate 权重 −0.48 → −0.10（docs/plan/m39_action_rate_20260928.md）。临时入口。
EXP_ACTION_RATE_010_TASK_ID = "SE3-WheelLegged-Rough-Exp-ActionRate010"
# M40（2026-09-28 用户定）：M39 + action_smoothness −0.12 → −0.06（docs/plan/m40_action_smooth_20260928.md）。临时入口。
EXP_ACTION_RATE_010_SMOOTH_006_TASK_ID = "SE3-WheelLegged-Rough-Exp-ActionRate010Smooth006"
# M41（2026-09-28 用户定）：删 action_smoothness、action_rate −0.01（docs/plan/m41_action_rate_001_20260928.md）。临时入口。
EXP_ACTION_RATE_001_NO_SMOOTH_TASK_ID = "SE3-WheelLegged-Rough-Exp-ActionRate001NoSmooth"
# M42（2026-09-28 用户定）：执行链对齐复旦量级（腿 scale 0.5、轮 10、腿 kp 20 / kd 1.5）+ 动作罚 −0.05 / −0.05
# （docs/plan/m42_fudan_actuation_20260928.md）。临时入口。
EXP_FUDAN_ACTUATION_TASK_ID = "SE3-WheelLegged-Rough-Exp-FudanActuation"
_M42 = {
    "action_rate_weight": -0.05,
    "action_smoothness_weight": -0.05,
    "leg_action_scale": 0.5,
    "wheel_action_scale": 10.0,
    "leg_kp": 20.0,
    "leg_kd": 1.5,
}
# M43（2026-09-28 用户定）：M42 + 速度跟踪核换成复旦的尖核 + 有界宽核（docs/plan/m43_fudan_tracking_20260928.md）。临时入口。
EXP_FUDAN_TRACKING_TASK_ID = "SE3-WheelLegged-Rough-Exp-FudanActuationTracking"
_M43 = {**_M42, "tracking_kernel": "fudan"}
# M44（2026-09-28 用户定）：M43 + 台阶列 yaw 角速度跟踪加回来（docs/plan/m44_stair_ang_vel_20260928.md）。临时入口。
EXP_STAIR_ANG_VEL_TASK_ID = "SE3-WheelLegged-Rough-Exp-FudanActuationTrackingStairYaw"
_M44 = {**_M43, "stair_ang_vel_tracking": True}
# M42（2026-09-28 用户定）：action_rate −0.05、action_smoothness −0.05（复旦上台阶3 的组合，对探索噪声的定价 k=0.4 与之等价，
# docs/plan/m42_fudan_action_penalty_20260928.md）。临时入口。
EXP_ACTION_RATE_005_SMOOTH_005_TASK_ID = "SE3-WheelLegged-Rough-Exp-ActionRate005Smooth005"


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
        task_id=EXP_ACTION_RATE_010_TASK_ID,
        env_cfg=env_cfg(action_rate_weight=-0.10),
        play_env_cfg=env_cfg(play=True, action_rate_weight=-0.10),
        rl_cfg=bind_task_name(rl_cfg(), EXP_ACTION_RATE_010_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_ACTION_RATE_010_SMOOTH_006_TASK_ID,
        env_cfg=env_cfg(action_rate_weight=-0.10, action_smoothness_weight=-0.06),
        play_env_cfg=env_cfg(play=True, action_rate_weight=-0.10, action_smoothness_weight=-0.06),
        rl_cfg=bind_task_name(rl_cfg(), EXP_ACTION_RATE_010_SMOOTH_006_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_ACTION_RATE_001_NO_SMOOTH_TASK_ID,
        env_cfg=env_cfg(action_rate_weight=-0.01, action_smoothness_weight=0.0),
        play_env_cfg=env_cfg(play=True, action_rate_weight=-0.01, action_smoothness_weight=0.0),
        rl_cfg=bind_task_name(rl_cfg(), EXP_ACTION_RATE_001_NO_SMOOTH_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_FUDAN_ACTUATION_TASK_ID,
        env_cfg=env_cfg(**_M42),
        play_env_cfg=env_cfg(play=True, **_M42),
        rl_cfg=bind_task_name(rl_cfg(), EXP_FUDAN_ACTUATION_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_FUDAN_TRACKING_TASK_ID,
        env_cfg=env_cfg(**_M43),
        play_env_cfg=env_cfg(play=True, **_M43),
        rl_cfg=bind_task_name(rl_cfg(), EXP_FUDAN_TRACKING_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_STAIR_ANG_VEL_TASK_ID,
        env_cfg=env_cfg(**_M44),
        play_env_cfg=env_cfg(play=True, **_M44),
        rl_cfg=bind_task_name(rl_cfg(), EXP_STAIR_ANG_VEL_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_ACTION_RATE_005_SMOOTH_005_TASK_ID,
        env_cfg=env_cfg(action_rate_weight=-0.05, action_smoothness_weight=-0.05),
        play_env_cfg=env_cfg(play=True, action_rate_weight=-0.05, action_smoothness_weight=-0.05),
        rl_cfg=bind_task_name(rl_cfg(), EXP_ACTION_RATE_005_SMOOTH_005_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )


__all__ = [
    "EXP_ACTION_RATE_001_NO_SMOOTH_TASK_ID",
    "EXP_ACTION_RATE_005_SMOOTH_005_TASK_ID",
    "EXP_ACTION_RATE_010_SMOOTH_006_TASK_ID",
    "EXP_ACTION_RATE_010_TASK_ID",
    "EXP_FUDAN_ACTUATION_TASK_ID",
    "EXP_FUDAN_TRACKING_TASK_ID",
    "EXP_STAIR_ANG_VEL_TASK_ID",
    "GRU_TASK_ID",
    "STAIR_EVAL_TASK_ID",
    "TASK_ID",
    "register",
]
