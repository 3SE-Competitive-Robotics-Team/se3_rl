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
# M47（2026-09-29 用户定）：M39 + 删 bad_orientation 终止（docs/plan/m47_no_bad_orientation_20260929.md）。临时入口。
EXP_ACTION_RATE_010_NO_BAD_ORI_TASK_ID = "SE3-WheelLegged-Rough-Exp-ActionRate010NoBadOri"
_M39 = {"action_rate_weight": -0.10}
_M47 = {**_M39, "bad_orientation_termination": False}
# M48（2026-09-29 用户定）：M47 + joint_mirror −0.179 → −5（docs/plan/m48_joint_mirror_20260929.md）。临时入口。
EXP_JOINT_MIRROR_TASK_ID = "SE3-WheelLegged-Rough-Exp-ActionRate010NoBadOriMirror5"
_M48 = {**_M47, "joint_mirror_weight": -5.0}
# M49（2026-09-29 用户定）：M48 + 删 joint_mirror、加轮心前后错位 Δx² −50（docs/plan/m49_wheel_fore_aft_20260929.md）；
# 从 M48 最新 checkpoint 完整续训。临时入口。
EXP_WHEEL_FORE_AFT_TASK_ID = "SE3-WheelLegged-Rough-Exp-ActionRate010NoBadOriWheelDx50"
_M49 = {**_M48, "joint_mirror_weight": 0.0, "wheel_fore_aft_weight": -50.0}
# M50（2026-09-29 用户筹划中）：M49 + 台阶列不采样 yaw 指令、初始朝向正对台阶 ±30°、台阶列 yaw 跟踪不再置零（docs/plan/m50_stair_yaw_20260929.md）。
# 改动还在累加，未启动。临时入口。
EXP_M50_TASK_ID = "SE3-WheelLegged-Rough-Exp-M50"
_M50 = {
    **_M49,
    # 改动一：台阶列不采样 yaw 指令，初始朝向正对台阶 ±30°。
    "stair_ang_vel_yaw_range": (0.0, 0.0),
    "stair_spawn_yaw_half_range_deg": 30.0,
    # 改动二：台阶列 yaw 跟踪不再置零（复用 M44 的开关）；指令恒 0 时它就是台阶列的 yaw 角速度罚。
    "stair_ang_vel_tracking": True,
}
# M51（2026-09-29 用户筹划中）：M50 + 随机粗糙、波浪、离散矮障碍三列地形，MDP 与 flat 一致（docs/plan/m51_random_terrain_20260929.md）。
# 未启动。临时入口。
EXP_M51_TASK_ID = "SE3-WheelLegged-Rough-Exp-M51"
_M51 = {**_M50, "random_terrains": True}
# M52（2026-09-30 用户定）：M51 + 训练 plant 去掉 300 N 膝气弹簧（docs/plan/m52_no_knee_spring_20260930.md）。临时入口。
EXP_M52_TASK_ID = "SE3-WheelLegged-Rough-Exp-M52"
_M52 = {**_M51, "knee_gas_spring": False}
# M53（2026-09-30 用户定）：M51 + 电机侧膝气弹簧前馈补偿（按 300 N，DR 270–330 N 照旧）+ 腿部 T-N 包络
# 按物理含义取参 ×0.8（32 N·m 平台）（docs/plan/m53_knee_spring_feedforward_20260930.md）。临时入口。
EXP_M53_TASK_ID = "SE3-WheelLegged-Rough-Exp-M53"
_M53 = {**_M51, "knee_gas_spring_compensation": True, "leg_torque_envelope_scale": 0.8}
# M54（2026-09-30 用户定）：M53 + 台阶列航向保持 −6、轮前后错位 ×0.25、出生朝向 ±15°
# （docs/plan/m54_stair_heading_hold_20260930.md）。临时入口。
EXP_M54_TASK_ID = "SE3-WheelLegged-Rough-Exp-M54"
_M54 = {
    **_M53,
    "stair_heading_hold_weight": -6.0,
    "stair_wheel_fore_aft_scale": 0.25,
    "stair_spawn_yaw_half_range_deg": 15.0,
}
# RJ1（2026-10-02 用户定）：M54 + 合入 J10 跳跃（平地列 30% 跳跃样本、高度固定 0.22、34 维观测含相位、无 RSI）
# （docs/plan/rj1_rough_jump_20261002.md）。从 M54 model_4999 热启动。临时入口。
EXP_RJ1_TASK_ID = "SE3-WheelLegged-Rough-Exp-RJ1"
_RJ1 = {**_M54, "jump_mimic": True}
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
# M45（2026-09-29 用户定）：M42 + 高度罚换成复旦式高度对（docs/plan/m45_fudan_height_20260929.md）。临时入口。
EXP_FUDAN_HEIGHT_TASK_ID = "SE3-WheelLegged-Rough-Exp-FudanActuationHeight"
_M45 = {**_M42, "height_shape": "fudan"}
# M46（2026-09-29 用户定）：M45 + 台阶类列高度参考改两轮支撑面（docs/plan/m46_stair_support_height_20260929.md）；
# 从 M45 model_600 完整续训（common_step_counter 随 checkpoint 恢复，平地热身不重来）。临时入口。
EXP_FUDAN_HEIGHT_SUPPORT_TASK_ID = "SE3-WheelLegged-Rough-Exp-FudanActuationHeightSupport"
_M46 = {**_M45, "fudan_height_stair_reference": "support"}
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
        task_id=EXP_ACTION_RATE_010_NO_BAD_ORI_TASK_ID,
        env_cfg=env_cfg(**_M47),
        play_env_cfg=env_cfg(play=True, **_M47),
        rl_cfg=bind_task_name(rl_cfg(), EXP_ACTION_RATE_010_NO_BAD_ORI_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_JOINT_MIRROR_TASK_ID,
        env_cfg=env_cfg(**_M48),
        play_env_cfg=env_cfg(play=True, **_M48),
        rl_cfg=bind_task_name(rl_cfg(), EXP_JOINT_MIRROR_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_WHEEL_FORE_AFT_TASK_ID,
        env_cfg=env_cfg(**_M49),
        play_env_cfg=env_cfg(play=True, **_M49),
        rl_cfg=bind_task_name(rl_cfg(), EXP_WHEEL_FORE_AFT_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_M50_TASK_ID,
        env_cfg=env_cfg(**_M50),
        play_env_cfg=env_cfg(play=True, **_M50),
        rl_cfg=bind_task_name(rl_cfg(), EXP_M50_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_M51_TASK_ID,
        env_cfg=env_cfg(**_M51),
        play_env_cfg=env_cfg(play=True, **_M51),
        rl_cfg=bind_task_name(rl_cfg(), EXP_M51_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_M52_TASK_ID,
        env_cfg=env_cfg(**_M52),
        play_env_cfg=env_cfg(play=True, **_M52),
        rl_cfg=bind_task_name(rl_cfg(), EXP_M52_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_M53_TASK_ID,
        env_cfg=env_cfg(**_M53),
        play_env_cfg=env_cfg(play=True, **_M53),
        rl_cfg=bind_task_name(rl_cfg(), EXP_M53_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_M54_TASK_ID,
        env_cfg=env_cfg(**_M54),
        play_env_cfg=env_cfg(play=True, **_M54),
        rl_cfg=bind_task_name(rl_cfg(), EXP_M54_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_RJ1_TASK_ID,
        env_cfg=env_cfg(**_RJ1),
        play_env_cfg=env_cfg(play=True, **_RJ1),
        rl_cfg=bind_task_name(rl_cfg(), EXP_RJ1_TASK_ID),
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
        task_id=EXP_FUDAN_HEIGHT_TASK_ID,
        env_cfg=env_cfg(**_M45),
        play_env_cfg=env_cfg(play=True, **_M45),
        rl_cfg=bind_task_name(rl_cfg(), EXP_FUDAN_HEIGHT_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_FUDAN_HEIGHT_SUPPORT_TASK_ID,
        env_cfg=env_cfg(**_M46),
        play_env_cfg=env_cfg(play=True, **_M46),
        rl_cfg=bind_task_name(rl_cfg(), EXP_FUDAN_HEIGHT_SUPPORT_TASK_ID),
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
    "EXP_ACTION_RATE_010_NO_BAD_ORI_TASK_ID",
    "EXP_ACTION_RATE_010_SMOOTH_006_TASK_ID",
    "EXP_ACTION_RATE_010_TASK_ID",
    "EXP_FUDAN_ACTUATION_TASK_ID",
    "EXP_FUDAN_HEIGHT_SUPPORT_TASK_ID",
    "EXP_FUDAN_HEIGHT_TASK_ID",
    "EXP_FUDAN_TRACKING_TASK_ID",
    "EXP_JOINT_MIRROR_TASK_ID",
    "EXP_M50_TASK_ID",
    "EXP_M51_TASK_ID",
    "EXP_M52_TASK_ID",
    "EXP_M53_TASK_ID",
    "EXP_M54_TASK_ID",
    "EXP_RJ1_TASK_ID",
    "EXP_STAIR_ANG_VEL_TASK_ID",
    "EXP_WHEEL_FORE_AFT_TASK_ID",
    "GRU_TASK_ID",
    "STAIR_EVAL_TASK_ID",
    "TASK_ID",
    "register",
]
