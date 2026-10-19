"""崎岖地形行走任务（MLP / GRU）、台阶定向评测入口，以及 CTS → Student128 部署链入口。"""

from mjlab.tasks.registry import register_mjlab_task

from se3_train.onnx_metadata import observation_term_width
from se3_train.rl_cfg import bind_task_name
from se3_train.tasks.common import Se3ProfiledOnPolicyRunner

from .env_cfg import (
    ROUGH_CTS_ROLE_GROUP,
    ROUGH_CTS_TEACHER_GROUP,
    ROUGH_STAIR_LIN_VEL_X_MIN_VX08,
    ROUGH_VX_OBSERVER_TARGET_GROUP,
    env_cfg,
)
from .rl_cfg import (
    ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU,
    ROUGH_CTS_STUDENT_WARMUP_ITERATIONS,
    cts_rl_cfg,
    gru_rl_cfg,
    rl_cfg,
)
from .terrains import stair_only_terrains_cfg

TASK_ID = "SE3-WheelLegged-Rough"
# M16：同一份 env_cfg，只把 actor/critic 换成 GRU（rl_cfg.gru_rl_cfg）。
GRU_TASK_ID = "SE3-WheelLegged-Rough-GRU"
STAIR_EVAL_TASK_ID = "SE3-WheelLegged-Rough-StairEval"
# 对照实验用临时入口的约定：并发 run 共用 Pod 上同一份仓库，中途切 commit 会让在跑的 run 把新 commit 写进 ONNX 溯源，
# 所以对照用任务入口而不是逐实验 commit 区分；对照结束、定下默认值后删除入口，复现用对应 commit。
# 2026-10-02：M39–M54 与 RJ1 的临时入口已全部删除，默认配置 = RJ1（M54 + 合入 J10 跳跃，env_cfg 模块常量），
# 各对照的改动与依据见 env_cfg.py 模块 docstring 与 M39–M54 与 RJ1 的计划文档（已删，见 docs/README.md）。
# 2026-10-04：Exp-Actor128（actor 128/64/32，已并入默认，见 rl_cfg.ROUGH_ACTOR_HIDDEN_DIMS）、
# Exp-Dec2（推理 100 Hz）与 Exp-Dec2-Steps48（100 Hz + 每轮 48 步）临时入口删除；维持 50 Hz。
# 复现 Dec2 用 69fd369，Dec2-Steps48 用 07fe1c2，Actor128 用 1545d9f / 635be5f。
# 2026-10-04：Exp-Orient24（tracking_orientation_l2 −24）并入默认（env_cfg.ROUGH_ORIENTATION_WEIGHT），入口删除；
# 复现 Orient24 / Base128 用 aeec423。
# 2026-10-04：Exp-HighStand01（high_stand_transition_prob 0.1）并入默认（env_cfg.ROUGH_HIGH_STAND_TRANSITION_PROB），
# 入口删除；复现用 1cb5d77。
# 2026-10-05：Exp-WideDR / -Oracle / -Rest05 / -NoRest 临时入口删除，NoRest 的 DR 并入默认（env_cfg.ROUGH_DR_*，DR1）；
# 复现 WideDR 用 80019b8，Oracle 用 1fa00ca，Rest05 用 17c91ba，NoRest 用 11f9d40。
# 2026-10-05：Exp-HeightSigma07（机身高度罚 σ 0.07）并入默认（env_cfg.ROUGH_BASE_HEIGHT_SIGMA），入口删除；
# 复现 HeightSigma07 与其同代码基线用 f090dbe。
# 2026-10-07（用户定）：CTS + 显式 vx（见 se3_train.cts_observer）——教师 env 用特权编码器 32 维隐向量 + 真实 vx，
# 学生 env 用 16 帧历史编码器回归它，共用 policy；部署走学生路径。新默认轮子包络，对照 VxObserver 与 sim2x 扫描。临时入口，结论后删除。
EXP_CTS_TASK_ID = "SE3-WheelLegged-Rough-Exp-CTS"
# 2026-10-07（用户定）：CTS 单变量对照——教师编码器输入去掉 critic 的 77 点高度扫描（critic 价值函数仍带）。
# Exp-CTS 的教师隐向量含盲学生推不出的前视地形，学生上台阶失真、师生单步奖励差拉到 0.11。临时入口，结论后删除。
EXP_CTS_NOSCAN_TASK_ID = "SE3-WheelLegged-Rough-Exp-CTS-NoScan"
# 2026-10-08（用户定）：NoScan 单变量对照——台阶类列前进指令下限 0.4 → 0.8（ROUGH_STAIR_LIN_VEL_X_MIN_VX08）。
# NoScan-4999 平地 0.4–1.0 超速是台阶列低指令冲速泄漏（见 env_cfg 常量注释）。临时入口，结论后删除。
EXP_CTS_NOSCAN_STAIR_VX08_TASK_ID = "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08"
# 2026-10-09（用户定）：单片机部署版——StairVx08 的学生编码器 480→256→128→33 换成 480→128→128→33
# （ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU），从 StairVx08 model_4999 续训；前 ROUGH_CTS_STUDENT_WARMUP_ITERATIONS 轮只训练学生编码器。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128"
)
# 2026-10-10 用户定：Student128 从零训练，腿 Kp/Kd=30/2、轮 Kd=0.1，动作缩放沿用 0.25/15。
# 无已训练教师，因此不启用换编码器续训专用的学生预热。对照完成并选定默认增益后删除此实验入口。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_LOW_GAINS_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-LowGains"
)
# 2026-10-06：Exp-VxObserver-SpringFF（估计器输出左右弹簧力并替代固定 300 N 做前馈，pcz5opfg）用户判断没用，入口与代码删除；
# 静站时左右反对称方向估计器只照抄上一拍、会漂（.scratch/yaw_diag/spring_probe.py）。复现用 6c4bd2a。
# 2026-10-07：Exp-VxObserver-WheelTN（轮子 T-N 包络换手册额定点口径，xdk0dnoa）并入默认（env_cfg.ROUGH_WHEEL_TORQUE_ENVELOPE），
# 入口删除；复现旧包络用 4832d15。
# 2026-10-19：诊断对照入口 DR1-Oracle / -OracleVel、Vel、Vx、VxObserver（及 -H5 / -Z3 / -YawRB05 / -WheelTN-Budget）、
# CTS-NoScan-StairVx08-JumpExitPen 已有结论，入口与专用开关删除；复现用 846c4fcd。


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
    cts_env_cfg = env_cfg(vx_observer=True, cts=True)
    cts_actor = cts_env_cfg.observations["actor"]
    register_mjlab_task(
        task_id=EXP_CTS_TASK_ID,
        env_cfg=cts_env_cfg,
        play_env_cfg=env_cfg(play=True, vx_observer=True, cts=True),
        rl_cfg=bind_task_name(
            cts_rl_cfg(
                history_length=int(cts_actor.history_length),
                frame_term_dims=tuple(observation_term_width(name) for name in cts_actor.terms),
                target_group=ROUGH_VX_OBSERVER_TARGET_GROUP,
                role_group=ROUGH_CTS_ROLE_GROUP,
            ),
            EXP_CTS_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    cts_noscan_env_cfg = env_cfg(vx_observer=True, cts=True, cts_teacher_height_scan=False)
    cts_noscan_actor = cts_noscan_env_cfg.observations["actor"]
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_TASK_ID,
        env_cfg=cts_noscan_env_cfg,
        play_env_cfg=env_cfg(play=True, vx_observer=True, cts=True, cts_teacher_height_scan=False),
        rl_cfg=bind_task_name(
            cts_rl_cfg(
                history_length=int(cts_noscan_actor.history_length),
                frame_term_dims=tuple(
                    observation_term_width(name) for name in cts_noscan_actor.terms
                ),
                target_group=ROUGH_VX_OBSERVER_TARGET_GROUP,
                role_group=ROUGH_CTS_ROLE_GROUP,
                teacher_obs_group=ROUGH_CTS_TEACHER_GROUP,
            ),
            EXP_CTS_NOSCAN_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    stair_vx08_kwargs = dict(
        vx_observer=True,
        cts=True,
        cts_teacher_height_scan=False,
        stair_lin_vel_x_min=ROUGH_STAIR_LIN_VEL_X_MIN_VX08,
    )
    stair_vx08_env_cfg = env_cfg(**stair_vx08_kwargs)
    stair_vx08_actor = stair_vx08_env_cfg.observations["actor"]
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_TASK_ID,
        env_cfg=stair_vx08_env_cfg,
        play_env_cfg=env_cfg(play=True, **stair_vx08_kwargs),
        rl_cfg=bind_task_name(
            cts_rl_cfg(
                history_length=int(stair_vx08_actor.history_length),
                frame_term_dims=tuple(
                    observation_term_width(name) for name in stair_vx08_actor.terms
                ),
                target_group=ROUGH_VX_OBSERVER_TARGET_GROUP,
                role_group=ROUGH_CTS_ROLE_GROUP,
                teacher_obs_group=ROUGH_CTS_TEACHER_GROUP,
            ),
            EXP_CTS_NOSCAN_STAIR_VX08_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    student128_env_cfg = env_cfg(**stair_vx08_kwargs)
    student128_actor = student128_env_cfg.observations["actor"]
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_TASK_ID,
        env_cfg=student128_env_cfg,
        play_env_cfg=env_cfg(play=True, **stair_vx08_kwargs),
        rl_cfg=bind_task_name(
            cts_rl_cfg(
                history_length=int(student128_actor.history_length),
                frame_term_dims=tuple(
                    observation_term_width(name) for name in student128_actor.terms
                ),
                target_group=ROUGH_VX_OBSERVER_TARGET_GROUP,
                role_group=ROUGH_CTS_ROLE_GROUP,
                teacher_obs_group=ROUGH_CTS_TEACHER_GROUP,
                student_hidden_dims=ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU,
                estimator_warmup_iterations=ROUGH_CTS_STUDENT_WARMUP_ITERATIONS,
            ),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    low_gains_kwargs = dict(stair_vx08_kwargs, leg_kp=30.0, leg_kd=2.0, wheel_kd=0.1)
    low_gains_env_cfg = env_cfg(**low_gains_kwargs)
    low_gains_actor = low_gains_env_cfg.observations["actor"]
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_LOW_GAINS_TASK_ID,
        env_cfg=low_gains_env_cfg,
        play_env_cfg=env_cfg(play=True, **low_gains_kwargs),
        rl_cfg=bind_task_name(
            cts_rl_cfg(
                history_length=int(low_gains_actor.history_length),
                frame_term_dims=tuple(
                    observation_term_width(name) for name in low_gains_actor.terms
                ),
                target_group=ROUGH_VX_OBSERVER_TARGET_GROUP,
                role_group=ROUGH_CTS_ROLE_GROUP,
                teacher_obs_group=ROUGH_CTS_TEACHER_GROUP,
                student_hidden_dims=ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU,
                estimator_warmup_iterations=0,
            ),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_LOW_GAINS_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )


__all__ = [
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_LOW_GAINS_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_TASK_ID",
    "EXP_CTS_NOSCAN_TASK_ID",
    "EXP_CTS_TASK_ID",
    "GRU_TASK_ID",
    "STAIR_EVAL_TASK_ID",
    "TASK_ID",
    "register",
]
