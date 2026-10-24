"""崎岖地形行走任务（MLP / GRU）、台阶定向评测入口，以及 CTS → Student128 部署链入口。"""

import math

from mjlab.tasks.registry import register_mjlab_task

from se3_train.onnx_metadata import observation_term_width
from se3_train.rl_cfg import ImitationLossCfg, PolicyDistillCfg, bind_task_name
from se3_train.tasks.common import Se3ProfiledOnPolicyRunner

from .env_cfg import (
    ROUGH_CTS_ROLE_GROUP,
    ROUGH_CTS_TEACHER_GROUP,
    ROUGH_FALL_TEACHER_TARGET_GROUP,
    ROUGH_RECOVER_AMP_GROUP,
    ROUGH_RECOVER_AMP_MASK_GROUP,
    ROUGH_STAIR_LIN_VEL_X_MIN_VX08,
    ROUGH_VX_OBSERVER_TARGET_GROUP,
    env_cfg,
)
from .rl_cfg import (
    ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU,
    ROUGH_FALL_TEACHER_IMITATION_LOSS_COEF,
    ROUGH_FALL_TEACHER_IMITATION_LOSS_HUBER_DELTA,
    ROUGH_RECOVER_DISTILL_TEACHER_CKPT,
    cts_rl_cfg,
    gru_rl_cfg,
    recover_amp_cfg,
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
# （ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU），当时从 StairVx08 model_4999 续训、前 200 轮只训学生编码器；
# 2026-10-19 用户定删掉预热默认值，入口按从零训练配置。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128"
)
# 2026-10-10 用户定：Student128 从零训练，腿 Kp/Kd=30/2、轮 Kd=0.1，动作缩放沿用 0.25/15。
# 无已训练教师，因此不启用换编码器续训专用的学生预热。对照完成并选定默认增益后删除此实验入口。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_LOW_GAINS_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-LowGains"
)
# 2026-10-22 用户定：Student128 单变量对照——学生隐向量 z_S 不 detach、由 PPO 端到端训练（去掉隐向量 MSE），
# v̂x 仍 detach 并由 vx MSE 监督；其余与 Student128 逐项相同（见 se3_train.cts_observer 的 student_latent_grad）。
# 以前没有干净对照：Z3（PPO 端到端 3 维隐向量）与 CTS-NoScan（教师 MSE）混了规模、隐向量维数与教师 env 占比。临时入口，结论后删除。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_STUDENT_PPO_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-StudentPPO"
)
# 2026-10-23 用户定：Student128 + 倒地起身脚本教师（6 月 recovery_teacher 路线移植，见 fall_teacher.py），
# 倒置与侧躺都接、固定正向 2.0 rad/s、夹角限速折叠；不带拉格朗日倒地约束与倒地动作平滑罚，其余与 Student128 逐项相同。
# 临时入口，结论后删除。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacher"
)
# 2026-10-23 用户定：FallTeacher + PPO 总损失里直接加模仿项（se3_train.ppo 的 imitation_loss，系数 0.2·α、Huber），
# 其余与 FallTeacher 逐项相同。FallTeacher（kf3o74t7）只有模仿奖励，前 200 轮策略与教师的动作差不降。临时入口，结论后删除。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC"
)
# 2026-10-23 用户定：FallTeacherBC + 倒地功率终止（M 内腿正功率连续 3 步超过 400 W 即终止，单独罚 300，
# 见 env_cfg.ROUGH_FALL_POWER_LIMIT_PENALTY），其余与 FallTeacherBC 逐项相同。教师退场后纯策略起身变猛（kf3o74t7），
# 拉格朗日 v2 罚价封顶压不住；对照 FallTeacherBC（m36nvar6）。临时入口，结论后删除。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_POWER_LIMIT_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-PowerLimit"
)
# 2026-10-23 用户定：PowerLimit 单变量对照——倒地功率终止限值 400 → 600 W，其余与 PowerLimit（k4re9m75）逐项相同。
# MLP 版 cpzyma4c（600 W）教师退场后仍能起身，而 400 W 的 d37wocyd / k4re9m75 退化；MLP 版走路太差，回到 CTS 验证。
# 临时入口，结论后删除。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_POWER_LIMIT600_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-PowerLimit600"
)
# 2026-10-23 用户定：PowerLimit600 单变量对照——倒地教师 α 退火提前、缩短：600–1300–1900 轮 → 400–600–800 轮
# （α 值 0.85 / 0.45 / 0 不变），其余与 PowerLimit600 逐项相同。看教师更早退场时策略能否学会起身。临时入口，结论后删除。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_POWER_LIMIT600_ALPHA800_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-PowerLimit600-Alpha800"
)
# 2026-10-24 用户定：recover 组——倒地起身单独成组、用自己的奖励（不按时间扣分、限时站稳、超限线性罚，见 env_cfg.ROUGH_RECOVER_*），
# 走路组不再倒地出生；去掉功率终止；倒地教师（α 400–600–800、模仿损失乘 α）只在 recover 组前期混合。其余与 PowerLimit600-Alpha800
# （dqkkieq0）相同。临时入口，结论后删除。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-Alpha800-Recover"
)
# 2026-10-24 用户定：Recover 单变量对照——速度课程 yaw 上限改由转向跟踪 EMA 驱动（yaw_gate），其余逐项相同。
# gyi9exiz 的 yaw 上限前约 500 轮就跟着直线跟踪放到 ±12 rad/s，原地转向跟踪到 2500 轮后才上 0.3。临时入口，结论后删除。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_YAW_GATE_TASK_ID = "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-Alpha800-Recover-YawGate"
# 2026-10-24 用户定：Recover 去掉倒地教师与模仿损失，改用 AMP 学起身风格——参考是用户认可的 gyi9exiz model_2200 在 recover 组
# 条件下的起身片段，判别器与风格奖励只作用在 recover 组（见 env_cfg.ROUGH_RECOVER_AMP_*、rl_cfg.recover_amp_cfg）；其余与 Recover
# （gyi9exiz）逐项相同（不开 yaw_gate）。q59ces4z 教师退场后倒置起身丢了、稀疏奖励找不回来，风格奖励每步都有梯度。临时入口，结论后删除。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_AMP_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-Recover-AMP"
)
# 2026-10-25 用户定：Recover-AMP + recover 组网络老师在线蒸馏（老师 gyi9exiz model_2200，全程保持，见 rl_cfg.ROUGH_RECOVER_DISTILL_*）；
# 从 6iwgpa2m（Recover-AMP）续训。6iwgpa2m 走路全面领先 gyi9exiz，但 1100 轮倒置起身仍为 0、侧躺停在 0.53。临时入口，结论后删除。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_AMP_DISTILL_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-Recover-AMP-Distill"
)
# 2026-10-24 用户定：PowerLimit600-Alpha800 + 教师退场后模仿损失一直保留（PersistBC）——模仿损失不乘 α（固定 0.2），
# α = 0 后教师状态机继续推进出目标但不执行，教师目标改限幅开环（实测 ±0.3 rad），其余逐项相同。dqkkieq0 教师退场后
# 动作模式完全变了（模仿项随 α 归零）。临时入口，结论后删除。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_POWER_LIMIT600_ALPHA800_PERSIST_BC_TASK_ID = "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-PowerLimit600-Alpha800-PersistBC"
# 2026-10-23 用户定：PowerLimit 去掉 CTS 编码器，换普通 MLP actor（单帧 30 维输入、128/64/32，rl_cfg），
# env 去掉 vx_observer / cts 开关（actor 不再开 16 帧历史，也没有 vx 监督目标与 CTS 分组观测组），
# critic、奖励、终止、倒地教师与模仿损失与 PowerLimit（k4re9m75）逐项相同。CTS 每轮更慢，先用纯 actor 验证起身路线。
# 临时入口，结论后删除。
EXP_STAIR_VX08_FALL_TEACHER_BC_POWER_LIMIT_MLP_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-StairVx08-FallTeacherBC-PowerLimit-MLP"
)
# 2026-10-23 用户定：PowerLimit-MLP 单变量对照——倒地功率终止限值 400 → 600 W，其余逐项相同。
# k4re9m75（400 W）约 3200 轮起倒置起身率掉到 ≈0，疑似 400 W 内找不到翻身动作而躺平。临时入口，结论后删除。
EXP_STAIR_VX08_FALL_TEACHER_BC_POWER_LIMIT600_MLP_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-StairVx08-FallTeacherBC-PowerLimit600-MLP"
)
# 2026-10-06：Exp-VxObserver-SpringFF（估计器输出左右弹簧力并替代固定 300 N 做前馈，pcz5opfg）用户判断没用，入口与代码删除；
# 静站时左右反对称方向估计器只照抄上一拍、会漂（.scratch/yaw_diag/spring_probe.py）。复现用 6c4bd2a。
# 2026-10-07：Exp-VxObserver-WheelTN（轮子 T-N 包络换手册额定点口径，xdk0dnoa）并入默认（env_cfg.ROUGH_WHEEL_TORQUE_ENVELOPE），
# 入口删除；复现旧包络用 4832d15。
# 2026-10-19：诊断对照入口 DR1-Oracle / -OracleVel、Vel、Vx、VxObserver（及 -H5 / -Z3 / -YawRB05 / -WheelTN-Budget）、
# CTS-NoScan-StairVx08-JumpExitPen 已有结论，入口与专用开关删除；复现用 846c4fcd。


def _recover_policy_distill(cfg) -> PolicyDistillCfg:
    """recover 组网络老师蒸馏：整圈周期按本入口动作项的腿部缩放算（与倒地教师模仿损失同一套对齐）。"""
    imitation = _fall_teacher_imitation_loss(cfg)
    return PolicyDistillCfg(
        teacher_checkpoint=str(ROUGH_RECOVER_DISTILL_TEACHER_CKPT),
        mask_group=ROUGH_RECOVER_AMP_MASK_GROUP,
        role_group=ROUGH_CTS_ROLE_GROUP,
        coef=imitation.coef,
        huber_delta=imitation.huber_delta,
        front_period=imitation.front_period,
        back_period=imitation.back_period,
        shift_back=imitation.shift_back,
    )


def _fall_teacher_imitation_loss(cfg, *, alpha_weighted: bool = True) -> ImitationLossCfg:
    """倒地教师模仿损失：整圈周期按本入口动作项的腿部缩放算（2π / scale），joint 语义下后杆同步平移。"""
    action = cfg.actions["delayed_action"]
    scales = tuple(float(s) for s in action.leg_scales)
    if scales[0] != scales[2] or scales[1] != scales[3]:
        raise ValueError(f"模仿损失按左右同周期对齐，腿部缩放须左右相同，实际 {scales}")
    return ImitationLossCfg(
        group=ROUGH_FALL_TEACHER_TARGET_GROUP,
        coef=ROUGH_FALL_TEACHER_IMITATION_LOSS_COEF,
        huber_delta=ROUGH_FALL_TEACHER_IMITATION_LOSS_HUBER_DELTA,
        front_period=2.0 * math.pi / scales[0],
        back_period=2.0 * math.pi / scales[1],
        shift_back=action.leg_action_semantics == "joint",
        alpha_weighted=alpha_weighted,
    )


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
            ),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    student_ppo_env_cfg = env_cfg(**stair_vx08_kwargs)
    student_ppo_actor = student_ppo_env_cfg.observations["actor"]
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_STUDENT_PPO_TASK_ID,
        env_cfg=student_ppo_env_cfg,
        play_env_cfg=env_cfg(play=True, **stair_vx08_kwargs),
        rl_cfg=bind_task_name(
            cts_rl_cfg(
                history_length=int(student_ppo_actor.history_length),
                frame_term_dims=tuple(
                    observation_term_width(name) for name in student_ppo_actor.terms
                ),
                target_group=ROUGH_VX_OBSERVER_TARGET_GROUP,
                role_group=ROUGH_CTS_ROLE_GROUP,
                teacher_obs_group=ROUGH_CTS_TEACHER_GROUP,
                student_hidden_dims=ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU,
                student_latent_grad="ppo",
            ),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_STUDENT_PPO_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    fall_teacher_kwargs = dict(stair_vx08_kwargs, fall_lagrange=False, fall_teacher=True)
    fall_teacher_env_cfg = env_cfg(**fall_teacher_kwargs)
    fall_teacher_actor = fall_teacher_env_cfg.observations["actor"]
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_TASK_ID,
        env_cfg=fall_teacher_env_cfg,
        play_env_cfg=env_cfg(play=True, **fall_teacher_kwargs),
        rl_cfg=bind_task_name(
            cts_rl_cfg(
                history_length=int(fall_teacher_actor.history_length),
                frame_term_dims=tuple(
                    observation_term_width(name) for name in fall_teacher_actor.terms
                ),
                target_group=ROUGH_VX_OBSERVER_TARGET_GROUP,
                role_group=ROUGH_CTS_ROLE_GROUP,
                teacher_obs_group=ROUGH_CTS_TEACHER_GROUP,
                student_hidden_dims=ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU,
            ),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    fall_teacher_bc_env_cfg = env_cfg(**fall_teacher_kwargs)
    fall_teacher_bc_actor = fall_teacher_bc_env_cfg.observations["actor"]
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_TASK_ID,
        env_cfg=fall_teacher_bc_env_cfg,
        play_env_cfg=env_cfg(play=True, **fall_teacher_kwargs),
        rl_cfg=bind_task_name(
            cts_rl_cfg(
                history_length=int(fall_teacher_bc_actor.history_length),
                frame_term_dims=tuple(
                    observation_term_width(name) for name in fall_teacher_bc_actor.terms
                ),
                target_group=ROUGH_VX_OBSERVER_TARGET_GROUP,
                role_group=ROUGH_CTS_ROLE_GROUP,
                teacher_obs_group=ROUGH_CTS_TEACHER_GROUP,
                student_hidden_dims=ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU,
                imitation_loss=_fall_teacher_imitation_loss(fall_teacher_bc_env_cfg),
            ),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    power_limit_kwargs = dict(fall_teacher_kwargs, fall_power_limit=True)
    power_limit_env_cfg = env_cfg(**power_limit_kwargs)
    power_limit_actor = power_limit_env_cfg.observations["actor"]
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_POWER_LIMIT_TASK_ID,
        env_cfg=power_limit_env_cfg,
        play_env_cfg=env_cfg(play=True, **power_limit_kwargs),
        rl_cfg=bind_task_name(
            cts_rl_cfg(
                history_length=int(power_limit_actor.history_length),
                frame_term_dims=tuple(
                    observation_term_width(name) for name in power_limit_actor.terms
                ),
                target_group=ROUGH_VX_OBSERVER_TARGET_GROUP,
                role_group=ROUGH_CTS_ROLE_GROUP,
                teacher_obs_group=ROUGH_CTS_TEACHER_GROUP,
                student_hidden_dims=ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU,
                imitation_loss=_fall_teacher_imitation_loss(power_limit_env_cfg),
            ),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_POWER_LIMIT_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    power_limit600_kwargs = dict(power_limit_kwargs, fall_power_limit_w=600.0)
    power_limit600_env_cfg = env_cfg(**power_limit600_kwargs)
    power_limit600_actor = power_limit600_env_cfg.observations["actor"]
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_POWER_LIMIT600_TASK_ID,
        env_cfg=power_limit600_env_cfg,
        play_env_cfg=env_cfg(play=True, **power_limit600_kwargs),
        rl_cfg=bind_task_name(
            cts_rl_cfg(
                history_length=int(power_limit600_actor.history_length),
                frame_term_dims=tuple(
                    observation_term_width(name) for name in power_limit600_actor.terms
                ),
                target_group=ROUGH_VX_OBSERVER_TARGET_GROUP,
                role_group=ROUGH_CTS_ROLE_GROUP,
                teacher_obs_group=ROUGH_CTS_TEACHER_GROUP,
                student_hidden_dims=ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU,
                imitation_loss=_fall_teacher_imitation_loss(power_limit600_env_cfg),
            ),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_POWER_LIMIT600_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    alpha800_kwargs = dict(power_limit600_kwargs, fall_teacher_alpha_iters=(400, 600, 800))
    alpha800_env_cfg = env_cfg(**alpha800_kwargs)
    alpha800_actor = alpha800_env_cfg.observations["actor"]
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_POWER_LIMIT600_ALPHA800_TASK_ID,
        env_cfg=alpha800_env_cfg,
        play_env_cfg=env_cfg(play=True, **alpha800_kwargs),
        rl_cfg=bind_task_name(
            cts_rl_cfg(
                history_length=int(alpha800_actor.history_length),
                frame_term_dims=tuple(
                    observation_term_width(name) for name in alpha800_actor.terms
                ),
                target_group=ROUGH_VX_OBSERVER_TARGET_GROUP,
                role_group=ROUGH_CTS_ROLE_GROUP,
                teacher_obs_group=ROUGH_CTS_TEACHER_GROUP,
                student_hidden_dims=ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU,
                imitation_loss=_fall_teacher_imitation_loss(alpha800_env_cfg),
            ),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_POWER_LIMIT600_ALPHA800_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    recover_kwargs = dict(
        fall_teacher_kwargs, fall_teacher_alpha_iters=(400, 600, 800), recover_group=True
    )
    recover_env_cfg = env_cfg(**recover_kwargs)
    recover_actor = recover_env_cfg.observations["actor"]
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_TASK_ID,
        env_cfg=recover_env_cfg,
        play_env_cfg=env_cfg(play=True, **recover_kwargs),
        rl_cfg=bind_task_name(
            cts_rl_cfg(
                history_length=int(recover_actor.history_length),
                frame_term_dims=tuple(observation_term_width(name) for name in recover_actor.terms),
                target_group=ROUGH_VX_OBSERVER_TARGET_GROUP,
                role_group=ROUGH_CTS_ROLE_GROUP,
                teacher_obs_group=ROUGH_CTS_TEACHER_GROUP,
                student_hidden_dims=ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU,
                imitation_loss=_fall_teacher_imitation_loss(recover_env_cfg),
            ),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    yaw_gate_kwargs = dict(recover_kwargs, yaw_gate=True)
    yaw_gate_env_cfg = env_cfg(**yaw_gate_kwargs)
    yaw_gate_actor = yaw_gate_env_cfg.observations["actor"]
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_YAW_GATE_TASK_ID,
        env_cfg=yaw_gate_env_cfg,
        play_env_cfg=env_cfg(play=True, **yaw_gate_kwargs),
        rl_cfg=bind_task_name(
            cts_rl_cfg(
                history_length=int(yaw_gate_actor.history_length),
                frame_term_dims=tuple(
                    observation_term_width(name) for name in yaw_gate_actor.terms
                ),
                target_group=ROUGH_VX_OBSERVER_TARGET_GROUP,
                role_group=ROUGH_CTS_ROLE_GROUP,
                teacher_obs_group=ROUGH_CTS_TEACHER_GROUP,
                student_hidden_dims=ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU,
                imitation_loss=_fall_teacher_imitation_loss(yaw_gate_env_cfg),
            ),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_YAW_GATE_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    recover_amp_kwargs = dict(
        stair_vx08_kwargs, fall_lagrange=False, recover_group=True, recover_amp=True
    )
    recover_amp_env_cfg = env_cfg(**recover_amp_kwargs)
    recover_amp_actor = recover_amp_env_cfg.observations["actor"]
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_AMP_TASK_ID,
        env_cfg=recover_amp_env_cfg,
        play_env_cfg=env_cfg(play=True, **recover_amp_kwargs),
        rl_cfg=bind_task_name(
            cts_rl_cfg(
                history_length=int(recover_amp_actor.history_length),
                frame_term_dims=tuple(
                    observation_term_width(name) for name in recover_amp_actor.terms
                ),
                target_group=ROUGH_VX_OBSERVER_TARGET_GROUP,
                role_group=ROUGH_CTS_ROLE_GROUP,
                teacher_obs_group=ROUGH_CTS_TEACHER_GROUP,
                student_hidden_dims=ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU,
                amp_cfg=recover_amp_cfg(
                    obs_group=ROUGH_RECOVER_AMP_GROUP, mask_obs_group=ROUGH_RECOVER_AMP_MASK_GROUP
                ),
            ),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_AMP_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_AMP_DISTILL_TASK_ID,
        env_cfg=env_cfg(**recover_amp_kwargs),
        play_env_cfg=env_cfg(play=True, **recover_amp_kwargs),
        rl_cfg=bind_task_name(
            cts_rl_cfg(
                history_length=int(recover_amp_actor.history_length),
                frame_term_dims=tuple(
                    observation_term_width(name) for name in recover_amp_actor.terms
                ),
                target_group=ROUGH_VX_OBSERVER_TARGET_GROUP,
                role_group=ROUGH_CTS_ROLE_GROUP,
                teacher_obs_group=ROUGH_CTS_TEACHER_GROUP,
                student_hidden_dims=ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU,
                policy_distill=_recover_policy_distill(recover_amp_env_cfg),
                amp_cfg=recover_amp_cfg(
                    obs_group=ROUGH_RECOVER_AMP_GROUP, mask_obs_group=ROUGH_RECOVER_AMP_MASK_GROUP
                ),
            ),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_AMP_DISTILL_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    persist_bc_kwargs = dict(
        alpha800_kwargs,
        fall_teacher_supervise_after_handoff=True,
        fall_teacher_target_leash_rad=0.3,
    )
    persist_bc_env_cfg = env_cfg(**persist_bc_kwargs)
    persist_bc_actor = persist_bc_env_cfg.observations["actor"]
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_POWER_LIMIT600_ALPHA800_PERSIST_BC_TASK_ID,
        env_cfg=persist_bc_env_cfg,
        play_env_cfg=env_cfg(play=True, **persist_bc_kwargs),
        rl_cfg=bind_task_name(
            cts_rl_cfg(
                history_length=int(persist_bc_actor.history_length),
                frame_term_dims=tuple(
                    observation_term_width(name) for name in persist_bc_actor.terms
                ),
                target_group=ROUGH_VX_OBSERVER_TARGET_GROUP,
                role_group=ROUGH_CTS_ROLE_GROUP,
                teacher_obs_group=ROUGH_CTS_TEACHER_GROUP,
                student_hidden_dims=ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU,
                imitation_loss=_fall_teacher_imitation_loss(
                    persist_bc_env_cfg, alpha_weighted=False
                ),
            ),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_POWER_LIMIT600_ALPHA800_PERSIST_BC_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    power_limit_mlp_kwargs = dict(
        stair_lin_vel_x_min=ROUGH_STAIR_LIN_VEL_X_MIN_VX08,
        fall_lagrange=False,
        fall_teacher=True,
        fall_power_limit=True,
    )
    power_limit_mlp_env_cfg = env_cfg(**power_limit_mlp_kwargs)
    register_mjlab_task(
        task_id=EXP_STAIR_VX08_FALL_TEACHER_BC_POWER_LIMIT_MLP_TASK_ID,
        env_cfg=power_limit_mlp_env_cfg,
        play_env_cfg=env_cfg(play=True, **power_limit_mlp_kwargs),
        rl_cfg=bind_task_name(
            rl_cfg(imitation_loss=_fall_teacher_imitation_loss(power_limit_mlp_env_cfg)),
            EXP_STAIR_VX08_FALL_TEACHER_BC_POWER_LIMIT_MLP_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    power_limit600_mlp_kwargs = dict(power_limit_mlp_kwargs, fall_power_limit_w=600.0)
    power_limit600_mlp_env_cfg = env_cfg(**power_limit600_mlp_kwargs)
    register_mjlab_task(
        task_id=EXP_STAIR_VX08_FALL_TEACHER_BC_POWER_LIMIT600_MLP_TASK_ID,
        env_cfg=power_limit600_mlp_env_cfg,
        play_env_cfg=env_cfg(play=True, **power_limit600_mlp_kwargs),
        rl_cfg=bind_task_name(
            rl_cfg(imitation_loss=_fall_teacher_imitation_loss(power_limit600_mlp_env_cfg)),
            EXP_STAIR_VX08_FALL_TEACHER_BC_POWER_LIMIT600_MLP_TASK_ID,
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
            ),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_LOW_GAINS_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )


__all__ = [
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_YAW_GATE_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_POWER_LIMIT600_ALPHA800_PERSIST_BC_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_POWER_LIMIT600_ALPHA800_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_POWER_LIMIT600_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_POWER_LIMIT_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_LOW_GAINS_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_AMP_DISTILL_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_AMP_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_STUDENT_PPO_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_TASK_ID",
    "EXP_CTS_NOSCAN_TASK_ID",
    "EXP_CTS_TASK_ID",
    "EXP_STAIR_VX08_FALL_TEACHER_BC_POWER_LIMIT600_MLP_TASK_ID",
    "EXP_STAIR_VX08_FALL_TEACHER_BC_POWER_LIMIT_MLP_TASK_ID",
    "GRU_TASK_ID",
    "STAIR_EVAL_TASK_ID",
    "TASK_ID",
    "register",
]
