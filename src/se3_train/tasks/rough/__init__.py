"""崎岖地形行走任务（MLP / GRU）、台阶定向评测入口，CTS → Student128 部署链入口（Student128 即最终配方），以及最终配方的来源入口 Recover / Recover-AMP。"""

import copy
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
    ROUGH_RECOVER_BASE_IMPACT_WEIGHT_STRONG,
    ROUGH_RECOVER_MASK_GROUP,
    ROUGH_RECOVER_POTENTIAL_GAMMA,
    ROUGH_STAIR_LIN_VEL_X_MIN_VX08,
    ROUGH_VX_OBSERVER_TARGET_GROUP,
    env_cfg,
)
from .rl_cfg import (
    ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU,
    ROUGH_FALL_TEACHER_IMITATION_LOSS_COEF,
    ROUGH_FALL_TEACHER_IMITATION_LOSS_HUBER_DELTA,
    ROUGH_RECOVER_DISTILL_TEACHER_CKPT,
    ROUGH_RECOVER_JUNE_DISTILL_TEACHER_CKPT,
    cts_rl_cfg,
    gru_rl_cfg,
    recover_amp_cfg,
    rl_cfg,
)
from .terrains import flat_only_terrains_cfg, stair_only_terrains_cfg

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
# 2026-10-25 起即最终配方（原 Recover-AMP-Distill，mb6sjulr）：env 默认的 recover 组 + AMP 起身风格，加 recover 组在线蒸馏
# gyi9exiz model_2200（见 rl_cfg.ROUGH_RECOVER_DISTILL_*，全程保持）。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128"
)
# 2026-11-04 用户定：Student128 的起身改学 6saj3n6y model_4600——关掉 AMP，recover 组奖励换成老师训练时那套
# （6 月版 + 悬空转速罚 + 触地速度罚，见 env_cfg.ROUGH_RECOVER_JUNE_*），在线蒸馏老师换成
# rl_cfg.ROUGH_RECOVER_JUNE_DISTILL_TEACHER_CKPT，地形与走路组同 Student128。临时入口，结论后删除。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_JUNE_TOUCHDOWN_DISTILL_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-JuneTouchdown-Distill"
)
# 2026-10-24 用户定：recover 组——倒地起身单独成组、用自己的奖励（不按时间扣分、限时站稳、超限线性罚，见 env_cfg.ROUGH_RECOVER_*），
# 走路组不再倒地出生；倒地脚本教师（α 400–600–800、模仿损失乘 α）只在 recover 组前期混合（gyi9exiz）。
# 保留原因：它的 model_2200 是最终配方在线蒸馏的老师与 AMP 参考数据来源，要能复现。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-Alpha800-Recover"
)
# 2026-10-30 用户定：Recover 的单变量对照——倒地教师翻正后从 tuck（收腿停顿）换成 catch（倾角 < 65° 就伸腿到站姿、
# 轮子接住，见 env_cfg.ROUGH_FALL_TEACHER_CATCH），压翻正后机身砸地；其余同 gyi9exiz。临时入口，结论后删除。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_CATCH_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-Alpha800-Recover-Catch"
)
# 2026-10-31 用户定：Recover-Catch 加翻正截止 3 s + 只计机身的撞击罚（3 倍体重、−0.1，见 env_cfg.ROUGH_RECOVER_FLIP_DEADLINE_S
# 注释），压教师退场后的机身砸地；对照 Recover-Catch（uglay7z1）。临时入口，结论后删除。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_CATCH_IMPACT_TASK_ID = "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-Alpha800-Recover-Catch-Impact"
# 2026-10-31 用户定：Recover-Catch-Impact 的单变量对照——机身撞击罚 −0.1 → −0.2（env_cfg.ROUGH_RECOVER_BASE_IMPACT_WEIGHT_STRONG），
# 对照 ub6z8ipr。临时入口，结论后删除。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_CATCH_IMPACT02_TASK_ID = "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-Alpha800-Recover-Catch-Impact02"
# 2026-10-31 用户定：Recover-Catch-Impact02 的单变量对照——轮子落地也计撞击罚（单轮 3 倍体重、−0.2，与机身同价，见
# env_cfg.ROUGH_RECOVER_WHEEL_IMPACT_*），对照 mzxd7dmq。临时入口，结论后删除。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_CATCH_IMPACT02_WHEEL_TASK_ID = "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-FallTeacherBC-Alpha800-Recover-Catch-Impact02-Wheel"
# 2026-11-01 用户定：纯 reward 起身老师——recover 组不带脚本教师 / AMP，站稳 +1/s 与 +20 换成势函数整形（不给快翻溢价，
# 见 env_cfg.ROUGH_RECOVER_POTENTIAL_*），带翻正截止 3 s 与机身撞击罚 −0.1。对照 6iwgpa2m（无教师）与 ub6z8ipr（同截止与撞击罚、
# 带 catch 教师）。临时入口，结论后删除。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_POTENTIAL_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-Recover-Potential"
)
# 2026-11-02 用户定：Recover-Potential 加反向课程出生（起身轨迹状态库，见 env_cfg.ROUGH_RECOVER_REVERSE_*），
# 地形只留平地（terrains.flat_only_terrains_cfg：recover 老师不需要地形，省时间；recover 组随之从约 1 成 env 变成 6 成）。
# 对照 ertzp9ja（倒置始终 ≤ 0.1），相对它改了反向课程与地形两处。临时入口，结论后删除。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_POTENTIAL_REVERSE_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-Recover-Potential-Reverse"
)
# 2026-11-03 用户定：recover 组奖励与终止整套换成 6 月上车版（model_5999_gru，见 env_cfg.ROUGH_RECOVER_JUNE_*），
# 不带教师 / AMP / 蒸馏 / 反向课程，地形只留平地。对照 yn8l1fk6（势函数 + 反向课程，同平地）。临时入口，结论后删除。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_JUNE_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-Recover-June"
)
# 2026-11-03 用户定：Recover-June 的单变量对照——recover 组加腿 + 轮正功率超额罚（见 env_cfg.ROUGH_RECOVER_JUNE_SHARED_TERMS
# 后的注释）。对照 sooaiehz。临时入口，结论后删除。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_JUNE_POWER_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-Recover-June-Power"
)
# 2026-11-04 用户定：Recover-June 的单变量对照——recover 组加轮子悬空转速罚（见 env_cfg.ROUGH_RECOVER_JUNE_WHEEL_AIR_*）。
# 对照 sooaiehz。临时入口，结论后删除。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_JUNE_WHEEL_AIR_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-Recover-June-WheelAir"
)
# 2026-11-04 用户定：Recover-June-WheelAir 的单变量对照——recover 组再加轮子触地速度罚
# （见 env_cfg.ROUGH_RECOVER_JUNE_TOUCHDOWN_*）。对照 fdrykn18。临时入口，结论后删除。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_JUNE_WHEEL_AIR_TOUCHDOWN_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-Recover-June-WheelAir-Touchdown"
)
# 2026-10-24 用户定：recover 组 + AMP 起身风格，不带教师（6iwgpa2m）。最终配方续训的起点，也是消融"去掉蒸馏"一臂。
EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_AMP_TASK_ID = (
    "SE3-WheelLegged-Rough-Exp-CTS-NoScan-StairVx08-Student128-Recover-AMP"
)
# 2026-10-29 整理（用户定）：
# - 删掉低撞击起身老师一线的 8 个入口 Recover-Impact、-RandDir、-RandDir-FreeFold-Impact、-RandDir-Impact、
#   -RandDir-Impact-BaseWheel、-RandDir-Impact015-BaseWheel-TeacherStay / -LossStay / -LossStay-NoLeash，
#   以及它们专用的 recover 组撞击罚、6 月随机方向教师（FallTeacherCfg.june_random）、夹角归策略（模仿损失 back_follow）、
#   退场后只监督不执行与目标限幅。结果：撞击罚加在会退场的老师上，倒置不是照样砸地（3zqym1vn、w0wawesv 1000 轮
#   机身峰值力中位约 1.0–1.1 kN，与 gyi9exiz 的 1.26 kN 同量级）就是躺平不翻（m6rxc7v9、w0wawesv 2500 轮后、
#   zxmvuilz）；NoLeash 没训。复现用 86c81a60。
# - 删掉 10-25 四臂消融的 −AMP（Ablate-NoAMP）与 −recover 组（Ablate-NoRecoverGroup）两个入口，以及只给后者用的
#   env_cfg 开关 recover_group / fall_spawn_prob（recover 组现为唯一做法）与起身掩码项 fall_recovering_obs。复现用 86c81a60。
# 2026-10-25 整理（用户定"只留最终配方链"）：删掉 LowGains、StudentPPO、FallTeacher、FallTeacherBC、FallTeacherBC-PowerLimit /
# -PowerLimit600 / -PowerLimit600-Alpha800 / -PersistBC、两个 PowerLimit-MLP、Recover-YawGate 入口，以及倒地拉格朗日约束、
# 倒地功率终止、PersistBC（教师退场后继续监督 + 目标限幅）、学生隐向量 PPO 端到端训练、env_cfg 的 yaw_gate 开关与
# recovery_mimic 任务；Recover-AMP-Distill 并入 Student128（即最终配方，mb6sjulr）。复现被删内容用 5343bf96。
# 2026-10-06：Exp-VxObserver-SpringFF（估计器输出左右弹簧力并替代固定 300 N 做前馈，pcz5opfg）用户判断没用，入口与代码删除；
# 静站时左右反对称方向估计器只照抄上一拍、会漂（.scratch/yaw_diag/spring_probe.py）。复现用 6c4bd2a。
# 2026-10-07：Exp-VxObserver-WheelTN（轮子 T-N 包络换手册额定点口径，xdk0dnoa）并入默认（env_cfg.ROUGH_WHEEL_TORQUE_ENVELOPE），
# 入口删除；复现旧包络用 4832d15。
# 2026-10-19：诊断对照入口 DR1-Oracle / -OracleVel、Vel、Vx、VxObserver（及 -H5 / -Z3 / -YawRB05 / -WheelTN-Budget）、
# CTS-NoScan-StairVx08-JumpExitPen 已有结论，入口与专用开关删除；复现用 846c4fcd。


def _recover_policy_distill(
    cfg, teacher_checkpoint=ROUGH_RECOVER_DISTILL_TEACHER_CKPT
) -> PolicyDistillCfg:
    """recover 组网络老师蒸馏：整圈周期按本入口动作项的腿部缩放算（与倒地教师模仿损失同一套对齐）。"""
    imitation = _fall_teacher_imitation_loss(cfg)
    return PolicyDistillCfg(
        teacher_checkpoint=str(teacher_checkpoint),
        mask_group=ROUGH_RECOVER_MASK_GROUP,
        role_group=ROUGH_CTS_ROLE_GROUP,
        coef=imitation.coef,
        huber_delta=imitation.huber_delta,
        front_period=imitation.front_period,
        back_period=imitation.back_period,
        shift_back=imitation.shift_back,
    )


def _recover_amp_cfg() -> dict:
    """recover 组 AMP：判别器读 AMP 运动帧组，风格奖励与策略样本只取 recover 组掩码组。"""
    return recover_amp_cfg(
        obs_group=ROUGH_RECOVER_AMP_GROUP, mask_obs_group=ROUGH_RECOVER_MASK_GROUP
    )


def _fall_teacher_imitation_loss(cfg) -> ImitationLossCfg:
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
    # 最终配方（Student128）：env 默认即 recover 组 + AMP；rl 加 AMP 判别器与 recover 组在线蒸馏 model_2200。
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
                policy_distill=_recover_policy_distill(student128_env_cfg),
                amp_cfg=_recover_amp_cfg(),
            ),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    # Student128-JuneTouchdown-Distill：Student128 去掉 AMP，recover 组换 6saj3n6y 的奖励，蒸馏老师换 6saj3n6y model_4600。
    june_distill_kwargs = dict(
        stair_vx08_kwargs,
        recover_amp=False,
        recover_june=True,
        recover_june_wheel_air=True,
        recover_june_touchdown=True,
    )
    june_distill_env_cfg = env_cfg(**june_distill_kwargs)
    june_distill_actor = june_distill_env_cfg.observations["actor"]
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_JUNE_TOUCHDOWN_DISTILL_TASK_ID,
        env_cfg=june_distill_env_cfg,
        play_env_cfg=env_cfg(play=True, **june_distill_kwargs),
        rl_cfg=bind_task_name(
            cts_rl_cfg(
                history_length=int(june_distill_actor.history_length),
                frame_term_dims=tuple(
                    observation_term_width(name) for name in june_distill_actor.terms
                ),
                target_group=ROUGH_VX_OBSERVER_TARGET_GROUP,
                role_group=ROUGH_CTS_ROLE_GROUP,
                teacher_obs_group=ROUGH_CTS_TEACHER_GROUP,
                student_hidden_dims=ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU,
                policy_distill=_recover_policy_distill(
                    june_distill_env_cfg, ROUGH_RECOVER_JUNE_DISTILL_TEACHER_CKPT
                ),
            ),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_JUNE_TOUCHDOWN_DISTILL_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    # Recover（gyi9exiz）：recover 组 + 倒地脚本教师，不带 AMP。
    recover_kwargs = dict(
        stair_vx08_kwargs,
        fall_teacher=True,
        fall_teacher_alpha_iters=(400, 600, 800),
        recover_amp=False,
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
    # Recover-Catch：Recover 只换倒地教师的翻正后释放方式。
    catch_kwargs = dict(recover_kwargs, fall_teacher_catch=True)
    catch_env_cfg = env_cfg(**catch_kwargs)
    catch_actor = catch_env_cfg.observations["actor"]
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_CATCH_TASK_ID,
        env_cfg=catch_env_cfg,
        play_env_cfg=env_cfg(play=True, **catch_kwargs),
        rl_cfg=bind_task_name(
            cts_rl_cfg(
                history_length=int(catch_actor.history_length),
                frame_term_dims=tuple(observation_term_width(name) for name in catch_actor.terms),
                target_group=ROUGH_VX_OBSERVER_TARGET_GROUP,
                role_group=ROUGH_CTS_ROLE_GROUP,
                teacher_obs_group=ROUGH_CTS_TEACHER_GROUP,
                student_hidden_dims=ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU,
                imitation_loss=_fall_teacher_imitation_loss(catch_env_cfg),
            ),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_CATCH_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    # Recover-Catch-Impact：Recover-Catch 加翻正截止与机身撞击罚。
    impact_kwargs = dict(catch_kwargs, recover_flip_deadline=True, recover_base_impact=True)
    impact_env_cfg = env_cfg(**impact_kwargs)
    impact_actor = impact_env_cfg.observations["actor"]
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_CATCH_IMPACT_TASK_ID,
        env_cfg=impact_env_cfg,
        play_env_cfg=env_cfg(play=True, **impact_kwargs),
        rl_cfg=bind_task_name(
            cts_rl_cfg(
                history_length=int(impact_actor.history_length),
                frame_term_dims=tuple(observation_term_width(name) for name in impact_actor.terms),
                target_group=ROUGH_VX_OBSERVER_TARGET_GROUP,
                role_group=ROUGH_CTS_ROLE_GROUP,
                teacher_obs_group=ROUGH_CTS_TEACHER_GROUP,
                student_hidden_dims=ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU,
                imitation_loss=_fall_teacher_imitation_loss(impact_env_cfg),
            ),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_CATCH_IMPACT_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    # Recover-Catch-Impact02：Recover-Catch-Impact 只把撞击罚加重到 −0.2。
    impact02_kwargs = dict(
        impact_kwargs, recover_base_impact_weight=ROUGH_RECOVER_BASE_IMPACT_WEIGHT_STRONG
    )
    impact02_env_cfg = env_cfg(**impact02_kwargs)
    impact02_actor = impact02_env_cfg.observations["actor"]
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_CATCH_IMPACT02_TASK_ID,
        env_cfg=impact02_env_cfg,
        play_env_cfg=env_cfg(play=True, **impact02_kwargs),
        rl_cfg=bind_task_name(
            cts_rl_cfg(
                history_length=int(impact02_actor.history_length),
                frame_term_dims=tuple(
                    observation_term_width(name) for name in impact02_actor.terms
                ),
                target_group=ROUGH_VX_OBSERVER_TARGET_GROUP,
                role_group=ROUGH_CTS_ROLE_GROUP,
                teacher_obs_group=ROUGH_CTS_TEACHER_GROUP,
                student_hidden_dims=ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU,
                imitation_loss=_fall_teacher_imitation_loss(impact02_env_cfg),
            ),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_CATCH_IMPACT02_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    # Recover-Catch-Impact02-Wheel：Recover-Catch-Impact02 加轮子撞击罚。
    impact02_wheel_kwargs = dict(impact02_kwargs, recover_wheel_impact=True)
    impact02_wheel_env_cfg = env_cfg(**impact02_wheel_kwargs)
    impact02_wheel_actor = impact02_wheel_env_cfg.observations["actor"]
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_CATCH_IMPACT02_WHEEL_TASK_ID,
        env_cfg=impact02_wheel_env_cfg,
        play_env_cfg=env_cfg(play=True, **impact02_wheel_kwargs),
        rl_cfg=bind_task_name(
            cts_rl_cfg(
                history_length=int(impact02_wheel_actor.history_length),
                frame_term_dims=tuple(
                    observation_term_width(name) for name in impact02_wheel_actor.terms
                ),
                target_group=ROUGH_VX_OBSERVER_TARGET_GROUP,
                role_group=ROUGH_CTS_ROLE_GROUP,
                teacher_obs_group=ROUGH_CTS_TEACHER_GROUP,
                student_hidden_dims=ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU,
                imitation_loss=_fall_teacher_imitation_loss(impact02_wheel_env_cfg),
            ),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_CATCH_IMPACT02_WHEEL_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    # Recover-Potential：recover 组势函数整形 + 翻正截止 + 机身撞击罚，不带教师 / AMP / 蒸馏。
    potential_kwargs = dict(
        stair_vx08_kwargs,
        recover_amp=False,
        recover_flip_deadline=True,
        recover_base_impact=True,
        recover_potential=True,
    )
    potential_env_cfg = env_cfg(**potential_kwargs)
    potential_actor = potential_env_cfg.observations["actor"]
    potential_rl_cfg = cts_rl_cfg(
        history_length=int(potential_actor.history_length),
        frame_term_dims=tuple(observation_term_width(name) for name in potential_actor.terms),
        target_group=ROUGH_VX_OBSERVER_TARGET_GROUP,
        role_group=ROUGH_CTS_ROLE_GROUP,
        teacher_obs_group=ROUGH_CTS_TEACHER_GROUP,
        student_hidden_dims=ROUGH_CTS_STUDENT_HIDDEN_DIMS_MCU,
    )
    if potential_rl_cfg.algorithm.gamma != ROUGH_RECOVER_POTENTIAL_GAMMA:
        raise ValueError("势函数整形的 γ 须等于 PPO 的 γ（ROUGH_RECOVER_POTENTIAL_GAMMA）")
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_POTENTIAL_TASK_ID,
        env_cfg=potential_env_cfg,
        play_env_cfg=env_cfg(play=True, **potential_kwargs),
        rl_cfg=bind_task_name(
            potential_rl_cfg, EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_POTENTIAL_TASK_ID
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    # Recover-Potential-Reverse：Recover-Potential 加反向课程出生、地形只留平地，rl 同 Recover-Potential。
    reverse_kwargs = dict(
        potential_kwargs, recover_reverse=True, terrain_generator=flat_only_terrains_cfg()
    )
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_POTENTIAL_REVERSE_TASK_ID,
        env_cfg=env_cfg(**reverse_kwargs),
        play_env_cfg=env_cfg(play=True, **reverse_kwargs),
        # bind_task_name 原地改名，须另拷一份
        rl_cfg=bind_task_name(
            copy.deepcopy(potential_rl_cfg),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_POTENTIAL_REVERSE_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    # Recover-June：recover 组换 6 月上车版奖励与终止、地形只留平地，rl 同 Recover-Potential。
    june_kwargs = dict(
        stair_vx08_kwargs,
        recover_amp=False,
        recover_june=True,
        terrain_generator=flat_only_terrains_cfg(),
    )
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_JUNE_TASK_ID,
        env_cfg=env_cfg(**june_kwargs),
        play_env_cfg=env_cfg(play=True, **june_kwargs),
        rl_cfg=bind_task_name(
            copy.deepcopy(potential_rl_cfg),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_JUNE_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    # Recover-June-Power：Recover-June 加腿 + 轮正功率超额罚。
    june_power_kwargs = dict(june_kwargs, recover_june_power=True)
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_JUNE_POWER_TASK_ID,
        env_cfg=env_cfg(**june_power_kwargs),
        play_env_cfg=env_cfg(play=True, **june_power_kwargs),
        rl_cfg=bind_task_name(
            copy.deepcopy(potential_rl_cfg),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_JUNE_POWER_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    # Recover-June-WheelAir：Recover-June 加轮子悬空转速罚。
    june_wheel_air_kwargs = dict(june_kwargs, recover_june_wheel_air=True)
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_JUNE_WHEEL_AIR_TASK_ID,
        env_cfg=env_cfg(**june_wheel_air_kwargs),
        play_env_cfg=env_cfg(play=True, **june_wheel_air_kwargs),
        rl_cfg=bind_task_name(
            copy.deepcopy(potential_rl_cfg),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_JUNE_WHEEL_AIR_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    # Recover-June-WheelAir-Touchdown：Recover-June-WheelAir 再加轮子触地速度罚。
    june_touchdown_kwargs = dict(june_wheel_air_kwargs, recover_june_touchdown=True)
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_JUNE_WHEEL_AIR_TOUCHDOWN_TASK_ID,
        env_cfg=env_cfg(**june_touchdown_kwargs),
        play_env_cfg=env_cfg(play=True, **june_touchdown_kwargs),
        rl_cfg=bind_task_name(
            copy.deepcopy(potential_rl_cfg),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_JUNE_WHEEL_AIR_TOUCHDOWN_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    # Recover-AMP（6iwgpa2m）：env 默认（recover 组 + AMP），rl 只有 AMP、不蒸馏。
    recover_amp_env_cfg = env_cfg(**stair_vx08_kwargs)
    recover_amp_actor = recover_amp_env_cfg.observations["actor"]
    register_mjlab_task(
        task_id=EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_AMP_TASK_ID,
        env_cfg=recover_amp_env_cfg,
        play_env_cfg=env_cfg(play=True, **stair_vx08_kwargs),
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
                amp_cfg=_recover_amp_cfg(),
            ),
            EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_AMP_TASK_ID,
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )


__all__ = [
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_CATCH_IMPACT02_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_CATCH_IMPACT02_WHEEL_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_CATCH_IMPACT_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_CATCH_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_FALL_TEACHER_BC_ALPHA800_RECOVER_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_AMP_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_JUNE_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_POTENTIAL_REVERSE_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_RECOVER_POTENTIAL_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_STUDENT128_TASK_ID",
    "EXP_CTS_NOSCAN_STAIR_VX08_TASK_ID",
    "EXP_CTS_NOSCAN_TASK_ID",
    "EXP_CTS_TASK_ID",
    "GRU_TASK_ID",
    "STAIR_EVAL_TASK_ID",
    "TASK_ID",
    "register",
]
