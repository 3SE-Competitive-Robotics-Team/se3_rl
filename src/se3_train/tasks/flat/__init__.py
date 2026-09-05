"""平地行走任务：GRU、单帧 MLP、五帧 History-MLP，以及抖动对照实验的 Exp-* 变体。"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from se3_train.rl_cfg import bind_task_name
from se3_train.tasks.common import Se3ProfiledOnPolicyRunner

from .env_cfg import (
    FLAT_ACTION_DELAY_RANGE_ONE_TO_THREE_STEPS_S,
    FLAT_ACTION_PENALTY_WHEEL_PRICING_UNIT,
    FLAT_ACTION_SMOOTHNESS_SPRING,
    FLAT_BAD_TILT_LIMITS_TIGHT_DEG,
    FLAT_CMD_VEL_DEADBAND_WIDE,
    FLAT_CURRICULUM_ADVANCE_THRESHOLD_STRICT,
    FLAT_CURRICULUM_ANG_VEL_YAW_STEP_FINE,
    FLAT_MAX_ANG_VEL_YAW_LOW,
    FLAT_WHEEL_ACTION_SCALE,
    FLAT_WHEEL_CONTACT_WEIGHT_HEAVY,
    env_cfg,
    history_env_cfg,
)
from .rl_cfg import FLAT_LEARNING_RATE, mlp_rl_cfg, rl_cfg

TASK_ID = "SE3-WheelLegged-Flat-GRU"
MLP_TASK_ID = "SE3-WheelLegged-Flat-MLP"
HISTORY_MLP_TASK_ID = "SE3-WheelLegged-Flat-History-MLP"

# 2026-09-03 抖动对照实验（单变量，全部以 Flat-MLP 为基线，仅换一个奖励/课程/延迟参数）。
# 基线本身用 SE3-WheelLegged-Flat-MLP 换随机种子重跑，用来量 run 间方差本底。
EXP_CMD_DEADBAND_TASK_ID = "SE3-WheelLegged-Flat-Exp-CmdDeadband"
EXP_WHEEL_CONTACT_TASK_ID = "SE3-WheelLegged-Flat-Exp-WheelContact"
EXP_TILT_BARRIER_TASK_ID = "SE3-WheelLegged-Flat-Exp-TiltBarrier"
EXP_ACTION_DELAY_TASK_ID = "SE3-WheelLegged-Flat-Exp-ActionDelay"
EXP_YAW_CURRICULUM_TASK_ID = "SE3-WheelLegged-Flat-Exp-YawCurriculum"

# 2026-09-04 C 批：yaw 上限一律保持 12（真实需求），只改课程的爬升方式；
# 外加一个把 A1 与 B1 两个已确证效果合并的入口。
EXP_YAW_GATE_TASK_ID = "SE3-WheelLegged-Flat-Exp-YawGate"
EXP_CURRICULUM_RETREAT_TASK_ID = "SE3-WheelLegged-Flat-Exp-CurriculumRetreat"
EXP_YAW_STEP_TASK_ID = "SE3-WheelLegged-Flat-Exp-YawStep"
EXP_ADVANCE_THRESHOLD_TASK_ID = "SE3-WheelLegged-Flat-Exp-AdvanceThreshold"
EXP_DEADBAND_TILT_TASK_ID = "SE3-WheelLegged-Flat-Exp-DeadbandTilt"

# 2026-09-04 动作语义改动：四维 action 直接是四根主动杆的绝对目标角。
# 契约变了（decoder serialleg_joint.v1），必须从头重训，旧 checkpoint 不可混用。
EXP_JOINT_ACTION_TASK_ID = "SE3-WheelLegged-Flat-Exp-JointAction"
# 2026-09-05 σ 平衡点实验：在 JointAction 之上只改动作罚项轮分量的定价（1/9 → 1.0，
# 即 action_rate 轮 1.0、action_smoothness 轮 2.0），对照 D2（steps24）看轮 σ 是否不再回升。
EXP_JOINT_ACTION_WHEEL_PRICE_TASK_ID = "SE3-WheelLegged-Flat-Exp-JointActionWheelPrice"
# 2026-09-05 critic 解耦实验：在 WheelPrice 之上只把 critic 的 LR 固定为 actor 初始值 6.5e-4，
# actor 仍走 KL 自适应。D4 诊断：σ 缩小后 KL 规则把共用 LR 压到 1e-5 地板，critic 被一起冻住，
# 4250 轮后 Loss/value 出现最高 27 的尖峰；critic 的回归目标与策略信任域无关，不该被限速。
EXP_JOINT_ACTION_WHEEL_PRICE_CRITIC_LR_TASK_ID = (
    "SE3-WheelLegged-Flat-Exp-JointActionWheelPriceCriticLr"
)

# Flat 三任务与 Exp-* 共享的基线契约：轮 scale 15 + 弹簧时代 action_smoothness 定价。
_FLAT_SPRING_BASE = {
    "wheel_action_scale": FLAT_WHEEL_ACTION_SCALE,
    "action_smoothness": FLAT_ACTION_SMOOTHNESS_SPRING,
}


def _register_flat_mlp_variant(
    task_id: str, *, rl_kwargs: dict[str, object] | None = None, **env_kwargs: object
) -> None:
    """注册一个只改单个旋钮的 Flat-MLP 变体；env 旋钮走 env_kwargs，PPO 旋钮走 rl_kwargs。"""
    register_mjlab_task(
        task_id=task_id,
        env_cfg=env_cfg(**_FLAT_SPRING_BASE, **env_kwargs),  # type: ignore[arg-type]
        play_env_cfg=env_cfg(play=True, **_FLAT_SPRING_BASE, **env_kwargs),  # type: ignore[arg-type]
        rl_cfg=bind_task_name(mlp_rl_cfg(**(rl_kwargs or {})), task_id),  # type: ignore[arg-type]
        runner_cls=Se3ProfiledOnPolicyRunner,
    )


def register() -> None:
    """注册平地 GRU、单帧 MLP、五帧 History-MLP 与 5 个 Exp-* 对照变体。"""
    # 2026-09-02 轮 scale 45→15 + 动作空间罚项 (15/45)² 补偿，以及 action_smoothness 弹簧时代重定价
    # （-0.12/cap 320/轮 2×）：仅 Flat 三个任务生效，env_cfg() 默认仍为 45 + LEGACY 供
    # rough/stair/jump/flow_match 继承线保持旧契约。
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg(
            wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
            action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
        ),
        play_env_cfg=env_cfg(
            play=True,
            wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
            action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
        ),
        rl_cfg=bind_task_name(rl_cfg(), TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=MLP_TASK_ID,
        env_cfg=env_cfg(
            wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
            action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
        ),
        play_env_cfg=env_cfg(
            play=True,
            wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
            action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
        ),
        rl_cfg=bind_task_name(mlp_rl_cfg(), MLP_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    # 五帧展平历史 MLP：网络与 PPO 配置同 Flat-MLP，仅 actor 观测为 34×5=170 维。
    register_mjlab_task(
        task_id=HISTORY_MLP_TASK_ID,
        env_cfg=history_env_cfg(
            wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
            action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
        ),
        play_env_cfg=history_env_cfg(
            play=True,
            wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
            action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
        ),
        rl_cfg=bind_task_name(mlp_rl_cfg(), HISTORY_MLP_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    # A1 放宽 command_velocity_error 死区：最大单项罚（占总罚 38%）且 2k 后不再下降，
    # 0.05 m/s 对轮式倒立摆物理不可达，等于在直接奖励高增益速度反馈。
    _register_flat_mlp_variant(
        EXP_CMD_DEADBAND_TASK_ID,
        command_velocity_deadband=FLAT_CMD_VEL_DEADBAND_WIDE,
    )
    # A2 加重轮离地罚：唯一一个越训越差的项（轮离地率 3%→6.6%），-10 拦不住策略去蹦。
    _register_flat_mlp_variant(
        EXP_WHEEL_CONTACT_TASK_ID,
        flat_wheel_contact_weight=FLAT_WHEEL_CONTACT_WEIGHT_HEAVY,
    )
    # B1 收紧倾角 barrier：实测平均倾角已在 12° 附近，soft 10° 起价太晚只收 0.056。
    _register_flat_mlp_variant(
        EXP_TILT_BARRIER_TASK_ID,
        bad_tilt_limits_deg=FLAT_BAD_TILT_LIMITS_TIGHT_DEG,
    )
    # B2 动作延迟拉到 1-3 个控制步：现在只有 4-6 ms，不足一个 20 ms 控制步，
    # 高频动作在训练里几乎没有真实代价。
    _register_flat_mlp_variant(
        EXP_ACTION_DELAY_TASK_ID,
        action_delay_range_s=FLAT_ACTION_DELAY_RANGE_ONE_TO_THREE_STEPS_S,
    )
    # B3 压 yaw 课程上限：课程已顶到 12 rad/s（688°/s，差速可行域上限），
    # 把策略逼进高增益区，零指令时用的是同一套权重。
    _register_flat_mlp_variant(
        EXP_YAW_CURRICULUM_TASK_ID,
        max_ang_vel_yaw=FLAT_MAX_ANG_VEL_YAW_LOW,
    )
    # C1 yaw 上限改由 yaw 跟踪 EMA 独立驱动；此前由线速度跟踪分推进，与 yaw 能力无关。
    _register_flat_mlp_variant(EXP_YAW_GATE_TASK_ID, curriculum_yaw_gate=True)
    # C2 课程可回退（滞回）；此前只扩不缩，冲过头后锁死在 yaw 9。
    _register_flat_mlp_variant(EXP_CURRICULUM_RETREAT_TASK_ID, curriculum_retreat=True)
    # C3 yaw 步长 1.0 → 0.25，到顶需 48 次推进而非 12 次。
    _register_flat_mlp_variant(
        EXP_YAW_STEP_TASK_ID,
        curriculum_ang_vel_yaw_step=FLAT_CURRICULUM_ANG_VEL_YAW_STEP_FINE,
    )
    # C4 推进阈值 0.5 → 0.75；cmd=0 阶段该分数反映站立稳定度，0.5 太容易过。
    _register_flat_mlp_variant(
        EXP_ADVANCE_THRESHOLD_TASK_ID,
        curriculum_advance_threshold=FLAT_CURRICULUM_ADVANCE_THRESHOLD_STRICT,
    )
    # C5 合并 A1 与 B1：两者各自把 action_rate raw 降 23-25%，机制不同，测叠加。
    _register_flat_mlp_variant(
        EXP_DEADBAND_TILT_TASK_ID,
        command_velocity_deadband=FLAT_CMD_VEL_DEADBAND_WIDE,
        bad_tilt_limits_deg=FLAT_BAD_TILT_LIMITS_TIGHT_DEG,
    )
    # 动作语义：四维直接是四根主动杆的绝对目标角，去掉夹角中间量与解码器夹紧。
    _register_flat_mlp_variant(EXP_JOINT_ACTION_TASK_ID, leg_action_semantics="joint")
    # 动作罚项轮分量按归一化动作单位计价，撤销 (15/45)² 折价；其余与 Exp-JointAction 逐项相同。
    _register_flat_mlp_variant(
        EXP_JOINT_ACTION_WHEEL_PRICE_TASK_ID,
        leg_action_semantics="joint",
        action_penalty_wheel_pricing=FLAT_ACTION_PENALTY_WHEEL_PRICING_UNIT,
    )
    # critic 固定 LR，其余与 Exp-JointActionWheelPrice 逐项相同（唯一差异在 algorithm 配置）。
    _register_flat_mlp_variant(
        EXP_JOINT_ACTION_WHEEL_PRICE_CRITIC_LR_TASK_ID,
        rl_kwargs={"critic_learning_rate": FLAT_LEARNING_RATE},
        leg_action_semantics="joint",
        action_penalty_wheel_pricing=FLAT_ACTION_PENALTY_WHEEL_PRICING_UNIT,
    )


__all__ = [
    "EXP_ACTION_DELAY_TASK_ID",
    "EXP_ADVANCE_THRESHOLD_TASK_ID",
    "EXP_CMD_DEADBAND_TASK_ID",
    "EXP_CURRICULUM_RETREAT_TASK_ID",
    "EXP_DEADBAND_TILT_TASK_ID",
    "EXP_JOINT_ACTION_TASK_ID",
    "EXP_JOINT_ACTION_WHEEL_PRICE_CRITIC_LR_TASK_ID",
    "EXP_JOINT_ACTION_WHEEL_PRICE_TASK_ID",
    "EXP_TILT_BARRIER_TASK_ID",
    "EXP_WHEEL_CONTACT_TASK_ID",
    "EXP_YAW_CURRICULUM_TASK_ID",
    "EXP_YAW_GATE_TASK_ID",
    "EXP_YAW_STEP_TASK_ID",
    "FLAT_ACTION_SMOOTHNESS_SPRING",
    "FLAT_WHEEL_ACTION_SCALE",
    "HISTORY_MLP_TASK_ID",
    "MLP_TASK_ID",
    "TASK_ID",
    "env_cfg",
    "history_env_cfg",
    "mlp_rl_cfg",
    "register",
    "rl_cfg",
]
