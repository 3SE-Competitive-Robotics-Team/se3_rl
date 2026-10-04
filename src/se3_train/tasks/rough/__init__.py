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
# 2026-10-04：Exp-Orient24（tracking_orientation_l2 −24）并入默认（env_cfg.ROUGH_ORIENTATION_WEIGHT），入口删除；
# 复现 Orient24 / Base128 用 aeec423。
# 2026-10-03（用户定）：加宽域随机化对照——真 Kp/Kd、质心 ±5 cm、质量 −1…+3 kg、恢复系数 0–1、气弹簧 ×0.9–1.5、
# 动作延迟 0–10 ms，其余与默认相同（见 env_cfg.ROUGH_WIDE_DR_* 注释）。临时入口：对照结束、定下默认后删除，复现用对应 commit。
EXP_WIDE_DR_TASK_ID = "SE3-WheelLegged-Rough-Exp-WideDR"
# 2026-10-04（用户定）：WideDR 的 Oracle 诊断——actor 额外观测真实 DR 参数 32 维，其余与 WideDR 相同（见 env_cfg._apply_oracle_dr_obs）。
# 不可部署；判别 WideDR 退化是信息不足还是容量 / 物理可行性问题。临时入口，结论后删除。
EXP_WIDE_DR_ORACLE_TASK_ID = "SE3-WheelLegged-Rough-Exp-WideDR-Oracle"
# 2026-10-04（用户定）：高姿态起步序列占比对照——high_stand_transition_prob 0.5 → 0.1，其余与默认（姿态罚 −24）相同。
# 依据：中等高度（0.26–0.34 m）0.5–1.5 m/s 稳态机身比指令高 4–6 cm，反事实账本显示贴合指令高度时奖励反而更高（没学好），
# 而任意时刻约 26% env 处在 0.36–0.38 m 高姿态静站 → 前进序列中（docs/plan/rough_orient24_20261004.md）。
# 风险：该序列是 M8 为解 0.38 m 静站起步的探索瓶颈加的，评测必须测高姿态静站后起步。临时入口，结论后删除。
EXP_HIGH_STAND01_TASK_ID = "SE3-WheelLegged-Rough-Exp-HighStand01"
EXP_HIGH_STAND01_PROB = 0.1


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
        task_id=EXP_WIDE_DR_TASK_ID,
        env_cfg=env_cfg(wide_dr=True),
        play_env_cfg=env_cfg(play=True, wide_dr=True),
        rl_cfg=bind_task_name(rl_cfg(), EXP_WIDE_DR_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_WIDE_DR_ORACLE_TASK_ID,
        env_cfg=env_cfg(wide_dr=True, oracle_dr_obs=True),
        play_env_cfg=env_cfg(play=True, wide_dr=True, oracle_dr_obs=True),
        rl_cfg=bind_task_name(rl_cfg(), EXP_WIDE_DR_ORACLE_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=EXP_HIGH_STAND01_TASK_ID,
        env_cfg=env_cfg(high_stand_transition_prob=EXP_HIGH_STAND01_PROB),
        play_env_cfg=env_cfg(play=True, high_stand_transition_prob=EXP_HIGH_STAND01_PROB),
        rl_cfg=bind_task_name(rl_cfg(), EXP_HIGH_STAND01_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )


__all__ = [
    "EXP_HIGH_STAND01_PROB",
    "EXP_HIGH_STAND01_TASK_ID",
    "EXP_WIDE_DR_ORACLE_TASK_ID",
    "EXP_WIDE_DR_TASK_ID",
    "GRU_TASK_ID",
    "STAIR_EVAL_TASK_ID",
    "TASK_ID",
    "register",
]
