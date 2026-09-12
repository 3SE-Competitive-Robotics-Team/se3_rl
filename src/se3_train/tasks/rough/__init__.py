"""崎岖地形 MLP 行走任务（含地形课程与地形感知高度下限）。"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from se3_train.rl_cfg import bind_task_name
from se3_train.tasks.common import Se3ProfiledOnPolicyRunner, Se3WarmStartRunner

from .a12_ablation import A12_ABLATIONS, a12_ablation_env_cfg, a12_ablation_rl_cfg
from .a13_tuned import a13_env_cfg, a13_rl_cfg
from .a14_last_action import a14_env_cfg, a14_rl_cfg
from .a15_pricing import a15_env_cfg, a15_rl_cfg
from .a16_reward_swap import a16_env_cfg, a16_rl_cfg
from .a17_flat_stairs_warmstart import flat_stairs_env_cfg, flat_stairs_rl_cfg
from .a18_height_gate import (
    a18_control_env_cfg,
    a18_height_gate_env_cfg,
    a18_warmstart_rl_cfg,
)
from .a20_high_stand_transition import a20_env_cfg, a20_rl_cfg
from .a21_full_height import a21_env_cfg, a21_rl_cfg
from .a22_stair_command import a22_env_cfg, a22_rl_cfg
from .a23_stair_no_gate import a23_env_cfg, a23_rl_cfg
from .a24_stair_height import a24_env_cfg, a24_rl_cfg, a25_env_cfg
from .env_cfg import env_cfg
from .reward_ablation import REWARD_ABLATIONS, reward_ablation_env_cfg
from .rl_cfg import amp_rl_cfg, rl_cfg
from .terrains import stair_only_terrains_cfg

TASK_ID = "SE3-WheelLegged-Rough"
# 只含上/下台阶与平地的定向评测入口：地形课程照常，用 terrain level 定位策略能上到多高的台阶。
STAIR_EVAL_TASK_ID = "SE3-WheelLegged-Rough-StairEval"
# AMP：加 amp 观测组 + 判别器风格奖励（se3_train.amp），专家数据集见 docs/amp_dataset.md。
AMP_TASK_ID = "SE3-WheelLegged-Rough-AMP"
# 消融入口（2026-09-09）：A12 实测 AMP 在台阶列贡献 +9.1/秒，占该列奖励信号约 90%，
# 而判别器只分到 ±0.24、style_reward 长期贴 0.6——疑似发的是常数存活奖金而非姿态信息。
# C1 只改风格奖励权重 15→3（A2 的值）；C2 只换成时序打乱的专家数据（逐帧分布不变，
# 跨帧结构破坏，见 scripts/make_shuffled_amp_dataset.py）。其余与 AMP_TASK_ID 逐项相同。
AMP_ABL_W3_TASK_ID = "SE3-WheelLegged-Rough-AMP-AblW3"
AMP_ABL_SHUFFLED_TASK_ID = "SE3-WheelLegged-Rough-AMP-AblShuffled"
# A13：按 A12 十组单因素消融的结论拼出的配置——有利项全开、有害项全关（见 a13_tuned.py）。
A13_TASK_ID = "SE3-WheelLegged-Rough-A13"
# A14：在 A13 之上只改 reset 帧的 last_actions 不再恒为 0（见 a14_last_action.py）。
# 对照端就是 A13_TASK_ID，env 其余逐位相同。
A14_TASK_ID = "SE3-WheelLegged-Rough-A14"
# A15：把台阶列早就改对的定价扩到其余五列（见 a15_pricing.py）。实测账本显示非台阶列上
# 站着不动净赚 2.407/秒、走路净亏 2.932/秒，站着才是该奖励函数的最优解；A15 改四个数值
# 把符号翻过来。对照端同为 A13_TASK_ID。
A15_TASK_ID = "SE3-WheelLegged-Rough-A15"
# A16：stairs_up 整列的奖励换成参考实现（se3_rl_competiition 的 cloud-changes 分支）
# 那 10 项，一项不改；其余五列保持 A15。对照端是 A15。见 a16_reward_swap.py。
A16_TASK_ID = "SE3-WheelLegged-Rough-A16"
# A17：以 A15 model_4999 只加载 actor/critic，从 iteration 0 重跑；stairs_up 奖励逐项对齐 flat。
A17_TASK_ID = "SE3-WheelLegged-Rough-A15-FlatStairsWarmStart"
# A18：A15 checkpoint 的无重复平地热身对照；A19 只额外门控移动时的目标高度惩罚。
A18_CONTROL_TASK_ID = "SE3-WheelLegged-Rough-A18-NoWarmupControl"
A19_HEIGHT_GATE_TASK_ID = "SE3-WheelLegged-Rough-A19-HeightGate"
# A20：在 A19 上加入平地高姿态静站到前进的显式指令转移分布，针对 sim2x 冷启动死锁。
A20_HIGH_STAND_TRANSITION_TASK_ID = "SE3-WheelLegged-Rough-A20-HighStandTransition"
# A21：保留 A20 的启动序列，恢复 A15 完整高度惩罚，验证能否兼得启动与高度跟踪。
A21_FULL_HEIGHT_TASK_ID = "SE3-WheelLegged-Rough-A21-FullHeight"
A22_STAIR_COMMAND_TASK_ID = "SE3-WheelLegged-Rough-A22-StairCommand"
A23_STAIR_NO_GATE_TASK_ID = "SE3-WheelLegged-Rough-A23-StairNoGate"
AMP_ABL_SHUFFLED_DATASET = "assets/amp/fudan_stairs20_shuffled/amp_training.pkl"


def register() -> None:
    """注册崎岖地形行走任务与定向评测、AMP 两个入口。"""
    for task_id, factory in (
        ("SE3-WheelLegged-Rough-A24-HeightControl", a24_env_cfg),
        ("SE3-WheelLegged-Rough-A25-StairHeight", a25_env_cfg),
    ):
        register_mjlab_task(
            task_id=task_id,
            env_cfg=factory(),
            play_env_cfg=factory(play=True),
            rl_cfg=bind_task_name(a24_rl_cfg(), task_id),
            runner_cls=Se3WarmStartRunner,
        )
    register_mjlab_task(
        task_id=A23_STAIR_NO_GATE_TASK_ID,
        env_cfg=a23_env_cfg(),
        play_env_cfg=a23_env_cfg(play=True),
        rl_cfg=bind_task_name(a23_rl_cfg(), A23_STAIR_NO_GATE_TASK_ID),
        runner_cls=Se3WarmStartRunner,
    )
    register_mjlab_task(
        task_id=A22_STAIR_COMMAND_TASK_ID,
        env_cfg=a22_env_cfg(),
        play_env_cfg=a22_env_cfg(play=True),
        rl_cfg=bind_task_name(a22_rl_cfg(), A22_STAIR_COMMAND_TASK_ID),
        runner_cls=Se3WarmStartRunner,
    )
    register_mjlab_task(
        task_id=A21_FULL_HEIGHT_TASK_ID,
        env_cfg=a21_env_cfg(),
        play_env_cfg=a21_env_cfg(play=True),
        rl_cfg=bind_task_name(a21_rl_cfg(), A21_FULL_HEIGHT_TASK_ID),
        runner_cls=Se3WarmStartRunner,
    )
    register_mjlab_task(
        task_id=A20_HIGH_STAND_TRANSITION_TASK_ID,
        env_cfg=a20_env_cfg(),
        play_env_cfg=a20_env_cfg(play=True),
        rl_cfg=bind_task_name(a20_rl_cfg(), A20_HIGH_STAND_TRANSITION_TASK_ID),
        runner_cls=Se3WarmStartRunner,
    )
    register_mjlab_task(
        task_id=A18_CONTROL_TASK_ID,
        env_cfg=a18_control_env_cfg(),
        play_env_cfg=a18_control_env_cfg(play=True),
        rl_cfg=bind_task_name(a18_warmstart_rl_cfg(), A18_CONTROL_TASK_ID),
        runner_cls=Se3WarmStartRunner,
    )
    register_mjlab_task(
        task_id=A19_HEIGHT_GATE_TASK_ID,
        env_cfg=a18_height_gate_env_cfg(),
        play_env_cfg=a18_height_gate_env_cfg(play=True),
        rl_cfg=bind_task_name(a18_warmstart_rl_cfg(), A19_HEIGHT_GATE_TASK_ID),
        runner_cls=Se3WarmStartRunner,
    )
    register_mjlab_task(
        task_id=A16_TASK_ID,
        env_cfg=a16_env_cfg(),
        play_env_cfg=a16_env_cfg(play=True),
        rl_cfg=bind_task_name(a16_rl_cfg(), A16_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=A17_TASK_ID,
        env_cfg=flat_stairs_env_cfg(),
        play_env_cfg=flat_stairs_env_cfg(play=True),
        rl_cfg=bind_task_name(flat_stairs_rl_cfg(), A17_TASK_ID),
        runner_cls=Se3WarmStartRunner,
    )
    register_mjlab_task(
        task_id=A15_TASK_ID,
        env_cfg=a15_env_cfg(),
        play_env_cfg=a15_env_cfg(play=True),
        rl_cfg=bind_task_name(a15_rl_cfg(), A15_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=A14_TASK_ID,
        env_cfg=a14_env_cfg(),
        play_env_cfg=a14_env_cfg(play=True),
        rl_cfg=bind_task_name(a14_rl_cfg(), A14_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=A13_TASK_ID,
        env_cfg=a13_env_cfg(),
        play_env_cfg=a13_env_cfg(play=True),
        rl_cfg=bind_task_name(a13_rl_cfg(), A13_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    for variant in A12_ABLATIONS:
        task_id = f"SE3-WheelLegged-Rough-A12Abl-{variant}"
        register_mjlab_task(
            task_id=task_id,
            env_cfg=a12_ablation_env_cfg(variant),
            play_env_cfg=a12_ablation_env_cfg(variant, play=True),
            rl_cfg=bind_task_name(a12_ablation_rl_cfg(variant), task_id),
            runner_cls=Se3ProfiledOnPolicyRunner,
        )
    for task_id, (progress_weight, support_weight) in REWARD_ABLATIONS.items():
        register_mjlab_task(
            task_id=task_id,
            env_cfg=reward_ablation_env_cfg(
                progress_weight=progress_weight, support_weight=support_weight
            ),
            play_env_cfg=reward_ablation_env_cfg(
                progress_weight=progress_weight, support_weight=support_weight, play=True
            ),
            rl_cfg=bind_task_name(rl_cfg(), task_id),
            runner_cls=Se3ProfiledOnPolicyRunner,
        )
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=bind_task_name(rl_cfg(), TASK_ID),
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
        task_id=AMP_TASK_ID,
        env_cfg=env_cfg(amp_enabled=True),
        play_env_cfg=env_cfg(play=True, amp_enabled=True),
        rl_cfg=bind_task_name(amp_rl_cfg(), AMP_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=AMP_ABL_W3_TASK_ID,
        env_cfg=env_cfg(amp_enabled=True),
        play_env_cfg=env_cfg(play=True, amp_enabled=True),
        rl_cfg=bind_task_name(amp_rl_cfg(reward_weight=3.0), AMP_ABL_W3_TASK_ID),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=AMP_ABL_SHUFFLED_TASK_ID,
        env_cfg=env_cfg(amp_enabled=True),
        play_env_cfg=env_cfg(play=True, amp_enabled=True),
        rl_cfg=bind_task_name(
            amp_rl_cfg(dataset_root=AMP_ABL_SHUFFLED_DATASET), AMP_ABL_SHUFFLED_TASK_ID
        ),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )


__all__ = [
    "A13_TASK_ID",
    "A14_TASK_ID",
    "A15_TASK_ID",
    "A16_TASK_ID",
    "A17_TASK_ID",
    "A18_CONTROL_TASK_ID",
    "A19_HEIGHT_GATE_TASK_ID",
    "AMP_ABL_SHUFFLED_DATASET",
    "AMP_ABL_SHUFFLED_TASK_ID",
    "AMP_ABL_W3_TASK_ID",
    "AMP_TASK_ID",
    "STAIR_EVAL_TASK_ID",
    "TASK_ID",
    "env_cfg",
    "register",
    "rl_cfg",
]
