"""四地形 stair teacher 任务 PPO 配置。"""

from __future__ import annotations

import os

from mjlab.rl import RslRlOnPolicyRunnerCfg

from se3_train.tasks.stair_ctbc.rl_cfg import rl_cfg as stair_ctbc_rl_cfg
from se3_train.teacher_student import (
    MaskConfig,
    StairTeacherPpoAlgorithmCfg,
    StairTeacherStudentConfig,
    TeacherCheckpointConfig,
    TeacherLossConfig,
)
from se3_train.teacher_student.config import stair_teacher_checkpoint_from_env

_STAIR_TERRAIN_TYPE_NAMES = ("stage_stairs", "inv_pyramid_stairs")


def rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """复用 stair_ctbc 的 GRU/warm-start 配置，并挂载 stair-only teacher loss。"""
    cfg = stair_ctbc_rl_cfg(smoke=smoke)
    smoke_enabled = smoke or os.environ.get("SE3_SMOKE", "0") == "1"
    cfg.experiment_name = "se3_wheel_leg_universal_stair_teacher"
    cfg.save_interval = int(os.environ.get("SE3_UNIVERSAL_STAIR_SAVE_INTERVAL", "100"))
    if smoke_enabled:
        cfg.max_iterations = 5
        cfg.resume = False
        cfg.algorithm.num_mini_batches = 1
    else:
        cfg.max_iterations = int(os.environ.get("SE3_UNIVERSAL_STAIR_MAX_ITERATIONS", "2000"))
        cfg.resume = True

    teacher_config = StairTeacherStudentConfig(
        enabled=not smoke_enabled,
        mask=MaskConfig(stair_type_names=_STAIR_TERRAIN_TYPE_NAMES),
        teacher_loss=TeacherLossConfig(
            initial_coef=float(os.environ.get("SE3_STAIR_TEACHER_COEF", "0.2")),
            end_iter=int(os.environ.get("SE3_STAIR_TEACHER_END_ITER", "1200")),
            loss_type=os.environ.get("SE3_STAIR_TEACHER_LOSS", "mse"),
        ),
        stair_teacher=TeacherCheckpointConfig(
            name="stair",
            path=stair_teacher_checkpoint_from_env(),
            strict=os.environ.get("SE3_STAIR_TEACHER_STRICT", "1").lower()
            in {"1", "true", "yes", "on"},
            enabled=True,
            actor_obs_dim=int(os.environ.get("SE3_STAIR_TEACHER_OBS_DIM", "31")),
        ),
    )
    cfg.algorithm = StairTeacherPpoAlgorithmCfg.from_base(cfg.algorithm, teacher_config)
    return cfg
