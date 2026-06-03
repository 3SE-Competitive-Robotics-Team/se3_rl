"""本任务使用的跳跃课程函数。"""

from __future__ import annotations

from se3_train.mdp.jump_curriculums import (
    jump_height_curriculum,
    jump_pretrain_constraint_weight_curriculum,
    jump_prob_curriculum,
    jump_quality_weight_curriculum,
)

__all__ = [
    "jump_height_curriculum",
    "jump_pretrain_constraint_weight_curriculum",
    "jump_prob_curriculum",
    "jump_quality_weight_curriculum",
]
