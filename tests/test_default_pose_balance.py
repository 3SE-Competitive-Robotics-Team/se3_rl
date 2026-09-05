"""默认站姿静平衡与 v2 高度条件默认姿态的一致性回归测试。

守护 2026-09-05 的重标定：RobotConfig.default_dof_pos、MJCF standing keyframe、
se3_shared.height_default（训练侧）与 se3_runtime._serialleg_v1（部署侧副本）必须给出同一套数值，
且默认站姿在机身水平、轮心落地时整机质心正对轮轴。
"""

from __future__ import annotations

import unittest

import mujoco
import numpy as np
import torch

from se3_runtime._serialleg_v1 import (
    HEIGHT_CONDITIONED_POLICY_DEFAULT_V2,
    height_conditioned_policy_default,
    height_conditioned_policy_default_v2,
)
from se3_shared import (
    HEIGHT_CONDITIONED_DEFAULT_STRATEGY,
    RobotConfig,
    policy_default_from_height_np,
    policy_default_from_height_torch,
)
from se3_shared.fourbar import policy_to_closedchain_passive_pos_np
from se3_shared.grounded_pose import _MJCF_PATH, GroundedPoseSolver

# Flat 线 height command 范围 0.20–0.32 m（se3_train.mdp.commands 默认 height_range）。
_HEIGHTS = np.round(np.arange(0.20, 0.3201, 0.01), 4)
# 2026-09-05 之前部署端 v1 在 0.22 m 的结果；v1 供旧 artifact 回放，不得随 v2 改动。
_LEGACY_V1_DEFAULT_AT_022 = (-0.237981949227, -1.550423887933, 0.237981949227, 1.550423887933)


class DefaultPoseBalanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = RobotConfig()
        cls.solver = GroundedPoseSolver()

    def _fk(self, height: float, q4: np.ndarray) -> tuple[float, float, float]:
        """返回机身水平时的轮心 x、轮心 z 与整机质心 x（世界系，base 在原点上方 height 处）。"""
        q6 = np.asarray([*q4, 0.0, 0.0], dtype=np.float64)
        left, right = self.solver.fk(height, q6)
        com = self.solver._center_of_mass()
        return (
            0.5 * float(left[0] + right[0]),
            0.5 * float(left[2] + right[2]),
            float(com[0]),
        )

    def test_default_pose_is_grounded_and_statically_balanced(self) -> None:
        wheel_x, wheel_z, com_x = self._fk(
            self.cfg.default_base_height,
            np.asarray(self.cfg.default_dof_pos[:4]),
        )
        self.assertAlmostEqual(wheel_z, self.solver.wheel_radius, delta=1.0e-4)
        self.assertLess(abs(com_x - wheel_x), 5.0e-4)

    def test_default_pose_equals_height_default_at_nominal_height(self) -> None:
        expected = policy_default_from_height_np(
            np.asarray([self.cfg.default_base_height]),
            self.cfg,
        )[0]
        np.testing.assert_allclose(self.cfg.default_dof_pos[:4], expected, rtol=0.0, atol=1.0e-9)

    def test_passive_joints_match_default_pose(self) -> None:
        passive = policy_to_closedchain_passive_pos_np(np.asarray(self.cfg.default_dof_pos[:4]))
        np.testing.assert_allclose(
            passive[[0, 2]],
            self.cfg.default_output_knee_pos,
            rtol=0.0,
            atol=1.0e-9,
        )
        np.testing.assert_allclose(
            passive[[1, 3]],
            self.cfg.default_coupler_pos,
            rtol=0.0,
            atol=1.0e-9,
        )

    def test_mjcf_standing_keyframe_matches_default_model_joint_pos(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(_MJCF_PATH))
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "standing")
        self.assertGreaterEqual(key_id, 0)
        qpos = model.key_qpos[key_id]
        self.assertAlmostEqual(float(qpos[2]), self.cfg.default_base_height, delta=1.0e-9)
        for joint_name, value in self.cfg.default_model_joint_pos.items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.assertGreaterEqual(joint_id, 0, joint_name)
            self.assertAlmostEqual(
                float(qpos[model.jnt_qposadr[joint_id]]),
                float(value),
                delta=1.0e-9,
                msg=joint_name,
            )

    def test_height_default_stays_grounded_and_near_balanced_across_command_range(self) -> None:
        defaults = policy_default_from_height_np(_HEIGHTS, self.cfg)
        for height, q4 in zip(_HEIGHTS, defaults, strict=True):
            wheel_x, wheel_z, com_x = self._fk(float(height), q4)
            self.assertAlmostEqual(
                wheel_z, self.solver.wheel_radius, delta=1.0e-4, msg=f"h={height}"
            )
            # 常量轮心 x 目标在 0.32 m 处残差约 2.4 mm（俯仰约 0.6°），远小于旧默认的 11–17 mm。
            self.assertLess(abs(com_x - wheel_x), 3.0e-3, msg=f"h={height}")

    def test_torch_and_numpy_height_defaults_agree(self) -> None:
        expected = policy_default_from_height_np(_HEIGHTS, self.cfg)
        actual = policy_default_from_height_torch(
            torch.as_tensor(_HEIGHTS, dtype=torch.float32),
            self.cfg,
        )
        np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=0.0, atol=2.0e-5)

    def test_runtime_v2_replica_matches_training_and_v1_is_untouched(self) -> None:
        self.assertEqual(HEIGHT_CONDITIONED_DEFAULT_STRATEGY, HEIGHT_CONDITIONED_POLICY_DEFAULT_V2)
        limits = self.cfg.active_rod_angle_limits
        expected = policy_default_from_height_np(_HEIGHTS, self.cfg)
        np.testing.assert_allclose(
            height_conditioned_policy_default_v2(_HEIGHTS, limits),
            expected,
            rtol=0.0,
            atol=1.0e-9,
        )
        legacy = height_conditioned_policy_default(0.22, limits).reshape(4)
        np.testing.assert_allclose(legacy, _LEGACY_V1_DEFAULT_AT_022, rtol=0.0, atol=1.0e-9)
        self.assertGreater(abs(float(legacy[0]) - float(self.cfg.default_dof_pos[0])), 0.05)


if __name__ == "__main__":
    unittest.main()
