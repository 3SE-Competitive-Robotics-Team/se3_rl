"""地形相对高度口径的回归测试。

2026-09-06 R1 崎岖地形训练在 1000 轮内崩溃：机身陷进 heightfield 后
`TerrainHeightSensor.heights` 被 backface 钳成 0.0，无界的 `flat_base_height`
二次罚给出 −144/s（正常总奖励约 −10/s），critic 回归目标进万级，策略被摧毁。
本文件钉住三条修复：平地口径不变、哨兵读数不再进入奖励、单项罚有上限。
"""

from __future__ import annotations

import unittest

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import se3_train  # noqa: F401  # 注册任务
from se3_train.mdp.rewards import flat_base_height_penalty_no_jump
from se3_train.mdp.terrain_height import frame_height_above_terrain
from se3_train.tasks.rough.env_cfg import env_cfg as rough_env_cfg

_SENSOR = "base_height_sensor"
_FLAT_BASE_HEIGHT_WEIGHT = -4.0
_SIGMA = 0.05
_MAX_ERROR = 0.15


def _place(env, dz: float) -> None:
    """把机器人放到各自地形原点上方 dz 处，并刷新传感器。"""
    robot = env.scene["robot"]
    quat = robot.data.root_link_pose_w[:, 3:7].clone()
    pos = env.scene.env_origins.clone()
    pos[:, 2] += dz
    for _ in range(2):  # 写两次，确保 sense() 看到的是新位姿
        robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1))
        env.sim.forward()
        env.sim.sense()


class FlatEquivalenceTests(unittest.TestCase):
    """平地上新口径必须与旧的逐射线净空逐位相同，否则冻结的 Flat 基线就被改了。"""

    @classmethod
    def setUpClass(cls) -> None:
        cfg = load_env_cfg("SE3-WheelLegged-Flat-MLP")
        cfg.scene.num_envs = 4
        cls.env = ManagerBasedRlEnv(cfg, device="cpu")
        cls.env.reset()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_matches_raw_sensor_heights_on_plane(self) -> None:
        for dz in (0.20, 0.30, 0.38, 0.50):
            _place(self.env, dz)
            old = self.env.scene[_SENSOR].data.heights[:, 0]
            new = frame_height_above_terrain(self.env, _SENSOR)
            self.assertLess(float((old - new).abs().max()), 1e-5, msg=f"dz={dz}")


class DegenerateReadingTests(unittest.TestCase):
    """实体地形上的两种哨兵读数必须被挡在奖励之外。"""

    @classmethod
    def setUpClass(cls) -> None:
        cfg = rough_env_cfg()
        cfg.scene.num_envs = 4
        cls.env = ManagerBasedRlEnv(cfg, device="cpu")
        cls.env.reset()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def _height_and_reward(self, dz: float) -> tuple[float, float]:
        _place(self.env, dz)
        term = self.env.command_manager.get_term("velocity_height")
        term._command[:, 4] = 0.30  # 高度指令
        term._command[:, 5] = 0.0  # jump_flag
        height = frame_height_above_terrain(self.env, _SENSOR)
        penalty = flat_base_height_penalty_no_jump(
            self.env, "velocity_height", _SENSOR, sigma=_SIGMA, max_error=_MAX_ERROR
        )
        return float(height[0]), float(_FLAT_BASE_HEIGHT_WEIGHT * penalty[0])

    def test_sunk_into_terrain_is_not_reported_as_zero_clearance(self) -> None:
        # backface：旧口径把净空钳成 0.0，看起来像"贴地站着"，实际是陷进去了。
        raw_before = None
        for dz in (-0.05, -0.20):
            _place(self.env, dz)
            raw = float(self.env.scene[_SENSOR].data.heights[0, 0])
            self.assertAlmostEqual(raw, 0.0, places=6, msg="前提：原始读数确实是哨兵 0")
            raw_before = raw
            height, _ = self._height_and_reward(dz)
            self.assertLess(height, 0.0, msg=f"dz={dz} 应给出负高度而不是 {raw_before}")
            self.assertAlmostEqual(height, dz, delta=0.02, msg=f"dz={dz}")

    def test_airborne_is_not_clipped_to_max_distance(self) -> None:
        # 射线全部打空：旧口径返回 max_distance=2.0，与真实高度 2.5 差 0.5 m。
        _place(self.env, 2.5)
        raw = float(self.env.scene[_SENSOR].data.heights[0, 0])
        self.assertAlmostEqual(raw, 2.0, places=6, msg="前提：原始读数确实是哨兵 max_distance")
        height, _ = self._height_and_reward(2.5)
        self.assertAlmostEqual(height, 2.5, delta=0.05)

    def test_penalty_is_bounded_by_the_error_clamp(self) -> None:
        # 上限 = weight * (max_error / sigma)^2 = -4.0 * (0.15/0.05)^2 = -36
        bound = _FLAT_BASE_HEIGHT_WEIGHT * (_MAX_ERROR / _SIGMA) ** 2
        for dz in (-0.05, -0.20, 2.5):
            _, reward = self._height_and_reward(dz)
            self.assertGreaterEqual(reward, bound - 1e-4, msg=f"dz={dz}")
        # 正常高度不该被罚。
        _, reward = self._height_and_reward(0.30)
        self.assertAlmostEqual(reward, 0.0, places=4)


class RewardWiringTests(unittest.TestCase):
    def test_flat_base_height_ships_with_the_clamp(self) -> None:
        # 默认配置必须带上夹紧；去掉它等于把 R1 的崩溃条件放回来。
        cfg = load_env_cfg("SE3-WheelLegged-Rough")
        params = cfg.rewards["flat_base_height"].params
        self.assertEqual(params.get("max_error", _MAX_ERROR), _MAX_ERROR)


if __name__ == "__main__":
    unittest.main()
