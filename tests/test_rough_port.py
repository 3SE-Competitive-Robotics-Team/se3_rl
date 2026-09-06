"""崎岖地形任务的移植回归测试。

来源：scutrobotlab/wheeled-legged_RL 的 V14 rough 线（地形课程 + step_up 状态机）。
移植原则是 rough 只相对冻结的 Flat 基线改地形、地形课程、台阶状态机和能耗三项定价，
其余逐项不动；本文件把这几条钉住。要改 rough 就一并改这里，并写清对照实验编号。
"""

from __future__ import annotations

import unittest

from mjlab.sensor import GridPatternCfg, TerrainHeightSensorCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import se3_train  # noqa: F401  # 注册任务
from se3_train.tasks.flat.env_cfg import (
    FLAT_ACTION_SMOOTHNESS_SPRING,
    FLAT_WHEEL_ACTION_SCALE,
)
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg
from se3_train.tasks.rough.commands import StepUpCommandCfg
from se3_train.tasks.rough.env_cfg import (
    ROUGH_ENERGY_PENALTY_SCALE,
    ROUGH_STEP_UP_LOOKAHEAD_M,
)
from se3_train.tasks.rough.terrains import _STEP_HEIGHT_RANGE

_ROUGH = "SE3-WheelLegged-Rough"
_STAIR_EVAL = "SE3-WheelLegged-Rough-StairEval"
_NO_STEP_UP = "SE3-WheelLegged-Rough-NoStepUp"
_FLAT_MLP = "SE3-WheelLegged-Flat-MLP"

# 参考仓库 rough 相对 flat 只放松能耗类罚项（wheel_power / joint_torque 各 ÷10）。
_ENERGY_REWARDS = ("leg_torques", "wheel_torques", "leg_power")
_SENSOR_NAME = "wheel_forward_sensor"


class RoughInheritsFlatBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_env_cfg(_ROUGH)
        cls.flat = flat_env_cfg(
            wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
            action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
        )

    def test_ppo_matches_flat_baseline_except_iterations(self) -> None:
        rough = load_rl_cfg(_ROUGH).algorithm
        flat = load_rl_cfg(_FLAT_MLP).algorithm
        for key in (
            "clip_param",
            "entropy_coef",
            "num_learning_epochs",
            "num_mini_batches",
            "learning_rate",
            "schedule",
            "gamma",
            "lam",
            "desired_kl",
            "max_grad_norm",
        ):
            self.assertEqual(getattr(rough, key), getattr(flat, key), msg=key)
        self.assertEqual(load_rl_cfg(_ROUGH).num_steps_per_env, 24)
        self.assertEqual(load_rl_cfg(_ROUGH).max_iterations, 5000)

    def test_action_and_command_contract_match_flat(self) -> None:
        action = self.cfg.actions["delayed_action"]
        self.assertEqual(action.leg_action_semantics, "joint")
        self.assertAlmostEqual(float(action.wheel_scale), FLAT_WHEEL_ACTION_SCALE)
        command = self.cfg.commands["velocity_height"]
        self.assertEqual(tuple(command.height_range), (0.20, 0.38))
        self.assertEqual(tuple(command.deployment_ranges["height"]), (0.20, 0.38))

    def test_only_energy_rewards_differ_from_flat(self) -> None:
        self.assertEqual(set(self.cfg.rewards), set(self.flat.rewards))
        for name, term in self.cfg.rewards.items():
            expected = float(self.flat.rewards[name].weight)
            if name in _ENERGY_REWARDS:
                expected *= ROUGH_ENERGY_PENALTY_SCALE
            self.assertAlmostEqual(float(term.weight), expected, places=12, msg=name)
        # 折价必须真的生效，否则上面那圈断言会退化成空对照。
        self.assertLess(ROUGH_ENERGY_PENALTY_SCALE, 1.0)

    def test_domain_randomization_matches_flat(self) -> None:
        self.assertAlmostEqual(self.cfg.events["com"].params["com_range"], 0.005)


class RoughTerrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_env_cfg(_ROUGH)

    def test_terrain_curriculum_is_enabled(self) -> None:
        terrain = self.cfg.scene.terrain
        self.assertEqual(terrain.terrain_type, "generator")
        generator = terrain.terrain_generator
        assert generator is not None
        self.assertTrue(generator.curriculum)
        self.assertEqual(generator.num_rows, 10)
        # 上/下台阶、上/下斜坡成对出现，外加平地与随机起伏。
        self.assertEqual(
            list(generator.sub_terrains),
            ["flat", "stairs_up", "stairs_down", "slope_up", "slope_down", "random_rough"],
        )
        for name in ("stairs_up", "stairs_down"):
            self.assertEqual(
                tuple(generator.sub_terrains[name].step_height_range), _STEP_HEIGHT_RANGE
            )
        # 全员从最简单一行起步，难度只由 terrain_levels 课程放开。
        self.assertEqual(terrain.max_init_terrain_level, 0)
        self.assertIn("terrain_levels", self.cfg.curriculum)

    def test_contact_sensor_slots_cover_generator_terrain(self) -> None:
        # 生成器地形有几百个 terrain geom，默认 64 个匹配槽会溢出并让接触力读数失真。
        self.assertGreaterEqual(self.cfg.sim.contact_sensor_maxmatch, 500)

    def test_play_and_stair_eval_variants(self) -> None:
        stair_eval = load_env_cfg(_STAIR_EVAL).scene.terrain.terrain_generator
        assert stair_eval is not None
        self.assertEqual(list(stair_eval.sub_terrains), ["flat", "stairs_up", "stairs_down"])


class StepUpStateMachineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_env_cfg(_ROUGH)
        cls.command = cls.cfg.commands["velocity_height"]

    def _sensor(self) -> TerrainHeightSensorCfg:
        sensors = {sensor.name: sensor for sensor in self.cfg.scene.sensors or ()}
        sensor = sensors[_SENSOR_NAME]
        assert isinstance(sensor, TerrainHeightSensorCfg)
        return sensor

    def test_command_term_is_step_up_and_enabled(self) -> None:
        self.assertIsInstance(self.command, StepUpCommandCfg)
        self.assertTrue(self.command.step_up_enabled)
        self.assertEqual(self.command.step_up_sensor_name, _SENSOR_NAME)
        self.assertEqual(self.command.step_up_hold_s, 2.0)
        self.assertFalse(load_env_cfg(_NO_STEP_UP).commands["velocity_height"].step_up_enabled)

    def test_thresholds_bracket_the_terrain_step_range(self) -> None:
        # 台阶窗口必须盖住地形最高一级台阶，墙阈值必须在其之上，
        # 否则真台阶会被判成墙、episode 被无谓截断。
        self.assertLess(self.command.step_up_height_min, _STEP_HEIGHT_RANGE[1])
        self.assertGreater(self.command.step_up_height_max, _STEP_HEIGHT_RANGE[1])
        self.assertGreaterEqual(self.command.step_up_wall_height, _STEP_HEIGHT_RANGE[1])
        # 随机起伏最大 0.05 m，检测下限必须高于它，避免地面噪声误触发。
        self.assertGreater(self.command.step_up_height_min, 0.05)

    def test_forward_probe_ray_order(self) -> None:
        """commands.py 的 _RAY_BACKWARD/_RAY_UNDER/_RAY_FORWARD 索引硬编码为 0/1/2。"""
        sensor = self._sensor()
        self.assertEqual(sensor.reduction, "none")
        self.assertEqual(sensor.ray_alignment, "yaw")
        pattern = sensor.pattern
        assert isinstance(pattern, GridPatternCfg)
        offsets, directions = pattern.generate_rays(None, "cpu")
        self.assertEqual(tuple(offsets.shape), (3, 3))
        self.assertAlmostEqual(float(offsets[0, 0]), -ROUGH_STEP_UP_LOOKAHEAD_M, places=6)
        self.assertAlmostEqual(float(offsets[1, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(offsets[2, 0]), ROUGH_STEP_UP_LOOKAHEAD_M, places=6)
        self.assertAlmostEqual(float(directions[0, 2]), -1.0, places=6)

    def test_wall_termination_is_a_timeout(self) -> None:
        # 参考实现把墙算成截断而不是失败；注册成 time_out=False 会让 PPO 把它当成 0 值终局。
        self.assertTrue(self.cfg.terminations["wall_blocked"].time_out)


if __name__ == "__main__":
    unittest.main()
