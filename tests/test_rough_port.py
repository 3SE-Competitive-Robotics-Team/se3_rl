"""崎岖地形任务的移植回归测试。

来源：scutrobotlab/wheeled-legged_RL 的 V14 rough 线（地形课程 + step_up 状态机）。
移植原则是 rough 只相对冻结的 Flat 基线改地形、地形课程、台阶状态机和能耗三项定价，
其余逐项不动；本文件把这几条钉住。要改 rough 就一并改这里，并写清对照实验编号。
"""

from __future__ import annotations

import unittest

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import GridPatternCfg, TerrainHeightSensorCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.terrains import (
    BoxInvertedPyramidStairsTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    HfPyramidSlopedTerrainCfg,
)

import se3_train  # noqa: F401  # 注册任务
from se3_train.log_filter import keep_log_key
from se3_train.tasks.flat.env_cfg import (
    FLAT_ACTION_SMOOTHNESS_SPRING,
    FLAT_WHEEL_ACTION_SCALE,
)
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg
from se3_train.tasks.rough import curriculums, terminations
from se3_train.tasks.rough.commands import StepUpCommandCfg
from se3_train.tasks.rough.env_cfg import (
    ROUGH_ENERGY_PENALTY_SCALE,
    ROUGH_STEP_UP_LOOKAHEAD_M,
    ROUGH_TERRAIN_ANG_VEL_YAW_RANGE,
    ROUGH_TERRAIN_LIN_VEL_X_RANGE,
)
from se3_train.tasks.rough.env_cfg import env_cfg as rough_env_cfg
from se3_train.tasks.rough.terrains import (
    _STEP_HEIGHT_RANGE,
    ROUGH_TERRAIN_CLEARED_DISTANCE_M,
    ROUGH_TERRAIN_EXIT_DISTANCE_M,
)

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
        # 列名按出生点出发的实际方向：MJLab 正金字塔出生在顶部（向外是下行），
        # 反金字塔出生在底部（向外是爬升）。R2 之前两列是反的。
        self.assertIsInstance(generator.sub_terrains["stairs_up"], BoxInvertedPyramidStairsTerrainCfg)
        self.assertNotIsInstance(generator.sub_terrains["stairs_down"], BoxInvertedPyramidStairsTerrainCfg)
        self.assertIsInstance(generator.sub_terrains["stairs_down"], BoxPyramidStairsTerrainCfg)
        slope_up = generator.sub_terrains["slope_up"]
        slope_down = generator.sub_terrains["slope_down"]
        assert isinstance(slope_up, HfPyramidSlopedTerrainCfg)
        assert isinstance(slope_down, HfPyramidSlopedTerrainCfg)
        self.assertTrue(slope_up.inverted)
        self.assertFalse(slope_down.inverted)
        # 全员从最简单一行起步，难度只由 terrain_levels 课程放开。
        self.assertEqual(terrain.max_init_terrain_level, 0)
        self.assertIn("terrain_levels", self.cfg.curriculum)

    def test_terrain_command_override_config(self) -> None:
        command = self.cfg.commands["velocity_height"]
        assert isinstance(command, StepUpCommandCfg)
        self.assertTrue(command.terrain_command_override_enabled)
        self.assertEqual(command.terrain_command_flat_names, ("flat",))
        self.assertEqual(tuple(command.terrain_lin_vel_x_range), ROUGH_TERRAIN_LIN_VEL_X_RANGE)
        self.assertEqual(tuple(command.terrain_ang_vel_yaw_range), ROUGH_TERRAIN_ANG_VEL_YAW_RANGE)
        # 非平地列只准朝前直冲：vx 下界为正、上界不超过 Flat 课程终值，yaw 近零。
        self.assertGreater(ROUGH_TERRAIN_LIN_VEL_X_RANGE[0], 0.0)
        self.assertLessEqual(ROUGH_TERRAIN_LIN_VEL_X_RANGE[1], 2.4)
        self.assertLessEqual(abs(ROUGH_TERRAIN_ANG_VEL_YAW_RANGE[0]), 0.2)
        self.assertLessEqual(abs(ROUGH_TERRAIN_ANG_VEL_YAW_RANGE[1]), 0.2)
        # 单变量对照旋钮：关掉后逐位退回 Flat 的对称随机指令。
        off = rough_env_cfg(terrain_command_override=False).commands["velocity_height"]
        self.assertFalse(off.terrain_command_override_enabled)

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

    def test_step_up_rates_survive_log_filter(self) -> None:
        # 状态机占比走 Rough/ 命名空间，必须在常驻白名单里，否则 W&B 上看不到它有没有触发。
        for key in ("Rough/step_up_hold_rate", "Rough/step_up_detect_rate", "Rough/wall_blocked_rate"):
            self.assertTrue(keep_log_key(key), key)

    def test_terrain_cleared_is_a_timeout_beyond_the_last_step(self) -> None:
        self.assertTrue(self.cfg.terminations["terrain_cleared"].time_out)
        # 清块门槛 = 平台半宽 1.0 + 2 级 × 1.5 m 踏面；出块门槛在它之后、块边 4.5 m 之前。
        self.assertAlmostEqual(ROUGH_TERRAIN_CLEARED_DISTANCE_M, 4.0)
        self.assertGreater(ROUGH_TERRAIN_EXIT_DISTANCE_M, ROUGH_TERRAIN_CLEARED_DISTANCE_M)
        self.assertLess(ROUGH_TERRAIN_EXIT_DISTANCE_M, 4.5)


if __name__ == "__main__":
    unittest.main()


class RoughRuntimeTests(unittest.TestCase):
    """在 CPU 上建一个小环境，验证分列指令覆盖与只升不降的课程真的生效。"""

    @classmethod
    def setUpClass(cls) -> None:
        cfg = rough_env_cfg()
        cfg.scene.num_envs = 12  # 6 列各 2 个 env
        cls.env = ManagerBasedRlEnv(cfg, device="cpu")
        cls.env.reset()
        cls.term = cls.env.command_manager.get_term("velocity_height")
        terrain = cls.env.scene.terrain
        assert terrain is not None
        names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
        cls.flat_col = names.index("flat")
        cls.terrain_types = terrain.terrain_types.to(dtype=torch.long)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_every_column_is_populated(self) -> None:
        self.assertEqual(sorted(set(self.terrain_types.tolist())), list(range(6)))

    def test_non_flat_columns_only_get_forward_commands(self) -> None:
        non_flat = self.terrain_types != self.flat_col
        mask = self.term._terrain_override_mask
        assert mask is not None
        self.assertTrue(torch.equal(mask.cpu(), non_flat.cpu()))
        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        vx_lo, vx_hi = ROUGH_TERRAIN_LIN_VEL_X_RANGE
        yaw_lo, yaw_hi = ROUGH_TERRAIN_ANG_VEL_YAW_RANGE
        for _ in range(50):  # 多抽几轮，静站样本（10%）若漏进非平地列一定会被抓到
            self.term._resample_command(env_ids)
            cmd = self.term.command[non_flat]
            self.assertTrue(bool((cmd[:, 0] >= vx_lo - 1e-6).all()), cmd[:, 0])
            self.assertTrue(bool((cmd[:, 0] <= vx_hi + 1e-6).all()), cmd[:, 0])
            self.assertTrue(bool((cmd[:, 1] >= yaw_lo - 1e-6).all()), cmd[:, 1])
            self.assertTrue(bool((cmd[:, 1] <= yaw_hi + 1e-6).all()), cmd[:, 1])
            self.assertFalse(bool(self.term._standing_mask[non_flat].any()))

    def test_flat_column_keeps_flat_command_ranges(self) -> None:
        # Flat 速度课程起点是 vx=yaw=0，平地列 reset 后指令必须仍是 0，而不是被覆盖成前向直行。
        flat = self.terrain_types == self.flat_col
        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        self.term._resample_command(env_ids)
        self.assertTrue(bool((self.term.command[flat][:, :2].abs() < 1e-6).all()))

    def _place_offset(self, dx: float, dy: float) -> None:
        robot = self.env.scene["robot"]
        pose = robot.data.root_link_pose_w.clone()
        pose[:, 0] = self.env.scene.env_origins[:, 0] + dx
        pose[:, 1] = self.env.scene.env_origins[:, 1] + dy
        robot.write_root_link_pose_to_sim(pose)
        self.env.sim.forward()

    def test_terrain_levels_never_demote(self) -> None:
        terrain = self.env.scene.terrain
        assert terrain is not None
        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        self._place_offset(0.0, 0.0)
        terrain.terrain_levels[:] = 3
        before = terrain.terrain_levels.clone()
        # 机器人还在出生点：净位移 0，旧判据会全员降级，新判据必须原地不动。
        curriculums.terrain_levels(self.env, env_ids, command_name="velocity_height")
        self.assertTrue(torch.equal(terrain.terrain_levels, before))

    def test_terrain_levels_ignore_the_initial_reset(self) -> None:
        # 首次 reset 前机器人还在世界原点附近，到出生点的距离远超门槛，但不能算升级。
        terrain = self.env.scene.terrain
        assert terrain is not None
        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        terrain.terrain_levels[:] = 3
        self._place_offset(30.0, 30.0)
        saved = self.env.common_step_counter
        self.env.common_step_counter = 0
        try:
            curriculums.terrain_levels(self.env, env_ids, command_name="velocity_height")
        finally:
            self.env.common_step_counter = saved
        self.assertTrue(bool((terrain.terrain_levels == 3).all()))
        self._place_offset(0.0, 0.0)

    def test_terrain_levels_use_chebyshev_distance(self) -> None:
        terrain = self.env.scene.terrain
        assert terrain is not None
        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        # 走过一步之后才结算；测试环境没 step 过，手动把计数拨到 1。
        self.env.common_step_counter = max(int(self.env.common_step_counter), 1)
        # 对角线上欧氏 4.5 m（旧门槛）只到 L∞ 3.18 m：两级台阶只爬了一级，不能升。
        terrain.terrain_levels[:] = 3
        self._place_offset(3.18, 3.18)
        curriculums.terrain_levels(self.env, env_ids, command_name="velocity_height")
        self.assertTrue(bool((terrain.terrain_levels == 3).all()))
        # 沿轴 L∞ 4.1 m：越过最外一级台阶，升一级。
        self._place_offset(4.1, 0.3)
        curriculums.terrain_levels(self.env, env_ids, command_name="velocity_height")
        self.assertTrue(bool((terrain.terrain_levels == 4).all()))
        # 升级后出生点已换到新一行；把机器人搬回新出生点，避免影响其他测试。
        self._place_offset(0.0, 0.0)

    def test_terrain_cleared_truncates_at_patch_edge(self) -> None:
        # 4.1 m 已越过最外一级台阶但还没到边框：不截断，留给课程结算。
        self._place_offset(4.1, 0.0)
        self.assertFalse(bool(terminations.terrain_cleared(self.env).any()))
        # 4.3 m 踩上边框：截断。
        self._place_offset(0.0, 4.3)
        self.assertTrue(bool(terminations.terrain_cleared(self.env).all()))
        self._place_offset(0.0, 0.0)
