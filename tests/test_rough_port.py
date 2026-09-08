"""崎岖地形任务的移植回归测试。

来源：scutrobotlab/wheeled-legged_RL 的 V14 rough 线（地形课程 + 台阶前的机身抬升）。
移植原则是 rough 只相对冻结的 Flat 基线改地形、地形课程、地形感知高度下限和能耗三项定价，
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
from se3_train.tasks.flat import rewards as flat_rewards
from se3_train.tasks.flat.env_cfg import (
    FLAT_ACTION_SMOOTHNESS_SPRING,
    FLAT_CMD_VEL_DEADBAND,
    FLAT_COMMAND_VELOCITY_ERROR_WEIGHT_LEGACY,
    FLAT_WHEEL_ACTION_SCALE,
)
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg
from se3_train.tasks.rough import ctbc, curriculums, events, observations, terminations
from se3_train.tasks.rough import rewards as rough_rewards
from se3_train.tasks.rough.commands import RoughCommandCfg
from se3_train.tasks.rough.env_cfg import (
    ROUGH_AMP_TERRAIN_TYPE_NAMES,
    ROUGH_BODY_COLLISION_BOTTOM_OFFSET,
    ROUGH_COMMAND_VELOCITY_ERROR_LIN_SCALE,
    ROUGH_ENERGY_PENALTY_SCALE,
    ROUGH_REWARD_TERRAIN_TYPE_NAMES,
    ROUGH_TERRAIN_ANG_VEL_YAW_RANGE,
    ROUGH_TERRAIN_HEIGHT_CLEARANCE,
    ROUGH_TERRAIN_LIN_VEL_X_RANGE,
    ROUGH_TERRAIN_STEP_HEIGHT_TYPE_NAMES,
)
from se3_train.tasks.rough.env_cfg import env_cfg as rough_env_cfg
from se3_train.tasks.rough.terrains import (
    _STEP_HEIGHT_RANGE,
    ROUGH_TERRAIN_CLEARED_DISTANCE_M,
    ROUGH_TERRAIN_EXIT_DISTANCE_M,
)

_ROUGH = "SE3-WheelLegged-Rough"
_STAIR_EVAL = "SE3-WheelLegged-Rough-StairEval"
_FLAT_MLP = "SE3-WheelLegged-Flat-MLP"

# 参考仓库 rough 相对 flat 只放松能耗类罚项（wheel_power / joint_torque 各 ÷10）。
_ENERGY_REWARDS = ("leg_torques", "wheel_torques", "leg_power")


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

    def test_only_energy_and_terrain_column_rewards_differ_from_flat(self) -> None:
        """rough 相对 Flat 基线只有三处奖励差异，逐处钉住。

        1. 能耗三项折价；2. 台阶列加回速度违令罚（Flat 已整项删除）；
        3. flat_base_height 换成按列置零的包装（权重与核参数不变）。
        """
        self.assertEqual(
            set(self.cfg.rewards) - set(self.flat.rewards), {"command_velocity_error"}
        )
        self.assertEqual(set(self.flat.rewards) - set(self.cfg.rewards), set())
        for name, term in self.cfg.rewards.items():
            if name == "command_velocity_error":
                continue
            expected = float(self.flat.rewards[name].weight)
            if name in _ENERGY_REWARDS:
                expected *= ROUGH_ENERGY_PENALTY_SCALE
            self.assertAlmostEqual(float(term.weight), expected, places=12, msg=name)
            # 除了按列置零的高度罚，奖励函数本身必须与 Flat 是同一个对象。
            if name != "flat_base_height":
                self.assertIs(term.func, self.flat.rewards[name].func, msg=name)
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
        # A4 起 env 集中到上行列：flat 25 / stairs_up 40 / slope_up 30 / random_rough 5，下行列 0。
        self.assertEqual(
            {n: c.proportion for n, c in generator.sub_terrains.items()},
            {"flat": 0.25, "stairs_up": 0.40, "stairs_down": 0.0, "slope_up": 0.30, "slope_down": 0.0, "random_rough": 0.05},
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
        assert isinstance(command, RoughCommandCfg)
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
        # 地形列 vx 上限跟随平地课程。
        self.assertTrue(command.terrain_lin_vel_x_follow_curriculum)

    def test_flat_warmup_and_strict_advance_threshold(self) -> None:
        # 2026-09-07 用户定（R7）：前 500 轮全平地，之后换回原列。推进阈值 R7–A4 为 0.75，2026-09-08（A5）改回 Flat 的 0.5。
        self.assertEqual(next(iter(self.cfg.curriculum)), "flat_warmup")
        params = self.cfg.curriculum["flat_warmup"].params
        self.assertEqual(params["iterations"], 500)
        self.assertEqual(params["steps_per_policy_iter"], load_rl_cfg(_ROUGH).num_steps_per_env)
        self.assertLess(list(self.cfg.curriculum).index("flat_warmup"), list(self.cfg.curriculum).index("terrain_levels"))
        self.assertAlmostEqual(self.cfg.curriculum["command_vel"].params["advance_threshold"], 0.5)
        off = rough_env_cfg(flat_warmup_iterations=0)
        self.assertNotIn("flat_warmup", off.curriculum)

    def test_flat_velocity_curriculum_reads_flat_column_only(self) -> None:
        self.assertEqual(self.cfg.events["set_curriculum_env_mask"].mode, "startup")
        self.assertEqual(
            tuple(self.cfg.events["set_curriculum_env_mask"].params["terrain_type_names"]), ("flat",)
        )
        params = self.cfg.curriculum["command_vel"].params
        self.assertEqual(params["tracking_log_key"], "Locomotion/tracking_lin_vel_reward_curriculum")
        # 关掉旋钮即退回 Flat 的全体均值判据。
        off = rough_env_cfg(flat_curriculum_signal_only=False)
        self.assertNotIn("set_curriculum_env_mask", off.events)
        self.assertNotIn("tracking_log_key", off.curriculum["command_vel"].params or {})

    def test_contact_sensor_slots_cover_generator_terrain(self) -> None:
        # 生成器地形有几百个 terrain geom，默认 64 个匹配槽会溢出并让接触力读数失真。
        self.assertGreaterEqual(self.cfg.sim.contact_sensor_maxmatch, 500)

    def test_play_and_stair_eval_variants(self) -> None:
        stair_eval = load_env_cfg(_STAIR_EVAL).scene.terrain.terrain_generator
        assert stair_eval is not None
        self.assertEqual(list(stair_eval.sub_terrains), ["flat", "stairs_up", "stairs_down"])


class TerrainColumnRewardTests(unittest.TestCase):
    """台阶列分列定价（2026-09-08 用户定，A6）：速度违令罚只在台阶列加回来，机身高度罚在台阶列置零。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_env_cfg(_ROUGH)

    def test_command_velocity_error_is_gated_to_the_stairs_column(self) -> None:
        term = self.cfg.rewards["command_velocity_error"]
        self.assertIs(term.func, rough_rewards.command_velocity_error_on_terrain)
        # 权重与死区沿用 2026-09-06 从 Flat 删除前的历史值，这次只改生效范围。
        self.assertAlmostEqual(float(term.weight), FLAT_COMMAND_VELOCITY_ERROR_WEIGHT_LEGACY)
        self.assertEqual(
            (term.params["lin_deadband"], term.params["yaw_deadband"]), FLAT_CMD_VEL_DEADBAND
        )
        self.assertEqual(term.params["terrain_type_names"], ROUGH_REWARD_TERRAIN_TYPE_NAMES)

    def test_lin_scale_keeps_the_penalty_inside_its_quadratic_region(self) -> None:
        """误差归一化尺度必须让整个可达误差区间都落在封顶以内，否则这项就退化成常数。"""
        term = self.cfg.rewards["command_velocity_error"]
        scale = term.params["lin_vel_scale"]
        self.assertAlmostEqual(scale, ROUGH_COMMAND_VELOCITY_ERROR_LIN_SCALE)
        # 台阶列最坏情况：满指令 2.4 m/s 而机器人不动。此时罚值必须仍未顶到 max_penalty。
        worst_excess = ROUGH_TERRAIN_LIN_VEL_X_RANGE[1] - term.params["lin_deadband"]
        self.assertLess((worst_excess / scale) ** 2, term.params["max_penalty"])
        # 且封顶后的量级不能压过正奖励预算（is_alive 1 + 跟踪三项 4/3/2 = 10/s）。
        self.assertLess(abs(term.weight) * (worst_excess / scale) ** 2, 10.0)

    def test_base_height_penalty_is_zeroed_on_the_stairs_column(self) -> None:
        term = self.cfg.rewards["flat_base_height"]
        self.assertIs(term.func, rough_rewards.base_height_penalty_off_terrain)
        self.assertEqual(term.params["terrain_type_names"], ROUGH_REWARD_TERRAIN_TYPE_NAMES)
        # 只改生效范围：权重与核参数继续跟随 Flat 基线。
        self.assertAlmostEqual(float(term.weight), -4.0)
        self.assertAlmostEqual(term.params["sigma"], 0.05)

    def test_all_three_column_switches_point_at_the_same_column(self) -> None:
        # AMP 掩码、地形感知高度下限、分列定价必须是同一组列，
        # 否则"只改台阶列"这句话在三处的含义就不一样了。
        self.assertEqual(ROUGH_REWARD_TERRAIN_TYPE_NAMES, ROUGH_AMP_TERRAIN_TYPE_NAMES)
        self.assertEqual(ROUGH_REWARD_TERRAIN_TYPE_NAMES, ROUGH_TERRAIN_STEP_HEIGHT_TYPE_NAMES)

    def test_both_knobs_fall_back_to_the_flat_baseline(self) -> None:
        off = rough_env_cfg(command_velocity_error_weight=None, zero_base_height_on_terrain=False)
        self.assertNotIn("command_velocity_error", off.rewards)
        self.assertIs(
            off.rewards["flat_base_height"].func, flat_rewards.flat_base_height_penalty_no_jump
        )


class TerrainAwareHeightFloorTests(unittest.TestCase):
    """台阶前的机身抬升：2026-09-08 用户定，删掉 step_up 状态机，改用地形感知高度下限。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_env_cfg(_ROUGH)
        cls.command = cls.cfg.commands["velocity_height"]

    def _floor(self, difficulty: float) -> float:
        """按 mdp/commands._terrain_aware_min_height 的公式复算该难度下的高度指令下限。"""
        step_low, step_high = _STEP_HEIGHT_RANGE
        step_height = step_low + difficulty * (step_high - step_low)
        required = (
            step_height
            + self.command.terrain_height_clearance
            - self.command.body_collision_bottom_offset
        )
        return max(self.command.height_range[0], required)

    def test_terrain_aware_height_floor_is_configured(self) -> None:
        self.assertIsInstance(self.command, RoughCommandCfg)
        self.assertTrue(self.command.terrain_aware_height)
        # clearance 必须为正，否则 _apply_terrain_aware_height / _sample_terrain_aware_height
        # 整段短路，下限静默失效（这正是 A1–A5 之前的状态）。
        self.assertGreater(self.command.terrain_height_clearance, 0.0)
        self.assertEqual(self.command.terrain_height_clearance, ROUGH_TERRAIN_HEIGHT_CLEARANCE)
        self.assertEqual(
            self.command.body_collision_bottom_offset, ROUGH_BODY_COLLISION_BOTTOM_OFFSET
        )
        self.assertLess(self.command.body_collision_bottom_offset, 0.0)

    def test_step_height_type_names_match_the_terrain_columns(self) -> None:
        """列名对不上时下限静默失效——基类默认值是 stair 线的列名，rough 必须自己配。"""
        generator = self.cfg.scene.terrain.terrain_generator
        assert generator is not None
        names = tuple(self.command.terrain_step_height_type_names)
        self.assertEqual(names, ROUGH_TERRAIN_STEP_HEIGHT_TYPE_NAMES)
        self.assertTrue(names)
        for name in names:
            self.assertIn(name, generator.sub_terrains)
            # 没有 step_height_range 的列（斜坡、随机起伏）会被 _terrain_aware_min_height 跳过。
            self.assertIsNotNone(getattr(generator.sub_terrains[name], "step_height_range", None))

    def test_floor_spans_the_step_range_without_saturating(self) -> None:
        low, high = self.command.height_range
        # 最低难度的台阶（0.02 m）不该顶起下限，否则平地段的高度指令分布也跟着变。
        self.assertAlmostEqual(self._floor(0.0), low)
        # 最高难度的台阶（0.20 m）必须顶起来，且不能顶到上界——顶满就退化成定值指令。
        self.assertGreater(self._floor(1.0), low)
        self.assertLess(self._floor(1.0), high)

    def test_step_up_state_machine_is_gone(self) -> None:
        # 前瞻传感器、墙终止、状态机字段都不该再出现；留着就是死代码 + 每步一次白跑的射线。
        sensor_names = {sensor.name for sensor in self.cfg.scene.sensors or ()}
        self.assertNotIn("wheel_forward_sensor", sensor_names)
        self.assertNotIn("wall_blocked", self.cfg.terminations)
        self.assertFalse(hasattr(self.command, "step_up_enabled"))

    def test_terrain_diagnostics_survive_log_filter(self) -> None:
        # 下限有没有顶起来、台阶列的速度违令罚吃了多少，只能从这两个键看出来。
        for key in ("Rough/height_cmd_terrain_mean", "Rough/command_velocity_error_terrain"):
            self.assertTrue(keep_log_key(key), key)

    def test_terrain_cleared_is_a_timeout_beyond_the_last_step(self) -> None:
        self.assertTrue(self.cfg.terminations["terrain_cleared"].time_out)
        # 清块门槛 = 平台半宽 1.0 + 2 级 × 1.5 m 踏面；出块门槛在它之后、块边 4.5 m 之前。
        self.assertAlmostEqual(ROUGH_TERRAIN_CLEARED_DISTANCE_M, 4.0)
        self.assertGreater(ROUGH_TERRAIN_EXIT_DISTANCE_M, ROUGH_TERRAIN_CLEARED_DISTANCE_M)
        self.assertLess(ROUGH_TERRAIN_EXIT_DISTANCE_M, 4.5)


class FlatWarmupRuntimeTests(unittest.TestCase):
    """平地热身：前 N 轮全员平地，之后 reset 时换回原列，两个按列掩码随之刷新。"""

    @classmethod
    def setUpClass(cls) -> None:
        cfg = rough_env_cfg(flat_warmup_iterations=500)
        cfg.scene.num_envs = 12
        cls.env = ManagerBasedRlEnv(cfg, device="cpu")
        cls.env.reset()
        cls.terrain = cls.env.scene.terrain
        names = list(cls.terrain.cfg.terrain_generator.sub_terrains.keys())
        cls.flat_col = names.index("flat")
        cls.term = cls.env.command_manager.get_term("velocity_height")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_everyone_starts_on_flat_then_returns_to_own_column(self) -> None:
        types = self.terrain.terrain_types
        original = getattr(self.env, curriculums.FLAT_WARMUP_ORIGINAL_TYPES_ATTR)
        self.assertEqual(len(set(original.tolist())), 6)  # 原列分配保留了六列
        self.assertTrue(bool((types == self.flat_col).all()))
        self.assertTrue(bool((self.terrain.terrain_levels == 0).all()))
        # 热身期：前向指令覆盖对谁都不生效，课程信号掩码全员为真。
        self.assertFalse(bool(self.term._terrain_override_mask.any()))
        self.assertTrue(bool(getattr(self.env, events.CURRICULUM_ENV_MASK_ATTR).all()))
        # 热身期 terrain_levels 不升级：把机器人放到清块距离之外也不动。
        robot = self.env.scene["robot"]
        pose = robot.data.root_link_pose_w.clone()
        pose[:, 0] = self.env.scene.env_origins[:, 0] + 4.1
        robot.write_root_link_pose_to_sim(pose)
        self.env.sim.forward()
        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        self.env.common_step_counter = 10 * 24
        curriculums.terrain_levels(self.env, env_ids, command_name="velocity_height")
        self.assertTrue(bool((self.terrain.terrain_levels == 0).all()))
        # 500 轮后：reset 到的 env 换回原列、第 0 行，掩码按新列重算。
        self.env.common_step_counter = 500 * 24
        half = env_ids[:6]
        curriculums.flat_warmup(self.env, half, command_name="velocity_height", iterations=500)
        self.assertTrue(torch.equal(types[:6], original[:6]))
        self.assertTrue(bool((types[6:] == self.flat_col).all()))
        curriculums.flat_warmup(self.env, env_ids, command_name="velocity_height", iterations=500)
        self.assertTrue(torch.equal(types, original))
        self.assertTrue(bool((self.terrain.terrain_levels == 0).all()))
        self.assertTrue(torch.equal(self.term._terrain_override_mask, original != self.flat_col))
        self.assertTrue(torch.equal(getattr(self.env, events.CURRICULUM_ENV_MASK_ATTR), original == self.flat_col))
        # 换列后升级恢复。
        pose[:, 0] = self.env.scene.env_origins[:, 0] + 4.1
        robot.write_root_link_pose_to_sim(pose)
        self.env.sim.forward()
        curriculums.terrain_levels(self.env, env_ids, command_name="velocity_height")
        self.assertTrue(bool((self.terrain.terrain_levels == 1).all()))


if __name__ == "__main__":
    unittest.main()


class CtbcPortTests(unittest.TestCase):
    """CTBC 从 stair 线移植到 rough 的接线：传感器、事件、观测槽位、旋钮。默认关闭，显式打开来测。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = rough_env_cfg(ctbc_enabled=True)

    def test_ctbc_is_off_by_default(self) -> None:
        # R4/R5：第 0 轮起注入前馈把策略教成回避接触，平地跟踪一起退化，默认关闭。
        default = load_env_cfg(_ROUGH)
        self.assertNotIn("init_ctbc_state", default.events)
        self.assertNotIn("wheel_riser_sensor", {s.name for s in default.scene.sensors or ()})
        self.assertIsNot(default.observations["actor"].terms["jump_commands"].func, ctbc.ctbc_obs)

    def test_riser_sensor_and_events_are_wired(self) -> None:
        sensors = {sensor.name: sensor for sensor in self.cfg.scene.sensors or ()}
        self.assertIn("wheel_riser_sensor", sensors)
        self.assertIn("normal", sensors["wheel_riser_sensor"].fields)
        for name in ("init_ctbc_state", "step_ctbc_state", "reset_ctbc_state"):
            self.assertIn(name, self.cfg.events)
        self.assertEqual(self.cfg.events["init_ctbc_state"].mode, "startup")
        self.assertEqual(self.cfg.events["step_ctbc_state"].mode, "interval")
        self.assertEqual(self.cfg.events["reset_ctbc_state"].mode, "reset")
        params = self.cfg.events["init_ctbc_state"].params
        # 2026-09-07 用户定：500 轮前满幅，500→1500 线性退火，之后关闭；只在上台阶列触发。
        self.assertEqual(params["ann_start_iter"], 500)
        self.assertEqual(params["ann_end_iter"], 1500)
        self.assertEqual(
            tuple(self.cfg.events["step_ctbc_state"].params["terrain_type_names"]), ("stairs_up",)
        )

    def test_ctbc_obs_replaces_jump_slots_without_changing_dims(self) -> None:
        flat = flat_env_cfg(
            wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
            action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
        )
        for group in ("actor", "critic"):
            rough_terms = [t for t in self.cfg.observations[group].terms if t != "height_scan"]
            flat_terms = list(flat.observations[group].terms)
            # 项名、顺序、个数与 Flat 完全一致：部署契约（se3-sim2x policy_contract）只认 jump_commands，
            # 这里只换实现函数。
            self.assertEqual(rough_terms, flat_terms)
            self.assertIs(self.cfg.observations[group].terms["jump_commands"].func, ctbc.ctbc_obs)

    def test_ctbc_can_be_switched_off_for_ablation(self) -> None:
        off = rough_env_cfg(ctbc_enabled=False)
        self.assertNotIn("init_ctbc_state", off.events)
        self.assertIsNot(off.observations["actor"].terms["jump_commands"].func, ctbc.ctbc_obs)
        self.assertNotIn("wheel_riser_sensor", {s.name for s in off.scene.sensors or ()})


class CriticHeightScanTests(unittest.TestCase):
    """critic 特权地形扫描（照 yly-true/fudan_rl_wheel_leg）：只进 critic，actor 契约不变。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_env_cfg(_ROUGH)

    def test_scan_sensor_matches_reference_grid(self) -> None:
        sensors = {sensor.name: sensor for sensor in self.cfg.scene.sensors or ()}
        sensor = sensors["critic_height_scan"]
        assert isinstance(sensor, TerrainHeightSensorCfg)
        assert isinstance(sensor.pattern, GridPatternCfg)
        # 参考仓库 measured_points_x = -0.5..0.5、measured_points_y = -0.3..0.3，步长 0.1 → 11×7。
        self.assertEqual(tuple(sensor.pattern.size), (1.0, 0.6))
        self.assertAlmostEqual(sensor.pattern.resolution, 0.1)
        self.assertEqual(sensor.ray_alignment, "yaw")
        self.assertEqual(sensor.reduction, "none")
        offsets, _ = sensor.pattern.generate_rays(None, device="cpu")
        self.assertEqual(int(offsets.shape[0]), 77)

    def test_scan_only_in_critic_group(self) -> None:
        self.assertIn("height_scan", self.cfg.observations["critic"].terms)
        self.assertNotIn("height_scan", self.cfg.observations["actor"].terms)
        self.assertIs(self.cfg.observations["critic"].terms["height_scan"].func, observations.height_scan_obs)
        off = rough_env_cfg(critic_height_scan=False)
        self.assertNotIn("height_scan", off.observations["critic"].terms)
        self.assertNotIn("critic_height_scan", {s.name for s in off.scene.sensors or ()})


class RoughRuntimeTests(unittest.TestCase):
    """在 CPU 上建一个小环境，验证分列指令覆盖与只升不降的课程真的生效。"""

    @classmethod
    def setUpClass(cls) -> None:
        # 运行时测试覆盖 CTBC 链路，显式打开；关掉平地热身，让 env 一开始就在各自的列上。
        cfg = rough_env_cfg(ctbc_enabled=True, flat_warmup_iterations=0)
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

    def test_terrain_column_rewards_only_bite_on_the_stairs_column(self) -> None:
        """台阶列吃速度违令罚、不吃高度罚；其余列正好相反。"""
        names = list(self.env.scene.terrain.cfg.terrain_generator.sub_terrains.keys())
        up = (self.terrain_types == names.index("stairs_up")).nonzero().flatten()
        flat = (self.terrain_types == self.flat_col).nonzero().flatten()
        cmd = self.env.command_manager.get_command("velocity_height")
        saved = cmd.clone()
        try:
            # 造一个所有 env 都同时违令的状态：指令 vx=2 而机器人基本不动，
            # 高度指令远低于实际高度（误差顶到 max_error=0.15，罚值封顶）。
            cmd[:, 0] = 2.0
            cmd[:, 1] = 0.0
            cmd[:, 4] = 0.0
            cmd[:, 5] = 0.0
            vel_pen = rough_rewards.command_velocity_error_on_terrain(
                self.env,
                command_name="velocity_height",
                terrain_type_names=("stairs_up",),
            )
            height_pen = rough_rewards.base_height_penalty_off_terrain(
                self.env,
                command_name="velocity_height",
                height_sensor_name="base_height_sensor",
                terrain_type_names=("stairs_up",),
            )
        finally:
            cmd[:] = saved
        self.assertGreater(float(vel_pen[up].min()), 0.0)
        self.assertEqual(float(vel_pen[flat].abs().max()), 0.0)
        self.assertEqual(float(height_pen[up].abs().max()), 0.0)
        self.assertGreater(float(height_pen[flat].min()), 0.0)

    def test_terrain_aware_height_floor_lifts_the_stairs_up_command(self) -> None:
        """最高难度行上，上台阶列的高度指令下界被顶到台阶高 + 余量；平地列不受影响。

        配置侧的断言（TerrainAwareHeightFloorTests）只能保证参数填对了，
        真正会静默失效的是 terrain_levels/terrain_types 这条查表链路，只能在运行时钉。
        """
        terrain = self.env.scene.terrain
        names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
        up = (self.terrain_types == names.index("stairs_up")).nonzero().flatten()
        flat = (self.terrain_types == self.flat_col).nonzero().flatten()
        cfg = self.term.cfg
        expected = (
            _STEP_HEIGHT_RANGE[1]
            + cfg.terrain_height_clearance
            - cfg.body_collision_bottom_offset
        )
        self.assertLess(expected, cfg.height_range[1])
        levels = terrain.terrain_levels.clone()
        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        flat_min = float("inf")
        try:
            terrain.terrain_levels[:] = terrain.terrain_origins.shape[0] - 1
            for _ in range(8):  # 高度是区间内均匀采样，多抽几次才钉得住下界
                self.term._resample_command(env_ids)
                height = self.term.command[:, 4]
                self.assertGreaterEqual(float(height[up].min()), expected - 1e-6)
                self.assertLessEqual(float(height[up].max()), cfg.height_range[1] + 1e-6)
                self.assertGreaterEqual(float(height[flat].min()), cfg.height_range[0] - 1e-6)
                flat_min = min(flat_min, float(height[flat].min()))
        finally:
            terrain.terrain_levels[:] = levels
        # 下限只对上台阶列生效：平地列必须能抽到低于该下限的高度，否则是全局抬高了。
        self.assertLess(flat_min, expected)

    def test_height_scan_reads_step_rise_ahead(self) -> None:
        terrain = self.env.scene.terrain
        names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
        up = (self.terrain_types == names.index("stairs_up")).nonzero().flatten()
        flat = (self.terrain_types == self.flat_col).nonzero().flatten()
        # 上台阶列的 env 放到第 3 行（8 cm 台阶）坑底，朝 +x 站在 x=0.7：网格 x≥+0.3 的射线打在第一级台面上。
        levels = terrain.terrain_levels.clone()
        terrain.terrain_levels[up] = 3
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        robot = self.env.scene["robot"]
        pose = robot.data.root_link_pose_w.clone()
        pose[:, 0] = self.env.scene.env_origins[:, 0] + 0.7
        pose[:, 1] = self.env.scene.env_origins[:, 1]
        pose[:, 2] = self.env.scene.env_origins[:, 2] + 0.30
        pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        try:
            for _ in range(2):
                robot.write_root_link_pose_to_sim(pose)
                self.env.sim.forward()
                self.env.sim.sense()
            scan = observations.height_scan_obs(self.env, "critic_height_scan")
            self.assertEqual(tuple(scan.shape), (self.env.num_envs, 77))
            step_h = 0.02 + 0.18 * 3 / 9
            # 平地列：四周全平，读数应接近 0。腿/轮上的自击只在抬升 >3 cm 时才被过滤，
            # 所以允许 3 cm 以内的残余。
            self.assertLess(float(scan[flat].abs().max()), 0.03)
            # 上台阶列：前方最高读数等于一级台阶高，身后仍是坑底。
            self.assertAlmostEqual(float(scan[up].max(dim=1).values.mean()), step_h, delta=0.02)
            self.assertLess(float(scan[up].min(dim=1).values.abs().max()), 0.02)
        finally:
            terrain.terrain_levels[:] = levels
            terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
            self.env.reset()

    def test_ctbc_state_attached_and_injects_leg_action(self) -> None:
        state = getattr(self.env, ctbc.CTBC_STATE_ATTR, None)
        self.assertIsNotNone(state)
        obs = ctbc.ctbc_obs(self.env)
        self.assertEqual(tuple(obs.shape), (self.env.num_envs, 3))
        self.assertEqual(float(obs.abs().sum()), 0.0)
        # 手动触发 env 0 左侧前馈：动作项必须只对 env 0 的腿部动作注入增量。
        state.update_iter(0)
        self.assertEqual(state.kff, 1.0)
        state._ff_phase[0, 0] = max(1, state.ff_rise_steps // 2)
        term = self.env.action_manager.get_term("delayed_action")
        term.process_actions(torch.zeros(self.env.num_envs, self.env.action_manager.total_action_dim))
        delta = term.ctbc_action_delta
        self.assertGreater(float(delta[0, :4].abs().sum()), 1e-4)
        self.assertEqual(float(delta[1:, :4].abs().sum()), 0.0)
        obs = ctbc.ctbc_obs(self.env)
        self.assertGreater(float(obs[0, 0]), 0.0)
        self.assertEqual(float(obs[0, 2]), 1.0)
        self.assertEqual(float(obs[1:].abs().sum()), 0.0)
        # 退火时间表：500 轮前 1.0，1000 轮 0.5，1500 轮起 0。
        for iteration, expected in ((499, 1.0), (1000, 0.5), (1500, 0.0)):
            state.update_iter(iteration)
            self.assertAlmostEqual(state.kff, expected, msg=f"iter {iteration}")
        # 退火结束（kff=0）后观测必须全 0，与没有状态机的部署端一致。
        state.update_iter(10**6)
        self.assertEqual(state.kff, 0.0)
        self.assertEqual(float(ctbc.ctbc_obs(self.env).abs().sum()), 0.0)
        state.update_iter(0)
        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        state.reset(env_ids)
        self.assertEqual(float(ctbc.ctbc_obs(self.env).abs().sum()), 0.0)

    def test_ctbc_only_runs_on_the_stairs_up_column(self) -> None:
        state = getattr(self.env, ctbc.CTBC_STATE_ATTR)
        terrain = self.env.scene.terrain
        names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
        up = (self.terrain_types == names.index("stairs_up")).nonzero().flatten()
        down = (self.terrain_types == names.index("stairs_down")).nonzero().flatten()
        state.update_iter(0)
        # 两列各挑一个 env 手动置成前馈进行中，跑一次 step 事件：下台阶列必须被清掉，上台阶列保留推进。
        state._ff_phase[up[0], 0] = 3
        state._ff_phase[down[0], 1] = 3
        ctbc.step_ctbc_state(self.env, None, terrain_type_names=("stairs_up",))
        self.assertEqual(int(state.ff_phase[down[0], 1]), -1)
        self.assertEqual(int(state.ff_phase[up[0], 0]), 4)
        state.reset(torch.arange(self.env.num_envs, device=self.env.device))

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

    def test_terrain_vx_upper_bound_follows_flat_curriculum(self) -> None:
        non_flat = self.terrain_types != self.flat_col
        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        saved = self.term.cfg.lin_vel_x_range
        try:
            # 课程起点 (0,0)：地形列拿到下界 0.4 的定速指令。
            self.term.cfg.lin_vel_x_range = (0.0, 0.0)
            self.term._resample_command(env_ids)
            self.assertTrue(bool(((self.term.command[non_flat][:, 0] - 0.4).abs() < 1e-5).all()))
            # 课程推到 ±0.8：地形列上限 0.8。
            self.term.cfg.lin_vel_x_range = (-0.8, 0.8)
            for _ in range(20):
                self.term._resample_command(env_ids)
                vx = self.term.command[non_flat][:, 0]
                self.assertTrue(bool((vx >= 0.4 - 1e-5).all()) and bool((vx <= 0.8 + 1e-5).all()))
            # 课程到顶 ±2.4：上限回到 terrain_lin_vel_x_range 的 2.4。
            self.term.cfg.lin_vel_x_range = (-2.4, 2.4)
            seen_max = 0.0
            for _ in range(30):
                self.term._resample_command(env_ids)
                seen_max = max(seen_max, float(self.term.command[non_flat][:, 0].max()))
            self.assertGreater(seen_max, 1.5)
        finally:
            self.term.cfg.lin_vel_x_range = saved
            self.term._resample_command(env_ids)

    def test_curriculum_signal_mask_marks_flat_column_and_is_logged(self) -> None:
        mask = getattr(self.env, events.CURRICULUM_ENV_MASK_ATTR, None)
        assert mask is not None
        self.assertTrue(torch.equal(mask.cpu(), (self.terrain_types == self.flat_col).cpu()))
        # 强制每步写日志，走一步，课程专用的跟踪分键必须出现。
        self.env._se3_reward_log_interval_steps = 1
        self.env.step(torch.zeros(self.env.num_envs, self.env.action_manager.total_action_dim))
        log = self.env.extras.get("log", {})
        self.assertIn("Locomotion/tracking_lin_vel_reward_curriculum", log)
        self.assertIn("Locomotion/tracking_lin_vel_reward_all", log)

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
