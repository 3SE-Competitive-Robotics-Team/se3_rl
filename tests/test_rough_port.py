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
from se3_train.tasks.rough import ctbc, curriculums, events, observations, terminations
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
        # 地形列 vx 上限跟随平地课程。
        self.assertTrue(command.terrain_lin_vel_x_follow_curriculum)

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


class CtbcPortTests(unittest.TestCase):
    """CTBC 从 stair 线移植到 rough 的接线：传感器、事件、观测槽位、旋钮。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_env_cfg(_ROUGH)

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
            self.assertNotIn("jump_commands", rough_terms)
            self.assertIn("ctbc", rough_terms)
            # 槽位一一对应：只是把 jump_commands 换成 ctbc，顺序与个数不变。
            self.assertEqual(
                [("jump_commands" if t == "ctbc" else t) for t in rough_terms], flat_terms
            )
            self.assertIs(self.cfg.observations[group].terms["ctbc"].func, ctbc.ctbc_obs)

    def test_ctbc_can_be_switched_off_for_ablation(self) -> None:
        off = rough_env_cfg(ctbc_enabled=False)
        self.assertNotIn("init_ctbc_state", off.events)
        self.assertIn("jump_commands", off.observations["actor"].terms)
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
            # 平地列：四周全平，读数应接近 0。
            self.assertLess(float(scan[flat].abs().max()), 0.02)
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
