"""崎岖地形任务的回归测试。

rough = 冻结的 Flat 基线 + 一层薄覆盖：地形/升降级课程/截断/高度扫描用 mjlab 官方件，
自己的部分只有指令分列覆盖、台阶专项奖励、按列奖励包装、平地热身。本文件把这几条钉住，
默认定价（A15）改动必须同步改这里并在提交信息里写对照实验编号。
"""

from __future__ import annotations

import collections
import unittest

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import height_scan
from mjlab.envs.mdp.rewards import is_terminated
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.mdp.curriculums import terrain_levels_vel
from mjlab.tasks.velocity.mdp.terminations import out_of_terrain_bounds, terrain_edge_reached
from mjlab.terrains import (
    BoxInvertedPyramidStairsTerrainCfg,
    BoxPyramidStairsTerrainCfg,
)

import se3_train  # noqa: F401  # 注册任务
from se3_train.log_filter import keep_log_key
from se3_train.tasks.flat.env_cfg import (
    FLAT_ACTION_SMOOTHNESS_SPRING,
    FLAT_CMD_VEL_DEADBAND,
    FLAT_WHEEL_ACTION_SCALE,
)
from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg
from se3_train.tasks.rough import curriculums, events
from se3_train.tasks.rough import rewards as rough_rewards
from se3_train.tasks.rough.columns import column_mask, non_flat_column_mask
from se3_train.tasks.rough.commands import RoughCommandCfg
from se3_train.tasks.rough.env_cfg import (
    ROUGH_ALL_TERRAIN_TYPE_NAMES,
    ROUGH_BASE_HEIGHT_SIGMA,
    ROUGH_COMMAND_VELOCITY_ERROR_LIN_SCALE,
    ROUGH_CRITIC_HEIGHT_SCAN_SENSOR_NAME,
    ROUGH_FALL_PENALTY,
    ROUGH_FLAT_VZ_WEIGHT,
    ROUGH_FLAT_WARMUP_ITERATIONS,
    ROUGH_FLAT_WARMUP_RAMP_ITERATIONS,
    ROUGH_MAX_INIT_TERRAIN_LEVEL,
    ROUGH_NCONMAX,
    ROUGH_NJMAX,
    ROUGH_OFF_STAIR_TRACKING_SIGMA_MOVE,
    ROUGH_REWARD_TERRAIN_TYPE_NAMES,
    ROUGH_ROBOT_COLLISION_GEOM_GROUP,
    ROUGH_STAIR_ANG_VEL_YAW_RANGE,
    ROUGH_STAIR_COMMAND_TERRAIN_NAMES,
    ROUGH_STAIR_HEIGHT_RANGE,
    ROUGH_STAIR_LIN_VEL_X_RANGE,
    ROUGH_STAIR_TRACKING_SIGMA_MOVE,
    ROUGH_STAIRS_ZEROED_REWARDS,
    ROUGH_TERRAIN_ANG_VEL_YAW_RANGE,
    ROUGH_TERRAIN_EDGE_THRESHOLD_FRACTION,
    ROUGH_BASE_HEIGHT_OFF_COLUMNS,
    ROUGH_TERRAIN_HEIGHT_CLEARANCE,
    ROUGH_TERRAIN_LIN_VEL_X_RANGE,
    ROUGH_TERRAIN_STEP_HEIGHT_TYPE_NAMES,
    ROUGH_TERRAIN_VZ_WEIGHT,
)
from se3_train.tasks.rough.env_cfg import env_cfg as rough_env_cfg
from se3_train.tasks.rough.terrains import (
    ROUGH_PATCH_SIZE,
    ROUGH_STEP_HEIGHT_RANGE,
    ROUGH_TERRAIN_PROPORTIONS,
)

_ROUGH = "SE3-WheelLegged-Rough"
_STAIR_EVAL = "SE3-WheelLegged-Rough-StairEval"
_FLAT_MLP = "SE3-WheelLegged-Flat-MLP"
_WRAPPED = (
    "flat_base_height",
    "tracking_lin_vel",
    "tracking_ang_vel",
    *ROUGH_STAIRS_ZEROED_REWARDS,
)
_ROUGH_ONLY = (
    "command_velocity_error",
    "stair_climb_progress",
    "stair_support_height",
    "fall_penalty",
)


def _all_ids(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.arange(env.num_envs, device=env.device)


def _step_once(env: ManagerBasedRlEnv) -> None:
    """走一步零动作，让 RewardManager 的逐项缓冲 `_step_reward` 有真实值（reset 后它是全 0）。"""
    env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))


def _place_offset(env: ManagerBasedRlEnv, dx: float, dy: float) -> None:
    robot = env.scene["robot"]
    pose = robot.data.root_link_pose_w.clone()
    pose[:, 0] = env.scene.env_origins[:, 0] + dx
    pose[:, 1] = env.scene.env_origins[:, 1] + dy
    robot.write_root_link_pose_to_sim(pose)
    env.sim.forward()


class RoughInheritsFlatBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_env_cfg(_ROUGH)
        cls.flat = flat_env_cfg(
            wheel_action_scale=FLAT_WHEEL_ACTION_SCALE,
            action_smoothness=FLAT_ACTION_SMOOTHNESS_SPRING,
        )

    def test_ppo_matches_flat_baseline_except_iterations(self) -> None:
        rough = load_rl_cfg(_ROUGH)
        flat = load_rl_cfg(_FLAT_MLP)
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
            self.assertEqual(getattr(rough.algorithm, key), getattr(flat.algorithm, key), msg=key)
        self.assertEqual(rough.num_steps_per_env, 24)
        self.assertEqual(rough.max_iterations, 5000)

    def test_action_and_command_contract_match_flat(self) -> None:
        action = self.cfg.actions["delayed_action"]
        self.assertEqual(action.leg_action_semantics, "joint")
        self.assertAlmostEqual(float(action.wheel_scale), FLAT_WHEEL_ACTION_SCALE)
        command = self.cfg.commands["velocity_height"]
        self.assertIsInstance(command, RoughCommandCfg)
        self.assertEqual(tuple(command.height_range), (0.20, 0.38))
        self.assertEqual(tuple(command.deployment_ranges["height"]), (0.20, 0.38))
        # actor 观测契约不变：高度扫描只进 critic。
        self.assertNotIn("height_scan", self.cfg.observations["actor"].terms)
        self.assertEqual(
            list(self.cfg.observations["actor"].terms), list(self.flat.observations["actor"].terms)
        )

    def test_rough_only_adds_three_rewards_and_wraps_three(self) -> None:
        """相对 Flat：新增两项台阶专项奖励与违令罚；三项换成按列包装；其余项函数与权重逐位相同。"""
        self.assertEqual(set(self.cfg.rewards) - set(self.flat.rewards), set(_ROUGH_ONLY))
        self.assertEqual(set(self.flat.rewards) - set(self.cfg.rewards), set())
        for name, term in self.cfg.rewards.items():
            if name in _ROUGH_ONLY:
                continue
            base = self.flat.rewards[name]
            self.assertAlmostEqual(float(term.weight), float(base.weight), places=12, msg=name)
            if name in _WRAPPED:
                continue
            self.assertIs(term.func, base.func, msg=name)
            self.assertEqual(term.params, base.params, msg=name)

    def test_a15_pricing_on_non_stair_columns(self) -> None:
        """默认定价 = A15（h85eljnj）：高度 σ 0.10、运动核 0.5、平地 vz 0、违令罚全六列。"""
        height = self.cfg.rewards["flat_base_height"]
        self.assertIs(height.func, rough_rewards.base_height_penalty_off_terrain)
        self.assertAlmostEqual(height.params["sigma"], ROUGH_BASE_HEIGHT_SIGMA)
        # M3：高度罚不再在任何列置零（空列名 → 包装退化为 Flat 原函数）。
        self.assertEqual(tuple(height.params["terrain_type_names"]), ROUGH_BASE_HEIGHT_OFF_COLUMNS)
        self.assertEqual(ROUGH_BASE_HEIGHT_OFF_COLUMNS, ())
        self.assertEqual(
            height.params.get("max_error"),
            self.flat.rewards["flat_base_height"].params.get("max_error"),
        )

        track = self.cfg.rewards["tracking_lin_vel"]
        self.assertIs(track.func, rough_rewards.tracking_lin_vel_terrain_vz)
        self.assertAlmostEqual(track.params["sigma_move"], ROUGH_OFF_STAIR_TRACKING_SIGMA_MOVE)
        self.assertAlmostEqual(track.params["vz_weight"], ROUGH_FLAT_VZ_WEIGHT)
        self.assertAlmostEqual(track.params["terrain_vz_weight"], ROUGH_TERRAIN_VZ_WEIGHT)
        self.assertAlmostEqual(track.params["stair_sigma_move"], ROUGH_STAIR_TRACKING_SIGMA_MOVE)
        self.assertEqual(
            track.params["sigma_stand"], self.flat.rewards["tracking_lin_vel"].params["sigma_stand"]
        )

        ang = self.cfg.rewards["tracking_ang_vel"]
        self.assertIs(ang.func, rough_rewards.tracking_ang_vel_off_terrain)
        self.assertEqual(tuple(ang.params["terrain_type_names"]), ROUGH_REWARD_TERRAIN_TYPE_NAMES)

        err = self.cfg.rewards["command_velocity_error"]
        self.assertIs(err.func, rough_rewards.command_velocity_error_on_terrain)
        self.assertLess(float(err.weight), 0.0)
        self.assertEqual(tuple(err.params["terrain_type_names"]), ROUGH_ALL_TERRAIN_TYPE_NAMES)
        self.assertAlmostEqual(err.params["lin_vel_scale"], ROUGH_COMMAND_VELOCITY_ERROR_LIN_SCALE)
        self.assertAlmostEqual(err.params["lin_deadband"], float(FLAT_CMD_VEL_DEADBAND[0]))
        # 误差尺度要让台阶列的典型误差（1.6–2.4 m/s）落在二次区而不是贴封顶。
        self.assertLess(
            (2.4 / ROUGH_COMMAND_VELOCITY_ERROR_LIN_SCALE) ** 2, err.params["max_penalty"]
        )
        self.assertEqual(self.cfg.rewards["stair_climb_progress"].weight, 3.0)
        self.assertEqual(self.cfg.rewards["stair_support_height"].weight, 4.0)

    def test_domain_randomization_matches_flat(self) -> None:
        self.assertAlmostEqual(self.cfg.events["com"].params["com_range"], 0.005)

    def test_m2_stairs_column_pricing_and_fall_penalty(self) -> None:
        """M2：台阶列 is_alive / flat_wheel_contact / collision 置零（权重与原参数沿用 Flat），加一次性摔倒罚。"""
        for name in ROUGH_STAIRS_ZEROED_REWARDS:
            term = self.cfg.rewards[name]
            base = self.flat.rewards[name]
            self.assertIs(term.func, rough_rewards.off_column, msg=name)
            self.assertIs(term.params["inner"], base.func, msg=name)
            self.assertEqual(term.params["params"], base.params, msg=name)
            self.assertEqual(
                tuple(term.params["terrain_type_names"]), ROUGH_REWARD_TERRAIN_TYPE_NAMES, msg=name
            )
            self.assertAlmostEqual(float(term.weight), float(base.weight), places=12, msg=name)
        fall = self.cfg.rewards["fall_penalty"]
        self.assertIs(fall.func, is_terminated)
        step_dt = float(self.cfg.sim.mujoco.timestep) * int(self.cfg.decimation)
        # RewardManager 按 dt 缩放，权重反缩放后每次非超时终止恰好扣 ROUGH_FALL_PENALTY。
        self.assertTrue(self.cfg.scale_rewards_by_dt)
        self.assertAlmostEqual(float(fall.weight) * step_dt, -ROUGH_FALL_PENALTY, places=9)
        self.assertNotIn("fall_penalty", self.flat.rewards)

    def test_robot_collision_geoms_leave_group_zero_only_in_rough(self) -> None:
        """rough 把碰撞 geom 挪出 group 0，高度射线（只看 group 0）就不会打到自己；Flat 不动。"""
        rough_groups = collections.Counter(
            g.group for g in self.cfg.scene.entities["robot"].spec_fn().geoms
        )
        flat_groups = collections.Counter(
            g.group for g in self.flat.scene.entities["robot"].spec_fn().geoms
        )
        self.assertNotIn(0, rough_groups)
        self.assertEqual(rough_groups[ROUGH_ROBOT_COLLISION_GEOM_GROUP], flat_groups[0])
        self.assertGreater(flat_groups[0], 0)
        self.assertEqual(sum(rough_groups.values()), sum(flat_groups.values()))


class RoughTerrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_env_cfg(_ROUGH)

    def test_terrain_generator_is_official_curriculum_grid(self) -> None:
        terrain = self.cfg.scene.terrain
        assert terrain is not None
        gen = terrain.terrain_generator
        assert gen is not None
        self.assertTrue(gen.curriculum)
        self.assertEqual(terrain.max_init_terrain_level, ROUGH_MAX_INIT_TERRAIN_LEVEL)
        # 2026-09-13：只剩 flat 与 stairs_up 两列。比例为 0 的列不能靠设 0 关掉——mjlab 课程模式
        # 仍会生成几何并分 1 个 env，之前四个死列白占 160 个 geom、每轮多 0.6 s。
        self.assertEqual(list(gen.sub_terrains), list(ROUGH_ALL_TERRAIN_TYPE_NAMES))
        self.assertEqual(list(gen.sub_terrains), ["flat", "stairs_up"])
        self.assertEqual(tuple(gen.size), ROUGH_PATCH_SIZE)
        for name, sub in gen.sub_terrains.items():
            self.assertAlmostEqual(sub.proportion, ROUGH_TERRAIN_PROPORTIONS[name], msg=name)
            self.assertGreater(sub.proportion, 0.0, msg=name)
            self.assertEqual(tuple(sub.size), ROUGH_PATCH_SIZE, msg=name)
        # 上台阶出生在坑底向外爬升（反金字塔）。
        self.assertIsInstance(gen.sub_terrains["stairs_up"], BoxInvertedPyramidStairsTerrainCfg)
        self.assertEqual(
            tuple(gen.sub_terrains["stairs_up"].step_height_range), ROUGH_STEP_HEIGHT_RANGE
        )
        self.assertGreaterEqual(self.cfg.sim.contact_sensor_maxmatch, 500)

    def test_sim_pool_sizes_follow_overflow_measurement(self) -> None:
        # 2026-09-13 用 M1 的 model_2400 按训练方式采样、8192 env、起步行 0–9、2000 步压测：
        # 每世界约束峰值 54、接触峰值 10；256 / 64 全程零溢出（scripts/check_sim_overflow.py）。
        measured_nefc_peak, measured_ncon_peak = 54, 10
        self.assertEqual(self.cfg.sim.njmax, ROUGH_NJMAX)
        self.assertEqual(self.cfg.sim.nconmax, ROUGH_NCONMAX)
        self.assertGreaterEqual(ROUGH_NJMAX, 3 * measured_nefc_peak)
        self.assertGreaterEqual(ROUGH_NCONMAX, 3 * measured_ncon_peak)
        # 台阶定向评测共用同一套覆盖层。
        stair_eval = load_env_cfg(_STAIR_EVAL)
        self.assertEqual(stair_eval.sim.njmax, ROUGH_NJMAX)
        self.assertEqual(stair_eval.sim.nconmax, ROUGH_NCONMAX)

    def test_stair_eval_variant(self) -> None:
        gen = load_env_cfg(_STAIR_EVAL).scene.terrain.terrain_generator
        assert gen is not None
        self.assertEqual(list(gen.sub_terrains), ["flat", "stairs_up", "stairs_down"])
        self.assertTrue(gen.curriculum)
        # 下台阶出生在顶部平台向外下行（正金字塔），与上台阶相反。
        self.assertIsInstance(gen.sub_terrains["stairs_down"], BoxPyramidStairsTerrainCfg)
        self.assertNotIsInstance(
            gen.sub_terrains["stairs_down"], BoxInvertedPyramidStairsTerrainCfg
        )

    def test_terrain_curriculum_is_the_official_one_and_runs_before_warmup(self) -> None:
        cur = self.cfg.curriculum
        self.assertIs(cur["terrain_levels"].func, terrain_levels_vel)
        self.assertEqual(cur["terrain_levels"].params, {"command_name": "velocity_height"})
        self.assertIs(cur["flat_warmup"].func, curriculums.flat_warmup)
        params = cur["flat_warmup"].params
        self.assertEqual(params["iterations"], ROUGH_FLAT_WARMUP_ITERATIONS)
        self.assertEqual(params["ramp_iterations"], ROUGH_FLAT_WARMUP_RAMP_ITERATIONS)
        self.assertEqual(params["steps_per_policy_iter"], load_rl_cfg(_ROUGH).num_steps_per_env)
        # 先结算升降级，再换列（换列那次 reset 的位移是旧地块上的，不能拿来升级）。
        self.assertLess(list(cur).index("terrain_levels"), list(cur).index("flat_warmup"))
        self.assertEqual(
            cur["command_vel"].params["tracking_log_key"],
            "Locomotion/tracking_lin_vel_reward_curriculum",
        )
        self.assertEqual(self.cfg.events["set_curriculum_env_mask"].mode, "startup")
        # play 模式沿用 Flat：没有任何课程项。
        self.assertEqual(rough_env_cfg(play=True).curriculum, {})

    def test_terminations_are_official_truncations(self) -> None:
        edge = self.cfg.terminations["terrain_edge_reached"]
        self.assertIs(edge.func, terrain_edge_reached)
        self.assertTrue(edge.time_out)
        self.assertAlmostEqual(
            edge.params["threshold_fraction"], ROUGH_TERRAIN_EDGE_THRESHOLD_FRACTION
        )
        # 截断门槛不能低于官方升级门槛（欧氏距离 > 块半边长），否则永远升不了级。
        self.assertGreaterEqual(ROUGH_TERRAIN_EDGE_THRESHOLD_FRACTION, 1.0)
        bounds = self.cfg.terminations["out_of_terrain_bounds"]
        self.assertIs(bounds.func, out_of_terrain_bounds)
        self.assertTrue(bounds.time_out)
        for name in ("time_out", "bad_orientation", "catastrophic_state", "leg_contact"):
            self.assertIn(name, self.cfg.terminations)

    def test_critic_height_scan_is_official_and_terrain_only(self) -> None:
        term = self.cfg.observations["critic"].terms["height_scan"]
        self.assertIs(term.func, height_scan)
        self.assertEqual(term.params["sensor_name"], ROUGH_CRITIC_HEIGHT_SCAN_SENSOR_NAME)
        sensor = next(
            s for s in self.cfg.scene.sensors if s.name == ROUGH_CRITIC_HEIGHT_SCAN_SENSOR_NAME
        )
        self.assertEqual(tuple(sensor.include_geom_groups), (0,))
        self.assertEqual(tuple(sensor.pattern.size), (1.0, 0.6))
        self.assertAlmostEqual(sensor.pattern.resolution, 0.1)
        self.assertNotIn("height_scan", self.cfg.observations["actor"].terms)

    def test_command_ranges_and_height_floor_are_configured(self) -> None:
        cmd = self.cfg.commands["velocity_height"]
        self.assertTrue(cmd.terrain_command_override_enabled)
        self.assertEqual(tuple(cmd.terrain_lin_vel_x_range), ROUGH_TERRAIN_LIN_VEL_X_RANGE)
        self.assertEqual(tuple(cmd.terrain_ang_vel_yaw_range), ROUGH_TERRAIN_ANG_VEL_YAW_RANGE)
        self.assertFalse(cmd.terrain_lin_vel_x_follow_curriculum)
        self.assertEqual(tuple(cmd.stair_command_terrain_names), ROUGH_STAIR_COMMAND_TERRAIN_NAMES)
        self.assertEqual(tuple(cmd.stair_lin_vel_x_range), ROUGH_STAIR_LIN_VEL_X_RANGE)
        self.assertEqual(tuple(cmd.stair_ang_vel_yaw_range), ROUGH_STAIR_ANG_VEL_YAW_RANGE)
        self.assertEqual(tuple(cmd.stair_height_range), ROUGH_STAIR_HEIGHT_RANGE)
        self.assertEqual(cmd.high_stand_transition_prob, 0.0)
        # 地形感知高度下限：跨完整台阶区间且不撞到高度上界。
        self.assertTrue(cmd.terrain_aware_height)
        self.assertAlmostEqual(cmd.terrain_height_clearance, ROUGH_TERRAIN_HEIGHT_CLEARANCE)
        self.assertEqual(
            tuple(cmd.terrain_step_height_type_names), ROUGH_TERRAIN_STEP_HEIGHT_TYPE_NAMES
        )
        gen = self.cfg.scene.terrain.terrain_generator
        for name in ROUGH_TERRAIN_STEP_HEIGHT_TYPE_NAMES:
            self.assertTrue(hasattr(gen.sub_terrains[name], "step_height_range"), name)
        floor_top = (
            ROUGH_STEP_HEIGHT_RANGE[1]
            + cmd.terrain_height_clearance
            - cmd.body_collision_bottom_offset
        )
        self.assertLess(floor_top, cmd.height_range[1])
        self.assertGreater(floor_top, cmd.height_range[0])
        # 高度下限、分列奖励与台阶指令指向同一列。
        self.assertEqual(ROUGH_TERRAIN_STEP_HEIGHT_TYPE_NAMES, ROUGH_REWARD_TERRAIN_TYPE_NAMES)
        self.assertEqual(ROUGH_STAIR_COMMAND_TERRAIN_NAMES, ROUGH_REWARD_TERRAIN_TYPE_NAMES)

    def test_column_diagnostics_survive_log_filter(self) -> None:
        for key in (
            "Rough/height_cmd_terrain_mean",
            "Rough/base_vx_terrain",
            "Rough/stair_supported_steps",
            f"{events.REWARD_SPLIT_LOG_PREFIX}tracking_lin_vel_stairs",
            "Curriculum/terrain_levels/stairs_up",
            "Curriculum/flat_warmup/active",
            "Episode_Termination/terrain_edge_reached",
        ):
            self.assertTrue(keep_log_key(key), key)


class RoughRuntimeTests(unittest.TestCase):
    """在 CPU 上建一个小环境，验证分列指令、分列奖励与官方课程/截断在本机器人上真的生效。"""

    @classmethod
    def setUpClass(cls) -> None:
        cfg = rough_env_cfg()
        # 热身会把首次 reset 的全部 env 放到平地列；这里要两列各占 6 个 env。
        cfg.curriculum.pop("flat_warmup")
        cfg.scene.num_envs = 12
        cls.env = ManagerBasedRlEnv(cfg, device="cpu")
        cls.env.reset()
        cls.term = cls.env.command_manager.get_term("velocity_height")
        terrain = cls.env.scene.terrain
        assert terrain is not None
        cls.names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
        cls.flat_col = cls.names.index("flat")
        cls.types = terrain.terrain_types.to(dtype=torch.long)
        cls.stairs = cls.types == cls.names.index("stairs_up")
        cls.flat = cls.types == cls.flat_col

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_every_column_is_populated_and_masks_agree(self) -> None:
        self.assertEqual(sorted(set(self.types.tolist())), list(range(len(self.names))))
        self.assertEqual(len(self.names), 2)
        self.assertTrue(torch.equal(column_mask(self.env, ("stairs_up",)), self.stairs))
        self.assertTrue(torch.equal(non_flat_column_mask(self.env), ~self.flat))
        self.assertIsNone(column_mask(self.env, ("no_such_column",)))
        self.assertTrue(torch.equal(self.term._terrain_override_mask, ~self.flat))
        self.assertTrue(torch.equal(self.term._stair_mask, self.stairs))
        mask = getattr(self.env, events.CURRICULUM_ENV_MASK_ATTR)
        self.assertTrue(torch.equal(mask, self.flat))

    def test_non_flat_columns_only_get_forward_commands(self) -> None:
        # 2026-09-13 起训练地形只有 flat 与 stairs_up：非平地列就是台阶列，通用地形列覆盖
        # （ROUGH_TERRAIN_*_RANGE）只在以后再加列时才会被用到。
        self.assertTrue(torch.equal(~self.flat, self.stairs))
        yaw_lo, yaw_hi = ROUGH_TERRAIN_ANG_VEL_YAW_RANGE
        vx_lo, vx_hi = ROUGH_STAIR_LIN_VEL_X_RANGE
        for _ in range(50):  # 多抽几轮，静站样本（10%）若漏进非平地列一定会被抓到
            self.term._resample_command(_all_ids(self.env))
            cmd = self.term.command[self.stairs]
            self.assertTrue(
                bool((cmd[:, 0] >= vx_lo - 1e-6).all()) and bool((cmd[:, 0] <= vx_hi + 1e-6).all())
            )
            self.assertTrue(
                bool((cmd[:, 1] >= yaw_lo - 1e-6).all())
                and bool((cmd[:, 1] <= yaw_hi + 1e-6).all())
            )
            self.assertEqual(float(self.term.command[self.stairs, 1].abs().max()), 0.0)
            self.assertFalse(bool(self.term._standing_mask[~self.flat].any()))
        # Flat 速度课程起点 vx=yaw=0，平地列 reset 后指令必须仍是 0。
        self.assertTrue(bool((self.term.command[self.flat][:, :2].abs() < 1e-6).all()))

    def test_terrain_vx_is_decoupled_from_the_flat_curriculum(self) -> None:
        lo, hi = ROUGH_STAIR_LIN_VEL_X_RANGE
        saved = self.term.cfg.lin_vel_x_range
        try:
            for flat_range in ((0.0, 0.0), (-2.4, 2.4)):
                self.term.cfg.lin_vel_x_range = flat_range
                for _ in range(10):
                    self.term._resample_command(_all_ids(self.env))
                    vx = self.term.command[self.stairs][:, 0]
                    self.assertGreaterEqual(float(vx.min()), lo - 1e-5)
                    self.assertLessEqual(float(vx.max()), hi + 1e-5)
        finally:
            self.term.cfg.lin_vel_x_range = saved

    def test_stair_height_follows_range_and_terrain_floor(self) -> None:
        terrain = self.env.scene.terrain
        cfg = self.term.cfg
        low, high = (float(v) for v in cfg.stair_height_range)
        step_low, step_high = ROUGH_STEP_HEIGHT_RANGE
        num_rows = int(terrain.terrain_origins.shape[0])
        saved = terrain.terrain_levels.clone()
        try:
            for row in (0, num_rows - 1):
                terrain.terrain_levels[self.stairs] = row
                step = step_low + row / (num_rows - 1) * (step_high - step_low)
                floor = min(
                    high,
                    max(
                        low, step + cfg.terrain_height_clearance - cfg.body_collision_bottom_offset
                    ),
                )
                seen_min = 1.0
                other_min = 1.0
                for _ in range(20):
                    self.term._resample_command(_all_ids(self.env))
                    h = self.term.command[self.stairs, 4]
                    self.assertGreaterEqual(float(h.min()), floor - 1e-6, msg=f"row={row}")
                    self.assertLessEqual(float(h.max()), high + 1e-6, msg=f"row={row}")
                    seen_min = min(seen_min, float(h.min()))
                    other_min = min(other_min, float(self.term.command[~self.stairs, 4].min()))
                # 下界真的贴着地板，而不是恒等于某个端点。
                self.assertLess(seen_min, floor + 0.03, msg=f"row={row}")
                if row == num_rows - 1:
                    # 最高行的下限 0.34 只作用在台阶列：其余列仍能抽到矮站姿。
                    self.assertGreater(floor, low)
                    self.assertLess(other_min, floor)
            # 台阶高度改完必须同步刷新高度条件默认腿姿缓存。
            from se3_shared import RobotConfig, policy_default_from_height_torch
            from se3_train.mdp.height_default_cache import get_policy_default_from_height_cache

            cache = get_policy_default_from_height_cache(
                self.env, "velocity_height", device=torch.device("cpu"), dtype=torch.float32
            )
            expected = policy_default_from_height_torch(self.term.command[:, 4], RobotConfig())
            self.assertLess(float((cache - expected).abs().max()), 1e-5)
        finally:
            terrain.terrain_levels[:] = saved

    def test_column_rewards_only_bite_where_configured(self) -> None:
        """包装函数点名列时台阶列不吃高度罚；配置里（M3）高度罚全列生效；yaw 工资台阶列为 0；违令罚全列（A15）。"""
        _step_once(self.env)
        cmd = self.env.command_manager.get_command("velocity_height")
        saved = cmd.clone()
        try:
            cmd[:, 0] = 2.0
            cmd[:, 1] = 0.0
            cmd[:, 4] = 0.0
            cmd[:, 5] = 0.0
            vel_pen = rough_rewards.command_velocity_error_on_terrain(
                self.env,
                command_name="velocity_height",
                terrain_type_names=ROUGH_ALL_TERRAIN_TYPE_NAMES,
                lin_vel_scale=3.0,
            )
            height_pen = rough_rewards.base_height_penalty_off_terrain(
                self.env,
                command_name="velocity_height",
                height_sensor_name="base_height_sensor",
                sigma=0.10,
            )
        finally:
            cmd[:] = saved
        self.assertGreater(float(vel_pen.min()), 0.0)
        self.assertEqual(float(height_pen[self.stairs].abs().max()), 0.0)
        self.assertGreater(float(height_pen[~self.stairs].min()), 0.0)
        manager = self.env.reward_manager
        # M3：配置里的 flat_base_height 在台阶列真的在扣（reset 后高度指令与实际高度不一致）。
        h_index = manager.active_terms.index("flat_base_height")
        self.assertLess(float(manager._step_reward[self.stairs, h_index].min()), 0.0)
        self.assertLess(float(manager._step_reward[self.flat, h_index].min()), 0.0)
        index = manager.active_terms.index("tracking_ang_vel")
        self.assertEqual(float(manager._step_reward[self.stairs, index].abs().max()), 0.0)
        self.assertGreater(float(manager._step_reward[self.flat, index].abs().max()), 0.0)

    def test_reward_split_logs_every_term_on_the_stairs_column(self) -> None:
        _step_once(self.env)
        events.log_reward_split_by_column(
            self.env, _all_ids(self.env), terrain_type_names=("stairs_up",)
        )
        log = self.env.extras.get("log", {})
        manager = self.env.reward_manager
        terms = list(manager.active_terms)
        keys = [k for k in log if k.startswith(events.REWARD_SPLIT_LOG_PREFIX)]
        self.assertEqual(len(keys), len(terms))
        for name in terms:
            logged = float(log[f"{events.REWARD_SPLIT_LOG_PREFIX}{name}_stairs"])
            manual = float(manager._step_reward[self.stairs, terms.index(name)].mean())
            self.assertAlmostEqual(logged, manual, places=6, msg=name)

    def test_curriculum_tracking_key_is_logged(self) -> None:
        self.env._se3_reward_log_interval_steps = 1
        self.env.step(torch.zeros(self.env.num_envs, self.env.action_manager.total_action_dim))
        log = self.env.extras.get("log", {})
        self.assertIn("Locomotion/tracking_lin_vel_reward_curriculum", log)
        self.assertIn("Rough/base_vx_terrain", log)

    def test_official_height_scan_sees_terrain_only(self) -> None:
        """平地列上 77 条射线全打在同一平面上：机身足迹内的射线不再打到腿和轮子。"""
        scan = height_scan(self.env, ROUGH_CRITIC_HEIGHT_SCAN_SENSOR_NAME)
        self.assertEqual(tuple(scan.shape), (self.env.num_envs, 77))
        flat_scan = scan[self.flat]
        # 每个 env 内 77 条射线读数一致（不同 env 的机身高度可以不同）。
        spread = flat_scan.max(dim=1).values - flat_scan.min(dim=1).values
        self.assertLess(float(spread.max()), 0.01, msg=str(flat_scan))
        self.assertLess(float(flat_scan.max()), 1.0)  # 没有打空的射线（打空会填 max_distance=2.0）
        obs = self.env.observation_manager.compute()["critic"]
        self.assertGreaterEqual(obs.shape[1], 77)

    def test_official_curriculum_promotes_demotes_and_freezes(self) -> None:
        terrain = self.env.scene.terrain
        env_ids = _all_ids(self.env)
        cmd = self.env.command_manager.get_command("velocity_height")
        saved_levels = terrain.terrain_levels.clone()
        saved_cmd = cmd.clone()
        saved_step = self.env.common_step_counter
        try:
            self.env.common_step_counter = 100
            cmd[:, 0] = 1.6
            cmd[:, 1] = 0.0
            terrain.terrain_levels[:] = 3
            terrain.env_origins[:] = terrain.terrain_origins[
                terrain.terrain_levels, terrain.terrain_types
            ]
            # 欧氏距离 4.6 m > 块半边长 4.5 m：升一级。
            _place_offset(self.env, 4.6, 0.0)
            terrain_levels_vel(self.env, env_ids, command_name="velocity_height")
            self.assertTrue(bool((terrain.terrain_levels == 4).all()))
            # 只走了 1 m，指令 1.6 m/s × 20 s 的一半是 16 m：降一级。
            _place_offset(self.env, 1.0, 0.0)
            terrain_levels_vel(self.env, env_ids, command_name="velocity_height")
            self.assertTrue(bool((terrain.terrain_levels == 3).all()))
            # 静站指令下没有应走距离，不升不降。
            cmd[:, 0] = 0.0
            _place_offset(self.env, 1.0, 0.0)
            terrain_levels_vel(self.env, env_ids, command_name="velocity_height")
            self.assertTrue(bool((terrain.terrain_levels == 3).all()))
            # 首次 reset（还没走过任何一步）冻结，不能凭出生前的位置升级。
            self.env.common_step_counter = 0
            _place_offset(self.env, 30.0, 30.0)
            terrain_levels_vel(self.env, env_ids, command_name="velocity_height")
            self.assertTrue(bool((terrain.terrain_levels == 3).all()))
        finally:
            self.env.common_step_counter = saved_step
            cmd[:] = saved_cmd
            terrain.terrain_levels[:] = saved_levels
            terrain.env_origins[:] = terrain.terrain_origins[
                terrain.terrain_levels, terrain.terrain_types
            ]
            _place_offset(self.env, 0.0, 0.0)

    def test_edge_truncation_coincides_with_promotion_threshold(self) -> None:
        saved = self.env.episode_length_buf.clone()
        try:
            self.env.episode_length_buf[:] = 10
            _place_offset(self.env, 4.4, 0.0)
            self.assertFalse(
                bool(
                    terrain_edge_reached(
                        self.env, threshold_fraction=ROUGH_TERRAIN_EDGE_THRESHOLD_FRACTION
                    ).any()
                )
            )
            _place_offset(self.env, 4.6, 0.0)
            self.assertTrue(
                bool(
                    terrain_edge_reached(
                        self.env, threshold_fraction=ROUGH_TERRAIN_EDGE_THRESHOLD_FRACTION
                    ).all()
                )
            )
            self.assertFalse(bool(out_of_terrain_bounds(self.env).any()))
        finally:
            self.env.episode_length_buf[:] = saved
            _place_offset(self.env, 0.0, 0.0)

    def test_jump_metrics_are_host_sync_free(self) -> None:
        """Jump/* 诊断在行走线默认关闭；打开时也不能有 .item() 之类的主机同步（2026-09-13 每步 184 次）。"""
        term = self.term
        self.assertFalse(term.cfg.enable_jump_metrics)
        term.cfg.enable_jump_metrics = True
        original_item = torch.Tensor.item

        def _forbidden_item(tensor: torch.Tensor):
            raise AssertionError("Jump/* 诊断不得调用 .item()")

        try:
            self.env.extras["log"] = {}
            torch.Tensor.item = _forbidden_item  # type: ignore[method-assign]
            term._update_metrics()
        finally:
            torch.Tensor.item = original_item  # type: ignore[method-assign]
            term.cfg.enable_jump_metrics = False
        log = self.env.extras["log"]
        self.assertIn("Jump/jump_flag_ratio", log)
        self.assertIn("Jump/diag_takeoff_vz_progress_ema", log)
        self.assertIn("Jump/diag_standing_joint_mirror_raw", log)
        for key, value in log.items():
            # Jump/diag_leg_contact_* 由 terminations.leg_contact() 写入，不在本函数的改动范围。
            if key.startswith("Jump/") and not key.startswith("Jump/diag_leg_contact_"):
                self.assertIsInstance(value, torch.Tensor, key)
                self.assertEqual(value.dim(), 0, key)
                self.assertTrue(torch.isfinite(value).all(), key)
        # 行走线没有跳跃：空中样本为 0 时各项应为 0，而不是 NaN。
        self.assertEqual(float(log["Jump/diag_max_airborne_vz"]), 0.0)
        self.assertEqual(float(log["Jump/diag_jump_success_rate"]), 0.0)

    def test_stairs_column_pays_no_wage_and_no_climbing_tax(self) -> None:
        """M2 运行时：台阶列上三项为 0；平地列 is_alive 照发；摔倒罚项已挂上。"""
        _step_once(self.env)
        manager = self.env.reward_manager
        for name in ROUGH_STAIRS_ZEROED_REWARDS:
            idx = manager.active_terms.index(name)
            self.assertEqual(
                float(manager._step_reward[self.stairs, idx].abs().max()), 0.0, msg=name
            )
        alive = manager._step_reward[self.flat, manager.active_terms.index("is_alive")]
        self.assertGreater(float(alive.min()), 0.0)
        self.assertIn("fall_penalty", manager.active_terms)


class FlatWarmupRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cfg = rough_env_cfg()
        cfg.curriculum["flat_warmup"].params.update({"iterations": 2, "ramp_iterations": 0})
        cfg.scene.num_envs = 12
        cls.env = ManagerBasedRlEnv(cfg, device="cpu")
        cls.env.reset()
        terrain = cls.env.scene.terrain
        assert terrain is not None
        cls.terrain = terrain
        cls.flat_col = list(terrain.cfg.terrain_generator.sub_terrains.keys()).index("flat")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_everyone_starts_on_flat_then_returns_to_own_column(self) -> None:
        original = getattr(self.env, curriculums.FLAT_WARMUP_ORIGINAL_TYPES_ATTR)
        self.assertTrue(bool((self.terrain.terrain_types == self.flat_col).all()))
        self.assertGreater(len(set(original.tolist())), 1)
        term = self.env.command_manager.get_term("velocity_height")
        self.assertFalse(bool(term._terrain_override_mask.any()))
        # 热身结束：所有 env 在下一次 reset 回到原列第 0 行，掩码同步刷新。
        self.env.common_step_counter = 3 * 24
        curriculums.flat_warmup(
            self.env,
            _all_ids(self.env),
            "velocity_height",
            iterations=2,
            ramp_iterations=0,
            steps_per_policy_iter=24,
        )
        self.assertTrue(torch.equal(self.terrain.terrain_types, original))
        self.assertTrue(bool((self.terrain.terrain_levels == 0).all()))
        self.assertTrue(torch.equal(term._terrain_override_mask, original != self.flat_col))
        self.assertTrue(
            torch.equal(
                getattr(self.env, events.CURRICULUM_ENV_MASK_ATTR), original == self.flat_col
            )
        )
        self.assertTrue(bool(getattr(self.env, curriculums.FLAT_WARMUP_DONE_ATTR).all()))


if __name__ == "__main__":
    unittest.main()
