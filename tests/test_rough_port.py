"""崎岖地形任务的回归测试。

rough = 冻结的 Flat 基线 + 一层薄覆盖：地形/升降级课程/截断/高度扫描用 mjlab 官方件，
自己的部分只有指令分列覆盖、台阶专项奖励、按列奖励包装、平地热身。本文件把这几条钉住，
默认定价（A15）改动必须同步改这里并在提交信息里写对照实验编号。
"""

from __future__ import annotations

import collections
import math
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
    ROUGH_BASE_HEIGHT_SUPPORT_COLUMNS,
    ROUGH_BASE_HEIGHT_SUPPORT_SENSOR,
    ROUGH_CATASTROPHIC_MIN_BASE_HEIGHT,
    ROUGH_COMMAND_VELOCITY_ERROR_LIN_SCALE,
    ROUGH_CONTACT_TAX_FREE_COLUMNS,
    ROUGH_CRITIC_HEIGHT_SCAN_SENSOR_NAME,
    ROUGH_FALL_PENALTY,
    ROUGH_FLAT_VZ_WEIGHT,
    ROUGH_FLAT_WARMUP_ITERATIONS,
    ROUGH_FLAT_WARMUP_RAMP_ITERATIONS,
    ROUGH_HIGH_STAND_TRANSITION_PROB,
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
    ROUGH_TERRAIN_COMMAND_FLAT_NAMES,
    ROUGH_TERRAIN_EDGE_THRESHOLD_FRACTION,
    ROUGH_TERRAIN_HEIGHT_CLEARANCE,
    ROUGH_TERRAIN_LIN_VEL_X_RANGE,
    ROUGH_TERRAIN_STEP_HEIGHT_TYPE_NAMES,
    ROUGH_TERRAIN_VZ_WEIGHT,
    ROUGH_TRACKING_LIN_VEL_NARROW_SIGMA,
    ROUGH_TRACKING_LIN_VEL_NARROW_STAIR_COLUMNS,
    ROUGH_TRACKING_LIN_VEL_NARROW_STAIR_WEIGHT,
    ROUGH_TRACKING_LIN_VEL_NARROW_WEIGHT,
    ROUGH_TRACKING_LIN_VEL_WEIGHT,
    ROUGH_WHEEL_HEIGHT_DIFF_COLUMNS,
    ROUGH_WHEEL_HEIGHT_DIFF_DEAD_ZONE_M,
    ROUGH_WHEEL_HEIGHT_DIFF_WEIGHT,
    ROUGH_WHEEL_OFFSET_COLUMNS,
    ROUGH_WHEEL_OFFSET_DEAD_ZONE_M,
    ROUGH_WHEEL_OFFSET_WEIGHT,
)
from se3_train.tasks.rough.env_cfg import env_cfg as rough_env_cfg
from se3_train.tasks.rough.terrains import (
    ROUGH_PATCH_SIZE,
    ROUGH_PLATFORM_WIDTH,
    ROUGH_STAIR_LIKE_COLUMNS,
    ROUGH_STEP_HEIGHT_RANGE,
    ROUGH_STEP_WIDTH,
    ROUGH_TERRAIN_PROPORTIONS,
    ROUGH_TWO_STEP_DOWN_COLUMN,
    ROUGH_TWO_STEP_GATE_LEVEL,
    ROUGH_TWO_STEP_SECOND_WIDTH_RANGE,
    ROUGH_TWO_STEP_UP_COLUMN,
    TwoStepStairsTerrainCfg,
)

_ROUGH = "SE3-WheelLegged-Rough"
_ROUGH_GRU = "SE3-WheelLegged-Rough-GRU"
_STAIR_EVAL = "SE3-WheelLegged-Rough-StairEval"
_FLAT_MLP = "SE3-WheelLegged-Flat-MLP"
_WRAPPED = (
    "flat_base_height",
    "tracking_lin_vel",
    "tracking_ang_vel",
    *ROUGH_STAIRS_ZEROED_REWARDS,
)
_REWEIGHTED = ("tracking_lin_vel",)
_ROUGH_ONLY = (
    "command_velocity_error",
    "stair_climb_progress",
    "stair_support_height",
    "fall_penalty",
    "tracking_lin_vel_narrow",  # M15（7d0e6ca）：全程叠加的窄核速度跟踪
    "wheel_fore_aft_offset",  # M18：左右轮前后错位罚
    "wheel_height_diff",  # M23：上台阶列左右轮高度差罚
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

    def test_gru_variant_only_swaps_the_network(self) -> None:
        """M16：GRU 入口除网络外与 MLP 入口逐项相同；rollout 保持 24 步，按轮计数的课程才不会平移。"""
        mlp = load_rl_cfg(_ROUGH)
        gru = load_rl_cfg(_ROUGH_GRU)
        self.assertEqual(mlp.actor.class_name, "MLPModel")
        for model in (gru.actor, gru.critic):
            self.assertEqual(model.class_name, "RNNModel")
            self.assertEqual(model.rnn_type, "gru")
            self.assertEqual(model.rnn_num_layers, 1)
            self.assertEqual(model.rnn_hidden_dim, 512)
        for name in ("actor", "critic"):
            for key in ("hidden_dims", "activation", "obs_normalization", "distribution_cfg"):
                self.assertEqual(
                    getattr(getattr(gru, name), key),
                    getattr(getattr(mlp, name), key),
                    msg=f"{name}.{key}",
                )
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
            self.assertEqual(getattr(gru.algorithm, key), getattr(mlp.algorithm, key), msg=key)
        self.assertEqual(gru.num_steps_per_env, mlp.num_steps_per_env)
        self.assertEqual(gru.num_steps_per_env, 24)
        self.assertEqual(gru.max_iterations, mlp.max_iterations)
        self.assertEqual(gru.save_interval, mlp.save_interval)
        # env 侧不允许有任何差别：同一份 env_cfg()，课程折算步数也与 rollout 一致。
        gru_env = load_env_cfg(_ROUGH_GRU)
        self.assertEqual(
            {k: v.weight for k, v in gru_env.rewards.items()},
            {k: v.weight for k, v in self.cfg.rewards.items()},
        )
        self.assertEqual(
            gru_env.curriculum["flat_warmup"].params["steps_per_policy_iter"], gru.num_steps_per_env
        )
        self.assertEqual(gru_env.commands["velocity_height"], self.cfg.commands["velocity_height"])
        self.assertEqual(
            gru_env.scene.terrain.terrain_generator, self.cfg.scene.terrain.terrain_generator
        )

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
            if name in _REWEIGHTED:
                # M7：只有这一项相对 Flat 改了权重，具体值由 test_a15_pricing_on_non_stair_columns 钉死。
                self.assertNotAlmostEqual(float(term.weight), float(base.weight), msg=name)
                continue
            self.assertAlmostEqual(float(term.weight), float(base.weight), places=12, msg=name)
            if name in _WRAPPED:
                continue
            self.assertIs(term.func, base.func, msg=name)
            self.assertEqual(term.params, base.params, msg=name)

    def test_narrow_kernel_is_weighted_per_column(self) -> None:
        """M15（7d0e6ca）加窄核 w=1；M17（a6c9767）全列 w=3；M20（0c69f77）台阶列置零；
        M22（对照 M21 ee88a1d）台阶列恢复 w=1、其余四列仍 3。改这些数必须同步改这里并在提交信息写对照编号。"""
        narrow = self.cfg.rewards["tracking_lin_vel_narrow"]
        self.assertIs(narrow.func, rough_rewards.column_scaled)
        self.assertIs(narrow.params["inner"], rough_rewards.tracking_lin_vel_narrow)
        self.assertAlmostEqual(float(narrow.weight), ROUGH_TRACKING_LIN_VEL_NARROW_WEIGHT)
        self.assertAlmostEqual(float(narrow.weight), 3.0)
        self.assertEqual(
            narrow.params["params"],
            {"command_name": "velocity_height", "sigma": ROUGH_TRACKING_LIN_VEL_NARROW_SIGMA},
        )
        self.assertAlmostEqual(float(ROUGH_TRACKING_LIN_VEL_NARROW_SIGMA), 0.04)
        self.assertEqual(
            tuple(narrow.params["terrain_type_names"]), ROUGH_TRACKING_LIN_VEL_NARROW_STAIR_COLUMNS
        )
        self.assertEqual(ROUGH_TRACKING_LIN_VEL_NARROW_STAIR_COLUMNS, ROUGH_STAIR_LIKE_COLUMNS)
        # 台阶列的有效权重 = weight × scale，必须正好是 M15 的 1.0（不是 M17 的 3，也不是 M20 的 0）。
        self.assertAlmostEqual(
            float(narrow.weight) * float(narrow.params["scale"]),
            ROUGH_TRACKING_LIN_VEL_NARROW_STAIR_WEIGHT,
        )
        self.assertAlmostEqual(ROUGH_TRACKING_LIN_VEL_NARROW_STAIR_WEIGHT, 1.0)
        self.assertAlmostEqual(
            float(self.cfg.rewards["tracking_lin_vel"].weight), ROUGH_TRACKING_LIN_VEL_WEIGHT
        )

    def test_m23_wheel_height_diff_penalty_is_configured(self) -> None:
        """M23（对照 M22 8e7e3fc）：上台阶列新增左右轮高度差罚，权重 −40/m²、死区 8 cm。

        罚的是 Δz 不是 Δx：爬升段 |Δz| p95 目标流形 ≤4.2 cm、走梯 ≥14 cm，而 |Δx| 两组重叠
        （M15 均值 −10.5 比 M22 的 −7.4 还大）。改这些数必须同步改这里并在提交信息写对照编号。
        """
        term = self.cfg.rewards["wheel_height_diff"]
        self.assertIs(term.func, rough_rewards.wheel_height_diff)
        self.assertAlmostEqual(float(term.weight), -ROUGH_WHEEL_HEIGHT_DIFF_WEIGHT)
        self.assertAlmostEqual(float(term.weight), -40.0)
        self.assertEqual(tuple(term.params["apply_type_names"]), ROUGH_WHEEL_HEIGHT_DIFF_COLUMNS)
        self.assertEqual(ROUGH_WHEEL_HEIGHT_DIFF_COLUMNS, ROUGH_STAIR_LIKE_COLUMNS)
        self.assertAlmostEqual(
            float(term.params["dead_zone_m"]), ROUGH_WHEEL_HEIGHT_DIFF_DEAD_ZONE_M
        )
        self.assertAlmostEqual(float(ROUGH_WHEEL_HEIGHT_DIFF_DEAD_ZONE_M), 0.08)
        # 死区之上的定价：14 cm 罚 0.144/s、18 cm 0.40/s，与台阶列窄核收益（0.125/s）同量级。
        for dz, cost in ((0.04, 0.0), (0.08, 0.0), (0.14, 0.144), (0.18, 0.40)):
            excess = max(abs(dz) - ROUGH_WHEEL_HEIGHT_DIFF_DEAD_ZONE_M, 0.0)
            self.assertAlmostEqual(ROUGH_WHEEL_HEIGHT_DIFF_WEIGHT * excess**2, cost, places=2)
        # 平地的前后错位罚（M18/M19）是另一项，两者互不覆盖。
        self.assertNotEqual(
            tuple(self.cfg.rewards["wheel_fore_aft_offset"].params["apply_type_names"]),
            ROUGH_WHEEL_HEIGHT_DIFF_COLUMNS,
        )

    def test_m19_wheel_offset_penalty_is_configured(self) -> None:
        """M19（对照 M18 ad9dd67）：几何错位罚只在平地列生效、死区 10 cm、w=40/m²。改这些数必须同步改这里并在提交信息写对照编号。"""
        term = self.cfg.rewards["wheel_fore_aft_offset"]
        self.assertIs(term.func, rough_rewards.wheel_fore_aft_offset)
        self.assertAlmostEqual(float(term.weight), -ROUGH_WHEEL_OFFSET_WEIGHT)
        self.assertAlmostEqual(float(term.weight), -40.0)
        self.assertEqual(tuple(term.params["apply_type_names"]), ROUGH_WHEEL_OFFSET_COLUMNS)
        self.assertEqual(ROUGH_WHEEL_OFFSET_COLUMNS, ("flat",))
        self.assertAlmostEqual(float(term.params["dead_zone_m"]), ROUGH_WHEEL_OFFSET_DEAD_ZONE_M)
        self.assertAlmostEqual(ROUGH_WHEEL_OFFSET_DEAD_ZONE_M, 0.10)

    def test_a15_pricing_on_non_stair_columns(self) -> None:
        """默认定价 = A15（h85eljnj）：高度 σ 0.10、运动核 0.5、平地 vz 0、违令罚全六列。"""
        height = self.cfg.rewards["flat_base_height"]
        flat_height = self.flat.rewards["flat_base_height"]
        self.assertIs(height.func, rough_rewards.base_height_penalty_support_on_terrain)
        self.assertAlmostEqual(height.params["sigma"], ROUGH_BASE_HEIGHT_SIGMA)
        self.assertAlmostEqual(float(height.weight), float(flat_height.weight))
        # M3：高度罚全列生效；M21（对照 M20 0c69f77）：上台阶列的地面参考改为轮子支撑面（stair_reward_height），
        # 机身射线传感器、夹紧 ±0.15 与 Flat 相同。改这些必须同步改这里并在提交信息写对照编号。
        self.assertEqual(
            tuple(height.params["terrain_type_names"]), ROUGH_BASE_HEIGHT_SUPPORT_COLUMNS
        )
        self.assertEqual(ROUGH_BASE_HEIGHT_SUPPORT_COLUMNS, ROUGH_STAIR_LIKE_COLUMNS)
        self.assertEqual(height.params["support_sensor_name"], ROUGH_BASE_HEIGHT_SUPPORT_SENSOR)
        self.assertEqual(ROUGH_BASE_HEIGHT_SUPPORT_SENSOR, "stair_reward_height")
        self.assertIn(
            ROUGH_BASE_HEIGHT_SUPPORT_SENSOR, [sensor.name for sensor in self.cfg.scene.sensors]
        )
        self.assertEqual(
            height.params["height_sensor_name"], flat_height.params["height_sensor_name"]
        )
        self.assertEqual(height.params.get("max_error"), flat_height.params.get("max_error"))

        track = self.cfg.rewards["tracking_lin_vel"]
        self.assertIs(track.func, rough_rewards.tracking_lin_vel_terrain_vz)
        # M7：跟踪权重 4 → 6，把运动从每秒亏 15 翻成赚（m6_ledger_full_20260914.md）。
        self.assertAlmostEqual(float(track.weight), ROUGH_TRACKING_LIN_VEL_WEIGHT)
        self.assertGreater(float(track.weight), float(self.flat.rewards["tracking_lin_vel"].weight))
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
                tuple(term.params["terrain_type_names"]), ROUGH_CONTACT_TAX_FREE_COLUMNS, msg=name
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
        # 2026-09-15 起五列，M24（2026-09-21）加 stairs_two_step 成六列。比例为 0 的列不能靠设 0 关掉——mjlab 课程模式仍会生成几何并分 1 个 env，
        # 之前四个死列白占 160 个 geom、每轮多 0.6 s；要么给正比例，要么从字典里删掉。
        self.assertEqual(list(gen.sub_terrains), list(ROUGH_ALL_TERRAIN_TYPE_NAMES))
        self.assertEqual(
            list(gen.sub_terrains),
            [
                "flat",
                "stairs_up",
                ROUGH_TWO_STEP_UP_COLUMN,
                ROUGH_TWO_STEP_DOWN_COLUMN,
                "stairs_down",
                "slope_up",
                "slope_down",
            ],
        )
        self.assertEqual(tuple(gen.size), ROUGH_PATCH_SIZE)
        for name, sub in gen.sub_terrains.items():
            self.assertAlmostEqual(sub.proportion, ROUGH_TERRAIN_PROPORTIONS[name], msg=name)
            self.assertGreater(sub.proportion, 0.0, msg=name)
            self.assertEqual(tuple(sub.size), ROUGH_PATCH_SIZE, msg=name)
        # 上台阶出生在坑底向外爬升（反金字塔）。
        self.assertIsInstance(gen.sub_terrains["stairs_up"], BoxInvertedPyramidStairsTerrainCfg)
        # M24：二级台阶列复刻复旦 sim2sim 那道「20 cm/0.6 m + 15 cm/0.15 m + 下 5 cm」的凸棱。
        two_step = gen.sub_terrains[ROUGH_TWO_STEP_UP_COLUMN]
        self.assertIsInstance(two_step, TwoStepStairsTerrainCfg)
        self.assertEqual(tuple(two_step.step_height_range), (0.05, 0.20))
        self.assertEqual(tuple(two_step.second_step_height_range), (0.04, 0.15))
        self.assertAlmostEqual(float(two_step.step_width), 0.60)
        self.assertEqual(tuple(two_step.second_step_width_range), ROUGH_TWO_STEP_SECOND_WIDTH_RANGE)
        self.assertEqual(ROUGH_TWO_STEP_SECOND_WIDTH_RANGE, (0.60, 0.15))
        self.assertAlmostEqual(float(two_step.outer_drop), 0.05)
        # 难度 1 的等效几何要能把两级分别数成 1 和 2（阶高取第二级，否则第二级白爬）。
        step_h, start, length, count = two_step.se3_stair_geometry(1.0)
        self.assertAlmostEqual(float(step_h), 0.15)
        self.assertAlmostEqual(float(start), 1.0)
        self.assertAlmostEqual(float(length), 0.75)
        self.assertAlmostEqual(float(count), 2.0)
        self.assertEqual(int((0.20 + 0.015) // float(step_h)), 1)
        self.assertEqual(int((0.35 + 0.015) // float(step_h)), 2)
        self.assertFalse(two_step.descending)
        # 下行那列是同一道剖面反着走：先上 5 cm 窄棱、再下 15 cm、再下 20 cm（复旦场景从平台往台阶方向）。
        down = gen.sub_terrains[ROUGH_TWO_STEP_DOWN_COLUMN]
        self.assertIsInstance(down, TwoStepStairsTerrainCfg)
        self.assertTrue(down.descending)
        rings = down.profile(1.0)
        heights = [z for _, z in rings]
        self.assertAlmostEqual(heights[0], 0.30)
        self.assertAlmostEqual(heights[1], 0.35)
        self.assertAlmostEqual(heights[2], 0.20)
        self.assertAlmostEqual(heights[3], 0.00)
        self.assertAlmostEqual(rings[1][0] - rings[0][0], 0.15)  # 窄棱在内
        self.assertAlmostEqual(rings[2][0] - rings[1][0], 0.60)
        # 上行剖面：出生 −0.30，向外 +0.20（踏面 0.60）、+0.15（踏面 0.15）、−0.05 回到 0。
        up_rings = two_step.profile(1.0)
        self.assertAlmostEqual(up_rings[0][1], -0.30)
        self.assertAlmostEqual(up_rings[1][1], -0.10)
        self.assertAlmostEqual(up_rings[2][1], 0.05)
        self.assertAlmostEqual(up_rings[3][1], 0.00)
        self.assertEqual(
            tuple(gen.sub_terrains["stairs_up"].step_height_range), ROUGH_STEP_HEIGHT_RANGE
        )
        self.assertGreaterEqual(self.cfg.sim.contact_sensor_maxmatch, 500)

    def test_sim_pool_sizes_follow_overflow_measurement(self) -> None:
        """池容量按**真实训练日志**里的 nefc overflow 定，不是按压测峰值。

        2026-09-15 的教训：两列地形时压测（4096 env × 2000 步）峰值 54/10，据此定的 256/64
        在五列地形下一直溢出——M9/M10/M11 的 train.log 里 "increase njmax to N" 刷了
        3900/32433/10892 次，N 实测 294–402，M11 从 754 轮（热身一结束踩上五列）就开始。
        压测规模和时长都远小于训练，只能证伪不能证明够用。
        """
        # 五列地形下训练日志实测的约束需求上界（M9–M11）与压测的接触峰值。
        measured_nefc_peak, measured_ncon_peak = 402 / 3, 10
        self.assertEqual(self.cfg.sim.njmax, ROUGH_NJMAX)
        self.assertEqual(self.cfg.sim.nconmax, ROUGH_NCONMAX)
        self.assertGreaterEqual(ROUGH_NJMAX, 3 * measured_nefc_peak)  # ≥ 402，训练实测上界
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

    def test_catastrophic_height_floor_clears_every_descending_column(self) -> None:
        """M10：下行地形的出生点在顶部平台，正常往下走不能被 catastrophic_state 判成物理发散。

        M9 就栽在这：Flat 的 −0.5 m 下限让 stairs_down / slope_down 一走下去就终止，
        1697 轮时 catastrophic 2.63/轮、回报从 34 掉到 8，而课程照升（升级只看水平位移）。
        """
        term = self.cfg.terminations["catastrophic_state"]
        floor = float(term.params["min_base_height"])
        self.assertAlmostEqual(floor, ROUGH_CATASTROPHIC_MIN_BASE_HEIGHT)
        self.assertLess(floor, -0.5)  # 必须比 Flat 基线更深

        gen = self.cfg.scene.terrain.terrain_generator
        assert gen is not None
        half = (min(ROUGH_PATCH_SIZE) - 2 * 0.5 - ROUGH_PLATFORM_WIDTH) / 2
        n_steps = int(half / ROUGH_STEP_WIDTH)

        def extent(sub) -> float:
            step_range = getattr(sub, "step_height_range", None)
            if step_range is not None:
                return float(step_range[1]) * n_steps
            slope_range = getattr(sub, "slope_range", None)
            if slope_range is not None:
                return float(slope_range[1]) * half
            return 0.0

        deepest = max(
            (extent(sub) for name, sub in gen.sub_terrains.items() if "down" in name), default=0.0
        )
        highest = max(
            (extent(sub) for name, sub in gen.sub_terrains.items() if "up" in name), default=0.0
        )
        self.assertGreater(deepest, 0.0, "没有下行列？此用例要跟着地形一起更新")
        self.assertLess(deepest, abs(floor), f"最深下行 {deepest:.2f} m 超过下限 {floor} m")
        self.assertLess(highest, float(term.params["max_base_height"]))

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

    def test_high_stand_transition_is_enabled_on_flat(self) -> None:
        """M8：平地列注入高姿起步转移，参数沿用 A20/A21 验证过的那组。"""
        cmd = self.cfg.commands["velocity_height"]
        self.assertAlmostEqual(cmd.high_stand_transition_prob, ROUGH_HIGH_STAND_TRANSITION_PROB)
        self.assertGreater(cmd.high_stand_transition_prob, 0.0)
        # 静站高度要落在死锁区（>0.34），否则练不到要练的那个状态。
        self.assertGreaterEqual(cmd.high_stand_height_range[0], 0.34)
        self.assertLessEqual(cmd.high_stand_height_range[1], cmd.height_range[1])
        # 切换后的速度要越过跟踪核的零梯度段，下界不能太小。
        self.assertGreaterEqual(cmd.high_stand_move_vx_range[0], 0.8)
        self.assertGreater(cmd.high_stand_duration_range_s[0], 0.0)

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
        # M8 起打开（原为 0.0 关闭）；具体值与参数由 test_high_stand_transition_is_enabled_on_flat 钉死。
        self.assertAlmostEqual(cmd.high_stand_transition_prob, ROUGH_HIGH_STAND_TRANSITION_PROB)
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


class RoughRuntimeTests(unittest.TestCase):
    """在 CPU 上建一个小环境，验证分列指令、分列奖励与官方课程/截断在本机器人上真的生效。"""

    @classmethod
    def setUpClass(cls) -> None:
        cfg = rough_env_cfg()
        # 热身会把首次 reset 的全部 env 放到平地列；这里要各列都分到 env。
        cfg.curriculum.pop("flat_warmup")
        # 二级台阶门控在 stairs_up 均级到 5 之前会清空那两列，本类要验证各列的奖励行为，所以也关掉；
        # 门控本身由 TwoStepGateRuntimeTests 钉住。
        cfg.curriculum.pop("two_step_gate")
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
        # M24 起台阶类开关作用于 ROUGH_STAIR_LIKE_COLUMNS（stairs_up + stairs_two_step），
        # 断言"台阶列如何如何"一律用这个并集；cls.stairs 只在需要单独指名 stairs_up 时用。
        stair_ids = [cls.names.index(n) for n in ROUGH_STAIR_LIKE_COLUMNS]
        cls.stair_like = torch.zeros_like(
            cls.flat if hasattr(cls, "flat") else cls.types, dtype=torch.bool
        )
        cls.stair_like = torch.isin(cls.types, torch.tensor(stair_ids, dtype=cls.types.dtype))
        cls.flat = cls.types == cls.flat_col

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_every_column_is_populated_and_masks_agree(self) -> None:
        self.assertEqual(sorted(set(self.types.tolist())), list(range(len(self.names))))
        self.assertEqual(len(self.names), 7)
        self.assertTrue(torch.equal(column_mask(self.env, ("stairs_up",)), self.stairs))
        self.assertTrue(
            torch.equal(column_mask(self.env, ROUGH_STAIR_LIKE_COLUMNS), self.stair_like)
        )
        self.assertTrue(torch.equal(non_flat_column_mask(self.env), ~self.flat))
        self.assertIsNone(column_mask(self.env, ("no_such_column",)))
        # M9：指令侧的"非平地覆盖"只剩 stairs_up，其余列按平地方式发指令（±2.4 + yaw）。
        flat_like = column_mask(self.env, ROUGH_TERRAIN_COMMAND_FLAT_NAMES)
        self.assertTrue(torch.equal(self.term._terrain_override_mask, ~flat_like))
        self.assertTrue(torch.equal(self.term._terrain_override_mask, self.stair_like))
        self.assertTrue(torch.equal(self.term._stair_mask, self.stair_like))
        mask = getattr(self.env, events.CURRICULUM_ENV_MASK_ATTR)
        self.assertTrue(torch.equal(mask, self.flat))

    def test_non_flat_columns_only_get_forward_commands(self) -> None:
        # M9：只有上台阶类列走非平地覆盖（再被台阶覆盖压一层）；M14（751eab3）起台阶覆盖为
        # vx 0.4–2.4 前向、yaw ±0.3（不再恒 0）；M24 起这类列有 stairs_up 与 stairs_two_step 两条。
        # 下台阶与上下坡按平地方式发指令，由 test_new_columns_get_flat_style_commands 钉住。
        self.assertTrue(torch.equal(self.term._terrain_override_mask, self.stair_like))
        yaw_lo, yaw_hi = ROUGH_STAIR_ANG_VEL_YAW_RANGE
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
            self.assertFalse(bool(self.term._standing_mask[self.stairs].any()))
        # Flat 速度课程起点 vx=yaw=0，平地列 reset 后指令必须仍是 0。
        self.assertTrue(bool((self.term.command[self.flat][:, :2].abs() < 1e-6).all()))

    def test_new_columns_get_flat_style_commands(self) -> None:
        """M9（2026-09-15 用户定）：下台阶与上下坡按平地方式发指令——速度跟平地课程（终值 ±2.4）、有偏航跟踪。"""
        names = list(self.names)
        for col in ("stairs_down", "slope_up", "slope_down"):
            self.assertIn(col, names, msg=col)
            self.assertIn(col, ROUGH_TERRAIN_COMMAND_FLAT_NAMES, msg=col)
        # 这三列不能落进"非平地覆盖"（那条路是 0.4–0.8 前向 + yaw ±0.2）。
        newmask = column_mask(self.env, ("stairs_down", "slope_up", "slope_down"))
        self.assertFalse(bool((self.term._terrain_override_mask & newmask).any()))
        # 偏航跟踪奖励只在 stairs_up 置零，新列保留。
        self.assertNotIn("stairs_down", ROUGH_REWARD_TERRAIN_TYPE_NAMES)
        # 速度课程终值必须能到 ±2.4。
        params = dict(self.env.cfg.curriculum["command_vel"].params or {})
        self.assertAlmostEqual(float(params["max_lin_vel_x"]), 2.4)
        # 接触税在上下台阶都免，坡面不免。
        self.assertEqual(
            ROUGH_CONTACT_TAX_FREE_COLUMNS,
            (*ROUGH_STAIR_LIKE_COLUMNS, "stairs_down", ROUGH_TWO_STEP_DOWN_COLUMN),
        )
        # M24：二级台阶的下行列按平地待遇——指令走平地那一路，不进台阶类列。
        self.assertIn(ROUGH_TWO_STEP_DOWN_COLUMN, ROUGH_TERRAIN_COMMAND_FLAT_NAMES)
        self.assertNotIn(ROUGH_TWO_STEP_DOWN_COLUMN, ROUGH_STAIR_LIKE_COLUMNS)
        self.assertNotIn(ROUGH_TWO_STEP_DOWN_COLUMN, ROUGH_REWARD_TERRAIN_TYPE_NAMES)
        self.assertIn(ROUGH_TWO_STEP_UP_COLUMN, ROUGH_STAIR_LIKE_COLUMNS)

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
        """包装函数点名列时台阶列不吃高度罚；配置里高度罚全列生效（M3 加回、M21 台阶列改支撑面参考）；
        yaw 工资台阶列为 0；违令罚全列（A15）。"""
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
        # 这里直接调包装函数，用的是它的默认列名 ("stairs_up",)，不是配置里的 ROUGH_STAIR_LIKE_COLUMNS。
        self.assertEqual(float(height_pen[self.stairs].abs().max()), 0.0)
        self.assertGreater(float(height_pen[~self.stairs].min()), 0.0)
        manager = self.env.reward_manager
        # M3：配置里的 flat_base_height 在台阶列真的在扣（reset 后高度指令与实际高度不一致）。
        h_index = manager.active_terms.index("flat_base_height")
        self.assertLess(float(manager._step_reward[self.stair_like, h_index].min()), 0.0)
        self.assertLess(float(manager._step_reward[self.flat, h_index].min()), 0.0)
        index = manager.active_terms.index("tracking_ang_vel")
        self.assertEqual(float(manager._step_reward[self.stair_like, index].abs().max()), 0.0)
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

    def test_wheel_height_diff_is_geometric_stairs_only_with_dead_zone(self) -> None:
        """M23 运行时：罚值 = max(|左轮z − 右轮z| − 0.08, 0)²·门控，只在台阶列，其余列恒 0；死区内恰为 0。"""
        from se3_train.mdp.rewards import _DEFAULT_ASSET_CFG, _tracking_upright_gate

        _step_once(self.env)
        robot = self.env.scene[_DEFAULT_ASSET_CFG.name]
        wheel_ids, _ = robot.find_bodies(("l_wheel_Link", "r_wheel_Link"), preserve_order=True)
        dz = (
            robot.data.body_link_pos_w[:, wheel_ids[0], 2]
            - robot.data.body_link_pos_w[:, wheel_ids[1], 2]
        )
        gate = _tracking_upright_gate(robot.data.projected_gravity_b[:, 2], 0.7)
        # 死区设 0 时应逐位等于 |Δz|²·门控，并且只在台阶列非零。
        bare = rough_rewards.wheel_height_diff(
            self.env, apply_type_names=ROUGH_WHEEL_HEIGHT_DIFF_COLUMNS, dead_zone_m=0.0
        )
        expected = dz.square() * gate
        self.assertTrue(torch.allclose(bare[self.stair_like], expected[self.stair_like], atol=1e-6))
        self.assertEqual(float(bare[~self.stair_like].abs().max()), 0.0)
        # 默认死区下同样逐位对得上（reset 后落地未稳，台阶列 |Δz| 可能有几厘米，不能假设它一定在死区内）。
        got = rough_rewards.wheel_height_diff(
            self.env,
            apply_type_names=ROUGH_WHEEL_HEIGHT_DIFF_COLUMNS,
            dead_zone_m=ROUGH_WHEEL_HEIGHT_DIFF_DEAD_ZONE_M,
        )
        excess = torch.clamp(dz.abs() - ROUGH_WHEEL_HEIGHT_DIFF_DEAD_ZONE_M, min=0.0)
        self.assertTrue(
            torch.allclose(
                got[self.stair_like], (excess.square() * gate)[self.stair_like], atol=1e-6
            )
        )
        self.assertEqual(float(got[~self.stair_like].abs().max()), 0.0)
        self.assertLessEqual(float(got.max()), float(bare.max()))
        # 死区语义：死区大到盖住所有 |Δz| 时必须恰好为 0，而不是很小的正数。
        wide = rough_rewards.wheel_height_diff(
            self.env, apply_type_names=ROUGH_WHEEL_HEIGHT_DIFF_COLUMNS, dead_zone_m=1.0
        )
        self.assertEqual(float(wide.abs().max()), 0.0)
        # 列名对不上（plane 口径）时恒 0：这是台阶专项，不该凭空全局生效。
        none_mask = rough_rewards.wheel_height_diff(
            self.env, apply_type_names=("no_such_column",), dead_zone_m=0.0
        )
        self.assertEqual(float(none_mask.abs().max()), 0.0)
        manager = self.env.reward_manager
        value = manager._step_reward[:, manager.active_terms.index("wheel_height_diff")]
        self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertLessEqual(float(value.max()), 0.0)
        self.assertEqual(float(value[~self.stair_like].abs().max()), 0.0)
        log = self.env.extras.get("log", {})
        self.assertIn("Rough/wheel_dz_abs", log)
        self.assertIn("Rough/wheel_dz_abs_stairs", log)

    def test_wheel_offset_penalty_is_geometric_flat_only_with_dead_zone(self) -> None:
        """M19 运行时：平地列罚值 = −40·max(|Δx|−0.10, 0)²·门控（Δx 为机身系左右轮心 x 差），其余四列恒 0。"""
        from se3_train.mdp.rewards import (
            _DEFAULT_ASSET_CFG,
            _tracking_upright_gate,
            _wheel_pos_body_frame,
        )

        _step_once(self.env)
        manager = self.env.reward_manager
        idx = manager.active_terms.index("wheel_fore_aft_offset")
        value = manager._step_reward[:, idx]
        self.assertEqual(float(value[~self.flat].abs().max()), 0.0)
        wheel_b = _wheel_pos_body_frame(self.env, _DEFAULT_ASSET_CFG)
        dx = wheel_b[:, 0, 0] - wheel_b[:, 1, 0]
        gate = _tracking_upright_gate(self.env.scene["robot"].data.projected_gravity_b[:, 2], 0.7)
        excess = torch.clamp(dx.abs() - ROUGH_WHEEL_OFFSET_DEAD_ZONE_M, min=0.0)
        # 数值口径用纯函数逐位比对；_step_reward 比这里读到的状态晚一个物理子步，逐位比对不可靠，
        # 只查符号与有限性（与 M21/M22/M23 三处同样处理）。
        got = rough_rewards.wheel_fore_aft_offset(
            self.env,
            apply_type_names=ROUGH_WHEEL_OFFSET_COLUMNS,
            dead_zone_m=ROUGH_WHEEL_OFFSET_DEAD_ZONE_M,
        )
        self.assertTrue(
            torch.allclose(got[self.flat], (excess.square() * gate)[self.flat], atol=1e-6)
        )
        self.assertEqual(float(got[~self.flat].abs().max()), 0.0)
        self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertLessEqual(float(value.max()), 0.0)
        # 死区内的 env 罚值必须恰好为 0，而不是很小的负数。
        inside = self.flat & (dx.abs() < ROUGH_WHEEL_OFFSET_DEAD_ZONE_M - 0.01)
        if bool(inside.any()):
            self.assertEqual(float(value[inside].abs().max()), 0.0)
        log = self.env.extras.get("log", {})
        self.assertIn("Rough/wheel_dx_abs", log)
        self.assertIn("Rough/wheel_dx_abs_flat", log)

    def test_m21_stairs_height_reference_is_wheel_support_surface(self) -> None:
        """M21 运行时：台阶列高度 = 机身射线 frame_z − 两轮下方射线的地面均值，其余列与 Flat 原函数逐位相同。

        把台阶列 env 平移到"机身与左轮在第一级踏面、右轮仍在中心平台"并悬空一步：机身射线参考抬一整阶，
        支撑面参考只抬半阶，两种口径相差半阶，罚值必须跟着支撑面走。
        """
        from se3_train.mdp.rewards import _recovery_reset_mask
        from se3_train.mdp.terrain_height import frame_height_above_terrain, ground_height_estimate
        from se3_train.tasks.flat.rewards import flat_base_height_penalty_no_jump
        from se3_train.tasks.rough.stair_rewards import _geometry

        env = self.env
        robot = env.scene["robot"]
        try:
            _step_once(env)
            _, step_h, start, _, _ = _geometry(env, ("stairs_up",))
            ids = torch.nonzero(self.stairs).flatten()
            pose = robot.data.root_link_pose_w.clone()
            origins = env.scene.env_origins
            pose[ids, 0] = origins[ids, 0]
            # 机身在 y = 平台边界 + 0.10（射线环半径 0.05 全在第一级踏面），左轮 +0.217 也在踏面，右轮 −0.217 仍在平台。
            pose[ids, 1] = origins[ids, 1] + start[ids] + 0.10
            pose[ids, 2] = pose[ids, 2] + 0.25  # 悬空一步，左轮不穿进踏面
            pose[ids, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=pose.device)
            robot.write_root_link_pose_to_sim(pose)
            env.sim.forward()
            _step_once(env)

            cmd = env.command_manager.get_command("velocity_height")
            active = (~(cmd[:, 5] > 0.5)) & (~_recovery_reset_mask(env))
            ground_ray = ground_height_estimate(env, "base_height_sensor")
            ground_support = ground_height_estimate(env, ROUGH_BASE_HEIGHT_SUPPORT_SENSOR)
            gap = ground_ray - ground_support
            # 只有 stairs_up 的 env 被平移过（平移量按金字塔台阶的几何算），所以差半阶只对它们成立。
            self.assertGreater(float(step_h[self.stairs].min()), 0.01)
            self.assertTrue(
                torch.allclose(gap[self.stairs], 0.5 * step_h[self.stairs], atol=2e-3), gap
            )
            # 平地列两种参考逐位相同。
            self.assertLess(float(gap[self.flat].abs().max()), 1e-4)

            sigma = ROUGH_BASE_HEIGHT_SIGMA
            kwargs = dict(
                command_name="velocity_height",
                height_sensor_name="base_height_sensor",
                sigma=sigma,
                max_error=None,
            )
            got = rough_rewards.base_height_penalty_support_on_terrain(
                env,
                support_sensor_name=ROUGH_BASE_HEIGHT_SUPPORT_SENSOR,
                terrain_type_names=ROUGH_BASE_HEIGHT_SUPPORT_COLUMNS,
                **kwargs,
            )
            flat_pen = flat_base_height_penalty_no_jump(env, **kwargs)
            frame_z = env.scene["base_height_sensor"].data.frame_pos_w[:, 0, 2]
            err_support = frame_z - ground_support - cmd[:, 4]
            err_ray = frame_height_above_terrain(env, "base_height_sensor") - cmd[:, 4]
            expected = torch.where(
                self.stair_like, err_support.square() / sigma**2 * active.float(), flat_pen
            )
            self.assertTrue(torch.allclose(got, expected, atol=1e-5))
            self.assertTrue(torch.equal(got[~self.stair_like], flat_pen[~self.stair_like]))
            # 台阶列的值跟着支撑面而不是机身射线走。
            ray_pen = err_ray.square() / sigma**2 * active.float()
            check = self.stairs & active
            self.assertTrue(bool(check.any()))
            self.assertGreater(float((got - ray_pen)[check].abs().min()), 0.05)

            # 配置里挂的就是这一项：台阶列有效 env 在扣钱，平地列也在扣（数值口径由上面的纯函数比对钉住；
            # _step_reward 比这里读到的状态晚一个物理子步，逐位比对不可靠，所以只查符号与有限性）。
            manager = env.reward_manager
            h_index = manager.active_terms.index("flat_base_height")
            value = manager._step_reward[:, h_index]
            self.assertTrue(bool(torch.isfinite(value).all()))
            self.assertLess(float(value[check].max()), 0.0)
            self.assertLessEqual(float(value.max()), 0.0)
            self.assertLess(float(env.cfg.rewards["flat_base_height"].weight), 0.0)
            log = env.extras.get("log", {})
            self.assertIn("Rough/base_height_err_support_stairs", log)
            self.assertIn("Rough/base_height_err_ray_stairs", log)
            # 悬空姿态下支撑面参考比机身射线参考低（左轮在踏面、右轮还在平台，取均值只抬半阶），
            # 所以支撑面口径的 |误差| 更大。M24 起这两个键对 stairs_up + stairs_two_step 一起求均值，
            # 而本用例只平移了 stairs_up 的 env，差值不再等于半阶，只断言方向；数值由上面的纯函数比对钉住。
            log_gap = float(log["Rough/base_height_err_support_stairs"]) - float(
                log["Rough/base_height_err_ray_stairs"]
            )
            self.assertGreater(log_gap, 0.0)
        finally:
            env.reset()

    def test_narrow_kernel_is_scaled_down_on_stairs(self) -> None:
        """M22 运行时：窄核在台阶列是其余四列的 1/3（有效 w=1），scale=0 时退化为 M20 的置零。

        把速度指令设成每个 env 当前的实际速度（误差 0 → 核值 = 直立门控），这样核值不依赖 reset 的随机指令；
        σ=0.04 很窄，误差 0.6 m/s 时核值就下溢到 0，直接拿随机指令比对会 flaky。
        """
        _step_once(self.env)
        robot = self.env.scene["robot"]
        cmd = self.env.command_manager.get_command("velocity_height")
        saved = cmd.clone()
        inner_kwargs = {
            "command_name": "velocity_height",
            "sigma": ROUGH_TRACKING_LIN_VEL_NARROW_SIGMA,
        }
        scale = ROUGH_TRACKING_LIN_VEL_NARROW_STAIR_WEIGHT / ROUGH_TRACKING_LIN_VEL_NARROW_WEIGHT
        try:
            cmd[:, 0] = robot.data.root_link_lin_vel_b[:, 0]
            raw = rough_rewards.tracking_lin_vel_narrow(self.env, **inner_kwargs)
            got = rough_rewards.column_scaled(
                self.env,
                inner=rough_rewards.tracking_lin_vel_narrow,
                params=inner_kwargs,
                terrain_type_names=ROUGH_TRACKING_LIN_VEL_NARROW_STAIR_COLUMNS,
                scale=scale,
            )
            zeroed = rough_rewards.column_scaled(
                self.env,
                inner=rough_rewards.tracking_lin_vel_narrow,
                params=inner_kwargs,
                terrain_type_names=ROUGH_TRACKING_LIN_VEL_NARROW_STAIR_COLUMNS,
                scale=0.0,
            )
        finally:
            cmd[:] = saved
        # 误差为 0 时核值就是门控值，reset 后全员直立所以接近 1。
        self.assertGreater(float(raw.min()), 0.5)
        self.assertTrue(
            torch.allclose(got[self.stair_like], raw[self.stair_like] * scale, atol=1e-6)
        )
        self.assertTrue(torch.equal(got[~self.stair_like], raw[~self.stair_like]))
        self.assertAlmostEqual(
            float(got[self.stair_like].mean() / raw[self.stair_like].mean()), 1.0 / 3.0, places=5
        )
        self.assertEqual(float(zeroed[self.stair_like].abs().max()), 0.0)
        self.assertTrue(torch.equal(zeroed[~self.stair_like], raw[~self.stair_like]))
        # 这一项确实挂在奖励表里且是奖励不是罚（分列权重由 test_narrow_kernel_is_weighted_per_column 钉住；
        # _step_reward 比这里读到的状态晚一个物理子步，逐位比对不可靠，所以只查符号与有限性）。
        manager = self.env.reward_manager
        idx = manager.active_terms.index("tracking_lin_vel_narrow")
        value = manager._step_reward[:, idx]
        self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertGreaterEqual(float(value.min()), 0.0)

    def test_stairs_column_pays_no_wage_and_no_climbing_tax(self) -> None:
        """M2 运行时：台阶列上三项为 0；平地列 is_alive 照发；摔倒罚项已挂上。"""
        _step_once(self.env)
        manager = self.env.reward_manager
        for name in ROUGH_STAIRS_ZEROED_REWARDS:
            idx = manager.active_terms.index(name)
            self.assertEqual(
                float(manager._step_reward[self.stair_like, idx].abs().max()), 0.0, msg=name
            )
        alive = manager._step_reward[self.flat, manager.active_terms.index("is_alive")]
        self.assertGreater(float(alive.min()), 0.0)
        self.assertIn("fall_penalty", manager.active_terms)


class TwoStepGateRuntimeTests(unittest.TestCase):
    """M24 门控：stairs_up 均级到 5 之前二级台阶两列没有 env，达标后一次性放开且不再回收。"""

    @classmethod
    def setUpClass(cls) -> None:
        cfg = rough_env_cfg()
        cfg.curriculum.pop("flat_warmup")  # 热身会把所有 env 压到平地列，掩盖门控效果
        cfg.scene.num_envs = 14
        cls.env = ManagerBasedRlEnv(cfg, device="cpu")
        cls.env.reset()
        terrain = cls.env.scene.terrain
        assert terrain is not None
        cls.terrain = terrain
        cls.names = list(terrain.cfg.terrain_generator.sub_terrains.keys())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def _run(self) -> dict:
        return curriculums.two_step_gate(
            self.env,
            _all_ids(self.env),
            command_name="velocity_height",
            gate_terrain_name="stairs_up",
            gate_level=ROUGH_TWO_STEP_GATE_LEVEL,
            gated_columns=(
                (ROUGH_TWO_STEP_UP_COLUMN, "stairs_up"),
                (ROUGH_TWO_STEP_DOWN_COLUMN, "flat"),
            ),
        )

    def test_gate_holds_then_opens_once(self) -> None:
        up_col = self.names.index(ROUGH_TWO_STEP_UP_COLUMN)
        down_col = self.names.index(ROUGH_TWO_STEP_DOWN_COLUMN)
        stairs_col = self.names.index("stairs_up")
        flat_col = self.names.index("flat")
        original = getattr(self.env, curriculums.TWO_STEP_GATE_ORIGINAL_TYPES_ATTR, None)
        if original is None:
            original = self.terrain.terrain_types.clone()
        self.assertIn(up_col, original.tolist())
        self.assertIn(down_col, original.tolist())

        # 未达标：两列清空，上行的 env 去 stairs_up、下行的去 flat。
        self.terrain.terrain_levels[:] = 0
        out = self._run()
        self.assertEqual(float(out["opened"]), 0.0)
        self.assertEqual(float(out["gate_level"]), 0.0)
        types = self.terrain.terrain_types
        self.assertEqual(int((types == up_col).sum()), 0)
        self.assertEqual(int((types == down_col).sum()), 0)
        self.assertTrue(bool((types[original == up_col] == stairs_col).all()))
        self.assertTrue(bool((types[original == down_col] == flat_col).all()))

        # 热身期原生 stairs_up env 在平地列上被官方升降级抬到高等级：不算台阶难度，门控不能开（M24 的 bug）。
        native = original == stairs_col
        self.terrain.terrain_types[native] = flat_col
        self.terrain.terrain_levels[native] = 6
        out = self._run()
        self.assertEqual(float(out["opened"]), 0.0)
        self.terrain.terrain_types[native] = stairs_col
        self.terrain.terrain_levels[:] = 0

        # stairs_up 列当前的 env（原生 + 门控期迁入的二级上行）均级到 5：门控打开，
        # 两列拿回自己的 env 且从第 0 行起步。
        self.terrain.terrain_levels[self.terrain.terrain_types == stairs_col] = 5
        out = self._run()
        self.assertEqual(float(out["opened"]), 1.0)
        self.assertGreaterEqual(float(out["gate_level"]), ROUGH_TWO_STEP_GATE_LEVEL)
        types = self.terrain.terrain_types
        self.assertTrue(bool((types[original == up_col] == up_col).all()))
        self.assertTrue(bool((types[original == down_col] == down_col).all()))
        self.assertTrue(bool((self.terrain.terrain_levels[original == up_col] == 0).all()))

        # 等级掉回去也不回收（一次性放开）。
        self.terrain.terrain_levels[:] = 0
        out = self._run()
        self.assertEqual(float(out["opened"]), 1.0)
        types = self.terrain.terrain_types
        self.assertTrue(bool((types[original == up_col] == up_col).all()))
        self.assertTrue(bool((types[original == down_col] == down_col).all()))

    def test_gate_is_registered_after_flat_warmup(self) -> None:
        """门控必须排在热身之后：热身期全体在平地列，先跑门控会把还在热身的 env 提前拽到 stairs_up。"""
        order = list(rough_env_cfg().curriculum)
        self.assertIn("two_step_gate", order)
        self.assertGreater(order.index("two_step_gate"), order.index("flat_warmup"))
        self.assertGreater(order.index("flat_warmup"), order.index("terrain_levels"))


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
        # M9：override 只覆盖上台阶类列，不再是"所有非平地列"；M24 起是 stairs_up + stairs_two_step。
        names = list(self.terrain.cfg.terrain_generator.sub_terrains)
        stair_cols = torch.tensor(
            [names.index(n) for n in ROUGH_STAIR_LIKE_COLUMNS], dtype=original.dtype
        )
        self.assertTrue(torch.equal(term._terrain_override_mask, torch.isin(original, stair_cols)))
        self.assertTrue(
            torch.equal(
                getattr(self.env, events.CURRICULUM_ENV_MASK_ATTR), original == self.flat_col
            )
        )
        self.assertTrue(bool(getattr(self.env, curriculums.FLAT_WARMUP_DONE_ATTR).all()))


class StairSpeedCapRuntimeTests(unittest.TestCase):
    """台阶列逐 env 速度上限：按 episode 速度达成率升降、夹在区间内，并写进台阶列的采样上界。"""

    @classmethod
    def setUpClass(cls) -> None:
        cfg = rough_env_cfg(stair_speed_cap=True)
        cfg.curriculum.pop("flat_warmup")  # 热身会把所有 env 压到平地列，台阶列就没有 env
        cfg.scene.num_envs = 14
        cls.env = ManagerBasedRlEnv(cfg, device="cpu")
        cls.env.reset()
        cls.term = cls.env.command_manager.get_term("velocity_height")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_registered_only_when_enabled(self) -> None:
        self.assertNotIn("stair_speed_cap", rough_env_cfg().curriculum)
        self.assertFalse(rough_env_cfg().commands["velocity_height"].stair_speed_cap_enabled)
        order = list(rough_env_cfg(stair_speed_cap=True).curriculum)
        self.assertGreater(order.index("stair_speed_cap"), order.index("two_step_gate"))
        play = rough_env_cfg(play=True, stair_speed_cap=True)
        self.assertFalse(play.commands["velocity_height"].stair_speed_cap_enabled)

    def _fill(self, ids: torch.Tensor, ratio: float, steps: int) -> None:
        self.term._stair_cmd_sum[ids] = float(steps)
        self.term._stair_vx_sum[ids] = float(ratio) * steps
        self.term._stair_steps[ids] = steps

    def test_caps_follow_speed_ratio_and_stay_in_range(self) -> None:
        term, cfg = self.term, self.term.cfg
        stairs = term._stair_mask.nonzero(as_tuple=False).flatten()
        self.assertGreaterEqual(len(stairs), 3)
        slow, fast, short = stairs[0:1], stairs[1:2], stairs[2:3]
        min_steps = math.ceil(cfg.stair_speed_cap_min_episode_s / self.env.step_dt)
        term._stair_speed_cap[:] = 1.5
        self._fill(slow, 0.2, min_steps)
        self._fill(fast, 0.9, min_steps)
        self._fill(short, 0.2, min_steps - 1)  # 在台阶列上待得太短，不调
        ids = torch.cat((slow, fast, short))
        curriculums.stair_speed_cap(self.env, ids, command_name="velocity_height")

        cap = term._stair_speed_cap
        self.assertAlmostEqual(float(cap[slow]), 1.5 - cfg.stair_speed_cap_shrink_step, places=5)
        self.assertAlmostEqual(float(cap[fast]), 1.5 + cfg.stair_speed_cap_grow_step, places=5)
        self.assertAlmostEqual(float(cap[short]), 1.5, places=5)
        self.assertEqual(int(term._stair_steps[ids].sum()), 0)
        ranges = term._lin_vel_x_range_override
        self.assertTrue(torch.allclose(ranges[ids, 1], cap[ids]))
        self.assertTrue(bool((ranges[ids, 0] == cfg.stair_lin_vel_x_range[0]).all()))

        for _ in range(20):
            self._fill(slow, 0.0, min_steps)
            self._fill(fast, 1.0, min_steps)
            curriculums.stair_speed_cap(self.env, torch.cat((slow, fast)), "velocity_height")
        self.assertAlmostEqual(float(cap[slow]), cfg.stair_speed_cap_min, places=5)
        self.assertAlmostEqual(float(cap[fast]), cfg.stair_lin_vel_x_range[1], places=5)

    def test_refresh_reapplies_caps_only_on_stairs(self) -> None:
        term = self.term
        term._stair_speed_cap[:] = 1.2
        term.refresh_terrain_override()
        stairs = term._stair_mask
        ids = torch.arange(self.env.num_envs, device=self.env.device)
        _, lin_high, _, _ = term._velocity_ranges(ids)
        self.assertTrue(torch.allclose(lin_high[stairs], torch.full_like(lin_high[stairs], 1.2)))
        self.assertTrue(bool((lin_high[~stairs] != 1.2).all()))


class StairHeightWindowRuntimeTests(unittest.TestCase):
    """复旦口径的上台阶列高度罚：窗口均值参考 + 有界形状；其余列与 Flat 原函数逐位相同。"""

    @classmethod
    def setUpClass(cls) -> None:
        cfg = rough_env_cfg(stair_height_reference="window")
        cfg.curriculum.pop("flat_warmup")
        cfg.scene.num_envs = 14
        cls.env = ManagerBasedRlEnv(cfg, device="cpu")
        cls.env.reset()
        _step_once(cls.env)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_default_stays_support(self) -> None:
        default = rough_env_cfg().rewards["flat_base_height"]
        self.assertIs(default.func, rough_rewards.base_height_penalty_support_on_terrain)
        self.assertIs(
            self.env.cfg.rewards["flat_base_height"].func,
            rough_rewards.base_height_penalty_window_on_terrain,
        )
        with self.assertRaises(ValueError):
            rough_env_cfg(stair_height_reference="ray")

    def test_window_is_bounded_and_other_columns_match_flat(self) -> None:
        from se3_train.mdp.terrain_height import ground_height_estimate
        from se3_train.tasks.flat.rewards import flat_base_height_penalty_no_jump

        params = dict(self.env.cfg.rewards["flat_base_height"].params)
        got = rough_rewards.base_height_penalty_window_on_terrain(self.env, **params)
        flat = flat_base_height_penalty_no_jump(
            self.env,
            command_name=params["command_name"],
            height_sensor_name=params["height_sensor_name"],
            sigma=params["sigma"],
            max_error=params.get("max_error", 0.15),
        )
        stairs = column_mask(self.env, params["terrain_type_names"])
        self.assertTrue(bool(stairs.any()) and bool((~stairs).any()))
        self.assertTrue(torch.equal(got[~stairs], flat[~stairs]))
        self.assertTrue(bool((got[stairs] >= 0.0).all()) and bool((got[stairs] < 1.0).all()))

        cmd = self.env.command_manager.get_command("velocity_height")
        frame_z = self.env.scene[params["height_sensor_name"]].data.frame_pos_w[:, 0, 2]
        ground = ground_height_estimate(self.env, params["window_sensor_name"])
        error = frame_z - ground - cmd[:, 4]
        expected = 1.0 - torch.exp(-error.square() / params["sigma"] ** 2)
        self.assertTrue(torch.allclose(got[stairs], expected[stairs], atol=1e-6))


class FudanRewardRuntimeTests(unittest.TestCase):
    """M28 复旦 v3 奖励集：全地形统一 15 项、逐项限幅到每秒 ±1，速度课程的跟踪分日志照常写出。"""

    @classmethod
    def setUpClass(cls) -> None:
        cfg = rough_env_cfg(reward_set="fudan_v3")
        cfg.curriculum.pop("flat_warmup")
        cfg.scene.num_envs = 14
        cls.env = ManagerBasedRlEnv(cfg, device="cpu")
        cls.env.reset()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_default_reward_set_unchanged(self) -> None:
        default = rough_env_cfg().rewards
        self.assertIn("fall_penalty", default)
        self.assertNotIn("tracking_lin_vel_enhance", default)
        with self.assertRaises(ValueError):
            rough_env_cfg(reward_set="kami")

    def test_terms_are_uniform_and_clipped(self) -> None:
        from se3_train.tasks.rough.fudan_rewards import FUDAN_V3_SCALES

        manager = self.env.reward_manager
        self.assertEqual(set(manager.active_terms), set(FUDAN_V3_SCALES))
        self.assertTrue(all(cfg.weight == 1.0 for cfg in manager._term_cfgs))
        seen_curriculum_key = False
        generator = torch.Generator().manual_seed(0)
        for _ in range(80):
            action = torch.randn(
                self.env.num_envs, self.env.action_manager.total_action_dim, generator=generator
            )
            self.env.step(0.5 * action)
            # _step_reward 是每秒量：逐项限幅后绝对值不超过 1。
            self.assertLessEqual(float(manager._step_reward.abs().max()), 1.0 + 1e-6)
            log = self.env.extras.get("log", {})
            seen_curriculum_key |= "Locomotion/tracking_lin_vel_reward_curriculum" in log
        self.assertTrue(seen_curriculum_key)


if __name__ == "__main__":
    unittest.main()
