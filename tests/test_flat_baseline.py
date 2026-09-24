"""冻结的 Flat 基线回归测试。

基线 = D11（W&B `mher9vfk`，commit 236666c，2026-09-06）。不带任何命令行覆盖直接跑
`SE3-WheelLegged-Flat-MLP` 即应复现该 run，本测试逐项钉住它的配置，防止后续改动无声漂移。

数值来源与依据见 docs/task_architecture.md 的 Flat 基线行、`tasks/flat/env_cfg.py` 与
`tasks/flat/rl_cfg.py` 的注释。要改基线就一并改这里，并在提交信息里写清对照实验编号。
"""

from __future__ import annotations

import unittest

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import se3_train  # noqa: F401  # 注册任务

_MLP = "SE3-WheelLegged-Flat-MLP"
_HISTORY = "SE3-WheelLegged-Flat-History-MLP"
_GRU = "SE3-WheelLegged-Flat-GRU"

# PPO：2026-09-06 对齐 BioInnov/kyber_rl_lab 的 locomotion 基线（commit f8057ab）。
_ALGORITHM = {
    "value_loss_coef": 1.0,
    "use_clipped_value_loss": True,
    "clip_param": 0.2,
    "entropy_coef": 0.01,
    "num_learning_epochs": 5,
    "num_mini_batches": 4,
    "learning_rate": 1.0e-3,
    "schedule": "adaptive",
    "gamma": 0.99,
    "lam": 0.95,
    "desired_kl": 0.01,
    "max_grad_norm": 1.0,
}

# 奖励表：commit e1946ba 把 D2–D8 已验证的改动合并为默认值。
_REWARD_WEIGHTS = {
    "tracking_lin_vel": 4.0,
    "tracking_ang_vel": 3.0,
    "tracking_lin_yaw_joint": 2.0,
    "tracking_orientation_l2": -12.0,
    "flat_base_height": -4.0,
    "bad_tilt": -6.0,
    "ang_vel_xy": -0.146,
    "angular_momentum": -5.0e-5,
    "leg_torques": -2.0e-4,
    "wheel_torques": -1.0e-4,
    "stand_still": -1.0,
    "leg_power": -1.03e-4,
    "leg_dof_acc": -2.17e-07,
    "action_rate": -0.48,
    "action_smoothness": -0.12,
    "joint_mirror": -0.179,
    "dof_pos_limits": -5.0,
    "collision": -16.0,
    "contact_forces": -1.07e-3,
    "flat_wheel_contact": -10.0,
    "flat_leg_contact": -25.0,
    "is_alive": 1.0,
}
# D7 起删除：该项 99% 的代价来自指令阶跃后 1 s 的不可达瞬态。
_REMOVED_REWARDS = ("command_velocity_error",)


class FlatBaselineAlgorithmTests(unittest.TestCase):
    def test_ppo_hyperparameters_are_frozen(self) -> None:
        for task in (_MLP, _HISTORY, _GRU):
            algorithm = load_rl_cfg(task).algorithm
            for key, expected in _ALGORITHM.items():
                actual = getattr(algorithm, key)
                if isinstance(expected, (bool, str)):
                    self.assertEqual(actual, expected, msg=f"{task}.{key}")
                else:
                    self.assertAlmostEqual(
                        float(actual), float(expected), places=9, msg=f"{task}.{key}"
                    )

    def test_rollout_length_and_iterations(self) -> None:
        # D 系列全部用 24 步；GRU 的该值同时是 BPTT 窗口，保留 64。
        self.assertEqual(load_rl_cfg(_MLP).num_steps_per_env, 24)
        self.assertEqual(load_rl_cfg(_HISTORY).num_steps_per_env, 24)
        self.assertEqual(load_rl_cfg(_GRU).num_steps_per_env, 64)
        for task in (_MLP, _HISTORY, _GRU):
            cfg = load_rl_cfg(task)
            self.assertEqual(cfg.max_iterations, 3500, msg=task)
            self.assertEqual(cfg.save_interval, 100, msg=task)

    def test_baseline_uses_stock_ppo_not_critic_lr_variant(self) -> None:
        # critic 固定 LR 在 D5 与 D9 两次失败，不得进入基线。
        algorithm = load_rl_cfg(_MLP).algorithm
        self.assertIsNone(getattr(algorithm, "critic_learning_rate", None))
        self.assertNotIn("Se3PPO", str(getattr(algorithm, "class_name", "")))


class FlatBaselineEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_env_cfg(_MLP)

    def test_reward_table_is_frozen(self) -> None:
        self.assertEqual(set(self.cfg.rewards), set(_REWARD_WEIGHTS))
        for name, weight in _REWARD_WEIGHTS.items():
            self.assertAlmostEqual(
                float(self.cfg.rewards[name].weight), float(weight), places=10, msg=name
            )
        for name in _REMOVED_REWARDS:
            self.assertNotIn(name, self.cfg.rewards)

    def test_action_penalty_wheel_pricing_is_unit(self) -> None:
        # D4 对 D2：轮 σ 0.922 → 0.256，撤销 (15/45)² 折价。
        self.assertAlmostEqual(self.cfg.rewards["action_rate"].params["wheel_scale"], 1.0)
        self.assertAlmostEqual(self.cfg.rewards["action_smoothness"].params["wheel_scale"], 2.0)
        self.assertAlmostEqual(self.cfg.rewards["action_rate"].params["leg_scale"], 1.0)
        self.assertAlmostEqual(self.cfg.rewards["action_smoothness"].params["leg_scale"], 1.0)

    def test_action_semantics_and_scale(self) -> None:
        action = self.cfg.actions["delayed_action"]
        self.assertEqual(action.leg_action_semantics, "joint")
        self.assertAlmostEqual(float(action.wheel_scale), 15.0)
        self.assertFalse(action.height_conditioned_action_default)

    def test_command_ranges(self) -> None:
        command = self.cfg.commands["velocity_height"]
        self.assertEqual(tuple(command.height_range), (0.20, 0.38))
        self.assertEqual(tuple(command.standing_height_range), (0.20, 0.38))
        self.assertEqual(tuple(command.deployment_ranges["height"]), (0.20, 0.38))
        self.assertEqual(tuple(command.deployment_ranges["lin_vel_x"]), (-2.4, 2.4))

    def test_domain_randomization_scale(self) -> None:
        # 2026-09-06：±20 mm 等于整机质心 ±16.9 mm、配平倾角 ±7.8°，比静平衡 bug 还大；收到 ±5 mm。
        self.assertAlmostEqual(self.cfg.events["com"].params["com_range"], 0.005)
        self.assertEqual(tuple(self.cfg.events["base_mass"].params["mass_range"]), (-0.5, 1.5))
        self.assertEqual(tuple(self.cfg.events["inertia"].params["inertia_range"]), (0.8, 1.2))
        self.assertEqual(tuple(self.cfg.events["friction"].params["friction_range"]), (0.2, 1.5))
        self.assertEqual(
            tuple(self.cfg.events["default_dof_pos"].params["offset_range"]), (-0.05, 0.05)
        )

    def test_curriculum_terms(self) -> None:
        # 只有速度课程与推力课程；高度范围不走课程。
        self.assertEqual(set(self.cfg.curriculum), {"command_vel", "push_disturbance"})


if __name__ == "__main__":
    unittest.main()
