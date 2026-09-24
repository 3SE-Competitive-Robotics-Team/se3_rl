"""critic 学习率解耦（se3_train.ppo）的回归测试。

守护两件事：ActorCriticAdam 每次 step 前恢复 critic 的固定 lr（抵消 rsl_rl 的 KL 自适应规则对全部
param group 的改写），以及 Flat 的 CriticLr 变体与其基线任务除该项外逐项相同。

背景见 se3_train/ppo.py 与 docs/task_architecture.md：rsl_rl 的 KL 规则作用在 actor 与 critic 共用的
唯一 optimizer 上，σ 缩小后共用 lr 被压到 1e-5 地板，critic 一起冻住。
"""

from __future__ import annotations

import unittest
from dataclasses import asdict

import torch

import se3_train  # noqa: F401  # 注册任务
from se3_train.ppo import ActorCriticAdam
from se3_train.tasks.flat.rl_cfg import FLAT_LEARNING_RATE

_BASELINE_TASK = "SE3-WheelLegged-Flat-Exp-JointActionWheelPriceNoCmdErr"
_CRITIC_LR_TASK = "SE3-WheelLegged-Flat-Exp-JointActionWheelPriceNoCmdErrCriticLr"
# rsl_rl 的 KL 自适应地板；D4/D8 长期贴在这里，正是本改动要隔离的值。
_KL_FLOOR_LR = 1.0e-5


class ActorCriticAdamTests(unittest.TestCase):
    @staticmethod
    def _optimizer() -> tuple[ActorCriticAdam, torch.nn.Parameter, torch.nn.Parameter]:
        actor = torch.nn.Parameter(torch.zeros(3))
        critic = torch.nn.Parameter(torch.zeros(3))
        optimizer = ActorCriticAdam(
            [actor], [critic], lr=FLAT_LEARNING_RATE, critic_lr=FLAT_LEARNING_RATE
        )
        return optimizer, actor, critic

    def test_param_groups_are_split_actor_then_critic(self) -> None:
        optimizer, actor, critic = self._optimizer()
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertIs(optimizer.param_groups[0]["params"][0], actor)
        self.assertIs(optimizer.param_groups[1]["params"][0], critic)

    def test_step_restores_critic_lr_after_kl_rule_overwrites_all_groups(self) -> None:
        optimizer, actor, critic = self._optimizer()
        actor.grad = torch.ones(3)
        critic.grad = torch.ones(3)

        # rsl_rl.PPO.update 在 KL 超阈时对全部 param group 写入同一个 lr。
        for group in optimizer.param_groups:
            group["lr"] = _KL_FLOOR_LR

        optimizer.step()

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], _KL_FLOOR_LR, places=12)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], FLAT_LEARNING_RATE, places=12)
        # critic 的实际步长必须大于被压到地板的 actor，否则解耦没有生效。
        self.assertGreater(float(critic.detach().abs().max()), float(actor.detach().abs().max()))

    def test_critic_lr_survives_repeated_overwrites(self) -> None:
        optimizer, actor, critic = self._optimizer()
        for _ in range(5):
            actor.grad = torch.ones(3)
            critic.grad = torch.ones(3)
            for group in optimizer.param_groups:
                group["lr"] = _KL_FLOOR_LR
            optimizer.step()
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], FLAT_LEARNING_RATE, places=12)


class FlatCriticLrTaskTests(unittest.TestCase):
    def test_variant_differs_from_baseline_only_in_critic_learning_rate(self) -> None:
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

        baseline = load_rl_cfg(_BASELINE_TASK)
        variant = load_rl_cfg(_CRITIC_LR_TASK)

        self.assertEqual(getattr(baseline.algorithm, "critic_learning_rate", None), None)
        self.assertAlmostEqual(variant.algorithm.critic_learning_rate, FLAT_LEARNING_RATE)
        self.assertEqual(variant.algorithm.class_name, "se3_train.ppo:Se3PPO")

        base_alg = asdict(baseline.algorithm)
        var_alg = asdict(variant.algorithm)
        ignored = {"class_name", "critic_learning_rate"}
        shared = (set(base_alg) | set(var_alg)) - ignored
        for key in sorted(shared):
            self.assertEqual(base_alg.get(key), var_alg.get(key), msg=f"algorithm.{key}")
        for field in ("num_steps_per_env", "max_iterations", "save_interval", "obs_groups"):
            self.assertEqual(getattr(baseline, field), getattr(variant, field), msg=field)

        base_env = load_env_cfg(_BASELINE_TASK)
        var_env = load_env_cfg(_CRITIC_LR_TASK)
        self.assertEqual(set(base_env.rewards), set(var_env.rewards))
        for name, term in base_env.rewards.items():
            self.assertEqual(term.weight, var_env.rewards[name].weight, msg=name)
        self.assertEqual(
            tuple(base_env.commands["velocity_height"].height_range),
            tuple(var_env.commands["velocity_height"].height_range),
        )
        # 该基线自 2026-09-05 起不含速度违令罚，变体必须继承这一点。
        self.assertNotIn("command_velocity_error", var_env.rewards)


if __name__ == "__main__":
    unittest.main()
