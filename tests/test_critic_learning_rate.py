"""critic 学习率解耦（se3_train.ppo）的回归测试。

验证：ActorCriticAdam 每次 step 前恢复 critic 的固定 lr（抵消 rsl_rl 的 KL 自适应规则对全部
param group 的改写）。旧 Flat CriticLr 任务已移除，这里仅保留优化器验证。

背景见 se3_train/ppo.py 与 docs/task_architecture.md：rsl_rl 的 KL 规则作用在 actor 与 critic 共用的
唯一 optimizer 上，σ 缩小后共用 lr 被压到 1e-5 地板，critic 一起冻住。
"""

from __future__ import annotations

import unittest

import torch

from se3_train.ppo import ActorCriticAdam
from se3_train.tasks.flat.rl_cfg import FLAT_LEARNING_RATE

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


if __name__ == "__main__":
    unittest.main()
