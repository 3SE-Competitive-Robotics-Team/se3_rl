"""验收 Recovery-Discovery 五帧历史观测的数据与配置契约。"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_rl_cfg

from se3_shared import ObservationConfig
from se3_train.tasks.recovery_discovery import HISTORY_MLP_TASK_ID, MLP_TASK_ID
from se3_train.tasks.recovery_discovery.env_cfg import (
    RECOVERY_DISCOVERY_HISTORY_LENGTH,
    env_cfg,
    history_env_cfg,
)
from se3_train.tasks.recovery_discovery.rl_cfg import mlp_rl_cfg


def _callable_contract(value: Any) -> tuple[object, ...]:
    code = getattr(value, "__code__", None)
    closure = getattr(value, "__closure__", None) or ()
    return (
        getattr(value, "__module__", ""),
        getattr(value, "__qualname__", repr(value)),
        None if code is None else code.co_code,
        None if code is None else code.co_consts,
        tuple(_canonical(cell.cell_contents) for cell in closure),
    )


def _canonical(value: Any) -> Any:
    if callable(value):
        return ("callable", _callable_contract(value))
    if is_dataclass(value):
        return (
            type(value).__qualname__,
            tuple((field.name, _canonical(getattr(value, field.name))) for field in fields(value)),
        )
    if isinstance(value, dict):
        return tuple((key, _canonical(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return (str(value.dtype), tuple(value.shape), value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return (str(value.dtype), tuple(value.shape), value.tolist())
    return value


def _assert_config_contract() -> None:
    baseline = env_cfg()
    history = history_env_cfg()
    actor = history.observations["actor"]
    if actor.history_length != RECOVERY_DISCOVERY_HISTORY_LENGTH:
        raise AssertionError(f"actor history length 错误: {actor.history_length}")
    if not actor.flatten_history_dim:
        raise AssertionError("actor history 必须展开为 MLP 输入")
    if history.observations["critic"].history_length is not None:
        raise AssertionError("critic 不应引入历史观测")

    history.observations["actor"] = replace(
        actor,
        history_length=baseline.observations["actor"].history_length,
        flatten_history_dim=baseline.observations["actor"].flatten_history_dim,
    )
    if _canonical(history) != _canonical(baseline):
        raise AssertionError("除 actor history 外，环境配置发生漂移")
    history_rl = load_rl_cfg(HISTORY_MLP_TASK_ID)
    baseline_rl = load_rl_cfg(MLP_TASK_ID)
    if _canonical(history_rl) != _canonical(baseline_rl):
        raise AssertionError("History-MLP 与单帧 MLP 的 PPO 配置不一致")
    if _canonical(history_rl) != _canonical(mlp_rl_cfg()):
        raise AssertionError("History-MLP 注册的 PPO 配置与配置工厂不一致")


def _split_term_history(env: ManagerBasedRlEnv, actor: torch.Tensor) -> dict[str, torch.Tensor]:
    history_length = RECOVERY_DISCOVERY_HISTORY_LENGTH
    result: dict[str, torch.Tensor] = {}
    cursor = 0
    for name, cfg in zip(
        env.observation_manager.active_terms["actor"],
        env.observation_manager._group_obs_term_cfgs["actor"],
        strict=True,
    ):
        current = cfg.func(env, **cfg.params)
        term_size = int(current.shape[-1])
        block_size = history_length * term_size
        result[name] = actor[:, cursor : cursor + block_size].reshape(
            env.num_envs, history_length, term_size
        )
        cursor += block_size
    if cursor != actor.shape[-1]:
        raise AssertionError(f"历史切片未覆盖 actor: cursor={cursor}, dim={actor.shape[-1]}")
    return result


def _assert_rollout_contract() -> None:
    cfg = history_env_cfg(play=True)
    cfg.scene.num_envs = 2
    cfg.seed = 17
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
    try:
        observations, _ = env.reset(seed=17)
        actor0 = observations["actor"]
        critic0 = observations["critic"]
        base_num_obs = ObservationConfig().num_obs
        expected_actor_dim = base_num_obs * RECOVERY_DISCOVERY_HISTORY_LENGTH
        if actor0.shape != (2, expected_actor_dim):
            raise AssertionError(f"actor shape 错误: {tuple(actor0.shape)}")
        if critic0.shape != (2, 40):
            raise AssertionError(f"critic shape 错误: {tuple(critic0.shape)}")
        if not torch.isfinite(actor0).all() or not torch.isfinite(critic0).all():
            raise AssertionError("reset observation 含 NaN/Inf")

        terms0 = _split_term_history(env, actor0)
        for name, values in terms0.items():
            expected = values[:, -1:, :].expand_as(values)
            if not torch.equal(values, expected):
                raise AssertionError(f"reset 未用首帧回填全部历史槽位: {name}")

        actions = torch.tensor(
            [[0.2, -0.1, -0.2, 0.1, 0.4, -0.4], [-0.3, 0.2, 0.3, -0.2, -0.5, 0.5]],
            dtype=torch.float32,
        )
        observations1, *_ = env.step(actions)
        actor1 = observations1["actor"]
        terms1 = _split_term_history(env, actor1)
        for name in terms0:
            if not torch.equal(terms1[name][:, :-1], terms0[name][:, 1:]):
                raise AssertionError(f"历史时间顺序或移位错误: {name}")
        if torch.equal(terms1["last_actions"][:, -1], terms0["last_actions"][:, -1]):
            raise AssertionError("step 后 last_actions 最新帧没有更新")

        env_ids = torch.tensor([0], dtype=torch.long)
        observations2, _ = env.reset(env_ids=env_ids)
        terms2 = _split_term_history(env, observations2["actor"])
        for name, values in terms2.items():
            expected = values[0, -1:].expand_as(values[0])
            if not torch.equal(values[0], expected):
                raise AssertionError(f"单环境 reset 未独立回填历史: {name}")
        if not torch.isfinite(observations2["actor"]).all():
            raise AssertionError("单环境 reset 后 actor 含 NaN/Inf")
    finally:
        env.close()


def main() -> int:
    _assert_config_contract()
    _assert_rollout_contract()
    print(
        "recovery history observation contract ok: "
        f"34D x {RECOVERY_DISCOVERY_HISTORY_LENGTH} = "
        f"{ObservationConfig().num_obs * RECOVERY_DISCOVERY_HISTORY_LENGTH}D actor, 40D critic"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
