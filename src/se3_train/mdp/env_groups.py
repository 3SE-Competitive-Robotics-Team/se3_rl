"""向量化环境的固定分组与 MDP term 过滤工具。"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.manager_base import ManagerTermBaseCfg


def _resolve_env_ids(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | Sequence[int] | slice | None,
) -> torch.Tensor:
    """把常见 env id 输入统一转换到环境设备。"""
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        start, stop, step = env_ids.indices(env.num_envs)
        return torch.arange(start, stop, step, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _resolve_nested_scene_cfgs(value: Any, env: ManagerBasedRlEnv) -> Any:
    """递归解析 wrapped term 参数中的 SceneEntityCfg。"""
    if isinstance(value, SceneEntityCfg):
        value.resolve(env.scene)
        return value
    if isinstance(value, dict):
        return {key: _resolve_nested_scene_cfgs(item, env) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_nested_scene_cfgs(item, env) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_nested_scene_cfgs(item, env) for item in value)
    return value


def _parse_wrapped_term(
    cfg: ManagerTermBaseCfg,
    env: ManagerBasedRlEnv,
) -> tuple[Any, dict[str, Any]]:
    """解析过滤器内部的 MDP term，并实例化 class-based term。"""
    wrapped_cfg = cfg.params.get("wrapped_term", cfg.params.get("wrapped_event"))
    if not isinstance(wrapped_cfg, dict):
        raise ValueError("过滤器需要 dict 类型的 wrapped_term 或 wrapped_event 配置。")

    term_func = wrapped_cfg.get("func")
    if not callable(term_func):
        raise ValueError("wrapped term 必须提供可调用的 func。")
    term_params = wrapped_cfg.get("params", {})
    if not isinstance(term_params, dict):
        raise ValueError("wrapped term 的 params 必须是 dict。")
    term_params = _resolve_nested_scene_cfgs(term_params, env)

    if inspect.isclass(term_func):
        if not issubclass(term_func, ManagerTermBase):
            raise TypeError(f"wrapped term class 必须继承 ManagerTermBase，实际为 {term_func}。")
        wrapped_term_cfg = replace(cfg, func=term_func, params=term_params)
        term_func = term_func(cfg=wrapped_term_cfg, env=env)
    return term_func, term_params


def _normalized_group_counts(probabilities: torch.Tensor, num_envs: int) -> torch.Tensor:
    """按最大余数法把组概率转换成总数严格等于 num_envs 的整数计数。"""
    expected = probabilities * num_envs
    counts = torch.floor(expected).to(dtype=torch.long)
    remainder = int(num_envs - counts.sum().item())
    if remainder > 0:
        fractional = expected - counts.float()
        counts[torch.topk(fractional, k=remainder).indices] += 1
    return counts


class AssignEnvGroups(ManagerTermBase):
    """在环境创建时按固定比例分配并打乱 ``env_group_ids``。"""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        env_groups = cfg.params.get("env_groups")
        probabilities = cfg.params.get("group_probabilities")

        if env_groups is not None:
            if not isinstance(env_groups, dict) or not env_groups:
                raise ValueError("env_groups 必须是非空的 {组名: 概率} 字典。")
            group_names = tuple(str(name) for name in env_groups)
            probabilities = tuple(float(value) for value in env_groups.values())
        else:
            if probabilities is None:
                raise ValueError("AssignEnvGroups 需要 env_groups 或 group_probabilities。")
            group_names = tuple(f"group_{index}" for index in range(len(probabilities)))

        probs = torch.as_tensor(probabilities, device=env.device, dtype=torch.float32)
        if probs.ndim != 1 or probs.numel() == 0:
            raise ValueError("分组概率必须是一维非空序列。")
        if torch.any(probs < 0):
            raise ValueError(f"分组概率不能为负数：{probabilities}")
        total = probs.sum()
        if not torch.isfinite(total) or total <= 0:
            raise ValueError(f"分组概率总和必须为有限正数：{probabilities}")
        probs = probs / total
        counts = _normalized_group_counts(probs, env.num_envs)

        group_ids = torch.empty(env.num_envs, device=env.device, dtype=torch.long)
        offset = 0
        for group_id, count in enumerate(counts.tolist()):
            if count > 0:
                group_ids[offset : offset + count] = group_id
                offset += count
        permutation = torch.randperm(env.num_envs, device=env.device)
        env.env_group_ids = group_ids[permutation]
        env.env_group_names = group_names
        env.env_group_counts = counts
        env.num_env_groups = int(probs.numel())

        # 分组在 class-based startup term 构造时已经完成；startup apply 只需调用空操作。
        cfg.params = {}

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | slice | None,
    ) -> None:
        del env, env_ids


def env_group_id_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """返回 ``[num_envs, 1]`` 的 int64 环境组编号观测。"""
    group_ids = getattr(env, "env_group_ids", None)
    if not isinstance(group_ids, torch.Tensor):
        raise RuntimeError("env_group_ids 尚未初始化，请先配置 AssignEnvGroups startup event。")
    if group_ids.shape != (env.num_envs,) or group_ids.dtype != torch.long:
        raise RuntimeError(
            "env_group_ids 必须是形状 "
            f"({env.num_envs},) 的 torch.long，实际为 {group_ids.shape}/{group_ids.dtype}。"
        )
    return group_ids.unsqueeze(-1)


def env_group_mask(
    env: ManagerBasedRlEnv,
    group_ids: int | Sequence[int],
) -> torch.Tensor:
    """返回属于指定一个或多个环境组的布尔 mask。"""
    current = env_group_id_obs(env).squeeze(-1)
    requested = torch.as_tensor(group_ids, device=env.device, dtype=torch.long).reshape(-1)
    if requested.numel() == 0:
        raise ValueError("group_ids 不能为空。")
    return torch.isin(current, requested)


class _FilteredGroupTermBase(ManagerTermBase):
    """按 ``env_group_ids`` 过滤 wrapped MDP term 的公共实现。"""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        self.wrapped_func, self.wrapped_params = _parse_wrapped_term(cfg, env)
        filter_cfg = cfg.params.get("filter")
        explicit_group_ids = cfg.params.get("group_ids")
        if explicit_group_ids is not None:
            requested = torch.as_tensor(
                explicit_group_ids,
                device=env.device,
                dtype=torch.long,
            ).reshape(-1)
            self.group_ids = tuple(int(value) for value in requested.detach().cpu().tolist())
        elif isinstance(filter_cfg, dict):
            if filter_cfg.get("field_name", "env_group_ids") != "env_group_ids":
                raise ValueError("当前过滤器只支持 field_name='env_group_ids'。")
            operation = filter_cfg.get("op", "eq")
            if operation == "eq":
                self.group_ids = (int(filter_cfg["value"]),)
            elif operation == "in":
                self.group_ids = tuple(int(value) for value in filter_cfg["values"])
            else:
                raise ValueError(f"环境组过滤只支持 eq/in，实际为 {operation}。")
        else:
            raise ValueError("过滤器需要 group_ids 或 filter 配置。")
        if not self.group_ids:
            raise ValueError("环境组过滤列表不能为空。")

        if hasattr(self.wrapped_func, "model_fields"):
            self.model_fields = self.wrapped_func.model_fields
        if hasattr(self.wrapped_func, "recompute"):
            self.recompute = self.wrapped_func.recompute
        cfg.params = {}

    def _mask(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        return env_group_mask(env, self.group_ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        reset_func = getattr(self.wrapped_func, "reset", None)
        if not callable(reset_func):
            return
        requested = _resolve_env_ids(self._env, env_ids)
        active_ids = requested[self._mask(self._env)[requested]]
        if active_ids.numel() > 0:
            reset_func(env_ids=active_ids)


class FilteredEventWrapper(_FilteredGroupTermBase):
    """只对指定环境组执行 wrapped reset/startup/interval event。"""

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | slice | None,
    ) -> None:
        requested = _resolve_env_ids(env, env_ids)
        active_ids = requested[self._mask(env)[requested]]
        if active_ids.numel() > 0:
            self.wrapped_func(env, active_ids, **self.wrapped_params)


class FilteredRewardWrapper(_FilteredGroupTermBase):
    """只在指定环境组保留 wrapped reward，其余环境返回零。"""

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        reward = self.wrapped_func(env, **self.wrapped_params)
        if not isinstance(reward, torch.Tensor) or reward.shape != (env.num_envs,):
            shape = getattr(reward, "shape", None)
            raise ValueError(f"wrapped reward 必须返回 ({env.num_envs},)，实际为 {shape}。")
        reward = reward.to(device=env.device)
        return torch.where(self._mask(env), reward, torch.zeros_like(reward))


class FilteredTerminationWrapper(_FilteredGroupTermBase):
    """只在指定环境组保留 wrapped termination。"""

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        active = self._mask(env)
        terminated = self.wrapped_func(env, **self.wrapped_params)
        if not isinstance(terminated, torch.Tensor) or terminated.shape != (env.num_envs,):
            shape = getattr(terminated, "shape", None)
            raise ValueError(f"wrapped termination 必须返回 ({env.num_envs},)，实际为 {shape}。")
        terminated = terminated.to(device=env.device, dtype=torch.bool)

        # stateful termination 会在全环境上更新计数，需清掉非本组状态，避免切组后继承。
        reset_func = getattr(self.wrapped_func, "reset", None)
        inactive_ids = torch.nonzero(~active, as_tuple=False).flatten()
        if callable(reset_func) and inactive_ids.numel() > 0:
            reset_func(env_ids=inactive_ids)
        return terminated & active


__all__ = [
    "AssignEnvGroups",
    "FilteredEventWrapper",
    "FilteredRewardWrapper",
    "FilteredTerminationWrapper",
    "env_group_id_obs",
    "env_group_mask",
]
