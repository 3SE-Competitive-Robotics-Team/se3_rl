"""任务契约注册表。"""

from __future__ import annotations

from .base import TaskContract
from .legacy_locomotion import LEGACY_LOCOMOTION_CONTRACT
from .task_mode_locomotion import TASK_MODE_LOCOMOTION_CONTRACT

TASK_CONTRACTS: dict[str, TaskContract] = {
    LEGACY_LOCOMOTION_CONTRACT.name: LEGACY_LOCOMOTION_CONTRACT,
    TASK_MODE_LOCOMOTION_CONTRACT.name: TASK_MODE_LOCOMOTION_CONTRACT,
}


def task_contract(name: str) -> TaskContract:
    """按名称读取任务契约。"""
    try:
        return TASK_CONTRACTS[name]
    except KeyError as exc:
        known = ", ".join(sorted(TASK_CONTRACTS))
        raise KeyError(f"未知任务契约: {name}; 可用契约: {known}") from exc


def task_contract_for_num_obs(num_obs: int) -> TaskContract:
    """按 actor 观测维度匹配任务契约。"""
    matches = [contract for contract in TASK_CONTRACTS.values() if contract.num_obs == num_obs]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(contract.name for contract in matches)
        raise ValueError(f"观测维度 {num_obs} 对应多个任务契约: {names}")
    known = ", ".join(f"{contract.name}={contract.num_obs}" for contract in TASK_CONTRACTS.values())
    raise ValueError(f"无法按观测维度 {num_obs} 匹配任务契约; 已知: {known}")


__all__ = ["TASK_CONTRACTS", "task_contract", "task_contract_for_num_obs"]
