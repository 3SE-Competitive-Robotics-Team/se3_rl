"""以完整 A12 为基准，每次撤掉一项改动的十组必要性消融。"""

from mjlab.envs import ManagerBasedRlEnvCfg

from se3_train.rl_cfg import RslRlOnPolicyRunnerCfg

from .env_cfg import env_cfg
from .rl_cfg import amp_rl_cfg

A12_ABLATIONS = {
    "A0": "完整 A12",
    "A1": "恢复窄速度核",
    "A2": "恢复竖直速度项",
    "A3": "恢复高度惩罚",
    "A4": "恢复 yaw 奖励",
    "A5": "删除速度违令罚",
    "A6": "恢复能耗原价",
    "A7": "删除前进进度奖励",
    "A8": "删除双轮支撑奖励",
    "A9": "关闭 AMP",
}


def a12_ablation_env_cfg(variant: str, *, play: bool = False) -> ManagerBasedRlEnvCfg:
    """保留 A12 地形、指令、课程和观测，仅覆盖该组指定因素。"""
    if variant not in A12_ABLATIONS:
        raise ValueError(f"未知 A12 消融组: {variant}")
    cfg = env_cfg(
        play=play,
        amp_enabled=True,
        terrain_vz_weight=2.0 if variant == "A2" else 0.0,
        zero_base_height_on_terrain=variant != "A3",
        zero_tracking_ang_vel_on_terrain=variant != "A4",
        command_velocity_error_weight=None if variant == "A5" else -2.0,
        energy_penalty_scale=1.0 if variant == "A6" else 0.1,
    )
    if variant == "A1":
        cfg.rewards["tracking_lin_vel"].params["stair_sigma_move"] = 0.08
    if variant == "A7":
        cfg.rewards["stair_climb_progress"].weight = 0.0
    if variant == "A8":
        cfg.rewards["stair_support_height"].weight = 0.0
    return cfg


def a12_ablation_rl_cfg(variant: str) -> RslRlOnPolicyRunnerCfg:
    """A9 关闭 AMP 训练与奖励，保留同一 PPO 类和观测组以隔离变量。"""
    if variant not in A12_ABLATIONS:
        raise ValueError(f"未知 A12 消融组: {variant}")
    cfg = amp_rl_cfg()
    if variant == "A9":
        cfg.algorithm.amp_cfg = None
    return cfg
