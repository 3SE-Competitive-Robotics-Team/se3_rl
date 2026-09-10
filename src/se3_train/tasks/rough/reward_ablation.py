"""统一平地与台阶基础奖励的四组专项奖励消融。"""

from mjlab.envs import ManagerBasedRlEnvCfg

from .env_cfg import env_cfg

# 四组共享地形、指令、课程和 PPO，仅切换两个专项奖励的权重。
REWARD_ABLATIONS = {
    "SE3-WheelLegged-Rough-RewardAbl-Base": (0.0, 0.0),
    "SE3-WheelLegged-Rough-RewardAbl-Progress": (3.0, 0.0),
    "SE3-WheelLegged-Rough-RewardAbl-Support": (0.0, 4.0),
    "SE3-WheelLegged-Rough-RewardAbl-Both": (3.0, 4.0),
}


def reward_ablation_env_cfg(
    *, progress_weight: float, support_weight: float, play: bool = False
) -> ManagerBasedRlEnvCfg:
    """保留 Rough 采样与共同能耗折价，基础奖励全线采用当前平地列定价。"""
    cfg = env_cfg(
        play=play,
        amp_enabled=False,
        ctbc_enabled=False,
        command_velocity_error_weight=None,
        zero_base_height_on_terrain=False,
        zero_tracking_ang_vel_on_terrain=False,
        terrain_vz_weight=2.0,
    )
    # 保留分列日志，但关闭台阶独有的宽核，所有列使用同一个 sigma_move。
    cfg.rewards["tracking_lin_vel"].params["stair_sigma_move"] = None
    cfg.rewards["stair_climb_progress"].weight = progress_weight
    cfg.rewards["stair_support_height"].weight = support_weight
    return cfg
