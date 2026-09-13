"""A27 热身检查点之后，仅替换上台阶奖励为复旦上台阶3配方。"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg, RewardTermCfg

from se3_train.rl_cfg import RslRlOnPolicyRunnerCfg

from . import fudan_rewards
from .a24_stair_height import a24_rl_cfg
from .a26_stair_sampling import a27_warmup_env_cfg
from .stair_column_rewards import off_column, on_column


def a28_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    """跳过已完成的热身，保留两列地形几何及原指令、课程、终止配置。"""
    cfg = a27_warmup_env_cfg(play=play)
    if not play:
        cfg.curriculum["flat_warmup"].params["iterations"] = 0
    cfg.rewards = {
        name: RewardTermCfg(
            func=off_column,
            weight=term.weight,
            params={"inner": term.func, "params": dict(term.params)},
        )
        for name, term in cfg.rewards.items()
    }
    for name in fudan_rewards.WEIGHTS:
        cfg.rewards["fudan_" + name] = RewardTermCfg(
            func=on_column,
            weight=1.0,
            params={"inner": fudan_rewards.reward, "params": {"term": name}},
        )
    cfg.events["reset_fudan_history"] = EventTermCfg(func=fudan_rewards.reset_history, mode="reset")
    return cfg


def a28_rl_cfg(*, smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """加载 A27 热身末网络，重置优化器和轮数，再训练一千轮。"""
    cfg = a24_rl_cfg(smoke=smoke)
    cfg.load_run = "2026-09-12_23-42-48_rough-A27-warmup500-seed42-6x8192-1500"
    cfg.load_checkpoint = "model_500.pt"
    return cfg
