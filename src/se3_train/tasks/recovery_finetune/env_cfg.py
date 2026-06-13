"""倒地自启 Stage II FineTune 环境配置。"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sim import MujocoCfg, SimulationCfg

from se3_shared import RobotConfig as SharedRobotConfig
from se3_train.mdp import events as mdp_events
from se3_train.robot_cfg import get_serialleg_cfg
from se3_train.tasks.recovery.env_cfg import env_cfg as recovery_env_cfg

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_STAGE2_MJCF_PATH = (
    _PROJECT_ROOT
    / "assets"
    / "robots"
    / "serialleg"
    / "mjcf"
    / "serialleg_fourbar_surrogate_stair_visualbase_coacd_train.xml"
)
_STAGE2_CACHE_PATH = _PROJECT_ROOT / "assets" / "recovery_states" / "serialleg_stair_v3_40k.npz"
_ROBOT_DEFAULTS = SharedRobotConfig()
_RECOVERY_WHEEL_KD = 0.08


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构造使用 v3 cache 的 Stage II 自起精调环境。"""
    cfg = recovery_env_cfg(play=play)
    cfg.scene.entities["robot"] = get_serialleg_cfg(
        mjcf_path=_STAGE2_MJCF_PATH,
        wheel_kd_override=_RECOVERY_WHEEL_KD,
    )

    command_cfg = cfg.commands["velocity_height"]
    command_cfg.lin_vel_x_range = (0.0, 0.0)
    command_cfg.ang_vel_yaw_range = (0.0, 0.0)
    command_cfg.height_range = (0.24, 0.30)
    command_cfg.standing_height_range = (0.24, 0.30)
    command_cfg.standing_ratio = 0.0

    cfg.events["reset_root_state"] = EventTermCfg(
        func=mdp_events.reset_root_state_full,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "recovery_prob": 1.0,
            "recovery_state_cache_path": str(_STAGE2_CACHE_PATH),
            "recovery_state_cache_prob": 1.0,
            "recovery_state_cache_split": "train",
            "recovery_grace_steps": 400,
            "recovery_command_height": None,
            "recovery_zero_velocity_command": False,
        },
    )
    cfg.events["reset_joints"] = EventTermCfg(
        func=mdp_events.reset_joints,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_offset_range": 0.0,
            "joint_vel_range": (0.0, 0.0),
            "wheel_joint_vel_range": (0.0, 0.0),
            "joint_randomization_prob": 0.0,
            "full_joint_randomization": False,
            "align_root_height_to_wheels": False,
        },
    )
    cfg.events.pop("push_robots", None)

    if "tracking_height" in cfg.rewards:
        cfg.rewards["tracking_height"].weight = -1500.0
        cfg.rewards["tracking_height"].params["use_pose_end_gate"] = False
        cfg.rewards["tracking_height"].params["use_upright_gate"] = False
        cfg.rewards["tracking_height"].params["use_inverted_free_upright_height_gate"] = True
    if "upward" in cfg.rewards:
        cfg.rewards["upward"].weight = 3.0

    cfg.sim = SimulationCfg(
        nconmax=256,
        njmax=1040,
        mujoco=MujocoCfg(
            timestep=_ROBOT_DEFAULTS.sim_dt,
            iterations=12,
            ls_iterations=8,
            ccd_iterations=15,
            tolerance=1e-6,
        ),
    )
    cfg.clip_observations = 100.0
    return cfg
