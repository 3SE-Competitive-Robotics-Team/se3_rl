# 控制频率基准

本项目默认采用 **500 Hz 物理仿真 + 50 Hz policy/action 更新**：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `RobotConfig.sim_dt` | `0.002 s` | MuJoCo / MJLab 物理积分步长，500 Hz |
| `RobotConfig.control_decimation` | `10` | 每 10 个物理步执行一次 policy action |
| `RobotConfig.control_dt` | `0.020 s` | policy/action 更新周期，50 Hz |

这个频率基准对齐 Unitree 官方 `unitree_rl_mjlab`、`unitree_rl_lab` 和 `unitree_rl_gym` 的常见配置：`0.005 s * 4 = 0.020 s`。我们保留更小的 `0.002 s` 物理步长，是为了闭链四连杆、轮地接触和跳跃落地阶段有更细的积分精度；只通过 `control_decimation=10` 把策略频率降到 50 Hz。

## 统一来源

控制频率的单一来源是 `se3_shared.RobotConfig`。训练端 `ManagerBasedRlEnvCfg.decimation`、sim2sim 默认参数、真机 runtime 导出配置都必须从这套共享配置派生，不能各自写死。

当前约定：

- 训练端：`src/se3_train/env_cfg.py` 读取 `_ROBOT_DEFAULTS.control_decimation` 和 `_ROBOT_DEFAULTS.sim_dt`。
- sim2sim：`src/se3_sim2sim/config.py` 和 CLI 默认值读取 `se3_shared.RobotConfig`。
- 奖励/课程中按秒换算 step 时，优先读取环境实际 `step_dt`；没有 `step_dt` 时才用 `physics_dt * decimation` 兜底。

## 修改频率时的检查项

如果后续需要重新调整控制频率，必须同步检查：

1. `src/se3_shared/robot.py` 的 `sim_dt`、`control_decimation`。
2. `src/se3_train/env_cfg.py` 是否仍从共享配置读取物理步长和 decimation。
3. `src/se3_sim2sim` 的 CLI 帮助文本、运行日志和配置导出。
4. `src/se3_train/mdp` 下所有按秒换算 step 的奖励、课程、状态机逻辑。
5. `docs/` 和真机 runtime 文档中的频率说明。

不要只改训练端 decimation。训练、sim2sim 和真机端频率漂移会直接制造 sim2sim gap，并让动作延迟、课程时长和奖励时间窗失真。
