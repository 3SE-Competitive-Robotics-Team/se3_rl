# kami-submission 台阶奖励参考核对

固定提交：17b97ad8bc2d0ee56f9b389f0aba5c3120b066d4。仅参考用户指定kami-submission，不混用此前A16的cloud-changes来源。未修改训练代码。

台阶终止项的GroupTermination groups=[0]，因此按reward variants里的第0组提取。配置外层GroupReward weight=1，各项下表为组内配置权重；GroupReward实现未提交，不能进一步验证运行时数学。

| term | 配置权重 | 函数 | 关键参数 |
|---|---:|---|---|
| tracking_lin_vel | 3.5 | se3_train.tasks.flat.rewards.tracking_lin_vel | {"command_name": "velocity_height", "sigma": "0.245"} |
| tracking_ang_vel | 1.5 | se3_train.tasks.flat.rewards.tracking_ang_vel | {"command_name": "velocity_height", "sigma": "0.1"} |
| tracking_height | 2.0 | se3_train.tasks.flat.rewards.tracking_height | {"command_name": "velocity_height", "sigma": "0.00125", "support_sensor_name": "stair_wheel_ground"} |
| tracking_combined | 1.5 | se3_train.tasks.flat.rewards.tracking_combined | {"command_name": "velocity_height", "lin_sigma": "0.245", "yaw_sigma": "0.1", "height_sigma": "0.00125", "support_sensor_name": "stair_wheel_ground"} |
| orientation | -8.0 | mjlab.envs.mdp.rewards.flat_orientation_l2 | {} |
| upright_quality | 0.2 | se3_train.tasks.flat.rewards.upright_quality | {"scale": "0.17453292519943295"} |
| action_smoothness | 0.3 | se3_train.tasks.flat.rewards.action_smoothness_quality | {"scale": "0.25"} |
| action_rate | -0.05 | mjlab.envs.mdp.rewards.action_rate_l2 | {} |
| lateral_velocity | -0.1 | se3_train.tasks.flat.rewards.lateral_velocity_l2 | {} |
| is_alive | 0.05 | mjlab.envs.mdp.rewards.is_alive | {} |
| leg_pos_limits | -1.0 | mjlab.envs.mdp.rewards.joint_pos_limits | {} |
| actuator_effort | -1.0e-05 | mjlab.envs.mdp.rewards.joint_torques_l2 | {} |
| positive_mechanical_power | -0.0001 | mjlab.envs.mdp.rewards.electrical_power_cost | {} |
| leg_acceleration | -1.0e-08 | mjlab.envs.mdp.rewards.joint_acc_l2 | {} |
| wheel_acceleration | -1.0e-10 | mjlab.envs.mdp.rewards.joint_acc_l2 | {} |
| leg_mirror | -2.0 | se3_train.tasks.flat.stair_rewards.leg_mirror_penalty | {} |
| base_angular_velocity | -0.05 | mjlab.tasks.velocity.mdp.rewards.body_angular_velocity_penalty | {} |
| base_terrain_contact | -1.0 | mjlab.tasks.velocity.mdp.rewards.self_collision_cost | {"sensor_name": "stair_base_contact"} |
| leg_upright | -2.0 | se3_train.tasks.flat.stair_rewards.leg_upright_penalty | {"allowed_angle_deg": "5.0"} |

共19项。没有climb_progress、step_height_progress、stair_support_height或额外command_velocity_error；没有轮子离地惩罚，也没有A15同名的腿触地惩罚，但有stair_base_contact传感器对应的接触项。不能把未同名当作所有物理约束完全不存在。

重要差别：

- 速度核.245、权重3.5，A15台阶核1.44、权重4；可见源码tracking_lin_vel使用exp(-err²/sigma)，因此前者对超速/欠速更敏感，但大误差启动信号也更弱。
- 配置保留tracking_height(+2)和tracking_combined(+1.5)，两者台阶组都传stair_wheel_ground。README解释高度按轮下地面算净空，不用世界z直接罚。相关函数未提交，不能确认联合项是乘积或其具体公式，也不能按sigma直接计算高度容差。
- leg_mirror(-2)与leg_upright(-2,allowed_angle_deg=5)针对腿姿。README说其初版曾学会前后叉腿、侧绕上阶，加入后改善；这是作者报告，不是本次独立评测。不能将其leg_mirror直接等同本仓joint_mirror。
- 有orientation、upright_quality、base_angular_velocity、leg/wheel_acceleration、action_rate和action_smoothness等；并非极简奖励方案，不能用它证明之前建议删的项无用。
- 除奖励外还有route_departure、stuck_without_progress、route_finished等终止项，函数缺失，阈值与细节UNKNOWN。这些也可能改变爬阶行为与收益，不能仅凭无进度奖励就断定其全部成功来自tracking。
- README三阶段平地→恢复→混合训练，最终agent.yaml resume=true，load_checkpoint=model_330.pt。不是从零训练对照证据；用户当前明确不想续训，因此不能照搬训练履历。

源码完整性限制：提交有env.yaml/agent.yaml/model.onnx/readme，但git tree缺少stair_rewards.py、mixed_cfg.py、stair_env_cfg.py、mixed_recovery_cfg.py、recovery_cfg.py。现有flat/rewards.py也缺少配置引用的tracking_height、tracking_combined、upright_quality、action_smoothness_quality、lateral_velocity_l2。故能确认配置意图，不能宣称完整复刻数学或复现训练。没有执行该仓库代码或反序列化其pt/pickle；YAML仅解析节点与别名，不构造Python对象。

对A15的参考建议：优先研究明确的腿姿约束与相对地面高度目标；减少依赖无速度上限的位移奖励；但此前A12删进度组失败，应把无进度版本视为独立实验假设而非直接删除结论。若要精简，不应照搬19项；尤其联合跟踪项和多个姿态/平滑项有重叠，需要先拿到函数实现再定。

来源：
- [submission/training_config/env.yaml](https://github.com/jinhuixie666-cmyk/se3_rl_competiition/blob/17b97ad8bc2d0ee56f9b389f0aba5c3120b066d4/submission/training_config/env.yaml)
- [submission/training_config/agent.yaml](https://github.com/jinhuixie666-cmyk/se3_rl_competiition/blob/17b97ad8bc2d0ee56f9b389f0aba5c3120b066d4/submission/training_config/agent.yaml)
- [submission/kami_readme.md](https://github.com/jinhuixie666-cmyk/se3_rl_competiition/blob/17b97ad8bc2d0ee56f9b389f0aba5c3120b066d4/submission/kami_readme.md)
- [src/se3_train/tasks/flat/rewards.py](https://github.com/jinhuixie666-cmyk/se3_rl_competiition/blob/17b97ad8bc2d0ee56f9b389f0aba5c3120b066d4/src/se3_train/tasks/flat/rewards.py)
