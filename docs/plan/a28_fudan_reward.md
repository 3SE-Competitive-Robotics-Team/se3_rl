# A28：复旦上台阶3奖励对照

来源：yly-true/fudan_rl_wheel_leg，commit 8204e853dfd2ed06d85a322e1a998c3d20a3be2c，plane/logs/wheel_legged/上台阶3 的配置和奖励函数。

## 固定条件

从A27-Warmup500的2026-09-12_23-42-48_rough-A27-warmup500-seed42-6x8192-1500/model_500.pt加载actor和critic；重置优化器、迭代与环境课程状态，不重复热身。维持两列几何，但首次重置将所有环境放到stairs_up第0级。指令[0,2.4]、原有只升不降课程、PPO、观测/动作、终止和域随机化不变。用户要求4卡、每卡8192；总计1000轮，seed42，save100。

每轮总样本为32768×24，较旧六卡减少三分之一，所以相同迭代并非相同总采样量；本实验是奖励配方工程验证，不宣称严格等预算消融。

## 奖励

保留旧项作为非台阶奖励，通过off_column在stairs_up归零；添加16个fudan_项，仅stairs_up生效。旧项仍可计算诊断，不代表它们参与台阶训练奖励。

权重：tracking_lin_vel 1.5，tracking_lin_vel_enhance 1.5，tracking_ang_vel 1；base_height 1，base_height_enhance 1；nominal_state -1；lin_vel_z -.1；ang_vel_xy -.05；orientation -10；dof_vel -5e-5；dof_acc -2.5e-7；torques -.0001；action_rate -.05；action_smooth -.05；collision -1；dof_pos_limits -1。

速度双核exp(-ev²/.25)、exp(-ev²/2.5)-1；高度双核1.5exp(-1000eh²)、exp(-eh²/.01)-1。每项先乘权重裁[-1,1]，外层RewardManager权重1再乘dt，禁止重复dt缩放或只裁总分。不叠加存活/爬升/支撑奖励。来源碰撞惩罚对象为空，collision保留为0；本机原有终止条件不变。

## 明确的机器人适配

- 高度沿用base_height_sensor和frame_height_above_terrain，保持本机测高语义；没有声称复现其77点地形平均高度。
- 名义腿角由实际髋body到轮心的向量转到机身系，比较左右相对向下轴的角度；不用参考串联腿长/膝角计算闭链几何。
- 关节速度/限位使用四个主动电机坐标，六关节加速度按policy tick速度差分。关节软限位为本机物理限位宽度的97%。
- 力矩取六电机实际actuator_force，排除气弹簧。action_rate覆盖六维，二阶action_smooth只覆盖四腿维度，均为原始动作差分，不复用本机带周期处理/姿态门控的同名奖励。
- 每步共享一次计算缓存，历史按episode reset清零；复位其他env不清除未重置env的历史。

## 验证

后续课程调整：最高行号限制为 3，仍从 0 开始逐级升级，达到 3 后保持。保留原来的十行地形及各行几何，不把行数改成四行，避免难度重新插值；原始 A28 已完成的四卡训练不包含此限制。

16环境运行检查：全部stairs_up、16项有限且各自绝对值≤1、旧项全零；核对速度误差1.6m/s与高度误差.1m双核的独立数值；重复读取不推进历史、reset清空历史。CPU A28 1env 5轮完成并导出ONNX；Flat smoke与远端GPU smoke/正式启动状态以handoff记录为准。
