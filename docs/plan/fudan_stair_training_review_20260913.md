# 复旦轮腿仓库台阶训练对照（2026-09-13）

核对公开 main HEAD `8204e853dfd2ed06d85a322e1a998c3d20a3be2c`，源码隔离克隆在 `.scratch/fudan-review-20260913`，没有运行其代码或安装依赖。本次只读分析，没有修改训练实现或启动实验。

## 证据范围

README称plane为通用运动入口，当前默认配置并非台阶专项。实际检查了plane/logs/wheel_legged下“上台阶1/2/3”、angz+以及混合地形v1/v2/v3实验的环境/配置快照，并追踪共享地形生成器和PPO代码。配置快照说明当时保存的参数，不能凭文件夹名确定哪个模型最好，也不能确认它们的精确续训父子关系。未发现这套台阶流程使用自动500轮平地热身的证据。

## 可确认的差异

### 1. 地形分布并非始终100%上台阶

上台阶2/3配置比例[0,0,0,.5,.5,0]。按terrain.py实际分支，半数负step_height金字塔台阶、半数正step_height金字塔台阶；从中心出生分别对应向外上/下台阶。“上台阶3_angz+添加随机地形，进一步训练v1/v2/v3”为[.3,.2,0,.3,.2,0]，即30%平地、20%坡地、50%两种台阶。当前默认又是另一套比例，不能拿默认代表所有台阶实验。

地块8×8m，台阶踏面0.7m，中央平台4m；生成器高度0.05+0.18*difficulty，10行difficulty=i/10，名义5–21.2cm，随后有5mm高度栅格量化。A27地块9×9m、踏面1.5m、中央平台2m，首次遇障距离和连续接触序列均不同。这些不是同一个地形任务。

来源：[台阶配置](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/logs/wheel_legged/%E4%B8%8A%E5%8F%B0%E9%98%B63/legged_robot_config.py#L54)、[生成器](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/utils/terrain.py#L125)。

### 2. 课程有升有降，完成最高难度后重新随机采样

上台阶3快照：离出生点平面距离>4m升级；未升级且整回合tracking_lin_vel奖励/最大回合时长低于该奖励名义权重40%，降一级。初始难度随机0..5，最高级完成后随机回到已有级别。没有只升不降的永久难度积累。

A27只升不降。推测：偶发探索成功可把环境推到确定性策略尚不能稳定处理的难度，易样本逐渐流失；目前退化结果与此机制相容，尚未通过消融证实。不要把它的阈值原样复制：其tracking奖励被限幅，且分母是最大episode时长；我们的奖励与终止口径不同。

来源：[课程实现](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/logs/wheel_legged/%E4%B8%8A%E5%8F%B0%E9%98%B63/legged_robot.py#L848)。

### 3. 速度是窄核正奖励+宽核有界负项，每项再统一限幅

上台阶3的原始加权速度项：

- 1.5*exp(-e_v²/0.25)
- 1.5*(exp(-e_v²/2.5)-1)

第二项覆盖较大的速度误差，但有界。环境把每一项加权奖励裁剪到[-dt,+dt]后求和（clip_single_reward=1，权重先乘dt），等效每项每秒[-1,+1]，并非只裁剪总奖励。负总奖励允许保留。配置中的1.5、orientation=-10不能当作最终贡献上限。

A22/A27没有同一套每项统一±1限幅；实际奖励函数虽有各自局部上限，也不等价。不能孤立抄某个权重，更不能只看奖励名称。

来源：[奖励汇总](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/logs/wheel_legged/%E4%B8%8A%E5%8F%B0%E9%98%B63/legged_robot.py#L317)、[速度公式](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/logs/wheel_legged/%E4%B8%8A%E5%8F%B0%E9%98%B63/legged_robot.py#L1838)。

### 4. 台阶上仍保留高度跟踪，且使用有界双核

上台阶3权重base_height=1、base_height_enhance=1，原始项为1.5*exp(-1000*e_h²)与exp(-e_h²/0.01)-1，各自再受上述限幅。高度误差是base z相对77个地形采样点平均高度，与我们高度传感器的语义不同。没有按台阶mask关掉高度项。

这支持试验有界高度目标，但不证明给A27加回原二次罚就能好；此前弱高度罚实验也没有稳定解决跟踪退化。高度定义、奖励范围、其他成本必须一并看清。

来源：[高度计算](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/logs/wheel_legged/%E4%B8%8A%E5%8F%B0%E9%98%B63/legged_robot.py#L654)、[高度奖励](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/logs/wheel_legged/%E4%B8%8A%E5%8F%B0%E9%98%B63/legged_robot.py#L1731)。

### 5. Actor显式使用历史速度估计

上台阶3配置ActorCriticSequence，25维本体观测、5帧历史、latent_dim=3。共享实现用encoder估计critic前三维的缩放机身线速度，单独MSE监督；actor输入当前观测+估计速度，PPO路径detach latent。77点地形高度给critic，actor没有直接看地形高度图。

A27是单帧MLP，actor没有机身线速度。这影响辨别空转、受阻与真正前进，属于值得单独验证的结构差异；不是保证成功的证据。引入估计器需要同时维护ONNX runtime历史/导出契约，不能直接往34维输入塞特权真值。

来源：[actor输入](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/rsl_rl/modules/actor_critic_sequence.py#L181)、[监督损失](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/rsl_rl/algorithms/ppo.py#L255)。

### 6. 没有地形分组advantage归一化；频率也不同

共享rollout_storage为整份rollout统一(adv-mean)/(std+eps)，没有按地形分组。它不能为“必须组间归一化”提供证据。

sim.dt=.005、decimation2，即100Hz策略；A27是dt=.005、decimation4，即50Hz。每轮48步与我们24步，物理rollout长度均.48s，但决策频率不同。没有支持把rough物理步长增大的依据。IsaacGym/PhysX与MuJoCo、URDF机构/PD/力矩限制也不同，不应整体复制。

来源：[归一化](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/rsl_rl/storage/rollout_storage.py#L160)、[配置](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/logs/wheel_legged/%E4%B8%8A%E5%8F%B0%E9%98%B63/legged_robot_config.py#L246)。

## 建议的验证顺序

1. 优先从A27热身末model500做配对对照：同样的100%台阶、A22奖励与步数，仅把只升不降改成可回退课程；按最近完整回合通行/跟踪质量设计退级，保留原始统计并防止零速样本误降。比较固定难度成功率、平地指令网格、台阶样本难度分布及随机/确定性动作差异。课程级别均值不能代替能力。
2. 独立验证固定混合分布保留平地样本，作为防止行走/高度能力遗忘的对照；与课程修改分开，避免同时改变多因素。
3. 若考虑采用其奖励，先绘制实际限幅后每项账本和饱和率，再单独比较有界速度/高度项。不能把所有惩罚统一压小而忽视碰撞等约束。
4. 最后单独比较历史MLP/GRU或速度估计器。它更接近参考的部署信息结构，但改动和runtime成本明显高于课程消融。

这些是建议，尚未启动新训练。现有证据最支持检查“热身后能力如何被训练分布与课程保留”，未证明advantage是根因。
