# AMP 运动输入约定

第一版 AMP 只比较运动状态。actor 保持原有 34D 观测；接触力用于标记撞阶、离地与落地，
不作为判别器输入。实现位于 `se3_shared.amp`，训练端读取入口为
`se3_train.mdp.amp_observations.amp_motion_frame`。

## 单帧 19D

所有向量采用右手「前、左、上」机身坐标，轮子顺序为物理左、右。

| 切片 | 内容 | 单位 |
|---|---|---|
| `[0:3]` | 机身坐标下重力单位向量，直立为 `[0,0,-1]` | 无量纲 |
| `[3:6]` | 机身角速度 | rad/s |
| `[6:9]` | 机身 link 原点线速度 | m/s |
| `[9:13]` | 左右轮心相对髋轴位置 `[Lx,Lz,Rx,Rz]` | m |
| `[13:17]` | 上述位置的随体时间导数 `[Lvx,Lvz,Rvx,Rvz]` | m/s |
| `[17:19]` | 左右轮自转速度，正值均表示向前滚动 | rad/s |

轮心和髋轴使用真实 link 原点，不用质心、惯性主轴系或策略目标。
设 R 将机身坐标转到世界坐标，则：

```text
p_relative_body = R.T @ (p_wheel_world - p_hip_world)
v_relative_body = R.T @ (v_wheel_world - v_hip_world) - omega_body × p_relative_body
```

先在三维中完成转换和旋转项扣除，再取 XZ。若所有腿关节固定，机器人只做整体旋转，
轮心的相对速度应为零。世界平移或整体改变航向也不应改变这些运动特征。

SerialLeg 的机身 +X 为训练前进方向，左轮轴为 +Y，右轮轴为 -Y，因此实际关节轮速乘
`[+1,-1]`。此转换只用于 AMP，不能更改原策略的 action/observation 契约。

## 相邻两帧 38D

时间间隔固定为 20 ms，排列为 `[frame(t-20ms), frame(t)]`。
`AmpTransitionHistory` 每个控制步更新一次；首帧没有有效 transition。
在自动 reset 后调用时，传入 `terminated | truncated`，跨 episode 的配对被屏蔽。
非有限状态同样无效；无效配对置零，但仍必须按返回的 `valid` 掩码排除出损失与统计量更新。

历史保存原始物理量。未来判别器对示范和策略使用同一份 19D 均值/标准差，再分别归一化
前、后两帧，不能沿用 actor 缩放，也不能给两种数据分别拟合统计量。
`normalize_amp_frame` 只应用传入统计量；目前没有从单条示范单独拟合归一化参数。

```python
raw_pair, valid = history.update(amp_motion_frame(env), reset_mask=reset_mask)
frames = raw_pair[valid].unflatten(-1, (2, 19))
discriminator_input = normalize_amp_frame(frames, mean19, std19).flatten(-2)
```

## 复旦数据

`scripts/export_fudan_amp_features.py` 从已记录的 qpos/qvel 重建 link 运动学，
使用同一个 `amp_frame_from_world`。仅重建状态，不推进物理仿真。
源机身车头沿 -X，因此完整坐标旋转为 `diag(-1,-1,1)`，不能只反转 X。
转换后按髋轴横向位置确定左右：本模型物理左轮是 `r_wheel_Link`，右轮是 `l_wheel_Link`；
轮速符号由轮轴与统一横向轴的点积确定。

当前导出位于 `assets/amp/fudan_step20_source/`：40 帧 19D 状态与 39 条 38D transition。
元数据明确标记 `retargeted_to_serialleg=false`、`ready_for_discriminator_training=false`。
这些数据验证坐标和输入口径，仍需把源运动重定向到 SerialLeg 的几何与可达范围。

本阶段没有添加判别器、AMP reward、经验池或 PPO 更新逻辑，也没有启用新的训练任务。
