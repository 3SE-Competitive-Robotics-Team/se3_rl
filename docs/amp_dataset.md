# AMP 训练接入（判别器、奖励、数据集加载）

运动输入契约（19 维单帧、38 维相邻帧、20 ms）见 [amp_input.md](amp_input.md)，本文只讲训练侧怎么用。

## 1. 组件

| 位置 | 内容 |
|---|---|
| `se3_shared.amp` | 契约：`amp_frame_from_world`、`amp_transition`、`AMP_FEATURE_NAMES` |
| `se3_train.mdp.amp_observations.amp_motion_frame` | 训练端从 mjlab link 运动学取 19 维帧；与复旦导出脚本共用同一个几何函数 |
| `se3_train.amp.AMP` | 判别器扩展，移植自 BioInnov/rsl_rl_bioin：LSGAN + R1、风格奖励、rollout 采窗口 |
| `se3_train.ppo.Se3PPO` | 钩子：写 storage 前加风格奖励，清 storage 前更新判别器，`ext_state_dict["amp"]` 存档 |
| `SE3-WheelLegged-Rough-AMP` | 任务入口：rough 环境 + `amp` 观测组 + AMP 算法配置 |

`amp` 观测组只有一项 `motion_frame`，actor/critic 观测与 ONNX 契约不变。

## 2. 数据集

`dataset_root`（默认 `assets/amp/fudan_stairs20_20260907`，环境变量 `SE3_AMP_DATASET_ROOT` 覆盖）可以是：

- 含 `source_dataset.npz` 的目录（`scripts/package_fudan_amp_dataset.py` 产物，`frames [N,19]` + `frame_offsets`）；
- 含若干 `<sample>/amp/source_features.npz` 的目录（`scripts/export_fudan_amp_features.py` 产物）；
- 单个上述 `.npz`。

加载规则：按 `frame_offsets` 切段，不跨段配对；契约固定 20 ms，训练 `step_dt` 必须相等，不重采样；
`mirror_augmentation=True` 追加左右镜像段（重力 y 取反，角速度 x、z 取反，线速度 y 取反，左右轮位置/速度/自转互换）。
导出元数据里 `ready_for_discriminator_training=false` 时启动只打印提示，不阻断。

## 3. 配置（`tasks/rough/rl_cfg.py` 的 `amp_cfg_dict`，默认值照 kyber g1 velocity AMP）

```
reward_weight 3.0                      风格奖励每秒系数；每步加 3.0 × 0.02 × r，r∈[0,1]
reward_warmup_updates 100              前 100 次判别器更新内线性升高
discriminator_updates 2                每轮 PPO 更新后的判别器步数
discriminator_batch_size 4096
discriminator_grad_penalty_weight 10   R1 惩罚，实现里乘 0.5
learning_rate 1e-4
model_cfg [512, 256] + 输入归一化       统计量在 19 维单帧上、专家与策略共用，两帧各自归一化，用专家帧热启动
```

## 4. 训练时看什么

W&B `Loss/amp/*`：`style_reward`（策略 transition 的平均 r，长期贴 0 = 判别器把策略完全分开、风格信号无梯度；
贴 1 = 判别器没学到）、`discriminator_loss`、`expert_score` / `policy_score`（目标 +1 / −1）、`grad_penalty`、`reward_scale`。

## 5. 已知边界

复旦机器人与 SerialLeg 的髋轴到轮心几何不同，`left/right_wheel_x/z` 的分布本身就有差异，判别器可能仅凭这一点区分。
导出元数据已标 `retargeted_to_serialleg=false`。若 `style_reward` 长期贴 0，先做重定向再训。
