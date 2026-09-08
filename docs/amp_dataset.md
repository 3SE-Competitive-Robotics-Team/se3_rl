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

## 2. 数据集（照 kyber 的 MotionLoader / amp_dataset_factory）

| 位置 | 内容 |
|---|---|
| `se3_train.motion_loader.MotionLoader` | 读 `se3.amp.pkl.v1`：校验 format/motion_contract/feature_names，按 `simulation_dt` 线性重采样，镜像增广，可选字段子集，产出 `format` / `metadata` / `get_dataset_dict()` |
| `se3_train.amp_dataset_factory.build_amp_dataset` | 解析字段 → MotionLoader → 校验字段顺序 → 校验 env `amp` 观测组维数 |

`dataset_root`（默认 `assets/amp/fudan_stairs20_20260907/amp_training.pkl`，环境变量 `SE3_AMP_DATASET_ROOT` 覆盖）
可以是单个 `.pkl`，或按 `dataset_glob="*.pkl"` 匹配的目录。文件由 `scripts/export_fudan_amp_pkl.py` 产出，payload：

```
format="se3.amp.pkl.v1", motion_contract="se3.amp.motion.v1", fps, dt, frame_dim=19, transition_dim=38,
feature_names[19], sequences: list[np.ndarray[T,19]], transitions: list[[T-1,38]], lengths, time_s, source_time_s,
annotations: list[dict]（台阶高、指令、接触力标注，不进判别器）, retargeted_to_serialleg
```

加载规则：每条 sequence 单独成段，不跨段配对；fps 与训练 `1/step_dt` 不同时线性重采样（与 kyber 一致）；
`mirror_augmentation=True` 时镜像段紧跟原段（重力 y 取反，角速度 x、z 取反，线速度 y 取反，左右轮位置/速度/自转互换）；
`retargeted_to_serialleg=false` 只打印提示，不阻断。

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
