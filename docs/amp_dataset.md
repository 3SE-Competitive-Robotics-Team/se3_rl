# AMP 训练接入（判别器、奖励、数据集加载）

运动输入契约（19 维单帧、38 维相邻帧、20 ms）见 [amp_input.md](amp_input.md)，本文只讲训练侧怎么用。

**判别器实际输入（2026-09-08 起，A2）**：契约帧按 `se3_train.mdp.amp_observations.AMP_DISCRIMINATOR_FIELDS` 切列，
去掉 `left/right_wheel_spin`（专家数据的轮速是打滑/悬空读数，判别器仅凭它就能分开两边），剩 17 维；
窗口 `transition_frames=5`（100 ms），判别器输入 5×17=85 维。env 的 `amp` 观测组与数据集 `dataset_kwargs.fields` 用同一份字段表。

## 1. 组件

| 位置 | 内容 |
|---|---|
| `se3_shared.amp` | 契约：`amp_frame_from_world`、`amp_transition`、`AMP_FEATURE_NAMES` |
| `se3_train.mdp.amp_observations.amp_motion_frame` | 训练端从 mjlab link 运动学取 19 维契约帧再按 `fields` 切列；与复旦导出脚本共用同一个几何函数 |
| `se3_train.amp.AMP` | 判别器扩展，移植自 BioInnov/rsl_rl_bioin：LSGAN + R1、风格奖励、rollout 采窗口 |
| `se3_train.ppo.Se3PPO` | 钩子：写 storage 前加风格奖励，清 storage 前更新判别器，`ext_state_dict["amp"]` 存档 |
| `SE3-WheelLegged-Rough-AMP` | 任务入口：rough 环境 + `amp` 观测组 + AMP 算法配置 |

`amp` 观测组只有一项 `motion_frame`（params `fields`），actor/critic 观测与 ONNX 契约不变。

**只对上台阶列生效**：另有 `amp_mask` 观测组（一项 `terrain`，env 是否在 `ROUGH_AMP_TERRAIN_TYPE_NAMES=("stairs_up",)` 列上），AMP 用它做 fork 里 `enabled_group_mask` 的事：不在列上的 env 不拿风格奖励、其 transition 不进判别器；平地热身期全员在平地列，掩码全 0，判别器不更新、预热计数不走，换列后才开始。env_cfg 旋钮 `amp_terrain_type_names`，空元组即全部 env。

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
transition_frames 5                    判别器窗口帧数（A1 用 2）
reward_weight 15.0                     风格奖励每秒系数；每步加 15 × 0.02 × r = 0.3 r，r∈[0,1]（A1/A2 用 3.0）
reward_warmup_updates 100              前 100 次判别器更新内线性升高
discriminator_updates 2                每轮 PPO 更新后的判别器步数
discriminator_batch_size 4096
discriminator_grad_penalty_weight 10   R1 惩罚，实现里乘 0.5
learning_rate 1e-4
model_cfg [512, 256] + 输入归一化       统计量在 17 维单帧上、专家与策略共用，各帧各自归一化，用专家帧热启动
dataset_kwargs.fields                  = AMP_DISCRIMINATOR_FIELDS（17 个字段名，加载器按名切列）
```

## 4. 训练时看什么

W&B `Loss/amp/*`：`style_reward`（策略 transition 的平均 r，长期贴 0 = 判别器把策略完全分开、风格信号无梯度；
贴 1 = 判别器没学到）、`discriminator_loss`、`expert_score` / `policy_score`（目标 +1 / −1）、`grad_penalty`、`reward_scale`。

## 5. 已知边界

复旦机器人与 SerialLeg 同尺寸，A1 实测 `left/right_wheel_x/z` 分布一致，不需要重定向（导出元数据的 `retargeted_to_serialleg=false` 只是标记）。
A1（19 维、2 帧）在 1073 轮 style_reward 0.60、expert 0.22 / policy −0.25：专家与策略差在 base_velocity_x/z、omega_y、轮心速度和轮速，
轮速差是打滑/悬空造成的非风格差异，故 A2 去掉。若 `style_reward` 长期贴 0，先看哪一列被判别器单独拿来分。
