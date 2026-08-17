# 环境分组训练

`env_group_ids` 用于在同一个向量化环境中固定划分不同训练子任务。每个组可以拥有
独立的 reset、reward 和 termination，但所有组共享同一个 PPO actor/critic 与优化器。

该实现参考 `BioInnov/rsl_rl_bioin` 的 grouped PPO 数据契约，只迁移纯 RL 分组需要的
环境能力。AMP、多专家蒸馏和 RL/distill 混合 loss 不属于当前迁移范围。

## 1. 分配环境组

在 startup event 中声明组名和比例。比例会自动归一化，并用最大余数法转换为总数
严格等于 `num_envs` 的整数计数。组编号按字典顺序生成，然后在环境维度随机打乱。
业务配置应使用稳定组名选择环境组，不要硬编码依赖字典顺序的整数编号。
以下片段假设已有待修改的 `ManagerBasedRlEnvCfg` 实例 `cfg`；各节代码处于同一个配置
模块上下文中。

```python
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from se3_train.mdp import env_groups, events, rewards

cfg.events["assign_env_groups"] = EventTermCfg(
    func=env_groups.AssignEnvGroups,
    mode="startup",
    params={
        "env_groups": {
            "locomotion": 0.40,
            "recovery": 0.30,
            "rough": 0.30,
        }
    },
)
```

环境创建后会提供：

- `env.env_group_ids`：形状 `[num_envs]` 的 `torch.long`
- `env.env_group_names`：按 group id 排列的组名
- `env.env_group_name_to_id`：当前配置下解析稳定组名的映射
- `env.env_group_counts`：每组环境数量
- `env.num_env_groups`：组数

## 2. 按组配置 reset

`FilteredEventWrapper` 只把当前 reset 的环境中属于目标组的 id 传给 wrapped event。

```python
cfg.events["reset_recovery_root"] = EventTermCfg(
    func=env_groups.FilteredEventWrapper,
    mode="reset",
    params={
        "group_names": ("recovery",),
        "wrapped_term": {
            "func": events.reset_root_state_full,
            "params": {"recovery_prob": 1.0},
        },
    },
)
```

同一组通常需要分别包裹 root reset 和 joint reset。不要再保留一个会覆盖所有环境的
通用 reset event，否则后执行的 event 会覆盖前面的分组状态。

## 3. 按组配置奖励和终止

wrapped reward 在非目标组自动归零，wrapped termination 在非目标组自动变为 false。

```python
cfg.rewards["recovery_upright"] = RewardTermCfg(
    func=env_groups.FilteredRewardWrapper,
    weight=4.0,
    params={
        "group_names": ("recovery",),
        "wrapped_term": {
            "func": rewards.recovery_upright,
            "params": {},
        },
    },
)
```

也兼容参考仓库使用的 filter 写法。新配置建议按组名过滤：

```python
"filter": {"field_name": "env_group_names", "op": "eq", "value": "recovery"}
```

为了兼容已有配置，`group_ids=(1,)` 和
`{"field_name": "env_group_ids", "op": "eq", "value": 1}` 仍然可用；修改
`env_groups` 的排列时，使用整数编号的调用方必须同步更新。

## 4. 是否把 group id 交给网络

可创建独立 observation group：

```python
cfg.observations["env_group"] = ObservationGroupCfg(
    terms={
        "group_id": ObservationTermCfg(func=env_groups.env_group_id_obs),
    },
    concatenate_terms=True,
    enable_corruption=False,
)

# 在现有任务 runner 配置中显式绑定 observation group。
runner_cfg = RslRlOnPolicyRunnerCfg(
    obs_groups={
        "actor": ("actor",),
        "critic": ("critic", "env_group"),
    },
)
```

- critic 建议始终读取 `env_group`，否则不同奖励函数对应的 value target 会互相混淆。
- actor 是否读取取决于任务定义。如果相同状态和 command 在不同组要求不同动作，actor
  必须把 actor 绑定改为 `("actor", "env_group")`；如果希望策略仅凭机器人状态自然
  切换行为，则保留 `("actor",)`。

当前 RSL-RL 会把各组 rollout 合并为同一个 PPO batch。纯 RL 分组不需要修改算法库；
各组奖励总尺度仍应接近，避免某一组长期主导联合梯度。
