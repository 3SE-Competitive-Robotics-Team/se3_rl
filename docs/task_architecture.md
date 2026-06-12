# 训练任务架构

`src/se3_train/tasks/` 是训练任务的唯一入口。新机器人从现在开始采用 `tasks/<robot>/<task>/` 两层结构，目录内收拢该任务相关的环境配置、RL 配置、观测、奖励、指令、课程、事件和终止条件。

旧的 `src/se3_train/env_cfg.py` 和 `src/se3_train/rl_cfg.py` 汇总入口已经删除，不再恢复。新增机器人实验必须进入 `tasks/<robot>/<task>/`。

现有 SerialLeg / WheelLeg 任务暂时保留在一级目录（例如 `flat/`、`rough/`、`jump_pretrain/`）。原因是这些任务已经被训练、FlowMatch 和 sim2sim 入口引用，一次性迁移会带来较大的注册名、导入路径和历史 checkpoint 回归风险。后续只有在专门做迁移时，才把这些目录移动到 `tasks/wheel_leg/`，本次 WheelDog 新任务不扩大这部分改动。

## 当前任务

| 目录 | task id | 用途 |
| --- | --- | --- |
| `rough/` | `SE3-WheelLegged-Rough` | 崎岖地形行走任务 |
| `flat/` | `SE3-WheelLegged-Flat-GRU` | 平地行走 GRU 基模 |
| `recovery/` | `SE3-WheelLegged-Recovery-GRU` | 倒地自启任务，复用平地任务结构并使用全角度 recovery reset |
| `flow_match/wheel/` | `SE3-WheelLegged-FlowMatch-Wheel-GRU` | FlowMatch `WHEEL=0` 单标签平地轮式能力任务 |
| `flow_match/gait_stage1/` | `SE3-WheelLegged-FlowMatch-Gait-Stage1-GRU` | FlowMatch `GAIT=1` 平地 `0-1.05m/s` 基础步态任务 |
| `flow_match/gait_stage2/` | `SE3-WheelLegged-FlowMatch-Gait-Stage2-GRU` | FlowMatch `GAIT=1` 低矮随机地形和最高 `8cm` 台阶任务 |
| `flow_match/gait_stage3/` | `SE3-WheelLegged-FlowMatch-Gait-Stage3-GRU` | FlowMatch `GAIT=1` `8-24cm` 上台阶专项任务 |
| `flow_match/wheel_leg/` | `SE3-WheelLegged-FlowMatch-WheelLeg-GRU` | FlowMatch `WHEEL_LEG=2` 单标签能力任务 |
| `flow_match/gait_wheel/` | `SE3-WheelLegged-FlowMatch-GaitWheel-GRU` | FlowMatch `GAIT_WHEEL=3` 单标签能力任务 |
| `flow_match/jump/` | `SE3-WheelLegged-FlowMatch-Jump-GRU` | FlowMatch `JUMP=4` 单标签能力任务 |
| `jump_pretrain/` | `SE3-WheelLegged-Jump-PreTrain-GRU` | 跳跃预训练阶段，包含 EFGCL 辅助和参考轨迹约束 |
| `jump_finetune/` | `SE3-WheelLegged-Jump-FineTune-GRU` | 跳跃 FineTune 阶段，从 PreTrain checkpoint 继续训练 |
| `wheel_dog/flat/` | `SE3-WheelDog-Flat-GRU` | 20kg、16DOF 四足轮腿 WheelDog 平地速度跟随任务，M20 风格前后反向屈膝，目标 base_link 高度 `0.32m`，速度课程最终到前后 `±2.0m/s`、左右 `±0.6m/s` |

阶段命名写在 task id 里。跳跃任务目前只有 `PreTrain` 和 `FineTune` 两个正式入口。Recovery 目前只有 `Recovery-GRU` 一个正式入口，不注册 recovery-stand 或 stair/NX 实验入口。

FlowMatch 任务用于先训练独立语义标签能力，再作为 FlowMatch 蒸馏源。正式语义标签为 `WHEEL=0`、`GAIT=1`、`WHEEL_LEG=2`、`GAIT_WHEEL=3`、`JUMP=4`。每个 FlowMatch 单标签 task 都固定自己的 `TaskMode`，关闭 episode 内模式切换；其中 `GAIT` 使用 Stage1/Stage2/Stage3 三段正式课程，阶段交接通过 CLI 显式传 `--agent.resume True --agent.load-run <上一阶段run> --agent.load-checkpoint <checkpoint>`，不在配置里写死 checkpoint。

### `WHEEL_LEG` 与 `GAIT_WHEEL` 语义边界

`WHEEL_LEG` 是轮式 locomotion 的腿增强版：轮子是主要推进源，腿的职责是抬升机身、越障、解卡和维持轮地接触。判断标准是，即使腿不形成稳定左右交替步态，机器人仍应主要靠轮速完成前进。

`GAIT_WHEEL` 是步态 locomotion 的轮增强版：腿部左右交替摆动/蹬地形成主要步态节律，轮子在支撑和滑行阶段继续贡献前进速度与转向修正，目标行为类似“滑旱冰”。判断标准是，去掉交替步态后行为不应退化成纯轮式巡航；去掉轮子主动推进后也应明显损失滑行效率。

reward 函数可以集中复用，但 reward 表属于具体 task。新增或调整语义任务时，应在对应 `tasks/<robot>/<task>/env_cfg.py` 中显式列出该任务的 reward term、weight 和 params，避免把语义藏进共享 mode-gated 大表。旧 WheelLeg 目录在迁移前仍按现有一级路径执行。

## 单个 task 的目录结构

```text
tasks/<robot_name>/<task_name>/
├── __init__.py       # task_id、register()、runner_cls
├── env_cfg.py        # 场景、观测、动作、指令、奖励、终止、课程、事件
├── rl_cfg.py         # PPO / GRU / checkpoint / logger
├── observations.py   # 本任务 actor/critic 观测项
├── rewards.py        # 本任务奖励函数
├── commands.py       # 本任务指令项
├── curriculums.py    # 本任务课程函数
├── terminations.py   # 本任务终止条件
└── events.py         # 本任务 reset / startup 事件
```

旧 WheelLeg 任务在迁移前仍使用 `tasks/<task_name>/`，但新增机器人任务不再使用一级目录。

`env_cfg.py` 可以复用更基础任务的配置，再覆盖当前任务的差异。例如 `jump_finetune` 基于 `jump_pretrain`，移除 EFGCL 辅助并调整 FineTune 阶段的奖励和课程。

`observations.py`、`rewards.py`、`commands.py`、`curriculums.py`、`terminations.py` 和 `events.py` 可以转发共享实现，也可以放本任务独有实现。原则是从 task 目录能看出该任务实际依赖了哪些 MDP 代码。

需要自定义 runner 时，应继承 `Se3OnPolicyRunner` 或 `Se3WarmStartRunner`，保留 W&B 失败时降级 TensorBoard 并继续保存 checkpoint 的保护。

## 注册流程

`src/se3_train/__init__.py` 调用 `se3_train.tasks.register_all_tasks()`。`tasks/__init__.py` 只负责导入并注册当前正式任务。

单个 task 的 `__init__.py` 负责：

```python
TASK_ID = "SE3-WheelLegged-Example-GRU"


def register() -> None:
    """注册 Example 任务。"""
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=rl_cfg(),
        runner_cls=Se3WarmStartRunner,
    )
```

没有注册到 `tasks/__init__.py` 的目录不属于正式训练入口。

## 新增实验

1. 复制最接近的目录，例如：

   ```bash
   cp -R src/se3_train/tasks/wheel_dog/flat src/se3_train/tasks/wheel_dog/rough
   ```

2. 修改新 task 的 `__init__.py`：

   - `TASK_ID` 使用明确阶段名，例如 `SE3-WheelDog-Rough-GRU`
   - docstring 写清楚观测维度、训练阶段和用途
   - runner 继续使用当前机器人任务已经验证过的 runner，除非新任务确实需要更换

3. 在 `env_cfg.py` 里改任务差异：

   - 观测维度和观测项
   - command 分布
   - reward 项和权重
   - termination 条件
   - curriculum 调度
   - reset / startup events

4. 在 `rl_cfg.py` 里改训练差异：

   - GRU / MLP 结构
   - `max_iterations`
   - `save_interval`
   - `resume`
   - `load_run`
   - `load_checkpoint`
   - logger 配置

5. 在 `tasks/__init__.py` 导入并调用 `register()`。

6. 更新本文档的“当前任务”表。实验还不准备作为正式入口时，不注册到 `tasks/__init__.py`。

## 验证

修改训练任务后至少运行：

```bash
uv run ruff format --check src/se3_train
uv run ruff check src/se3_train
git diff --check
```

然后做 task 构造 smoke：

```bash
uv run python - <<'PY'
from se3_train.tasks import flat, jump_finetune, jump_pretrain, recovery, rough
from se3_train.tasks.flow_match import (
    gait_stage1,
    gait_stage2,
    gait_stage3,
    gait_wheel,
    jump,
    wheel,
    wheel_leg,
)
from se3_train.tasks.wheel_dog import flat as wheel_dog_flat

for module in (
    rough,
    flat,
    wheel,
    gait_stage1,
    gait_stage2,
    gait_stage3,
    wheel_leg,
    gait_wheel,
    jump,
    recovery,
    jump_pretrain,
    jump_finetune,
    wheel_dog_flat,
):
    cfg = module.env_cfg(play=True)
    rl = module.rl_cfg(smoke=True)
    print(module.TASK_ID, len(cfg.observations["actor"].terms), rl.max_iterations)
PY
```

改了跳跃 task 时，再跑对应 CLI smoke：

```bash
uv run se3-train SE3-WheelLegged-Jump-FineTune-GRU \
  --env.scene.num-envs 1 \
  --gpu-ids None \
  --agent.max-iterations 1 \
  --agent.logger tensorboard \
  --agent.resume False
```
