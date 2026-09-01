# 训练任务架构

`src/se3_train/tasks/` 是训练任务的唯一入口。每个子目录表示一个完整 task，目录内收拢该任务相关的环境配置、RL 配置、观测、奖励、指令、课程、事件和终止条件。

旧的 `src/se3_train/env_cfg.py` 和 `src/se3_train/rl_cfg.py` 汇总入口已经删除，不再恢复。新增实验必须进入 `tasks/<task>/`。

## 当前任务

| 目录 | task id | 用途 |
| --- | --- | --- |
| `rough/` | `SE3-WheelLegged-Rough` | 崎岖地形行走任务 |
| `flat/` | `SE3-WheelLegged-Flat-GRU` | 平地行走 GRU 基模 |
| `recovery_discovery/` | `SE3-WheelLegged-Recovery-Discovery-GRU` / `SE3-WheelLegged-Recovery-Discovery-MLP` / `SE3-WheelLegged-Recovery-Discovery-History-MLP` / `SE3-WheelLegged-Recovery-Loco-Grouped-MLP` / `SE3-WheelLegged-Recovery-Loco-Grouped-Gentle-MLP` / `SE3-WheelLegged-Recovery-Loco-Grouped-Teacher-MLP` / `SE3-WheelLegged-Recovery-Loco-Grouped-Gentle-Teacher-MLP` / `SE3-WheelLegged-Recovery-Loco-Grouped-Gentle-TorqueAssist-MLP` / `SE3-WheelLegged-Recovery-Discovery-Ungrouped-MLP` | 唯一倒地自启任务；各入口共享奖励、课程和 PPO 配置，Grouped 使用 loco/recover 分组与五帧历史观测；Teacher 入口变换动作，TorqueAssist 入口保持策略动作原样并按当前机身倾角施加外部扭矩；两种引导均在 play/eval 关闭 |
| `stair/` | `SE3-WheelLegged-Stair-GRU` | CTBC 倒金字塔台阶任务，从 stair checkpoint warm start |
| `jump_pretrain/` | `SE3-WheelLegged-Jump-PreTrain-GRU` | 跳跃预训练阶段，包含 EFGCL 辅助和参考轨迹约束 |
| `jump_finetune/` | `SE3-WheelLegged-Jump-FineTune-GRU` | 跳跃 FineTune 阶段，从 PreTrain checkpoint 继续训练 |

阶段命名写在 task id 里。跳跃任务目前只有 `PreTrain` 和 `FineTune` 两个正式入口。

`tasks/recovery/` 仅保留 Recovery-Discovery 使用的环境基配置、奖励、事件和课程实现，
不注册独立 task；所有倒地自启训练必须从 `recovery_discovery/` 的三个正式入口启动。

## 台阶任务

`stair/` 是当前台阶训练入口，注册 `SE3-WheelLegged-Stair-GRU`，并保留 `SE3-WheelLegged-Stair-GRU-TrainView` 作为历史 watch/play 别名。正式远程训练和本地值守脚本默认使用原始 task id；只有需要兼容旧 watch 流程时才显式使用 `*-TrainView`。

台阶任务的核心差异集中在 `src/se3_train/tasks/stair/`：

- `env_cfg.py` 使用沿世界系 +x 上升的直线台阶地形 `BoxForwardStairsTerrainCfg`，当前训练 MJCF 为真实闭链 `serialleg_closed_chain_v3_train_obb_trim.xml`。
- `state.py`、`events.py` 和 `observations.py` 管理 CTBC 前馈状态机；actor 仍为 34 维观测，最后 3 维扩展槽在台阶任务中输出 CTBC 左右摆动相位和触发位。
- `rewards.py`、`curriculums.py` 提供台阶爬升奖励、地形等级课程和诊断项。
- `env_cfg.py` 同时接入 recovery replay 状态缓存，用于提升台阶训练中跌倒后的恢复覆盖率。

台阶任务的远程值守使用 `./scripts/run_sim2x.sh` 启动 native MuJoCo/Viser，并在
`Models` 页签选择对应 experiment、run id 和 ONNX。checkpoint 来源、远程连接和本地缓存
目录由当前 machine profile 决定，不属于任务架构契约。

## 单个 task 的目录结构

```text
tasks/<task_name>/
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

`env_cfg.py` 可以复用更基础任务的配置，再覆盖当前任务的差异。例如 `jump_finetune` 基于 `jump_pretrain`，移除 EFGCL 辅助并调整 FineTune 阶段的奖励和课程。

`observations.py`、`rewards.py`、`commands.py`、`curriculums.py`、`terminations.py` 和 `events.py` 可以转发共享实现，也可以放本任务独有实现。原则是从 task 目录能看出该任务实际依赖了哪些 MDP 代码。

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
   cp -R src/se3_train/tasks/jump_finetune src/se3_train/tasks/jump_high
   ```

2. 修改 `jump_high/__init__.py`：

   - `TASK_ID` 使用明确阶段名，例如 `SE3-WheelLegged-JumpHigh-FineTune-GRU`
   - docstring 写清楚观测维度、训练阶段和用途
   - runner 继续使用 `Se3WarmStartRunner`，除非新任务确实不需要 warm start 逻辑

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
from se3_train.tasks import (
    flat,
    jump_finetune,
    jump_pretrain,
    recovery_discovery,
    rough,
    stair,
)

for module in (
    rough,
    flat,
    recovery_discovery,
    stair,
    jump_pretrain,
    jump_finetune,
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
