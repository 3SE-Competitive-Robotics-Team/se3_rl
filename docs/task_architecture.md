# 训练任务架构

`src/se3_train/tasks/` 是训练任务的唯一入口。每个子目录表示一个完整 task，目录内收拢该任务相关的环境配置、RL 配置、观测、奖励、指令、课程、事件和终止条件。

旧的 `src/se3_train/env_cfg.py` 和 `src/se3_train/rl_cfg.py` 汇总入口已经删除，不再恢复。新增实验必须进入 `tasks/<task>/`。

## 当前任务

| 目录 | task id | 用途 |
| --- | --- | --- |
| `rough/` | `SE3-WheelLegged-Rough` / `SE3-WheelLegged-Rough-GRU` / `SE3-WheelLegged-Rough-StairEval` | 冻结的 Flat 基线 + 一层薄覆盖（2026-09-13 重写）。地形 preset、升降级课程 `terrain_levels_vel`、出块截断 `terrain_edge_reached` / `out_of_terrain_bounds`、critic 高度扫描 `height_scan` 全部用 mjlab 官方件；自己的部分只有高度指令的地形感知下限与分列指令覆盖（`commands.py`）、台阶进度/支撑奖励与按列奖励包装（`stair_rewards.py` / `rewards.py`）、平地热身（`curriculums.py`）。默认定价取 A15，见 `docs/plan/stair_training_wandb_review_20260913.md` 与 `rough_official_base_survey_20260913.md`。`-GRU` 入口（M16，2026-09-20）与 MLP 共用同一份 env_cfg，只把 actor/critic 换成单层 GRU 512、rollout 仍 24 步（`rough/rl_cfg.py`）。历史实验用对应 Git commit 复现。 |
| `rough/`（临时） | `SE3-WheelLegged-Rough-Exp-StairSpeedCap` / `SE3-WheelLegged-Rough-Exp-HeightWindow` / `SE3-WheelLegged-Rough-Exp-FudanReward` / `SE3-WheelLegged-Rough-Exp-FudanRewardOri10` / `SE3-WheelLegged-Rough-Exp-FudanRewardOri10Std15` / `SE3-WheelLegged-Rough-Exp-FudanRewardOri10Std15NoWarmup` | M26/M27 与 M25 并发对照，各只翻一个开关（见 `docs/plan/m25_m27_pushfix_speedcap_heightwindow_20260925.md`）；M28 整张奖励表换成复旦 v3（见 `docs/plan/m28_fudan_v3_reward_20260926.md`）；M29 在 M28 上把姿态罚改回 −10、高度参考改成复旦格点口径（见 `docs/plan/m29_fudan_v3_ori10_20260926.md`）；M30 在 M29 上把 actor 初始 std 改成 1.5（见 `docs/plan/m30_fudan_v3_ori10_std15_20260926.md`）；M31 在 M30 上去掉平地热身（见 `docs/plan/m31_fudan_v3_ori10_std15_nowarmup_20260926.md`）；对照结束后删除 |
| `flat/` | `SE3-WheelLegged-Flat-MLP` | 仅保留 D11 单帧 MLP 基线；共享配置仍供其他任务复用 |
| `stair/` | `SE3-WheelLegged-Stair-GRU` | CTBC 倒金字塔台阶任务，从 stair checkpoint warm start |
| `jump_pretrain/` | `SE3-WheelLegged-Jump-PreTrain-GRU` | 跳跃预训练阶段，包含 EFGCL 辅助和参考轨迹约束 |
| `jump_finetune/` | `SE3-WheelLegged-Jump-FineTune-GRU` | 跳跃 FineTune 阶段，从 PreTrain checkpoint 继续训练 |

**Flat 基线已于 2026-09-06 冻结**，取 D11 的配置（W&B `mher9vfk`，commit `236666c`）：不带任何命令行覆盖直接跑
`SE3-WheelLegged-Flat-MLP` 即可复现。除已合并的奖励与动作改动外，`num_steps_per_env` 由 64 改为 24（D 系列
全部实验的实际取值），`max_iterations` 由 5000 改为 3500（逐轮曲线显示
有信息量的窗口在 3500 轮以内），`randomize_com` 由 ±20 mm 收到 ±5 mm。全部数值由
`tests/test_flat_baseline.py` 逐项守护，改基线必须同步改该测试并在提交信息里写明对照实验编号。

**2026-09-25 修正推力课程的轮次换算**：推力课程按 PPO 轮次分档，但一直没传 `steps_per_policy_iter`，
落到默认 64；rollout 改成 24 之后课程时钟慢了 2.67 倍，首档（2000 轮、±0.3 m/s）实际要到 5333 轮才出现，
所以 D11 冻结的 Flat 基线与之后的 Rough 5000 轮训练都**从未推过**。修正后 Flat 在第 2000 轮开始推、Rough 在第 5000 轮升到
±0.5 m/s。复现修正前的 D11 用 commit `236666c`。runner 启动时会校验所有按轮次推进的课程/事件与
`num_steps_per_env` 一致，不一致直接报错。

Flat 基线已合并 D2–D8 的已验证改动。2026-09-13 清理后，GRU、History-MLP 和全部 Flat-Exp 注册入口已删除；复现历史实验请使用对应 Git commit。

阶段命名写在 task id 里。跳跃任务目前只有 `PreTrain` 和 `FineTune` 两个正式入口。

倒地自启的 Recovery-Discovery 任务族（含 `tasks/recovery/` 共享实现）已于 2026-09-25 删除；
复现历史实验请使用对应 Git commit。stair 的 recovery rehearsal 仍复用 `mdp/` 下的 recovery
复位、奖励与状态掩码。

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
    rough,
    stair,
)

for module in (
    rough,
    flat,
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
