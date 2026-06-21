# AGENTS.md — se3_wheel_leg

## 项目概述

轮腿机器人（SerialLeg）强化学习训练框架。基于 MJLab（MuJoCo-Warp GPU 加速）训练，sim2sim 验证。

- 7 个 Python 包：`se3_shared`（训练和验证共享配置）、`se3_train`（MJLab 训练）、`se3_sim2sim`（sim2sim 验证，计划 deprecated）、`se3_deploy`（NX 真机部署 runtime，计划 deprecated）、`se3_tools`（诊断工具）、`se3_jump_to`（跳跃参考轨迹生成）、`se3_flow_match`（Flow Matching，暂不可用待迁移 34D）
- 机器人：6 DOF policy-order（LF0/LB/RF0/RB/L_WHEEL/R_WHEEL）
- 控制方式：腿部关节位置目标 + 轮子速度目标，支持训练端和 sim2sim 共享动作延迟配置

迁移约定：`se3_sim2sim` 和 `se3_deploy` 为历史兼容模块，后续建议迁移到 `https://github.com/3SE-Competitive-Robotics-Team/se3-mono`。Agent 在主动使用 `se3_deploy` 或执行 NX 真机部署/调试命令前，必须先询问主人是否继续使用旧链路；用户已经明确要求时可以继续。

## 术语表 (Glossary)

对话时出现的训练诊断指标统一在此解释。新接触的术语首次出现时应括注中文全称。

### 跳跃诊断指标 (wandb — Jump/*)

- `diag_mean_airborne_vz`：空中平均垂直速度（m/s），越大跳得越高
- `diag_max_airborne_vz`：空中最大垂直速度峰值（m/s）
- `diag_jump_success_rate`：空中且 vz > 成功阈值（1.0 m/s）的 env 比例
- `diag_tilt_airborne_deg`：**空中姿态倾斜角（度）**，腾空阶段机身相对竖直方向的平均倾角。由 `acos(-projected_gravity_z)` 计算，0°=完全直立，>15° 通常表示 RSI 注入或域随机化过强导致空中不稳
- `diag_complete_jumps`：完整走完 grounded→air→landing→grounded 的跳跃次数（每 iter 均值）
- `diag_active_takeoffs`：策略主动蹬腿离地（非 RSI 注入）的次数（每 iter 均值）
- `diag_rsi_takeoffs`：RSI 参考轨迹注入触发的蹬腿离地次数（每 iter 均值）
- `diag_active_success_rate`：主动蹬腿中飞行高度达标的比例
- `diag_active_takeoff_ratio_per_jump_flag`：jump_flag=1 的 env 中采取主动蹬腿的比例
- `leg_contact_walk`：jump_flag=0（行走阶段）的 env 中腿部有地面接触的比例，正常应接近 0。轮足机器人接地点是轮子，腿部触地属于结构性异常姿态
- `leg_contact_jump`：jump_flag=1（跳跃阶段）的 env 中腿部有地面接触的比例，空中应接近 0
- `leg_contact_rsi`：RSI 注入窗口期（前 debounce+2 步）腿部接触比例，排查 RSI 过早触地
- `leg_contact_landing`：跳跃 landing 阶段（stage==2）腿部接触比例，校准落地时序
- `leg_contact_termination`：因非预期腿部触地导致 episode 终止的比例

### 跳跃状态机阶段

- `grounded`：双脚着地 / 行走阶段
- `air`：腾空飞行阶段
- `landing`：落地缓冲阶段

### 状态估计

- `projected_gravity`：机身坐标系下的重力方向投影（3 维），z 分量 -1=完全直立、+1=完全倒置
- `base_ang_vel`：机身角速度
- `base_lin_vel`：机身线速度（critic 特权观测）
- `command`：指令向量 [vx, vy, vyaw, base_height, jump_target_height]

### 训练/仿真术语

- `RSI`（Reference State Initialization）：用参考轨迹的关节位置+角速度+姿态初始化 episode 初始状态，用于跳跃任务的状态分布扩展
- `domain randomization`（域随机化）：训练时随机扰动质量、摩擦、PD 增益等参数，提升策略鲁棒性
- `sim2sim gap`：训练仿真器（MJLab/MuJoCo-Warp）与验证仿真器（原生 MuJoCo）之间的行为差异
- `privileged observation`：训练时 critic 可见但 actor 不可见的观测（如 base 线速度、接触力、base 高度），部署时不可用
- `EFGCL`（External Force Guided Curriculum Learning）：训练早期通过外部物理辅助让策略经历成功状态，再按成功率逐步撤掉辅助。本仓库当前用于跳跃 PreTrain 的空中姿态 spotting，详见 `docs/efgcl_spotting.md`。

## 核心命令

本仓库按功能包直接调用对应 CLI，不使用单一任务运行器工作流。

```bash
# Setup / 检查
uv sync
uv run prek install
uv run python --version
uv run python -c "import mujoco, torch; from importlib.metadata import version; print('mujoco:', mujoco.__version__); print('torch:', torch.__version__); print('rerun-sdk:', version('rerun-sdk'))"
uv run python -c "import torch; print('CUDA 可用:', torch.cuda.is_available()); print('GPU 数量:', torch.cuda.device_count())"

# 代码质量
uv run ruff format .
uv run ruff check . --fix
uv run prek run --all-files

# Smoke 验证（5 轮，不上传 W&B）
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1 --gpu-ids None
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1024

# 训练（需要 NVIDIA GPU + CUDA 12.4+，macOS 不支持训练）
uv run --env-file .env se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1024
uv run --env-file .env se3-train SE3-WheelLegged-Rough --env.scene.num-envs 1024
uv run --env-file .env se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1 --gpu-ids None

# 评估 / sim2sim（纯 MuJoCo CPU + Rerun，macOS 可运行）
uv run se3-sim2sim --max-steps 3000 --course walk-sweep
uv run se3-sim2sim --checkpoint <checkpoint> --max-steps 3000 --course walk-sweep
uv run se3-sim2sim --viewer none --max-steps 200 --print-every 20 --course walk-sweep
uv run se3-sim2sim --checkpoint <checkpoint> --viewer none --max-steps 200 --print-every 20 --course walk-sweep

# 清理
rm -rf logs/ wandb/ replays/ MUJOCO_LOG.TXT
```

跳跃任务专用命令：

```bash
# 跳跃训练
uv run --env-file .env se3-train SE3-WheelLegged-Jump-PreTrain-GRU --env.scene.num-envs 1024
uv run --env-file .env se3-train SE3-WheelLegged-Jump-FineTune-GRU --env.scene.num-envs 1024

# 跳跃参考轨迹生成
uv run se3-jump-to --height 0.4 --output assets/trajectories/jump_0.4m.npz
uv run se3-jump-to --height 0.6 --output assets/trajectories/jump_0.6m.npz

# 跳跃 sim2sim 验证（每隔 5s 触发一次原地跳跃）
uv run se3-sim2sim --checkpoint <ckpt> --jump-interval-s 5.0 --jump-target-height 0.4
```

## 开发流程

**每次修改训练相关代码后，必须先运行 smoke 模式验证环境不会崩溃：**

```bash
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1 --gpu-ids None
# 修改了跳跃相关代码时用这条：
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Jump-FineTune-GRU --env.scene.num-envs 1 --gpu-ids None
```

Smoke 模式特点：
- 仅训练 5 轮
- 不上传到 wandb
- 用于验证代码修改不会导致环境崩溃

确认 smoke 通过后，再运行完整训练。

## 测试策略

- 本实验一般不为单个实现细节添加用于"锁住某些行为"的测试用例
- 不为 checkpoint key、内部排序、私有 helper 等行为补专门回归测试
- 当前以 smoke 训练作为主要有效性验证手段
- 后续需要把 smoke 训练接入 GitHub Actions，但现在不做

## 强制规范

### 第一性原理与长期主义
- **追问根因，不接受临时对策。** 遇到问题先问「为什么会发生」，而不是「怎么绕过去」。能用一行 workaround 掩盖的问题，往往在三步后变成更难修的 bug。
- **解法必须能活过下一个迭代。** 评估任何修改时，问自己：「如果任务数量翻倍、训练时长翻倍、换一块硬件，这个方案还成立吗？」成立才做，不成立就找根本解。
- **不为短期指标牺牲结构。** 奖励权重调参、域随机化范围微调是合理的工程手段；但绕过物理约束、用 magic number 修复奇怪行为、在不理解原因的情况下改超参——这些都是技术债，必须在合并前消除。
- **临时对策的唯一合法形式是带删除日期的注释。** 如果某段代码是权宜之计，必须在注释里写明「为什么是临时的」和「什么条件下删除」，否则视为永久方案，按永久标准审查。

### 错误处理
- Don't fight errors! Whenever you encounter the same error twice, research the web and find 3-5 possible ways to fix it. Then choose the most efficient solution and implement it.

### 工具链
- **禁止直接使用 python/pip**，所有操作必须通过 `uv` 执行
- **禁止直接修改 `pyproject.toml` 添加依赖**，必须使用 `uv add <package>` 命令
- 开发依赖在 `[dependency-groups] dev` 中，使用 `uv sync` 安装
- prek.toml 配置了 pre-commit 钩子：ruff format → ruff check

### 语言
- **所有注释和 docstring 必须使用中文**
- 保留技术术语原文（如 PPO、MuJoCo、MJLab、sim2sim）
- 代码中的变量名、函数名保持英文

### Python 规范
- 目标版本：Python 3.11+，强制使用现代类型标注
- 使用 `list[str]` 而非 `List[str]`，`dict[str, int]` 而非 `Dict[str, int]`
- 使用 `X | None` 而非 `Optional[X]`，`tuple[float, float]` 而非 `Tuple[float, float]`
- 已配置 ruff `UP` 规则自动检查

### Git 规范
- 提交格式：`<feat/chore/fix/del/enh/docs>(<module>): <content>`
- body 用 `- ` 列表
- PR 提交前必须 rebase
- 示例：
  ```
  feat(se3_train): 新增动作延迟配置

  - 训练端按 reset 采样动作延迟
  - sim2sim 支持同一套延迟参数
  - 添加命令行覆盖参数
  ```

### 可视化
- **动态日志/回放**（sim2sim 轨迹、训练曲线、实时诊断）：必须使用 `rerun-sdk`，禁止 matplotlib
- **静态分析小工具**（参数曲线、几何示意图、包络线等一次性绘图脚本）：允许使用 matplotlib，放在 `scripts/` 目录，参考 `scripts/plot_tn_envelope.py`
- 可视化是仓库长期建设的一部分，必须严肃对待

### 机械结构与弹簧建模

SerialLeg 的传动不是简单串联链，实际结构为：
- 所有电机在**机身内部**，通过**共轴链轮**传动
- 膝关节通过**四连杆机构**（驱动杆 AB → 连杆 BC → 小腿上段 CD）传动
- 膝关节安装有**气弹簧**，P₁ 在驱动杆对侧（A 下方），P₂ 在小腿对侧（D 下方）

运行 `scripts/plot_spring_geometry.py` 可生成带真实 MuJoCo FK 的机构示意图，理解四连杆拓扑和弹簧挂点位置关系。详细方案见 `docs/plan/knee_spring_modeling.md`。

## 架构关键点

### 共享配置（核心设计决策）
`se3_shared` 是训练端和 sim2sim 的单一参数来源，覆盖关节语义、默认姿态、PD 增益、动作缩放、任务级观测契约、控制频率和动作延迟。这样设计的原因：
1. 训练端和验证端使用同一套机器人常量
2. 避免动作缩放、默认姿态、观测布局、控制频率或延迟参数漂移导致 sim2sim gap
3. 后续添加 GRU、恢复任务或部署导出时，有明确的 runtime contract

### MJLab 环境结构
```
se3_train/
├── __init__.py      # 调用 tasks.register_all_tasks()
├── robot_cfg.py     # EntityCfg（MJCF + 初始状态）
├── cli.py           # 命令行入口
├── tasks/           # 每个训练任务的最小独立单元
│   ├── rough/       # SE3-WheelLegged-Rough
│   ├── flat/        # SE3-WheelLegged-Flat-GRU
│   ├── stair/       # SE3-WheelLegged-Stair-GRU（爬楼梯）
│   ├── recovery/    # SE3-WheelLegged-Recovery-GRU（倒地自启）
│   ├── jump_pretrain/  # SE3-WheelLegged-Jump-PreTrain-GRU
│   ├── jump_finetune/  # SE3-WheelLegged-Jump-FineTune-GRU
│   └── wheel_dog/   # WheelDog 任务（独立 minidog 机器人）
└── mdp/
    ├── actions.py          # SerialLegDelayedAction — 自定义 6D 动作项
    ├── observations.py     # 34 维 actor 观测（6-joint policy-order）
    ├── rewards.py          # 行走奖励函数
    ├── jump_rewards.py     # 跳跃专属奖励函数
    ├── commands.py         # 速度+高度指令生成器
    ├── jump_commands.py    # 跳跃指令 + 状态机（JumpCommandTerm，8 维）
    ├── curriculums.py      # 行走课程
    ├── jump_curriculums.py # 跳跃课程（jump_prob 动态调度）
    ├── efgcl_stabilizer.py # EFGCL 空中姿态 spotting 辅助（仅 PreTrain）
    ├── jump_traj_tracking.py  # 参考轨迹跟踪奖励
    ├── events.py           # 域随机化事件 + RSI 轨迹注入
    └── terminations.py     # 超时终止 + 腿部接触终止
```

每个 `tasks/<task>/` 目录包含一个训练任务的完整配置和专属 MDP 代码。新增训练任务时
复制最接近的 task 目录，在 `tasks/__init__.py` 注册即可；不要恢复旧的 `se3_train/env_cfg.py`
或 `se3_train/rl_cfg.py` 汇总入口。详细约定见 `docs/task_architecture.md`。

### 跳跃轨迹优化结构
```
se3_jump_to/
├── cli.py      # se3-jump-to 命令行入口，生成 assets/trajectories/*.npz
├── kinematics.py  # SerialLeg FK/IK
└── replay.py   # se3-jump-to-replay 回放入口
```

轨迹文件字段：`base_pos`、`base_vel`、`q_ref`、`q_vel`（关节角速度，用于 RSI）、`t_stance`、`dt` 等。

### 观测空间（34 维 actor）
```
[0:3]   base_ang_vel × 0.25
[3:6]   projected_gravity
[6:11]  commands × (2.0, 0.25, 5.0, 5.0, 5.0)
[11:17] leg_joint_pos [sin(LF), cos(LF), left_active, sin(RF), cos(RF), right_active]
[17:21] leg_joint_vel × 0.25
[21:23] wheel_pos_zero（固定 0，保留兼容槽位）
[23:25] wheel_vel × 0.05
[25:31] last_actions
[31:34] jump_commands  [jump_flag, jump_target_height, jump_phase]
                        jump_phase: 0→1 连续相位，grounded=0，飞行段随轨迹推进
```

critic 在 actor 观测基础上额外包含 base 线速度、轮子接触力和 base height 特权观测。

### 动作空间（6 维）
```
[LF0, LB, RF0, RB, L_WHEEL, R_WHEEL]
腿部：action × 0.25 + default_dof_pos
轮子：action × 20.0 rad/s
```

默认动作延迟配置在 `se3_shared.ActionDelayConfig` 中：名义 5 ms，reset 时在 4-6 ms 间随机采样。训练端和 sim2sim 都应使用同一套配置。

## RL 训练调优 Skill

**触发条件**（满足其一即加载 `.agents/skills/rl-tuning/SKILL.md`）：
- 训练不收敛、奖励停滞或持续下降
- 策略行为不符合预期（跳不高、走路抖动、高度偏低、跪地）
- 需要分析为什么某个能力学不会
- 发现奖励函数 bug 或门控失效
- 需要设计或修改奖励权重、课程、RSI
- 需要判断某次修改是否真正生效
- 需要从 wandb 指标推断根因

**已验证案例库**：`.agents/skills/rl-tuning/references/symptom-to-hypothesis.md`（持续追加）

---

## 远程训练机运维 Skill

涉及远程训练机的任何操作，加载 `.agents/skills/remote-dev-se3/SKILL.md`。

**触发条件**（满足其一即加载）：
- 提到远程训练机、GPU 机器、云机器、wuyinyun、无影云、阿里云、腾讯云
- 需要建立 SSH 连接、代理隧道、反向隧道（用 `boring` 管理，配置在 `~/.boring.toml`）
- 需要启动、停止、监控训练进程
- 需要查看训练日志、wandb 数据
- 需要拉取 checkpoint 到本地
- 需要在远程机器执行 uv sync / git pull
- 询问 tmux 会话管理

各机器特定参数（IP、用户名、SSH 别名、GPU 型号）在 `.agents/skills/remote-dev-se3/machines/` 下对应文件。
当前已注册：`wuyinyun`（无影云 RTX 5880）。

## 环境限制

- **训练**：仅支持 Linux + NVIDIA GPU（CUDA 12.4+）
- **评估/sim2sim**：支持 macOS、Linux、Windows (WSL)
- 推荐环境数：1024（6 DOF 机器人，白天）/ 2048（wuyinyun 夜间）
- 推荐 GPU 显存：8GB+（RTX 3090/4090 训练约 2-3 小时）

## 踩坑记录

### 远程训练进程管理

**问题**：通过 SSH `nohup ... &` 启动训练后，`kill $PID` 杀的是 `uv run` 的 shell wrapper 进程，实际的 Python 训练子进程不会被杀掉。多次"重启"后会出现多个训练进程同时跑，争抢 GPU 且日志混乱。

**正确做法**：
```bash
# 杀进程时必须找到实际 python 子进程
ps aux | grep se3-train | grep python | grep -v grep | awk '{print $2}' | xargs kill

# 或者用 pkill
pkill -f "se3-train"
```

### wandb 保存依赖

**历史问题**：旧版 RSL-RL 的 checkpoint 保存逻辑为 `if self.logger.writer is not None and it % save_interval == 0`。当 wandb 初始化失败（网络超时）时 `writer=None`，会导致**整个训练过程不保存任何 checkpoint**，但训练本身正常跑、日志正常打印，极难察觉。

**解决方案**：
- 当前 `Se3OnPolicyRunner` 已给 W&B 初始化和写入加超时保护，失败后自动降级到本地 TensorBoard，并继续保存 checkpoint
- 远程长时间训练仍应先确认代理可用，避免丢失在线日志与模型上传

### checkpoint 文件名排序

**问题**：`model_900.pt` 在字典序中排在 `model_1000.pt` 之后（"9" > "1"），导致 `ls | sort` 或 `ls -t` 给出错误的"最新 checkpoint"。

**正确做法**：
```bash
# 按数字排序
ls model_*.pt | sort -V

# 或者用 find + stat 按时间
find . -name "model_*.pt" -printf '%T@ %p\n' | sort -n | tail -1
```

### 观测维度不对齐

所有任务统一为 34 维观测。从行走 checkpoint fine-tune 跳跃任务时，需要 `strict=False` 加载，输入层随机初始化，其余权重复用。

## 常见错误手册

`docs/common_mistakes.md` 记录了本仓库开发过程中反复出现的、容易忽视的、或修复成本较高的常见错误。

常见错误文档里放一个具体的条目，以及大概 200-300 字来龙去脉说明，帮助大家快速勘误。

当前条目：
- [1. 关节轴方向：MJCF 中 6 个受控关节的物理约束](docs/common_mistakes.md#1-关节轴方向mjcf-中-6-个受控关节的物理约束)

## 文件结构

```
se3_wheel_leg/
├── pyproject.toml          # 项目配置 + ruff 配置
├── prek.toml               # pre-commit 钩子（ruff format + check）
├── .python-version         # 3.11
├── .env                    # 环境变量（API keys，不提交）
├── .env.example            # 环境变量模板
├── assets/
│   ├── robots/serialleg/mjcf/
│   │   └── serialleg_fidelity_cylinder_wheels.xml
│   ├── base_model/         # 训练完成的基模（_gru.pt / _mlp.pt）
│   └── trajectories/       # 跳跃参考轨迹（jump_0.4m.npz 等）
├── src/
│   ├── se3_flow_match/    # Flow Matching（暂不可用，待迁移 34D）
│   ├── se3_shared/         # 共享机器人、观测和动作延迟配置
│   ├── se3_train/          # MJLab 训练环境
│   ├── se3_sim2sim/        # sim2sim 验证（计划 deprecated，迁移 se3-mono）
│   ├── se3_deploy/         # NX 真机部署 runtime（计划 deprecated，使用前先确认）
│   ├── se3_tools/          # 关节诊断和模型查看工具
│   └── se3_jump_to/        # 跳跃参考轨迹生成
```
