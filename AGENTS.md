# AGENTS.md — se3_wheel_leg

## 项目概述

轮腿机器人（SerialLeg）强化学习训练框架。基于 MJLab（MuJoCo-Warp GPU 加速）训练，sim2sim 验证。

- 4 个 Python 包：`se3_shared`（训练和验证共享配置）、`se3_train`（MJLab 训练）、`se3_sim2sim`（sim2sim 验证）、`se3_tools`（诊断工具）
- 机器人：6 DOF（左腿 lf0/lf1/l_wheel + 右腿 rf0/rf1/r_wheel）
- 控制方式：腿部关节位置目标 + 轮子速度目标，支持训练端和 sim2sim 共享动作延迟配置

## 核心命令

```bash
# Smoke 模式（验证环境，每次修改训练代码后必须运行）
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1 --gpu-ids None

# 训练（需要 NVIDIA GPU + CUDA 12.4+，macOS 不支持训练）
# 所有训练操作都需要 --env-file .env 以上传指标到 wandb
uv run --env-file .env se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1024
uv run --env-file .env se3-train SE3-WheelLegged-Rough --env.scene.num-envs 1024

# CPU 训练（仅用于调试，速度极慢）
uv run --env-file .env se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1 --gpu-ids None

# 评估/回放 + sim2sim 验证（纯 MuJoCo CPU + Rerun，macOS 可运行）
uv run se3-sim2sim --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_1999.pt --max-steps 3000

# 无 GUI smoke 验证
uv run se3-sim2sim --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_1999.pt --viewer none --max-steps 200

# 格式化 + lint（必须在提交前执行）
uv run ruff format .
uv run ruff check . --fix
```

## 开发流程

**每次修改训练相关代码后，必须先运行 smoke 模式验证环境不会崩溃：**

```bash
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1 --gpu-ids None
```

Smoke 模式特点：
- 仅训练 5 轮
- 不上传到 wandb
- 用于验证代码修改不会导致环境崩溃

确认 smoke 通过后，再运行完整训练。

## 测试策略

- 本实验一般不为单个实现细节添加用于“锁住某些行为”的测试用例
- 不为 checkpoint key、内部排序、私有 helper 等行为补专门回归测试
- 当前以 smoke 训练作为主要有效性验证手段
- 后续需要把 smoke 训练接入 GitHub Actions，但现在不做

## 强制规范

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
- **禁止使用 matplotlib 或其他临时可视化方案**
- 必须使用 `rerun-sdk` 进行所有可视化
- 可视化是仓库长期建设的一部分，必须严肃对待

## 架构关键点

### 共享配置（核心设计决策）
`se3_shared` 是训练端和 sim2sim 的单一参数来源，覆盖关节语义、默认姿态、PD 增益、动作缩放、观测维度、控制频率和动作延迟。这样设计的原因：
1. 训练端和验证端使用同一套机器人常量
2. 避免动作缩放、默认姿态、控制频率或延迟参数漂移导致 sim2sim gap
3. 后续添加 GRU、恢复任务或部署导出时，有明确的 runtime contract

### MJLab 环境结构
```
se3_train/
├── __init__.py      # register_mjlab_task 注册任务
├── env_cfg.py       # ManagerBasedRlEnvCfg 工厂函数
├── rl_cfg.py        # PPO 超参数（Optuna 调优值）
├── robot_cfg.py     # EntityCfg（MJCF + 初始状态）
├── cli.py           # 命令行入口
└── mdp/
    ├── actions.py   # SerialLegDelayedAction — 自定义 6D 动作项
    ├── observations.py  # 29 维 actor 观测
    ├── rewards.py   # 17 个奖励函数
    ├── commands.py  # 速度+高度指令生成器
    ├── events.py    # 域随机化事件
    └── terminations.py  # 仅有超时终止
```

### 奖励函数与原始实现的对齐
奖励函数经过逐一比对，已与原始 Isaac Gym 实现（`wheel_legged_fzqver.py`）完全对齐。关键公式差异点：
- `base_height`：使用 `exp(-err²/0.05)` 指数衰减（非线性绝对误差）
- `joint_pos_penalty`：使用 L2 范数（`torch.linalg.norm`，非 L2 平方）
- `collision`：惩罚 body 包含 base（indices 0,1,2,4,5），力阈值 0.1N
- `contact_forces`：超出阈值后除以 100 归一化
- `joint_mirror`：按镜像对数求平均（`sum / 2`）

### 观测空间（29 维 actor）
```
[0:3]   base_ang_vel × 0.25
[3:6]   projected_gravity
[6:11]  commands × (2.0, 0.25, 5.0, 5.0, 5.0)
[11:15] leg_joint_pos（相对默认姿态）
[15:19] leg_joint_vel × 0.25
[19:21] wheel_pos
[21:23] wheel_vel × 0.05
[23:29] last_actions
```

critic 在 actor 观测基础上额外包含 base 线速度、轮子接触力和 base height 特权观测。

### 动作空间（6 维）
```
[lf0, lf1, rf0, rf1, l_wheel, r_wheel]
腿部：action × 0.25 + default_dof_pos
轮子：action × 20.0 rad/s
```

默认动作延迟配置在 `se3_shared.ActionDelayConfig` 中：名义 5 ms，reset 时在 4-6 ms 间随机采样。训练端和 sim2sim 都应使用同一套配置。

## 环境限制

- **训练**：仅支持 Linux + NVIDIA GPU（CUDA 12.4+）
- **评估/sim2sim**：支持 macOS、Linux、Windows (WSL)
- 推荐环境数：1024（6 DOF 机器人，4096 过大）
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

**问题**：RSL-RL 的 checkpoint 保存逻辑为 `if self.logger.writer is not None and it % save_interval == 0`。当 wandb 初始化失败（网络超时）时 `writer=None`，导致**整个训练过程不保存任何 checkpoint**，但训练本身正常跑、日志正常打印，极难察觉。

**解决方案**：
- 无外网环境使用 `WANDB_MODE=offline`（writer 不为 None，checkpoint 正常保存，日志事后 `wandb sync` 上传）
- 有代理时设置 `HTTP_PROXY=http://... HTTPS_PROXY=http://...`
- **绝对不要**依赖 wandb 在线模式在网络不稳定的环境跑长时间训练

### checkpoint 文件名排序

**问题**：`model_900.pt` 在字典序中排在 `model_1000.pt` 之后（"9" > "1"），导致 `ls | sort` 或 `ls -t` 给出错误的"最新 checkpoint"。

**正确做法**：
```bash
# 按数字排序
ls model_*.pt | sort -V

# 或者用 find + stat 按时间
find . -name "model_*.pt" -printf '%T@ %p\n' | sort -n | tail -1
```

## 文件结构

```
se3_wheel_leg/
├── pyproject.toml          # 项目配置 + ruff 配置
├── prek.toml               # pre-commit 钩子（ruff format + check）
├── .python-version         # 3.11
├── .env                    # 环境变量（API keys，不提交）
├── .env.example            # 环境变量模板
├── assets/robots/serialleg/mjcf/
│   └── serialleg_fidelity_cylinder_wheels.xml
├── src/
│   ├── se3_shared/         # 共享机器人、观测和动作延迟配置
│   ├── se3_train/          # MJLab 训练环境
│   ├── se3_sim2sim/        # sim2sim 验证
│   ├── se3_tools/          # 关节诊断和模型查看工具
```
