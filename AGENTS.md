# AGENTS.md — se3_wheel_leg

## 项目概述

轮腿机器人（SerialLeg）强化学习训练框架。基于 MJLab（MuJoCo-Warp GPU 加速）训练，sim2sim 验证。

- 3 个 Python 包：`vmc_kinematics`（独立 VMC 模块）、`se3_train`（MJLab 训练）、`se3_sim2sim`（sim2sim 验证）
- 机器人：6 DOF（左腿 lf0/lf1/l_wheel + 右腿 rf0/rf1/r_wheel）
- 控制方式：VMC 极坐标空间（L0 腿长、theta0 腿角）→ 关节力矩

## 核心命令

```bash
# Smoke 模式（验证环境，每次修改训练代码后必须运行）
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1 --gpu-ids None

# 训练（需要 NVIDIA GPU + CUDA 12.4+，macOS 不支持训练）
# 所有训练操作都需要 --env-file .env 以上传指标到 SwanLab
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

# 查看 SwanLab 实验仪表盘
uv run swanlab watch swanlog
```

## 开发流程

**每次修改训练相关代码后，必须先运行 smoke 模式验证环境不会崩溃：**

```bash
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1 --gpu-ids None
```

Smoke 模式特点：
- 仅训练 5 轮
- 不上传到 SwanLab
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
- 保留技术术语原文（如 VMC、theta0、L0、PPO、MuJoCo）
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
  feat(se3_train): 新增 VMC 动作项

  - 实现极坐标空间 PD 控制器
  - 支持动态前馈重力补偿
  - 添加力矩限幅和 NaN 防护
  ```

### 可视化
- **禁止使用 matplotlib 或其他临时可视化方案**
- 必须使用 `rerun-sdk` 进行所有可视化
- 可视化是仓库长期建设的一部分，必须严肃对待

## 架构关键点

### VMC 控制器（核心设计决策）
VMC 运动学作为独立包 `vmc_kinematics` 存在，同时被 `se3_train` 和 `se3_sim2sim` 依赖。这样设计的原因：
1. 训练端（MJLab/torch）和验证端（MuJoCo/numpy）共享同一套运动学公式
2. 避免公式不一致导致的 sim2sim gap
3. 所有函数自动适配 numpy/torch（通过 `hasattr(x, "is_cuda")` 分发）

### MJLab 环境结构
```
se3_train/
├── __init__.py      # register_mjlab_task 注册任务
├── env_cfg.py       # ManagerBasedRlEnvCfg 工厂函数
├── rl_cfg.py        # PPO 超参数（Optuna 调优值）
├── robot_cfg.py     # EntityCfg（MJCF + 初始状态）
├── cli.py           # 命令行入口
└── mdp/
    ├── actions.py   # VMCActionTerm — 自定义动作项
    ├── observations.py  # 27 维观测
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

### 观测空间（27 维）
```
[0:3]   base_ang_vel × 0.25
[3:6]   projected_gravity
[6:9]   commands × (2.0, 0.25, 5.0)
[9:11]  theta0（VMC 腿角）
[11:13] theta0_dot × 0.05
[13:15] L0（VMC 腿长）× 5.0
[15:17] L0_dot × 0.25
[17:19] wheel_pos（取反）
[19:21] wheel_vel × 0.05
[21:27] last_actions
```

### 动作空间（6 维）
```
[theta0_ref_L, l0_ref_L, wheel_vel_ref_L, theta0_ref_R, l0_ref_R, wheel_vel_ref_R]
缩放：theta0 × π, l0 × 0.096 + 0.22, wheel_vel × 25.0
```

## 环境限制

- **训练**：仅支持 Linux + NVIDIA GPU（CUDA 12.4+）
- **评估/sim2sim**：支持 macOS、Linux、Windows (WSL)
- 推荐环境数：1024（6 DOF 机器人，4096 过大）
- 推荐 GPU 显存：8GB+（RTX 3090/4090 训练约 2-3 小时）

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
│   ├── vmc_kinematics/     # 独立 VMC 运动学包
│   ├── se3_train/          # MJLab 训练环境
│   ├── se3_sim2sim/        # sim2sim 验证（从 3se_sim2sim 迁移）
│   └── swanlab_rsl_rl/     # SwanLab RSL-RL 集成插件
```
