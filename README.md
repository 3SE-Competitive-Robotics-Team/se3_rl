# SE3 Wheel Leg

SerialLeg 轮腿机器人（6 DOF）的强化学习训练与 sim2sim 验证框架。训练端用 MJLab（MuJoCo-Warp GPU 加速）跑 PPO，验证端用标准 MuJoCo CPU，两端通过 `se3_shared` 共享机器人常量、观测维度和动作延迟配置，保证 sim2sim gap 可控。

> 当前 sim2sim 入口为 [`se3-sim2x`](https://github.com/3SE-Competitive-Robotics-Team/se3-sim2x)。

## 前置条件

- Python 3.11（`>=3.11,<3.12`）
- [uv](https://github.com/astral-sh/uv) 包管理器
- 训练：Linux + NVIDIA GPU + CUDA 12.4+
- sim2sim / 诊断工具：macOS、Linux、Windows（WSL）均可

## 项目结构

```text
src/
├── se3_shared/     # 训练和 sim2sim 共享配置，包含关节语义、PD 增益、动作缩放、延迟参数
├── se3_train/      # MJLab 训练环境，含 MDP（奖励、观测、事件）和 PPO 配置
├── se3_tools/      # 关节方向和默认姿态诊断工具
├── se3_jump_to/    # 跳跃参考轨迹生成与回放
└── se3_flow_match/ # Flow Matching 蒸馏（暂不可用，待迁移 34D 观测）

submodules/
└── se3-sim2x/      # 共用 ONNX runtime 与 MuJoCo/Viser adapter
```

机器人模型位于 `assets/robots/serialleg/`，训练产物默认写入
`logs/rsl_rl/<task_name>/<run_id>/`。

新的部署入口是带 `se3.meta.v1` 的单个 ONNX artifact。submodule 中的 `se3_runtime` 已统一处理
MLP、History-MLP、GRU 的 observation/state/action 语义；完整调用约定见
[ONNX metadata runtime](docs/onnx_runtime.md)。

## 环境准备

```bash
git submodule update --init --recursive
uv sync
uv run prek install
```

`uv run prek install` 把提交前检查接入 Git，之后每次 `git commit` 自动运行 ruff format 和 ruff check。

## Quick Start

本仓库按功能包直接调用对应 CLI，避免把训练、验证、诊断和工具脚本塞进单一任务入口。

```bash
uv sync
uv run prek install
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1 --gpu-ids None
uv run --env-file .env se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1024
./scripts/run_sim2x.sh
```

训练指标按 Task 分别上传到同名 W&B Project，结构为 `Entity → Task Project → Run`。

## 常用命令

### Setup & 检查

```bash
uv sync
uv run prek install
uv run python --version
uv run python -c "import mujoco, torch; from importlib.metadata import version; print('mujoco:', mujoco.__version__); print('torch:', torch.__version__); print('viser:', version('viser'))"
uv run python -c "import torch; print('CUDA 可用:', torch.cuda.is_available()); print('GPU 数量:', torch.cuda.device_count())"
```

### Smoke 验证

修改训练代码后先跑这个，5 轮训练验证环境不崩溃，不上传 W&B。

```bash
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1 --gpu-ids None
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1024
```

### 训练

需要 `.env` 以上传指标到 W&B。正式训练前先确认 `.env` 存在。

```bash
uv run --env-file .env se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1024
uv run --env-file .env se3-train SE3-WheelLegged-Rough --env.scene.num-envs 1024
uv run --env-file .env se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1 --gpu-ids None
```

### Sim2sim Viser

```bash
./scripts/run_sim2x.sh
```

启动后访问终端打印的地址，通常为 `http://127.0.0.1:8080/`。服务持续扫描
`logs/rsl_rl/<task_name>/<run_id>/onnx/*.onnx`，可在 Viser 的 `Models` 页签依次选择
Task、Run ID 和 ONNX；训练新导出的模型会自动出现在列表中。

完整用法和 artifact 要求见 [ONNX metadata runtime](docs/onnx_runtime.md)。

### 代码质量

```bash
uv run ruff format .
uv run ruff check . --fix
uv run prek run --all-files
```

### 清理

```bash
rm -rf logs/ wandb/ replays/ MUJOCO_LOG.TXT
```

## 文档

- [文档索引](docs/README.md)
- [新手入门](docs/how_to_start.md)
- [训练指南](docs/train.md)
- [训练性能记录](docs/perf.md)
- [训练任务架构](docs/task_architecture.md)
- [ONNX metadata runtime](docs/onnx_runtime.md)
- 远程机器运维：先读 [remote-dev-se3 公用流程](.agents/skills/remote-dev-se3/SKILL.md)，再选择对应 machine profile
- [MoE 多速度域方案](docs/plan/moe_multi_speed.md)
- [膝关节弹簧建模方案](docs/plan/knee_spring_modeling.md)
- [碰撞模型优化记录](docs/todo/collision_model_optimization.md)

## 注意事项

- 所有 Python 命令通过 `uv` 执行，不直接用 `python` 或 `pip`。
- `.env`、`logs/`、`wandb/` 和本地回放文件不应提交。
- 训练 checkpoint 较大，分享仓库时单独传 `model_*.pt`，不要提交到 Git。
- W&B 初始化或运行期写入失败时，runner 会自动降级到本地 TensorBoard，训练和 checkpoint 保存继续进行；远程长训仍建议先确保代理可用，避免丢在线日志。
