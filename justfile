# justfile for se3_wheel_leg — 轮腿机器人强化学习训练框架
# 运行 `just` 查看所有可用命令

# 默认：显示帮助
@_default:
    @just --list

# ---- Setup ----

# 安装项目依赖并配置 pre-commit hook
setup:
    uv sync
    uv run prek install

# 环境健康检查：验证 Python、GPU、W&B、prek 状态
check:
    @echo "=== Python ==="
    uv run python --version
    @echo ""
    @echo "=== uv ==="
    uv --version
    @echo ""
    @echo "=== 核心依赖 ==="
    uv run python -c "import mujoco, torch; from importlib.metadata import version; print('  mujoco:', mujoco.__version__); print('  torch:', torch.__version__); print('  rerun-sdk:', version('rerun-sdk'))"
    @echo ""
    @echo "=== GPU / CUDA ==="
    @uv run python -c "import torch; print('  CUDA 可用:', torch.cuda.is_available()); print('  GPU 数量:', torch.cuda.device_count())" || echo "  ⚠️  无 GPU 或 CUDA 不可用（macOS 正常现象）"
    @echo ""
    @echo "=== .env / W&B ==="
    @[ -f .env ] && echo "  .env 已配置" || echo "  ⚠️  .env 缺失 — 运行: cp .env.example .env  并填入 WANDB_API_KEY"
    @echo ""
    @echo "=== prek hook ==="
    @uv run prek check 2>/dev/null && echo "  prek hook 已安装" || echo "  ⚠️  prek hook 未安装 — 运行: just setup"

# ---- 代码质量 ----

# ruff 格式化
fmt:
    uv run ruff format .

# ruff lint + 自动修复
lint:
    uv run ruff check . --fix

# 格式化 + lint（提交前执行）
check-code: fmt lint

# ---- 训练 ----

# CPU smoke 验证（5 轮，不上传 W&B）— 修改训练代码后必跑
smoke:
    SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-FlowMatch-Wheel-GRU --env.scene.num-envs 1 --gpu-ids None

# GPU smoke 验证（5 轮，不上传 W&B）
smoke-gpu:
    SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-FlowMatch-Wheel-GRU --env.scene.num-envs 1024

# 倒地自启 CPU smoke 验证（5 轮，不上传 W&B）
smoke-recovery:
    SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Recovery-GRU --env.scene.num-envs 4 --gpu-ids None

# FlowMatch WHEEL 单标签训练（需要 GPU + .env）
train:
    @[ -f .env ] || { echo "❌ 缺少 .env 文件。请先: cp .env.example .env  并填入 WANDB_API_KEY"; exit 1; }
    uv run --env-file .env se3-train SE3-WheelLegged-FlowMatch-Wheel-GRU --env.scene.num-envs 1024

# 倒地自启训练（需要 GPU + .env）
train-recovery:
    @[ -f .env ] || { echo "❌ 缺少 .env 文件。请先: cp .env.example .env  并填入 WANDB_API_KEY"; exit 1; }
    uv run --env-file .env se3-train SE3-WheelLegged-Recovery-GRU --env.scene.num-envs 4096

# 崎岖地形训练（需要 GPU + .env）
train-rough:
    @[ -f .env ] || { echo "❌ 缺少 .env 文件。请先: cp .env.example .env  并填入 WANDB_API_KEY"; exit 1; }
    uv run --env-file .env se3-train SE3-WheelLegged-Rough --env.scene.num-envs 1024

# CPU 调试训练（极慢，仅用于调试）
train-cpu:
    @[ -f .env ] || { echo "❌ 缺少 .env 文件。请先: cp .env.example .env  并填入 WANDB_API_KEY"; exit 1; }
    uv run --env-file .env se3-train SE3-WheelLegged-FlowMatch-Wheel-GRU --env.scene.num-envs 1 --gpu-ids None

# ---- 评估 / 回放 ----

# ---- mjlab play（行走 / 跳跃，原生 viewer）----

# mjlab play 行走回放（自动选最新 checkpoint，本地 viewer）
play:
    uv run se3-play SE3-WheelLegged-FlowMatch-Wheel-GRU --num-envs 1

# mjlab play 行走回放（指定 checkpoint）
# 用法: just play-ckpt logs/.../model_4999.pt
play-ckpt checkpoint:
    uv run se3-play SE3-WheelLegged-FlowMatch-Wheel-GRU --checkpoint-file {{checkpoint}} --num-envs 1

# mjlab play 跳跃回放（自动选最新 checkpoint，本地 viewer）
play-jump:
    uv run se3-play SE3-WheelLegged-Jump-FineTune-GRU --num-envs 1

# mjlab play 跳跃回放（指定 checkpoint）
# 用法: just play-jump-ckpt logs/.../model_4999.pt
play-jump-ckpt checkpoint:
    uv run se3-play SE3-WheelLegged-Jump-FineTune-GRU --checkpoint-file {{checkpoint}} --num-envs 1

# ---- sim2sim（行走模型）—— 默认跑 walk-sweep 历程 ----

# sim2sim 回放（自动选最新 checkpoint，Rerun，walk-sweep 历程）
sim:
    uv run se3-sim2sim --max-steps 3000 --course walk-sweep

# sim2sim 指定 checkpoint（Rerun，walk-sweep 历程）
# 用法: just sim-ckpt logs/.../model_4999.pt
sim-ckpt checkpoint:
    uv run se3-sim2sim --checkpoint {{checkpoint}} --max-steps 3000 --course walk-sweep

# sim2sim 无 GUI 快速验证（自动选最新 checkpoint）
sim-headless:
    uv run se3-sim2sim --viewer none --max-steps 200 --print-every 20 --course walk-sweep

# sim2sim 无 GUI 验证（指定 checkpoint）
# 用法: just sim-headless-ckpt logs/.../model_4999.pt
sim-headless-ckpt checkpoint:
    uv run se3-sim2sim --checkpoint {{checkpoint}} --viewer none --max-steps 200 --print-every 20 --course walk-sweep

# ---- sim2sim（跳跃模型） —— 默认跑 jump-sweep 历程 ----

# sim2sim 跳跃回放（自动选最新 checkpoint，Rerun，jump-sweep 历程）
sim-jump:
    uv run se3-sim2sim --max-steps 5000 --course jump-sweep

# sim2sim 指定 checkpoint 跳跃回放（Rerun，jump-sweep 历程）
# 用法: just sim-jump-ckpt logs/.../model_4999.pt
sim-jump-ckpt checkpoint:
    uv run se3-sim2sim --checkpoint {{checkpoint}} --max-steps 5000 --course jump-sweep

# sim2sim 跳跃无 GUI 快速验证（指定 checkpoint）
# 用法: just sim-jump-headless-ckpt logs/.../model_4999.pt
sim-jump-headless-ckpt checkpoint:
    uv run se3-sim2sim --checkpoint {{checkpoint}} --viewer none --max-steps 5000 --print-every 50 --course jump-sweep

# ---- 清理 ----

# 清理训练日志、wandb 缓存、回放文件
clean:
    rm -rf logs/ wandb/ replays/ MUJOCO_LOG.TXT
