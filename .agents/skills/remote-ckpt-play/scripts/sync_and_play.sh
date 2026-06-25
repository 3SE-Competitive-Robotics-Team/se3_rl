#!/usr/bin/env bash
set -euo pipefail

REMOTE="wuyingyun"
REMOTE_LOG_DIR="~/project/se3_rl/logs/rsl_rl/se3_wheel_leg"
LOCAL_BASE="logs/remote_watch"
PROJECT_DIR="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$PROJECT_DIR"

# 查询远端最新 checkpoint
echo "[1] 查询远端..."
read -r REMOTE_RUN LATEST_CKPT < <(ssh "$REMOTE" bash -s << 'EOF'
run=$(ls -dt ~/project/se3_rl/logs/rsl_rl/se3_wheel_leg/*/ 2>/dev/null | head -1)
echo "$(basename "$run") $(basename "$(ls -t "$run"model_*.pt 2>/dev/null | head -1)")"
EOF
)
echo "  Run: $REMOTE_RUN"
echo "  Ckpt: $LATEST_CKPT"

# Rsync
LOCAL_DIR="$LOCAL_BASE/$REMOTE_RUN"
LOCAL_PATH="$LOCAL_DIR/$LATEST_CKPT"
mkdir -p "$LOCAL_DIR"
if [ ! -f "$LOCAL_PATH" ]; then
  rsync -avP "$REMOTE:~/project/se3_rl/logs/rsl_rl/se3_wheel_leg/$REMOTE_RUN/$LATEST_CKPT" "$LOCAL_DIR/"
fi

# 清理旧 Viser
lsof -ti :8080 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 2

# 启动 play
cd "$PROJECT_DIR"
nohup uv run se3-play SE3-WheelLegged-Flat-GRU \
  --checkpoint-file "$LOCAL_PATH" --viewer viser --num-envs 1 \
  > /tmp/remote_ckpt_play.log 2>&1 &

for i in $(seq 1 15); do
  sleep 1; lsof -i :8080 >/dev/null 2>&1 && break
done
echo ""
echo "=== http://localhost:8080/ ==="
