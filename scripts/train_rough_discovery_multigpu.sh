#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${SE3_LOGGER:=tensorboard}"
: "${SE3_ROUGH_DISCOVERY_UPLOAD_MODEL:=0}"
: "${SE3_ROUGH_DISCOVERY_SAVE_INTERVAL:=100}"
: "${SE3_ROUGH_DISCOVERY_ENVS_PER_GPU:=4096}"
: "${SE3_ROUGH_DISCOVERY_GPU_IDS:=all}"
: "${SE3_ROUGH_DISCOVERY_TORCHRUNX_LOG_DIR:=/tmp/se3_rl_torchrunx/${USER:-user}}"

export SE3_LOGGER
export SE3_ROUGH_DISCOVERY_UPLOAD_MODEL
export SE3_ROUGH_DISCOVERY_SAVE_INTERVAL

has_torchrunx_log_dir=0
for arg in "$@"; do
  case "$arg" in
    --torchrunx-log-dir|--torchrunx-log-dir=*)
      has_torchrunx_log_dir=1
      ;;
  esac
done

torchrunx_args=()
if [[ -z "${TORCHRUNX_LOG_DIR+x}" && "$has_torchrunx_log_dir" -eq 0 ]]; then
  torchrunx_args=(--torchrunx-log-dir "$SE3_ROUGH_DISCOVERY_TORCHRUNX_LOG_DIR")
fi

uv_args=(run)
if [[ -f .env ]]; then
  uv_args+=(--env-file .env)
fi

exec uv "${uv_args[@]}" se3-train SE3-WheelLegged-Rough-Discovery-GRU \
  --gpu-ids "$SE3_ROUGH_DISCOVERY_GPU_IDS" \
  --env.scene.num-envs "$SE3_ROUGH_DISCOVERY_ENVS_PER_GPU" \
  "${torchrunx_args[@]}" \
  "$@"
