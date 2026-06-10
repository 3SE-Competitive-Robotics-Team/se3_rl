#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -d ".venv" ]]; then
    # runtime 包在 NX 上会自带轻量 venv；完整训练仓库可继续用 uv run。
    # shellcheck disable=SC1091
    . .venv/bin/activate
fi
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

exec python -m se3_deploy.visualize_cdc_state \
    --local-cdc \
    --host "${CDC_VIS_HOST:-0.0.0.0}" \
    --viewer-port "${CDC_VIS_PORT:-8081}" \
    "$@"
