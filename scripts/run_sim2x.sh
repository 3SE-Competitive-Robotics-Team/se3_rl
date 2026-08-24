#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SUBMODULE_ROOT="${REPO_ROOT}/submodules/se3-sim2x"

if command -v uv >/dev/null 2>&1; then
  uv_bin="uv"
elif command -v uv.exe >/dev/null 2>&1; then
  uv_bin="uv.exe"
else
  echo "错误：未找到 uv，请先安装 uv。" >&2
  exit 1
fi

if [[ ! -f "${SUBMODULE_ROOT}/pyproject.toml" ]]; then
  echo "[se3-sim2x] 正在初始化 submodule..."
  git -C "${REPO_ROOT}" submodule update --init --recursive
fi

echo "[se3-sim2x] 监听 ${REPO_ROOT}/logs/rsl_rl"
echo "[se3-sim2x] 在 Viser 中按 experiment / run_id / ONNX 切换模型"

cd -- "${REPO_ROOT}"
exec "${uv_bin}" run --no-sync --with-editable ./submodules/se3-sim2x se3-sim2x-browser
