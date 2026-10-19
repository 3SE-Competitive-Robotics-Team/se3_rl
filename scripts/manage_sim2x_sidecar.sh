#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
用法：
  manage_sim2x_sidecar.sh <start|ensure|stop|restart|status|log> [选项]

选项：
  --repo-root PATH      se3_wheel_leg 仓库根目录
  --run-root PATH       experiment/run_id/onnx 的扫描根目录
  --state-dir PATH      PID、metadata 与日志目录
  --env-file PATH       启动前 source 的机器环境文件
  --host HOST           Viser 监听地址，仅允许 127.0.0.1（默认）
  --port PORT           Viser 端口，默认 8080
  --cpu-list LIST       taskset CPU 列表，例如 0-1
  --scan-interval SEC   ONNX 扫描周期，默认 1.0
  --wait-timeout SEC    首个 ONNX 等待上限，默认 600
  --label TEXT          Viser server 标签
  --physics-hz HZ       覆盖 MuJoCo solver 频率（透传给 se3-sim2x-browser）
EOF
}

die() {
  echo "错误：$*" >&2
  exit 1
}

action="${1:-}"
if [[ -z "$action" || "$action" == "-h" || "$action" == "--help" ]]; then
  usage
  exit 0
fi
shift

repo_root="$DEFAULT_REPO_ROOT"
run_root=""
state_dir=""
env_file=""
host="127.0.0.1"
port="8080"
cpu_list=""
scan_interval="1.0"
wait_timeout="600"
label="se3-sim2x-sidecar"
physics_hz=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) repo_root="${2:?--repo-root 缺少值}"; shift 2 ;;
    --run-root) run_root="${2:?--run-root 缺少值}"; shift 2 ;;
    --state-dir) state_dir="${2:?--state-dir 缺少值}"; shift 2 ;;
    --env-file) env_file="${2:?--env-file 缺少值}"; shift 2 ;;
    --host) host="${2:?--host 缺少值}"; shift 2 ;;
    --port) port="${2:?--port 缺少值}"; shift 2 ;;
    --cpu-list) cpu_list="${2:?--cpu-list 缺少值}"; shift 2 ;;
    --scan-interval) scan_interval="${2:?--scan-interval 缺少值}"; shift 2 ;;
    --wait-timeout) wait_timeout="${2:?--wait-timeout 缺少值}"; shift 2 ;;
    --label) label="${2:?--label 缺少值}"; shift 2 ;;
    --physics-hz) physics_hz="${2:?--physics-hz 缺少值}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

[[ "$action" =~ ^(start|ensure|stop|restart|status|log)$ ]] || die "未知动作：$action"
[[ -n "$state_dir" ]] || die "必须提供 --state-dir"
[[ "$host" == "127.0.0.1" ]] || die "sidecar 只允许监听 127.0.0.1，并通过 SSH 转发访问"
[[ "$port" =~ ^[0-9]+$ ]] && ((port >= 1 && port <= 65535)) || die "--port 必须位于 [1, 65535]"
[[ "$wait_timeout" =~ ^[0-9]+$ ]] || die "--wait-timeout 必须是非负整数"

mkdir -p -- "$state_dir"
state_dir="$(cd -- "$state_dir" && pwd -P)"
pid_file="${state_dir}/sim2x.pid"
pgid_file="${state_dir}/sim2x.pgid"
meta_file="${state_dir}/sim2x.meta"
log_file="${state_dir}/sim2x.log"
lock_file="${state_dir}/sim2x.lock"

exec 9>"$lock_file"
flock -w 10 9 || die "无法取得 sidecar 生命周期锁：$lock_file"

read_pid() {
  if [[ -s "$pid_file" ]]; then
    tr -d '[:space:]' < "$pid_file"
  fi
}

is_expected_process() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  local command_line
  command_line="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  [[ "$command_line" == *"se3-sim2x-browser"* ]]
}

http_ready() {
  curl --fail --silent --show-error \
    --connect-timeout 1 --max-time 2 \
    "http://${host}:${port}/" \
    -o /dev/null 2>/dev/null
}

latest_onnx() {
  [[ -n "$run_root" && -d "$run_root" ]] || return 0
  find "$run_root" -type f -path '*/onnx/*.onnx' -printf '%T@ %p\n' 2>/dev/null \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

stop_process() {
  local pid="$1"
  is_expected_process "$pid" || return 0
  local pgid
  pgid="$(ps -p "$pid" -o pgid= | tr -d '[:space:]')"
  [[ "$pgid" =~ ^[0-9]+$ && "$pgid" -gt 1 ]] || die "拒绝停止非法 PGID：$pgid"
  kill -TERM -- "-${pgid}"
  for _ in $(seq 1 50); do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
    sleep 0.1
  done
  kill -KILL -- "-${pgid}"
}

stop_sidecar() {
  local pid
  pid="$(read_pid || true)"
  if [[ -n "$pid" ]] && is_expected_process "$pid"; then
    stop_process "$pid"
    echo "[se3-sim2x] 已停止 PID=${pid}"
  else
    echo "[se3-sim2x] 未发现运行中的 sidecar"
  fi
  rm -f -- "$pid_file" "$pgid_file"
}

status_sidecar() {
  local pid
  pid="$(read_pid || true)"
  if [[ -n "$pid" ]] && is_expected_process "$pid"; then
    local pgid
    pgid="$(ps -p "$pid" -o pgid= | tr -d '[:space:]')"
    echo "status=running pid=${pid} pgid=${pgid} url=http://${host}:${port}/"
    local model
    model="$(latest_onnx || true)"
    [[ -z "$model" ]] || echo "latest_onnx=${model}"
    http_ready || return 2
    return 0
  fi
  echo "status=stopped"
  return 1
}

start_sidecar() {
  [[ -d "$repo_root" ]] || die "仓库不存在：$repo_root"
  repo_root="$(cd -- "$repo_root" && pwd -P)"
  [[ -x "$repo_root/.venv/bin/se3-sim2x-browser" ]] \
    || die "未找到已安装的 se3-sim2x-browser：$repo_root/.venv/bin/se3-sim2x-browser"
  [[ -n "$run_root" && -d "$run_root" ]] || die "ONNX 扫描根目录不存在：$run_root"
  run_root="$(cd -- "$run_root" && pwd -P)"
  if [[ -n "$env_file" ]]; then
    [[ -f "$env_file" ]] || die "环境文件不存在：$env_file"
    # shellcheck disable=SC1090
    source "$env_file"
  fi

  local existing_pid
  existing_pid="$(read_pid || true)"
  if [[ -n "$existing_pid" ]] && is_expected_process "$existing_pid"; then
    if [[ -f "$meta_file" ]] \
      && { ! grep -Fqx "repo_root=${repo_root}" "$meta_file" \
        || ! grep -Fqx "run_root=${run_root}" "$meta_file" \
        || ! grep -Fqx "host=${host}" "$meta_file" \
        || ! grep -Fqx "port=${port}" "$meta_file"; }; then
      die "运行中的 sidecar 配置与请求不一致，请显式 restart"
    fi
    http_ready || die "sidecar PID=${existing_pid} 存活但 HTTP 未就绪，请先 restart"
    echo "[se3-sim2x] 已在运行 PID=${existing_pid} URL=http://${host}:${port}/"
    return 0
  fi
  rm -f -- "$pid_file" "$pgid_file"

  local wait_started
  wait_started="$(date +%s)"
  while [[ -z "$(latest_onnx || true)" ]]; do
    if (( $(date +%s) - wait_started >= wait_timeout )); then
      die "等待首个 ONNX 超时：${run_root}"
    fi
    sleep 1
  done

  if ss -H -ltn "sport = :${port}" | grep -q .; then
    die "端口已被其他进程占用：${host}:${port}"
  fi

  local -a browser_command
  browser_command=(
    "$repo_root/.venv/bin/se3-sim2x-browser"
    --run-root "$run_root"
    --host "$host"
    --port "$port"
    --scan-interval "$scan_interval"
    --label "$label"
  )
  if [[ -n "$physics_hz" ]]; then
    browser_command+=(--physics-hz "$physics_hz")
  fi
  if [[ -n "$cpu_list" ]]; then
    browser_command=(taskset -c "$cpu_list" "${browser_command[@]}")
  fi

  {
    printf '\n[%s] 启动 sidecar\n' "$(date -Ins)"
    printf 'run_root=%s host=%s port=%s cpu_list=%s\n' "$run_root" "$host" "$port" "${cpu_list:-none}"
  } >> "$log_file"

  cd -- "$repo_root"
  nohup setsid env \
    CUDA_VISIBLE_DEVICES='' \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    "${browser_command[@]}" \
    9>&- >> "$log_file" 2>&1 < /dev/null &
  local pid="$!"
  printf '%s\n' "$pid" > "${pid_file}.tmp"
  mv -f -- "${pid_file}.tmp" "$pid_file"

  local pgid=""
  for _ in $(seq 1 600); do
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -80 "$log_file" >&2 || true
      rm -f -- "$pid_file" "$pgid_file"
      die "sidecar 启动后提前退出"
    fi
    if is_expected_process "$pid"; then
      pgid="$(ps -p "$pid" -o pgid= | tr -d '[:space:]')"
    fi
    if [[ -n "$pgid" ]] && http_ready; then
      break
    fi
    sleep 0.1
  done
  if [[ -z "$pgid" ]] || ! is_expected_process "$pid" || ! http_ready; then
    stop_process "$pid"
    tail -80 "$log_file" >&2 || true
    rm -f -- "$pid_file" "$pgid_file"
    die "sidecar HTTP 健康检查超时"
  fi

  printf '%s\n' "$pgid" > "${pgid_file}.tmp"
  mv -f -- "${pgid_file}.tmp" "$pgid_file"
  {
    printf 'pid=%s\n' "$pid"
    printf 'pgid=%s\n' "$pgid"
    printf 'repo_root=%s\n' "$repo_root"
    printf 'run_root=%s\n' "$run_root"
    printf 'host=%s\n' "$host"
    printf 'port=%s\n' "$port"
    printf 'cpu_list=%s\n' "$cpu_list"
    printf 'physics_hz=%s\n' "${physics_hz:-default}"
    printf 'started_at=%s\n' "$(date -Ins)"
  } > "${meta_file}.tmp"
  mv -f -- "${meta_file}.tmp" "$meta_file"
  echo "[se3-sim2x] 已启动 PID=${pid} PGID=${pgid} URL=http://${host}:${port}/"
}

case "$action" in
  start|ensure)
    start_sidecar
    ;;
  stop)
    stop_sidecar
    ;;
  restart)
    stop_sidecar
    start_sidecar
    ;;
  status)
    status_sidecar
    ;;
  log)
    tail -100 "$log_file"
    ;;
esac
