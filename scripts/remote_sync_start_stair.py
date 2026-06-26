"""同步当前代码并在 A800 Kubernetes 容器启动台阶续训。"""

from __future__ import annotations

import argparse
import base64
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    capture: bool = False,
    display: str | None = None,
) -> str:
    """执行本地命令，失败时立即终止。"""
    print("+", display or subprocess.list2cmdline(command), flush=True)
    capture_kwargs = (
        {"capture_output": True, "text": True, "encoding": "utf-8", "errors": "replace"}
        if capture
        else {}
    )
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        **capture_kwargs,
    )
    if capture and result.stdout:
        print(result.stdout, end="")
    return result.stdout if capture else ""


def ssh_args(args: argparse.Namespace, command: str) -> list[str]:
    """生成单跳或两跳 SSH 命令。"""
    if args.inner_host:
        encoded = base64.b64encode(command.encode("utf-8")).decode("ascii")
        inner_command = f"echo {shlex.quote(encoded)} | base64 -d | bash"
        nested = f"ssh {shlex.quote(args.inner_host)} {shlex.quote(inner_command)}"
        return ["ssh", args.entry_host, nested]
    return ["ssh", args.entry_host, command]


def pod_bash(
    args: argparse.Namespace,
    *,
    namespace: str,
    pod: str,
    script: str,
    capture: bool = False,
    label: str = "执行远端脚本",
) -> str:
    """通过 SSH 和 base64 在目标 pod 内执行 Bash 脚本。"""
    payload = base64.b64encode(script.encode("utf-8")).decode("ascii")
    remote_command = (
        f"echo {payload} | base64 -d | "
        f"kubectl exec -i -n {shlex.quote(namespace)} {shlex.quote(pod)} -- bash -s"
    )
    route = f"{args.entry_host}->{args.inner_host}" if args.inner_host else args.entry_host
    display = f"ssh {route} kubectl exec -n {namespace} {pod} -- bash -s  # {label}"
    return run(ssh_args(args, remote_command), capture=capture, display=display)


def copy_to_remote_host(args: argparse.Namespace, local_path: Path, remote_path: str) -> None:
    """把本地文件复制到执行 kubectl 的远端主机。"""
    if not args.inner_host:
        run(["scp", str(local_path), f"{args.entry_host}:{remote_path}"])
        return
    entry_tmp = f"/tmp/{Path(remote_path).name}.entry"
    try:
        run(["scp", str(local_path), f"{args.entry_host}:{entry_tmp}"])
        nested = f"scp {shlex.quote(entry_tmp)} {shlex.quote(args.inner_host + ':' + remote_path)}"
        run(["ssh", args.entry_host, nested])
    finally:
        cleanup = f"Remove-Item -Force -ErrorAction SilentlyContinue '{entry_tmp}'"
        run(["ssh", args.entry_host, cleanup])


def build_code_archive(archive: Path, *, include_base_model: bool = False) -> None:
    """打包代码并排除训练产物与本地虚拟环境。"""
    if archive.exists():
        archive.unlink()
    command = [
        "tar",
        "-czf",
        str(archive),
        "--exclude=.git",
        "--exclude=.venv",
        "--exclude=logs",
        "--exclude=outputs",
        "--exclude=replays",
        "--exclude=.ruff_cache",
        "--exclude=__pycache__",
    ]
    if not include_base_model:
        command.append("--exclude=assets/base_model")
    command.append(".")
    run(command, cwd=REPO_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync local code and start remote stair training.")
    parser.add_argument("--entry-host", default="a800")
    parser.add_argument(
        "--inner-host",
        default=None,
        help="两跳 SSH 的内层主机，例如从 laptop 再 ssh 到 a800。",
    )
    parser.add_argument("--namespace", default="gczx-project06")
    parser.add_argument("--pod", default="abbtask-79cdb78487-mgx44")
    parser.add_argument(
        "--remote-project",
        default="/workspace/3SE-Competitive-Robotics-Team/se3_wheel_leg",
    )
    parser.add_argument(
        "--cuda-compat-dir",
        default="/workspace/cudacompat/usr/local/cuda-12.6/compat",
    )
    parser.add_argument("--cuda-toolkit-lib-dir", default="/usr/local/cuda-12.2/lib64")
    parser.add_argument(
        "--task",
        default="SE3-WheelLegged-Stair-NoCTBC-GRU-WarmStart",
    )
    parser.add_argument(
        "--load-run",
        default="2026-06-01_12-35-40_stair_user_state_ff",
    )
    parser.add_argument("--load-checkpoint", default=r"model_4100\.pt")
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="不检查或加载 checkpoint，并显式设置 --agent.resume False.",
    )
    parser.add_argument(
        "--full-resume",
        action="store_true",
        help="完整恢复 checkpoint 的 optimizer、iteration 和 env_state；默认仍为 warm-start.",
    )
    parser.add_argument(
        "--sync-base-model",
        action="store_true",
        help="同步 assets/base_model；默认不传基模以减少 A800 同步时间。",
    )
    parser.add_argument("--envs", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument(
        "--gpu-ids",
        default="all",
        help="传给 se3-train 的 --gpu-ids; 配合 --cuda-visible-devices 时通常保持 all.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="限制本次训练可见的物理 GPU, 例如 1,2,3,4,5,6,7; 为空则保持原来的 unset 行为.",
    )
    parser.add_argument(
        "--gpu-memory-busy-threshold-mib",
        type=int,
        default=1000,
        help="指定 --cuda-visible-devices 时, 选定 GPU 显存超过该值则拒绝启动.",
    )
    parser.add_argument(
        "--stair-local-iter-offset",
        type=int,
        default=None,
        help="CTBC 台阶阶段 local iter 偏移; 从中途 checkpoint 续训时用于避免重新打开 CTBC.",
    )
    parser.add_argument(
        "--run-name",
        default="stair_progress_fix_noctbc_m4100_20260610",
    )
    parser.add_argument(
        "--job-name",
        default="stair_progress_fix_noctbc_m4100_20260610",
    )
    parser.add_argument("--watch-terrain-level", type=int, choices=range(10), default=6)
    parser.add_argument(
        "--watch-command-height",
        type=float,
        default=None,
        help="生成 watch 命令时固定 height command；省略则按训练课程随机采样。",
    )
    parser.add_argument("--watch-interval-iters", type=int, default=100)
    return parser.parse_args()


def watch_remote_command(args: argparse.Namespace, run_dir: str | None = None) -> str:
    """生成与本次远端训练参数对应的本地 watcher 指令。"""
    run_dir_arg = f"  --run-dir {run_dir} `\n" if run_dir else ""
    route_args = (
        f"  --host {args.inner_host} `\n"
        f"  --entry-host {args.entry_host} `\n"
        f"  --inner-host {args.inner_host} `\n"
        if args.inner_host
        else f"  --host {args.entry_host} `\n"
    )
    command_height_arg = (
        f"  --command-height {args.watch_command_height:g} `\n"
        if args.watch_command_height is not None
        else ""
    )
    return (
        "uv run --no-sync python scripts/watch_remote_train_local.py `\n"
        f"{route_args}"
        f"  --namespace {args.namespace} `\n"
        f"  --pod {args.pod} `\n"
        f"  --remote-project {args.remote_project} `\n"
        f"{run_dir_arg}"
        f"  --task {args.task} `\n"
        f"  --terrain-level {args.watch_terrain_level} `\n"
        f"{command_height_arg}"
        f"  --interval-iters {args.watch_interval_iters} `\n"
        "  --poll-seconds 60 `\n"
        "  --viewer viser"
    )


def main() -> None:
    args = parse_args()
    archive = Path(tempfile.gettempdir()) / "se3_wheel_leg_code_sync.tar.gz"
    remote_archive = "/tmp/se3_wheel_leg_code_sync.tar.gz"
    log_path = f"/tmp/train_{args.job_name}.log"
    pid_path = f"/tmp/train_{args.job_name}.pid"
    checkpoint_file = args.load_checkpoint.replace(r"\.", ".").strip("^$")
    cuda_visible_devices = (args.cuda_visible_devices or "").strip()
    if "/" in checkpoint_file or "\\" in checkpoint_file:
        raise ValueError("--load-checkpoint 必须是 checkpoint 文件名或其转义形式")
    if cuda_visible_devices and not re.fullmatch(r"\d+(,\d+)*", cuda_visible_devices):
        raise ValueError("--cuda-visible-devices 必须是逗号分隔的 GPU 编号, 例如 1,2,3")

    if cuda_visible_devices:
        active_check = f"""
active="$(ps -eo stat=,cmd= | awk '$1 !~ /^Z/ && ($0 ~ /[s]e3-train/ || $0 ~ /[t]orchrunx/ || $0 ~ /[t]orch.distributed.run/) {{print}}')"
if [ -n "$active" ]; then
  echo "检测到训练相关进程; 本次只检查选定 GPU 是否空闲:"
  echo "$active"
fi
busy="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v selected={shlex.quote(cuda_visible_devices)} -v threshold={args.gpu_memory_busy_threshold_mib} '
BEGIN {{
  split(selected, ids, ",")
  for (i in ids) wanted[ids[i]] = 1
}}
{{
  gsub(/ /, "", $1)
  gsub(/ /, "", $2)
  if (($1 in wanted) && ($2 + 0) > threshold) print $0
}}')"
if [ -n "$busy" ]; then
  echo "选定 GPU 显存占用超过阈值, 拒绝启动:"
  echo "$busy"
  exit 20
fi
"""
    else:
        active_check = """
active="$(ps -eo stat=,cmd= | awk '$1 !~ /^Z/ && ($0 ~ /[s]e3-train/ || $0 ~ /[t]orchrunx/ || $0 ~ /[t]orch.distributed.run/) {print}')"
if [ -n "$active" ]; then
  echo "检测到仍在运行的训练进程:"
  echo "$active"
  exit 20
fi
"""

    checkpoint_check = (
        ""
        if args.from_scratch
        else (
            "test -f logs/rsl_rl/se3_wheel_leg/"
            f"{shlex.quote(args.load_run)}/{shlex.quote(checkpoint_file)}"
        )
    )
    precheck_checkpoint_check = "" if args.sync_base_model else checkpoint_check
    precheck_script = f"""
set -euo pipefail
cd {shlex.quote(args.remote_project)}
test -d .venv/bin
test -d {shlex.quote(args.cuda_compat_dir)}
test -d {shlex.quote(args.cuda_toolkit_lib_dir)}
{precheck_checkpoint_check}
{active_check}
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
"""
    pod_bash(
        args,
        namespace=args.namespace,
        pod=args.pod,
        script=precheck_script,
        label="预检查远端环境和训练进程",
    )

    build_code_archive(archive, include_base_model=args.sync_base_model)
    copy_to_remote_host(args, archive, remote_archive)
    run(
        ssh_args(
            args,
            (
                f"kubectl cp {shlex.quote(remote_archive)} "
                f"{shlex.quote(args.namespace)}/{shlex.quote(args.pod)}:"
                f"{shlex.quote(remote_archive)} -n {shlex.quote(args.namespace)}"
            ),
        )
    )

    sync_script = f"""
set -euo pipefail
cd {shlex.quote(args.remote_project)}
tar -xzf {shlex.quote(remote_archive)}
export CUDA_COMPAT_DIR={shlex.quote(args.cuda_compat_dir)}
export CUDA_TOOLKIT_LIB_DIR={shlex.quote(args.cuda_toolkit_lib_dir)}
export LD_LIBRARY_PATH="$CUDA_COMPAT_DIR:$CUDA_TOOLKIT_LIB_DIR"
export PYTHONPATH="$PWD/src${{PYTHONPATH:+:$PYTHONPATH}}"
export MUJOCO_GL=egl
compile_targets=(src scripts)
[ -d experiments ] && compile_targets+=(experiments)
./.venv/bin/python -m compileall -q "${{compile_targets[@]}}"
{checkpoint_check}
"""
    pod_bash(
        args,
        namespace=args.namespace,
        pod=args.pod,
        script=sync_script,
        label="解包同步代码并编译检查",
    )

    launch_args = [
        "./.venv/bin/se3-train",
        args.task,
        "--gpu-ids",
        args.gpu_ids,
        "--env.scene.num-envs",
        str(args.envs),
        "--agent.max-iterations",
        str(args.iterations),
        "--agent.run-name",
        args.run_name,
    ]
    if args.from_scratch:
        launch_args.extend(["--agent.resume", "False"])
    else:
        launch_args.extend(
            [
                "--agent.load-run",
                args.load_run,
                "--agent.load-checkpoint",
                args.load_checkpoint,
            ]
        )
    launch_command = " ".join(shlex.quote(value) for value in launch_args)
    cuda_visible_line = (
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(cuda_visible_devices)}"
        if cuda_visible_devices
        else "unset CUDA_VISIBLE_DEVICES"
    )
    stair_offset_line = (
        f"export SE3_STAIR_LOCAL_ITER_OFFSET={args.stair_local_iter_offset}"
        if args.stair_local_iter_offset is not None
        else "unset SE3_STAIR_LOCAL_ITER_OFFSET"
    )
    full_resume_value = "1" if args.full_resume else "0"
    launch_script = f"""
set -euo pipefail
cd {shlex.quote(args.remote_project)}
export CUDA_COMPAT_DIR={shlex.quote(args.cuda_compat_dir)}
export CUDA_TOOLKIT_LIB_DIR={shlex.quote(args.cuda_toolkit_lib_dir)}
export LD_LIBRARY_PATH="$CUDA_COMPAT_DIR:$CUDA_TOOLKIT_LIB_DIR"
export PYTHONPATH="$PWD/src${{PYTHONPATH:+:$PYTHONPATH}}"
export MUJOCO_GL=egl
export WANDB_MODE=offline
export SE3_LOGGER=wandb
export SE3_FULL_RESUME={full_resume_value}
{stair_offset_line}
export PYTHONUNBUFFERED=1
{cuda_visible_line}
rm -f {shlex.quote(log_path)} {shlex.quote(pid_path)}
nohup {launch_command} > {shlex.quote(log_path)} 2>&1 < /dev/null &
pid=$!
echo "$pid" > {shlex.quote(pid_path)}
sleep 3
kill -0 "$pid"
run_dir=""
for _ in $(seq 1 30); do
  if [ -d logs/rsl_rl/se3_wheel_leg ]; then
    run_dir_lines=$(find logs/rsl_rl/se3_wheel_leg -mindepth 1 -maxdepth 1 -type d \
      -name '*_{args.run_name}' -printf '%T@ %f\n' | sort -nr)
    run_dir=$(printf '%s\n' "$run_dir_lines" | sed -n '1s/^[^ ]* //p')
  fi
  if [ -n "$run_dir" ]; then
    break
  fi
  sleep 1
done
echo "TRAIN_PID=$pid"
echo "TRAIN_LOG={log_path}"
echo "TRAIN_RUN_DIR=$run_dir"
"""
    launch_output = pod_bash(
        args,
        namespace=args.namespace,
        pod=args.pod,
        script=launch_script,
        capture=True,
        label="启动远端训练",
    )
    run_dir_match = re.search(r"^TRAIN_RUN_DIR=(.+)$", launch_output, re.MULTILINE)
    run_dir = run_dir_match.group(1).strip() if run_dir_match else None

    print("\n训练已启动。日志监控命令:")
    print(
        f'ssh {args.entry_host} "kubectl exec -n {args.namespace} {args.pod} -- '
        f"bash -lc 'tail -f {log_path}'\""
    )
    print("\nWatch Remote 指令:")
    print(watch_remote_command(args, run_dir=run_dir))


if __name__ == "__main__":
    main()
