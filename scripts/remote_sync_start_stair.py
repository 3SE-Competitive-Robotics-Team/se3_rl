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


def pod_bash(
    *,
    entry_host: str,
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
    display = f"ssh {entry_host} kubectl exec -n {namespace} {pod} -- bash -s  # {label}"
    return run(["ssh", entry_host, remote_command], capture=capture, display=display)


def build_code_archive(archive: Path, *, exclude_base_model: bool = False) -> None:
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
    if exclude_base_model:
        command.append("--exclude=assets/base_model")
    command.append(".")
    run(command, cwd=REPO_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync local code and start remote stair training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Resume notes:\n"
            "  Warm-start is the default: only policy/value weights are loaded and the new run\n"
            "  starts its runner iteration, optimizer, env common_step_counter, and CTBC local\n"
            "  iteration from zero. For yaw repair or other curriculum-offset warm starts,\n"
            "  pass --warm-start-iteration N; this initializes env common_step_counter to\n"
            "  N * 64 while still starting the runner/optimizer from zero.\n\n"
            "  For full training resume from an in-task checkpoint, pass --full-resume. The\n"
            "  checkpoint restores optimizer, runner iteration, and env common_step_counter.\n"
            "  The stair CTBC state object is still recreated on startup, so set\n"
            "  --stair-local-iter-offset to the checkpoint iteration when resuming from the\n"
            "  middle of the stair task. Example from model_800.pt:\n"
            "    --full-resume --load-checkpoint model_800\\.pt --stair-local-iter-offset 800\n"
            "  This keeps the walking phase skipped and makes CTBC enter local iteration 800\n"
            "  instead of reopening from local iteration 0.\n\n"
            "  --iterations is the number of iterations to run in this launch. With\n"
            "  --full-resume the final iteration is checkpoint_iter + --iterations, so to\n"
            "  finish a model_800.pt resume at total iteration 3800, pass --iterations 3000.\n\n"
            "  Terrain mastery target level is runtime-only state, not serialized in the\n"
            "  runner checkpoint. When resuming a stair-stage checkpoint and you already\n"
            "  know the intended level, pass --stair-initial-target-level N; otherwise the\n"
            "  new process starts the target terrain level at 0 again."
        ),
    )
    parser.add_argument("--entry-host", default="target-via-phone")
    parser.add_argument("--namespace", default="gczx-project06")
    parser.add_argument("--pod", default="gpu-8-b6994457c-kv2rj")
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
        help=(
            "完整恢复 checkpoint 的 optimizer、runner iteration 和 env common_step_counter；"
            "从台阶任务中途 checkpoint 续训时通常还要配合 --stair-local-iter-offset。"
        ),
    )
    parser.add_argument("--envs", type=int, default=8192)
    parser.add_argument(
        "--iterations",
        type=int,
        default=2000,
        help=(
            "本次启动继续训练的轮数；full resume 时最终轮数 = checkpoint 轮数 + 该值，"
            "例如 model_800.pt 训练到总 3800 轮应传 3000。"
        ),
    )
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
        "--robot-mjcf-variant",
        default=None,
        help=(
            "远端训练使用的 SE3_ROBOT_MJCF_VARIANT，例如 fourbar-surrogate 或 closedchain；"
            "省略则使用训练代码默认值。"
        ),
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
        help=(
            "CTBC 台阶阶段 local iter 对齐值；full resume 从 model_N.pt 续训时设为 N，"
            "例如 model_800.pt 配 800，避免新建 CTBC 状态机后 local iter 从 0 重开。"
        ),
    )
    parser.add_argument(
        "--warm-start-iteration",
        type=int,
        default=None,
        help=(
            "仅 warm-start 时使用：把新 run 的 env common_step_counter 初始化到该 PPO iteration。"
            "用于从 checkpoint 权重开始补训但让课程从指定轮数继续，例如 yaw repair 从 700 开始。"
        ),
    )
    parser.add_argument(
        "--stair-initial-target-level",
        type=int,
        choices=range(10),
        default=None,
        help=(
            "台阶全局 mastery curriculum 的初始 target level；从中途 checkpoint full resume "
            "时可设为当时已经达到的难度，避免课程状态回到 0。"
        ),
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
    robot_mjcf_variant_arg = (
        f"  --robot-mjcf-variant {args.robot_mjcf_variant} `\n"
        if args.robot_mjcf_variant
        else ""
    )
    command_height_arg = (
        f"  --command-height {args.watch_command_height:g} `\n"
        if args.watch_command_height is not None
        else ""
    )
    return (
        "uv run python scripts/watch_remote_train_local.py `\n"
        f"  --host {args.entry_host} `\n"
        f"  --namespace {args.namespace} `\n"
        f"  --pod {args.pod} `\n"
        f"  --remote-project {args.remote_project} `\n"
        f"{run_dir_arg}"
        f"  --task {args.task} `\n"
        f"{robot_mjcf_variant_arg}"
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
    robot_mjcf_variant = (args.robot_mjcf_variant or "").strip()
    if "/" in checkpoint_file or "\\" in checkpoint_file:
        raise ValueError("--load-checkpoint 必须是 checkpoint 文件名或其转义形式")
    if cuda_visible_devices and not re.fullmatch(r"\d+(,\d+)*", cuda_visible_devices):
        raise ValueError("--cuda-visible-devices 必须是逗号分隔的 GPU 编号, 例如 1,2,3")
    if robot_mjcf_variant and not re.fullmatch(r"[A-Za-z0-9_.-]+", robot_mjcf_variant):
        raise ValueError("--robot-mjcf-variant 只能包含字母、数字、下划线、点和短横线")

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
    precheck_script = f"""
set -euo pipefail
cd {shlex.quote(args.remote_project)}
test -d .venv/bin
test -d {shlex.quote(args.cuda_compat_dir)}
test -d {shlex.quote(args.cuda_toolkit_lib_dir)}
{checkpoint_check}
{active_check}
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
"""
    pod_bash(
        entry_host=args.entry_host,
        namespace=args.namespace,
        pod=args.pod,
        script=precheck_script,
        label="预检查远端环境和训练进程",
    )

    build_code_archive(archive, exclude_base_model=args.from_scratch)
    run(["scp", str(archive), f"{args.entry_host}:{remote_archive}"])
    run(
        [
            "ssh",
            args.entry_host,
            (
                f"kubectl cp {shlex.quote(remote_archive)} "
                f"{shlex.quote(args.namespace)}/{shlex.quote(args.pod)}:"
                f"{shlex.quote(remote_archive)} -n {shlex.quote(args.namespace)}"
            ),
        ]
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
"""
    pod_bash(
        entry_host=args.entry_host,
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
    robot_mjcf_variant_line = (
        f"export SE3_ROBOT_MJCF_VARIANT={shlex.quote(robot_mjcf_variant)}"
        if robot_mjcf_variant
        else "unset SE3_ROBOT_MJCF_VARIANT"
    )
    stair_offset_line = (
        f"export SE3_STAIR_LOCAL_ITER_OFFSET={args.stair_local_iter_offset}"
        if args.stair_local_iter_offset is not None
        else "unset SE3_STAIR_LOCAL_ITER_OFFSET"
    )
    stair_target_level_line = (
        f"export SE3_STAIR_INITIAL_TARGET_LEVEL={args.stair_initial_target_level}"
        if args.stair_initial_target_level is not None
        else "unset SE3_STAIR_INITIAL_TARGET_LEVEL"
    )
    warm_start_iteration_line = (
        f"export SE3_WARM_START_ITERATION={args.warm_start_iteration}"
        if args.warm_start_iteration is not None and not args.full_resume
        else "unset SE3_WARM_START_ITERATION"
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
{stair_target_level_line}
{warm_start_iteration_line}
{robot_mjcf_variant_line}
export SE3_WARM_START_STEPS_PER_ITER=64
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
        entry_host=args.entry_host,
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
