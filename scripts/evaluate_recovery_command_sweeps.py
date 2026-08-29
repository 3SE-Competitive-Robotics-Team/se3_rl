"""Recovery/Loco 策略的指令跟踪扫描评测。

覆盖固定姿态评测（``evaluate_recovery_discovery_fixed_poses.py``）不覆盖的部分：
从标准站立起步，扫描速度、偏航、高度指令，并做长时零指令保持，
用于判断「站起来之后能不能好好走」以及动作/接触质量。

所有 case 共用一个 env 实例：只改命令项的 cfg 再 reset，避免重复的
MuJoCo-Warp 编译开销。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

import se3_train  # noqa: F401
from se3_train.mdp.rewards import _contact_diagnostic_stats

DEFAULT_TASK_NAME = "SE3-WheelLegged-Recovery-Loco-Grouped-MLP"
STANDING_POSE_WEIGHTS = (1.0, 0.0, 0.0, 0.0, 0.0)
CMD_VX, CMD_YAW, CMD_HEIGHT = 0, 1, 4


# 评测必须复现 checkpoint 训练时的 action 零点语义。该 flag 由 CLI 显式指定，
# 因为 ONNX metadata 目前不记录它（见 backlog B18）。None = 沿用任务注册的默认值。
_ACTION_DEFAULT_OVERRIDE: bool | None = None


def _apply_action_default_override(cfg) -> None:
    if _ACTION_DEFAULT_OVERRIDE is None:
        return
    term = cfg.actions["delayed_action"]
    term.height_conditioned_action_default = bool(_ACTION_DEFAULT_OVERRIDE)
    print(
        f"[cfg] height_conditioned_action_default = {term.height_conditioned_action_default}",
        flush=True,
    )


@dataclass(frozen=True)
class Case:
    """一个指令工况。``vx``/``yaw``/``height`` 为下发指令，``seconds`` 为总时长。"""

    name: str
    vx: float = 0.0
    yaw: float = 0.0
    height: float = 0.26
    seconds: float = 8.0
    standing: bool = False


@dataclass
class CaseResult:
    name: str
    cmd_vx: float
    cmd_yaw: float
    cmd_height: float
    actual_cmd_vx: float
    actual_cmd_yaw: float
    vx_mean: float
    vx_err_abs: float
    vy_abs_mean: float
    yaw_rate_mean: float
    yaw_err_abs: float
    height_mean: float
    height_err_abs: float
    tilt_deg_mean: float
    tilt_deg_max: float
    wheel_contact_rate: float
    leg_contact_rate: float
    collision_rate: float
    wheel_first_contact_step: float | None
    wheel_first_contact_rate: float
    wheel_contact_frac_first_1s: float
    leg_contact_frac_first_1s: float
    base_contact_frac_first_1s: float
    max_wheel_force_n: float
    max_leg_force_n: float
    max_base_force_n: float
    max_abs_action: float
    leg_action_abs_mean: float
    action_rate: float
    net_dx_m: float
    net_dy_m: float
    accum_yaw_deg: float
    terminated_rate: float
    extras: dict = field(default_factory=dict)


def _default_cases() -> list[Case]:
    cases: list[Case] = [
        Case("zero_hold_20s", 0.0, 0.0, 0.26, seconds=20.0, standing=True),
        Case("zero_hold_8s", 0.0, 0.0, 0.26, seconds=8.0, standing=True),
    ]
    for v in (0.5, 1.0, 1.5, 1.89):
        cases.append(Case(f"vx_+{v}", vx=v))
        cases.append(Case(f"vx_-{v}", vx=-v))
    for w in (0.5, 1.0, 2.5):
        cases.append(Case(f"yaw_+{w}", yaw=w))
        cases.append(Case(f"yaw_-{w}", yaw=-w))
    for h in (0.195, 0.22, 0.30, 0.34, 0.39):
        cases.append(Case(f"height_{h}", height=h, standing=True))
    cases.append(Case("combo_vx1.0_yaw1.0", vx=1.0, yaw=1.0))
    cases.append(Case("combo_vx1.0_yaw-1.0", vx=1.0, yaw=-1.0))
    return cases


def _build_env_cfg(task_name: str, num_envs: int):
    """play 模式 + 强制标准站立 reset，关闭自动重置与随机化。"""
    cfg = load_env_cfg(task_name, play=True)
    _apply_action_default_override(cfg)
    cfg.scene.num_envs = int(num_envs)
    cfg.auto_reset = False
    cfg.episode_length_s = 1.0e6

    root_params = cfg.events["reset_root_state"].params
    # 分组任务的 *_by_group 优先级高于标量参数，定姿前必须清掉。
    root_params.pop("pose_weights_by_group", None)
    root_params.pop("source_curriculum_stages_by_group", None)
    root_params.update(
        {
            "pos_xy_range": (0.0, 0.0),
            "height_offset_range": (0.0, 0.0),
            "yaw_range": (0.0, 0.0),
            "roll_jitter_range": (0.0, 0.0),
            "pitch_jitter_range": (0.0, 0.0),
            "lin_vel_range": (0.0, 0.0),
            "ang_vel_range": (0.0, 0.0),
            "clearance_range": (0.003, 0.003),
            "pose_weights": STANDING_POSE_WEIGHTS,
            "source_curriculum_stages": [],
            "standard_curriculum_stages": [],
            "use_iterations": False,
            "offset_iter": 0,
        }
    )
    if root_params.get("pose_weights") != STANDING_POSE_WEIGHTS:
        raise RuntimeError("站立 reset 契约未生效")

    joint_params = cfg.events["reset_joints"].params
    joint_params.update(
        {
            "joint_offset_range": 0.0,
            "joint_vel_range": (0.0, 0.0),
            "joint_randomization_prob": 0.0,
            "align_root_height_to_wheels": True,
            "curriculum_stages": [],
            "use_iterations": False,
            "offset_iter": 0,
        }
    )
    joint_params.pop("randomization_group_names", None)
    if "wheel_joint_vel_range" in joint_params:
        joint_params["wheel_joint_vel_range"] = (0.0, 0.0)
    return cfg


def _load_policy(env: RslRlVecEnvWrapper, checkpoint: Path, device: str, task_name: str):
    agent_cfg = load_rl_cfg(task_name)
    runner_cls = load_runner_cls(task_name) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    return runner.get_inference_policy(device=device)


def _command_term(base_env: ManagerBasedRlEnv, name: str):
    manager = base_env.command_manager
    getter = getattr(manager, "get_term", None)
    if callable(getter):
        return getter(name)
    terms = getattr(manager, "_terms", None)
    if isinstance(terms, dict) and name in terms:
        return terms[name]
    raise RuntimeError(f"无法从 command manager 取出 term: {name}")


def _apply_case(term, case: Case) -> None:
    """把 case 写进命令项 cfg，使下一次采样落到固定值。"""
    cfg = term.cfg
    cfg.lin_vel_x_range = (case.vx, case.vx)
    cfg.ang_vel_yaw_range = (case.yaw, case.yaw)
    cfg.height_range = (case.height, case.height)
    cfg.standing_height_range = (case.height, case.height)
    cfg.standing_ratio = 1.0 if case.standing else 0.0
    # 单次采样，评测期间不重采样。
    cfg.resampling_time_range = (1.0e6, 1.0e6)


def _run_case(
    base_env: ManagerBasedRlEnv,
    env: RslRlVecEnvWrapper,
    policy,
    case: Case,
    settle_frac: float,
) -> CaseResult:
    term = _command_term(base_env, "velocity_height")
    _apply_case(term, case)

    env_ids = torch.arange(base_env.num_envs, device=base_env.device)
    base_env.reset(env_ids=env_ids)
    reset_fn = getattr(policy, "reset", None)
    if reset_fn is not None:
        reset_fn()

    robot = base_env.scene["robot"]
    height_sensor = base_env.scene["base_height_sensor"]
    step_dt = float(base_env.step_dt)
    total_steps = max(1, math.ceil(case.seconds / step_dt))
    settle_start = int(total_steps * (1.0 - settle_frac))
    first_second_steps = min(total_steps, max(1, math.ceil(1.0 / step_dt)))

    start_xy = robot.data.root_link_pos_w[:, :2].clone()
    prev_yaw = None
    accum_yaw = torch.zeros(base_env.num_envs, device=base_env.device)
    prev_action = None
    alive = torch.ones(base_env.num_envs, device=base_env.device, dtype=torch.bool)
    wheel_first_contact = torch.full(
        (base_env.num_envs,),
        -1,
        device=base_env.device,
        dtype=torch.long,
    )
    wheel_contact_first_1s = torch.zeros(base_env.num_envs, device=base_env.device)
    leg_contact_first_1s = torch.zeros(base_env.num_envs, device=base_env.device)
    base_contact_first_1s = torch.zeros(base_env.num_envs, device=base_env.device)
    max_wheel_force = torch.zeros(base_env.num_envs, device=base_env.device)
    max_leg_force = torch.zeros(base_env.num_envs, device=base_env.device)
    max_base_force = torch.zeros(base_env.num_envs, device=base_env.device)
    contact_trace: dict[str, list[float]] = {
        name: []
        for name in (
            "wheel_contact_ratio",
            "leg_contact_ratio",
            "base_contact_ratio",
            "wheel_max_force_n",
            "leg_max_force_n",
            "base_max_force_n",
        )
    }

    acc: dict[str, list[torch.Tensor]] = {
        k: []
        for k in (
            "vx",
            "vy",
            "yaw_rate",
            "height",
            "tilt",
            "wheel",
            "leg",
            "coll",
            "max_a",
            "leg_a",
            "a_rate",
            "cmd_vx",
            "cmd_yaw",
        )
    }
    tilt_max = torch.zeros(base_env.num_envs, device=base_env.device)

    for step_idx in range(total_steps):
        with torch.no_grad():
            obs = env.get_observations()
            actions = policy(obs)
            _, _, dones, _ = env.step(actions)

        alive &= ~dones.to(device=base_env.device, dtype=torch.bool)

        lin_b = robot.data.root_link_lin_vel_b
        ang_b = robot.data.root_link_ang_vel_b
        pg_z = robot.data.projected_gravity_b[:, 2]
        tilt = torch.rad2deg(torch.acos(torch.clamp(-pg_z, -1.0, 1.0)))
        tilt_max = torch.maximum(tilt_max, tilt)
        height = torch.nan_to_num(height_sensor.data.heights[:, 0], nan=0.0)

        # 世界系偏航累积（处理绕圈）
        quat = robot.data.root_link_quat_w
        siny = 2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2])
        cosy = 1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2)
        yaw_w = torch.atan2(siny, cosy)
        if prev_yaw is not None:
            d = yaw_w - prev_yaw
            d = torch.atan2(torch.sin(d), torch.cos(d))
            accum_yaw += d
        prev_yaw = yaw_w

        wheel_ratio, wheel_force, _ = _contact_diagnostic_stats(base_env, "wheel_sensor", 1.0)
        leg_ratio, leg_force, _ = _contact_diagnostic_stats(base_env, "leg_contact_sensor", 1.0)
        coll_ratio, base_force, _ = _contact_diagnostic_stats(base_env, "collision_sensor", 1.0)
        wheel_full_contact = wheel_ratio >= 0.999
        first_contact = (wheel_first_contact < 0) & wheel_full_contact
        wheel_first_contact[first_contact] = step_idx
        if step_idx < first_second_steps:
            wheel_contact_first_1s += wheel_full_contact.float()
            leg_contact_first_1s += (leg_ratio > 0.0).float()
            base_contact_first_1s += (coll_ratio > 0.0).float()
        max_wheel_force = torch.maximum(max_wheel_force, wheel_force)
        max_leg_force = torch.maximum(max_leg_force, leg_force)
        max_base_force = torch.maximum(max_base_force, base_force)
        contact_trace["wheel_contact_ratio"].append(float(wheel_ratio.mean().item()))
        contact_trace["leg_contact_ratio"].append(float(leg_ratio.mean().item()))
        contact_trace["base_contact_ratio"].append(float(coll_ratio.mean().item()))
        contact_trace["wheel_max_force_n"].append(float(wheel_force.max().item()))
        contact_trace["leg_max_force_n"].append(float(leg_force.max().item()))
        contact_trace["base_max_force_n"].append(float(base_force.max().item()))

        a = actions.to(base_env.device)
        a_rate = (
            torch.sum((a - prev_action) ** 2, dim=1)
            if prev_action is not None
            else torch.zeros(base_env.num_envs, device=base_env.device)
        )
        prev_action = a.clone()

        cmd = base_env.command_manager.get_command("velocity_height")

        if step_idx >= settle_start:
            acc["vx"].append(lin_b[:, 0])
            acc["vy"].append(lin_b[:, 1])
            acc["yaw_rate"].append(ang_b[:, 2])
            acc["height"].append(height)
            acc["tilt"].append(tilt)
            acc["wheel"].append((wheel_ratio >= 0.999).float())
            acc["leg"].append((leg_ratio > 0.0).float())
            acc["coll"].append((coll_ratio > 0.0).float())
            acc["max_a"].append(a.abs().max(dim=1).values)
            acc["leg_a"].append(a[:, :4].abs().mean(dim=1))
            acc["a_rate"].append(a_rate)
            acc["cmd_vx"].append(cmd[:, CMD_VX])
            acc["cmd_yaw"].append(cmd[:, CMD_YAW])

    def m(key: str) -> float:
        if not acc[key]:
            return float("nan")
        return float(torch.stack(acc[key]).mean().item())

    cmd_vx_actual = m("cmd_vx")
    cmd_yaw_actual = m("cmd_yaw")
    vx_stack = torch.stack(acc["vx"]) if acc["vx"] else torch.zeros(1, 1)
    yaw_stack = torch.stack(acc["yaw_rate"]) if acc["yaw_rate"] else torch.zeros(1, 1)
    net = robot.data.root_link_pos_w[:, :2] - start_xy
    wheel_contact_observed = wheel_first_contact >= 0
    wheel_first_contact_step = (
        float(wheel_first_contact[wheel_contact_observed].float().mean().item())
        if wheel_contact_observed.any()
        else None
    )
    first_second_denominator = float(first_second_steps)

    return CaseResult(
        name=case.name,
        cmd_vx=case.vx,
        cmd_yaw=case.yaw,
        cmd_height=case.height,
        actual_cmd_vx=cmd_vx_actual,
        actual_cmd_yaw=cmd_yaw_actual,
        vx_mean=m("vx"),
        vx_err_abs=float((vx_stack - cmd_vx_actual).abs().mean().item()),
        vy_abs_mean=float(torch.stack(acc["vy"]).abs().mean().item())
        if acc["vy"]
        else float("nan"),
        yaw_rate_mean=m("yaw_rate"),
        yaw_err_abs=float((yaw_stack - cmd_yaw_actual).abs().mean().item()),
        height_mean=m("height"),
        height_err_abs=abs(m("height") - case.height),
        tilt_deg_mean=m("tilt"),
        tilt_deg_max=float(tilt_max.max().item()),
        wheel_contact_rate=m("wheel"),
        leg_contact_rate=m("leg"),
        collision_rate=m("coll"),
        wheel_first_contact_step=wheel_first_contact_step,
        wheel_first_contact_rate=float(wheel_contact_observed.float().mean().item()),
        wheel_contact_frac_first_1s=float(
            (wheel_contact_first_1s / first_second_denominator).mean().item()
        ),
        leg_contact_frac_first_1s=float(
            (leg_contact_first_1s / first_second_denominator).mean().item()
        ),
        base_contact_frac_first_1s=float(
            (base_contact_first_1s / first_second_denominator).mean().item()
        ),
        max_wheel_force_n=float(max_wheel_force.max().item()),
        max_leg_force_n=float(max_leg_force.max().item()),
        max_base_force_n=float(max_base_force.max().item()),
        max_abs_action=float(torch.stack(acc["max_a"]).max().item())
        if acc["max_a"]
        else float("nan"),
        leg_action_abs_mean=m("leg_a"),
        action_rate=m("a_rate"),
        net_dx_m=float(net[:, 0].mean().item()),
        net_dy_m=float(net[:, 1].mean().item()),
        accum_yaw_deg=float(torch.rad2deg(accum_yaw).mean().item()),
        terminated_rate=float((~alive).float().mean().item()),
        extras={
            "contact_trace": contact_trace,
            "contact_trace_aggregation": {
                "contact_ratio": "env_mean",
                "max_force_n": "env_max",
            },
        },
    )


def _format_table(results: list[CaseResult]) -> str:
    head = (
        "| case | cmd vx | act vx | vx err | cmd yaw | yaw rate | yaw err | "
        "height | h err | tilt | tilt max | wheel | leg | coll | wheel 1st | "
        "wheel 1s | leg 1s | base 1s | wheel F | leg F | max|a| | a rate | term |"
    )
    sep = "|---|" + "---:|" * 22
    lines = [head, sep]
    for r in results:
        wheel_first = (
            "-" if r.wheel_first_contact_step is None else f"{r.wheel_first_contact_step:.1f}"
        )
        lines.append(
            f"| {r.name} | {r.cmd_vx:+.2f} | {r.vx_mean:+.3f} | {r.vx_err_abs:.3f} | "
            f"{r.cmd_yaw:+.2f} | {r.yaw_rate_mean:+.3f} | {r.yaw_err_abs:.3f} | "
            f"{r.height_mean:.4f} | {r.height_err_abs:.4f} | {r.tilt_deg_mean:.2f} | "
            f"{r.tilt_deg_max:.2f} | {r.wheel_contact_rate:.3f} | {r.leg_contact_rate:.3f} | "
            f"{r.collision_rate:.3f} | {wheel_first} | {r.wheel_contact_frac_first_1s:.3f} | "
            f"{r.leg_contact_frac_first_1s:.3f} | {r.base_contact_frac_first_1s:.3f} | "
            f"{r.max_wheel_force_n:.1f} | {r.max_leg_force_n:.1f} | "
            f"{r.max_abs_action:.3f} | {r.action_rate:.4f} | "
            f"{r.terminated_rate:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--task", default=DEFAULT_TASK_NAME)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--settle-frac", type=float, default=0.5, help="取末段多大比例做统计")
    parser.add_argument("--cases", default=None, help="逗号分隔的 case 名，缺省跑全部")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--height-conditioned-action-default",
        choices=("true", "false"),
        default=None,
        help="覆盖 action 零点语义，必须与 checkpoint 训练时一致；缺省沿用任务默认。",
    )
    args = parser.parse_args()
    global _ACTION_DEFAULT_OVERRIDE
    if args.height_conditioned_action_default is not None:
        _ACTION_DEFAULT_OVERRIDE = args.height_conditioned_action_default == "true"

    configure_torch_backends()
    cases = _default_cases()
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c.name in wanted]
        if not cases:
            raise SystemExit(f"没有匹配的 case: {sorted(wanted)}")

    env_cfg = _build_env_cfg(args.task, args.num_envs)
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env)
    policy = _load_policy(env, args.checkpoint, args.device, args.task)
    env.reset()

    results: list[CaseResult] = []
    failures: list[dict] = []
    for case in cases:
        # 单个 case 失败不应带走整轮扫描；记录后继续。
        try:
            res = _run_case(base_env, env, policy, case, args.settle_frac)
        except Exception as exc:
            failures.append({"case": case.name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[FAIL] {case.name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        results.append(res)
        print(f"[done] {case.name}", flush=True)
    if failures:
        print(f"\n{len(failures)} 个 case 失败:", flush=True)
        for f in failures:
            print(f"  {f['case']}: {f['error']}", flush=True)

    env.close()
    table = _format_table(results)
    print()
    print(table)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": args.task,
            "checkpoint": str(args.checkpoint),
            "num_envs": args.num_envs,
            "device": args.device,
            "settle_frac": args.settle_frac,
            "results": [asdict(r) for r in results],
            "failures": failures,
        }
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output_json}")


if __name__ == "__main__":
    main()
