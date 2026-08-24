"""Viser + 终端交互式关节诊断工具。

通过终端命令逐个测试关节和轮子方向，同时在 Viser 显示 MuJoCo 状态。
远程无 GUI 检查可使用 ``--viewer none``。

用法:
    uv run se3-joint-diag
    uv run se3-joint-diag --mode sweep       # 自动扫描所有关节
    uv run se3-joint-diag --mode interactive  # 手动输入力矩测试
"""

import argparse

import mujoco

from se3_runtime_mujoco import MujocoSceneViewer
from se3_shared import JointGroup, RobotConfig

MJCF_PATH = "assets/robots/serialleg/mjcf/serialleg_closed_chain_v3_train_obb_trim.xml"

ACTUATOR_NAMES = list(JointGroup.POLICY_JOINT_NAMES)
_ROBOT_CFG = RobotConfig()


def load_model(mjcf_path: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """加载正式闭链模型并添加六个诊断力矩执行器。"""
    spec = mujoco.MjSpec.from_file(mjcf_path)
    for joint_name in JointGroup.POLICY_JOINT_NAMES:
        actuator = spec.add_actuator()
        actuator.name = f"{joint_name}_diag_motor"
        actuator.target = joint_name
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
        actuator.gainprm[0] = 1.0
    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data


def reset_standing(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    height: float = 0.22,
) -> None:
    """把模型重置到共享配置的站立姿态。"""
    mujoco.mj_resetData(model, data)
    data.qpos[2] = height
    for joint_name, angle in _ROBOT_CFG.default_model_joint_pos.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            data.qpos[model.jnt_qposadr[joint_id]] = float(angle)
    mujoco.mj_forward(model, data)


def get_base_state(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, object]:
    """读取终端交互所需的基座状态。"""
    base_id = model.body("base_link").id
    return {
        "x": data.qpos[0],
        "y": data.qpos[1],
        "z": data.xpos[base_id, 2],
        "quat": data.qpos[3:7].tolist(),
    }


def _step_simulation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    steps: int,
    viewer: MujocoSceneViewer | None,
    sequence_step: int,
) -> int:
    """推进仿真并逐步更新 Viser，返回新的全局步号。"""
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.update(data)
        sequence_step += 1
    return sequence_step


def mode_sweep(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    args: argparse.Namespace,
    viewer: MujocoSceneViewer | None,
) -> None:
    """自动扫描每个 actuator 的正/负方向响应。"""
    print("=" * 60)
    print("SWEEP: 逐个 actuator 施力,观察响应")
    print("=" * 60)
    print(f"{'actuator':15s} | {'ctrl':>5s} | {'qvel':>10s} | {'base_dx':>8s} | {'base_dz':>8s}")
    print("-" * 60)

    sequence_step = 0
    base_id = model.body("base_link").id
    for act_id in range(model.nu):
        for ctrl_val in [+5.0, -5.0]:
            reset_standing(model, data, args.height)
            x0, z0 = data.qpos[0], data.xpos[base_id, 2]

            data.ctrl[act_id] = ctrl_val
            sequence_step = _step_simulation(
                model,
                data,
                args.steps,
                viewer,
                sequence_step,
            )

            jnt_id = model.actuator(act_id).trnid[0]
            dof_adr = model.jnt_dofadr[jnt_id]
            qvel = data.qvel[dof_adr]
            dx = data.qpos[0] - x0
            dz = data.xpos[base_id, 2] - z0

            print(
                f"{ACTUATOR_NAMES[act_id]:15s} | {ctrl_val:+5.1f} | "
                f"{qvel:+10.4f} | {dx:+8.4f} | {dz:+8.4f}"
            )

    print()
    print("解读:")
    print("  qvel > 0: 关节正方向转动")
    print("  base_dx > 0: 机器人向 +X 方向移动")
    print("  base_dz > 0: 机器人升高")


def mode_interactive(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    args: argparse.Namespace,
    viewer: MujocoSceneViewer | None,
) -> None:
    """手动输入 ctrl 值测试。"""
    print("=" * 60)
    print("INTERACTIVE: 手动输入 ctrl 值")
    print("=" * 60)
    print("Actuator 列表:")
    for i, name in enumerate(ACTUATOR_NAMES):
        r = model.actuator(i).ctrlrange
        print(f"  [{i}] {name:15s} range=[{r[0]:.1f}, {r[1]:.1f}]")
    print()
    print("输入格式: <actuator_id> <ctrl_value> <steps>")
    print("输入 'q' 退出, 'r' 重置, 's' 显示状态, 'all <v> <steps>' 给所有施力")
    print()

    reset_standing(model, data, args.height)
    sequence_step = 0
    if viewer is not None:
        viewer.update(data)
    sequence_step += 1

    while True:
        state = get_base_state(model, data)
        try:
            cmd = input(f"[z={state['z']:.4f} x={state['x']:.4f}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == "q":
            break
        elif cmd == "r":
            reset_standing(model, data, args.height)
            print("  -> 重置完成")
            continue
        elif cmd == "s":
            print(f"  base: {state}")
            for i in range(model.nu):
                jnt_id = model.actuator(i).trnid[0]
                qpos_adr = model.jnt_qposadr[jnt_id]
                dof_adr = model.jnt_dofadr[jnt_id]
                print(
                    f"  {ACTUATOR_NAMES[i]:15s}: "
                    f"pos={data.qpos[qpos_adr]:+.4f} "
                    f"vel={data.qvel[dof_adr]:+.4f} "
                    f"ctrl={data.ctrl[i]:+.4f}"
                )
            continue
        elif cmd.startswith("all"):
            parts = cmd.split()
            val = float(parts[1]) if len(parts) > 1 else 5.0
            steps = int(parts[2]) if len(parts) > 2 else 50
            data.ctrl[:] = val
            sequence_step = _step_simulation(
                model,
                data,
                steps,
                viewer,
                sequence_step,
            )
            state_after = get_base_state(model, data)
            print(
                f"  -> all ctrl={val}, {steps} steps: z={state_after['z']:.4f} x={state_after['x']:.4f}"
            )
            continue

        parts = cmd.split()
        if len(parts) < 2:
            print("  格式: <id> <ctrl> [steps]")
            continue

        act_id = int(parts[0])
        ctrl_val = float(parts[1])
        steps = int(parts[2]) if len(parts) > 2 else 50

        data.ctrl[:] = 0
        data.ctrl[act_id] = ctrl_val
        sequence_step = _step_simulation(
            model,
            data,
            steps,
            viewer,
            sequence_step,
        )

        state_after = get_base_state(model, data)
        jnt_id = model.actuator(act_id).trnid[0]
        dof_adr = model.jnt_dofadr[jnt_id]
        print(
            f"  -> {ACTUATOR_NAMES[act_id]} ctrl={ctrl_val:+.1f}, {steps} steps: "
            f"z={state_after['z']:.4f} x={state_after['x']:.4f} "
            f"jvel={data.qvel[dof_adr]:+.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Viser + 终端交互式关节诊断工具")
    parser.add_argument("--mjcf", default=MJCF_PATH, help="MJCF 模型路径")
    parser.add_argument("--height", type=float, default=0.22, help="初始基座高度")
    parser.add_argument("--steps", type=int, default=50, help="sweep 模式每次仿真步数")
    parser.add_argument(
        "--viewer",
        choices=["viser", "none"],
        default="viser",
        help="动态诊断可视化后端",
    )
    parser.add_argument(
        "--mode",
        choices=["sweep", "interactive"],
        default="sweep",
        help="运行模式",
    )
    args = parser.parse_args()

    model, data = load_model(args.mjcf)

    viewer = MujocoSceneViewer(model, label="se3_joint_diag") if args.viewer == "viser" else None
    try:
        if args.mode == "sweep":
            mode_sweep(model, data, args, viewer)
        elif args.mode == "interactive":
            mode_interactive(model, data, args, viewer)
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
