"""Rough benchmark 的场景定义（版本 rough-v1）：五项评测 A–E 的指令网格、时长、判据与 quick / full 两档。

- A 平地跟踪：vx × 高度网格 + yaw 网格（含包络顶端 ±12 rad/s），测稳态跟踪、高度误差与 5–15 Hz 俯仰抖动。
- B 静站与起步：零指令静站 30 s 的漂移，静站后阶跃前进到 90% 指令的时间（对照只站 1 s 的冷起步）。
- C 台阶：训练地形台阶四列（上行 / 二级上行 / 下行 / 二级下行）逐级，助跑开局与静止开局；上行列记立面事件。
- D 跳跃：平地按参考目标离地间隙起跳，测成功率、离地间隙误差与落地恢复。
- E 鲁棒：WideDR 那组 11 种扰动 plant × 平地跟踪 / 静站 / 侧推 / 第 6 级台阶。

训练中每 1000 轮与训练结束都跑 full 档（Pod 128 核 32 进程约 15 s）；quick 档只用于手动快速检查。
改动任何场景、网格或判据都必须升 `BENCH_VERSION`。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from se3_tools.bench.sim import Plant

BENCH_VERSION = "rough-v1"

# ---- 判据常量 ----
STAIR_CLEAR_DISTANCE_M = 4.0
"""与 Rough 训练 terrain_levels 升级判据一致：机身到出生点切比雪夫距离 ≥ 4.0 m 算走出该块。"""
STAIR_EPISODE_S = 20.0
RUNNING_START_HEIGHT_M = 0.24
"""助跑开局：前 RUNNING_START_S 秒用低姿态高度指令起步，再切到目标高度（把「爬升能力」与「高姿态静止起步死锁」分开）。"""
RUNNING_START_S = 0.6
ACTION_NOISE_STD = 0.21
"""训练后期 Mean action std ≈ 0.21；台阶带噪声回放用，避免确定性回放卡在不动点时误判为不会爬。"""
JUMP_STAND_HEIGHT_M = 0.22
JUMP_SUCCESS_RATIO = 0.8
"""离地间隙达到目标的 80% 且落地 2 s 后未摔、高度回到 ±3 cm 算一次成功跳跃。"""

PLANTS: dict[str, Plant] = {
    "nominal": Plant(),
    "com_x+5cm": Plant(com=(0.05, 0.0, 0.0)),
    "com_x-5cm": Plant(com=(-0.05, 0.0, 0.0)),
    "com_z+5cm": Plant(com=(0.0, 0.0, 0.05)),
    "mass+3kg": Plant(mass=3.0),
    "spring×1.5": Plant(spring=1.5),
    "kpkd×0.9": Plant(kp=0.9, kd=0.9),
    "delay0": Plant(delay_steps=0),
    "delay2(10ms)": Plant(delay_steps=2),
    "friction0.3": Plant(friction=0.3),
    "worst": Plant(com=(0.05, 0.0, 0.0), mass=3.0, spring=1.5, delay_steps=2),
}


@dataclass(frozen=True)
class StairRules:
    """台阶列指令下限规则（训练端 commands.RoughCommandTerm 的地形感知高度下限）。"""

    height_range: tuple[float, float]
    clearance: float
    body_bottom_offset: float
    step_heights: dict[str, tuple[float, float]]
    """列名 → step_height_range（第 r 行台阶高 = lo + r/9 × (hi − lo)）。"""
    num_rows: int

    def step_height(self, column: str, row: int) -> float:
        lo, hi = self.step_heights[column]
        return lo + (row / max(self.num_rows - 1, 1)) * (hi - lo)

    def height_floor(self, column: str, row: int) -> float:
        low, high = self.height_range
        need = self.step_height(column, row) + self.clearance - self.body_bottom_offset
        return round(min(max(low, need), high), 3)


def rough_stair_rules() -> StairRules:
    """从当前 checkout 的 Rough 配置读规则（只在主进程调用一次，worker 不导入 se3_train）。"""
    from se3_train.tasks.rough import commands as rough_commands
    from se3_train.tasks.rough.terrains import rough_terrains_cfg

    cfg = rough_terrains_cfg()
    steps = {
        name: tuple(float(v) for v in cfg.sub_terrains[name].step_height_range)
        for name in ("stairs_up", "stairs_two_step_up", "stairs_down", "stairs_two_step_down")
    }
    return StairRules(
        height_range=tuple(float(v) for v in rough_commands.ROUGH_STAIR_HEIGHT_RANGE),
        clearance=float(rough_commands.ROUGH_TERRAIN_HEIGHT_CLEARANCE),
        body_bottom_offset=float(rough_commands.ROUGH_BODY_COLLISION_BOTTOM_OFFSET),
        step_heights=steps,
        num_rows=int(cfg.num_rows),
    )


@dataclass
class Case:
    """一条评测：kind 决定执行函数（见 cases.run_case），params 为场景参数。"""

    suite: str
    name: str
    kind: str
    params: dict = field(default_factory=dict)
    plant: str = "nominal"
    scene: str | None = None
    """None = 机器人 MJCF 自带平地；"stairs" = 本次生成的台阶网格。"""
    spawn: tuple[float, float, float] | None = None
    noise_std: float = 0.0
    seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def _flat_cases(full: bool) -> list[Case]:
    cases: list[Case] = []
    if full:
        speeds = (-0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4)
        heights = (0.22, 0.26, 0.30, 0.34, 0.38)
        yaws = (-12.0, -9.0, -6.0, -3.0, -1.0, 1.0, 3.0, 6.0, 9.0, 12.0)
        yaw_heights = (0.22, 0.30, 0.38)
    else:
        speeds = (-0.4, 0.8, 1.6, 2.4)
        heights = (0.22, 0.30, 0.38)
        yaws = (-12.0, 3.0, 6.0, 12.0)
        yaw_heights = (0.30,)
    for h in heights:
        for v in speeds:
            cases.append(
                Case("A", f"flat_vx{v:+.1f}_h{h:.2f}", "flat", dict(vx=v, yaw=0.0, height=h))
            )
    for h in yaw_heights:
        for w in yaws:
            cases.append(
                Case("A", f"flat_yaw{w:+.0f}_h{h:.2f}", "flat", dict(vx=0.0, yaw=w, height=h))
            )
    if full:
        for h in (0.22, 0.30):
            cases.append(
                Case("A", f"flat_vx+1.0_yaw+3_h{h:.2f}", "flat", dict(vx=1.0, yaw=3.0, height=h))
            )
    return cases


def _standing_cases(full: bool) -> list[Case]:
    heights = (0.22, 0.30, 0.38) if full else (0.30, 0.38)
    speeds = (1.0, 1.6) if full else (1.0,)
    cases: list[Case] = []
    for h in heights:
        for v in speeds:
            for stand_s in (30.0, 1.0):
                tag = "stand30" if stand_s > 1.0 else "cold"
                cases.append(
                    Case(
                        "B",
                        f"{tag}_h{h:.2f}_vx{v:.1f}",
                        "standing",
                        dict(height=h, vx=v, stand_s=stand_s),
                    )
                )
    return cases


def _stair_cases(full: bool, rules: StairRules) -> list[Case]:
    cases: list[Case] = []
    up_rows = tuple(range(1, 10)) if full else (3, 6, 9)
    down_rows = (3, 6, 9)
    up_speeds = (0.8, 1.6) if full else (1.2,)
    down_speeds = (0.8, 1.6) if full else (0.8,)
    for column in ("stairs_up", "stairs_two_step_up"):
        rows = up_rows if column == "stairs_up" else down_rows
        for r in rows:
            floor = rules.height_floor(column, r)
            heights = sorted({floor, round((floor + 0.38) / 2, 3), 0.38}) if full else (floor, 0.38)
            for v in up_speeds:
                for h in heights:
                    for mode in ("running", "static"):
                        cases.append(
                            Case(
                                "C",
                                f"{column}_r{r}_vx{v:.1f}_h{h:.2f}_{mode}",
                                "stairs",
                                dict(column=column, row=r, vx=v, height=h, mode=mode),
                                scene="stairs",
                            )
                        )
        if full and column == "stairs_up":
            # 带训练式探索噪声的重复：确定性回放可能卡在不动点，噪声下能走才说明能力还在。
            for r in up_rows:
                floor = rules.height_floor(column, r)
                for seed in range(3):
                    cases.append(
                        Case(
                            "C",
                            f"{column}_r{r}_vx1.2_h{floor:.2f}_running_noise{seed}",
                            "stairs",
                            dict(column=column, row=r, vx=1.2, height=floor, mode="running"),
                            scene="stairs",
                            noise_std=ACTION_NOISE_STD,
                            seed=seed + 1,
                        )
                    )
    for column in ("stairs_down", "stairs_two_step_down"):
        for r in down_rows:
            for v in down_speeds:
                cases.append(
                    Case(
                        "C",
                        f"{column}_r{r}_vx{v:.1f}_h0.30",
                        "stairs",
                        dict(column=column, row=r, vx=v, height=0.30, mode="running"),
                        scene="stairs",
                    )
                )
    return cases


def _jump_cases(full: bool) -> list[Case]:
    targets = (0.2, 0.3, 0.4, 0.5) if full else (0.3, 0.5)
    speeds = (0.0, 0.8) if full else (0.0,)
    return [
        Case(
            "D",
            f"jump_t{t:.1f}_vx{v:.1f}",
            "jump",
            dict(target=t, vx=v),
        )
        for t in targets
        for v in speeds
    ]


def _robust_cases(rules: StairRules) -> list[Case]:
    floor6 = rules.height_floor("stairs_up", 6)
    tests = [
        ("flat_vx1.0_h0.30", "flat", dict(vx=1.0, yaw=0.0, height=0.30), None),
        ("stand_h0.38", "standing_drift", dict(height=0.38, seconds=8.0), None),
        ("push_h0.30", "push", dict(height=0.30, dv=1.5), None),
        (
            f"stairs_up_r6_h{floor6:.2f}",
            "stairs",
            dict(column="stairs_up", row=6, vx=1.2, height=floor6, mode="running"),
            "stairs",
        ),
    ]
    return [
        Case("E", f"{plant}/{name}", kind, params, plant=plant, scene=scene)
        for plant in PLANTS
        for name, kind, params, scene in tests
    ]


def build_cases(suite: str, rules: StairRules) -> list[Case]:
    """suite = "quick"（A 粗网格、B、C 第 3/6/9 级、D）或 "full"（全部五项）。"""
    if suite not in ("quick", "full"):
        raise ValueError(f"未知 suite {suite!r}")
    full = suite == "full"
    cases = [
        *_flat_cases(full),
        *_standing_cases(full),
        *_stair_cases(full, rules),
        *_jump_cases(full),
    ]
    if full:
        cases += _robust_cases(rules)
    return cases


def stair_rows(cases: list[Case]) -> tuple[int, ...]:
    return tuple(sorted({int(c.params["row"]) for c in cases if c.scene == "stairs"}))


__all__ = [
    "ACTION_NOISE_STD",
    "BENCH_VERSION",
    "PLANTS",
    "Case",
    "StairRules",
    "build_cases",
    "rough_stair_rules",
    "stair_rows",
]
