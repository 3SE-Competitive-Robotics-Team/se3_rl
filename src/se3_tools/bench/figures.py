"""每次评测的诊断图：平地跟踪热力图、静站与起步、台阶分级图、跳跃、扰动 plant 矩阵。

图里文字用英文（Pod 上通常没有中文字体）；输入是 cli 写出的 payload（含逐条结果与 params）。
返回 {图名: png 路径}，由 report 贴到 W&B。缺某项数据就不出对应的图。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

CLEAR_COLOR = "#2e9d5b"
STUCK_COLOR = "#e0a43a"
FELL_COLOR = "#d1495b"
ERROR_COLOR = "#9a9a9a"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _ok(r: dict) -> bool:
    return "error" not in r and "skipped" not in r and "params" in r


def _heat(ax, matrix, xlabels, ylabels, *, fmt, cmap, vmin, vmax, title, fell=None):
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_title(title, fontsize=10)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if fell is not None and fell[i, j]:
                ax.text(
                    j,
                    i,
                    "FELL",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    bbox=dict(facecolor=FELL_COLOR, edgecolor="none", pad=1),
                )
            elif np.isfinite(matrix[i, j]):
                ax.text(j, i, fmt.format(matrix[i, j]), ha="center", va="center", fontsize=7)
    return im


def flat_tracking(results: list[dict], path: Path) -> Path | None:
    flat = [r for r in results if r["suite"] == "A" and _ok(r)]
    vx_cases = [r for r in flat if r["params"]["yaw"] == 0.0]
    yaw_cases = [r for r in flat if r["params"]["vx"] == 0.0 and r["params"]["yaw"] != 0.0]
    if not vx_cases:
        return None
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
    speeds = sorted({r["params"]["vx"] for r in vx_cases})
    heights = sorted({r["params"]["height"] for r in vx_cases})
    err = np.full((len(heights), len(speeds)), np.nan)
    herr = np.full_like(err, np.nan)
    fell = np.zeros_like(err, dtype=bool)
    for r in vx_cases:
        i, j = heights.index(r["params"]["height"]), speeds.index(r["params"]["vx"])
        fell[i, j] = r.get("fell", False)
        if not fell[i, j]:
            err[i, j] = r["vx_mean"] - r["params"]["vx"]
            herr[i, j] = r["height_err"] * 100
    lim = max(0.2, float(np.nanmax(np.abs(err))) if np.isfinite(err).any() else 0.2)
    im = _heat(
        axes[0],
        err,
        [f"{v:g}" for v in speeds],
        [f"{h:.2f}" for h in heights],
        fmt="{:+.2f}",
        cmap="RdBu_r",
        vmin=-lim,
        vmax=lim,
        title="vx tracking error (m/s), steady 5 s",
        fell=fell,
    )
    axes[0].set_xlabel("vx command (m/s)")
    axes[0].set_ylabel("height command (m)")
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    hlim = max(2.0, float(np.nanmax(np.abs(herr))) if np.isfinite(herr).any() else 2.0)
    im = _heat(
        axes[1],
        herr,
        [f"{v:g}" for v in speeds],
        [f"{h:.2f}" for h in heights],
        fmt="{:+.1f}",
        cmap="PuOr_r",
        vmin=-hlim,
        vmax=hlim,
        title="height error (cm)",
        fell=fell,
    )
    axes[1].set_xlabel("vx command (m/s)")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    if yaw_cases:
        yaws = sorted({r["params"]["yaw"] for r in yaw_cases})
        yh = sorted({r["params"]["height"] for r in yaw_cases})
        ratio = np.full((len(yh), len(yaws)), np.nan)
        yfell = np.zeros_like(ratio, dtype=bool)
        for r in yaw_cases:
            i, j = yh.index(r["params"]["height"]), yaws.index(r["params"]["yaw"])
            yfell[i, j] = r.get("fell", False)
            if not yfell[i, j]:
                ratio[i, j] = r.get("wz_ratio", np.nan)
        im = _heat(
            axes[2],
            ratio,
            [f"{w:g}" for w in yaws],
            [f"{h:.2f}" for h in yh],
            fmt="{:.2f}",
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.2,
            title="yaw tracking ratio (achieved / command)",
            fell=yfell,
        )
        axes[2].set_xlabel("yaw command (rad/s)")
        fig.colorbar(im, ax=axes[2], shrink=0.8)
    else:
        axes[2].axis("off")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def standing(results: list[dict], path: Path) -> Path | None:
    stand = [r for r in results if r["suite"] == "B" and _ok(r)]
    s30 = [r for r in stand if r["params"]["stand_s"] > 1.0]
    if not s30:
        return None
    plt = _plt()
    labels = [f"h{r['params']['height']:.2f}\nvx{r['params']['vx']:g}" for r in s30]
    x = np.arange(len(s30))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    # 静站段与随后的前进指令无关，同一高度只画一根（取该高度第一条）。
    by_height: dict[float, dict] = {}
    for r in s30:
        by_height.setdefault(r["params"]["height"], r)
    hx = np.arange(len(by_height))
    for ax, key, title, color in (
        (axes[0], "drift_m", "30 s zero-command drift (m)", "#4c78a8"),
        (axes[1], "yaw_drift_deg", "30 s zero-command yaw drift (deg)", "#72b7b2"),
    ):
        ax.bar(hx, [r.get(key, np.nan) for r in by_height.values()], color=color)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(hx, [f"h{h:.2f}" for h in by_height], fontsize=8)
        for i, r in enumerate(by_height.values()):
            if r.get("fell"):
                ax.text(i, 0, "FELL", ha="center", va="bottom", color=FELL_COLOR, fontsize=8)

    def t90(r):
        return min(float(r.get("t90_s", math.inf)), 5.0) if not r.get("fell") else np.nan

    cold = {
        (r["params"]["height"], r["params"]["vx"]): r
        for r in stand
        if r["params"]["stand_s"] <= 1.0
    }
    width = 0.38
    axes[2].bar(
        x - width / 2, [t90(r) for r in s30], width, label="after 30 s stand", color="#f58518"
    )
    axes[2].bar(
        x + width / 2,
        [
            t90(cold[(r["params"]["height"], r["params"]["vx"])])
            if (r["params"]["height"], r["params"]["vx"]) in cold
            else np.nan
            for r in s30
        ],
        width,
        label="cold (1 s stand)",
        color="#bab0ac",
    )
    axes[2].axhline(5.0, color=FELL_COLOR, lw=0.8, ls="--")
    axes[2].set_title("time to 90% of vx command (s, 5 = never)", fontsize=10)
    axes[2].legend(fontsize=8)
    axes[2].set_xticks(x, labels, fontsize=8)
    for i, r in enumerate(s30):
        if r.get("fell"):
            axes[2].text(i, 0, "FELL", ha="center", va="bottom", color=FELL_COLOR, fontsize=8)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _stair_cell(r: dict) -> tuple[str, str]:
    if not _ok(r):
        return ERROR_COLOR, "err"
    if r.get("cleared"):
        return CLEAR_COLOR, f"{r['far_m']:.1f}"
    if r.get("fell"):
        return FELL_COLOR, f"{r['far_m']:.1f}"
    return STUCK_COLOR, f"{r['far_m']:.1f}"


def stairs(results: list[dict], path: Path) -> Path | None:
    stair = [r for r in results if r["suite"] == "C" and "params" in r]
    if not stair:
        return None
    plt = _plt()
    from matplotlib.patches import Patch

    columns = [
        c
        for c in ("stairs_up", "stairs_two_step_up", "stairs_down", "stairs_two_step_down")
        if any(r["params"]["column"] == c for r in stair)
    ]
    fig, axes = plt.subplots(
        1,
        len(columns) + 1,
        figsize=(5.2 * (len(columns) + 1), 5.2),
        layout="constrained",
        gridspec_kw={"width_ratios": [3] + [1.6] * (len(columns) - 1) + [2.2]},
    )
    for ax, column in zip(axes[: len(columns)], columns, strict=True):
        rows_c = [r for r in stair if r["params"]["column"] == column and not r.get("noise_std")]
        levels = sorted({r["params"]["row"] for r in rows_c})

        # 每级的高度指令按该级内排序标成 floor（地形感知下限）/ mid / top，不同级之间才能对齐成同一列。
        ranks = {
            lv: sorted({x["params"]["height"] for x in rows_c if x["params"]["row"] == lv})
            for lv in levels
        }

        def key(r, ranks=ranks):
            p = r["params"]
            heights = ranks[p["row"]]
            names = ["floor", "mid", "top"] if len(heights) == 3 else ["floor", "top"]
            tag = names[heights.index(p["height"])] if len(heights) <= 3 else f"h{p['height']:.2f}"
            if len(heights) == 1:
                tag = f"h{p['height']:.2f}"
            return (p["mode"], p["vx"], tag)

        keys = sorted(
            {key(r) for r in rows_c}, key=lambda k: (k[0] != "running", k[1], k[2] != "floor", k[2])
        )
        for r in rows_c:
            i, j = levels.index(r["params"]["row"]), keys.index(key(r))
            color, text = _stair_cell(r)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=color))
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color="white")
        noisy = [r for r in stair if r["params"]["column"] == column and r.get("noise_std")]
        if noisy:
            j = len(keys)
            for lv in levels:
                runs = [r for r in noisy if r["params"]["row"] == lv]
                if runs:
                    rate = np.mean([bool(r.get("cleared")) for r in runs])
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, levels.index(lv) - 0.5), 1, 1, color=plt.cm.RdYlGn(rate)
                        )
                    )
                    ax.text(
                        j, levels.index(lv), f"{rate:.0%}", ha="center", va="center", fontsize=7
                    )
            keys = [*keys, ("noise", "", "")]
        ax.set_xlim(-0.5, len(keys) - 0.5)
        ax.set_ylim(-0.5, len(levels) - 0.5)
        ax.set_yticks(range(len(levels)), [f"L{lv}" for lv in levels])
        ax.set_xticks(
            range(len(keys)),
            [
                "noise\nσ0.21 x3" if k[0] == "noise" else f"{k[0][:3]}\nvx{k[1]:g}\n{k[2]}"
                for k in keys
            ],
            fontsize=7,
        )
        ax.set_title(column, fontsize=10)
    axes[0].legend(
        handles=[
            Patch(color=CLEAR_COLOR, label="cleared (L∞ ≥ 4 m)"),
            Patch(color=STUCK_COLOR, label="stuck"),
            Patch(color=FELL_COLOR, label="fell"),
        ],
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(0, -0.12),
        ncol=3,
    )
    # 立面姿态分级：每级上行台阶的事件占比
    ax = axes[-1]
    ups = [r for r in stair if r["params"]["column"] == "stairs_up" and not r.get("noise_std")]
    levels = sorted({r["params"]["row"] for r in ups})
    shares = []
    for lv in levels:
        events = [e for r in ups if r["params"]["row"] == lv for e in r.get("riser_events", [])]
        n = max(len(events), 1)
        target = sum(e["target_style"] for e in events) / n
        jumped = sum(e["jumped"] for e in events) / n
        shares.append((target, jumped, (len(events) and 1.0 - target - jumped) or 0.0, len(events)))
    shares = np.array(shares) if shares else np.zeros((0, 4))
    if len(shares):
        y = np.arange(len(levels))
        ax.barh(y, shares[:, 0], color=CLEAR_COLOR, label="target style")
        ax.barh(y, shares[:, 1], left=shares[:, 0], color=FELL_COLOR, label="jumped onto step")
        ax.barh(y, shares[:, 2], left=shares[:, 0] + shares[:, 1], color=STUCK_COLOR, label="other")
        for i, n in enumerate(shares[:, 3]):
            ax.text(1.01, i, f"n={int(n)}", va="center", fontsize=7)
        ax.set_yticks(y, [f"L{lv}" for lv in levels])
        ax.set_xlim(0, 1.15)
    ax.set_title("riser style, stairs_up (deterministic)", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.08), ncol=3)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def jump(results: list[dict], path: Path) -> Path | None:
    jumps = [r for r in results if r["suite"] == "D" and _ok(r)]
    if not jumps:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 4), layout="constrained")
    labels = [f"t{r['params']['target']:g}\nvx{r['params']['vx']:g}" for r in jumps]
    x = np.arange(len(jumps))
    ax.bar(x, [r["params"]["target"] for r in jumps], color="#d9d9d9", label="target clearance")
    colors = [
        FELL_COLOR if r.get("fell") else (CLEAR_COLOR if r.get("success") else STUCK_COLOR)
        for r in jumps
    ]
    ax.bar(
        x,
        [r.get("clearance_peak_m", 0.0) for r in jumps],
        width=0.45,
        color=colors,
        label="achieved",
    )
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("wheel clearance (m)")
    ax.set_title("jump: achieved vs target (green = success)", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def robust(results: list[dict], path: Path) -> Path | None:
    rob = [r for r in results if r["suite"] == "E" and "params" in r]
    if not rob:
        return None
    plt = _plt()
    plants = list(dict.fromkeys(r["plant"] for r in rob))
    tests = list(dict.fromkeys(r["name"].split("/", 1)[1] for r in rob))
    fig, ax = plt.subplots(
        figsize=(1.9 * len(tests) + 2, 0.42 * len(plants) + 1.5), layout="constrained"
    )
    for r in rob:
        i, j = plants.index(r["plant"]), tests.index(r["name"].split("/", 1)[1])
        if not _ok(r):
            color, text = ERROR_COLOR, "err"
        elif r.get("fell"):
            color, text = FELL_COLOR, "FELL"
        elif r["kind"] == "flat":
            color, text = plt.cm.RdYlGn(1 - min(r["vx_err"] / 0.5, 1)), f"e{r['vx_err']:.2f}"
        elif r["kind"] == "standing_drift":
            color, text = plt.cm.RdYlGn(1 - min(r["drift_m"] / 1.0, 1)), f"{r['drift_m']:.2f} m"
        elif r["kind"] == "push":
            color, text = (
                plt.cm.RdYlGn(1 - min(r["tilt_max_deg"] / 45, 1)),
                f"{r['tilt_max_deg']:.0f}°",
            )
        else:
            color, text = (
                (CLEAR_COLOR if r.get("cleared") else STUCK_COLOR),
                f"{r.get('far_m', 0):.1f} m",
            )
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=color))
        ax.text(j, i, text, ha="center", va="center", fontsize=7)
    ax.set_xlim(-0.5, len(tests) - 0.5)
    ax.set_ylim(len(plants) - 0.5, -0.5)
    ax.set_xticks(range(len(tests)), tests, fontsize=8)
    ax.set_yticks(range(len(plants)), plants, fontsize=8)
    ax.set_title("perturbed plants (vx err / drift / push tilt / stairs L6 distance)", fontsize=10)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


FIGURES = {
    "A_flat_tracking": flat_tracking,
    "B_standing": standing,
    "C_stairs": stairs,
    "D_jump": jump,
    "E_robust": robust,
}


def render_all(payload: dict, out_dir: Path) -> dict[str, str]:
    """出全部图；单张失败只跳过并在返回里缺省。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    made: dict[str, str] = {}
    for name, fn in FIGURES.items():
        try:
            path = fn(payload["results"], out_dir / f"{name}.png")
        except Exception as error:
            print(f"[se3-bench] 出图 {name} 失败：{error}", flush=True)
            continue
        if path is not None:
            made[name] = str(path)
    return made


__all__ = ["FIGURES", "render_all"]
