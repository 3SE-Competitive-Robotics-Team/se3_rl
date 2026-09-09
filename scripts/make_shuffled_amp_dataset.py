"""生成"时序打乱"的 AMP 专家数据集，用于消融 C2。

问题：A12 里 AMP 在台阶列贡献 +9.1/秒（`reward_weight × step_dt × style_reward`
= 15 × 0.02 × 0.608，折每秒 9.1），占该列全部奖励信号的约 90%，而判别器只把专家/策略
分到 ±0.24（LSGAN 目标是 ±1），`style_reward` 长期贴在 0.6 缓慢下滑。这很像 AMP 并没有在
传递复旦那套爬楼姿态，只是在发一笔"你在台阶列上且活着"的近似常数奖金。

C2 就是这个假设的判别实验：把每条专家序列**在时间维上随机置换**，逐帧的姿态分布一模一样
（同一批帧，只是顺序乱了），但跨帧的时序结构——抬轮/落轮的先后、速度的连贯性——被彻底破坏。
判别器窗口是 5 帧，打乱之后窗口里就是五个互不相关的姿态。

- 若打乱后 stairs_up 照样爬到 6，说明 AMP 给的不是姿态信息，那 9.1 分可以用任何常数替代；
- 若明显变差，说明时序内容确实在起作用，AMP 的模仿是实的。

只动 `sequences`（loader 只消费它，见 motion_loader._validate_sequences_present）与随之
失效的 `transitions`；`lengths`、`feature_names`、契约字段原样保留，帧数与维度不变。

用法：
    uv run python scripts/make_shuffled_amp_dataset.py
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

DEFAULT_SRC = Path("assets/amp/fudan_stairs20_20260907/amp_training.pkl")
DEFAULT_DST = Path("assets/amp/fudan_stairs20_shuffled/amp_training.pkl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = pickle.loads(args.src.read_bytes())
    rng = np.random.default_rng(args.seed)

    sequences = data["sequences"]
    shuffled = []
    for frames in sequences:
        arr = np.asarray(frames)
        shuffled.append(arr[rng.permutation(arr.shape[0])].copy())
    data["sequences"] = shuffled

    # transitions 是"相邻帧拼接"，打乱后它与 sequences 不再自洽；loader 不消费它，
    # 但留着会误导后来的人，所以按新顺序重建。
    if isinstance(data.get("transitions"), list):
        data["transitions"] = [
            np.concatenate([s[:-1], s[1:]], axis=1) if s.shape[0] > 1 else s[:0]
            for s in shuffled
        ]
    data["shuffled_from"] = str(args.src).replace("\\", "/")
    data["shuffle_seed"] = int(args.seed)

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    args.dst.write_bytes(pickle.dumps(data))

    src_all = np.concatenate([np.asarray(s) for s in sequences], axis=0)
    dst_all = np.concatenate(shuffled, axis=0)
    print(f"写出 {args.dst}")
    print(f"  序列 {len(shuffled)} 条，帧 {dst_all.shape[0]}，维度 {dst_all.shape[1]}")
    print(f"  逐帧分布必须不变：均值最大差 {np.abs(src_all.mean(0) - dst_all.mean(0)).max():.3e}"
          f"，标准差最大差 {np.abs(src_all.std(0) - dst_all.std(0)).max():.3e}")
    d_src = np.concatenate([np.diff(np.asarray(s), axis=0) for s in sequences], axis=0)
    d_dst = np.concatenate([np.diff(s, axis=0) for s in shuffled], axis=0)
    print(f"  时序必须被破坏：相邻帧差分的标准差 {np.abs(d_src).std():.4f} -> {np.abs(d_dst).std():.4f}"
          f"（放大 {np.abs(d_dst).std() / max(np.abs(d_src).std(), 1e-9):.1f} 倍）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
