"""se3-train 的 CLI 入口。"""

from __future__ import annotations

import sys


def main() -> None:
    """训练入口,捕获 KeyboardInterrupt 实现优雅退出。"""
    sys.argv[0] = "se3-train"

    try:
        from mjlab.scripts.train import main as mjlab_train

        mjlab_train()
    except KeyboardInterrupt:
        print("\n✓ 训练已停止")
        sys.exit(0)


if __name__ == "__main__":
    main()
