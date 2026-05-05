"""se3-train 的 CLI 入口。"""
import sys


def main():
    from mjlab.scripts.train import main as mjlab_train

    sys.argv[0] = "se3-train"
    mjlab_train()


if __name__ == "__main__":
    main()
