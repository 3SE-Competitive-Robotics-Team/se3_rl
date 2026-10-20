"""训练后自动评测（benchmark）：在原生 MuJoCo + se3_runtime 部署链路上回放导出的 ONNX，按固定场景出指标。

只读 ONNX 契约，不依赖训练 env：测的是要上车的那条链路，sim2sim gap 一并计入。
场景、指令网格、种子与判据都固定并带版本号（`spec.BENCH_VERSION`），改动场景要升版本，不同版本的数字不直接比。
入口：`se3-bench run / compare`（见 cli）；训练中由 `se3_train.bench_hook` 在保存 checkpoint 时旁路触发。

runtime 版本：优先导入本仓库自己检出的 `submodules/se3-sim2x/src`，与训练代码配套的子模块指针为准，
而不是 venv 里 editable 安装指向的那份（Pod 上 worktree 软链主 checkout 的 .venv，后者可能是别的分支的旧版）。
子模块没检出时退回 venv 里的版本。
"""

import sys
from pathlib import Path

_SIM2X_SRC = Path(__file__).resolve().parents[3] / "submodules" / "se3-sim2x" / "src"
if (_SIM2X_SRC / "se3_runtime" / "__init__.py").is_file() and "se3_runtime" not in sys.modules:
    sys.path.insert(0, str(_SIM2X_SRC))

from se3_tools.bench.spec import BENCH_VERSION  # noqa: E402

__all__ = ["BENCH_VERSION"]
