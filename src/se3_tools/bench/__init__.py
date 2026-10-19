"""训练后自动评测（benchmark）：在原生 MuJoCo + se3_runtime 部署链路上回放导出的 ONNX，按固定场景出指标。

只读 ONNX 契约，不依赖训练 env：测的是要上车的那条链路，sim2sim gap 一并计入。
场景、指令网格、种子与判据都固定并带版本号（`spec.BENCH_VERSION`），改动场景要升版本，不同版本的数字不直接比。
入口：`se3-bench run / compare`（见 cli）；训练中由 `se3_train.bench_hook` 在保存 checkpoint 时旁路触发。
"""

from se3_tools.bench.spec import BENCH_VERSION

__all__ = ["BENCH_VERSION"]
