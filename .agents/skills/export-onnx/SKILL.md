---
name: export-onnx
description: 将 SE3 GRU actor checkpoint 导出为 ONNX 格式用于 NX 真机部署。当需要导出 ONNX 模型、转换 checkpoint、或准备部署模型时使用。支持 34D 观测输入、6D 动作输出的 GRU 策略。
---

# SE3 ONNX 模型导出

将训练产出的 PyTorch checkpoint（.pt）转换为 ONNX 单文件（.onnx），仅导出 actor 网络。

## 契约

当前固定资产标准：

| 参数 | 值 |
|------|-----|
| 观测维度 | 34 |
| 动作维度 | 6 |
| 网络结构 | GRU(512, 1层) → MLP(512→256→128→6), ELU |

模型接口：

- 输入 `obs` (batch×34) + `hidden_in` (1×batch×512)
- 输出 `action` (batch×6) + `hidden_out` (1×batch×512)
- 首帧 `hidden_in` 传零，后续每步用上一帧的 `hidden_out`

## 使用

```bash
uv run python .agents/skills/export-onnx/scripts/export.py [--checkpoint <path>] [--output <path>] [--opset 18]
```

默认 checkpoint 为 `assets/base_model/` 下的 `.pt` 文件，输出同目录 `.onnx`。

## 当前资产

| 文件 | 说明 |
|------|------|
| `assets/base_model/model_2999_flat_20260702.pt` | 原始 checkpoint（29MB） |
| `assets/base_model/model_2999_flat_20260702.onnx` | ONNX 单文件（4.9MB） |
