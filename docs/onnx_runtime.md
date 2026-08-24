# ONNX metadata producer 与 se3-sim2x 接入

训练仓库负责从运行中的 MJLab 环境生成部署契约，并在保存 checkpoint 时导出单个带
`se3.meta.v1` JSON 的 ONNX artifact。契约消费、策略状态、动作延迟、执行器控制和平台
adapter 已由 [`se3-sim2x`](../submodules/se3-sim2x/README.md) submodule 统一维护。

```text
运行中的训练环境 + actor groups
→ build_deployment_onnx_metadata()
→ checkpoint 保存时导出 onnx/model_N.onnx
→ 嵌入单个 se3.meta.v1 JSON
→ se3-sim2x: PolicyBundle 严格校验 schema/hash/ONNX I/O
→ se3-sim2x: PolicyRuntime + PolicyControlLoop
→ MuJoCo/Viser adapter（真机 adapter 暂缓）
```

## 主仓库职责

- [`src/se3_train/onnx_metadata.py`](../src/se3_train/onnx_metadata.py) 从 live env 读取
  observation/history、command ranges、action delay、actuator PD/T-N 和 MuJoCo solver 配置。
- [`src/se3_train/runner.py`](../src/se3_train/runner.py) 在 checkpoint 保存成功后自动生成
  同名 `onnx/model_N.onnx`，临时文件完成嵌入和校验后再原子替换最终文件。
- 训练任务和机器人资产继续留在本仓库；runtime 不得反向依赖 MJLab 或 `se3_train`。

Metadata 必须来自当前 live env，不能从全局默认配置推测。这样 recovery、stair 等任务级
actuator override，以及 MLP、History-MLP、GRU 的真实 I/O 都会进入 artifact。

## Submodule 接入

首次 clone 或切换到包含该 submodule 的提交后执行：

```bash
git submodule update --init --recursive
uv sync
```

根项目通过 uv workspace editable dependency 使用 `se3-sim2x`，源码位置为
`submodules/se3-sim2x`。日常运行启动常驻 Viser ONNX 浏览器：

```bash
./scripts/run_sim2x.sh
```

命令必须从主仓库根目录执行。启动成功后终端会打印 Viser URL，默认是
`http://127.0.0.1:8080/`；端口被占用时以实际输出为准。服务保持前台运行，使用
`Ctrl-C` 停止。

它持续扫描固定目录布局：

```text
logs/rsl_rl/
└── <experiment>/
    └── <run_id>/
        └── onnx/
            ├── model_0.onnx
            └── model_N.onnx
```

Viser 的 `Models` 页签提供 `Experiment → Run ID → ONNX` 三级选择。目录每秒刷新，
新导出的模型不需要重启服务；候选模型通过 metadata、ONNX I/O 和 MJCF hash 校验并完成
session reset 后才会替换当前模型。第一版只允许在同一个 MJCF hash 内热切换，加载失败时
继续运行原模型。

模型列表为空时，先确认训练 run 下存在 `onnx/model_N.onnx`。这里加载的是训练保存
checkpoint 时自动导出的 ONNX，不直接加载 `model_N.pt`。较早生成且缺少完整
`meta.sim.mujoco` 等 v1 字段的 artifact 会被严格拒绝，需要从 checkpoint 重新导出。

等价命令为 `uv run se3-sim2x-browser`。单个 artifact 的诊断入口
`se3-sim2x-mujoco --onnx ...` 仍然保留。

完整 runtime schema、History-MLP/GRU 状态语义、physics-tick delay 和 adapter 边界以
[`submodules/se3-sim2x/docs/runtime-contract.md`](../submodules/se3-sim2x/docs/runtime-contract.md)
为唯一文档来源。

开发期间早于 actuator curve 与 MuJoCo solver 契约生成的临时 ONNX 不满足完整 v1 schema，
必须从 checkpoint 重新导出；runtime 会明确拒绝缺字段 artifact，不会猜测旧默认值。
