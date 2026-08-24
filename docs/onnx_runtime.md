# ONNX metadata producer 与 se3-sim2x 接入

新导出的策略使用与 Kyber 相同组织方式的 `se3.meta.v2` descriptor。ONNX 自身负责描述
网络输入、输出和 recurrent state，metadata 只保留部署需要且无法从 graph 获得的信息。
`se3-sim2x` 优先读取 v2，同时继续兼容已有的 `se3.meta.v1` artifact。

```text
运行中的训练环境 + actor groups
→ build_deployment_onnx_metadata()
→ checkpoint 保存时导出 onnx/model_N.onnx
→ 嵌入单个 se3.meta.v2 JSON
→ PolicyBundle 从 ONNX graph 推断 MLP/GRU 与 I/O shape
→ v2 descriptor 归一化为 SerialLeg runtime contract
→ PolicyRuntime + PolicyControlLoop
→ MuJoCo/Viser adapter（真机 adapter 暂缓）
```

## v2 descriptor

顶层结构直接沿用 Kyber：

```text
meta
├── schema_name
├── assets
├── training
├── env_groups
└── sim                 # sim_dt / decimation / step_dt / inference_hz
robot
├── policy_joint_names
├── KP / KD / armature
├── effort_limit / velocity_limit
└── init_state / links
commands
└── velocity_height.dimension
policy_io
├── groups.*.terms      # name / scale / history_length / clip / params
├── references          # joint_pos_ref / joint_vel_ref
└── action              # name / scale / offset / clip
```

以下信息不再重复写进 JSON：

- ONNX inputs、outputs、opset 和完整 graph 副本；
- observation 的 width、source、transform、field name 等 SerialLeg 固定语义；
- contract hash 和 MJCF 文件 hash；
- MuJoCo solver 全量配置；
- action channel、decoder、delay 等 runtime 固定语义；
- checkpoint 文件名。

`meta.training.is_rnn` 只作为 graph 的交叉校验。实际策略类型按 ONNX I/O 判定：

- `obs → actions`：MLP，包括单帧 MLP 和 History-MLP；
- `obs + h_in → actions + h_out`：GRU。

因此网络输入输出维度和 GRU hidden shape 始终以 ONNX graph 为唯一事实来源。

## Producer 职责

[`src/se3_train/onnx_metadata.py`](../src/se3_train/onnx_metadata.py) 从 live env 读取：

- inference 时序；
- actor observation term 顺序、scale、clip 和 history length；
- command 名称和维度；
- policy-order 关节、PD、armature、effort/velocity limit；
- action scale、offset 和 clip。

[`src/se3_train/runner.py`](../src/se3_train/runner.py) 在 checkpoint 保存后自动导出同名
ONNX，并只补充 `policy_iteration`、`is_rnn`、`export_timestamp` 和 `export_tag`。临时文件
完成 graph/descriptor 一致性检查后才会原子替换最终文件。

## Runtime 固定语义

Kyber-style descriptor 有意不携带项目内部实现细节。`se3.meta.v2` 对 SerialLeg 的解释由
runtime 版本固定：

- `asset_name=serialleg_closed_chain_v3_train_obb_trim` 映射到仓库内正式 MJCF；
- observation term name 映射到固定 source/transform/width；
- History-MLP 使用 per-term buffer、`oldest_to_newest`、`term_major`；
- action 使用闭链主动杆 decoder 和高度条件默认姿态；
- action delay 使用共享默认的 4–6 ms，并按 physics tick 推进；
- command 默认值为 `height=0.22`，其余字段为 `0`；
- command 交互控件边界由 runtime 固定，只用于输入校验和 Viser，不代表训练范围；
- 未导出的 MuJoCo solver 字段使用 MJLab 默认配置。

v2 不含资产 hash。单模型加载通过固定 `asset_name` 选取 MJCF；Viser 热切换要求候选
session 解析到同一个 MJCF 路径。旧 v1 仍按原规则严格校验 contract hash 和 MJCF hash。

## 使用

首次 clone 后执行：

```bash
git submodule update --init --recursive
uv sync
```

常驻 Viser ONNX 浏览器：

```bash
./scripts/run_sim2x.sh
```

它持续扫描：

```text
logs/rsl_rl/<experiment>/<run_id>/onnx/*.onnx
```

新模型完成 metadata、ONNX I/O、资产和 session reset 校验后才会替换当前模型。单 artifact
诊断入口仍为：

```bash
uv run se3-sim2x-mujoco --onnx /path/to/model.onnx --viewer viser --steps 0
```

完整 runtime 状态、动作延迟和 adapter 边界见
[`submodules/se3-sim2x/docs/runtime-contract.md`](../submodules/se3-sim2x/docs/runtime-contract.md)。
