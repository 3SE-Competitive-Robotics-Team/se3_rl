# 训练指南

## 环境准备

### 机器人模型文件

训练和仿真用的 MJCF 与 mesh 文件已放在 `assets/robots/serialleg/`。重新导出模型时，保持 MJCF 中的关节名和 mesh 相对路径不变。

当前所有注册的 `SE3-WheelLegged-*` 训练任务和 sim2sim 都固定使用真实闭链 OBB 模型，保持 policy 的 `[LF, LB, RF, RB, l_wheel, r_wheel]` 主动杆语义。MJCF 目录只保留 `serialleg_closed_chain_v3_train_obb_trim.xml`；旧高保真模型和两个四连杆 surrogate 均已删除。

policy 动作顺序固定为 `[LF, LB, RF, RB, l_wheel, r_wheel]`，其中 `LB/RB` 对应 `l_drive_bar_Joint/r_drive_bar_Joint`。闭链限位语义是同侧两根主动杆夹角；当前装配分支下左腿为 `LF-LB`，右腿为 `RB-RF`，允许范围为 `0.0~1.50954 rad`（`129.95° - 43.46° = 86.49°`），对应腿长下限约 `0.135 m`；当前默认夹角为 `1.31668 rad`，不是后主动杆的绝对角。

当前无气弹簧默认站姿按“腿长 0.16 m、base_link 距地约 0.22 m、轮心落在整机质心投影下、base/腿部几何离地”的几何平衡点重标定：

```text
default_dof_pos = [-0.275422946189, -1.592100148957, 0.275422946189, 1.592100148957, 0.0, 0.0]
default_output_knee_pos = [-1.242259649307, 1.242259649307]
default_coupler_pos = [1.401266340000, -1.401269410000]
default_base_height = 0.22 m
```

这只是两轮倒立系统的 reset 几何基点；零轮速开环 PD 仍不能替代策略的轮子平衡反馈。

闭链模型基础检查：

```bash
uv run python scripts/check_closedchain_model.py
uv run se3-joint-viewer --geom-view both
```

### 安装依赖

```bash
uv sync
```

## 训练命令

### Smoke 模式（验证环境）

修改训练代码后先跑一次 smoke，5 轮训练，不上传日志，确认环境不崩：

```bash
# CPU smoke（任何机器都能跑）
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1 --gpu-ids None

# GPU smoke
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1024
```

### Viser 训练值守（必开）

远程训练的连接方式、checkpoint 交换通道和本地目录由选定的 machine profile 决定，
公用训练文档不提供某台机器的默认入口。ONNX 生成后，值守侧统一运行 native MuJoCo
`se3-sim2x`，并确认策略、模型和碰撞地形都与训练契约一致。

```bash
./scripts/run_sim2x.sh
```

打开终端输出的 URL，在 `Models` 页签按 `Experiment → Run ID → ONNX` 切换。服务会持续
监听 `logs/rsl_rl`，无需为每个 checkpoint 重启。

当前 run 尚未生成第一个 ONNX 时，先检查训练日志和 MuJoCo 模型；ONNX 出现后再做策略
值守，不用零动作 viewer 冒充策略验收。远端产物来源和同步方式从当前 machine profile 获取。

### 本地/单卡 GPU 训练

需要 NVIDIA GPU + CUDA 12.4+。本节是本地/单卡示例，环境数使用 1024；远程机器的
推荐环境数和 CUDA 特殊配置以对应 machine profile 为准。

```bash
# 平地训练
uv run se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1024

# 崎岖地形训练
uv run se3-train SE3-WheelLegged-Rough --env.scene.num-envs 1024
```

### 远程多卡训练

远程训练先按 `.agents/skills/remote-dev-se3/SKILL.md` 选择精确 machine profile。
profile 负责记录连接参数、GPU 数量、推荐环境数、计费策略和 CUDA 特殊配置。

MJLab 多卡训练使用 `--gpu-ids all`。多卡时 `--env.scene.num-envs` 是每张卡的环境数，
不是全局环境数：

```bash
# 多卡：每张卡的环境数由 machine profile 给出
uv run se3-train SE3-WheelLegged-Recovery-Discovery-GRU \
  --gpu-ids all \
  --env.scene.num-envs <envs-per-gpu>
```

Recovery-Discovery 的默认训练长度为 `5000` 轮，确保能够覆盖延伸到第 `4200` 轮的状态缓存、速度指令和 push disturbance（推搡扰动）课程。需要短训时显式传 `--agent.max-iterations`，不要把短训轮数写回正式默认配置。

若 profile 要求 CUDA Forward Compatibility，按该 profile 的路径和验证条件执行，
不要把某台机器的 driver、compat 目录或 `LD_LIBRARY_PATH` 写入本公用文档。

### CPU 训练

macOS 或无 GPU 时可跑，速度很慢，只用于调试。环境数设为 1，否则会更慢：

```bash
# 平地训练（CPU 模式）
uv run se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1 --gpu-ids None

# 崎岖地形训练（CPU 模式）
uv run se3-train SE3-WheelLegged-Rough --env.scene.num-envs 1 --gpu-ids None
```

## 训练参数

轮数和保存间隔在对应任务的 `src/se3_train/tasks/<task>/rl_cfg.py` 里配置：

```python
max_iterations=5000,  # 默认 5000 轮
save_interval=100,    # 每 100 轮保存一次 checkpoint
```

## 训练产物

checkpoint 保存在 `logs/rsl_rl/se3_wheel_leg/<timestamp>/`，每 100 轮一个 `.pt` 文件：

```
logs/rsl_rl/se3_wheel_leg/2026-05-05_23-13-57/
├── model_0.pt
├── model_100.pt
├── model_200.pt
├── ...
├── model_4900.pt
├── model_4999.pt     # 5000 轮训练的最终模型
├── onnx/
│   ├── model_0.onnx
│   ├── model_100.onnx
│   └── model_4999.onnx
└── params/
```

## Sim2sim 验证

从主仓库根目录启动常驻浏览器：

```bash
./scripts/run_sim2x.sh
```

在 `Models` 页签按 `Experiment → Run ID → ONNX` 选择 artifact。切换成功会同时重置仿真、
history/hidden state、previous action 和动作延迟状态。具体契约见
[ONNX metadata runtime](onnx_runtime.md)。

## 常见问题

### ImportError: cannot import name 'XmlMotorActuatorCfg'

mjlab 版本更新，API 已变化，改用 `XmlActuatorCfg`。

### ValueError: Error opening file '../meshes/...'

缺少机器人 mesh 文件，参考「机器人模型文件」章节。

### IndexError: list index out of range (GPU)

没有可用的 NVIDIA GPU，加 `--gpu-ids None` 切到 CPU 模式。
