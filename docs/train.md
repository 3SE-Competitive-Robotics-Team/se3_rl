# 训练指南

## 环境准备

### 机器人模型文件

训练和仿真用的 MJCF 与 mesh 文件已放在 `assets/robots/serialleg/`。重新导出模型时，保持 MJCF 中的关节名和 mesh 相对路径不变。

当前默认模型是解析四连杆等效开树版本，用来降低闭链求解成本，同时保持 policy 的 `[LF, LB, RF, RB, l_wheel, r_wheel]` 主动杆语义。它移除 `drive_bar/coupler/equality`，但通过解析 FK/IK 在训练动作、观测和力矩映射中复现四连杆关系；collision 沿用闭链 OBB 裁剪方案，base 为 3 个保守 box，轮子为窄接地 cylinder，四个主腿 link 各使用一个沿视觉 mesh 长轴拟合并裁掉关节重叠的有向 box。

```text
assets/robots/serialleg/mjcf/serialleg_fourbar_surrogate_train.xml
```

MJCF 目录保留解析四连杆等效开树模型、OBB 裁剪闭链模型和旧开链模型。训练端默认 `SE3_ROBOT_MJCF_VARIANT=fourbar-surrogate`；需要显式验证真实闭链求解时用 `SE3_ROBOT_MJCF_VARIANT=closedchain`；需要回退定位时用 `SE3_ROBOT_MJCF_VARIANT=openchain`。临时测试其它导出文件时，用 `SE3_ROBOT_MJCF` 显式指定路径。

policy 动作顺序固定为 `[LF, LB, RF, RB, l_wheel, r_wheel]`，其中 `LB/RB` 对应 `l_drive_bar_Joint/r_drive_bar_Joint`。闭链限位语义是同侧两根主动杆夹角；当前装配分支下左腿为 `LF-LB`，右腿为 `RB-RF`，允许范围为 `0.0~1.46945 rad`，对应腿长下限约 `0.14 m`；当前默认夹角为 `1.31668 rad`，不是后主动杆的绝对角。

当前无气弹簧默认站姿按“腿长 0.16 m、base_link 距地约 0.22 m、轮心落在整机质心投影下、base/腿部几何离地”的几何平衡点重标定：

```text
default_dof_pos = [-0.275422946189, -1.592100148957, 0.275422946189, 1.592100148957, 0.0, 0.0]
default_output_knee_pos = [-1.242259649307, 1.242259649307]
default_coupler_pos = [1.401266340000, -1.401269410000]
default_base_height = 0.22 m
```

这只是两轮倒立系统的 reset 几何基点；零轮速开环 PD 仍不能替代策略的轮子平衡反馈。旧开链模型仍保留为显式回退 variant：

```bash
SE3_ROBOT_MJCF_VARIANT=closedchain uv run se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1 --gpu-ids None
SE3_ROBOT_MJCF_VARIANT=openchain uv run se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1 --gpu-ids None
uv run se3-sim2sim --model assets/robots/serialleg/mjcf/serialleg_fidelity_cylinder_wheels.xml --viewer none --max-steps 200
```

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

修改训练代码后先跑一次 smoke，5 轮训练，不上传 wandb，确认环境不崩：

```bash
# CPU smoke（任何机器都能跑）
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1 --gpu-ids None

# 开链回退 smoke（仅用于 A/B 定位）
SE3_SMOKE=1 SE3_ROBOT_MJCF_VARIANT=openchain uv run se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1 --gpu-ids None

# 真实闭链 OBB smoke（用于 A/B 定位）
SE3_SMOKE=1 SE3_ROBOT_MJCF_VARIANT=closedchain uv run se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1 --gpu-ids None

# GPU smoke
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1024
```

### GPU 训练

需要 NVIDIA GPU + CUDA 12.4+，环境数推荐 1024，RTX 3090/4090 约 2-3 小时跑完。

```bash
# 平地训练
uv run --env-file .env se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1024

# 崎岖地形训练
uv run --env-file .env se3-train SE3-WheelLegged-Rough --env.scene.num-envs 1024
```

### 远程多卡训练

gpufree 等按量计费机器遵循“本地改代码、无卡模式准备、GPU 模式短验证、GPU 模式长训、产物同步后立即关机”的流程，具体见 `.agents/skills/remote-dev-se3/machines/gpufree.md`。

MJLab 多卡训练使用 `--gpu-ids all`。多卡时 `--env.scene.num-envs` 是每张卡的环境数，不是全局环境数。例如 5 张 RTX 4090 上设置 `1024`，全局约为 `5 * 1024 = 5120` 个环境。

```bash
uv run --env-file .env se3-train SE3-WheelLegged-Recovery-GRU --gpu-ids all --env.scene.num-envs 1024
```

如果把单卡 `4096` 直接搬到 5 卡，会变成全局约 `20480` 个环境。除非已经重新缩放课程阶段、总迭代数、保存间隔和评估频率，否则优先用每卡 `1024` 做 20-50 iter benchmark。

A800 四卡倒地自起训练当前推荐使用每卡 `2048` 个环境。2026-06-01 benchmark 显示 `2048 env/rank` 约 `155.7k steps/s`，比 `1024 env/rank` 的约 `91.8k steps/s` 明显更快，同时等样本量仍保留约 `1500` 次 PPO update。`3072` 和 `4096` 吞吐更高，但等样本量时 update 次数更少，先作为吞吐实验或夜间试训，不直接替代主线长训。详细数据见 `docs/perf.md`。

```bash
# A800 四卡倒地自起推荐长训档位
SE3_LOGGER=tensorboard ./.venv/bin/se3-train \
  SE3-WheelLegged-Recovery-Stand-GRU \
  --gpu-ids all \
  --env.scene.num-envs 2048 \
  --agent.max-iterations 1500
```

如果训练容器使用 CUDA Forward Compatibility，启动训练前确认 Warp 报 `Driver 12.6`，且没有 `CUDA Graphs disabled`。容器默认 `LD_LIBRARY_PATH=/usr/local/nvidia/lib64` 可能覆盖 ldconfig 中的 compat `libcuda`，必要时在训练 shell 中 `unset LD_LIBRARY_PATH`。

### CPU 训练

macOS 或无 GPU 时可跑，速度很慢，只用于调试。环境数设为 1，否则会更慢：

```bash
# 平地训练（CPU 模式）
uv run --env-file .env se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1 --gpu-ids None

# 崎岖地形训练（CPU 模式）
uv run --env-file .env se3-train SE3-WheelLegged-Rough --env.scene.num-envs 1 --gpu-ids None
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
└── params/
```

## 评估/回放

项目不用 `se3-play`，回放和验证走 `se3-sim2sim`，Rerun 负责可视化：

```bash
uv run se3-sim2sim --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_4999.pt --max-steps 3000
```

不传 `--checkpoint` 时，程序自动选 `logs/rsl_rl/se3_wheel_leg/` 下编号最高的 `model_*.pt`，编号相同时选较新的 run。

需要保存 Rerun 回放文件：

```bash
uv run se3-sim2sim --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_4999.pt --max-steps 3000 --rerun-record replays/local_manual/model_4999__walk-sweep.rrd
```

Rerun 录制后拉回本地必须按固定命名存放，避免不同机器、不同 run、不同 course 的 `.rrd` 互相覆盖：

```text
remote_artifacts/<experiment_id>/rrd/<run_label>/<checkpoint>__<course>[__<case>]__rec-<YYYYMMDD-HHMMSS>[__rN].rrd
remote_artifacts/<experiment_id>/summaries/<run_label>/<checkpoint>__<course>[__<case>]__rec-<YYYYMMDD-HHMMSS>[__rN].json
remote_artifacts/<experiment_id>/logs/<run_label>/<checkpoint>__<course>[__<case>]__rec-<YYYYMMDD-HHMMSS>[__rN].log
```

字段约定：
- `experiment_id`：本次实验的稳定名字，格式为 `<topic>_<YYYYMMDD>`；如果实验本身按小时批次区分，可用 `<topic>_<YYYYMMDD_HHMM>`。多机器或 A/B 对比时把机器/分支写进 topic，例如 `l40s_unilab_vs_mjlab_20260603`。
- `run_label`：同一实验内的短标签，只用小写字母、数字、下划线或连字符，例如 `mjlab`、`unilab`、`jump_pretrain`、`recovery_gru`。
- `checkpoint`：直接使用 checkpoint 文件名去掉 `.pt`，例如 `model_4999`，不要写成 `latest`、`final` 或 `best`。
- `course`：使用 sim2sim 的 course 名，例如 `walk-sweep`、`jump-sweep`；手工单场景也必须写一个稳定名字，例如 `manual-stand`。
- `case`：同一 checkpoint + course 下的初始姿态、目标高度或特殊扰动，例如 `roll90`、`h0p4`、`yaw-pid-off`；没有区分项时省略。
- `rec-<YYYYMMDD-HHMMSS>`：录制开始时间，使用远程训练机本地时间；跨机器 A/B 时统一使用北京时间。
- `rN`：只有在同一 checkpoint、course、case 重新录制且需要保留旧文件时使用，从 `r2` 开始。

示例：

```text
remote_artifacts/l40s_unilab_vs_mjlab_20260603/rrd/unilab/model_1800__walk-sweep__rec-20260603-141927.rrd
remote_artifacts/l40s_unilab_vs_mjlab_20260603/summaries/unilab/model_1800__walk-sweep__rec-20260603-141927.json
remote_artifacts/recovery_roll_sweep_20260603/rrd/recovery_gru/model_1200__recovery-sweep__roll90__rec-20260603-133809.rrd
remote_artifacts/jump_height_ab_20260603/rrd/pretrain/model_900__jump-sweep__h0p4__rec-20260603-104522.rrd
```

远程录制时可以先写 `.rrd.tmp`，确认文件非空后再原子重命名为最终 `.rrd`；拉回本地时只拉最终 `.rrd`、同名 `.json` summary 和同名 `.log`。禁止把远程绝对路径或机器随机目录名直接塞进 `.rrd` 文件名；除 `rec-<YYYYMMDD-HHMMSS>` 外的来源信息写到 `state/recorded.tsv` 或 summary JSON 里。

`replays/`、`remote_artifacts/` 和 `.rrd` 文件是本地验证产物，不提交。

## Sim2Sim 验证

纯 MuJoCo CPU，macOS 可跑：

```bash
uv run se3-sim2sim --checkpoint logs/rsl_rl/se3_wheel_leg/<timestamp>/model_4999.pt --viewer none --max-steps 200 --print-every 20
```

训练只上传 wandb 指标曲线，不录视频。策略回放、轨迹检查、控制量曲线和 `.rrd` 文件都由 `se3-sim2sim` 的 Rerun 产出。

## 常见问题

### ImportError: cannot import name 'XmlMotorActuatorCfg'

mjlab 版本更新，API 已变化，改用 `XmlActuatorCfg`。

### ValueError: Error opening file '../meshes/...'

缺少机器人 mesh 文件，参考「机器人模型文件」章节。

### IndexError: list index out of range (GPU)

没有可用的 NVIDIA GPU，加 `--gpu-ids None` 切到 CPU 模式。
