# Jetson Orin NX 真机部署笔记

本文记录当前已经验证过的 NX 连接方式，以及后续把训练好的 policy 部署到真机时必须遵守的边界。NX 是真机实时推理目标，不作为 MJLab 训练机使用；训练仍在 Linux + NVIDIA GPU 训练机完成，先通过 sim2sim 验证，再进入真机联调。

## 当前目标

- 阶段一只在 NX 上部署 `SE3-WheelLegged-Recovery-GRU` 倒地自启 policy。
- NX 只负责 policy 推理和上层状态机，通过 USB CDC 以 50 Hz 给 STM32 发送 6 维 raw policy action。
- STM32 继续负责 CAN、电机闭环、急停、限幅、限速、通信超时和底层安全。
- 阶段一不做 policy 切换；runtime 只运行 recovery 网络。
- 训练端、sim2sim 端和真机端必须沿用同一份 policy contract：动作顺序、观测布局、默认姿态、动作缩放和动作延迟不能漂移。

## 阶段一 Recovery-only Checkpoint

当前阶段一部署目标是纯自起网络，不包含行走、跳跃或双策略切换。只要进入 policy 运行态，NX 就持续运行这一份 recovery 网络；`recovery_success` 只作为日志和人工验收信号，不触发切换到其他 policy。

| 项目 | 值 |
|---|---|
| 任务 | `SE3-WheelLegged-Recovery-GRU` |
| 本地权重文件 | `logs/rsl_rl/se3_wheel_leg/<run>/model_<step>.pt` |
| checkpoint iter | `<step>` |
| checkpoint 内容 | `actor_state_dict`、`critic_state_dict`、`optimizer_state_dict`、`infos` |

注意：`replays/.../model_<step>` 是 sim2sim 验证结果目录，只包含 `json` 和 Rerun 回放，不是部署时加载的 policy checkpoint。部署到 NX 时应传 `logs/.../model_<step>.pt`。

Recovery runtime 的 command 固定为：

```text
vx = 0.0
yaw_rate = 0.0
pitch = 0.0
roll = 0.0
height = default_base_height
jump_flag = 0.0
jump_target_height = 0.0
jump_phase = 0.0
```

NX 每帧输出网络的 6 维归一化 raw action，动作顺序固定为：

```text
[LF, LB, RF, RB, l_wheel, r_wheel]
```

STM32 按共享常量把 raw action 转换为物理目标并限幅：腿部按 `RobotConfig.action_scale[:4]` 和当前高度默认腿型解码，轮子为 `action[4:6] * 45.0 rad/s`。

## NX Recovery Runtime

仓库内新增阶段一 runtime 入口：

```bash
uv run se3-export-nx-policy \
  --checkpoint logs/rsl_rl/se3_wheel_leg/<run>/model_<step>.pt \
  --output logs/deploy/model_recovery_gru.npz

uv run se3-nx-recovery \
  --checkpoint logs/deploy/model_recovery_gru.npz \
  --port /dev/ttyUSB0 \
  --rate-hz 50
```

不接 STM32 时先跑 dry-run，确认 checkpoint、GRU hidden 和 32 维观测拼装能正常推理：

```bash
just nx-recovery-dry-run
```

对应源码：

- `src/se3_deploy/protocol.py`：USB CDC 帧格式、CRC32、流解析。
- `src/se3_deploy/cdc.py`：Linux `/dev/ttyUSB*` / `/dev/ttyACM*` 非阻塞读写，不依赖 pyserial。
- `src/se3_deploy/observation.py`：真机 recovery-only 32 维 actor 观测拼装。
- `src/se3_deploy/export_npz.py`：把 PyTorch checkpoint 导出为 NX 轻量 NumPy 权重。
- `src/se3_deploy/numpy_policy.py`：不依赖 torch 的 GRU actor NumPy 推理后端。
- `src/se3_deploy/recovery_runtime.py`：50 Hz recovery policy 主循环。

当前 NX 环境检查结果：JetPack R36.4.3，系统 Python 3.10.12，已在用户目录安装 `uv` 和 Python 3.11.15。由于 PyTorch cu128 wheel 对 NX 阶段一过重，当前实测路线为本地导出 `logs/deploy/model_recovery_gru.npz`，NX 上只安装 `numpy` / `pydantic` 并用 NumPy 后端推理。NX 测试目录：

```bash
~/project/se3_wheel_leg_nx_runtime
```

NX 上已生成便捷启动脚本：

```bash
cd ~/project/se3_wheel_leg_nx_runtime
./run_recovery.sh --dry-run --max-steps 5 --print-every 1
./run_recovery.sh --max-steps 5 --print-every 1
```

第二条会打开 `/dev/ttyUSB0`。没有 STM32 协议帧时，预期只打印 `states=0 actions=0`，用于确认 USB CDC 端口可打开；接入 STM32 state 帧后才会发送 action。

### USB CDC 帧格式

协议使用小端二进制帧：

```text
magic[2] = "S3"
version  = 1
msg_type = 1(state) / 2(action)
payload_len uint16
payload
crc32(header + payload) uint32
```

STM32 → NX state payload：

```text
seq uint32
timestamp_us uint32
status_bits uint32
base_ang_vel_body[3] float32
projected_gravity[3] float32
dof_pos[6] float32       policy 顺序 [LF, LB, RF, RB, l_wheel, r_wheel]
dof_vel[6] float32       policy 顺序 [LF, LB, RF, RB, l_wheel, r_wheel]
motor_status[6] uint16
```

NX → STM32 action payload：

```text
seq uint32
source_state_seq uint32
timestamp_us uint32
mode uint16              1 = recovery-only
flags uint16             bit0=state timeout, bit1=non-finite sanitized, bit2=dry-run
action[6] float32        raw policy action，policy 顺序 [LF, LB, RF, RB, l_wheel, r_wheel]
```

STM32 必须检查 magic、version、长度、CRC、`source_state_seq` 新鲜度和 `flags`；任何错误、超时或急停都应退出 policy action 接收路径，回到底层安全逻辑。

## 已验证连接信息

| 项目 | 值 |
|---|---|
| 设备 | Jetson Orin NX |
| 主机名 | `tegra-ubuntu` |
| 系统 | Linux `5.15.148-tegra` |
| 架构 | `aarch64` |
| 直连 IP | `192.168.137.100` |
| 用户名 | `amov` |
| 本机网卡 | Windows `以太网` |
| 本机直连地址 | `192.168.137.1/24` |
| 认证方式 | 本机 `~/.ssh/id_ed25519.pub` 已写入 NX `~/.ssh/authorized_keys` |
| 验证日期 | 2026-06-01 |

首次连接时如果本机还没有 `192.168.137.0/24` 地址，需要在管理员 PowerShell 中给有线网卡临时追加地址：

```powershell
New-NetIPAddress -InterfaceAlias '以太网' -IPAddress 192.168.137.1 -PrefixLength 24 -PolicyStore ActiveStore
```

`ActiveStore` 只对当前系统会话生效，重启后会消失；这适合现场调试，不会永久改写网卡配置。

验证 SSH：

```powershell
ssh -i "$env:USERPROFILE\.ssh\id_ed25519" -o IdentitiesOnly=yes amov@192.168.137.100 "hostname && uname -m && whoami && hostname -I"
```

期望至少包含：

```text
tegra-ubuntu
aarch64
amov
192.168.137.100
```

建议在本机 `~/.ssh/config` 增加别名：

```sshconfig
Host serialleg-nx
    HostName 192.168.137.100
    User amov
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
```

之后可直接：

```bash
ssh serialleg-nx "hostname && uname -a"
```

## 部署前置门槛

真机部署前必须先满足这些条件：

1. checkpoint 在训练端保存完整，不能依赖 W&B 在线 writer 才保存。
2. 对应 checkpoint 已跑过 sim2sim headless 验证。
3. policy 动作顺序固定为 `[LF, LB, RF, RB, l_wheel, r_wheel]`。
4. actor 观测布局与 `se3_shared` / `se3_sim2sim` runtime contract 一致。
5. 真机端只使用 actor 可用观测，不能读取训练专用 privileged observation。
6. 真机启动默认进入安全禁能状态，必须有急停、限幅、限速和超时保护。

推荐验证链路：

```bash
uv run se3-sim2sim --checkpoint <ckpt> --viewer none --max-steps 1000 --print-every 100
uv run se3-sim2sim --checkpoint <ckpt> --max-steps 3000
```

## 源码和产物同步

源码变更走 git，不用 `scp` 手工覆盖源码文件。NX 上建议放在：

```bash
~/project/se3_wheel_leg
```

首次准备：

```bash
ssh serialleg-nx "mkdir -p ~/project"
ssh serialleg-nx "cd ~/project && git clone https://github.com/3SE-Competitive-Robotics-Team/se3_wheel_leg.git"
```

checkpoint、导出的 policy、Rerun 回放等产物可以单独传输。传 checkpoint 时仍要保留 run timestamp 和 model step，避免后续无法追溯：

```bash
scp logs/rsl_rl/se3_wheel_leg/<timestamp>/model_<step>.pt \
  serialleg-nx:~/project/se3_wheel_leg/logs/rsl_rl/se3_wheel_leg/<timestamp>/
```

## 真机 runtime 后续补齐

当前仓库已经具备阶段一 recovery-only NX runtime、policy 导出入口和 USB CDC 协议框架。下一步部署开发应继续补齐：

- 真机观测适配层实测校准：IMU、关节编码器、轮速、上一帧动作和 command 输入。
- USB CDC 联调：STM32 上行 policy 顺序物理状态，NX 下行 6 维 raw policy action。
- NX 上层状态机：禁能、recovery-only policy 运行、GRU hidden reset、故障/人工停止回退；阶段一不实现多 policy 切换。
- STM32 底层安全状态机：CAN、电机闭环、急停、限幅、限速、通信超时和安全回退。
- 频率与延迟对齐：policy/action 默认 50 Hz（`control_dt=0.02 s`，`control_decimation=4`），动作延迟和 actuator 限幅必须显式记录。
- 日志与回放：动态日志继续优先使用 Rerun，便于和 sim2sim 对齐排查。

不要在 NX runtime 或 STM32 firmware 中各自发明一套关节顺序、动作缩放常量或控制频率；应以 `se3_shared` 作为部署 contract 的单一来源，避免部署端和训练端漂移。当前频率基准见 `docs/control_frequency.md`。

## 风险记录

- NX 直连依赖本机有线网卡处于 `192.168.137.0/24`；如果 SSH 超时，先查本机网卡 IP，不要先怀疑密码。
- `192.168.137.100:22` 可达才进入认证排查；端口不通时优先检查网段、网线、NX SSH 服务。
- 明文密码不写入仓库；首次引导后使用 SSH key。
- 真机动作输出必须先限幅再下发，任何未经限幅的 raw policy action 都不能直接进电机控制链路。
