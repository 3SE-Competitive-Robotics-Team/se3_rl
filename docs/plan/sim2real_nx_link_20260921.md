# 真机链路重启：NX adapter 与下位机契约对齐（2026-09-21）

2026-07-12 暂停的 sim2real 迁移本轮重启。下位机代码为
`https://gitee.com/SEU-3SE/serialleg2026.git` 的 `rl_deploy` 分支（本地
`D:\RoboMaster\serialleg2026`），MCU 是 STM32H723 + FreeRTOS，Keil MDK-ARM 工程。

## 审计结论：下位机侧基本做完，缺的是 NX 侧

`Custom/Tasks/Src/task_nx_comm.c` 与 `task_control_core.c` 已实现完整的 USB CDC 协议、
状态上报、目标接收、RC 使能门控、100 ms 超时清零与 DWT 延迟回报。`tools/nx_cdc_latency_test.py`
是上位机侧的帧层种子，但其 `STATE` 结构体与 `MAX_PAYLOAD=96` 都已落后于当前固件
（现 payload 160 B、上限 192 B），按原样跑会把每一帧都判成 `bad_state`。

语义上已经对齐的三件事：

- joint 顺序 LF / LB / RF / RB 与 policy order 一致；
- `command[5]` 正好是 `velocity_height[0:5]`，字段顺序逐项吻合；
- `wheel_pos` 发 0 —— 训练侧 `se3_shared.policy_io` 同样把这两个观测槽位硬置 0（轮子是
  连续关节，累计位置无界），固件这个「看起来没写完」的地方其实是对的。

## 数值失配（会直接改变机器人行为）

| # | 项 | 训练 / artifact 契约 | 固件现值 | 建议 |
|---|---|---|---|---|
| 1 | 轮速环增益 | `RobotConfig.wheel_kd = 0.2` | `pid_initialize(&WheelTorqueInt[*], 0.08f, …)` | 改 `0.2f`。固件停在 2026-09-02 改动前的旧值，同样轮速误差真机只出 40% 力矩 |
| 2 | 轮力矩限幅 | `M3508_C620_14.rated_torque` = 3.0×14/19 = **2.21 N·m** + T-N | PID limit `3.0f` | 改 `2.21f`；要复刻 T-N 需按轮速插值 |
| 3 | 腿力矩限幅 | `min(40·(1−ω/16.75), 20)`，零速 20 N·m，ω>8.4 rad/s 继续降 | `NX_POLICY_JOINT_TORQUE_LIMIT 24.0f` 平限 | 先降到 `20.0f` 拿到零速一致；要完全一致需实现 T-N 包络 |
| 4 | 高度指令包络 | rough `stair_height_range=(0.20, 0.38)`、high-stand `(0.36, 0.38)` | `0.195–0.390`，默认从 0.26 改到 0.36（未提交） | 固件侧无需改；已改 sim2x 兼容边界，见下 |
| 5 | 膝气弹簧 | `knee_gas_spring_force=300`、`compensation_enabled=False`（策略直接面对带弹簧的 plant） | 无前馈 | **已核对一致**（2026-09-21） |

第 5 项的核对依据：用户确认真机已装 300 N 气弹簧且挂点与 MJCF 一致；固件侧
`hipMotorTorque` 只在 `nx_policy_output_update` 由 PD 写入，`task_control_output.c:35`
直接 `DM_Mit_control` 下发（带符号翻转），全链路没有任何弹簧或重力前馈——与训练端
`compensation_enabled=False` 的语义吻合，弹簧那 10–18 N·m 抗重力力矩由硬件提供、策略
自己利用。这也意味着第 3 项的力矩包络更要紧：策略学到的动作是在「有弹簧 + 上限 20 N·m」
的 plant 上探索出来的。

腿 PD `kp=60 / kd=3` 与 `leg_kp / leg_kd` 已一致；轮速目标限幅 45 rad/s 与 rough 线的
`action_scale` 45 一致。

## 本轮改动

### se3-sim2x

新增 `se3_runtime_nx`，与 `se3_runtime_mujoco` 平级：

- `protocol.py`：帧编解码 + 增量解析器，CRC/结构体与固件逐字段对齐，纯函数可离线单测；
- `transport.py`：termios 打开 `/dev/ttyACM*`，不引入 pyserial；`NxTransport` 是 Protocol；
- `adapter.py`：`SerialLegNxAdapter` —— STATE → `RobotState`、command 夹紧、按 decimation
  推进动作 FIFO、下发 TARGET、使能/超时/异常降级、链路遥测；
- `cli.py`：`se3-sim2x-nx --onnx … --dev /dev/ttyACM0`。

核心侧只加了一处：`PolicyActionPipeline` / `PolicyRuntime` 支持 `delay_steps_override`。
真机默认置 0 —— metadata 里的 4–6 ms 在仿真里模拟的就是「推理 → 通讯 → 执行器」这段真实
耗时，真机上它物理存在，再叠一次 FIFO 是双份延迟。

`policy_descriptor.py` 的兼容 command 边界从 `lin_vel_x (-1.5, 1.5)` / `height (0.20, 0.32)`
放宽到 `(-2.4, 2.4)` / `(0.195, 0.39)`。旧值是 runtime 侧单方收窄的结果，已窄于 rough 线的
实际训练包络，真机 adapter 收到第一帧 `height=0.36` 会直接判错。这就是 handoff 里挂着的
「上实机前需拍板契约归属」，证据在训练侧：rough 从未声明 `deployment_ranges`，只有 flat 与
recovery_discovery 声明了，所以 rough artifact 一直走这张兜底表。

### 未做（下一轮）

1. 固件四个常量对齐（已获授权直接改本地 `rl_deploy`，不 push，由用户 review 后烧录）；
2. 给 rough / stair / jump 各线补 `deployment_ranges`，让 artifact 自带部署包络，不再依赖
   sim2x 的兜底表；
3. `tools/nx_cdc_latency_test.py` 的 STATE 结构体与 payload 上限同步到当前固件；
4. 离线对拍：录一段 STATE 序列同时喂 sim2x 与固件，逐帧比 `target_joint_pos`，验证符号、
   零点与坐标系（固件 `base_ang_vel = (−roll_w, −pitch_w, yaw_w)`、
   `projected_gravity = (g_y, −g_x, g_z)` 这两处重映射尚未独立验证过）。

## 上机顺序

1. NX 上装 submodule，`se3-sim2x-nx --ignore-output-enabled` 空跑，只看遥测：STATE 周期应
   ≈10 ms，固件 rx→输出 p95 与端到端 RTT 落在什么量级；
2. 机器人吊起、轮子离地，RC 使能，观察四个关节是否朝默认站姿收拢；
3. 对齐固件四个常量后重复 2；
4. 落地静站（height 默认），再给小 vx；
5. 台阶前先确认 20 s episode 之外的长历史问题：GRU 线上机必须先 Reset 再给指令。

## 风险

- 训练 checkpoint 尚未定版（rough M23 `7om3sfb1` 在训）。链路调通不等于有可上机策略。
- 原生 MuJoCo 与 MJLab 的轮地牵引不一致这条老问题仍未解（`work-log.md:801`），真机数据
  回来之前不要用任一后端的数字去定实机参数。
