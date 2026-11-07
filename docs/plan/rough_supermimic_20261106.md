# Rough-SuperMimic：世界模型 + BPTT 训练 Rough 速度跟踪（2026-11-06）

## 背景

SuperMimic（Liao et al., arXiv 2610.28480, CoRL 2026）的做法：用模拟器采数据、拟合一个确定性可微世界模型，
冻结模型后做 32 步想象 rollout，用一阶梯度 BPTT 更新确定性策略，critic 用 GAE(λ) 估计视野外回报。论文在 G1
动作跟踪上报告，同样墙钟时间下比 PPO 高一截。官方没有放代码。

用户 11-06 定：用开源复现 [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)
（MIT，mjlab + MuJoCo Warp，取 commit `ba61053`）接本仓库训练，目标任务是 **Rough 速度跟踪**，
奖励走**解析 + 学习混合头**。

复现仓库是 G1 专用（29 DoF、可微 FK、动作跟踪奖励），并要求 Python 3.12、锁了更新的 mjlab commit，不能当依赖装。
本仓库把与机器人无关的骨架移植进 `src/se3_train/supermimic/`，G1 专用部分按 SerialLeg 重写；许可全文见
`LICENSES/SuperMimic-Reproduced.txt`。

## 分阶段

- **第 1 阶段（本次）**：Rough 默认配置换纯平地地形，去掉 recover 组与推扰，把整条链路跑通；同 env 开 PPO 对照。
- **第 2 阶段**：全地形。启动时把地形采成高度图，世界模型输入加预测机身位姿处双线性插值得到的局部高度扫描
  （对 x、y、yaw 可微），ξ 加地形列 / 等级；台阶类奖励走学习头；多卡数据并行。

## 入口

| task id | runner | 用途 |
|---|---|---|
| `SE3-WheelLegged-Rough-SuperMimic` | `SuperMimicRunner` | SuperMimic |
| `SE3-WheelLegged-Rough-SuperMimic-PPO` | `Se3ProfiledOnPolicyRunner` | 同 env 的 Rough MLP PPO 对照 |

env（`tasks/rough_supermimic/env_cfg.py`）= `rough.env_cfg(terrain_generator=flat_only_terrains_cfg())` 再去掉：
- recover 组整套：倒地出生、零速指令、`recover_*` 奖励与终止、`amp` / `recover_mask` 观测组、critic 的 `recover_*` 项；
- 推扰：`push_robots` 事件、`push_disturbance` 课程（论文同样去掉：状态突变确定性模型拟合不了）。

其余（DR1、动作延迟 0–10 ms、轮力矩滞后 5–15 ms、30 维 actor 观测、ONNX 契约、actor 128/64/32）与 Rough 默认相同。
两个入口只差 runner；SuperMimic 入口多挂一个回放 recorder（不改 MDP）。

## 世界模型

**状态 s（41 维）** = qpos 17 + qvel 16 + 上一拍执行动作 6 + 轮执行器滞后状态 2。后两项是执行链的隐状态：
物理步 5 ms × decimation 4、动作延迟 0–2 个物理步，每个控制步前几个子步执行的仍是上一拍动作；
`LaggedDcMotorActuator` 的一阶滞后输出同理。不带它们 s 不是马尔可夫的。

**规范化特征**航向不变；8 个非轮铰链用 sin/cos（前杆可整周转）；轮角不进特征。

**动力学**沿用复现仓库的参数化：网络输出速度增量与位置修正，叠加在显式积分上，最后一层零初始化（匀速外推先验）。
MLP 2048×3，输入 = 特征 + 裁剪后的动作 + 归一化 ξ。

**ξ（41 维）** = `dr_model_params_obs` 28 + 气弹簧力 2 + 腿编码器偏置 4 + 腿默认站姿 4 + 延迟步数 1 + 轮滞后 τ 2。
归一化用首次采到的全体 env 统计量（DR 是 startup 事件，每 env 常量），随 checkpoint 保存。

**解析观测头**：30 维 actor 观测直接从状态算，复用 `policy_leg_phase_active_obs_torch` 等同一批函数；噪声读 env 配置。

**奖励头**（`heads.RewardHeads`，按 env 奖励项的函数身份分类）：
- 解析：tracking_lin_vel、tracking_ang_vel、tracking_orientation_l2、flat_base_height、ang_vel_xy、stand_still、
  action_rate、action_smoothness、dof_pos_limits、is_alive、tracking_lin_vel_narrow、wheel_fore_aft_offset（可微树 FK）、upward；
- 摔倒罚：`fall_penalty` = 权重 × 终止头概率 d̂_{t+1}；
- 学习：其余项（力矩 / 功率 / 关节加速度 / 角动量 / 接触类），一个小 MLP 回归逐项加权值，按逐项 RMS 归一化，
  在 k 步窗口里与状态损失一起训练（Eq. 1 的 r̂ 项）。纯平地上恒 0 的项（台阶类、跳跃 mimic）也落在学习头，学成 0。

**终止头**学习，输入特征 + ξ + 指令，BCE 拟合 `reset_terminated`；time_out 与出界截断不算终止。

## 训练循环

- 策略就是 rsl_rl PPO 构造出的 actor（确定性输出），不再用 PPO 更新；checkpoint / ONNX / bench 全部沿用。
  obs 归一化只在真实采集时更新。
- 采集：执行动作 = 策略动作 + N(0, 0.1²)；缓存同时存干净动作与执行动作，世界模型用执行动作。
- 一次 learning iteration = 24 个 env step（与 Rough PPO 每轮采样量相同），UTD 0.4，约 10 次（世界模型 + 策略）更新。
- 想象中的指令取缓存里起点之后同一 episode 的记录指令，出 episode 后保持最后一帧。
- 超参默认取论文 Table A.5：窗口 8、每次 10×2048 窗口、想象 4096 条 × H 32、γ 0.99、λ 0.95、Adam 2e-4、
  余弦衰减到 0.03×（按总 env 步数的 250/320）、梯度裁剪范数 1.0 + 逐元素 0.1、缓存每 env 1000 步。
- 第 1 阶段单卡；checkpoint 多一个 `supermimic_{it}.pt`（世界模型、critic、优化器）。

## 性能（2026-11-07，nulltask1 A800，1024 env）

首版 smoke 每轮 19 s。剖析：一次更新 2.2 s 里矩阵乘只占 117 ms，其余是每步几百个逐元素小算子的 kernel launch 与
Python 开销（四元数、FK、13 项解析奖励、积分与误差，世界模型 8 步 × 10 批、想象 32 步前向 + 反传各来一遍）。修法：

1. `WorldModel._loss_step` 与 `ImaginationStep.observe / transition` 用 `torch.compile` 编译成固定形状的融合图
   （复现仓库同样编译了这两步，移植时漏了）；
2. 编译后仍被切成几十张图：mjlab 的四元数函数带 `@torch.jit.script`，dynamo 追不进 → 换成 `supermimic/quat.py`
   纯 torch 版（双精度误差 ≤ 4e-16）；se3_shared 的前杆周期解析走 numpy → 构造时解析一次；
3. fused Adam；更新指标留在设备上，每轮只同步一次主机。

结果：世界模型更新 878 → 228 ms，actor–critic 更新 1393 → 193 ms，三张编译图均 0 断点、无重编译；编译前后窗口损失
相对差 3e-7。smoke 稳态 4.7–5.2 s/轮（首轮含编译 12 s），5000 轮单卡约 7 h。剩余开销主要是 2048×3 世界模型的
矩阵乘（实测约 40 TFLOPS），也就是论文超参（UTD 0.4、每次 10×2048 窗口 + 4096×32 想象）本身的计算量；
再快只能动超参或换 bf16。剖析脚本 `.scratch/supermimic/profile_update.py`。

## 相对复现仓库的修正

复现仓库在 episode 结束时把最后一个 transition 行也标成终止（`done = terminated`），该行状态被当作窗口中间目标时
终止头会拿到错标签；这里只给终止行打标签。

## 论文未指明事项的取舍

| 事项 | 取舍 |
|---|---|
| 状态内容（U1） | qpos / qvel + 执行链隐状态（见上） |
| 名义误差（U2） | 位置 5 cm、姿态 0.1 rad、线速度 0.5 m/s、角速度 1 rad/s、腿角 0.1 rad、腿速 2 rad/s、轮速 5 rad/s、轮滞后 0.5 N·m |
| 解析 / 学习头（U3） | 见奖励头 |
| 残差参数化（U4） | 沿用复现仓库：显式积分 + 零初始化增量 |
| 上一动作进观测（U9） | 想象中用干净动作；仿真里是加噪后的执行动作 |
| critic 输入（U10） | 特征、指令、ξ 全部 detach |
| UTD 与迭代（U12） | 按 env step 计，10 个批顺序做 10 次梯度步 |

## 验收

1. GPU 逐项核对（`.scratch/supermimic/check_heads.py`）：解析观测 vs env 观测（关噪声）、每个解析奖励项 vs
   `_step_reward`、缓存窗口完整性、执行链隐状态读取。
2. GPU smoke：`SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Rough-SuperMimic --env.scene.num-envs 1024` 与 PPO 对照入口各一次。
3. 正式对照（需用户明示开训）：两入口同卡数、同 env 数，比 W&B `Train/mean_reward` 与确定性回放的跟踪误差。
