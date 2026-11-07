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

### 第二轮（2026-11-07，4096 env）

开训代码（7f02927）在 4096 env 下真实训练 5.65 s/轮（采集 1.24 + 学习 4.4，每轮 9.7 次更新）。剖析后更新阶段已是
GPU 满载，剩下三处：

1. 回放采样每批对 4096×1000 整张掩码做 `multinomial` 并重算 `window_mask`，占世界模型更新 GPU 时间约 22% →
   每次更新只算一次有效行、各批 `randint`（与权重全 1 的有放回 multinomial 同分布）；
2. 世界模型每批调 8 次前向 + 8 次反传的编译函数、想象每步两张图，每次调用 CPU 侧 1.3–2.4 ms →
   k 步窗口整体编译（`WorldModel._window`），想象单步把观测、策略、世界模型、奖励合成一张图
   （策略直接调 actor 的 obs 归一化与 MLP，learn 开始时与 actor 前向核对）；
3. 各处零碎主机同步。

结果（983976c）：世界模型更新 265 → 180 ms、actor–critic 207 → 187 ms，两张图均 0 断点；smoke（4096 env）
4.5–4.8 s/轮，指标与改动前吻合。此时每次更新的 GPU kernel 约 310 ms，其中 TF32 矩阵乘约 200 ms，
即论文超参下的下限约 3.7 s/轮（9.7 × 0.31 + 采集）；再快要动世界模型宽度 / UTD / 想象批量，或换 bf16。

### 第三轮：CUDA Graphs（2026-11-07）

983976c 之后每次更新仍有约 2.7 万次 kernel launch。世界模型缩到 1024 宽只快约 10%（GPU 先算完，又回到等 CPU），
说明瓶颈是 launch 而不是算力。`compile_mode="reduce-overhead"`（CUDA Graphs）后每张图回放只算一次 launch：

| 4096 env | 学习段 / 轮 | 世界模型 / 次 | actor–critic / 次 | launch / 次更新 |
|---|---|---|---|---|
| default，2048 | 3.73 s | 180 ms | 206 ms | 约 2.7 万 |
| reduce-overhead，2048 | 3.11 s | 161 ms | 160 ms | 约 3600 |
| reduce-overhead，1024 | 2.34 s | 93 ms | 149 ms | 约 3600 |

（剖析脚本数字；runner smoke 2048：每轮 3.65–3.94 s，采集约 1.3 s。）窗口损失与 default 相对差 7e-8，显存不变。

踩坑：rsl_rl 的 EmpiricalNormalization.update 每步重新赋值 `_mean/_std`（新地址），CUDA Graphs 把模块缓冲区当作
地址固定的静态输入，想象图每次更新整张重录（每轮 360 s）。runner 改为让想象中的策略读一份地址固定的副本，
每次 actor–critic 更新前原地同步。凡是进编译图的模块张量都必须原地更新。

### 第四轮：bf16 + 世界模型 1024 宽（2026-11-07 用户定，010acde 起可选，之后设为默认）

bf16 只作用在三个 MLP 内部（autocast，输出转回 fp32），状态积分、误差、损失、Adam 主权重仍是 fp32。
与 tf32 相比：窗口损失相对差 1.5e-6、1 步预测差 7e-7（模型自身误差 0.32）、想象 32 步奖励均差 2.8e-3（量级 3）。

| 4096 env、CUDA Graphs | 学习段 / 轮（剖析） | 显存峰值 |
|---|---|---|
| 2048 + tf32 | 3.11 s | 6.6 GiB |
| 2048 + bf16 | 2.65 s | 5.2 GiB |
| 1024 + tf32 | 1.97 s | 5.0 GiB |
| 1024 + bf16 | 1.91 s | 4.0 GiB |

runner smoke（1024 + bf16，8 轮）：稳态每轮 2.52–2.75 s（采集约 1.27 s、学习约 1.3–1.45 s），第 7 轮 8 步误差 0.126
（2048 + tf32 同期 0.092）。1024 宽时 actor–critic 又回到 CPU 侧受限，bf16 对它无效；采集比裸 env.step（0.7 s）
多出约 0.55 s 的 runner 开销，是下一个可压的点。注意：11-07 12:29 开训的对照 p7dv60ea 用的是 2048 + tf32。

### 第五轮：采集开销核查（2026-11-07）

日志里采集约 1.1–1.27 s，而剖析脚本单测 env.step 只要 0.72 s。逐项排查（`.scratch/supermimic/profile_collect*.py`、py-spy）：

- runner 自身（obs 归一化、actor、加噪、logger）合计 0.6 ms/步，recorder 0.5 ms/步，不是开销来源；
- 早期日志的 1.27 s 有一部分是计时归属：更新异步排队，被下一个 env.step 的首次同步吃进去。现在每批更新后同步一次再计时（CUDA event 不可靠：cudagraph 回放不一定在当前 stream 上排队）；
- 剩下的约 1.1 s 是 Rough env 的真实开销：训练用 `init_at_random_ep_len`，每步约 2–3% 的 env 在 reset，`_reset_idx`
  占 env.step 约 25%（重采关节、抬机身两次全量 forward、重采指令）。单测脚本所有 env 同时从头开始、几乎不 reset，所以只有 0.72 s。
  PPO 同样付这部分（每轮约 1.07 s 里几乎全是它），不是 SuperMimic 特有的。

顺带修掉两处更新侧的损耗：回放采样里 `nonzero` 的主机同步（py-spy 占全进程 15.7%，改为无同步的拒绝采样，
GPU 核对：抽到无效行 0、按行分布与纯随机一致）；编译与 CUDA Graphs 录制后 `gc.freeze()`（GC 开 / 关每次更新 172 / 161 ms）。
smoke（1024 + bf16，12 轮）稳态每轮 2.50–2.72 s（采集约 1.1 s、学习 1.42 / 1.58 s，交替是 9 / 10 次更新）。

### 第六轮：Rough env reset 抬机身（2026-11-07 用户定接受，9b79b35，所有 Rough / Flat 任务共用）

`_lift_root_to_wheel_clearance` 每次 reset 对全部 world 做两次全量 forward。原点模式只需要轮 body 的 xpos，改为
`mujoco_warp.kinematics`（forward 第一阶段同一 kernel）；写回根位姿后不再 forward（env.step 在 reset 后统一 forward，
后续 reset 事件与指令重采样都不读 reset env 的派生量）。

- 速度：4096 env、随机起始 episode 长度，env.step 43.1 / 43.3 → 39.2 / 37.9 ms；smoke SuperMimic 2.37–2.53 s/轮、
  SuperMimic-PPO 0.97–1.00 s/轮（此前 1.07–1.09）；默认 Rough、Flat-MLP smoke 同样通过。
- CPU 逐位核对（`.scratch/supermimic/check_reset_equiv.py`，Rough-SuperMimic-PPO 与完整 Rough，各 200 步）：qpos、qvel、
  actor 观测、奖励、done 逐位相同。critic 的轮接触力、关节加速度、机身碰撞力三项不同，且差异步与「这一步有 env reset」
  的步完全重合（62/62、59/59）：这三项是约束求解器输出，旧版只要有 env reset 就对全部 env 多跑两次 forward，求解初值不同；
  新版每步都只在物理步后 forward 一次，口径一致。只影响 PPO critic 的特权输入，此后 run 与之前的在这三项上不逐位可比。

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
