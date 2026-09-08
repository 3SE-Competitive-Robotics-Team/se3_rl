# 训练任务架构

`src/se3_train/tasks/` 是训练任务的唯一入口。每个子目录表示一个完整 task，目录内收拢该任务相关的环境配置、RL 配置、观测、奖励、指令、课程、事件和终止条件。

旧的 `src/se3_train/env_cfg.py` 和 `src/se3_train/rl_cfg.py` 汇总入口已经删除，不再恢复。新增实验必须进入 `tasks/<task>/`。

## 当前任务

| 目录 | task id | 用途 |
| --- | --- | --- |
| `rough/` | `SE3-WheelLegged-Rough` / `SE3-WheelLegged-Rough-StairEval` / `SE3-WheelLegged-Rough-AMP` | 崎岖地形行走任务。2026-09-06 按 scutrobotlab/wheeled-legged_RL 的 V14 rough 线重写：环境继承冻结的 Flat 基线（轮 scale 15 + 弹簧时代 action_smoothness），只换四样东西——带课程的地形集（平地/上下台阶/上下斜坡/随机起伏，台阶 0.02–0.20 m、踏面 1.5 m，10 级难度，全员从第 0 级起步）、`terrain_levels` 地形难度课程、台阶前的机身抬升、能耗三项（`leg_torques`/`wheel_torques`/`leg_power`）÷10。PPO 与 Flat 基线逐项相同，只把轮数改为 5000。`-StairEval` 只留平地与上下台阶做定向评测。2026-09-07（R3 诊断后）：非平地列只发前向直行指令（vx 0.4–2.4、yaw ±0.2、无静站）；地形课程只升不降，升级判据改为切比雪夫距离越过最外一级台阶（4.0 m），走到块边 4.25 m 即 `terrain_cleared` 截断；stairs_up/slope_up 改用反金字塔（出生在低处向外爬）。同日移植 stair 线的 CTBC（`rough/ctbc.py`，状态机在 `mdp/ctbc_state.py`）：轮子顶住台阶立面（`wheel_riser_sensor` 法向 |n_z|≤0.5）持续 3 步超 10 N 就触发单侧轮端后缩抬升前馈，只在 stairs_up 列触发，500 轮前满幅、500→1500 线性退火、之后关闭；actor 的 jump_commands 3 维扩展槽实现换成 CTBC 左右相位 + 触发位（项名不变，部署契约 se3-sim2x 只认 jump_commands；退火完恒 0，部署端填 0）。回放验证：model_500 在 4 cm 坑底无 CTBC 停在平台边缘，加 CTBC 后 4 个 env 中 3 个翻过两级出块。R4 前另两处修正：平地速度课程只按平地列的跟踪分推进（startup 事件挂 `_se3_curriculum_env_mask`，奖励侧多记 `Locomotion/tracking_lin_vel_reward_curriculum`，课程读该键；R3 里全体均值被地形列拖在 0.3，平地列整场 vx=0）；地形列 vx 上限跟随平地课程当前上限（起点即 0.4 定速，随课程爬到 2.4；R3 从第 0 轮给满 2.4，500 轮策略对 vx≥1 原地不动）。critic 特权观测加 77 点地形高度扫描（`rough/observations.height_scan_obs`，照 yly-true/fudan_rl_wheel_leg 的 legged_gym 配方：机身系 yaw 对齐网格 x ±0.5 m、y ±0.3 m、间距 0.1 m；值为各点地面相对脚下地面的抬升，clip ±1 m；打空与自击记 0），只进 critic，actor 契约不变，旋钮 `critic_height_scan`。R7：前 500 轮全部 env 在平地列（`curriculums.flat_warmup`，之后各 env 在下一次 reset 时换回原列第 0 行，并刷新地形列指令覆盖与平地课程信号掩码；热身期 terrain_levels 不升级），Flat 速度课程推进阈值改用严格档 0.75；CTBC 默认关闭（R4/R5 第 0 轮注入前馈把策略教成回避接触、平地跟踪一起退化）。AMP：`SE3-WheelLegged-Rough-AMP` 入口 = rough 环境 + `amp` 观测组（19 维运动帧，契约 docs/amp_input.md）+ `se3_train.amp.AMP`（移植 BioInnov/rsl_rl_bioin 的 AMP 扩展：LSGAN + R1、风格奖励 reward_weight×dt×r、rollout 采窗口），钩子在 `se3_train.ppo.Se3PPO`；数据集加载与配置见 docs/amp_dataset.md，由 `tests/test_amp.py` 守护。2026-09-08（A6）用户定：删掉 step_up 前瞻状态机（原做法是身前 0.5 m 三射线探到 0.06–0.22 m 抬升就把高度指令 +0.10 保持 2 s、更高的障碍判为墙转 time_out），改用 `mdp/commands.py` 原有的**地形感知抬高下限**——重采样时按 env 所在列与难度行算 `台阶高 + 0.02 − (−0.12)`，把高度指令采样区间的下界顶到该值（台阶 0.02→0.20 m 对应下限 0.20→0.34 m，上界仍 0.38），只在 `stairs_up` 列生效；随之删掉 `wheel_forward_sensor`、`wall_blocked` 终止与 `-NoStepUp` 入口，`Rough/step_up_*` 三个日志键换成 `Rough/height_cmd_terrain_mean`。标定沿用 stair 线（`tasks/stair/env_cfg.py`）：机体碰撞网格底面在 base_link 下方 0.12 m，再留 0.02 m 余量。与状态机的差别是不再需要前瞻传感器、按列按行静态生效而非事件触发，部署端也不必复现状态机才能对齐训练期的高度指令。同日（A6）用户定的台阶列分列定价，只改生效范围不改数值（`rough/rewards.py`）：① 速度违令二次罚 `command_velocity_error`（2026-09-06 D7 对 D4 从 Flat 整项删除）只在 `stairs_up` 列加回来，权重 -2.0、死区 (0.05, 0.10)、封顶 9 都用删除前的历史值——删它的依据全部来自平地（那里误差小且短暂，99% 代价来自指令阶跃后 1 s 内），而台阶列是 `tracking_lin_vel` 高斯核 σ_move=0.08 压零、没有梯度的那一段（A5 换列后跟踪 2.2→0.5 再没回来）；② 机身高度罚 `flat_base_height` 在 `stairs_up` 列置零（爬升时机身相对脚下地面的高度必然大幅偏离指令，该项等于按爬升幅度罚钱），姿态改由 AMP 与地形感知高度下限管。两项都走按列掩码，平地热身期与其余列与 Flat 基线逐位相同；旋钮 `command_velocity_error_weight`（None 关）与 `zero_base_height_on_terrain`。AMP 掩码、高度下限、分列定价三处指同一组列。2026-09-08（A7）用户定的四处，全部冲着「换列后跟踪暴跌」这一个根因：本机 sim2x 回放证明策略是真坏（A6 model_500 在 2 m/s 上误差 0.02，换列 100 轮后的 model_600 掉到 0.93，A5 同型），不是记账假象；根因是换列那一刻 75% 的 env 拿到跟不上的指令，`tracking_lin_vel` 的核 exp(−err²/0.08) 在误差 0.8 以上恒为 0，既无分也无梯度，而罚项照常，于是唯一有梯度的方向是「别动」，共享 actor 把它学进去连平地一起退化。① 地形列 vx 从 (0.4, 2.4) 收到 (0.4, 0.8) 并与平地课程脱钩（`terrain_lin_vel_x_follow_curriculum=False`）——A6 的 `Rough/command_velocity_error_terrain`=0.95 反解出地形列误差约 1.5 m/s、指令均值 1.4，误差和指令一样大；② 换列改成线性 ramp，地形 env 比例在 500→1000 轮从 0 涨到 1（`ROUGH_FLAT_WARMUP_RAMP_ITERATIONS`，逐 env 固定阈值、单调不回头）；③ 非平地列把核里的 vz 项关掉（`terrain_vz_weight=0`，平地仍 2.0）——爬升必须有垂直速度，而核按 vz² 扣分，vz 0.2 m/s 就乘掉 0.37；④ 日志：`Locomotion/` 六个键进白名单，并在 `rough/rewards.py` 加按列拆开的 `Rough/tracking_lin_vel_{terrain,flat}`、`Rough/{cmd_vx,base_vx,base_vx_error}_terrain`。2026-09-08（A8）用户定：把 `stairs_up` 从通用地形列里拆出来单独定价——vx 1.0–2.4、机身高度指令 0.35–0.38（`ROUGH_STAIR_*`，实现在 `RoughCommandTerm._apply_stair_height` 与`refresh_terrain_override` 的二次覆盖）。起因：A7 把所有非平地列 vx 收到 (0.4, 0.8) 之后，平地能力（sim2x 2 m/s 误差 0.03）与地形列梯度（`base_vx_error_terrain` 0.37、`tracking_lin_vel_terrain` 0.45）都回来了，slope_up 爬到 6.39、平地 6.1，但 stairs_up 1500 轮只从 1.09 挪到 1.11——6 cm 轮子靠 0.8 m/s 的动量翻不过 4 cm 立面。vx 区间与专家数据（Fudan 12–20 cm 爬升，1.5–2.4 m/s）同段。两个已知副作用：① 台阶列高度区间整体高于地形感知抬高下限在最高难度行的 0.34，那条下限在本列被完全吞掉（测试钉住该关系，避免两套机制互相盖）；② vx 1.0–2.4 会把台阶列推回「指令不可达 → 核恒零」的区间，与 A6 的区别是只有 40% 的 env 在其中（A6 为 75%）、迁移有 ramp、且 `command_velocity_error` 专补远场梯度（封顶约 −4.9/s）——平地是否被带塌，看 `Rough/tracking_lin_vel_flat` 与 sim2x 回放。另注：0.35–0.38 显著高于专家的 0.22–0.27，AMP 风格奖励与任务指令的姿态目标更不一致，本轮未动 AMP。2026-09-09（A9）用户定：地形 env 分配改为 flat 30% / stairs_up 70%，slope 与随机起伏、两个下行列的 `proportion` 全设 0（列与日志键保留，mjlab 每列仍留 1 个 env）；ramp 沿用 500→1000。顺带修掉一个观测问题——A8 里 slope_up 爬到 6.4 级正常清块，把 `Rough/*_terrain` 这些「非平地」口径整个稀释了，slope 一去该口径实质等于台阶列。同时新增 `events.log_reward_split_by_column`：每步把奖励表**每一项**在台阶列上的均值记一份（`Rough/rw_<项名>_stairs`，23 项），直接读 RewardManager 的 `_step_reward` 缓冲做掩码均值，不重算奖励也不改奖励数学；A8 只能读出 `command_velocity_error` 在台阶列是 −2.48/s，其余罚项无从定位。由 `tests/test_rough_port.py` 守护 |
| `flat/` | `SE3-WheelLegged-Flat-GRU` / `SE3-WheelLegged-Flat-MLP` / `SE3-WheelLegged-Flat-History-MLP` | 平地行走基模：GRU、单帧 MLP、五帧展平历史 MLP 三个入口共享环境与 PPO 配置，仅网络/观测历史不同。2026-09-06 合并 D2–D8 已验证的改动为默认：腿动作语义 `joint`、动作罚轮分量按归一化单位计价（`action_rate` 轮 1.0、`action_smoothness` 轮 2.0）、删除 `command_velocity_error`、整机质心正对轮轴的静平衡默认站姿与高度默认 v2、高度指令 0.20–0.38、PPO 超参数对齐 kyber_rl_lab（lr 1e-3、entropy_coef 0.01、epochs 5、clip 0.2）。合并后 `Flat-MLP` 的环境与 `Exp-JointActionWheelPriceNoCmdErr` 逐项相同 |
| `flat/` | `SE3-WheelLegged-Flat-Exp-CmdDeadband` / `-Exp-WheelContact` / `-Exp-TiltBarrier` / `-Exp-ActionDelay` / `-Exp-YawCurriculum` | 2026-09-03 抖动对照实验入口：网络与 PPO 完全同 `Flat-MLP`，各自只改一个旋钮（速度违令死区 0.15/0.30、轮离地罚 -30、bad_tilt 6°/25°、动作延迟 20-60 ms、yaw 课程上限 6 rad/s）；基线用 `Flat-MLP` 换随机种子重跑 |
| `flat/` | `SE3-WheelLegged-Flat-Exp-YawGate` / `-Exp-CurriculumRetreat` / `-Exp-YawStep` / `-Exp-AdvanceThreshold` / `-Exp-DeadbandTilt` | 2026-09-04 课程对照实验入口：yaw 上限一律保持 12 rad/s，只改爬升方式（yaw 由 yaw 跟踪 EMA 独立门控、课程可回退滞回、yaw 步长 0.25、推进阈值 0.75），外加把已确证的速度死区与 bad_tilt barrier 两个改动合并的入口 |
| `flat/` | `SE3-WheelLegged-Flat-Exp-JointAction` | 2026-09-04 动作语义改动：4 维 action 直接是四根主动杆的绝对目标角（`target = default + action × scale`），去掉夹角中间量与解码器夹紧；隐含夹角越界交给 MJCF 的 `active_rod` tendon 限位承接。ONNX 契约 decoder 变为 `serialleg_joint.v1`，必须从头重训，旧 checkpoint 与新 sim2x 不可混用 |
| `flat/` | `SE3-WheelLegged-Flat-Exp-JointActionWheelPrice` | 2026-09-05 σ 平衡点实验：在 `Exp-JointAction` 之上只改动作罚项轮分量的定价，撤销 (15/45)² 折价（`action_rate` 轮 1.0、`action_smoothness` 轮 2.0）。诊断：σ 与 entropy_coef 都按归一化动作维度计，折价后一单位轮噪声的代价只剩腿的 1/6.3，平衡点 σ_wheel≈0.85-0.9，且收敛后两项动作罚 72-104% 是纯探索噪声地板；同一定价下 4gs3te0p 曾把 σ 退火到 0.23。预测改后轮 σ 平衡点≈0.28 |
| `flat/` | `SE3-WheelLegged-Flat-Exp-JointActionWheelPriceCriticLr` | 2026-09-05 critic 学习率解耦实验：在 `Exp-JointActionWheelPrice` 之上只把 critic 的 LR 固定为 actor 的初始学习率 `FLAT_LEARNING_RATE`（`se3_train.ppo.Se3PPO`，两 param group 的 Adam，每次 step 前恢复 critic lr），actor 仍走 KL 自适应。诊断：D4 σ 缩到 0.25 以下后共用 LR 被压到 1e-5 地板，critic 一起冻住，Loss/value 出现尖峰 |
| `flat/` | `SE3-WheelLegged-Flat-Exp-JointActionWheelPriceOrient` | 2026-09-05 腿部摆动实验：在 `Exp-JointActionWheelPrice` 之上只把 `tracking_orientation_l2` 权重 -12 → -120。诊断：D4 确定性策略站立时有 0.67 Hz 极限环（腿峰峰 24°、俯仰 rms 2.2°），训练 rollout 里被探索噪声淹没，-12 下这段慢摆只花 0.03/s；-120 时 0.31/s 与动作罚项同量级，安静站立与行进俯仰不受影响 |
| `flat/` | `SE3-WheelLegged-Flat-Exp-JointActionWheelPricePoseHold` | 2026-09-05 腿部摆动实验（选定方案）：在 `Exp-JointActionWheelPrice` 之上只加 `joint_pos_penalty` -1.0（腿关节偏离高度条件默认姿态的 L2 范数，静止 ×5，与 recovery 线同参数）。量级：D4 确定性站立慢摆 1.6/s、D2 式安静站立 0.18/s、行进 0.3 到 0.5/s |
| `flat/` | `SE3-WheelLegged-Flat-Exp-JointActionWheelPriceNoCmdErrCriticLr` | 2026-09-06 critic 学习率解耦复测：在 `Exp-JointActionWheelPriceNoCmdErr` 之上只把 critic 的 LR 固定为 `FLAT_LEARNING_RATE`（`se3_train.ppo.Se3PPO`）。诊断：D8 从 2750 轮起共用 LR 贴 1e-5 地板，此后 `Loss/value` 尾部指数发散（分段最大 0.6 → 1151，中位数始终约 0.5，尖峰为单轮脉冲且下一轮即恢复，故权重未损坏，是估值在新访问状态上失准），确定性站立腿峰峰从 1300 轮的 4.7° 退回 7.4°，reward 峰值 123 → 终点 113。横向对照：贴地板的 D4/D8 最大 `Loss/value` 27.5/991，不贴地板的 D5/D6/D7 仅 4.1/3.7/1.0。D5 在旧基线（仍带 `command_velocity_error`）测过同一改动但结论模糊，故复测 |
| `flat/` | `SE3-WheelLegged-Flat-Exp-JointActionWheelPriceNoCmdErr` | 2026-09-05 速度违令罚实验：在 `Exp-JointActionWheelPrice` 之上只删除 `command_velocity_error`。诊断：该项 99% 的代价来自指令阶跃后 1 s 内物理上跟不上的瞬态（训练平均 -1.44/s，最大单项罚），稳态只有 0.003/s，实际效果是奖励指令跳变后猛冲；高斯核 σ=0.08 在稳态误差处梯度是它的 7 倍 |
| `recovery_discovery/` | `SE3-WheelLegged-Recovery-Discovery-GRU` / `SE3-WheelLegged-Recovery-Discovery-MLP` / `SE3-WheelLegged-Recovery-Discovery-History-MLP` / `SE3-WheelLegged-Recovery-Loco-Grouped-MLP` / `SE3-WheelLegged-Recovery-Loco-Grouped-Gentle-MLP` / `SE3-WheelLegged-Recovery-Loco-Grouped-Teacher-MLP` / `SE3-WheelLegged-Recovery-Loco-Grouped-Gentle-Teacher-MLP` / `SE3-WheelLegged-Recovery-Loco-Grouped-Gentle-TorqueAssist-MLP` / `SE3-WheelLegged-Recovery-Discovery-Ungrouped-MLP` | 唯一倒地自启任务；各入口共享奖励、课程和 PPO 配置，Grouped 使用 loco/recover 分组与五帧历史观测；Teacher 入口变换动作，TorqueAssist 入口保持策略动作原样并按当前机身倾角施加外部扭矩；两种引导均在 play/eval 关闭 |
| `stair/` | `SE3-WheelLegged-Stair-GRU` | CTBC 倒金字塔台阶任务，从 stair checkpoint warm start |
| `jump_pretrain/` | `SE3-WheelLegged-Jump-PreTrain-GRU` | 跳跃预训练阶段，包含 EFGCL 辅助和参考轨迹约束 |
| `jump_finetune/` | `SE3-WheelLegged-Jump-FineTune-GRU` | 跳跃 FineTune 阶段，从 PreTrain checkpoint 继续训练 |

**Flat 基线已于 2026-09-06 冻结**，取 D11 的配置（W&B `mher9vfk`，commit `236666c`）：不带任何命令行覆盖直接跑
`SE3-WheelLegged-Flat-MLP` 即可复现。除已合并的奖励与动作改动外，`num_steps_per_env` 由 64 改为 24（D 系列
全部实验的实际取值；GRU 线的该值同时是 BPTT 窗口，保留 64），`max_iterations` 由 5000 改为 3500（逐轮曲线显示
有信息量的窗口在 3500 轮以内），`randomize_com` 由 ±20 mm 收到 ±5 mm。全部数值由
`tests/test_flat_baseline.py` 逐项守护，改基线必须同步改该测试并在提交信息里写明对照实验编号。

2026-09-06 起 Flat 基线默认值已合并 D2–D8 的已验证改动，早于该日期的 `Flat-Exp-*` 入口（A/B/C 批课程与抖动对照）当时是相对旧基线的单变量，现在跑会落在新基线上；复现旧实验请切到该实验的 commit。其中 `Exp-CmdDeadband` 与 `Exp-DeadbandTilt` 调的是已被删除的 `command_velocity_error` 死区，已显式钉回旧权重以保留对照含义。

阶段命名写在 task id 里。跳跃任务目前只有 `PreTrain` 和 `FineTune` 两个正式入口。

`tasks/recovery/` 仅保留 Recovery-Discovery 使用的环境基配置、奖励、事件和课程实现，
不注册独立 task；所有倒地自启训练必须从 `recovery_discovery/` 的三个正式入口启动。

TorqueAssist 对被采样到的 episode 施加满幅 20 N·m：倾角超过 30° 施力，回到直立带
立即撤力并清零本次辅助计时，再次跌出直立带重新获得完整 3 s 预算。退火只降 episode
采样概率（iter 0-199 全采样，200-499 线性降到 0），不降力矩幅值——原生扫频实测
倒置姿态下 ≤18 N·m 完全翻不起来、19 N·m 需 1.99 s、20 N·m 需 1.66 s，低于阈值的
档位等同于没施力。辅助同时改变动力学与回报，因此 critic 额外观测 3D 辅助状态
（是否被采样 / 是否在施力 / 本次跌倒剩余预算），actor 的 34D 部署契约不变；
play/eval 始终关闭辅助，该观测退化为恒零。

## 台阶任务

`stair/` 是当前台阶训练入口，注册 `SE3-WheelLegged-Stair-GRU`，并保留 `SE3-WheelLegged-Stair-GRU-TrainView` 作为历史 watch/play 别名。正式远程训练和本地值守脚本默认使用原始 task id；只有需要兼容旧 watch 流程时才显式使用 `*-TrainView`。

台阶任务的核心差异集中在 `src/se3_train/tasks/stair/`：

- `env_cfg.py` 使用沿世界系 +x 上升的直线台阶地形 `BoxForwardStairsTerrainCfg`，当前训练 MJCF 为真实闭链 `serialleg_closed_chain_v3_train_obb_trim.xml`。
- `state.py`、`events.py` 和 `observations.py` 管理 CTBC 前馈状态机；actor 仍为 34 维观测，最后 3 维扩展槽在台阶任务中输出 CTBC 左右摆动相位和触发位。
- `rewards.py`、`curriculums.py` 提供台阶爬升奖励、地形等级课程和诊断项。
- `env_cfg.py` 同时接入 recovery replay 状态缓存，用于提升台阶训练中跌倒后的恢复覆盖率。

台阶任务的远程值守使用 `./scripts/run_sim2x.sh` 启动 native MuJoCo/Viser，并在
`Models` 页签选择对应 experiment、run id 和 ONNX。checkpoint 来源、远程连接和本地缓存
目录由当前 machine profile 决定，不属于任务架构契约。

## 单个 task 的目录结构

```text
tasks/<task_name>/
├── __init__.py       # task_id、register()、runner_cls
├── env_cfg.py        # 场景、观测、动作、指令、奖励、终止、课程、事件
├── rl_cfg.py         # PPO / GRU / checkpoint / logger
├── observations.py   # 本任务 actor/critic 观测项
├── rewards.py        # 本任务奖励函数
├── commands.py       # 本任务指令项
├── curriculums.py    # 本任务课程函数
├── terminations.py   # 本任务终止条件
└── events.py         # 本任务 reset / startup 事件
```

`env_cfg.py` 可以复用更基础任务的配置，再覆盖当前任务的差异。例如 `jump_finetune` 基于 `jump_pretrain`，移除 EFGCL 辅助并调整 FineTune 阶段的奖励和课程。

`observations.py`、`rewards.py`、`commands.py`、`curriculums.py`、`terminations.py` 和 `events.py` 可以转发共享实现，也可以放本任务独有实现。原则是从 task 目录能看出该任务实际依赖了哪些 MDP 代码。

## 注册流程

`src/se3_train/__init__.py` 调用 `se3_train.tasks.register_all_tasks()`。`tasks/__init__.py` 只负责导入并注册当前正式任务。

单个 task 的 `__init__.py` 负责：

```python
TASK_ID = "SE3-WheelLegged-Example-GRU"


def register() -> None:
    """注册 Example 任务。"""
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=rl_cfg(),
        runner_cls=Se3WarmStartRunner,
    )
```

没有注册到 `tasks/__init__.py` 的目录不属于正式训练入口。

## 新增实验

1. 复制最接近的目录，例如：

   ```bash
   cp -R src/se3_train/tasks/jump_finetune src/se3_train/tasks/jump_high
   ```

2. 修改 `jump_high/__init__.py`：

   - `TASK_ID` 使用明确阶段名，例如 `SE3-WheelLegged-JumpHigh-FineTune-GRU`
   - docstring 写清楚观测维度、训练阶段和用途
   - runner 继续使用 `Se3WarmStartRunner`，除非新任务确实不需要 warm start 逻辑

3. 在 `env_cfg.py` 里改任务差异：

   - 观测维度和观测项
   - command 分布
   - reward 项和权重
   - termination 条件
   - curriculum 调度
   - reset / startup events

4. 在 `rl_cfg.py` 里改训练差异：

   - GRU / MLP 结构
   - `max_iterations`
   - `save_interval`
   - `resume`
   - `load_run`
   - `load_checkpoint`
   - logger 配置

5. 在 `tasks/__init__.py` 导入并调用 `register()`。

6. 更新本文档的“当前任务”表。实验还不准备作为正式入口时，不注册到 `tasks/__init__.py`。

## 验证

修改训练任务后至少运行：

```bash
uv run ruff format --check src/se3_train
uv run ruff check src/se3_train
git diff --check
```

然后做 task 构造 smoke：

```bash
uv run python - <<'PY'
from se3_train.tasks import (
    flat,
    jump_finetune,
    jump_pretrain,
    recovery_discovery,
    rough,
    stair,
)

for module in (
    rough,
    flat,
    recovery_discovery,
    stair,
    jump_pretrain,
    jump_finetune,
):
    cfg = module.env_cfg(play=True)
    rl = module.rl_cfg(smoke=True)
    print(module.TASK_ID, len(cfg.observations["actor"].terms), rl.max_iterations)
PY
```

改了跳跃 task 时，再跑对应 CLI smoke：

```bash
uv run se3-train SE3-WheelLegged-Jump-FineTune-GRU \
  --env.scene.num-envs 1 \
  --gpu-ids None \
  --agent.max-iterations 1 \
  --agent.logger tensorboard \
  --agent.resume False
```
