# RJ1：rough 合入跳跃（M54 + J10 跳跃）（2026-10-02，用户定）

## 决策来源

J7 → J10 验证了跳跃 mimic 的几件事：无下蹲参考（站姿 0.22 m）"flag 一到就蹬"可行（J7）；不需要 RSI（J9）；
时间信息必须给网络（J8 去掉参考观测学不出），一维相位 jump_phase 即可代替 20 维参考帧（J10，1000 轮时腿长跟踪略差于 J9）。
用户定：不等 J10 跑完，按 J10 的做法合入 rough；参考尾巴不缩短；跳跃样本的高度指令像 J10 一样固定；只在平地列跳，
平地列 30% 的样本；从头训（不从 M54 续训）。

## 改动

`SE3-WheelLegged-Rough-Exp-RJ1` = M54 + `env_cfg(jump_mimic=True)`：

- **观测不变**：actor / critic 与 M54 完全相同（34 维 actor 里本来就有 [jump_flag, 目标高度, jump_phase] 三个槽位，M54 恒 0）。
- **指令项** `RoughJumpCommandTerm`（rough/commands.py）：关掉旧 JumpCommandTerm 的跳跃生命周期（原地跳、旧轨迹），挂上
  与 J 系列共用的参考时钟 `jump_mimic.clock.JumpReferenceClock`；
  - 每回合 reset 时在平地列按 30% 抽跳跃样本：整回合高度指令固定 0.22（参考站姿），不参加高姿态起步序列；
  - 只有跳跃样本触发跳跃：站满 1 s 后每秒 0.2 次（J10 为 0.5，跳跃时间占 29%，rough 降低免得挤占地形训练）；
  - 参考 `jump_ref_v2_nocrouch_h022`（0.20 / 0.30 / 0.40 / 0.50 m，约 1.4 s），jump_phase = 参考时刻 / 1.5 s；
  - 触发时 vx 夹到 ±1.5（J10 范围）、yaw / pitch / roll 置 0，跳跃期间冻结速度 / 姿态 / 高度指令；
  - 部署包络 jump_flag (0, 1)、jump_target_height (0, 0.5)、jump_phase (0, 0.973)，ONNX 带顶层 jump_reference。
- **奖励**：J10 的四项模仿奖励（腿长 3.0、机身高度 3.0、竖直速度 1.5、接触 1.0）只计跳跃样本（非跳跃样本高度 0.20–0.38，
  不能被 0.22 站姿帧拉住）；跳跃期间静站罚、接触力罚置零，速度跟踪核去掉 vz 项（`zero_vz_when_jumping`）；机身高度罚、
  轮 / 腿离地罚本来就按 jump_flag 屏蔽。
- **终止**：跳跃期间机身高度或腿长偏离参考 > 0.12 m 提前终止（同 J10）；摔倒罚改为 `is_terminated_except`，偏离参考终止
  不吃 −500。平地列 env 原点即地面（轮底相对原点约 −0.002 m），参考机身高度口径成立。
- 其余全部照 M54：地形、课程、台阶奖励、执行链（T-N ×0.8、气弹簧前馈）、PPO。

## 共用参考时钟

`jump_mimic/clock.py` 的 `JumpReferenceClock`（触发、推进、写指令跳跃三维、导出部署参考）由 J 系列的 `JumpMimicCommandTerm`
与 rough 的 `RoughJumpCommandTerm` 共用。J 系列改用它后，同一种子 250 步 rollout 的观测 / 奖励 / 指令哈希与重构前完全一致
（J7 含 RSI、J10 相位两种配置）。

## 验证

- RJ1 与 M54 配置对比：指令项换成 RoughJumpCommandCfg（旧生命周期关）；新增 4 项模仿奖励、改 4 项（速度跟踪、静站罚、
  接触力罚、摔倒罚）、新增 mimic_deviation 终止；观测完全相同。
- 功能检查（`.scratch/rj1_check.py`，CPU 96 env，平地热身期全在平地列）：跳跃样本 30（占平地 31%），高度指令全为 0.22，
  非跳跃样本 0.203–0.38；250 步触发 35 次全是跳跃样本，触发时 vx ≤ 1.5、yaw / pitch / roll = 0、flag = 1；
  非跳跃样本模仿奖励恒 0；只因偏离参考终止的 env 不吃摔倒罚；奖励与观测有限。
- `tests.test_rough_port` + `tests.test_onnx_metadata` + `tests.test_flat_baseline` 79 项通过；CPU smoke（1 env、5 轮）通过，
  导出 ONNX 被 runtime 加载为 34 维、四条参考、相位尺度 1.5，触发后指令 [1, 0.4, 0.0133] 与训练端一致。

## 验收

- 跳：平地列回放四档 × 四速的离地高度、起跳延迟、落地（`.scratch/j4_moving_jump.py` 需在 rough 场景平地上跑）。
- 台阶不退化：`riser_events.py` 与二级台阶回放，对照 M54。
