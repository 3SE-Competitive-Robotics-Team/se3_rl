# M28：整张奖励表换成复旦 v3 的全地形统一奖励（2026-09-26，用户定）

来源：yly-true/fudan_rl_wheel_leg `8204e853` 上台阶 v3 快照（`上台阶3_angz+添加随机地形，进一步训练v3_速度上限拉大…`）。
两边奖励的完整清单与差异见 2026-09-26 的对照（我们 29 项、其中 11 项按列生效或改参数、无统一限幅；复旦 16 项、
实际生效 15 项、全地形一致、逐项限幅）。

## 改动

任务入口 `SE3-WheelLegged-Rough-Exp-FudanReward`（`env_cfg(reward_set="fudan_v3")`），与 `SE3-WheelLegged-Rough`
只差奖励表：指令、课程（速度课程、推力、官方升降级、平地热身、二级台阶门控）、终止、观测、域随机化全部不变。

奖励表整张替换为 15 项（`tasks/rough/fudan_rewards.py`），每项在函数内乘复旦权重后限幅到每秒 ±1、注册权重固定 1，
与复旦 `clip(原值·权重·dt, ±dt)` 逐项等价；没有存活奖励、终止罚、接触罚与任何按列开关：

| 项 | 复旦权重 | 原值 |
|---|---|---|
| tracking_lin_vel | 1.0 | 1.3·exp(−e²/0.25) |
| tracking_lin_vel_enhance | 1.0 | 1.3·(exp(−e²/2.5) − 1) |
| tracking_ang_vel | 1.0 | exp(−e²/0.25) |
| base_height | 1.0 | 1.5·exp(−1000·e²)，高度相对机身周围 77 点地面均值 |
| base_height_enhance | 1.0 | exp(−e²/0.01) − 1 |
| nominal_state | −1.0 | 左右虚拟腿角差² |
| lin_vel_z | −0.1 | vz² |
| ang_vel_xy | −0.07 | ωx² + ωy² |
| orientation | −20 | gx² + gy² |
| dof_vel | −5e-5 | 四个腿主动杆速度² |
| dof_acc | −2.5e-7 | 六个受控关节加速度² |
| torques | −1e-4 | 六个执行器力矩² |
| action_rate | −0.05 | 六维动作一阶差分² |
| action_smooth | −0.05 | 腿四维动作二阶差分²（前两步缓存 reset 清零） |
| dof_pos_limits | −1.0 | 腿软限位越界量（闭链口径） |

原 tracking_lin_vel 只作 shadow 调用：速度课程读它写的 `Locomotion/tracking_lin_vel_reward_curriculum`，
去掉速度上限就永远停在 0；它的返回值不计入奖励，`Locomotion/*`、`Rough/*_terrain`、`Rough/*_stairs` 诊断照常。

## 已知口径差异（按复旦原权重照搬，不做换算）

- 控制 50 Hz（复旦 100 Hz）、腿动作缩放 0.25 rad、轮 45 rad/s（复旦 0.5 rad、10 rad/s）：action_rate / action_smooth 的
  物理价格与复旦不同。
- 虚拟腿角用髋关节（lf0/rf0_Link 原点）→ 轮心连线在机身系相对竖直的夹角（闭链等效虚拟腿），复旦是两连杆 FK。
- 高度指令沿用本仓库 0.20–0.38 m（复旦 0.09–0.33，车不同）；pitch/roll 指令仍在观测里，但复旦奖励只把姿态压向水平，
  这两维指令不再有奖励意义。
- 终止沿用本仓库（倾角 30° 持续 2 s 等，复旦是 84° 持续 1 s），且复旦没有摔倒罚，本 run 也没有。

## run 设置与对照

nulltask1 六卡 × 8192 envs、8000 轮、保存间隔 200、seed 42，W&B project `SE3-WheelLegged-Rough`。

对照：代码上最接近的是 M25（同 commit 系列、同配置只差奖励），但 M25 是两卡；六卡且配置接近的只有 M24（旧代码：门控 bug、
从未推过）。**批量与代码两个混杂只能二选一**，读结果时按"与 M25 比奖励影响、与 M24 比量级"两头看。

## 判据

- 能学起来：`Train/mean_episode_length` 与平地跟踪 `Rough/tracking_lin_vel_flat` 不塌；`Curriculum/command_vel/lin_vel_x_max`
  在热身期正常推到 2.4（速度课程信号没断）。
- 台阶：`Curriculum/terrain_levels/stairs_up`、二级台阶两列的等级，台阶列 `Rough/base_vx_error_stairs`。
- 没有存活奖励与摔倒罚：看 `Episode_Termination/*` 是否出现"主动求死"（终止率随训练上升、episode 变短）。
- 确定性回放：2000/4000/6000/7999 跑台阶通关（`scripts/eval_stair_climb.py`）、立面事件（riser_events）与固定指令扫描，
  与 M25 同轮次比较。

## 启动记录

2026-09-26 10:46（Pod 时区）启动，commit `6f25ca9`（Pod 仓库 `xyh/925`，子模块 `e753ce6`：外部提交 `079cc8d` 的子模块更新
一并以 bundle 同步，Pod 工作树干净），启动器 DryRun 通过。启动前本地 114 个 unittest 通过、Fudan 入口 CLI smoke 5/5，
真实 env 核对：15 项、注册权重全 1、逐项每秒贡献 |·| ≤ 1、课程日志键在写。

| 标签 | run | W&B | PGID | state dir |
|---|---|---|---|---|
| M28 | `2026-09-26_10-46-35_rough-M28-fudanv3reward-seed42-6x8192-8k` | `ayrbsbfy` | 562939 | `20260926T024628Z-17473` |

首轮核验：在迭代、无报错、无 nefc overflow，热身期 2.94–3.00 s/轮；GPU0 13.3 GB / 59%，GPU1–5 12.9 GB / 87–91%。
36 轮时 `Rough/tracking_lin_vel_flat` 0.70、速度课程 `vel_ema` 0.38 在涨（shadow 调用写的课程键在刷新；
`vel_tracking_lin_vel` 显示 nan 是当步没有新样本的占位，既有行为）。热身结束 env 分到七列后按 `7c29rmkd` 的经验
升到约 3.8 s/轮，8000 轮约 8 h。
