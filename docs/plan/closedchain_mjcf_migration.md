# 闭链 MJCF 主模型迁移计划

## 目标

将 SerialLeg 训练和 sim2sim 的默认机器人模型从串联开链 MJCF 切换为显式闭链四连杆 MJCF。闭链模型成为唯一主模型；旧开链 XML 保留为显式 `openchain` variant，用于 A/B 定位和回退诊断。

第一版目标是跑通基础站立、行走、恢复和 sim2sim，不迁移跳跃参考轨迹、RSI 和轨迹跟踪。

## 已确认决策

1. 默认模型切换为闭链四连杆 + 气弹簧 MJCF。
2. 旧开链 XML 文件保留，训练端和 sim2sim 端保留显式 `openchain` variant。
3. policy 仍为 6 维动作，actor 观测维度保持 32 维。
4. 腿部 action 和 actor 腿部观测改为主动杆坐标。
5. action 顺序固定为 `[LF, LB, RF, RB, l_wheel, r_wheel]`。
6. `LF/RF` 对应 `lf0_Joint/rf0_Joint`，`LB/RB` 对应 `l_drive_bar_Joint/r_drive_bar_Joint`。
7. `lf1_Joint/rf1_Joint` 是被动输出小腿角，不进 action，不作为 actor 主关节观测。
8. `l_drive_bar_Joint/r_drive_bar_Joint` 必须是相对 base 的独立主动铰，不能作为 `lf0_Link/rf0_Link` 子关节。
9. 闭链端点约束使用 `connect site1/site2`，表达销轴点约束。
10. MJCF 保留气弹簧 actuator，第一版用 300 N 常力。
11. 6 个电机 actuator 继续由训练端和 sim2sim 代码程序化添加。
12. 四杆/五杆几何采用 Infantry 3/4 共同参数，不绑定 Infantry 3 或 Infantry 4 的编码器 offset。
13. 默认站姿保持当前 2027 MJCF 外形，通过反解得到主动杆默认角。
14. 跳跃参考轨迹第一版先不迁移，避免和闭链主模型切换耦合。

## 参考依据

- 实机五杆解算：`D:/robomaster/good_code/2026/serial_Leg_whole/serialleg2026/Custom/Controller/planar5rod.c`
  - `pfr_forward_kinematic()` 使用两根主动杆 `th[1] / th[2]`，先由差值解四杆输出角，再计算轮心 `L0/phi0` 和 Jacobian。
- 实机 profile 参数：`D:/robomaster/good_code/2026/serial_Leg_whole/serialleg2026/Custom/Config/robot_profile.c`
  - Infantry 3/4 共同几何：`four_rod = [60.5, 170.0, 65.0, 180.0] mm`，`five_rod_l = [180.0, 200.0] mm`，`spring_length = [194.1768, 64.895] mm`。
- 实机左右符号：`D:/robomaster/good_code/2026/serial_Leg_whole/serialleg2026/Custom/Tasks/Src/task_estimate.c`
  - 右腿电机角反馈取负，仿真轴向和导出映射需要单独验证。
- MuJoCo 闭链建模参考：`Albusgive/mujoco_learning` 的 `MJCF/Chapter11-equality/connecting_rod.xml`。
- MuJoCo 官方 XML Reference：`equality/connect/weld/tendon/actuator` 语义。

## 目标接口契约

### Joint 语义

闭链主模型需要区分主动电机坐标和输出机构坐标：

| 语义 | 左腿 | 右腿 | 用途 |
| --- | --- | --- | --- |
| 主动髋杆 | `lf0_Joint` | `rf0_Joint` | action、actor 腿部观测、PD 控制 |
| 主动膝驱动杆 | `l_drive_bar_Joint` | `r_drive_bar_Joint` | action、actor 腿部观测、PD 控制 |
| 输出小腿角 | `lf1_Joint` | `rf1_Joint` | 几何、奖励、终止、诊断、可选 critic 特权观测 |
| 轮子 | `l_wheel_Joint` | `r_wheel_Joint` | action、actor 轮子观测、速度控制 |

### Action 顺序

第一版闭链 policy contract：

```text
[LF, LB, RF, RB, l_wheel, r_wheel]
```

这与 Infantry 分支的实机控制语义对齐：`LF/RF` 是前主动杆，`LB/RB` 是后主动杆。

### JointGroup 建议

`src/se3_shared/robot.py` 需要从“6 个关节等于全部关节”的假设切换为按语义分组：

- `JointGroup.LEGS`：主动杆坐标 `[LF, LB, RF, RB]`。
- `JointGroup.WHEELS`：轮子 `[l_wheel, r_wheel]`。
- `JointGroup.OUTPUT_LEGS`：输出机构角 `[lf0, lf1, rf0, rf1]` 或至少提供 `OUTPUT_KNEES=[lf1, rf1]`。
- `JointGroup.POLICY_JOINTS`：policy 观测/动作使用的 6 个关节。
- `JointGroup.ALL_MODEL_JOINTS`：需要时包含被动输出和闭链辅助关节，不能被 actor 默认使用。

避免继续用硬编码列索引表达 actuator force 或 joint tensor 语义；闭链模型会引入额外被动关节和气弹簧 actuator。

## 实施阶段

### Phase 0：锁定验证工具

先补最小验证脚本，避免模型改完后只能靠训练崩不崩判断。

- 新增或扩展一个模型 introspection 脚本，打印 `njnt/nq/nv/nu/neq/ntendon`、joint 名称、qpos/qvel 地址、actuator 名称和 transmission。
- 新增闭链 FK 校验脚本：
  - 输入主动杆角 `[LF, LB, RF, RB]`。
  - 读取 MuJoCo 中 `lf1/rf1` 输出角、轮心位置、equality residual。
  - 与 `planar5rod.c` 等价 Python 解算结果对比。
- 新增默认站姿反解脚本：
  - 保持当前 2027 外形目标，包括 base height、轮心位置和左右对称。
  - 反解主动杆默认角，写入候选 `RobotConfig.default_dof_pos`。

验收：

- 两个 XML 都能被 MuJoCo 编译。
- 闭链 XML 的 equality residual 在默认站姿下接近 0。
- 主动杆默认角能复现当前外形站姿，轮心高度和左右对称误差在可解释范围内。

### Phase 1：重建闭链 MJCF

旧草稿 `serialleg_fidelity_cylinder_wheels_closedchain_spring.xml` 保留为参考；当前交付模型为 `serialleg_closed_chain_v2_spring.xml`，几何来自 `robot_lab_serialleg` 历史闭链模型。

要做：

- 用 Infantry 3/4 共同参数重建四杆和五杆：
  - `L_AB = 60.5 mm`
  - `L_BC = 170.0 mm`
  - `L_CD = 65.0 mm`
  - `L_AD = 180.0 mm`
  - 主腿/轮心几何：`180.0 mm + 200.0 mm`
- 将 `l_drive_bar/r_drive_bar` 改为从 `base_link` 派生的独立主动铰。
- 保留输出小腿 `lf1/rf1` 作为被动 hinge。
- 用 `connect site1/site2` 闭合四杆末端。
- 保留 spatial tendon 表示气弹簧路径。
- 将气弹簧 actuator 常力从 100 N 改为 300 N。
- 确认闭链辅助杆几何 `contype=0 conaffinity=0`，避免辅助连杆参与地面碰撞。

验收：

- `uv run` 的 MuJoCo 编译检查通过。
- `se3-joint-viewer --closedchain-spring` 可打开并拖动主动杆。
- 默认站姿无明显爆约束、穿地或自碰问题。
- 交付闭链 MJCF 文件路径和查看命令。查看命令必须固定 `base_link`，允许主动拖动 `LF/LB/RF/RB` 主动杆，其他杆件在重力下自由响应，且 300 N 气弹簧力实际生效。

### Phase 2：共享配置和索引重构

修改 `src/se3_shared/robot.py` 是核心风险点。

要做：

- 将 `Joint` 枚举扩展为主动杆和轮子的 policy joint。
- 新增输出机构 joint 名称常量。
- `RobotConfig.default_dof_pos` 改为新 action 顺序 `[LF, LB, RF, RB, l_wheel, r_wheel]`。
- `action_scale` 和 `torque_limits` 按新 action 顺序重排。
- 删除或隔离 `LEG_ACTUATORS=[0,1,2,3]` 这类全局硬编码，改用 actuator 名称解析。

验收：

- actor 观测仍是 32 维。
- action 仍是 6 维。
- `JointGroup.LEGS` 不再指向 `lf1/rf1`。
- 所有用到输出膝角的奖励、终止和诊断显式使用输出组。

### Phase 3：训练端切默认闭链

修改 `src/se3_train/robot_cfg.py`：

- 默认 variant 改为闭链。
- `openchain/default` 显式返回旧 XML，`closedchain/closedchain_spring` 返回新 XML。
- leg actuator target 从旧 `[lf0, lf1, rf0, rf1]` 改为 `[lf0, l_drive_bar, rf0, r_drive_bar]`。
- 初始状态只给 policy joints 和必要被动输出 joint；被动闭链 joint 的初值必须由默认站姿反解或 `mj_forward` 稳定处理。
- 训练端 actuator force 日志和 torque penalty 按 actuator 名称取，不按列号猜。

需要扫描并分类修改：

- `src/se3_train/mdp/observations.py`：actor 腿部观测用主动杆。
- `src/se3_train/mdp/rewards.py`：torque penalty 用主动电机 actuator；几何类奖励用输出机构。
- `src/se3_train/mdp/jump_rewards.py`：第一版闭链下跳跃轨迹相关项先禁用或显式报错。
- `src/se3_train/mdp/terminations.py`：关节限位/姿态类逻辑分清主动杆和输出杆。
- `src/se3_train/mdp/events.py`：reset 默认姿态、随机姿态、RSI 注入不要误写被动输出角。

验收：

- `SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1 --gpu-ids None` 不崩。
- recovery/flat smoke 中 actor obs shape 和 action shape 与原 policy contract 一致。
- 日志中主动杆角、输出膝角、equality residual 都能观测。

### Phase 4：sim2sim 对齐

修改 `src/se3_sim2sim/robot.py` 和配置：

- 默认 `model_path` 改为闭链 XML。
- `_build_model()` 仍程序化添加 6 个电机 actuator。
- MJCF 中已有 2 个气弹簧 actuator，所以 `data.ctrl` 不能再整体赋值为 6 维数组。
- 新增 actuator name resolver：
  - 找到 `lf0_Joint_motor/l_drive_bar_Joint_motor/rf0_Joint_motor/r_drive_bar_Joint_motor/l_wheel_Joint_motor/r_wheel_Joint_motor` 的 ctrl 地址。
  - 只写这 6 个电机槽位。
  - 气弹簧槽位保持 MJCF actuator 自身行为。
- `dof_pos/dof_vel` 按 policy joint 名称读取，不按 qpos 连续位置读取。
- telemetry 同时输出主动杆角和输出膝角。

验收：

- `uv run se3-sim2sim --viewer none --max-steps 200` 能加载闭链模型。
- `data.ctrl` 维度与 `model.nu` 一致，6 个电机 ctrl 只写对应槽位。
- 默认站姿静置 1-2 秒无 NaN、无爆约束、无非预期腿部触地。

### Phase 5：文档和操作入口

要做：

- 更新 `docs/common_mistakes.md`，加入“闭链后 joint 顺序不能按 MJCF qpos 位置猜”的条目。
- 更新 `docs/train.md`，说明默认闭链、`openchain` variant 的回退方式。
- 更新 `docs/background.md` 的关节语义和动作空间说明。
- 在 `justfile` 中保留显式 openchain smoke 命令或环境变量示例。

验收：

- 新人只看文档能知道：
  - action 顺序是 `[LF, LB, RF, RB, l_wheel, r_wheel]`。
  - `lf1/rf1` 是被动输出角。
  - 如何临时切回 openchain。

## 风险和缓解

### 风险 1：joint tensor 顺序被被动关节污染

闭链模型会多出被动关节，任何依赖 `joint_pos[:, 0:6]` 语义的代码都有风险。

缓解：所有训练和 sim2sim 的 policy joint 都按名称解析，输出机构角单独命名。

### 风险 2：actuator force 和 ctrl 槽位错位

MJCF 自带气弹簧 actuator 后，`model.nu` 不再等于 6。

缓解：按 actuator 名称维护 ctrl 写入索引和 torque 日志索引。

### 风险 3：默认站姿反解错误导致训练从坏状态开始

闭链默认角不能沿用旧开链 `default_dof_pos`。

缓解：先写反解和 MuJoCo FK 校验脚本，再改训练默认值。

### 风险 4：右腿轴向和实机符号不一致

实机右腿反馈角取负，MJCF 左右轴向必须用测试验证。

缓解：用小角度正向扰动检查轮心运动方向、输出膝角方向和 `planar5rod` 解算方向。

### 风险 5：跳跃轨迹误读新 joint 语义

旧 `q_ref` 是开链输出膝角语义，不能直接用于主动杆 policy contract。

缓解：第一版闭链主模型不迁移跳跃轨迹；若闭链下运行跳跃任务，相关轨迹跟踪项应显式禁用或报错。

## 推荐执行顺序

1. 写模型 introspection 和闭链 FK 校验脚本。
2. 用 Infantry 3/4 几何重建闭链 XML。
3. 反解默认站姿主动杆角。
4. 重构 `se3_shared` joint/action 语义。
5. 修改训练端 actuator target 和观测/奖励索引。
6. 修改 sim2sim ctrl 写入和 telemetry。
7. 跑 MuJoCo 编译检查、joint viewer、sim2sim 静置。
8. 跑 CPU smoke。
9. 更新训练/背景/踩坑文档。

## 第一版不做

- 不迁移跳跃参考轨迹和 RSI。
- 不复刻 Infantry 非线性气弹簧曲线。
- 不删除旧开链 XML。
- 不兼容旧 checkpoint。
- 不改变 actor observation 总维度。

## 完成标准

- 默认训练和 sim2sim 都使用闭链模型。
- openchain variant 仍可显式启用。
- policy action/actor 腿部观测语义为主动杆坐标。
- 闭链默认站姿稳定，无 NaN、无约束爆炸、无意外穿地。
- 已给出闭链 MJCF 文件和固定 base 的查看命令；该命令下可主动拖动主动杆，其他杆件受重力，气弹簧力生效。
- CPU smoke 通过。
- 文档说明新 joint/action contract 和回退方式。
