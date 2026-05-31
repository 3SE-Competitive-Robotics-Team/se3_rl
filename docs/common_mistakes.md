# 常见错误手册（Common Mistakes）

> 本书用于记录本仓库开发过程中反复出现的、容易忽视的、或修复成本较高的常见错误。
> 每一条包含：错误表象、根因、正确的做法、修复 commit 引用（如有）。
> 每一条的描述核心是 MJCF / 物理引擎内涵的不可变约束，而不是某次调参的经验。

---

## 1. 关节轴方向：MJCF 中 6 个受控关节的物理约束

SerialLeg 的 6 个受控关节在 `serialleg_fidelity_cylinder_wheels.xml` 中的 axis 定义如下：

| 关节 | 轴方向 | 与对侧关系 |
|---|---|---|
| `lf0_Joint` 左髋 | `(0, -1, 0)` | 与 rf0 **同向** |
| `lf1_Joint` 左膝 | `(0, -1, 0)` | 与 rf1 **同向** |
| `l_wheel_Joint` 左轮 | `(0, 1, 0)` | 与 r_wheel **反向** |
| `rf0_Joint` 右髋 | `(0, -1, 0)` | 与 lf0 **同向** |
| `rf1_Joint` 右膝 | `(0, -1, 0)` | 与 lf1 **同向** |
| `r_wheel_Joint` 右轮 | `(0, -1, 0)` | 与 l_wheel **反向** |

这是一个**不可变更的物理事实**：左右侧关节轴方向的一致性并不统一，处理左右对称计算时必须先查 MJCF，不可凭直觉假设。

### 对编程的影响

- **腿部（lf0/rf0、lf1/rf1）**：由于左右髋/膝轴方向相同，且左右腿通过基座两侧的安装位置（`lf0_Link` 在 `(0, 0.16885, 0)`，`rf0_Link` 在 `(0, -0.16885, 0)`）自然实现了运动学对称，**相同的广义关节位置即产生对称的机身姿态**。因此镜像约束可直接用 `q_L - q_R` 表达，无需符号反转。
- **轮子（l_wheel/r_wheel）**：左右轮轴方向相反。在广义坐标中，**同号**广义速度对应轮子反向旋转（yaw 扭转），**异号**广义速度对应轮子同向旋转（前进/后退）。这一关系与直觉相反，最容易出错。

### 检查清单

在以下场景中必须对照上表确认符号关系：
- 设计左右对称性奖励（`joint_mirror` 等）
- 写轮速相关的惩罚/奖励项
- sim2sim 里程计计算
- 诊断指标中涉及关节差/和的统计
- 参考轨迹生成（IK/FK）中的对称性假设

---

## 2. 闭链后不能按 MJCF qpos 顺序猜 policy 关节

闭链模型在旧 6 个关节之外新增了 `l_drive_bar_Joint/r_drive_bar_Joint` 和 `l_coupler_Joint/r_coupler_Joint`，MJCF 自带的气弹簧 actuator 也会排在程序化电机 actuator 前面。因此 `robot.data.joint_pos[:, 0:6]`、`data.ctrl[:] = torque6`、`actuator_force[:, 0:4]` 这类写法会把被动关节或气弹簧槽位误当成 policy 电机，表现为观测错位、力矩惩罚读到气弹簧、sim2sim 控制维度和 `model.nu` 不一致。

正确做法是按名称解析：policy 动作/actor 腿部观测只使用 `[lf0_Joint, l_drive_bar_Joint, rf0_Joint, r_drive_bar_Joint, l_wheel_Joint, r_wheel_Joint]`；输出膝角 `lf1_Joint/rf1_Joint` 单独用于几何诊断和必要的奖励/终止；闭链主动杆限位使用同侧两主动杆夹角，当前装配分支下左腿为 `LF-LB`，右腿为 `RB-RF`，不使用后主动杆绝对角；sim2sim 只写 6 个 `*_motor` ctrl 槽位，保留气弹簧 actuator 的 MJCF 行为。需要 A/B 定位时显式设置 `SE3_ROBOT_MJCF_VARIANT=openchain`，不要让闭链代码隐式退回旧索引假设。
