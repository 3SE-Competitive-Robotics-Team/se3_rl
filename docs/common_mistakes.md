# 常见错误手册（Common Mistakes）

> 本书用于记录本仓库开发过程中反复出现的、容易忽视的、或修复成本较高的常见错误。
> 每一条包含：错误表象、根因、正确的做法、修复 commit 引用（如有）。
> 每一条的描述核心是 MJCF / 物理引擎内涵的不可变约束，而不是某次调参的经验。

---

## 1. 关节轴方向：MJCF 中 6 个受控关节的物理约束

SerialLeg 的 6 个受控关节在正式 OBB 闭链模型中的 axis 定义如下：

| 关节 | 轴方向 | 与对侧关系 |
|---|---|---|
| `lf0_Joint` 左前主动杆 | `(0, 1, 0)` | 与 rf0 **反向** |
| `l_drive_bar_Joint` 左后主动杆 | `(0, 1, 0)` | 与右后主动杆 **反向** |
| `l_wheel_Joint` 左轮 | `(0, 1, 0)` | 与右轮 **反向** |
| `rf0_Joint` 右前主动杆 | `(0, -1, 0)` | 与 lf0 **反向** |
| `r_drive_bar_Joint` 右后主动杆 | `(0, -1, 0)` | 与左后主动杆 **反向** |
| `r_wheel_Joint` 右轮 | `(0, -1, 0)` | 与左轮 **反向** |

这是一个**不可变更的物理事实**：左右侧关节轴方向的一致性并不统一，处理左右对称计算时必须先查 MJCF，不可凭直觉假设。

### 对编程的影响

- **腿部主动杆**：左右轴方向相反，对称姿态使用 `q_L + q_R`，不能使用旧模型的同号假设。
- **轮子**：左右轮轴方向相反。在广义坐标中，同号广义速度对应轮子反向旋转，异号广义速度对应轮子同向旋转。

### 检查清单

在以下场景中必须对照上表确认符号关系：
- 设计左右对称性奖励（`joint_mirror` 等）
- 写轮速相关的惩罚/奖励项
- sim2sim 里程计计算
- 诊断指标中涉及关节差/和的统计
- 参考轨迹生成（IK/FK）中的对称性假设

---

## 2. 闭链后不能按 MJCF qpos 顺序猜 policy 关节

闭链模型在旧 6 个关节之外新增了 `l_drive_bar_Joint/r_drive_bar_Joint` 和 `l_coupler_Joint/r_coupler_Joint`。如果后续临时导入带气弹簧 actuator 的 MJCF，气弹簧 actuator 也可能排在程序化电机 actuator 前面。因此 `robot.data.joint_pos[:, 0:6]`、`data.ctrl[:] = torque6`、`actuator_force[:, 0:4]` 这类写法会把被动关节或气弹簧槽位误当成 policy 电机，表现为观测错位、力矩惩罚读到气弹簧、sim2sim 控制维度和 `model.nu` 不一致。

正确做法是按名称解析：policy 动作/actor 腿部观测只使用 `[lf0_Joint, l_drive_bar_Joint, rf0_Joint, r_drive_bar_Joint, l_wheel_Joint, r_wheel_Joint]`；输出膝角 `lf1_Joint/rf1_Joint` 单独用于几何诊断和必要的奖励/终止；闭链主动杆限位使用同侧两主动杆夹角，当前装配分支下左腿为 `LF-LB`，右腿为 `RB-RF`，不使用后主动杆绝对角；sim2sim 只写 6 个 `*_motor` ctrl 槽位。仓库不再提供其他模型语义的隐式回退。
