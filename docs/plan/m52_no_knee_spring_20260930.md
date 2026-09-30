# M52：M51 + 训练 plant 去掉 300 N 膝气弹簧（2026-09-30，用户定）

## 改动（相对 M51 唯一差异：膝气弹簧）

`env_cfg(knee_gas_spring=False)` → `get_serialleg_closedchain_cfg(knee_gas_spring=False)`：在内存 spec 里删掉
`l_knee_gas_spring` / `r_knee_gas_spring` 两个恒力 tendon actuator（MJCF 里 gain 0、bias 300 N）。MJCF 文件不动，
挂点 site 与 spatial tendon 保留（tendon 无刚度/阻尼，没有 actuator 就不出力）。

随弹簧一起去掉、只在有弹簧时有意义的两项：

| 项 | M51 | M52 |
|---|---|---|
| DR 事件 `knee_spring_force`（左右独立 ×0.9–1.1） | 有 | 删除 |
| critic 特权观测 `knee_gas_spring_force`（2D） | 有 | 删除（critic 输入少 2 维，actor 34 维不变） |
| 动作项 `knee_gas_spring_force` | 300 | 0（只用于前馈补偿与 ONNX 契约） |
| 电机侧前馈补偿 | 关 | 关（开着会报错） |

弹簧量级：默认站姿下 300 N 弹簧在每根主动杆上等效约 13.6 N·m（`knee_gas_spring_compensation_torque_np`），
是 DM8009P 额定 20 N·m 的 68%。去掉后腿部电机要自己扛这部分负载，PD 稳态误差与力矩余量都会变。

## 部署注意

真机与 sim2x 的 MJCF 都有弹簧。M52 的 ONNX 导出 `knee_gas_spring = {force: 0, compensation_enabled: false}`，
sim2x 按契约不做补偿，等于把无弹簧训练的策略放到有弹簧的 plant 上。要让部署端也面对「无弹簧」plant，
需要在部署时打开电机侧前馈补偿（`-F·dL/dα`，300 N）——这是契约层面的改动，本条只做训练对照，部署前另议。

## 验证

- 内存 spec 核验（`.scratch/m52_check_spring.py`）：M51 编译出两个气弹簧 actuator，M52 为 0；
  M52 无 `knee_spring_force` 事件、无 critic 弹簧观测、动作项弹簧力 0；地形仍为 M51 的十列。
- `tests.test_rough_port` + `tests.test_onnx_metadata` 70 项通过（默认入口不变）。
- M52 入口 CPU smoke（1 env、5 轮）通过，导出 ONNX 的 `knee_gas_spring` 为 `{force: 0.0, compensation_enabled: false}`。

## 启动记录

（未启动）
