# 碰撞模型优化

## 背景

当前训练中已经通过 `SimulationCfg` 显式设置 `nconmax=128`、`njmax=512`，解决了 MJWarp 训练过程反复输出 `nefc overflow` 的问题。这个修复只扩大求解器工作区，不改变机器人本体的物理语义，风险较低。

机器人本体的碰撞模型仍然偏复杂。尤其是 base 使用多段 mesh collision，机器人摔倒、翻滚、随机 reset 或被 push 时，多个碰撞分片可能同时接触地面，导致 contact 和 constraint 数量出现峰值。

## 当前判断

- 默认训练 MJCF 已切换到 `assets/robots/serialleg/mjcf/serialleg_closed_chain_v3_train_obb_trim.xml`。
- 该版本只替换 collision geom，不改变闭链拓扑、气弹簧挂点、关节默认姿态和 actuator 语义。
- `drive_bar/coupler` 暂不参与 collision，避免闭链内部细杆成为额外接触面。
- 腿部原始 STL mesh collision 只保留为预览/诊断思路，不作为默认训练方案；MuJoCo mesh collision 按凸包处理，孔洞和凹槽不会成为精确接触边界。

## 优化方向

- 保留 visual mesh，仅替换 collision geom。
- 将 base collision 从单个粗 box 简化为 3 个保守 box，减少大盒子填满镂空区域造成的误接触。
- 将 thigh、calf 的 collision mesh 替换为每个主腿 link 一个 OBB 裁剪 box：先用原始视觉 STL 计算有向包围盒，再沿长轴裁掉膝关节和轮轴附近重叠体积。
- 检查并过滤不需要的 self-collision，仅保留训练确实需要的接触关系。
- 评估 `contype`、`conaffinity`、`condim`、`margin` 对 contact 数量和训练稳定性的影响。
- 如需调整 friction cone 或 contact 参数，必须同步观察轮子接地、站立高度、摔倒恢复和奖励曲线。

## 验证方式

- 使用 `uv run se3-joint-viewer --geom-view both` 对比 visual 与 collision 位置。
- 使用 MuJoCo 或 Rerun 对比接触点数量、接触位置和姿态变化。
- 跑 smoke 训练，确认环境不崩溃且无 overflow warning。
- 做短训对比，观察奖励曲线、站立稳定性、轮子接地行为和跌倒后的接触表现。
- 如果容量需求明显下降，再评估是否可以降低 `nconmax/njmax` 以节省显存和计算。

## 注意事项

碰撞模型会直接影响接触力、摩擦、摔倒恢复和 sim2sim gap。任何简化都需要逐步做、逐项验证，不应一次性大范围替换。
