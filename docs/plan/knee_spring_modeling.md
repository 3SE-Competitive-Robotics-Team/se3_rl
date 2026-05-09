# 膝关节弹簧力矩建模方案

> 状态：方案阶段，尚未实现。目标是在训练环境和 sim2sim 中统一建模弹簧补偿力矩，消除 sim-to-real 中弹簧力的 gap。

## 动机

SerialLeg 膝关节（L2、R2）安装有物理弹簧，用于补偿重力力矩、降低电机峰值扭矩。如果仿真中不建模这根弹簧，策略在训练时会学到一个"自己抗重力"的运动模式，迁移到实物后弹簧的额外力矩会导致膝关节过伸或震荡。

核心诉求：

1. 训练端每物理步对膝关节施加弹簧等效力矩
2. sim2sim 端使用相同数学模型验证
3. 弹簧参数集中管理在 `se3_shared`，训练/验证共享单一来源
4. 支持刚度域随机化（DR），提升策略鲁棒性

## 数学模型

### 几何定义

弹簧安装在膝关节两侧的铰接点之间，构成一个基于连杆几何的变力臂弹簧系统。建模在膝关节旋转平面（2D）内进行：

```
坐标系原点：大腿连杆的弹簧挂点 P₁
关节轴位置：P₀ = [l, 0]（膝关节旋转中心）
```

三个关键点：

| 符号 | 含义 | 计算 |
|------|------|------|
| P₀ | 膝关节旋转中心 | `[l, 0]` |
| P₁ | 弹簧在大腿侧的铰接点 | `[a·cos(α), a·sin(α)]` |
| P₂ | 弹簧在小腿侧的铰接点 | `[l - b·sin(θ+β), -b·cos(θ+β)]` |

其中 θ 为膝关节角度（joint position）。

### 弹簧力计算

弹簧为线性压缩弹簧，自然长度 s₀，实际工作长度 s 随关节角度变化：

```
弹簧向量: dp = P₂ - P₁
有效长度: s = ‖dp‖ - δ₀ - δ₁
弹簧力大小: |F| = k · (s₀ - s)
弹簧力向量: F = |F| · dp / ‖dp‖
```

δ₀、δ₁ 为弹簧两端铰接件的固定长度偏移（铰链/球头长度）。

### 等效力矩

对膝关节旋转中心 P₀ 求力矩：

```
力臂向量: r = P₂ - P₀
力矩: τ = -(r × F) = -(rₓ·F_y - r_y·Fₓ)
```

负号使得弹簧力矩方向与关节角度方向约定一致（弹簧伸展时产生正力矩抵抗重力屈膝）。

### 参数表

| 参数 | 符号 | 默认值 | 单位 | 说明 |
|------|------|--------|------|------|
| 大腿挂点距离 | a | 0.014 | m | P₁ 到坐标原点距离 |
| 小腿挂点距离 | b | 0.015 | m | P₂ 到关节轴距离 |
| 关节轴偏移 | l | 0.07 | m | P₀ 的 x 坐标 |
| 大腿挂点角 | α | 5° | rad | P₁ 方位角 |
| 小腿挂点角 | β | 35° | rad | P₂ 基准角偏移 |
| 弹簧刚度 | k | 900 | N/m | 线性刚度 |
| 弹簧自然长度 | s₀ | 0.06 | m | 无载荷自由长度 |
| 铰接偏移 1 | δ₀ | 0.004 | m | 大腿侧球头长度 |
| 铰接偏移 2 | δ₁ | 0.0095 | m | 小腿侧球头长度 |

### 域随机化

训练时对刚度 k 进行域随机化：

- 每次 reset 从 `[k_min, k_max]` 均匀采样（推荐 `[900, 980]`）
- 左右腿独立采样，shape `(num_envs, 2)`
- 可选 curriculum：前 N 步线性缩放力矩从 0→1，避免初始策略被大弹簧力干扰

## 实现计划

### Phase 1：共享配置

在 `se3_shared` 中添加弹簧参数 dataclass：

```python
@dataclass
class SpringConfig:
    """膝关节弹簧几何与力学参数"""
    enable: bool = True
    joints: list[str] = field(default_factory=lambda: ["lf1", "rf1"])

    # 几何参数
    a: float = 0.014        # 大腿挂点距离 (m)
    b: float = 0.015        # 小腿挂点距离 (m)
    l: float = 0.07         # 关节轴偏移 (m)
    alpha: float = 0.0873   # 大腿挂点角 (rad, ~5°)
    beta: float = 0.6109    # 小腿挂点角 (rad, ~35°)

    # 弹簧力学参数
    k: float = 900.0        # 标称刚度 (N/m)
    k_min: float = 900.0    # DR 下界
    k_max: float = 980.0    # DR 上界
    s0: float = 0.06        # 自然长度 (m)
    delta_0: float = 0.004  # 铰接偏移 1 (m)
    delta_1: float = 0.0095 # 铰接偏移 2 (m)

    # Curriculum
    curriculum_steps: int | None = 24000  # 力矩线性缩放步数，None=不使用
```

### Phase 2：训练端集成

在 `se3_train/mdp/` 中实现 `apply_knee_spring_torque` event：

1. 作为 `ManagerTermBase` 子类，`mode="before_simulation"`
2. `__init__` 中从 `SpringConfig` 读取参数，预计算三角函数
3. `__call__` 中批量计算所有 env 的弹簧力矩并 `set_joint_effort_target`
4. 支持 curriculum 线性缩放

关键实现要点：

- 使用 PyTorch 张量运算，shape `(num_envs, 2)` 对应左右膝
- reset 时重新采样 k（域随机化）
- 弹簧力矩作为 critic 特权观测输入

### Phase 3：sim2sim 集成

在 `se3_sim2sim` 中实现 `SpringTorqueCalculator`：

1. 纯 NumPy 单环境计算（sim2sim 为单机器人）
2. 从 `se3_shared.SpringConfig` 读取参数
3. 作为 pre-physics hook 每步调用 `mj_data.qfrc_applied[joint_id] += torque`

### Phase 4：观测扩展

将弹簧力矩作为 actor 或 critic 观测的可选项：

- critic 特权观测：直接使用力矩值（shape `(num_envs, 2)`）
- actor 观测（可选）：归一化后的力矩，帮助策略感知当前弹簧状态

是否加入 actor 观测需实验验证。保守起步只加 critic。

## 验证标准

1. **Smoke 测试通过**：加入弹簧后 `SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat --env.scene.num-envs 1 --gpu-ids None` 正常完成
2. **力矩曲线合理**：θ ∈ [-π/4, π/4] 范围内力矩单调、量级在 ±2 N·m 以内
3. **sim2sim 一致性**：训练端和 sim2sim 端在同一 θ 序列下力矩误差 < 1e-6
4. **训练收敛**：flat 任务 2000 iter 后 reward 不低于无弹簧 baseline 的 90%

## 风险与注意事项

- 弹簧参数来源于 CAD 设计图或实测。实际装配后可能有偏差，需要 system identification 校准
- β=35° 对应的是弹簧在膝关节完全伸直时的挂点角，需确认和 MJCF 中的零位定义一致
- 力矩方向符号取决于关节正方向约定，集成后第一件事是检查弹簧力矩是否抵抗重力（而非加剧屈膝）
- 域随机化范围不宜过大，否则策略难收敛。初期 k ∈ [900, 980] 较保守
