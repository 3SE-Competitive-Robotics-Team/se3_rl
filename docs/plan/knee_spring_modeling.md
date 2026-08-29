# 膝关节弹簧力矩建模方案

> 本文是设计记录，不是当前运行手册。命令和 runtime 路径以 [`../README.md`](../README.md) 为准。

> 状态（2026-08-26 复核）：落地实现与本文最初的设计方案**不同**，以下为当前真实状态。
>
> - 已实现：MJCF `l_knee_spring` / `r_knee_spring` spatial tendon + 300 N **恒力** actuator；
>   `se3_shared.fourbar` 解析前馈力矩；训练端 `SerialLegDelayedAction` 与 sim2sim
>   `SerialLegActuatorController` 使用同一算法、同一插入点（PD 之后、T-N 限幅之前）。
> - **未实现**：本文原方案的线性刚度弹簧（`k=900/4000 N/m`）、`xfrc_applied` 施力路径、
>   刚度域随机化、Phase 4 观测扩展。这些段落保留为设计历史，不代表代码现状。
> - **2026-08-26 修正**：前馈力矩符号原本与弹簧同号（等于把弹簧加了两遍），已翻转为真正
>   抵消，见下方「符号约定」。`7c9c155` 之后、本次修正之前导出的 ONNX 与 checkpoint 所处
>   的 plant 与现在不同，不能直接对比。
> - **2026-08-26 决定：前馈默认关闭**（`RobotConfig.knee_gas_spring_compensation_enabled
>   = False`）。弹簧仍在 MJCF 里生效，策略直接面对带弹簧的 plant，先验证它能否学会利用这份
>   抗重力力矩；见下方「为什么默认不做补偿」。

## 动机

SerialLeg 膝关节（lf1/rf1）安装有物理弹簧，用于补偿重力力矩、降低电机峰值扭矩。如果仿真中不建模这根弹簧，策略在训练时会学到一个"自己抗重力"的运动模式，迁移到实物后弹簧的额外力矩会导致膝关节过伸或震荡。

核心诉求：

1. 训练端每物理步施加弹簧等效广义力（对膝关节和髋关节都有贡献）
2. sim2sim 端使用相同数学模型验证
3. 弹簧参数集中管理在 `se3_shared`，训练/验证共享单一来源
4. 支持刚度域随机化（DR），提升策略鲁棒性

## 实际实现（以代码与 MJCF 为准）

### 拓扑：弹簧挂在大腿与小腿之间，不在驱动杆上

正式 MJCF 里 `l_spring_p1` 是 **`lf0_Link`（大腿）** 上的 site，`l_spring_p2` 是
**`lf1_Link`（小腿）** 上的 site。弹簧跨过膝轴连接大腿和小腿，**不涉及驱动杆**。因此下文
「弹簧安装位置」「广义力分配」两节里 P₁ 在驱动杆上、需要经传动比 `n(θ)` 折算的分析
**不是当前模型**，仅保留为早期设计推演。

实测校验：把 `l_spring_p2` 的 MJCF 坐标减去膝轴坐标，与 `se3_shared/fourbar.py` 的
`_SPRING_P2_FROM_KNEE_{X,Z}` 逐位一致；`_KNEE_{X,Z}` 也与 MJCF 的膝轴位置逐位一致。

### MJCF 侧：300 N 恒力 actuator

```xml
<spatial name="l_knee_spring" width="0.003" rgba="0.2 0.6 1 1">
  <site site="l_spring_p1" />
  <site site="l_spring_p2" />
</spatial>
<general name="l_knee_gas_spring" tendon="l_knee_spring" gaintype="fixed" gainprm="0"
         biastype="affine" biasprm="300 0 0" forcelimited="true" forcerange="0 300" />
```

`gainprm=0` + `biasprm[0]=300` 使 actuator 力恒为 300 N，与 tendon 长度、速度、ctrl 均无关，
所以它是**恒力气弹簧**而不是线性弹簧。MuJoCo 的 actuator 约定为
`qfrc_actuator = +force · ∂L/∂q`（已用 `qfrc_actuator[lf1] / (300·∂L/∂q_lf1) = 1.000000` 实测确认），
正力**推长** tendon。

弹簧长度与等效力矩随主动杆夹角 α 的实测变化（`policy_pos=(0, -α, 0, α)`）：

| α (rad) | 0.000 | 0.377 | 0.755 | 1.132 | 1.510 |
|---------|-------|-------|-------|-------|-------|
| tendon 长度 L (m) | 0.2273 | 0.2051 | 0.1818 | 0.1601 | 0.1435 |
| 弹簧广义力矩 τ_α,spring (N·m) | -16.89 | -18.36 | -18.29 | -15.71 | -10.34 |

前馈力矩即 `-τ_α,spring`，同一位姿下为 `+16.89 … +10.34 N·m`。α 增大对应下蹲，
`τ_α,spring` 全程为负说明弹簧一直在推着腿伸直，即抬升机身。

全行程 ΔL = 0.0838 m，单腿 300 N 做功 25.2 J（双腿 50.3 J）。`|τ_α|` 落在
10.3–18.6 N·m，而 DM8009P 连续额定只有 20 N·m —— **前馈项本身就占用 52%–93% 的额定
力矩包络**，且每个 physics tick 都在施加。任何关于力矩余量的判断都必须把这一项算进去。

### 代码路径

| 层 | 位置 | 职责 |
|----|------|------|
| 共享数学 | `se3_shared/fourbar.py::knee_gas_spring_compensation_torque_{torch,np}` | `τ = F · (dL/dθ) · (dθ/dα)`，输出 policy 序 `[c, -c, -c, c]` |
| 训练端 | `se3_train/mdp/actions.py::SerialLegDelayedAction.apply_actions` | 用带 encoder bias 的测量角计算，写 `set_joint_effort_target`，mjlab 在 PD 之后叠加、再过 T-N 限幅 |
| sim2sim | `se3_runtime/_serialleg_v1.py::knee_gas_spring_compensation_torque` | 纯 NumPy 同算法复刻（`se3_runtime` 不得反向依赖 `se3_shared`） |
| sim2sim | `se3_runtime/policy_actuator.py::SerialLegActuatorController.compute` | 在 PD 之后、`np.clip` T-N 限幅之前叠加，与训练端插入点一致 |
| 契约 | `robot.knee_gas_spring = {force, compensation_enabled}` | 由 `se3_train/onnx_metadata.py` 导出，`policy_descriptor` → `policy_contract` 解析 |

两端 cadence 也一致：mjlab 的 `apply_action()` 每个 physics substep 执行一次，sim2sim 的
`apply_decoded_action()` 同样每个 physics tick 重算，因此前馈不是 policy 频率的零阶保持。

**旧 artifact 兼容**：metadata 里没有 `robot.knee_gas_spring` 的 artifact 按
`compensation_enabled=false` 解释，即保持接入前的 sim2sim 行为。要让老策略也走前馈必须重新导出 ONNX。

### 为什么默认不做补偿

弹簧的广义力矩在名义站立姿态就已经是这个量级（`height_conditioned_policy_default` 求出零点
姿态后代入，单腿）：

| 指令高度 (m) | 0.20 | 0.22 | 0.26 | 0.30 | 0.32 |
|--------------|------|------|------|------|------|
| α (rad) | 1.461 | 1.312 | 1.029 | 0.744 | 0.595 |
| \|τ_α,spring\| (N·m) | 11.16 | 13.46 | 16.70 | 18.33 | 18.58 |
| 占 DM8009P 额定 20 N·m | 55.8% | 67.3% | 83.5% | 91.6% | 92.9% |

方向上 `τ_α,spring` 全程推动腿伸直，正是对抗重力，这也是真机装这根弹簧的目的。开启前馈等于
让电机把这份力矩原样抵消掉：plant 回到无弹簧状态，但电机每个 tick 都要额外掏 11-18 N·m，
可用 PD 余量被压缩到额定的 7%-44%，包络打满时还抵消不干净。也就是说**开前馈比不装弹簧更差**
——同样的 plant，却多付一份力矩。

因此默认 `knee_gas_spring_compensation_enabled = False`：弹簧在 MJCF 里照常生效，策略直接
学带弹簧的 plant。前馈实现与契约字段全部保留，改成 `True` 即可复现「电机抵消弹簧」那一版。

### 符号约定

前馈取 `τ = -F · (dL/dθ) · (dθ/dα)`，即弹簧广义力矩的相反数，作用是让策略面对的 plant 回到
无弹簧状态，同时让 T-N 包络如实反映真机电机为抵消弹簧付出的力矩。

`7c9c155` 引入时符号写反了（与弹簧同号，等于把弹簧加了两遍），2026-08-26 已翻转。判定与
复核依据：

- MuJoCo 的 actuator 约定为 `qfrc_actuator = +force·∂L/∂q`，实测
  `qfrc_actuator[lf1] / (300·∂L/∂q_lf1) = 1.000000`，即正力推长 tendon；
- 约束一致的 reduced 坐标下弹簧广义力矩为 `τ_α,spring = +F·dL/dα`；
- 修正后 `comp[0] / (F·dL/dα) = -0.999999`，与弹簧严格反号。

闭环 sim2sim 对照（零动作策略、250 policy step、机身高度）：

| 配置 | 修正前 | 修正后 |
|------|--------|--------|
| 无弹簧 MJCF | 0.1849 m | 0.1849 m |
| 弹簧、无前馈 | 0.2537 m | 0.2537 m |
| 弹簧 + 前馈 | 0.2802 m | 0.2011 m |

修正后「弹簧 + 前馈」回到接近无弹簧 plant（高度差 16 mm、关节角差 0.036 rad），残差来自
T-N 限幅：该位姿下 PD + 前馈已经打满包络，电机没有余量把弹簧完全抵消掉。这是真机同样存在的
物理限制，不是实现误差。

## 实际传动机构（早期设计推演，非当前模型）

### 整体布局

所有电机安装在机身内部，通过共轴链轮传动将运动传递到腿部：

```
┌─ 机身 (base_link) ──────────────────────────────────────────┐
│                                                              │
│   [膝关节电机] ──链轮── (共轴) ──链轮── [髋关节电机]           │
│        │                                    │                │
└────────┼────────────────────────────────────┼────────────────┘
         │ 膝关节驱动杆                        │ (链轮驱动)
         │ (从机身穿出)                        │
         │                                    │
         B ─── 连杆 ─── C                     A ← 髋关节轴 (共轴)
                         │                    │
                         D ══ 膝关节轴 ════════╝
                         │
                       [小腿]
                         │
                       [轮子]
```

### 四连杆机构详解

膝关节的驱动是一个**平面四杆机构**，四个铰接点：

| 铰接点 | 位置 | 角色 |
|--------|------|------|
| A | 髋关节轴（大腿与机身的转点） | 固定铰（frame 上的） |
| B | 膝关节驱动杆的末端 | 驱动铰（电机通过链轮驱动） |
| C | 连杆与小腿的连接点 | 从动铰 |
| D | 膝关节轴（大腿与小腿的转点） | 输出铰 |

四根杆件：
- **大腿 (AD)**：frame 杆，连接髋轴 A 和膝轴 D
- **驱动杆 (AB)**：从髋轴 A 出发，由膝关节电机经链轮驱动旋转
- **连杆 (BC)**：连接驱动杆末端 B 和从动点 C
- **小腿上段 (CD)**：从膝轴 D 到连杆铰接点 C（小腿的一部分）

### 弹簧安装位置

弹簧安装在**四连杆铰接点的对侧延伸处**：

```
              B'（弹簧挂点 P₁）
              │
    A ════ B ─── 连杆 ─── C
    ║                      │
    ║      大腿            D ═══ 膝关节轴
    ║                      │
    ╚══════════════════════╝
                           │
                           D'（弹簧挂点 P₂）
                           │
                         [小腿]

    弹簧: P₁ ~~~弹簧~~~ P₂
    P₁ 在驱动杆 AB 的 B 侧对面延伸
    P₂ 在小腿 CD 的 D 侧对面延伸
```

关键特征：
- **P₁** 在膝关节驱动杆上，B 点的对侧延伸（与 A 方向相反）
- **P₂** 在小腿上，D 点（膝关节轴）的对侧延伸（远离连杆方向）
- 弹簧跨越了膝关节轴 D，连接的是**驱动杆**和**小腿**

## 力学分析：弹簧力的广义力分配

### 广义力分配总结

根据实际拓扑（驱动杆与大腿共轴但独立旋转），弹簧力的广义力分配如下：

| 力的来源 | 膝关节 (lf1) | 髋关节 (lf0) | 膝电机（链轮） |
|---------|-------------|-------------|--------------|
| +F 作用在 P₁（驱动杆上） | ✗ 电机经链轮吸收 | ✗ 轴承力过轴心 A，力臂=0 | ✓ 直接承受 |
| -F 作用在 P₂（小腿上） | ✓ 直接力矩 (P₂-D)×F | ✓ 间接经 D→大腿→A 传递 | ✗ |

关键物理推理：
- P₁ 在驱动杆上，驱动杆绕 A 旋转但与大腿**不刚性连接**（仅共轴轴承）
- 轴承在 A 点只传递径向力，不传递力矩 → 力臂为零 → 对大腿（髋关节 DOF）力矩为零
- 驱动杆上的弹簧力矩完全由膝电机通过链轮吸收
- P₂ 在小腿上，小腿通过膝轴 D 连接大腿 → 反力在 D 处传入大腿 → 对髋轴 A 有力臂 → 产生髋关节力矩

### 非平行四连杆传动比映射

由于膝关节传动是**非平行四连杆**，驱动杆角度 φ 和膝关节输出角度 θ 之间存在非线性映射关系：

```
φ = f(θ)     （四连杆正运动学）
dφ/dθ = n(θ) （瞬时传动比，随 θ 变化）
```

#### 四连杆闭环位置方程

给定四杆长度 L_AB, L_BC, L_CD, L_AD（大腿长），四连杆的闭环约束为：

```
A + L_AB·[cos(φ), sin(φ)] + L_BC·[cos(γ), sin(γ)] = D + L_CD·[cos(ψ), sin(ψ)]

其中:
  φ = 驱动杆角（输入，相对大腿方向）
  ψ = 小腿上段角（输出，相对大腿方向）
  γ = 连杆角（从 B→C 方向）
  θ = 膝关节角 = ψ 的函数（取决于 CD 和小腿的几何关系）
```

消去 γ 后得到 φ 和 ψ(θ) 的隐式关系，即 φ = f(θ)。

#### 瞬时传动比 n(θ)

对闭环方程求导：

```
n(θ) = dφ/dθ
```

这个传动比决定了：
1. **弹簧力矩映射**：P₁ 侧弹簧对驱动杆的力矩 τ_drive，等效到膝关节输出端为 τ_knee_P1 = τ_drive / n(θ)
2. **电机力矩映射**：电机输出力矩 τ_motor 到膝关节为 τ_knee = τ_motor / n(θ)
3. **速度映射**：膝关节角速度 ω_knee 到驱动杆为 ω_drive = n(θ) · ω_knee

#### 弹簧等效膝关节力矩（P₁ 侧）

```
τ_drive_from_P1 = (P₁ - A) × (+F)     # 弹簧对驱动杆的力矩
τ_knee_equiv_P1 = τ_drive_from_P1 / n(θ)  # 经四连杆映射到膝关节输出
```

#### 弹簧等效膝关节力矩（P₂ 侧，直接）

```
τ_knee_from_P2 = (P₂ - D) × (-F)      # 弹簧对小腿的力矩，直接作用于膝关节
```

#### 总膝关节弹簧力矩

```
τ_knee_total = τ_knee_equiv_P1 + τ_knee_from_P2
             = (P₁ - A) × F / n(θ) + (P₂ - D) × (-F)
```

#### 髋关节弹簧力矩（仅 P₂ 侧贡献）

P₂ 处的弹簧力 -F 作用在小腿上，小腿在 D 处对大腿产生反力。该反力对髋轴 A 的力矩：

```
F_reaction_at_D = 小腿对大腿在 D 处的约束力（由牛顿第三定律）
τ_hip = (D - A) × F_reaction_at_D
```

在仿真中用 `xfrc_applied` 对小腿 body 施加 -F，MuJoCo 自动通过 Jᵀ 计算这个间接贡献。

## 数学模型

### 几何定义

弹簧安装在驱动杆和小腿的对侧延伸处，构成一个基于四连杆几何的变力臂弹簧系统。建模在膝关节旋转平面（2D）内进行：

```
坐标系原点：髋关节轴 A（驱动杆与大腿的共轴转点）
膝关节轴：D = [L_thigh, 0]（大腿长度方向）
```

关键点（在大腿局部坐标系中）：

| 符号 | 含义 | 计算 |
|------|------|------|
| A | 髋关节轴 / 驱动杆旋转中心 | `[0, 0]` |
| D | 膝关节轴 | `[L_thigh, 0]` |
| P₁ | 弹簧在驱动杆上的挂点（B 的对侧） | `[-a·cos(φ+α), -a·sin(φ+α)]`（φ 为驱动杆角） |
| P₂ | 弹簧在小腿上的挂点（D 的对侧） | `D + [b·sin(θ+β), b·cos(θ+β)]`（θ 为膝关节角） |

其中：
- φ 为驱动杆相对大腿的角度（通过四连杆运动学，φ = f(θ)）
- θ 为膝关节角度（lf1_Joint position）

### 弹簧力计算

与之前相同：

```
弹簧向量: dp = P₂ - P₁
有效长度: s = ‖dp‖ - δ₀ - δ₁
弹簧力大小: |F| = k · (s₀ - s)
弹簧力向量: F = |F| · dp / ‖dp‖
```

### 等效广义力（正确做法）

使用虚功原理，弹簧力对各关节的广义力为：

```
τ_hip  = (∂P₁/∂q_hip)ᵀ · F - (∂P₂/∂q_hip)ᵀ · F
τ_knee = (∂P₁/∂q_knee)ᵀ · F - (∂P₂/∂q_knee)ᵀ · F
```

由于简化 MJCF 模型没有显式四连杆，实现时用 `xfrc_applied` 对 body 施力让 MuJoCo 自动处理。

### 参数表（设计稿，未落地）

> 下表是线性弹簧方案的占位参数，**代码里不存在这些常量**。真实几何见
> `se3_shared/fourbar.py` 的 `_SPRING_P1_{X,Z}` / `_SPRING_P2_FROM_KNEE_{X,Z}`，
> 真实力值见 `se3_shared/robot.py::RobotConfig.knee_gas_spring_force = 300.0`（恒力，无刚度）。

| 参数 | 符号 | 当前值 | 单位 | 说明 |
|------|------|--------|------|------|
| 驱动杆长 | L_AB | 0.060 | m | 四连杆曲柄（A→B），占位值需 CAD 确认 |
| 连杆长 | L_BC | 0.181 | m | 四连杆连杆（B→C）= 大腿长（平行四连杆） |
| 小腿上段长 | L_CD | 0.060 | m | 四连杆摇杆（D→C），占位值需 CAD 确认 |
| 大腿长 | L_AD | 0.181 | m | 四连杆机架（MJCF 实测） |
| 驱动杆挂点距离 | a | 0.040 | m | P₁ 到髋轴(A)距离 |
| 小腿挂点距离 | b | 0.040 | m | P₂ 到膝轴(D)距离 |
| 驱动杆挂点角 | α | 5° | deg | P₁ 相对驱动杆朝下方向的偏角 |
| 小腿挂点角 | β | 35° | deg | P₂ 相对小腿朝下方向的偏角 |
| 弹簧刚度 | k | 4000 | N/m | 线性刚度 |
| 弹簧自然长度 | s₀ | 0.200 | m | 无载荷自由长度 |
| 铰接偏移 1 | δ₀ | 0.004 | m | 驱动杆侧球头长度 |
| 铰接偏移 2 | δ₁ | 0.0095 | m | 小腿侧球头长度 |

### 域随机化（未实现）

线性弹簧方案曾计划对刚度 k 做域随机化：

- 每次 reset 从 `[k_min, k_max]` 均匀采样（推荐 `[900, 980]`）
- 左右腿独立采样，shape `(num_envs, 2)`
- 可选 curriculum：前 N 步线性缩放力矩从 0→1，避免初始策略被大弹簧力干扰

恒力方案下对应的随机化对象应是力值 F 而不是刚度 k，当前**没有任何气弹簧域随机化**：
MJCF 的 300 N 与 `RobotConfig.knee_gas_spring_force` 都是固定值。

## 实现计划（原设计，未按此执行）

### Phase 1：共享配置

在 `se3_shared` 中添加弹簧参数 dataclass：

```python
@dataclass
class SpringConfig:
    """膝关节弹簧几何与力学参数"""
    enable: bool = True

    # 作用关节
    knee_joints: list[str] = field(default_factory=lambda: ["lf1_Joint", "rf1_Joint"])
    hip_joints: list[str] = field(default_factory=lambda: ["lf0_Joint", "rf0_Joint"])

    # 四连杆杆长 (m)
    L_AB: float = 0.06      # 驱动杆长（待 CAD 确认）
    L_BC: float = 0.08      # 连杆长（待 CAD 确认）
    L_CD: float = 0.04      # 小腿上段长（待 CAD 确认）
    L_AD: float = 0.18      # frame 杆 = 大腿长（MJCF 已知）

    # 弹簧挂点几何参数
    a: float = 0.014        # P₁ 距髋轴 A 距离 (m)
    b: float = 0.015        # P₂ 距膝轴 D 距离 (m)
    alpha: float = 0.0873   # P₁ 相对驱动杆偏角 (rad, ~5°)
    beta: float = 0.6109    # P₂ 相对小腿偏角 (rad, ~35°)

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

### Phase 2：训练端集成（混合施力方案）

在 `se3_train/mdp/` 中实现 `apply_spring_force` event：

1. 作为 `ManagerTermBase` 子类，`mode="before_simulation"`
2. 每步根据膝关节角度 θ：
   - 通过四连杆正运动学求解驱动杆角度 φ = f(θ) 和传动比 n(θ)
   - 计算 P₁、P₂ 的位置和弹簧力向量 F
3. **P₁ 侧**：计算驱动杆上的力矩，经传动比映射后用 `qfrc_applied[knee]` 施加
4. **P₂ 侧**：对小腿 body 施加 -F（用 `xfrc_applied`），MuJoCo 自动分配到膝关节+髋关节
5. 支持 curriculum 线性缩放

关键实现要点：

- 使用 PyTorch 张量运算，shape `(num_envs, 2)` 对应左右腿
- 四连杆闭环方程需要数值求解（或预计算查表）
- reset 时重新采样 k（域随机化）
- 弹簧力矩作为 critic 特权观测输入

### Phase 3：sim2x 集成

在 `se3-sim2x` 的平台 adapter 中实现 `SpringForceCalculator`：

1. 纯 NumPy 单环境计算（sim2sim 为单机器人）
2. 从 `se3_shared.SpringConfig` 读取参数
3. 每步：
   - P₁ 侧力矩经传动比 → `mj_data.qfrc_applied[knee_joint_id]`
   - P₂ 侧力 → `mj_data.xfrc_applied[shank_body_id]`
4. MuJoCo 引擎自动处理 P₂ 侧的 Jᵀ 映射（含髋关节间接贡献）

### Phase 4：观测扩展

将弹簧力矩作为 actor 或 critic 观测的可选项：

- critic 特权观测：直接使用力矩值（shape `(num_envs, 2)`）
- actor 观测（可选）：归一化后的力矩，帮助策略感知当前弹簧状态

是否加入 actor 观测需实验验证。保守起步只加 critic。

## 验证标准

1. **Smoke 测试通过**：加入弹簧后 `SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat-GRU --env.scene.num-envs 1 --gpu-ids None` 正常完成 —— 未做。
2. **力矩曲线合理**：原判据「±2 N·m 以内」是线性弹簧方案的量级，恒力方案实测为
   10.3–18.6 N·m，判据本身已作废，应改为按额定力矩占比评估。
3. **sim2sim 一致性**：训练端和 sim2sim 端在同一 θ 序列下力矩误差 < 1e-6 —— **已通过**。
   4096 组随机 policy 位姿下 `se3_runtime` 与 `se3_shared` NumPy 实现的最大偏差为 `0.0`
   （逐位相同），与 torch float64 的偏差为 `1.6e-11`；训练实际用的 float32 张量偏差为
   `6.6e-3 N·m`，属于 dtype 精度而非算法差异。回归用例见
   `tests/test_runtime_mujoco.py::KneeGasSpringCompensationTests`。
4. **训练收敛**：flat 任务 2000 iter 后 reward 不低于无弹簧 baseline 的 90% —— 未做，
   目前没有任何带气弹簧的训练 run 记录。

## 风险与注意事项

当前有效：

- **力矩包络占用**：前馈项 10.3–18.6 N·m 对 DM8009P 额定 20 N·m 是 52%–93%，会显著压缩
  PD 可用余量，T-N 越界统计（`TnTrackedDcMotorActuator`）的读数含义随之改变。包络打满时
  弹簧无法被完全抵消，plant 会偏向「带弹簧」一侧。
- **符号翻转前后的 artifact 不可混比**：`7c9c155` 之后、2026-08-26 修正之前的导出物所处
  plant 与现在不同。
- **两端符号必须同步**：`se3_shared/fourbar.py` 与 `se3_runtime/_serialleg_v1.py` 是两份独立
  实现，只改一边会直接产生 sim2sim gap。回归用例
  `tests/test_runtime_mujoco.py::KneeGasSpringCompensationTests` 会挡住这种情况。
- **弹簧参数来源于 CAD 设计图或实测**，实际装配后可能有偏差，需要 system identification 校准。
- **没有域随机化**：F 固定 300 N，真机装配差异不在训练分布内。
- **旧 ONNX 不带 `robot.knee_gas_spring`**，sim2sim 会按无前馈解释；混跑新旧 artifact 时要注意
  这不是同一个 plant。

已随实现方案作废（保留说明来源）：

- β=35° 挂点角判据 —— 恒力方案不使用 α/β 挂点角参数。
- k ∈ [900, 980] 域随机化范围 —— 没有刚度这个量。
- 四连杆传动比 φ=f(θ) 需从 CAD 导出 —— 已由 `fourbar.py` 的 LUT 从 MJCF 几何解析求出。
- xfrc_applied 与简化 MJCF 的 Jacobian 分配 —— 现在弹簧由 MJCF tendon actuator 直接产生，
  广义力由 MuJoCo 在真实闭链上求解，不存在这条路径。
