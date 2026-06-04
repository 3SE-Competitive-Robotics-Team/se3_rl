# 电机过热模型方案

> 状态：待实现。

## 动机

sim 里没有热模型，策略在 2.5 m/s 指令下能稳定运行（tilt < 1°），但实机轮电机（M3508-C620-14:1）的连续输出力矩约为 2.21 N·m，长时间高速/大加速度工况仍可能触发 C620 限流保护或导致电机过热。训练出来的策略不会主动规避高热工况，sim2sim gap 的一部分就来自这里。

核心诉求：

1. 在训练端引入电机温度状态，策略能观测到热负荷并学会规避
2. sim2sim 端使用相同热模型验证迁移效果
3. 热模型参数集中在 `se3_shared`，训练/验证共享
4. 支持热时间常数域随机化（DR），提升策略对不同散热条件的鲁棒性

## 热模型

采用一阶集总热容模型（lumped thermal model），与电机数据手册标定方式一致：

```
dT/dt = (P_loss - (T - T_amb) / R_th) / C_th

P_loss = I² × R_phase     （铜损，主要热源）
I      = τ / (Kt × η)     （由输出转矩反推相电流）
```

参数说明：

| 符号 | 含义 | M3508-C620-14:1 参考值 |
|------|------|----------------------|
| `R_th` | 热阻（K/W） | 待标定，初始估计 ~3.0 K/W |
| `C_th` | 热容（J/K） | 待标定，初始估计 ~50 J/K |
| `T_amb` | 环境温度（K） | 固定 300 K（27°C） |
| `T_max` | 过热保护阈值 | ~358 K（85°C，C620 限流触发） |
| `τ_limit(T)` | 温度相关转矩上限 | 线性退化或硬截断，见下 |

热时间常数 `τ_th = R_th × C_th ≈ 150 s`，意味着电机在满载下约 2-3 分钟升温到保护阈值，与实测一致。

### 温度相关转矩上限

两种可选方案：

**方案 A：硬截断**（简单，策略学起来更容易）
```
τ_limit = rated_torque,          T < T_warn
τ_limit = 0,                     T ≥ T_max
```

**方案 B：线性退化**（更贴近 C620 实际限流行为）
```
τ_limit = rated_torque × max(0, (T_max - T) / (T_max - T_warn))
```

推荐先用方案 B，T_warn = 343 K（70°C）。

## 实现位置

### 训练端（`se3_train`）

在 `mdp/actions.py` 的 `SerialLegDelayedAction` 里维护每个环境每个关节的温度状态 `motor_temp`（shape `[num_envs, 6]`），每步更新：

```python
# 伪代码
delta_T = (P_loss - (motor_temp - T_amb) / R_th) / C_th * dt
motor_temp += delta_T
tau_limit = thermal_torque_limit(motor_temp)
effort = clip(effort, -tau_limit, tau_limit)
```

reset 时温度初始化为 `T_amb + Uniform(0, 20)` K（域随机化）。

### 观测扩展（`mdp/observations.py`）

将归一化温度作为特权观测加入 critic（不加入 actor），让 critic 更准确估值：

```
thermal_obs = (motor_temp - T_amb) / (T_max - T_amb)  # 归一化到 [0, 1]
```

共 6 维，加在现有 critic 特权观测后面。actor 不观测温度——策略通过感受到的转矩退化间接学会规避。

### sim2sim 端（`se3_sim2sim/robot.py`）

在 `WheelLeggedRobot` 里加 `motor_temp` 数组，`_compute_pd_torques` 里应用温度相关限流，`telemetry()` 暴露温度字段。

### 共享配置（`se3_shared`）

新增 `ThermalConfig` dataclass：

```python
@dataclass(frozen=True)
class ThermalConfig:
    r_th: float = 3.0       # K/W
    c_th: float = 50.0      # J/K
    t_amb: float = 300.0    # K
    t_warn: float = 343.0   # K（70°C，退化起点）
    t_max: float = 358.0    # K（85°C，完全限流）
    dr_init_range: tuple[float, float] = (0.0, 20.0)  # reset 初始温升范围 K
```

## 实现阶段

### Phase 1：热模型 + sim2sim 验证

- 在 `se3_shared` 加 `ThermalConfig`
- 在 `se3_sim2sim` 实现热状态更新和限流，`telemetry()` 暴露温度
- 用 2.5 m/s 持续跑 60 秒，观察温度曲线和限流触发时机
- 标定 `R_th` / `C_th`，使升温曲线与实测或 C620 手册匹配

### Phase 2：训练端接入

- 在 `SerialLegDelayedAction` 里加温度状态
- reset 时随机初始化温度
- 将归一化温度加入 critic 特权观测
- 跑 smoke 验证环境不崩

### Phase 3：奖励项（可选）

如果策略训练后仍然不主动规避高热工况，加一个软惩罚：

```python
thermal_penalty = sum(relu(motor_temp - T_warn) / (T_max - T_warn))
```

权重从小到大调，避免压制速度跟踪奖励。

## 参数标定说明

`R_th` 和 `C_th` 目前是估算值，需要实测标定：

1. 在静止状态施加额定电流，记录电机外壳温升曲线
2. 用一阶响应拟合得到 `τ_th = R_th × C_th` 和稳态温升 `ΔT_ss = P × R_th`
3. 分别解出 `R_th` 和 `C_th`

在有实测数据之前，训练中对 `R_th` 做 ±50% 的域随机化，让策略对不确定的散热条件保持鲁棒。
