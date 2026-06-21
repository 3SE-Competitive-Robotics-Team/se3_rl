# Jetson Orin NX 机器备忘录

> 通用操作流程见父目录 `SKILL.md`。本文件只记录 NX 真机部署目标的特有参数和坑。
> 当前 `se3_deploy` 和 `se3_sim2sim` 旧链路计划 deprecated，新开发建议迁移到 `https://github.com/3SE-Competitive-Robotics-Team/se3-mono`。Agent 在执行 NX 真机部署或调试命令前，必须先询问主人是否继续使用旧链路。

## 基本信息

| 项目 | 值 |
|---|---|
| SSH 别名 | `serialleg-nx` |
| 直连 IP | `192.168.137.100` |
| 用户名 | `amov` |
| 主机名 | `tegra-ubuntu` |
| 系统 | Linux `5.15.148-tegra` |
| 架构 | `aarch64` |
| 用途 | 真机 policy 部署与实时推理 |
| 仓库路径 | `~/project/se3_wheel_leg` |
| 认证 | 本机 `~/.ssh/id_ed25519.pub` 已写入 `~/.ssh/authorized_keys` |
| 连接验证 | 2026-06-01 |

NX 不是训练机；不要在 NX 上跑 MJLab 长训练。训练在 GPU 训练机完成，NX 只负责部署推理、真机 I/O 和安全状态机。

## 本机网卡配置

本机通过网线直连 NX 时，Windows 有线网卡需要在 `192.168.137.0/24`：

```powershell
New-NetIPAddress -InterfaceAlias '以太网' -IPAddress 192.168.137.1 -PrefixLength 24 -PolicyStore ActiveStore
```

该配置是临时配置，重启后会消失。若 SSH 超时，先检查：

```powershell
Get-NetIPAddress -InterfaceAlias '以太网' -AddressFamily IPv4
Test-NetConnection -ComputerName 192.168.137.100 -Port 22
```

## 本机 SSH 配置

建议写入本机 `~/.ssh/config`：

```sshconfig
Host serialleg-nx
    HostName 192.168.137.100
    User amov
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
```

验证：

```bash
ssh serialleg-nx "hostname && uname -m && whoami && hostname -I"
```

期望包含 `tegra-ubuntu`、`aarch64`、`amov`、`192.168.137.100`。

## 部署边界

- 源码同步走 git，checkpoint 和日志产物可以单独传输。
- 真机部署前必须先完成 sim2sim headless 验证。
- 真机 runtime 必须复用 `se3_shared` 的动作顺序、动作缩放、默认姿态和控制频率约定。
- 明文密码不写入仓库；首次引导后使用 SSH key。
- 任何 raw policy action 都必须经过限幅、限速和安全状态机后才能下发给电机。

详细部署计划见 `docs/nx_real_robot_deployment.md`。
