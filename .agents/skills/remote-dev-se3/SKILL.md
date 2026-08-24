---
name: remote-dev-se3
description: Use when managing se3_wheel_leg remote training, SSH or Kubernetes training targets, GPU selection, training launch, logs, checkpoint exchange, local Viser watch, or stopping remote jobs. The common skill contains workflow and routing only; connection details live in one explicitly selected machine profile.
---

# SE3 远端训练

## profile 路由

本 Skill 是公用入口，只保存流程、风险控制和路由规则。以下内容只能出现在
`machines/` 下对应的 machine profile 中：

- IP、端口、SSH 用户名、主机名和 SSH alias。
- Kubernetes namespace、pod、container 和集群入口。
- 远端仓库、虚拟环境、缓存、checkpoint 与本地中转目录。
- 机器所有者、GitHub 账号、密钥文件名和个人值守约定。

开始操作前，先从用户当前请求确定一个精确的 machine profile：

- 用户明确给出 profile 名或唯一机器别名时，只读取对应文件。
- 用户只说宽泛机器类型且可能对应多个 profile 时，先询问具体 profile，不猜测。
- 用户没有对应 profile 时，要求其提供连接参数；新参数应写入新的 machine profile，
  不能写回本 Skill 或其他公用文档。
- 不批量读取 `machines/`，也不从其他用户的 profile 借用默认值。

profile 是运行参数来源，不是凭据存储。密码、Token、私钥和临时验证码不得写入任何
已提交文件，只能使用本地凭据管理器、SSH agent 或未提交的环境配置。

## 通用原则

- **目标隔离**：共享集群中只操作用户明确指定的 namespace、pod 和进程。不得进入、
  `exec`、`kill`、`cp` 或读取其他用户的 pod。
- **显式目标**：远程脚本必须显式传入入口主机、namespace 和 pod；不要在通用脚本中
  设置个人机器默认值。
- **源码可追溯**：日常同步使用 git。开发机提交并 push，远端 checkout 同一 commit；
  临时 tar 同步只能用于明确接受不可复现风险的排障。
- **进程可追溯**：启动后记录 PID、日志和 run 目录。停止单个任务时只停止已确认的
  PID 或进程组，不默认执行全局 `pkill`。
- **先预检查**：启动前检查目标、checkpoint、虚拟环境、CUDA compat、GPU 占用和已有
  训练进程；启动后同时验证进程、GPU 利用率、iteration 和关键诊断指标。
- **产物不进 Git**：checkpoint 通过 Release asset 或用户指定的产物通道交换，不能提交
  `.pt` 文件到仓库历史。
- **复杂命令编码**：跨 PowerShell、SSH、kubectl 和 Bash 的复杂命令使用
  `scripts/remote_bash.ps1` 或 base64 模板，避免多层引号提前展开。

## 通用工作流

1. 确认目标 machine profile；需要个人资源时同时确认操作者有权使用该 profile。
2. 用只读命令确认入口主机、目标 namespace/pod、GPU 和远端 repo commit。
3. 本地完成修改、验证、提交和 push。
4. 远端拉取同一 commit，或使用明确选择的同步模式。
5. 执行 checkpoint、CUDA、GPU 与进程预检查。
6. 启动训练并记录 PID、日志、run 目录和实际命令。
7. 持续检查训练 iteration、关键指标、GPU 使用和错误日志。
8. 通过用户 profile 指定的产物通道同步 `onnx/model_*.onnx`，并在本机运行
   `./scripts/run_sim2x.sh` 完成 Viser/sim2sim 验收。

## 通用脚本约定

`scripts/remote_sync_start_training.py` 不提供个人目标默认值，至少显式传入：

```bash
uv run python scripts/remote_sync_start_training.py \
  --entry-host <host> \
  --namespace <namespace> \
  --pod <pod> \
  --remote-project <remote-project> \
  --cuda-compat-dir <cuda-compat-dir> \
  --cuda-toolkit-lib-dir <cuda-toolkit-lib-dir> \
  --task <task> \
  --iterations <iterations>
```

两跳 SSH 时同时传 `--inner-host`、用户、端口和 `--entry-temp-dir`。具体值只能来自已
通过身份门禁的 machine profile 或用户当前明确提供的信息。

`scripts/remote_bash.ps1` 同样要求显式传入口：

```powershell
.\scripts\remote_bash.ps1 `
  -HostAlias <entry-host> `
  -KubeNamespace <namespace> `
  -KubePod <pod> `
  -NoWorkdir `
  -ScriptPath .\tmp\check_training.sh
```

## 必查项

- CUDA compat/toolkit 路径必须由 profile 或显式参数给出，并由启动脚本检查。
- 无外网训练使用 `WANDB_MODE=offline`，避免日志上传阻塞训练或影响 checkpoint 保存。
- checkpoint 按数字版本排序，不能用字典序判断最新文件。
- 同步器必须先确认 ONNX 写入稳定，再原子替换本地文件，并保持
  `logs/rsl_rl/<experiment>/<run_id>/onnx/` 目录层级。
- Viser 验收同时检查策略、仿真模型、碰撞地形和实际 HTTP 可用性，不能只确认进程存在。
