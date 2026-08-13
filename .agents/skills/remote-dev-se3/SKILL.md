---
name: remote-dev-se3
description: Use when managing se3_wheel_leg remote training, SSH or Kubernetes training targets, GPU selection, training launch, logs, checkpoint exchange, local Viser watch, or stopping remote jobs. Personal machine profiles must pass the identity gate before they are read or used.
---

# SE3 远端训练

## 身份与 profile 门禁

本 Skill 是通用入口，不保存个人主机地址、pod 名或本地绝对路径。机器细节放在
`machines/` profile 中，读取前必须先确定当前操作者与目标 profile。

`a800-xyh-am345` 和 `docs/laptop_viser_play.md` 只属于 `xuyihao/xyh`（GitHub
账号 `am345`）的个人开发环境。只有满足以下任一条件时才能读取、引用或使用其中的
主机、IP、namespace、pod 和路径：

- 用户在当前对话中明确说明自己是 `xuyihao/xyh`，并要求使用该 profile。
- 当前已认证 GitHub 账号通过 `gh api user --jq .login` 确认为 `am345`。

不得根据仓库作者、commit、分支名、工作区路径或机器上残留的 SSH alias 推断身份。
身份不匹配或无法确认时：

- 不读取 `machines/a800-xyh-am345.md` 和 `docs/laptop_viser_play.md`。
- 不使用从历史命令中看到的个人地址、pod、路径或 checkpoint 中转仓库。
- 使用用户自己的 machine profile；若不存在，则要求用户提供目标参数，并保持脚本参数显式。

这是一条 Agent 路由规则，不是仓库访问控制。个人 profile 中不得存放密码、Token、
私钥或其他凭据；真正敏感的信息必须留在未提交的本地配置或凭据管理器中。

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

## profile 路由

- 用户明确指定 `wuyingyun`：读取 `machines/wuyingyun.md`。
- 用户明确指定 `gpufree`：读取 `machines/gpufree.md`。
- NX / Jetson 真机部署：读取 `machines/nx.md`，并先按仓库约定确认是否继续旧链路。
- `xuyihao/xyh` 或 GitHub `am345` 使用 A800：身份门禁通过后读取
  `machines/a800-xyh-am345.md`；需要 laptop Viser fallback 时再读取
  `docs/laptop_viser_play.md`。
- 其他用户：只读取属于该用户的 profile。没有 profile 时不得借用
  `a800-xyh-am345`，应先收集显式连接参数。

只读取当前任务所需的一个 machine profile，不要批量加载 `machines/`。

## 通用工作流

1. 确认操作者身份和目标 machine profile。
2. 用只读命令确认入口主机、目标 namespace/pod、GPU 和远端 repo commit。
3. 本地完成修改、验证、提交和 push。
4. 远端拉取同一 commit，或使用明确选择的同步模式。
5. 执行 checkpoint、CUDA、GPU 与进程预检查。
6. 启动训练并记录 PID、日志、run 目录和实际命令。
7. 持续检查训练 iteration、关键指标、GPU 使用和错误日志。
8. 通过用户 profile 指定的产物通道同步 checkpoint，并在本机完成 Viser/sim2sim 验收。

## 通用脚本约定

`scripts/remote_sync_start_training.py` 不提供个人目标默认值，至少显式传入：

```bash
uv run python scripts/remote_sync_start_training.py \
  --entry-host <host> \
  --namespace <namespace> \
  --pod <pod> \
  --task <task> \
  --iterations <iterations>
```

两跳 SSH 时同时传 `--inner-host`、用户和端口。具体值只能来自已通过身份门禁的
machine profile 或用户当前明确提供的信息。

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
- watcher 必须先确认 checkpoint 写入稳定，再原子替换本地文件。
- Viser 验收同时检查策略、仿真模型、碰撞地形和实际 HTTP 可用性，不能只确认进程存在。
