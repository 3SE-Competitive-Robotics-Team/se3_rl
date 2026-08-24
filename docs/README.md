# 文档索引

本文档目录按“当前操作手册”和“设计记录”分开。运行命令只以当前操作手册为准；
`plan/` 中的路径、CLI 和阶段描述可能记录的是当时方案，不能覆盖现行 runtime 契约。

## 当前操作手册

- [新手入门](how_to_start.md)：安装、smoke、训练、ONNX 和第一次 sim2sim。
- [训练指南](train.md)：训练命令、训练产物和 Viser 值守。
- [训练任务架构](task_architecture.md)：task 目录边界和新增任务约定。
- [ONNX metadata 与 sim2x](onnx_runtime.md)：producer、artifact、runtime 和浏览器入口。
- [控制频率](control_frequency.md)：200 Hz physics / 50 Hz policy 的唯一来源和校验路径。
- [常见错误](common_mistakes.md)：已经确认且容易复发的实现错误。

## 背景与专题

- [机器人背景](background.md)
- [训练性能记录](perf.md)
- [环境分组训练](env_group_training.md)
- [EFGCL spotting](efgcl_spotting.md)
- [EFGCL 论文摘要](efgcl_paper_summary.md)

## 设计记录

`plan/` 保存方案推演和阶段性决策，不是当前命令手册。阅读时应先对照
[ONNX metadata 与 sim2x](onnx_runtime.md) 和实际代码；其中出现的旧包名或旧 CLI 只表示
当时实现背景。

`todo/` 保存尚未完成的工作，不能当作已经实现的能力。

## 当前 sim2sim 入口

```bash
./scripts/run_sim2x.sh
```

服务扫描 `logs/rsl_rl/<experiment>/<run_id>/onnx/*.onnx`。旧 `se3_sim2sim` 和
`se3_deploy` 不再提供当前操作文档；真机 adapter 尚未开始搭建。
