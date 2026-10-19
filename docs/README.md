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
- [环境分组训练](env_group_training.md)
- [EFGCL spotting](efgcl_spotting.md)（只用于旧跳跃线 Jump-PreTrain）
- [RMUC 2027 场地场景](rmuc2027_field.md)

## 设计记录

`plan/` 只保留仍对应现行功能的设计文档，不是当前命令手册，阅读时先对照
[ONNX metadata 与 sim2x](onnx_runtime.md) 和实际代码：

- [膝气弹簧建模](plan/knee_spring_modeling.md)
- [趴地移动任务（导轮 MJCF、Crawl / Crawl-Slope）](plan/crawl_guide_wheels_20261017.md)
- [电机被动参数](plan/motor_passive_params.md)
- [Rough 单轮耗时分析](plan/rough_iteration_time_20261003.md)
- [从 SolidWorks 重建 MJCF](plan/mjcf_rebuild_from_solidworks.md)

### 已删除的实验记录

2026-10-19 起，逐次实验记录（Rough M1–M54、跳跃 J1–J10、A15/A28、复旦对照、`rough_*` 对照、早期方案调研等
共 90 份）与 `perf.md`、`efgcl_paper_summary.md`、`todo/` 已从主线删除。代码注释里仍会以不带路径的名字
（如 `m37_reward_prune_20260927`）提到它们，需要原文时从删除前的 main 取：

```bash
git show e64516b8:docs/plan/m37_reward_prune_20260927.md
```

2026-10-19 才收录又随即删除的 7 份（`a15_*`、`kami_*`、`rough_dr1_oracle_*`、`rough_training_status_*`、
`fudan_stair_training_review_*`、`wheel_channel_fit_*`）不在 `e64516b8` 里，在 PR #77 的分支历史中：
`git fetch origin pull/77/head` 后对其中的 `52a4f968` 用同样的 `git show`。

## 当前 sim2sim 入口

```bash
./scripts/run_sim2x.sh
```

服务扫描 `logs/rsl_rl/<experiment>/<run_id>/onnx/*.onnx`。真机链路见 `submodules/se3-sim2x/docs/nx-link.md`。
