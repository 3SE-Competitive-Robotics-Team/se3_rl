# A100 runtime upgrade 历史验证机

> 通用操作流程见父目录 `SKILL.md`。本 profile 只保存 2026-06-02 runtime upgrade
> 验证所用机器的历史路由参数；不要把它当作当前默认训练目标，也不得在此保存凭据。

## 历史入口

| 项目 | 值 |
|---|---|
| SSH 用户 | `root` |
| SSH 地址 | `120.209.70.195` |
| SSH 端口 | `30369` |
| GPU | `A100-SXM4-80GB * 2` |
| 验证用 GPU | `CUDA_VISIBLE_DEVICES=0` |

该入口是否仍有效必须重新做只读连通性检查；历史 plan 只记录验证结论，不复制这些
机器参数。
