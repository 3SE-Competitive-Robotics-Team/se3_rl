---
name: remote-ckpt-play
description: 从远端训练机拉取最新 checkpoint 到本地，自动启动 se3-play Viser 可视化。
---

# Remote Ckpt Play

## 流程

1. SSH 到远端，找到最新 run 的最新 checkpoint
2. Rsync 到 `logs/remote_watch/<run_name>/`
3. 清理旧 Viser（port 8080）
4. 启动 `se3-play --viewer viser --num-envs 1`
5. 输出 `http://localhost:8080/`

## 脚本一键执行

```bash
bash scripts/sync_and_play.sh
```

## 远端配置

- 别名：`wuyingyun`（见 `remote-dev-se3/machines/wuyingyun.md`）
- 日志路径：`~/project/se3_rl/logs/rsl_rl/se3_wheel_leg/`
- 本地缓存：`logs/remote_watch/`
