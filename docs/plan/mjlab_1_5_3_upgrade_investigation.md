> 固化自 `.tmp/full_perf_pipeline/version_upgrade_investigation.md`（2026-08-29），为 MJLab 1.5.3 迁移分支的输入。

# 版本升级兼容性调查

调查日期：2026-08-29  
固定项目 commit：`190fc4c44818a675440fca4936308c76cb8c52ea`

## 当前与候选版本

| 组合 | MJLab | MuJoCo Warp | MuJoCo | Warp | 状态 |
|---|---:|---:|---:|---:|---|
| 当前 A | 1.4.0 | 3.8.1 | 3.8.0 | 1.12.1 | 项目锁定版本 |
| 最初 B | 1.4.0 | 3.12.0 | 3.12.0 | 1.16.0 | 官方未支持的混装；8192 env 崩溃 |
| 正式 B | 1.5.3 | 3.10.0.3 | 3.10.0 | 1.14.0 | MJLab 官方支持的 simulator 组合；ABBA 已完成 |

## 官方依据

- [MJLab v1.4.0 `pyproject.toml`](https://github.com/mujocolab/mjlab/blob/v1.4.0/pyproject.toml) 约束 MuJoCo/MJWarp `~=3.8.0`。
- [MJLab v1.5.0 release](https://github.com/mujocolab/mjlab/releases/tag/v1.5.0) 明确整套迁移到 3.10，并说明上游删除 parallel linesearch，`SimulationCfg.ls_parallel` 从此弃用且忽略。
- [MJLab v1.6.0 release](https://github.com/mujocolab/mjlab/releases/tag/v1.6.0) 明确整套迁移到 3.11；截至调查日，[MJLab main 的依赖](https://github.com/mujocolab/mjlab/blob/main/pyproject.toml) 仍是 `mujoco-warp~=3.11.0`，没有正式支持 3.12。
- [MJLab PR #1091](https://github.com/mujocolab/mjlab/pull/1091) 修复新版 MJWarp 将共享 model field 表示成带 `_is_batched` 标记的真实 size-1 数组后，`TorchArray` 未扩展 world 维的问题；官方复现症状正是多 env 索引越界。
- [MJLab issue #1108](https://github.com/mujocolab/mjlab/issues/1108) 记录 RTX 4090、`num_envs>=128`、startup mass randomization 下 MJWarp 3.10.0.2 的 CUDA 700；官方最终要求升级到 MJLab 1.5.3，其中将 MJWarp 下限提高到 3.10.0.3。
- [MuJoCo Warp v3.12.0](https://github.com/google-deepmind/mujoco_warp/releases/tag/v3.12.0) 是可用最新版，但其存在不等于 MJLab 1.4 与之兼容。

## 两次失败与同步定位

- job 40、41：MJLab 1.4 + MJWarp/MuJoCo 3.12 + Warp 1.16，均在 startup 阶段触发 PyTorch CUDA `IndexKernel` 越界；异步栈表面落在 `randomize_restitution`。
- job 43：同配置增加 `CUDA_LAUNCH_BLOCKING=1`。首个真实失败点定位到：

  ```text
  randomize_friction
  env.sim.model.geom_friction[env_ids, :, 0] = friction
  mjlab/sim/sim_data.py -> self._tensor[idx] = value
  CUDA IndexKernel: index out of bounds
  ```

这与 PR #1091 的 world-shared size-1 field 未按 `nworld` 扩展完全一致，证明它是桥接层版本不兼容，不是 GPU 算力、接触容量或 solver 容量问题。

## 评估过的处理路径

1. 保持 1.4/3.8：兼容风险最低，作为 A 基线。
2. 给 1.4 回移 PR #1091，并继续强行运行 3.12：可用于诊断上限，但仍超出官方支持矩阵，不作为正式 ABBA。
3. 整套迁移到 1.5.3/3.10.0.3：最小的官方支持升级，包含 world 维修复和 RTX 4090 大批量 startup 修复；采用为正式 B。
4. 整套迁移到 1.6/3.11：受 CollisionCfg、CommandTerm 等 breaking changes 影响更大，仅在 1.5.3 不可用时再做。
5. 等待 MJLab 正式支持 3.12：若目标必须是 3.12，这是长期正确路径。

## 预检结果

- job 44：1 env × 24 steps，1.5.3/3.10.0.3 成功。
- job 45：8192 env × 24 steps + `model_4999.pt`，1.5.3/3.10.0.3 成功。
- 正式性能结论以 30 iteration、去除前 5 iteration 的 A→B→B→A 为准，单次预检耗时不参与比较。

## 正式 ABBA 结果

- jobs 46→47→48→49，A→B→B→A 全部成功。
- Collection 对称变化：-12.681%；两组配对为 -12.328%、-13.032%。
- moving-block 95% 区间：-13.115% 至 -12.121%。
- steps/s：+13.085%。
- 同一模型元数据、版本匹配 MJB 的官方 testspeed：physics step -13.03%，solve -29.90%，与训练端结果一致。

为隔离 simulator-stack，ABBA 把 checkpoint、actor、PPO 和 `rsl-rl-lib` 固定在 5.2.0。MJLab 1.5.3 正式依赖 RSL-RL 5.4.0；生产迁移还需单独完成该依赖与本仓库自定义 runner 的兼容验收。
