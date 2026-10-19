# 复旦 sim2sim 台阶场景

来源：[复旦仓库 `mujoco/assert_now/infantry_binglian_yuntai/infantry_V2/meshes/mjmodel.xml`](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/mujoco/assert_now/infantry_binglian_yuntai/infantry_V2/meshes/mjmodel.xml#L50-L100)。此处的“二级台阶”是源 sim2sim 场景中 20 cm 和 35 cm 的两道 box 障碍，不是训练端的金字塔台阶。

场景文件 `assets/robots/serialleg/mjcf/serialleg_fudan_sim2sim_steps.xml` 复制源场景的六个地形 box：`step_20cm`、侧向 `step_20rightcm`、`step_35cm`、`platform_30cm`、`fly_ramp_1`、`landing_box`，并复制蓝色方格地面。尺寸、颜色、斜板角度和编译后的地形接触参数保持一致。

源机器人出生在 x=-2.8，本项目的机器人出生在 x=0，因此所有障碍 x 坐标统一加 2.8 m；出生点仍落在 30 cm 落地区上。场景用 `se3_spawn_surface` 标记平台高度，sim2x reset 时把机器人起始 z 增加 0.30 m。机器人本体、关节和 ONNX metadata 仍使用本项目的模型与契约。源 XML 虽声明了 `terrain_hfield.png`，但 worldbody 没有使用它，因此场景未复制该文件。

```bash
./scripts/run_sim2x.sh --scene assets/robots/serialleg/mjcf/serialleg_fudan_sim2sim_steps.xml --host 127.0.0.1 --port 8080 --physics-hz 500
```

原场景的障碍位于机器人初始位置的 **-x 方向**；需要向该方向运动才能依次经过落地区、斜板、平台与两道台阶。机器人结构和控制策略与复旦项目不同，因此此场景用于验证几何障碍下的表现，不能据此直接比较两套策略的通过率。

Viser 键盘快捷键在整个 sim2x viewer 中可用：I/K 每按一次将前进速度增/减 0.2 m/s，J/L 将偏航速度增/减 0.5 rad/s，U/O 将机身高度减/增 0.01 m，空格将运动和转向指令清零。当前复旦场景的障碍在 -x，按 K 可逐步给出朝台阶的负向速度。滑块会同步显示键盘设置；WASD 和方向键仍用于移动相机。
