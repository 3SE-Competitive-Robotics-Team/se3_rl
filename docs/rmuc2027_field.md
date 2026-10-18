# RMUC 2027 场地 sim2sim 场景

来源：[keilking66/RMUC2027-map](https://github.com/keilking66/RMUC2027-map) Release v1.0.0，即官方 `RMUC2027_V1.0.0.stp`（2026-10-15 规则版）三角化后的 `field_meshes.npz`（644 分件、605 万三角面）。许可 CC BY-NC-SA 4.0（署名 RoboMaster 高校赛务；网格转换 keilking66），非商业使用。

## 生成

原始文件不进 Git，先下载到 `assets/fields/rmuc2027/raw/`：

```bash
mkdir -p assets/fields/rmuc2027/raw && cd assets/fields/rmuc2027/raw && curl -LO https://github.com/keilking66/RMUC2027-map/releases/download/v1.0.0/field_meshes.npz -LO https://github.com/keilking66/RMUC2027-map/releases/download/v1.0.0/SHA256SUMS.txt -LO https://raw.githubusercontent.com/keilking66/RMUC2027-map/main/manifest.json
```

```bash
uv run python scripts/make_rmuc2027_field_scene.py
```

约 1 分钟，产出 `assets/robots/serialleg/meshes/rmuc2027/`（高度场 + 视觉 STL，不进 Git）和 `assets/robots/serialleg/mjcf/serialleg_rmuc2027.xml`，最后做射线往返校验。

## 运行

```bash
./scripts/run_sim2x.sh --scene assets/robots/serialleg/mjcf/serialleg_rmuc2027.xml --host 127.0.0.1 --port 8080 --physics-hz 500
```

GUI 的地形下拉选 `rmuc2027`，难度 `r0` / `r1` 是两处出生点（默认 (-6.5, -1.6) 与 (6.5, 1.6)，中心对称），`--spawn X Y` 可重复传入自定义。

## 建模取舍

- **碰撞是 2 cm 高度场 + 悬空长方体**：MuJoCo mesh 碰撞按凸包算，会把台阶、坡道填平，所以全场栅格化成高度场。高度场表达不了悬空，2026-10-18 起逐格另算「朝下表面最低点」与其下方「朝上表面最高点」，净空 ≥ `--overhang-clearance`（默认 0.15 m）的格子高度场取下方地面，悬空部分按格做 [底面, 最高点] 竖柱、(底, 顶) 量化到 1 cm 后合并成长方体（当前约 7 万格 → 1.2 万个长方体，单步物理 118 → 322 µs，500 Hz 下仍远低于实时预算）。按块求凸包会把旁边的高塔一起包进来，已弃用。例：(4.6, 5.25) 处门框窗口宽 0.79 m、下沿约 0.02、横梁底约 0.32，此前被填成 0.42 m 高的墙。
- **视觉**：同色分件合并、1 cm 顶点聚类减面到约 37 万三角面，只做显示（group 2）；高度场在 group 3，默认隐藏，可在 GUI 打开对照。
- **坐标**：CAD 平移 (0, -1.6243, +1.6403)，规则矩形中心在原点，x 沿 28 m 长边。
- **地面不是水平面**：官方 CAD 地面沿 y 向两侧各有约 1.5° 的屋脊坡，脊线在中线（z=0），边缘低约 0.12–0.21 m。按 CAD 原样保留，没有改平。
- **围墙**：官方 CAD 没有外围围挡；脚本沿高度场边界加了 0.5 m 半透明墙，只是为了防止机器人掉出去，不是官方几何。
