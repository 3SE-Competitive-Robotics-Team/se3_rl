"""SerialLeg 跳跃轨迹优化（TO）模块。

基于质心动力学（centroidal dynamics）的 kino-dynamic 轨迹优化，
为 TO+RL 路线提供参考跳跃轨迹。

主要输出：
    - base_link 位置/速度轨迹
    - 关节角度轨迹（站立段）
    - 着陆时刻时间戳（用于 RL 的奖励松弛）
"""
