"""SerialLeg 跳跃参考轨迹模块。

reference.py 生成 se3.jump_ref.v1 原地跳跃运动学参考（policy 主动杆坐标、腿长、机身高度/竖直速度、
接触标志、阶段），供跳跃 mimic 任务使用；只保证运动学，动力学由训练 rollout 迭代。
kinematics.py / replay.py 保留给旧格式文件回放。
"""
