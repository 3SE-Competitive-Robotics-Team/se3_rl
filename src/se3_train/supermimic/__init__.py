"""SuperMimic（arXiv 2610.28480）的世界模型 + 一阶梯度 BPTT 训练器，SerialLeg 移植版。

骨架（世界模型、回放缓存、想象 rollout、actor / critic 训练、采集 recorder）移植自
CuboiLeo/SuperMimic-Reproduced（Yuxi Liu，MIT，许可全文见 LICENSES/SuperMimic-Reproduced.txt），
G1 专用部分（状态、FK、观测、奖励）按 SerialLeg 重写。设计与取舍见 docs/plan/rough_supermimic_20261106.md。
"""
