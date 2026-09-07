"""AMP 流水线回归测试：19 维运动帧、数据集加载与镜像、判别器可分性、任务注册、Se3PPO 端到端一轮、存档往返。

专家数据用合成的 `source_dataset.npz`（契约 se3.amp.motion.v1，见 docs/amp_input.md），不依赖真实数据集。
"""

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_from_euler_xyz

import se3_train  # noqa: F401  # 注册任务
from se3_shared.amp import AMP_FEATURE_NAMES, AMP_FRAME_DIM
from se3_train.amp import AMP, build_amp_dataset, load_amp_sequences, mirror_amp_frames
from se3_train.mdp.amp_observations import amp_motion_frame
from se3_train.tasks.rough.env_cfg import env_cfg as rough_env_cfg
from se3_train.tasks.rough.rl_cfg import amp_rl_cfg

_AMP_TASK = "SE3-WheelLegged-Rough-AMP"
_IDX = {name: i for i, name in enumerate(AMP_FEATURE_NAMES)}


def _write_synthetic_dataset(root: Path, *, lengths: tuple[int, ...] = (40, 30, 50)) -> None:
    """按 package 脚本的 source_dataset.npz 形态写合成序列：frames + frame_offsets。"""
    rng = np.random.default_rng(0)
    root.mkdir(parents=True, exist_ok=True)
    frames = []
    for length in lengths:
        walk = np.cumsum(rng.normal(scale=0.02, size=(length, AMP_FRAME_DIM)), axis=0)
        walk[:, 0:3] = walk[:, 0:3] * 0.05 + np.array([0.0, 0.0, -1.0])
        walk[:, 9:13] += np.array([-0.05, -0.20, -0.05, -0.20])
        frames.append(walk.astype(np.float32))
    offsets = np.concatenate([[0], np.cumsum([f.shape[0] for f in frames])])
    np.savez(root / "source_dataset.npz", frames=np.concatenate(frames), frame_offsets=offsets)


class AmpDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "ds"
        _write_synthetic_dataset(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_split_by_offsets(self) -> None:
        seqs = load_amp_sequences(self.root, 0.02)
        self.assertEqual([s.shape for s in seqs], [(40, 19), (30, 19), (50, 19)])
        with self.assertRaises(ValueError):
            load_amp_sequences(self.root, 0.01)  # 契约固定 20 ms

    def test_per_sample_features_layout(self) -> None:
        root = Path(self.tmp.name) / "samples"
        for k in range(2):
            d = root / f"step{k}" / "amp"
            d.mkdir(parents=True)
            np.savez(d / "source_features.npz", frames=np.zeros((10 + k, 19), np.float32), valid=np.ones(9 + k, bool))
        self.assertEqual([s.shape[0] for s in load_amp_sequences(root, 0.02)], [10, 11])

    def test_mirror(self) -> None:
        ds = build_amp_dataset(dataset_root=str(self.root), simulation_dt=0.02, mirror_augmentation=True)
        self.assertEqual(len(ds["sequences"]), 6)
        orig, mirrored = ds["sequences"][0], ds["sequences"][3]
        i = _IDX
        self.assertTrue(torch.allclose(mirrored[:, i["gravity_y"]], -orig[:, i["gravity_y"]]))
        self.assertTrue(torch.allclose(mirrored[:, i["gravity_z"]], orig[:, i["gravity_z"]]))
        self.assertTrue(torch.allclose(mirrored[:, i["base_omega_x"]], -orig[:, i["base_omega_x"]]))
        self.assertTrue(torch.allclose(mirrored[:, i["base_omega_y"]], orig[:, i["base_omega_y"]]))
        self.assertTrue(torch.allclose(mirrored[:, i["base_omega_z"]], -orig[:, i["base_omega_z"]]))
        self.assertTrue(torch.allclose(mirrored[:, i["base_velocity_y"]], -orig[:, i["base_velocity_y"]]))
        self.assertTrue(torch.allclose(mirrored[:, i["left_wheel_x"]], orig[:, i["right_wheel_x"]]))
        self.assertTrue(torch.allclose(mirrored[:, i["right_wheel_vz"]], orig[:, i["left_wheel_vz"]]))
        self.assertTrue(torch.allclose(mirrored[:, i["left_wheel_spin"]], orig[:, i["right_wheel_spin"]]))
        # 镜像两次回到原状。
        self.assertTrue(torch.allclose(mirror_amp_frames(mirror_amp_frames(orig)), orig))


class _DummyStorage:
    """给 AMP.individual_update 用的最小 storage 替身。"""

    def __init__(self, frames: torch.Tensor, dones: torch.Tensor) -> None:
        from tensordict import TensorDict

        self.observations = TensorDict({"amp": frames}, batch_size=frames.shape[:2])
        self.dones = dones
        self.step = frames.shape[0]


class AmpDiscriminatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "ds"
        _write_synthetic_dataset(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _make_amp(self, num_envs: int = 8) -> AMP:
        from tensordict import TensorDict

        obs = TensorDict({"amp": torch.zeros(num_envs, 19), "actor": torch.zeros(num_envs, 3)}, batch_size=[num_envs])
        cfg = {
            "obs_group": "amp",
            "transition_frames": 2,
            "reward_weight": 3.0,
            "reward_warmup_updates": 0,
            "discriminator_updates": 5,
            "discriminator_batch_size": 64,
            "discriminator_grad_penalty_weight": 1.0,
            "learning_rate": 3.0e-3,
            "model_cfg": {"hidden_dims": [32], "activation": "elu", "state_normalization": True},
            "dataset_kwargs": {"dataset_root": str(self.root)},
        }
        return AMP(num_envs=num_envs, step_dt=0.02, obs=obs, cfg=cfg, device="cpu")

    def test_reward_needs_two_frames_and_resets_on_done(self) -> None:
        from rsl_rl.storage import RolloutStorage
        from tensordict import TensorDict

        amp = self._make_amp(num_envs=4)
        # 与 fork 的单测一样把判别器奖励钉成 1，只测帧缓冲/done 逻辑。
        amp.discriminator.predict_reward = lambda sequences: torch.ones(sequences.shape[0])  # type: ignore[method-assign]
        obs = TensorDict({"amp": torch.zeros(4, 19)}, batch_size=[4])
        t = RolloutStorage.Transition()
        t.rewards = torch.zeros(4)
        t.dones = torch.zeros(4, dtype=torch.bool)
        amp.process_env_step(obs, t, {})
        self.assertTrue(torch.equal(t.rewards, torch.zeros(4)))  # 只有一帧，还给不出奖励
        t.rewards = torch.zeros(4)
        t.dones = torch.tensor([True, False, False, False])
        extras: dict = {}
        amp.process_env_step(obs, t, extras)
        self.assertEqual(float(t.rewards[0]), 0.0)  # done 的 env 清缓冲
        self.assertTrue(torch.allclose(t.rewards[1:], torch.full((3,), 3.0 * 0.02)))  # 3.0×dt×1
        self.assertIn("amp", extras["ext_reward"])

    def test_normalizer_is_per_frame_and_shared(self) -> None:
        amp = self._make_amp()
        self.assertEqual(tuple(amp.discriminator.normalizer._mean.shape[-1:]), (19,))

    def test_discriminator_learns_to_separate(self) -> None:
        amp = self._make_amp(num_envs=8)
        frames = torch.full((16, 8, 19), 2.0)  # 与专家明显不同的分布
        dones = torch.zeros(16, 8, 1, dtype=torch.uint8)
        dones[5, 2] = 1
        storage = _DummyStorage(frames, dones)
        before = amp.individual_update(storage)
        for _ in range(20):
            after = amp.individual_update(storage)
        self.assertLess(after["amp/discriminator_loss"], before["amp/discriminator_loss"])
        self.assertGreater(after["amp/expert_score"], after["amp/policy_score"])
        self.assertEqual(int(amp.amp_update_counter), 21)
        self.assertEqual(int(amp._rollout_valid_end_indices(storage).numel()), 16 * 8 - 8 - 2)


class AmpMotionFrameTests(unittest.TestCase):
    """amp_motion_frame 的几何：直立重力 [0,0,-1]、轮心在髋轴下方、整体绕 z 旋转不改变特征。"""

    @classmethod
    def setUpClass(cls) -> None:
        cfg = rough_env_cfg(amp_enabled=True, flat_warmup_iterations=0)
        cfg.scene.num_envs = 2
        cls.env = ManagerBasedRlEnv(cfg, device="cpu")
        cls.env.reset()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def _place(self, yaw: torch.Tensor) -> None:
        robot = self.env.scene["robot"]
        # reset 会随机腿姿（髋 ±90°），先把两个 env 都放回默认站姿再比较。
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_pos)
        )
        pose = robot.data.root_link_pose_w.clone()
        zeros = torch.zeros_like(yaw)
        pose[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
        robot.write_root_link_pose_to_sim(pose)
        robot.write_root_link_velocity_to_sim(torch.zeros(self.env.num_envs, 6))
        self.env.sim.forward()

    def test_frame_geometry_and_yaw_invariance(self) -> None:
        self._place(torch.tensor([0.0, 1.3]))
        frame = amp_motion_frame(self.env)
        self.assertEqual(tuple(frame.shape), (2, AMP_FRAME_DIM))
        self.assertEqual(list(self.env.observation_manager.compute()["amp"].shape), [2, AMP_FRAME_DIM])
        i = _IDX
        self.assertTrue(torch.allclose(frame[:, i["gravity_z"]], torch.full((2,), -1.0), atol=1e-3))
        # 轮心在髋轴下方（z<0），左右对称（x 相近）。
        self.assertTrue(bool((frame[:, i["left_wheel_z"]] < -0.1).all()))
        self.assertTrue(bool((frame[:, i["right_wheel_z"]] < -0.1).all()))
        self.assertLess(float((frame[:, i["left_wheel_x"]] - frame[:, i["right_wheel_x"]]).abs().max()), 0.03)
        # env 0（yaw 0）与 env 1（yaw 1.3）姿态相同，特征必须一致；容差留给逐 env 的模型随机化（毫米级）。
        self.assertTrue(torch.allclose(frame[0], frame[1], atol=1e-2))


class AmpTaskTests(unittest.TestCase):
    def test_rough_amp_task_has_group_and_algorithm(self) -> None:
        cfg = load_env_cfg(_AMP_TASK)
        self.assertEqual(list(cfg.observations["amp"].terms), ["motion_frame"])
        self.assertNotIn("amp", load_env_cfg("SE3-WheelLegged-Rough").observations)
        rl = load_rl_cfg(_AMP_TASK)
        self.assertEqual(rl.algorithm.class_name, "se3_train.ppo:Se3PPO")
        self.assertEqual(rl.algorithm.amp_cfg["reward_weight"], 3.0)
        base = asdict(load_rl_cfg("SE3-WheelLegged-Rough").algorithm)
        mine = asdict(rl.algorithm)
        for key in ("learning_rate", "entropy_coef", "num_learning_epochs", "clip_param", "gamma", "lam", "desired_kl"):
            self.assertEqual(base[key], mine[key], key)

    def test_dataset_root_env_override(self) -> None:
        old = os.environ.get("SE3_AMP_DATASET_ROOT")
        os.environ["SE3_AMP_DATASET_ROOT"] = "/tmp/somewhere"
        try:
            self.assertEqual(amp_rl_cfg().algorithm.amp_cfg["dataset_kwargs"]["dataset_root"], "/tmp/somewhere")
        finally:
            if old is None:
                del os.environ["SE3_AMP_DATASET_ROOT"]
            else:
                os.environ["SE3_AMP_DATASET_ROOT"] = old


class AmpEndToEndTests(unittest.TestCase):
    """CPU 上 4 个 env 跑一轮采集 + PPO 更新 + 判别器更新 + 存档往返。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name) / "ds"
        _write_synthetic_dataset(root)
        env_cfg = rough_env_cfg(amp_enabled=True, flat_warmup_iterations=0)
        env_cfg.scene.num_envs = 4
        agent = amp_rl_cfg(dataset_root=str(root))
        agent.logger = "tensorboard"
        agent.num_steps_per_env = 6
        agent.algorithm.amp_cfg["discriminator_batch_size"] = 8
        agent.algorithm.amp_cfg["discriminator_updates"] = 1
        agent.algorithm.amp_cfg["reward_warmup_updates"] = 0
        cls.env = ManagerBasedRlEnv(env_cfg, device="cpu")
        cls.wrapped = RslRlVecEnvWrapper(cls.env, clip_actions=agent.clip_actions)
        runner_cls = load_runner_cls(_AMP_TASK) or MjlabOnPolicyRunner
        cls.runner = runner_cls(cls.wrapped, asdict(agent), tempfile.mkdtemp(), "cpu")
        cls.agent = agent

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()
        cls.tmp.cleanup()

    def test_rollout_update_and_checkpoint(self) -> None:
        alg = self.runner.alg
        self.assertIsNotNone(alg.amp)
        self.assertAlmostEqual(alg.amp.step_dt, self.env.step_dt)
        self.assertEqual(alg.amp.amp_obs_dim, AMP_FRAME_DIM)
        obs = self.wrapped.get_observations()
        alg.train_mode()
        for _ in range(self.agent.num_steps_per_env):
            actions = alg.act(obs)
            obs, rewards, dones, extras = self.wrapped.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)
            self.assertIn("amp", extras.get("ext_reward", {}))
        alg.compute_returns(obs)
        loss = alg.update()
        for key in ("value", "surrogate", "amp/style_reward", "amp/reward_scale", "amp/discriminator_loss"):
            self.assertIn(key, loss)
        self.assertEqual(int(alg.amp.amp_update_counter), 1)
        saved = alg.save()
        self.assertIn("amp", saved["ext_state_dict"])
        with torch.no_grad():
            for p in alg.amp.discriminator.parameters():
                p.add_(1.0)
        alg.load(saved, None, True)
        flat_after = torch.cat([p.flatten() for p in alg.amp.discriminator.parameters()])
        flat_saved = torch.cat(
            [
                v.flatten()
                for k, v in saved["ext_state_dict"]["amp"]["module_state_dict"].items()
                if k.startswith("discriminator.backbone")
            ]
        )
        self.assertTrue(torch.allclose(flat_after.sort().values, flat_saved.sort().values))


if __name__ == "__main__":
    unittest.main()
