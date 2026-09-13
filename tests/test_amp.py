"""通用 AMP 测试：运动帧、数据集加载与镜像、判别器；Rough 的 AMP 训练入口已移除。

专家数据用合成的 se3.amp.pkl.v1（契约 se3.amp.motion.v1，见 docs/amp_input.md），不依赖真实数据集。
"""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_from_euler_xyz

import se3_train  # noqa: F401  # 注册任务
from se3_shared.amp import AMP_FEATURE_NAMES, AMP_FRAME_DIM
from se3_train.amp import AMP
from se3_train.amp_dataset_factory import build_amp_dataset
from se3_train.mdp.amp_observations import (
    AMP_DISCRIMINATOR_FIELDS,
    amp_field_indices,
    amp_motion_frame,
)
from se3_train.motion_loader import MotionLoader
from se3_train.tasks.rough.env_cfg import env_cfg as rough_env_cfg

_IDX = {name: i for i, name in enumerate(AMP_FEATURE_NAMES)}


def _write_synthetic_dataset(
    root: Path, *, lengths: tuple[int, ...] = (40, 30, 50), fps: float = 50.0
) -> Path:
    """按 scripts/export_fudan_amp_pkl.py 的 se3.amp.pkl.v1 形态写合成序列，返回 pkl 路径。"""
    rng = np.random.default_rng(0)
    root.mkdir(parents=True, exist_ok=True)
    sequences = []
    for length in lengths:
        walk = np.cumsum(rng.normal(scale=0.02, size=(length, AMP_FRAME_DIM)), axis=0)
        walk[:, 0:3] = walk[:, 0:3] * 0.05 + np.array([0.0, 0.0, -1.0])
        walk[:, 9:13] += np.array([-0.05, -0.20, -0.05, -0.20])
        sequences.append(walk.astype(np.float32))
    payload = dict(
        format="se3.amp.pkl.v1",
        motion_contract="se3.amp.motion.v1",
        fps=fps,
        dt=1.0 / fps,
        frame_dim=19,
        transition_dim=38,
        normalization="raw SI",
        retargeted_to_serialleg=False,
        feature_names=list(AMP_FEATURE_NAMES),
        sequences=sequences,
        transitions=[np.concatenate([s[:-1], s[1:]], axis=1) for s in sequences],
        lengths=[int(s.shape[0]) for s in sequences],
        time_s=[np.arange(s.shape[0]) / fps for s in sequences],
        source_time_s=[np.arange(s.shape[0]) / fps for s in sequences],
        annotations=[{"id": f"clip{i}", "step_height_m": 0.12} for i in range(len(sequences))],
    )
    path = root / "amp_training.pkl"
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=5)
    return path


class AmpDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "ds"
        self.pkl = _write_synthetic_dataset(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_loader_shapes_and_format(self) -> None:
        loader = MotionLoader(self.root, 0.02, mirror_augmentation=False)
        self.assertEqual([tuple(s.shape) for s in loader.sequences], [(40, 19), (30, 19), (50, 19)])
        self.assertEqual(loader.obs_dim, 19)
        self.assertEqual(loader.format["fields"], list(AMP_FEATURE_NAMES))
        self.assertEqual(loader.format["feature_slices"]["left_wheel_x"], [9, 10])
        self.assertFalse(loader.metadata["retargeted_to_serialleg"])
        self.assertEqual(loader.metadata["annotations"][1]["id"], "clip1")
        # 目录与单文件两种给法等价。
        self.assertEqual(MotionLoader(self.pkl, 0.02, mirror_augmentation=False).obs_dim, 19)

    def test_resample_to_control_dt(self) -> None:
        root = Path(self.tmp.name) / "fast"
        _write_synthetic_dataset(root, lengths=(41,), fps=100.0)
        loader = MotionLoader(root, 0.02, mirror_augmentation=False)
        self.assertEqual(int(loader.sequences[0].shape[0]), 21)  # 0.4 s @ 20 ms

    def test_field_subset(self) -> None:
        loader = MotionLoader(
            self.root, 0.02, fields=["gravity_z", "left_wheel_spin"], mirror_augmentation=False
        )
        self.assertEqual(loader.obs_dim, 2)
        with self.assertRaises(ValueError):
            MotionLoader(self.root, 0.02, fields=["nope"])

    def test_contract_checks(self) -> None:
        bad = Path(self.tmp.name) / "bad"
        bad.mkdir()
        with (bad / "x.pkl").open("wb") as f:
            pickle.dump({"format": "other", "motion_contract": "se3.amp.motion.v1", "fps": 50.0}, f)
        with self.assertRaises(ValueError):
            MotionLoader(bad, 0.02)
        with (bad / "x.pkl").open("wb") as f:
            payload = pickle.load(self.pkl.open("rb"))
            payload["feature_names"] = list(reversed(payload["feature_names"]))
            pickle.dump(payload, f)
        with self.assertRaises(ValueError):
            MotionLoader(bad, 0.02)

    def test_mirror(self) -> None:
        ds = build_amp_dataset(
            dataset_root=str(self.root), simulation_dt=0.02, mirror_augmentation=True
        )
        self.assertEqual(len(ds["sequences"]), 6)
        orig, mirrored = ds["sequences"][0], ds["sequences"][1]  # 镜像紧跟原序列（与 kyber 一致）
        i = _IDX
        self.assertTrue(torch.allclose(mirrored[:, i["gravity_y"]], -orig[:, i["gravity_y"]]))
        self.assertTrue(torch.allclose(mirrored[:, i["gravity_z"]], orig[:, i["gravity_z"]]))
        self.assertTrue(torch.allclose(mirrored[:, i["base_omega_x"]], -orig[:, i["base_omega_x"]]))
        self.assertTrue(torch.allclose(mirrored[:, i["base_omega_y"]], orig[:, i["base_omega_y"]]))
        self.assertTrue(torch.allclose(mirrored[:, i["base_omega_z"]], -orig[:, i["base_omega_z"]]))
        self.assertTrue(
            torch.allclose(mirrored[:, i["base_velocity_y"]], -orig[:, i["base_velocity_y"]])
        )
        self.assertTrue(torch.allclose(mirrored[:, i["left_wheel_x"]], orig[:, i["right_wheel_x"]]))
        self.assertTrue(
            torch.allclose(mirrored[:, i["right_wheel_vz"]], orig[:, i["left_wheel_vz"]])
        )
        self.assertTrue(
            torch.allclose(mirrored[:, i["left_wheel_spin"]], orig[:, i["right_wheel_spin"]])
        )
        # 镜像两次回到原状。
        twice = MotionLoader._mirror_frames(MotionLoader._mirror_frames(orig.numpy()))
        self.assertTrue(np.allclose(twice, orig.numpy()))

    def test_factory_validates_env_obs_dim(self) -> None:
        class _Env:
            step_dt = 0.02

            class observation_manager:
                @staticmethod
                def compute():
                    return {"amp": torch.zeros(2, 18)}

        with self.assertRaises(ValueError):
            build_amp_dataset(env=_Env(), dataset_root=str(self.root))


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

        obs = TensorDict(
            {"amp": torch.zeros(num_envs, 19), "actor": torch.zeros(num_envs, 3)},
            batch_size=[num_envs],
        )
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

    def test_mask_group_gates_reward_and_windows(self) -> None:
        from rsl_rl.storage import RolloutStorage
        from tensordict import TensorDict

        amp = self._make_amp(num_envs=4)
        amp.mask_obs_group = "amp_mask"
        amp.discriminator.predict_reward = lambda sequences: torch.ones(sequences.shape[0])  # type: ignore[method-assign]
        mask = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
        obs = TensorDict({"amp": torch.zeros(4, 19), "amp_mask": mask}, batch_size=[4])
        t = RolloutStorage.Transition()
        for _ in range(2):
            t.rewards = torch.zeros(4)
            t.dones = torch.zeros(4, dtype=torch.bool)
            amp.process_env_step(obs, t, {})
        self.assertTrue(
            torch.allclose(t.rewards, torch.tensor([0.06, 0.06, 0.0, 0.0]))
        )  # 未启用的 env 拿 0
        # 判别器窗口同样只来自启用的 env。
        from tensordict import TensorDict as TD

        frames = torch.zeros(6, 4, 19)
        storage = _DummyStorage(frames, torch.zeros(6, 4, 1, dtype=torch.uint8))
        storage.observations = TD(
            {"amp": frames, "amp_mask": mask.expand(6, 4, 1).clone()}, batch_size=[6, 4]
        )
        valid = amp._rollout_valid_end_indices(storage)
        self.assertEqual(int(valid.numel()), 5 * 2)  # 4 env 里 2 个启用，各 5 个窗口
        self.assertTrue(bool(((valid % 4) < 2).all()))

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
        cfg = rough_env_cfg()
        # 平地热身会把首次 reset 的全部 env 放到平地列；这里要六列各占一个 env。
        cfg.curriculum.pop("flat_warmup", None)
        # 通用 AMP 几何测试自行挂接观测，不恢复 Rough 的 AMP 训练入口。
        from mjlab.managers import ObservationGroupCfg

        from se3_train.mdp.amp_observations import build_amp_mask_terms, build_amp_obs_terms

        cfg.observations["amp"] = ObservationGroupCfg(
            terms=build_amp_obs_terms(), concatenate_terms=True, enable_corruption=False
        )
        cfg.observations["amp_mask"] = ObservationGroupCfg(
            terms=build_amp_mask_terms(("stairs_up",)),
            concatenate_terms=True,
            enable_corruption=False,
        )
        cfg.scene.num_envs = 6  # 六列各 1 个 env，含上台阶列
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

    def test_terrain_mask_marks_stairs_up_column(self) -> None:
        from se3_train.mdp.amp_observations import amp_terrain_mask

        terrain = self.env.scene.terrain
        names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
        expected = (
            (terrain.terrain_types == names.index("stairs_up")).to(torch.float32).unsqueeze(-1)
        )
        self.assertTrue(torch.equal(amp_terrain_mask(self.env, ("stairs_up",)), expected))
        self.assertTrue(torch.equal(self.env.observation_manager.compute()["amp_mask"], expected))
        self.assertTrue(bool((amp_terrain_mask(self.env, ()) == 1.0).all()))

    def test_frame_geometry_and_yaw_invariance(self) -> None:
        n = self.env.num_envs
        i = _IDX
        self._place(torch.zeros(n))
        frame = amp_motion_frame(self.env, fields=AMP_FEATURE_NAMES)
        self.assertEqual(tuple(frame.shape), (n, AMP_FRAME_DIM))
        # 任务默认切列：去掉两个轮速，17 维，列顺序与契约一致。
        sliced = self.env.observation_manager.compute(update_history=True)[
            "amp"
        ]  # 不带参会返回上一步缓存
        self.assertEqual(len(AMP_DISCRIMINATOR_FIELDS), 17)
        self.assertNotIn("left_wheel_spin", AMP_DISCRIMINATOR_FIELDS)
        self.assertEqual(list(sliced.shape), [n, 17])
        self.assertTrue(
            torch.allclose(sliced, frame[:, amp_field_indices(AMP_DISCRIMINATOR_FIELDS)])
        )
        self.assertTrue(torch.allclose(frame[:, i["gravity_z"]], torch.full((n,), -1.0), atol=1e-3))
        # 轮心在髋轴下方（z<0），左右对称（x 相近；容差留给逐 env 的模型随机化）。
        self.assertTrue(bool((frame[:, i["left_wheel_z"]] < -0.1).all()))
        self.assertTrue(bool((frame[:, i["right_wheel_z"]] < -0.1).all()))
        self.assertLess(
            float((frame[:, i["left_wheel_x"]] - frame[:, i["right_wheel_x"]]).abs().max()), 0.05
        )
        # 同一个 env 整体绕 z 转 1.3 rad，特征不变（不跨 env 比较，避免逐 env 随机化干扰）。
        self._place(torch.full((n,), 1.3))
        rotated = amp_motion_frame(self.env, fields=AMP_FEATURE_NAMES)
        self.assertTrue(torch.allclose(frame, rotated, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
