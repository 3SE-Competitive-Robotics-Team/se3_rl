"""SE3 训练 runner 扩展。"""

from __future__ import annotations

import os

import torch
from mjlab.rl import MjlabOnPolicyRunner
from rsl_rl.utils import check_nan

from se3_train.training_runtime import (
    IterationTimer,
    IterationTiming,
    detect_training_runtime,
    format_runtime_summary,
)


class Se3ProfiledOnPolicyRunner(MjlabOnPolicyRunner):
    """带 SE3 运行时画像的 MJLab on-policy runner。"""

    def __init__(self, *args, **kwargs) -> None:
        """初始化 runner，并采集一次训练运行时资源快照。"""
        super().__init__(*args, **kwargs)
        self._se3_runtime_info = detect_training_runtime()
        self._se3_runtime_info_logged = False
        if not self.is_distributed or self.gpu_global_rank == 0:
            print(format_runtime_summary(self._se3_runtime_info), flush=True)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """运行 PPO 训练循环，并记录采样、return、update 的分段耗时。"""
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        num_steps_per_env = int(self.cfg["num_steps_per_env"])
        for it in range(start_it, total_it):
            timer = IterationTimer(self.env.num_envs, num_steps_per_env)
            with torch.inference_mode():
                for _ in range(num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = (
                        self.alg.intrinsic_rewards
                        if self.cfg["algorithm"].get("rnd_cfg")
                        else None
                    )
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                timer.mark_collect_done()
                self.alg.compute_returns(obs)
                timer.mark_returns_done()

            loss_dict = self.alg.update()
            timing = timer.finish()
            self.current_learning_iteration = it

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=timing.collect_s,
                learn_time=timing.returns_s + timing.learn_s,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"].get("rnd_cfg") else None,
            )
            self._log_profile_scalars(it, timing)

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

        if self.logger.writer is not None:
            self.save(
                os.path.join(
                    self.logger.log_dir,  # type: ignore[arg-type]
                    f"model_{self.current_learning_iteration}.pt",
                )
            )
            self.logger.stop_logging_writer()

    def _log_profile_scalars(self, iteration: int, timing: IterationTiming) -> None:
        """把 SE3 runtime profile 写入 logger 后端。"""
        writer = self.logger.writer
        if writer is None:
            return
        for key, value in timing.as_log_dict().items():
            writer.add_scalar(key, value, iteration)
        writer.add_scalar("Perf/update_s", timing.learn_s, iteration)
        if not self._se3_runtime_info_logged:
            for key, value in self._se3_runtime_info.as_log_dict().items():
                writer.add_scalar(key, value, iteration)
            self._se3_runtime_info_logged = True


class Se3WarmStartRunner(Se3ProfiledOnPolicyRunner):
    """阶段切换专用 warm-start runner。

    上一阶段 checkpoint 只提供 actor/critic 初始权重。新阶段必须从新的 runner
    迭代、optimizer 和环境计数开始，否则课程和日志会继承旧 checkpoint 的训练进度。
    """

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        """只加载策略和价值网络权重，不恢复 optimizer、iter 和 env_state。"""
        warm_start_cfg = {
            "actor": True,
            "critic": True,
            "optimizer": False,
            "iteration": False,
            "rnd": False,
        }

        if map_location is None and not torch.cuda.is_available():
            map_location = "cpu"
        loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
        actor_state_dict = loaded_dict.get("actor_state_dict", {})
        if "std" in actor_state_dict:
            actor_state_dict["distribution.std_param"] = actor_state_dict.pop("std")
        if "log_std" in actor_state_dict:
            actor_state_dict["distribution.log_std_param"] = actor_state_dict.pop("log_std")

        self.alg.load(loaded_dict, warm_start_cfg if load_cfg is None else load_cfg, strict)
        self.current_learning_iteration = 0
        self.env.unwrapped.common_step_counter = 0
        return loaded_dict.get("infos", {})


Se3PretrainWarmStartRunner = Se3WarmStartRunner
