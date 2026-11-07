"""SuperMimicRunner：采集 → 拟合世界模型 → 穿过世界模型 BPTT 改进策略（论文 Algorithm A.1）。

改写自 SuperMimic-Reproduced（Yuxi Liu，MIT，见 LICENSES/SuperMimic-Reproduced.txt）的 `training/loop.py`。

策略就是 rsl_rl PPO 构造出来的 actor（MLPModel，确定性输出），只是不再用 PPO 更新：这样 `model_*.pt`、
`export_policy_to_onnx`、se3.meta.v1 元数据与 bench 钩子全部沿用 `Se3ProfiledOnPolicyRunner`，部署链不改。
世界模型、critic 与各自优化器另存 `supermimic_*.pt`，`load` 时一起恢复。

一次 learning iteration = `schedule.steps_per_iteration` 个 env step（默认 24，与 Rough PPO 每轮采样量相同），
按 UTD 穿插更新；`max_iterations`、`save_interval`、W&B 横轴与 PPO 对照一一对应。
第 1 阶段只支持单卡。
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from tensordict import TensorDict

from se3_train.runner import Se3ProfiledOnPolicyRunner
from se3_train.training_runtime import IterationTiming, write_training_status

from .config import supermimic_cfg_from_dict
from .heads import ObservationHead, RewardHeads
from .kinematics import TreeKinematics
from .recorder import RECORDER_NAME
from .state import HingeLayout, StateSpec
from .trainer import ActorCriticTrainer, Critic, WorldModelTrainer
from .world_model import WorldModel


def cosine_lr(
    env_step: int, total_env_steps: int, base: float, final_ratio: float, decay_fraction: float
) -> float:
    """余弦衰减到 final_ratio × base（在 decay_fraction × 总步数处），之后保持。"""
    horizon = max(1.0, decay_fraction * total_env_steps)
    progress = min(1.0, env_step / horizon)
    return base * (final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


class SuperMimicRunner(Se3ProfiledOnPolicyRunner):
    """世界模型 + 一阶梯度 BPTT 的训练 runner。"""

    def __init__(
        self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu", **kwargs
    ):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        if self.is_distributed:
            raise NotImplementedError(
                "SuperMimic 第 1 阶段只支持单卡（梯度 allreduce 未移植），请用 --gpu-ids 指定一张卡"
            )
        self.sm_cfg = supermimic_cfg_from_dict(train_cfg.get("supermimic"))
        sched = self.sm_cfg.schedule
        if int(self.cfg["num_steps_per_env"]) != int(sched.steps_per_iteration):
            raise ValueError(
                f"num_steps_per_env={self.cfg['num_steps_per_env']} 与 "
                f"supermimic.schedule.steps_per_iteration={sched.steps_per_iteration} 不一致"
            )
        base = self.env.unwrapped
        if RECORDER_NAME not in base.recorder_manager:
            raise RuntimeError(
                f"env 配置缺少 recorder `{RECORDER_NAME}`（见 tasks/rough_supermimic/env_cfg.py）"
            )
        self.recorder = base.recorder_manager.get_term(RECORDER_NAME)
        self.buffer = self.recorder.buffer
        reader = self.recorder.reader
        mj_model = base.sim.mj_model
        robot = base.scene["robot"]
        spec = StateSpec(
            HingeLayout.from_model(mj_model, tuple(robot.joint_names)), base.step_dt, device
        )
        body_ids = sorted(int(b) for b in robot.indexing.body_ids.tolist())
        kin = TreeKinematics(mj_model, body_ids, device)
        self.obs_head = ObservationHead(base)
        self.reward_heads = RewardHeads(base, spec, kin, mj_model)
        self.wm = WorldModel(
            self.sm_cfg.world_model,
            spec,
            len(self.reward_heads.learned_names),
            reader.action_clip,
            device,
        )
        self.critic = Critic(self.sm_cfg.imagination.critic_hidden_dims).to(device)
        actor = self.alg.actor
        if list(actor.obs_groups) != ["actor"]:
            raise NotImplementedError(
                f"SuperMimic 假定 actor 只读 actor 观测组，实际 {actor.obs_groups}"
            )
        clip = self.cfg.get("clip_actions")
        self._clip_actions = None if clip is None else float(clip)
        self.wm_trainer = WorldModelTrainer(
            self.wm, self.sm_cfg.world_model, self.reward_heads.learned_index
        )
        self.ac_trainer = ActorCriticTrainer(
            self.wm,
            self.obs_head,
            self.reward_heads,
            actor,
            self._policy,
            self.critic,
            self.sm_cfg.imagination,
        )
        self._utd_acc = 0.0
        self.env_steps = 0
        self.num_updates = 0
        n_params = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        print(
            f"[SuperMimic] 世界模型 {n_params(self.wm) / 1e6:.2f}M 参数，critic {n_params(self.critic) / 1e6:.3f}M，"
            f"actor {n_params(actor) / 1e6:.3f}M",
            flush=True,
        )
        print(f"[SuperMimic] 解析奖励项：{self.reward_heads.analytic_names}", flush=True)
        print(f"[SuperMimic] 学习奖励项：{self.reward_heads.learned_names}", flush=True)
        print(f"[SuperMimic] 摔倒罚权重：{self.reward_heads.fall_weight}", flush=True)

    # ------------------------------------------------------------------ 策略

    def _policy(self, obs: torch.Tensor) -> torch.Tensor:
        """想象中的策略：确定性输出，按包装器的 clip_actions 裁剪（等于 env 里 ActionManager.action）。"""
        action = self.alg.actor(TensorDict({"actor": obs}, batch_size=[obs.shape[0]]))
        if self._clip_actions is not None:
            action = torch.clamp(action, -self._clip_actions, self._clip_actions)
        return action

    # ------------------------------------------------------------------ checkpoint

    def _supermimic_path(self, path: str) -> Path:
        p = Path(path)
        return p.with_name(p.name.replace("model_", "supermimic_", 1))

    def save(self, path: str, infos: dict | None = None) -> None:
        super().save(path, infos=infos)
        torch.save(
            {
                "iter": self.current_learning_iteration,
                "env_steps": self.env_steps,
                "num_updates": self.num_updates,
                "world_model": self.wm.state_dict(),
                "critic": self.critic.state_dict(),
                "world_model_opt": self.wm_trainer.opt.state_dict(),
                "actor_opt": self.ac_trainer.actor_opt.state_dict(),
                "critic_opt": self.ac_trainer.critic_opt.state_dict(),
                "supermimic_cfg": asdict(self.sm_cfg),
            },
            self._supermimic_path(path),
        )

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ):
        infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
        sm_path = self._supermimic_path(path)
        if not sm_path.is_file():
            raise FileNotFoundError(
                f"缺少与 {path} 配对的 {sm_path}，SuperMimic 续训需要世界模型与 critic"
            )
        ck = torch.load(sm_path, map_location=self.device, weights_only=False)
        self.wm.load_state_dict(ck["world_model"])
        self.critic.load_state_dict(ck["critic"])
        self.wm_trainer.opt.load_state_dict(ck["world_model_opt"])
        self.ac_trainer.actor_opt.load_state_dict(ck["actor_opt"])
        self.ac_trainer.critic_opt.load_state_dict(ck["critic_opt"])
        self.env_steps = int(ck["env_steps"])
        self.num_updates = int(ck["num_updates"])
        return infos

    # ------------------------------------------------------------------ 主循环

    def _set_lr(self, total_env_steps: int) -> float:
        sched = self.sm_cfg.schedule
        wm_cfg, im_cfg = self.sm_cfg.world_model, self.sm_cfg.imagination
        ratio = cosine_lr(
            self.env_steps, total_env_steps, 1.0, sched.lr_final_ratio, sched.lr_decay_fraction
        )
        self.wm_trainer.set_lr(wm_cfg.learning_rate * ratio)
        self.ac_trainer.set_lr(
            im_cfg.actor_learning_rate * ratio, im_cfg.critic_learning_rate * ratio
        )
        return wm_cfg.learning_rate * ratio

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        sched = self.sm_cfg.schedule
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations().to(self.device)
        actor = self.alg.actor
        actor.train()  # 只为让 obs 归一化在真实采集时更新；前向始终是确定性输出
        self.recorder.prime()
        self.wm.fit_xi(self.recorder.reader.read_xi())
        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        steps_per_it = int(sched.steps_per_iteration)
        total_env_steps = total_it * steps_per_it
        lr = self._set_lr(total_env_steps)
        for it in range(start_it, total_it):
            t_collect = t_learn = 0.0
            agg: dict[str, float] = {}
            agg_n = 0
            for _ in range(steps_per_it):
                t0 = time.perf_counter()
                with torch.no_grad():
                    actor.update_normalization(obs)
                    action = actor(obs)
                    executed = action + sched.action_noise_std * torch.randn_like(action)
                    self.recorder.set_clean_action(action)
                    obs, rewards, dones, extras = self.env.step(executed.to(self.env.device))
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.logger.process_env_step(rewards, dones, extras, None)
                self.env_steps += 1
                t_collect += time.perf_counter() - t0

                if self.env_steps < sched.warmup_env_steps:
                    continue
                self._utd_acc += sched.updates_per_env_step
                while self._utd_acc >= 1.0:
                    self._utd_acc -= 1.0
                    t0 = time.perf_counter()
                    lr = self._set_lr(total_env_steps)
                    m_wm = self.wm_trainer.update(self.buffer)
                    m_ac = self.ac_trainer.update(self.buffer)
                    self.num_updates += 1
                    for k, v in m_wm.items():
                        agg[f"wm/{k}"] = agg.get(f"wm/{k}", 0.0) + v
                    for k, v in m_ac.items():
                        agg[f"policy/{k}"] = agg.get(f"policy/{k}", 0.0) + v
                    agg_n += 1
                    t_learn += time.perf_counter() - t0

            loss_dict = {k: v / agg_n for k, v in agg.items()} if agg_n else {}
            if agg_n and self.buffer.size > 2048:
                err = self.wm_trainer.online_error(self.buffer)
                loss_dict["wm/online_nse_1step"] = float(err[0])
                loss_dict[f"wm/online_nse_{err.numel()}step"] = float(err[-1])
            loss_dict["sm/updates"] = float(self.num_updates)
            loss_dict["sm/env_steps"] = float(self.env_steps)
            self.current_learning_iteration = it
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=t_collect,
                learn_time=t_learn,
                loss_dict=loss_dict,
                learning_rate=lr,
                action_std=torch.zeros(1),
                rnd_weight=None,
            )
            if self.logger.writer is not None:
                total_s = t_collect + t_learn
                write_training_status(
                    self.logger.log_dir,  # type: ignore[arg-type]
                    iteration=it,
                    total_iterations=total_it,
                    timing=IterationTiming(
                        collect_s=t_collect,
                        returns_s=0.0,
                        learn_s=t_learn,
                        total_s=total_s,
                        steps_per_second=steps_per_it * self.env.num_envs / max(total_s, 1.0e-9),
                    ),
                )
                if it % self.cfg["save_interval"] == 0:
                    self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

        if self.logger.writer is not None:
            self._se3_final_save = True
            self.save(
                os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")  # type: ignore[arg-type]
            )
            self.logger.stop_logging_writer()


__all__ = ["SuperMimicRunner", "cosine_lr"]
