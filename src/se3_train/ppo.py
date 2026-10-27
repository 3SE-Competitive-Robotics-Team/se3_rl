"""带独立 critic 学习率的 PPO。

背景（2026-09-05，D4 诊断）：rsl_rl 的 KL 自适应学习率作用在 actor 与 critic 共用的唯一
optimizer 上。σ 缩小后梯度方向变得一致，KL 规则把 actor 的 LR 压到 1e-5 地板，critic 的 LR 被一起压到
地板：D4 从 3580 轮起 97% 的轮次贴地板，4250 轮后 Loss/value 出现最高 27 的尖峰，critic 跟不上回报变化。
critic 的回归目标与策略信任域无关，不该被同一条规则限速。

做法：`critic_learning_rate` 非 None 时把 optimizer 换成两个 param group（actor / critic）的 Adam，
每次 `step()` 前把 critic group 的 lr 恢复为固定值；KL 规则仍照常改写全部 group 的 lr，因此 actor
行为与原版 PPO 逐位相同。`critic_learning_rate=None` 时不做任何改动，等价于 rsl_rl.PPO。

多卡梯度噪声尺度（2026-10-03）：多卡时额外记录 actor 的梯度噪声尺度 B_noise（McCandlish et al. 2018,
"An Empirical Model of Large-Batch Training" 附录 A），只读梯度、不改更新。单卡 mini-batch 梯度（样本数 B_小）
与各卡平均后的梯度（B_大 = 卡数 × B_小）给出两种 batch 下的 |g|²，由
    |G|² ≈ (B_大·|g_大|² − B_小·|g_小|²) / (B_大 − B_小)，tr(Σ) ≈ (|g_小|² − |g_大|²) / (1/B_小 − 1/B_大)
解出真实梯度模长与噪声，B_noise = tr(Σ) / |G|² 是"再加样本仍明显有用"的 batch 量级。两项先按整轮
mini-batch 求均值再相除（单个 mini-batch 的估计噪声大）。日志键 Loss/grad_noise_scale_actor（样本数），
Loss/grad_sq_small_actor、Loss/grad_sq_big_actor 供复核。单卡训练不记录。

显式 vx 估计器（2026-10-05 起；单独的 VxObserverMLPModel 2026-10-19 删除，现只有 CTS 用）：`estimator_learning_rate` 非 None 时 actor
必须带估计器。估计器用独立 Adam，在 PPO 各 epoch 结束后用同一批 rollout 做 MSE 回归（epoch 数、mini-batch
数与 PPO 相同），目标是观测组 `estimator_target_group`（同一时刻的真实机身系 vx）。放在 PPO 之后是为了让 PPO 各
epoch 的 log prob 与 rollout 时用的是同一个估计器，ratio 从 1 开始；rsl_rl 的 storage.clear() 只复位写指针、不清数据，
所以 PPO update 之后仍可读同一批 rollout。估计器参数也在 PPO optimizer 里（它属于 actor），但 v̂x 在 policy 前
detach，PPO 损失对它的梯度恒为 None，Adam 跳过，不会被 PPO 更新。日志 Loss/estimator_vx_mse、Loss/estimator_vx_rmse。
估计器带隐向量时（latent_dim > 0）隐向量不 detach，PPO 梯度经它进入估计器、由 PPO optimizer 一起更新；MSE 只监督 v̂x 一列。
另记 Loss/estimator_latent_std（隐向量各维 batch 标准差均值，看是否塌缩）。
actor 自带 estimator_loss(observations) 时（CTS，见 se3_train.cts_observer）改用它给出的损失（学生编码器的隐向量重建），
并按 cts_role 分别记教师 / 学生 env 的平均单步奖励（Loss/cts_reward_teacher、Loss/cts_reward_student）：
训练端其余指标被占多数的教师 env 主导，学生 env 才代表部署路径。actor 带 latent_std(observations) 时另记
Loss/estimator_latent_std（学生隐向量是否塌缩）。

换学生编码器续训（2026-10-09，单片机部署版 128/128）：`load` 发现 checkpoint 的 estimator 形状与当前不一致时，
只丢弃 `estimator.*` 参数与估计器独立优化器状态，其余（教师编码器、policy、critic、PPO 优化器）照常加载。
PPO 优化器能原样加载：估计器参数虽在 actor 里，但从未收到 PPO 梯度，Adam 里没有它们的状态，参数个数也不变。
`estimator_warmup_iterations` > 0 时本进程前若干轮跳过 PPO 更新、只训练估计器（日志 Loss/estimator_warmup = 1），
避免随机初始化的学生编码器给 25% 学生 env 的垃圾隐向量把共享 policy 带偏。

模仿损失（2026-10-23 用户定，倒地脚本教师 tasks.rough.fall_teacher 配套）：`imitation_loss` 非 None 时 PPO 总损失加
coef·mean_接管样本(α·Huber(μ − T))，μ 是本 mini-batch 重算的策略均值，T 与 α、接管标记取自观测组 `group`（与该样本观测同一时刻）。
只靠模仿奖励时（kf3o74t7）前 200 轮策略与教师的动作差不降反升：exp(−mse/σ) 在差距大时几乎为 0，没有梯度。
T 先按前杆整圈平移到 μ 那一圈（前杆 PD 走最短角，差整圈的动作物理上相同），joint 语义下同侧后杆同步平移。
`back_follow`（2026-10-27，教师夹角归策略时用）：后杆目标改为 μ_后 + k·(μ_前 − T_前)（μ 不回传梯度），只监督后杆跟前杆转、不监督夹角。
rsl_rl 5.4 的 PPO.update 只在 symmetry 钩子处允许往总损失里加项（与 PPO 损失同一次反传，走同样的多卡梯度平均与梯度裁剪），
这里借这个钩子挂模仿项、不做对称增广；同时配真正的 symmetry_cfg 会报错。日志 Loss/imitation（原 symmetry 键改名）。

网络老师在线蒸馏（2026-10-25 用户定，recover 组起身用）：`policy_distill` 非 None 时，从 checkpoint 载入一份与本 actor 同结构的
冻结老师（actor_state_dict，eval、不进优化器、不进存档），PPO 总损失加 coef·mean_掩码样本 Huber(μ − μ_老师)：μ_老师 由老师在同一批
观测上按部署路径（CTS 角色组置 0 → 学生编码器）前向得到，掩码取观测组 `mask_group`（recover 组）。状态来自学生自己的 rollout、
标签来自老师（DAgger 式），老师与学生观测布局必须相同。同样挂在 symmetry 钩子上（与 imitation_loss 互斥），前杆整圈对齐同上，
日志 Loss/imitation。

AMP 风格奖励（2026-10-24 从 867c7299 之前恢复，recover 组起身风格用，见 se3_train.amp）：`amp_cfg` 非 None 时，
process_env_step 在写 storage 之前把风格奖励加进 transition.rewards（在 time-out bootstrap 之前，与 fork 的钩子位置一致）；
update 在 PPO 更新之前用本轮 rollout 更新判别器（PPO.update 末尾会清空 storage）。actor / critic 网络与观测组不变，
导出的 ONNX 不受影响。日志 Loss/amp/*。
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.distributed
from rsl_rl.algorithms import PPO

from se3_train.amp import AMP


class ActorCriticAdam(torch.optim.Adam):
    """param_groups[0] 为 actor（lr 由 KL 自适应规则改写），param_groups[1] 为 critic（lr 固定）。"""

    def __init__(
        self,
        actor_params,
        critic_params,
        *,
        lr: float,
        critic_lr: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            [
                {"params": list(actor_params)},
                {"params": list(critic_params), "lr": float(critic_lr)},
            ],
            lr=float(lr),
            **kwargs,
        )
        self.critic_lr = float(critic_lr)

    def step(self, closure=None):  # type: ignore[override]
        """每次更新前恢复 critic 的固定 lr，抵消 PPO.update 对全部 group 的改写。"""
        self.param_groups[1]["lr"] = self.critic_lr
        return super().step(closure)


class _ImitationLoss:
    """挂在 rsl_rl PPO 的 symmetry 钩子上的模仿项（见模块 docstring）；augment_batch 不做任何增广。"""

    use_mirror_loss = True
    mirror_loss_coeff = 1.0

    def __init__(
        self,
        *,
        group: str,
        coef: float,
        huber_delta: float,
        front_period: float,
        back_period: float,
        shift_back: bool,
        back_follow: tuple[float, float] | None = None,
    ) -> None:
        if coef < 0.0 or huber_delta <= 0.0 or front_period <= 0.0:
            raise ValueError(
                f"模仿损失参数非法：coef={coef}, huber_delta={huber_delta}, front_period={front_period}"
            )
        if shift_back and back_period <= 0.0:
            raise ValueError(f"shift_back=True 需要正的 back_period，实际 {back_period}")
        self.group = group
        self.coef = float(coef)
        self.huber_delta = float(huber_delta)
        self.front_period = float(front_period)
        self.back_period = float(back_period)
        self.shift_back = bool(shift_back)
        self.back_follow = None if back_follow is None else tuple(float(k) for k in back_follow)

    def augment_batch(self, batch: Any, original_batch_size: int) -> None:
        del batch, original_batch_size

    def align(self, target: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        """把教师动作按前杆整圈平移到策略均值那一圈（turns 不回传梯度）。"""
        turns = torch.round((target[:, [0, 2]] - mean.detach()[:, [0, 2]]) / self.front_period)
        aligned = target.clone()
        aligned[:, [0, 2]] = target[:, [0, 2]] - turns * self.front_period
        if self.shift_back:
            aligned[:, [1, 3]] = target[:, [1, 3]] - turns * self.back_period
        return aligned

    def compute_loss(self, actor: Any, batch: Any, original_batch_size: int) -> torch.Tensor:
        mean = actor.output_mean[:original_batch_size]
        target = batch.observations[self.group][:original_batch_size]
        active = (target[:, 0] > 0.5).float()
        alpha = target[:, 1]
        teacher = self.align(target[:, 2:], mean)
        if self.back_follow is not None:
            # 夹角归策略（fall_teacher fold='policy'）：后杆目标 = μ_后 + k·(μ_前 − T_前)（μ 不回传），只监督跟转量、不监督夹角
            held = mean.detach()
            gain = torch.tensor(self.back_follow, device=mean.device, dtype=mean.dtype)
            teacher[:, [1, 3]] = held[:, [1, 3]] + gain * (held[:, [0, 2]] - teacher[:, [0, 2]])
        per_sample = torch.nn.functional.huber_loss(
            mean, teacher, reduction="none", delta=self.huber_delta
        ).mean(dim=1)
        weighted = (per_sample * alpha * active).sum() / active.sum().clamp(min=1.0)
        return self.coef * weighted


class _PolicyDistillLoss(_ImitationLoss):
    """网络老师在线蒸馏（见模块 docstring）：目标动作由冻结老师在同一批观测上前向给出，只在掩码组样本上求均值。"""

    def __init__(
        self,
        *,
        teacher: torch.nn.Module,
        mask_group: str,
        role_group: str | None,
        coef: float,
        huber_delta: float,
        front_period: float,
        back_period: float,
        shift_back: bool,
    ) -> None:
        super().__init__(
            group=mask_group,
            coef=coef,
            huber_delta=huber_delta,
            front_period=front_period,
            back_period=back_period,
            shift_back=shift_back,
        )
        self.teacher = teacher
        self.role_group = role_group

    @torch.no_grad()
    def teacher_mean(self, observations: Any) -> torch.Tensor:
        """老师按部署路径的确定性动作（CTS 角色组置 0 走学生编码器）。"""
        obs = observations
        if self.role_group is not None and self.role_group in observations.keys():  # noqa: SIM118
            obs = observations.clone(recurse=False)
            obs[self.role_group] = torch.zeros_like(observations[self.role_group])
        return self.teacher(obs)

    def compute_loss(self, actor: Any, batch: Any, original_batch_size: int) -> torch.Tensor:
        mean = actor.output_mean[:original_batch_size]
        observations = batch.observations[:original_batch_size]
        mask = (observations[self.group].reshape(-1) > 0.5).float()
        teacher = self.align(self.teacher_mean(observations), mean)
        per_sample = torch.nn.functional.huber_loss(
            mean, teacher, reduction="none", delta=self.huber_delta
        ).mean(dim=1)
        return self.coef * (per_sample * mask).sum() / mask.sum().clamp(min=1.0)


class Se3PPO(PPO):
    """rsl_rl.PPO 的透明扩展：可选的固定 critic 学习率、显式 vx 估计器监督、模仿损失 / 网络老师蒸馏与 AMP 风格奖励。"""

    def __init__(
        self,
        *args: Any,
        critic_learning_rate: float | None = None,
        estimator_learning_rate: float | None = None,
        estimator_target_group: str | None = None,
        estimator_warmup_iterations: int = 0,
        imitation_loss: dict[str, Any] | None = None,
        policy_distill: dict[str, Any] | None = None,
        amp_cfg: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if policy_distill is not None:
            if imitation_loss is not None or self.symmetry is not None:
                raise ValueError(
                    "网络老师蒸馏借用 symmetry 钩子，不能与 imitation_loss / symmetry_cfg 同时使用"
                )
            self.symmetry = self._build_policy_distill(dict(policy_distill))
        self.amp: AMP | None = None
        if amp_cfg:
            amp_cfg = dict(amp_cfg)
            step_dt = amp_cfg.pop("step_dt", None)
            if step_dt is None:
                raise ValueError(
                    "amp_cfg 缺少 step_dt；应由 Se3PPO.construct_algorithm 从 env 填入"
                )
            self.amp = AMP(
                num_envs=int(self.storage.observations.shape[1]),
                step_dt=float(step_dt),
                obs=self.storage.observations[0],
                cfg=amp_cfg,
                device=str(self.device),
                multi_gpu=self.is_multi_gpu,
            )
        if imitation_loss is not None:
            if self.symmetry is not None:
                raise ValueError("模仿损失借用 symmetry 钩子，不能与 symmetry_cfg 同时使用")
            self.symmetry = _ImitationLoss(**imitation_loss)
        self.estimator_warmup_remaining = int(estimator_warmup_iterations)
        self._noise_sq_small: list[float] = []
        self._noise_sq_big: list[float] = []
        self.estimator_target_group = estimator_target_group
        self.estimator_optimizer: torch.optim.Adam | None = None
        if estimator_learning_rate is not None:
            if not hasattr(self._raw_actor, "estimate_vx"):
                raise ValueError(
                    "estimator_learning_rate 需要 actor 带估计器（CTSVxObserverModel）"
                )
            if estimator_target_group is None:
                raise ValueError("estimator_learning_rate 需要同时给 estimator_target_group")
            self.estimator_optimizer = torch.optim.Adam(
                self._raw_actor.estimator.parameters(), lr=float(estimator_learning_rate)
            )
        if self.estimator_warmup_remaining > 0 and self.estimator_optimizer is None:
            raise ValueError("estimator_warmup_iterations 需要同时给 estimator_learning_rate")
        self.critic_learning_rate = (
            None if critic_learning_rate is None else float(critic_learning_rate)
        )
        if self.critic_learning_rate is None:
            return
        if kwargs.get("optimizer", "adam") != "adam":
            raise ValueError("critic_learning_rate 目前只支持 optimizer='adam'")
        if self.critic_learning_rate <= 0.0:
            raise ValueError(f"critic_learning_rate 必须为正数，收到 {self.critic_learning_rate}")
        self.optimizer = ActorCriticAdam(
            self.actor.parameters(),
            self.critic.parameters(),
            lr=self.learning_rate,
            critic_lr=self.critic_learning_rate,
        )

    def _build_policy_distill(self, cfg: dict[str, Any]) -> _PolicyDistillLoss:
        """复制一份本 actor 当老师骨架，载入 checkpoint 的 actor_state_dict（形状必须逐项一致），冻结。"""
        path = str(cfg.pop("teacher_checkpoint"))
        teacher = copy.deepcopy(self._raw_actor)
        state = torch.load(path, map_location=self.device, weights_only=False)["actor_state_dict"]
        teacher.load_state_dict(state, strict=True)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        print(f"[Se3PPO] 网络老师蒸馏：老师 {path}，掩码组 {cfg['mask_group']}，系数 {cfg['coef']}")
        return _PolicyDistillLoss(teacher=teacher, **cfg)

    def _actor_grad_sq(self) -> torch.Tensor:
        grads = [p.grad.reshape(-1) for p in self.actor.parameters() if p.grad is not None]
        return torch.cat(grads).square().sum()

    def reduce_parameters(self) -> None:
        """多卡梯度平均前后各取一次 actor 梯度模长平方，供 update() 汇总噪声尺度。"""
        sq_small = self._actor_grad_sq()
        torch.distributed.all_reduce(sq_small, op=torch.distributed.ReduceOp.SUM)
        sq_small /= self.gpu_world_size
        super().reduce_parameters()
        self._noise_sq_small.append(float(sq_small))
        self._noise_sq_big.append(float(self._actor_grad_sq()))

    # ---- AMP 钩子 ----
    @staticmethod
    def construct_algorithm(obs: Any, env: Any, cfg: dict, device: str) -> PPO:
        """在 PPO.construct_algorithm 之前把 env 的控制周期填进 amp_cfg。"""
        amp = cfg.get("algorithm", {}).get("amp_cfg")
        if amp:
            unwrapped = getattr(env, "unwrapped", env)
            amp.setdefault("step_dt", float(unwrapped.step_dt))
        return PPO.construct_algorithm(obs, env, cfg, device)

    def process_env_step(
        self, obs: Any, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
    ) -> None:
        if self.amp is None:
            super().process_env_step(obs, rewards, dones, extras)
            return
        # 照抄 rsl_rl 5.4.0 PPO.process_env_step，在 bootstrap 之前插入 AMP 钩子（与 fork 一致）。
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards
        self.amp.process_env_step(obs, self.transition, extras)
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
                1,
            )
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def train_mode(self) -> None:
        super().train_mode()
        if self.amp is not None:
            self.amp.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        if self.amp is not None:
            self.amp.eval()

    def update(self) -> dict[str, float]:
        """原版 update；多卡时附带本轮 actor 梯度噪声尺度。预热期只训练估计器。判别器在 PPO 之前更新。"""
        amp_metrics = self.amp.individual_update(self.storage) if self.amp is not None else {}
        if self.estimator_warmup_remaining > 0:
            self.estimator_warmup_remaining -= 1
            loss_dict = self._update_estimator()
            loss_dict["estimator_warmup"] = 1.0
            loss_dict.update(amp_metrics)
            self.storage.clear()
            return loss_dict
        b_small = (
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        )
        self._noise_sq_small.clear()
        self._noise_sq_big.clear()
        loss_dict = super().update()
        if isinstance(self.symmetry, _ImitationLoss) and "symmetry" in loss_dict:
            loss_dict["imitation"] = loss_dict.pop("symmetry")
        if self.is_multi_gpu and self._noise_sq_small:
            b_big = b_small * self.gpu_world_size
            sq_small = sum(self._noise_sq_small) / len(self._noise_sq_small)
            sq_big = sum(self._noise_sq_big) / len(self._noise_sq_big)
            grad_sq = (b_big * sq_big - b_small * sq_small) / (b_big - b_small)
            trace = (sq_small - sq_big) / (1.0 / b_small - 1.0 / b_big)
            loss_dict["grad_sq_small_actor"] = sq_small
            loss_dict["grad_sq_big_actor"] = sq_big
            loss_dict["grad_noise_scale_actor"] = trace / grad_sq if grad_sq > 0.0 else float("nan")
        if self.estimator_optimizer is not None:
            loss_dict.update(self._update_estimator())
        loss_dict.update(amp_metrics)
        return loss_dict

    def _update_estimator(self) -> dict[str, float]:
        """用本轮 rollout 对 vx 估计器做 MSE 回归；多卡时梯度跨卡平均，保证各卡估计器一致。"""
        assert self.estimator_optimizer is not None
        actor = self._raw_actor
        params = list(actor.estimator.parameters())
        custom_loss = getattr(actor, "estimator_loss", None)
        mse_sum = torch.zeros((), device=self.device)
        extra_sums: dict[str, torch.Tensor] = {}
        num_updates = 0
        for batch in self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        ):
            if custom_loss is not None:
                loss, extra = custom_loss(batch.observations)
                for key, value in extra.items():
                    extra_sums[key] = extra_sums.get(key, 0.0) + value
            else:
                target = batch.observations[self.estimator_target_group]
                loss = torch.nn.functional.mse_loss(actor.estimate_vx(batch.observations), target)
            self.estimator_optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                grads = torch.cat([p.grad.reshape(-1) for p in params])
                torch.distributed.all_reduce(grads, op=torch.distributed.ReduceOp.SUM)
                grads /= self.gpu_world_size
                offset = 0
                for p in params:
                    p.grad.copy_(grads[offset : offset + p.numel()].view_as(p))
                    offset += p.numel()
            self.estimator_optimizer.step()
            mse_sum += loss.detach()
            num_updates += 1
        # 梯度置 None，下一轮 PPO 的噪声尺度统计与 reduce_parameters 只看到 PPO 自己的梯度。
        self.estimator_optimizer.zero_grad(set_to_none=True)
        mse = float(mse_sum) / max(num_updates, 1)
        log = {"estimator_vx_mse": mse, "estimator_vx_rmse": mse**0.5}
        if custom_loss is not None:
            log = {"estimator_mse": mse}
            for key, value in extra_sums.items():
                log[key] = float(value) / max(num_updates, 1)
            if "estimator_vx_mse" in log:
                log["estimator_vx_rmse"] = log["estimator_vx_mse"] ** 0.5
        role_group = getattr(actor, "role_group", None)
        if role_group is not None:
            teacher = self.storage.observations[role_group] > 0.5
            rewards = self.storage.rewards
            log["cts_reward_teacher"] = float(rewards[teacher].mean())
            log["cts_reward_student"] = float(rewards[~teacher].mean())
        if hasattr(actor, "latent_std"):
            log["estimator_latent_std"] = actor.latent_std(batch.observations)
        return log

    def save(self) -> dict:
        saved_dict = super().save()
        if self.estimator_optimizer is not None:
            saved_dict["estimator_optimizer_state_dict"] = self.estimator_optimizer.state_dict()
        if self.amp is not None:
            saved_dict["ext_state_dict"] = {"amp": self.amp.save()}
        return saved_dict

    def _estimator_shape_changed(self, actor_state: dict) -> bool:
        """checkpoint 的 estimator.* 与当前 actor 的形状或键集合是否不同（换学生编码器续训）。"""
        if self.estimator_optimizer is None:
            return False
        current = {
            k: v.shape
            for k, v in self._raw_actor.state_dict().items()
            if k.startswith("estimator.")
        }
        saved = {k: v.shape for k, v in actor_state.items() if k.startswith("estimator.")}
        return current != saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
            }
        reset_estimator = bool(load_cfg.get("actor")) and self._estimator_shape_changed(
            loaded_dict["actor_state_dict"]
        )
        if reset_estimator:
            state = {
                k: v
                for k, v in loaded_dict["actor_state_dict"].items()
                if not k.startswith("estimator.")
            }
            missing, unexpected = self._raw_actor.load_state_dict(state, strict=False)
            bad = [k for k in missing if not k.startswith("estimator.")] + list(unexpected)
            if bad:
                raise RuntimeError(f"换学生编码器续训只允许 estimator.* 缺失，实际还缺 / 多：{bad}")
            print(
                "[Se3PPO] checkpoint 的学生编码器形状与当前不同：estimator.* 随机初始化，其余照常加载"
            )
            load_cfg = {**load_cfg, "actor": False}
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        load_optimizer = bool(load_cfg.get("optimizer"))
        if (
            self.estimator_optimizer is not None
            and load_optimizer
            and not reset_estimator
            and "estimator_optimizer_state_dict" in loaded_dict
        ):
            self.estimator_optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
        if self.amp is not None and load_cfg.get("ext", True):
            state = loaded_dict.get("ext_state_dict", {}).get("amp")
            if state is not None:
                self.amp.load(state, strict=strict)
        return load_iteration


__all__ = ["ActorCriticAdam", "Se3PPO"]
