"""SE3 训练 runner 扩展。"""

from __future__ import annotations

import os
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from mjlab.rl import MjlabOnPolicyRunner
from rsl_rl.utils import check_nan
from torch.utils.tensorboard import SummaryWriter

from se3_train.async_logging import Se3AsyncHostLogger, async_host_logger_enabled
from se3_train.training_runtime import (
    IterationTimer,
    IterationTiming,
    detect_training_runtime,
    format_runtime_summary,
    write_training_status,
)

_DEFAULT_WANDB_INIT_TIMEOUT_S = 30.0
_DEFAULT_WANDB_CALL_TIMEOUT_S = 2.0
_WANDB_CLEANUP_TIMEOUT_S = 5.0


class _WandbInitTimeoutError(TimeoutError):
    """W&B 初始化超过允许时间。"""


def _env_flag(name: str, default: bool) -> bool:
    """读取布尔环境变量。"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@contextmanager
def _deadline(seconds: float) -> Iterator[None]:
    """用 SIGALRM 给同步初始化设置硬超时。"""
    if (
        seconds <= 0.0
        or not hasattr(signal, "setitimer")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    def _raise_timeout(_signum: int, _frame: object | None) -> None:
        raise _WandbInitTimeoutError(f"W&B 初始化超过 {seconds:.1f}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _wandb_init_timeout_s(cfg: dict) -> float:
    """读取 W&B 初始化超时时间，默认 30 秒。"""
    raw_timeout = os.environ.get(
        "SE3_WANDB_INIT_TIMEOUT_S",
        cfg.get("wandb_init_timeout_s", _DEFAULT_WANDB_INIT_TIMEOUT_S),
    )
    try:
        return max(0.0, float(raw_timeout))
    except (TypeError, ValueError):
        print(
            "[WARN] SE3_WANDB_INIT_TIMEOUT_S 无法解析，"
            f"使用默认值 {_DEFAULT_WANDB_INIT_TIMEOUT_S:.1f}s。"
        )
        return _DEFAULT_WANDB_INIT_TIMEOUT_S


def _wandb_call_timeout_s(cfg: dict) -> float:
    """读取 W&B 单次写入超时时间，默认 2 秒。"""
    raw_timeout = os.environ.get(
        "SE3_WANDB_CALL_TIMEOUT_S",
        cfg.get("wandb_call_timeout_s", _DEFAULT_WANDB_CALL_TIMEOUT_S),
    )
    try:
        return max(0.0, float(raw_timeout))
    except (TypeError, ValueError):
        print(
            "[WARN] SE3_WANDB_CALL_TIMEOUT_S 无法解析，"
            f"使用默认值 {_DEFAULT_WANDB_CALL_TIMEOUT_S:.1f}s。"
        )
        return _DEFAULT_WANDB_CALL_TIMEOUT_S


def _cleanup_partial_wandb_run() -> None:
    """清理超时后可能残留的 W&B 后台状态。"""
    try:
        with _deadline(_WANDB_CLEANUP_TIMEOUT_S):
            import wandb

            if wandb.run is not None:
                wandb.finish(exit_code=1, quiet=True)
            teardown = getattr(wandb, "teardown", None)
            if callable(teardown):
                teardown()
    except Exception as exc:
        print(f"[WARN] W&B 超时清理失败，继续使用本地 TensorBoard：{exc}")


class _GuardedWandbWriter:
    """给 W&B writer 增加运行期超时和本地 TensorBoard 降级。"""

    def __init__(self, writer: Any, log_dir: str, cfg: dict) -> None:
        """保存外部 writer 和降级所需配置。"""
        self._writer = writer
        self._log_dir = log_dir
        self._timeout_s = _wandb_call_timeout_s(cfg)
        self._fallback_writer: SummaryWriter | None = None
        self._disabled = False

    def _fallback(self) -> SummaryWriter:
        """按需创建降级后的本地 TensorBoard writer。"""
        if self._fallback_writer is None:
            self._fallback_writer = SummaryWriter(log_dir=self._log_dir, flush_secs=10)
        return self._fallback_writer

    def _disable_wandb(self, reason: Exception) -> None:
        """停用 W&B，后续只写本地 TensorBoard。"""
        if self._disabled:
            return
        self._disabled = True
        print(
            f"[WARN] W&B 写入超时或失败，已停用在线日志；后续继续写本地 TensorBoard。原因：{reason}"
        )
        _cleanup_partial_wandb_run()

    def _call_wandb(self, method_name: str, *args: Any, **kwargs: Any) -> bool:
        """带超时调用 W&B writer 方法，失败时停用 W&B。"""
        if self._disabled:
            return False
        try:
            with _deadline(self._timeout_s):
                method = getattr(self._writer, method_name)
                method(*args, **kwargs)
            return True
        except Exception as exc:
            self._disable_wandb(exc)
            return False

    def store_config(self, *args: Any, **kwargs: Any) -> None:
        """上传配置，失败后切本地日志。"""
        self._call_wandb("store_config", *args, **kwargs)

    def save_file(self, *args: Any, **kwargs: Any) -> None:
        """上传文件，失败后切本地日志。"""
        self._call_wandb("save_file", *args, **kwargs)

    def save_model(self, *args: Any, **kwargs: Any) -> None:
        """上传模型，失败后切本地日志。"""
        self._call_wandb("save_model", *args, **kwargs)

    def save_video(self, *args: Any, **kwargs: Any) -> None:
        """上传视频，失败后切本地日志。"""
        self._call_wandb("save_video", *args, **kwargs)

    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        """写标量；W&B 不可用后继续写 TensorBoard。"""
        if not self._call_wandb("add_scalar", *args, **kwargs):
            self._fallback().add_scalar(*args, **kwargs)

    def stop(self) -> None:
        """结束 W&B run，并关闭降级 writer。"""
        if not self._disabled:
            self._call_wandb("stop")
        if self._fallback_writer is not None:
            self._fallback_writer.close()

    def __getattr__(self, name: str) -> Any:
        """透传未覆盖的 SummaryWriter 方法。"""
        if self._disabled:
            return getattr(self._fallback(), name)
        return getattr(self._writer, name)


def _init_tensorboard_fallback(logger: Any, reason: Exception) -> None:
    """W&B 不可用时降级到本地 TensorBoard，并保留 checkpoint 保存。"""
    print(
        "[WARN] W&B 初始化失败，已降级为本地 TensorBoard；"
        f"训练和 checkpoint 保存会继续。原因：{reason}"
    )
    logger.logger_type = "tensorboard"
    logger.writer = SummaryWriter(log_dir=logger.log_dir, flush_secs=10)
    logger._store_code_state()  # type: ignore[attr-defined]


def _init_logging_writer_with_wandb_guard(logger: Any) -> None:
    """初始化日志 writer，并给 W&B 初始化加硬超时保护。"""
    if logger.log_dir is None or logger.disable_logs:
        logger.writer = None
        return

    logger_type = logger.cfg.get("logger", "tensorboard").lower()
    logger.logger_type = logger_type
    if logger_type != "wandb":
        from rsl_rl.utils.logger import Logger

        Logger.init_logging_writer(logger)  # type: ignore[arg-type]
        return

    try:
        with _deadline(_wandb_init_timeout_s(logger.cfg)):
            from rsl_rl.utils.wandb_utils import WandbSummaryWriter

            writer = WandbSummaryWriter(
                log_dir=logger.log_dir,
                flush_secs=10,
                cfg=logger.cfg,
            )
            logger.writer = _GuardedWandbWriter(writer, logger.log_dir, logger.cfg)

            files_to_upload = logger._store_code_state()  # type: ignore[attr-defined]
            logger.writer.store_config(logger.env_cfg, logger.cfg)
            for path in files_to_upload:
                logger.writer.save_file(path)
    except Exception as exc:
        _cleanup_partial_wandb_run()
        _init_tensorboard_fallback(logger, exc)


class Se3OnPolicyRunner(MjlabOnPolicyRunner):
    """SE3 默认 runner，保护训练不被 W&B 网络初始化阻塞。"""

    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        """构建 runner，并安装 W&B 初始化超时保护。"""
        super().__init__(env, train_cfg, log_dir, device)
        self.logger.init_logging_writer = (  # type: ignore[method-assign]
            lambda: _init_logging_writer_with_wandb_guard(self.logger)
        )


class Se3ProfiledOnPolicyRunner(MjlabOnPolicyRunner):
    """带 SE3 运行时画像的 MJLab on-policy runner。"""

    def __init__(self, *args, **kwargs) -> None:
        """初始化 runner，并采集一次训练运行时资源快照。"""
        super().__init__(*args, **kwargs)
        self._se3_runtime_info = detect_training_runtime()
        self._se3_runtime_info_logged = False
        self._se3_async_host_logger_enabled = async_host_logger_enabled()
        self._se3_last_async_logger_flush_s = 0.0
        self._se3_check_nan_enabled = _env_flag(
            "SE3_CHECK_NAN",
            os.environ.get("SE3_SMOKE", "0") == "1",
        )
        if not self.is_distributed or self.gpu_global_rank == 0:
            print(format_runtime_summary(self._se3_runtime_info), flush=True)
            if self._se3_async_host_logger_enabled:
                print("[SE3 Runtime] async_host_logger=enabled", flush=True)
            print(
                f"[SE3 Runtime] check_nan={'enabled' if self._se3_check_nan_enabled else 'disabled'}",
                flush=True,
            )

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
        async_logger = (
            Se3AsyncHostLogger(self.logger) if self._se3_async_host_logger_enabled else None
        )

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        num_steps_per_env = int(self.cfg["num_steps_per_env"])
        for it in range(start_it, total_it):
            timer = IterationTimer(self.env.num_envs, num_steps_per_env)
            with torch.inference_mode():
                for _ in range(num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self._se3_check_nan_enabled and self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = (
                        self.alg.intrinsic_rewards if self.cfg["algorithm"].get("rnd_cfg") else None
                    )
                    if async_logger is None:
                        self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
                    else:
                        async_logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                timer.mark_collect_done()
                self.alg.compute_returns(obs)
                timer.mark_returns_done()

            loss_dict = self.alg.update()
            timing = timer.finish()
            self._se3_last_async_logger_flush_s = (
                async_logger.flush() if async_logger is not None else 0.0
            )
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
            if not self.is_distributed or self.gpu_global_rank == 0:
                write_training_status(
                    self.logger.log_dir,  # type: ignore[arg-type]
                    iteration=it,
                    total_iterations=total_it,
                    timing=timing,
                )

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
        writer.add_scalar(
            "Perf/async_host_logger_flush_s", self._se3_last_async_logger_flush_s, iteration
        )
        if not self._se3_runtime_info_logged:
            for key, value in self._se3_runtime_info.as_log_dict().items():
                writer.add_scalar(key, value, iteration)
            writer.add_scalar(
                "Runtime/async_host_logger_enabled",
                float(self._se3_async_host_logger_enabled),
                iteration,
            )
            writer.add_scalar(
                "Runtime/check_nan_enabled",
                float(self._se3_check_nan_enabled),
                iteration,
            )
            self._se3_runtime_info_logged = True


class Se3WarmStartRunner(Se3OnPolicyRunner):
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
