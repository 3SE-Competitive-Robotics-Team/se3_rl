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
from torch.utils.tensorboard import SummaryWriter

_DEFAULT_WANDB_INIT_TIMEOUT_S = 30.0
_DEFAULT_WANDB_CALL_TIMEOUT_S = 2.0
_WANDB_CLEANUP_TIMEOUT_S = 5.0


class _WandbInitTimeoutError(TimeoutError):
    """W&B 初始化超过允许时间。"""


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
