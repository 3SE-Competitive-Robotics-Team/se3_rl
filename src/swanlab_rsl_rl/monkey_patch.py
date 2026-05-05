"""SwanLab 集成模块。

通过 monkey patch 为 rsl_rl 添加 SwanLab 支持。
"""

from __future__ import annotations

from typing import Any

from rsl_rl.utils import logger as rsl_rl_logger


def patch_rsl_rl_logger() -> None:
    """Monkey patch rsl_rl Logger 以支持 SwanLab。

    在 rsl_rl 的 Logger.init_logging_writer 方法中添加 swanlab 支持。
    """

    def patched_init_logging_writer(self: Any) -> None:
        """支持 swanlab 的 init_logging_writer 实现。"""
        if self.log_dir is not None and not self.disable_logs:
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "swanlab":
                from swanlab_rsl_rl.swanlab_utils import SwanLabSummaryWriter

                self.writer = SwanLabSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
            elif self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError(
                    "Logger type not found. Please choose 'swanlab', 'wandb', 'neptune', or 'tensorboard'."
                )
        else:
            self.writer = None

        # 保存代码状态
        files_to_upload = self._store_code_state()

        # 上传配置和代码状态到外部日志服务
        if self.writer is not None and self.logger_type in [
            "wandb",
            "neptune",
            "swanlab",
        ]:
            self.writer.store_config(self.env_cfg, self.cfg)
            for path in files_to_upload:
                self.writer.save_file(path)

    rsl_rl_logger.Logger.init_logging_writer = patched_init_logging_writer
