"""SwanLab 集成工具。

为 rsl_rl 提供 SwanLab 日志记录支持。
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import asdict
from typing import Any

import swanlab
from torch.utils.tensorboard import SummaryWriter


class SwanLabSummaryWriter(SummaryWriter):
    """SwanLab 的 SummaryWriter 实现。

    继承 TensorBoard SummaryWriter，同时将指标同步到 SwanLab。
    """

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict[str, Any]) -> None:
        """初始化 SwanLab run。

        Args:
            log_dir: 日志目录路径
            flush_secs: 刷新间隔（秒）
            cfg: 训练配置字典
        """
        super().__init__(log_dir, flush_secs=flush_secs)

        # 从日志目录获取 run 名称
        run_name = os.path.split(log_dir)[-1]

        # 从环境变量或配置中获取项目信息
        project = cfg.get("swanlab_project", cfg.get("wandb_project", "se3_wheel_leg"))
        workspace = os.environ.get("SWANLAB_USERNAME", None)

        # 检查是否配置了 API key
        api_key = os.environ.get("SWANLAB_API_KEY", None)

        # 初始化 SwanLab
        if api_key:
            # 使用 API key 登录
            swanlab.login(api_key=api_key)
            swanlab.init(
                project=project,
                workspace=workspace,
                name=run_name,
                config={"log_dir": log_dir},
            )
        else:
            # 使用 local 模式（不上传到云端）
            swanlab.init(
                project=project,
                workspace=workspace,
                name=run_name,
                config={"log_dir": log_dir},
                mode="local",
            )

        # 记录已上传的视频文件
        self.logged_videos: set[str] = set()

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        """上传环境和训练配置到 SwanLab。

        Args:
            env_cfg: 环境配置
            train_cfg: 训练配置
        """
        config_update = {"train_cfg": train_cfg}
        try:
            config_update["env_cfg"] = env_cfg.to_dict()
        except Exception:
            try:
                config_update["env_cfg"] = asdict(env_cfg)
            except Exception:
                config_update["env_cfg"] = str(env_cfg)

        swanlab.config.update(config_update)

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        """记录标量到 TensorBoard 和 SwanLab。

        Args:
            tag: 指标名称
            scalar_value: 指标值
            global_step: 全局步数
            walltime: 墙钟时间
            new_style: 是否使用新样式
        """
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        swanlab.log({tag: scalar_value}, step=global_step)

    def stop(self) -> None:
        """结束 SwanLab run。"""
        swanlab.finish()

    def save_model(self, model_path: str, it: int) -> None:
        """上传模型 checkpoint 到 SwanLab。

        Args:
            model_path: 模型文件路径
            it: 迭代次数
        """
        swanlab.save(model_path)

    def save_file(self, path: str) -> None:
        """上传任意文件到 SwanLab。

        Args:
            path: 文件路径
        """
        swanlab.save(path)

    def save_video(self, video: pathlib.Path, it: int) -> None:
        """上传视频到 SwanLab（每个文件只上传一次）。

        Args:
            video: 视频文件路径
            it: 迭代次数
        """
        if video.name not in self.logged_videos:
            # SwanLab 支持视频记录
            swanlab.log({"video": swanlab.Video(str(video))}, step=it)
            self.logged_videos.add(video.name)
