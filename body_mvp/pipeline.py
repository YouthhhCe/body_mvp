from pathlib import Path

from loguru import logger

from body_mvp import stage1, stage2, stage3


def run(video_path: Path, height: float, weight: float, gender: str) -> None:
    logger.info("Starting pipeline: video={}, height={}, weight={}, gender={}", video_path, height, weight, gender)
    stage1.run(video_path, height, weight, gender)
    stage2.run()
    stage3.run()
