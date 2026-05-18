from pathlib import Path

from loguru import logger


def run(video_path: Path, height: float, weight: float, gender: str) -> None:
    logger.info("Stage 1 not implemented yet")


def sample_keyframes(*args, **kwargs):
    """M2: TODO"""
    raise NotImplementedError("M2")


def segment_keyframes(*args, **kwargs):
    """M3: TODO"""
    raise NotImplementedError("M3")


def estimate_smpl_params(*args, **kwargs):
    """M4: TODO"""
    raise NotImplementedError("M4")


def detect_keypoints(*args, **kwargs):
    """M5: TODO"""
    raise NotImplementedError("M5")


def predict_normals(*args, **kwargs):
    """M5: TODO"""
    raise NotImplementedError("M5")
