from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from body_mvp.config import settings

# Paired with checkpoints/sam2/sam2.1_hiera_small.pt
_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"


def _make_box_prompt(image: np.ndarray) -> np.ndarray:
    """Returns [x1, y1, x2, y2] heuristic center crop: 80% width, 90% height."""
    h, w = image.shape[:2]
    return np.array(
        [int(w * 0.10), int(h * 0.05), int(w * 0.90), int(h * 0.95)],
        dtype=np.float32,
    )


def segment_keyframes(keyframe_paths: list[Path], run_dir: Path) -> list[Path]:
    """SAM 2 person mask per keyframe. Returns list of mask PNG paths."""
    masks_dir = run_dir / "masks"

    model = build_sam2(_SAM2_CONFIG, str(settings.sam2_checkpoint), device=settings.device)
    predictor = SAM2ImagePredictor(model)
    logger.info("SAM 2 loaded: {}", settings.sam2_checkpoint)

    saved: list[Path] = []
    for kf_path in keyframe_paths:
        img_bgr = cv2.imread(str(kf_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        predictor.set_image(img_rgb)
        masks, scores, _ = predictor.predict(
            box=_make_box_prompt(img_rgb), multimask_output=True
        )

        best = int(np.argmax(scores))
        mask = masks[best].astype(bool)

        idx = kf_path.stem.split("_")[-1]  # "keyframe_03" -> "03"
        mask_path = masks_dir / f"mask_{idx}.png"
        cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))

        overlay = img_bgr.copy()
        green = np.zeros_like(img_bgr)
        green[mask] = (0, 200, 0)
        cv2.imwrite(
            str(masks_dir / f"overlay_{idx}.png"),
            cv2.addWeighted(overlay, 0.6, green, 0.4, 0),
        )

        logger.info(
            "mask_{}.png: score={:.4f}, coverage={:.1f}%",
            idx, scores[best], 100.0 * mask.sum() / mask.size,
        )
        saved.append(mask_path)

    return saved


def run(video_path: Path, height: float, weight: float, gender: str) -> None:
    logger.info("Stage 1 not implemented yet")


def sample_keyframes(*args, **kwargs):
    """M2: TODO"""
    raise NotImplementedError("M2")


def estimate_smpl_params(*args, **kwargs):
    """M4: TODO"""
    raise NotImplementedError("M4")


def detect_keypoints(*args, **kwargs):
    """M5: TODO"""
    raise NotImplementedError("M5")


def predict_normals(*args, **kwargs):
    """M5: TODO"""
    raise NotImplementedError("M5")
