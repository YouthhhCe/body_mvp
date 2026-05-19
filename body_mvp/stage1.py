from pathlib import Path

import cv2
import numpy as np
import torch
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.models import DEFAULT_CHECKPOINT, load_hmr2
from hmr2.utils import recursive_to
from hmr2.utils.renderer import Renderer, cam_crop_to_full
from loguru import logger
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from body_mvp.config import settings

# Paired with checkpoints/sam2/sam2.1_hiera_small.pt
_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

# Minimum mask coverage to use mask-derived bbox; below this
# we fall back to the M3 heuristic box. Threshold from M3
# run 20260519_160131 on test.mp4: good frames 11.3-14.6%,
# failed frames (03,04,08,09) 4.2-6.9%. 8% sits in the middle.
_MASK_COVERAGE_THRESHOLD = 0.08


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


def _mask_to_bbox(mask: np.ndarray, pad: float = 0.15) -> np.ndarray | None:
    """Bbox [x1, y1, x2, y2] from mask non-zero pixels, expanded by pad on each side.

    Returns None if coverage is below _MASK_COVERAGE_THRESHOLD.
    """
    if (mask > 0).mean() < _MASK_COVERAGE_THRESHOLD:
        return None
    ys, xs = np.where(mask > 0)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw, bh = x2 - x1, y2 - y1
    h, w = mask.shape[:2]
    x1 = max(0, x1 - int(bw * pad))
    y1 = max(0, y1 - int(bh * pad))
    x2 = min(w, x2 + int(bw * pad))
    y2 = min(h, y2 + int(bh * pad))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def extract_smpl_params(
    keyframe_paths: list[Path],
    mask_paths: list[Path],
    run_dir: Path,
) -> list[Path]:
    """4D Humans inference per keyframe. Returns list of .npz paths."""
    out_dir = run_dir / "smpl_params"
    out_dir.mkdir(exist_ok=True)

    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.to(settings.device)
    model.eval()
    logger.info("HMR2 loaded: {}", DEFAULT_CHECKPOINT)

    saved: list[Path] = []

    for kf_path, mask_path in zip(keyframe_paths, mask_paths):
        idx = kf_path.stem.split("_")[-1]  # "keyframe_03" -> "03"

        img_bgr = cv2.imread(str(kf_path))

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        bbox = _mask_to_bbox(mask)
        if bbox is not None:
            bbox_source = "mask-derived"
        else:
            bbox = _make_box_prompt(img_bgr)
            bbox_source = "heuristic-fallback"
        logger.info(
            "frame {}: bbox_source={}, bbox=[{:.0f},{:.0f},{:.0f},{:.0f}]",
            idx, bbox_source, bbox[0], bbox[1], bbox[2], bbox[3],
        )

        dataloader = torch.utils.data.DataLoader(
            ViTDetDataset(model_cfg, img_bgr, bbox[None]),
            batch_size=1, shuffle=False, num_workers=0,
        )
        batch = next(iter(dataloader))
        batch = recursive_to(batch, settings.device)

        try:
            with torch.no_grad():
                out = model(batch)
        except Exception:
            logger.exception("inference failed on frame {}", idx)
            raise

        body_pose_aa = matrix_to_axis_angle(
            out["pred_smpl_params"]["body_pose"][0]  # [23, 3, 3]
        ).cpu().numpy()  # [23, 3]
        global_orient_aa = matrix_to_axis_angle(
            out["pred_smpl_params"]["global_orient"][0]  # [1, 3, 3]
        ).squeeze(0).cpu().numpy()  # [3]
        betas = out["pred_smpl_params"]["betas"][0].cpu().numpy()  # [10]
        pred_cam = out["pred_cam"][0].cpu().numpy()  # [3]
        # out['focal_length'] is cfg.EXTRA.FOCAL_LENGTH * ones(batch, 2); fx == fy always
        focal_length = out["focal_length"][0].cpu().numpy()  # [2]

        # Single-frame inference; img_size.max() collapses batch+dim
        # correctly only when batch=1. Don't reuse this block in batched code.
        assert batch["img"].shape[0] == 1, "extract_smpl_params is per-frame"
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_fl = focal_length[0] / model_cfg.MODEL.IMAGE_SIZE * img_size.max().item()
        pred_cam_t = cam_crop_to_full(
            out["pred_cam"], box_center, box_size, img_size, scaled_fl
        )[0].cpu().numpy()  # [3]

        npz_path = out_dir / f"smpl_{idx}.npz"
        np.savez(
            str(npz_path),
            betas=betas,
            body_pose=body_pose_aa,
            global_orient=global_orient_aa,
            pred_cam=pred_cam,
            pred_cam_t=pred_cam_t,
            focal_length=focal_length,
            bbox=bbox,
            bbox_source=np.array(bbox_source),
        )
        betas_norm = float(np.linalg.norm(betas))
        pose_mean_mag = float(np.linalg.norm(body_pose_aa, axis=1).mean())
        logger.info(
            "smpl_{}.npz saved (betas_norm={:.2f}, pose_mean_mag={:.2f})",
            idx, betas_norm, pose_mean_mag,
        )
        saved.append(npz_path)

    return saved


def render_smpl_overlays(
    keyframe_paths: list[Path],
    npz_paths: list[Path],
    run_dir: Path,
) -> list[Path]:
    """SMPL mesh overlaid on keyframes for M4 acceptance review.

    Reads .npz files written by extract_smpl_params; does not call it.
    Not called by the pipeline.
    """
    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.to(settings.device)
    model.eval()

    renderer = Renderer(model_cfg, faces=model.smpl.faces)
    mesh_color = (0.65098039, 0.74117647, 0.85882353)  # light blue, matches demo.py

    out_dir = run_dir / "smpl_params"
    out_dir.mkdir(exist_ok=True)
    saved: list[Path] = []

    for kf_path, npz_path in zip(keyframe_paths, npz_paths):
        idx = kf_path.stem.split("_")[-1]

        img_bgr = cv2.imread(str(kf_path))
        img_h, img_w = img_bgr.shape[:2]

        data = np.load(str(npz_path), allow_pickle=True)
        betas            = data["betas"]            # [10]
        body_pose_aa     = data["body_pose"]         # [23, 3]
        global_orient_aa = data["global_orient"]     # [3]
        pred_cam_t       = data["pred_cam_t"]        # [3]
        focal_length     = data["focal_length"]      # [2]

        # Axis-angle → rotation matrix → model.smpl (same pose2rot=False path as hmr2)
        device = next(model.parameters()).device
        body_pose_rm = axis_angle_to_matrix(
            torch.tensor(body_pose_aa, dtype=torch.float32).unsqueeze(0).to(device)
        )  # [1, 23, 3, 3]
        global_orient_rm = axis_angle_to_matrix(
            torch.tensor(global_orient_aa, dtype=torch.float32).unsqueeze(0).to(device)
        ).unsqueeze(1)  # [1, 1, 3, 3]
        betas_t = torch.tensor(betas, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            smpl_out = model.smpl(
                betas=betas_t,
                body_pose=body_pose_rm,
                global_orient=global_orient_rm,
                pose2rot=False,
            )
        verts = smpl_out.vertices[0].cpu().numpy()  # [6890, 3]

        # Scaled focal length for full-image rendering (same formula as extract_smpl_params)
        scaled_fl = focal_length[0] / model_cfg.MODEL.IMAGE_SIZE * max(img_h, img_w)

        try:
            cam_view = renderer.render_rgba_multiple(
                [verts],
                cam_t=[pred_cam_t.copy()],
                render_res=[img_w, img_h],
                mesh_base_color=mesh_color,
                focal_length=scaled_fl,
            )
        except Exception:
            logger.exception("rendering failed on frame {}", idx)
            raise

        # Alpha-composite over original keyframe (following demo.py full_frame path)
        input_rgb = img_bgr.astype(np.float32)[:, :, ::-1] / 255.0
        alpha = cam_view[:, :, 3:]
        composite = np.clip(
            input_rgb * (1 - alpha) + cam_view[:, :, :3] * alpha, 0.0, 1.0
        )

        overlay_path = out_dir / f"overlay_{idx}.png"
        cv2.imwrite(
            str(overlay_path),
            (composite[:, :, ::-1] * 255).astype(np.uint8),
        )
        logger.info("overlay_{}.png saved", idx)
        saved.append(overlay_path)

    return saved


def detect_keypoints(*args, **kwargs):
    """M5: TODO"""
    raise NotImplementedError("M5")


def predict_normals(*args, **kwargs):
    """M5: TODO"""
    raise NotImplementedError("M5")
