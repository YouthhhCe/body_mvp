from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.models import DEFAULT_CHECKPOINT, load_hmr2
from hmr2.utils import recursive_to
from hmr2.utils.renderer import Renderer, cam_crop_to_full
from loguru import logger
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from rtmlib import RTMPose, YOLOX, draw_skeleton
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

# rtmlib 'balanced' mode input shapes (yolox_m + rtmpose-m_body7);
# must match the ONNX models pinned in settings.rtmpose_*_checkpoint.
_RTMPOSE_DET_INPUT_SIZE = (640, 640)
_RTMPOSE_POSE_INPUT_SIZE = (192, 256)
# Per-keypoint score threshold for the skeleton overlay only;
# saved scores are raw. 0.43 matches rtmlib's body.py example.
_KEYPOINT_VIS_THRESHOLD = 0.43

# Mask coverage below this triggers the M5 keypoint-bbox SAM re-run.
# From M5 coverage analysis on run 20260519_215818 (NOTES.md 2026-05-20):
# good frames 11.3-14.6%, failed (03,04,08,09) 4.2-6.9% — gap clean at 9%.
_MASK_REFINE_COVERAGE_THRESHOLD = 0.09
# Per-side padding (fraction of bbox w/h) around the keypoint min/max box.
_KEYPOINT_BBOX_PADDING = 0.10
# Minimum keypoint score to include when computing the bbox; filters
# severely occluded/missing keypoints that would skew the extent.
_KEYPOINT_BBOX_SCORE_THRESHOLD = 0.3

# Sapiens-Normal 0.3B preprocessing (NOTES.md 2026-05-20).
# Input is BGR (no RGB swap), pixels stay in 0-255 (no /255),
# then (x - mean) / std. Output is half-resolution; upsample
# before applying foreground mask.
_SAPIENS_INPUT_H = 1024
_SAPIENS_INPUT_W = 768
_SAPIENS_MEAN = np.array([123.5, 116.5, 103.5], dtype=np.float32)
_SAPIENS_STD = np.array([58.5, 57.0, 57.5], dtype=np.float32)


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


def extract_keypoints(keyframe_paths: list[Path], run_dir: Path) -> list[Path]:
    """RTMPose COCO-17 keypoints per keyframe via rtmlib (YOLOX-m + RTMPose-m).

    On multi-person detection, picks the largest bbox (max area).
    On no detection, saves zeros + bbox_source="none" so downstream
    code can skip the frame instead of trusting a corrupt bbox.

    Per-frame .npz contents:
      - keypoints [17, 2]:   pixel coords (x, y); zeros if bbox_source == "none"
      - scores [17]:         per-keypoint confidence in [0, 1]; zeros if "none"
      - bbox [4]:            xyxy of the YOLOX person bbox used by RTMPose;
                             do not use when bbox_source == "none"
      - bbox_source (str):   "detection" | "none"
    """
    out_dir = run_dir / "keypoints"
    out_dir.mkdir(exist_ok=True)

    det = YOLOX(
        onnx_model=str(settings.rtmpose_det_checkpoint),
        model_input_size=_RTMPOSE_DET_INPUT_SIZE,
        backend="onnxruntime",
        device=settings.device,
    )
    pose = RTMPose(
        onnx_model=str(settings.rtmpose_pose_checkpoint),
        model_input_size=_RTMPOSE_POSE_INPUT_SIZE,
        backend="onnxruntime",
        device=settings.device,
    )
    logger.info(
        "RTMPose loaded: det={}, pose={}",
        settings.rtmpose_det_checkpoint.name,
        settings.rtmpose_pose_checkpoint.name,
    )

    saved: list[Path] = []
    for kf_path in keyframe_paths:
        idx = kf_path.stem.split("_")[-1]
        img_bgr = cv2.imread(str(kf_path))

        bboxes = det(img_bgr)  # [N, 4] xyxy

        if len(bboxes) == 0:
            logger.warning("frame {}: no person detected; saving zeros", idx)
            keypoints = np.zeros((17, 2), dtype=np.float32)
            scores = np.zeros(17, dtype=np.float32)
            bbox = np.zeros(4, dtype=np.float32)
            bbox_source = "none"
        else:
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            best = int(np.argmax(areas))
            if len(bboxes) > 1:
                logger.info(
                    "frame {}: {} persons detected, using largest (idx {})",
                    idx, len(bboxes), best,
                )
            bbox = bboxes[best].astype(np.float32)
            # slice [best:best+1] preserves the leading person dim RTMPose expects
            kps_all, scores_all = pose(img_bgr, bboxes=bboxes[best : best + 1])
            keypoints = kps_all[0].astype(np.float32)  # [17, 2]
            scores = scores_all[0].astype(np.float32)  # [17]
            bbox_source = "detection"

        npz_path = out_dir / f"kp_{idx}.npz"
        np.savez(
            str(npz_path),
            keypoints=keypoints,
            scores=scores,
            bbox=bbox,
            bbox_source=np.array(bbox_source),
        )
        logger.info(
            "kp_{}.npz saved (source={}, mean_score={:.3f}, bbox=[{:.0f},{:.0f},{:.0f},{:.0f}])",
            idx, bbox_source, float(scores.mean()),
            bbox[0], bbox[1], bbox[2], bbox[3],
        )
        saved.append(npz_path)

    return saved


def render_keypoint_overlays(
    keyframe_paths: list[Path],
    npz_paths: list[Path],
    run_dir: Path,
) -> list[Path]:
    """COCO-17 skeleton overlay per keyframe for M5 acceptance review.

    Reads kp_NN.npz files written by extract_keypoints; does not call it.
    Not called by the pipeline. Frames with bbox_source == "none" are
    saved as the raw keyframe (no bbox / skeleton drawn).
    """
    out_dir = run_dir / "keypoints"
    out_dir.mkdir(exist_ok=True)
    saved: list[Path] = []

    for kf_path, npz_path in zip(keyframe_paths, npz_paths):
        idx = kf_path.stem.split("_")[-1]
        img_bgr = cv2.imread(str(kf_path))

        data = np.load(str(npz_path))
        bbox_source = str(data["bbox_source"])
        out = img_bgr.copy()

        if bbox_source == "detection":
            keypoints = data["keypoints"]  # [17, 2]
            scores = data["scores"]        # [17]
            bbox = data["bbox"]            # [4]
            cv2.rectangle(
                out,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 255), 2,
            )
            # draw_skeleton expects leading person dim
            out = draw_skeleton(
                out, keypoints[None], scores[None],
                openpose_skeleton=False, kpt_thr=_KEYPOINT_VIS_THRESHOLD,
            )

        overlay_path = out_dir / f"overlay_{idx}.png"
        cv2.imwrite(str(overlay_path), out)
        logger.info("keypoints/overlay_{}.png saved (source={})", idx, bbox_source)
        saved.append(overlay_path)

    return saved


def _keypoints_to_bbox(
    keypoints: np.ndarray,
    scores: np.ndarray,
    image_shape: tuple,
) -> np.ndarray | None:
    """xyxy bbox from keypoints filtered by score, expanded by padding.

    Returns None if fewer than 4 keypoints pass the score threshold
    (bbox would be too ill-defined to feed SAM safely).
    """
    keep = scores >= _KEYPOINT_BBOX_SCORE_THRESHOLD
    if int(keep.sum()) < 4:
        return None
    pts = keypoints[keep]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    bw, bh = x2 - x1, y2 - y1
    h, w = image_shape[:2]
    x1 = max(0.0, x1 - bw * _KEYPOINT_BBOX_PADDING)
    y1 = max(0.0, y1 - bh * _KEYPOINT_BBOX_PADDING)
    x2 = min(float(w), x2 + bw * _KEYPOINT_BBOX_PADDING)
    y2 = min(float(h), y2 + bh * _KEYPOINT_BBOX_PADDING)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def refine_masks_with_keypoints(
    keyframe_paths: list[Path],
    mask_paths: list[Path],
    keypoint_paths: list[Path],
    run_dir: Path,
) -> list[Path]:
    """Re-run SAM 2 on M3-failed frames using a keypoint-derived box prompt.

    For each frame whose current mask coverage is below the threshold,
    compute a keypoint bbox (min/max of confident keypoints + padding)
    and feed it to SAM 2 as the box prompt. Overwrites masks/mask_NN.png
    and masks/overlay_NN.png in place.

    Frames with bbox_source == "none" in the keypoint .npz, or with too
    few confident keypoints, are skipped — their masks are left as-is.

    Returns the same mask_paths list (files overwritten on disk).
    """
    masks_dir = run_dir / "masks"

    model = build_sam2(
        _SAM2_CONFIG, str(settings.sam2_checkpoint), device=settings.device
    )
    predictor = SAM2ImagePredictor(model)
    logger.info("SAM 2 loaded for mask refinement: {}", settings.sam2_checkpoint)

    refined = 0
    for kf_path, mask_path, kp_path in zip(
        keyframe_paths, mask_paths, keypoint_paths
    ):
        idx = kf_path.stem.split("_")[-1]

        current_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        coverage = float((current_mask > 0).mean())

        if coverage >= _MASK_REFINE_COVERAGE_THRESHOLD:
            logger.info(
                "frame {}: coverage={:.1%} OK, keeping current mask",
                idx, coverage,
            )
            continue

        kp_data = np.load(str(kp_path))
        if str(kp_data["bbox_source"]) == "none":
            logger.warning(
                "frame {}: coverage={:.1%} below threshold but no keypoints; skipping",
                idx, coverage,
            )
            continue

        img_bgr = cv2.imread(str(kf_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        new_box = _keypoints_to_bbox(
            kp_data["keypoints"], kp_data["scores"], img_bgr.shape
        )
        if new_box is None:
            logger.warning(
                "frame {}: <4 keypoints above score threshold; skipping", idx,
            )
            continue

        predictor.set_image(img_rgb)
        masks, scores, _ = predictor.predict(
            box=new_box, multimask_output=True
        )
        best = int(np.argmax(scores))
        new_mask = masks[best].astype(bool)
        new_coverage = float(new_mask.mean())

        cv2.imwrite(str(mask_path), (new_mask * 255).astype(np.uint8))

        green = np.zeros_like(img_bgr)
        green[new_mask] = (0, 200, 0)
        cv2.imwrite(
            str(masks_dir / f"overlay_{idx}.png"),
            cv2.addWeighted(img_bgr, 0.6, green, 0.4, 0),
        )

        logger.info(
            "frame {}: REFINED old_cov={:.1%} -> new_cov={:.1%}, "
            "sam_score={:.3f}, kp_box=[{:.0f},{:.0f},{:.0f},{:.0f}]",
            idx, coverage, new_coverage, scores[best],
            new_box[0], new_box[1], new_box[2], new_box[3],
        )
        refined += 1

    logger.info(
        "mask refinement complete: {}/{} frames refined",
        refined, len(keyframe_paths),
    )
    return mask_paths


def extract_normals(
    keyframe_paths: list[Path],
    mask_paths: list[Path],
    run_dir: Path,
) -> list[Path]:
    """Sapiens-Normal per-frame surface normals (camera coordinates).

    Feeds the FULL keyframe (no person crop — cropping degrades quality
    per Sapiens-Pytorch-Inference notes). Output is half-resolution
    [1, 3, 512, 384]; we bilinearly upsample to the original keyframe
    size, then zero out background using the SAM 2 foreground mask.

    Per-frame .npz contents:
      - normals [H, W, 3]: surface normals in CAMERA coordinates
        (Sapiens convention), float32, raw output (not necessarily
        unit length — normalize per pixel if downstream needs it)
      - foreground [H, W]: uint8, 1 where SAM mask is foreground, else 0
    """
    out_dir = run_dir / "normals"
    out_dir.mkdir(exist_ok=True)

    model = torch.jit.load(
        str(settings.sapiens_normal_checkpoint), map_location=settings.device
    ).eval()
    logger.info("Sapiens-Normal loaded: {}", settings.sapiens_normal_checkpoint.name)

    saved: list[Path] = []
    for kf_path, mask_path in zip(keyframe_paths, mask_paths):
        idx = kf_path.stem.split("_")[-1]

        img_bgr = cv2.imread(str(kf_path))
        H_orig, W_orig = img_bgr.shape[:2]

        # Sapiens preprocess: BGR (no RGB swap), 1024x768, 0-255 scale,
        # (x - mean) / std. cv2.resize takes (W, H).
        resized = cv2.resize(
            img_bgr, (_SAPIENS_INPUT_W, _SAPIENS_INPUT_H),
            interpolation=cv2.INTER_LINEAR,
        )
        arr = (resized.astype(np.float32) - _SAPIENS_MEAN) / _SAPIENS_STD
        tensor = (
            torch.from_numpy(arr.transpose(2, 0, 1))
            .unsqueeze(0)
            .to(settings.device)
        )

        with torch.inference_mode():
            out = model(tensor)  # [1, 3, 512, 384]
            out_up = F.interpolate(
                out, size=(H_orig, W_orig), mode="bilinear", align_corners=False,
            )
            normals = out_up[0].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]

        # Mask out background AFTER upsample
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
        normals_masked = normals * mask[..., None]
        foreground = mask.astype(np.uint8)

        npz_path = out_dir / f"normal_{idx}.npz"
        np.savez(
            str(npz_path),
            normals=normals_masked.astype(np.float32),
            foreground=foreground,
        )

        norms = np.linalg.norm(normals_masked[mask], axis=-1) if mask.any() else np.array([0.0])
        logger.info(
            "normal_{}.npz saved (fg_px={}, mean_norm={:.3f}, range=[{:.2f},{:.2f}])",
            idx, int(mask.sum()),
            float(norms.mean()), float(norms.min()), float(norms.max()),
        )
        saved.append(npz_path)

        del out, out_up, tensor

    return saved


def render_normal_overlays(
    keyframe_paths: list[Path],
    npz_paths: list[Path],
    run_dir: Path,
) -> list[Path]:
    """Normal-map RGB visualization per keyframe for M5 acceptance.

    Per-pixel: normalize to unit length, then map (n+1)/2 -> [0, 255].
    Mapped channels are RGB-ordered (R=X, G=Y, B=Z); we BGR-swap before
    cv2.imwrite. Background pixels are zero (black). Colors will differ
    across frames as the camera circles the subject — that's expected
    for camera-frame normals, not a bug.

    Reads normal_NN.npz; does not call extract_normals. Not pipeline-called.
    """
    out_dir = run_dir / "normals"
    out_dir.mkdir(exist_ok=True)
    saved: list[Path] = []

    for kf_path, npz_path in zip(keyframe_paths, npz_paths):
        idx = kf_path.stem.split("_")[-1]

        data = np.load(str(npz_path))
        normals = data["normals"]            # [H, W, 3] float32
        foreground = data["foreground"] > 0  # [H, W] bool

        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        unit = np.where(norm < 1e-6, 0.0, normals / np.where(norm < 1e-6, 1.0, norm))

        vis_rgb = ((unit + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
        vis_rgb[~foreground] = 0
        vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

        overlay_path = out_dir / f"vis_{idx}.png"
        cv2.imwrite(str(overlay_path), vis_bgr)
        logger.info("normals/vis_{}.png saved", idx)
        saved.append(overlay_path)

    return saved
