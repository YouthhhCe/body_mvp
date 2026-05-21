from dataclasses import dataclass
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

# COCO-17 torso anchors fed to SAM 2 as positive-point prompts alongside
# the YOLOX bbox: nose, L/R shoulder, L/R hip. Resolves the
# foreground/background inversion ambiguity that wide bboxes hit on
# frames 06/11 of test.mp4 (see M6 diagnosis). Wrists/ankles/face
# excluded — extremity points risk snapping SAM to silhouette edges.
_SAM_TORSO_KEYPOINTS = (0, 5, 6, 11, 12)
# Per-point confidence gate. Higher than _KEYPOINT_BBOX_SCORE_THRESHOLD
# (0.3) on purpose: a wrong positive point INSIDE the bbox actively
# pulls SAM off the body, so we only trust high-confidence joints here.
_SAM_KEYPOINT_SCORE_THRESHOLD = 0.5
# Fewer than this many torso points surviving the gate → fail-loud
# rather than ship a frame whose SAM prompt we don't trust.
_SAM_MIN_TORSO_POINTS = 2

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


@dataclass
class Stage1Result:
    """Aggregated output of Stage 1. Consumed by Stage 2 as the input contract.

    Persisted to a single .npz at the run dir root via save_stage1_result;
    reload via load_stage1_result. Pixel-level artifacts (masks, normals)
    are referenced by path so the bundle stays small; Stage 2 lazy-loads them.

    theta_per_frame layout: joint 0 = global_orient, joints 1-23 = body_pose.
    This is the 24-joint SMPL convention used throughout the Stage 2/3 contract;
    the underlying hmr2 outputs (split global_orient + body_pose) are kept in
    the per-frame smpl_NN.npz debug artifacts unchanged.
    """

    # Run metadata
    run_id: str
    run_dir: Path
    video_path: Path

    # User metadata (Stage 2 height-match loss consumes these)
    height_cm: float
    weight_kg: float
    gender: str

    # Keyframe geometry
    keyframe_paths: list[Path]
    n_frames: int
    image_size_wh: tuple[int, int]

    # Single source of truth for person bbox per frame (YOLOX detection,
    # shared by SAM and hmr2 under option (c)).
    bbox_xyxy: np.ndarray              # [N, 4] float32

    # SMPL params
    beta: np.ndarray                   # [10]      float32 — mean over keyframes
    betas_per_frame: np.ndarray        # [N, 10]   float32 — raw hmr2 outputs
    theta_per_frame: np.ndarray        # [N, 24, 3] float32 — axis-angle, joint 0 = global_orient
    pred_cam_per_frame: np.ndarray     # [N, 3]    float32 — hmr2 crop-frame camera
    pred_cam_t_per_frame: np.ndarray   # [N, 3]    float32 — full-image translation
    focal_length_per_frame: np.ndarray # [N]       float32 — scaled, fx == fy

    # 2D keypoints (COCO-17)
    keypoints_2d: np.ndarray           # [N, 17, 2] float32 — pixel coords
    keypoint_scores: np.ndarray        # [N, 17]    float32

    # Pixel-level artifacts (large) referenced by path; Stage 2 lazy-loads
    mask_paths: list[Path]             # masks/mask_NN.png   (uint8)
    normal_paths: list[Path]           # normals/normal_NN.npz (normals + foreground)

    def __post_init__(self) -> None:
        # Normalize tuple-coercible fields so callers can pass list or tuple.
        self.image_size_wh = tuple(self.image_size_wh)

        # Scalars
        if self.gender not in {"neutral", "male", "female"}:
            raise ValueError(
                f"gender: expected one of {{neutral, male, female}}, got {self.gender!r}"
            )
        if self.n_frames <= 0:
            raise ValueError(f"n_frames: must be > 0, got {self.n_frames}")
        if self.height_cm <= 0:
            raise ValueError(f"height_cm: must be > 0, got {self.height_cm}")
        if self.weight_kg <= 0:
            raise ValueError(f"weight_kg: must be > 0, got {self.weight_kg}")

        N = self.n_frames

        def _check_len(name: str, actual: int, expected: int) -> None:
            if actual != expected:
                raise ValueError(f"{name}: expected length {expected}, got {actual}")

        def _check_shape(name: str, arr: np.ndarray, expected: tuple) -> None:
            if arr.shape != expected:
                raise ValueError(
                    f"{name}: expected shape {expected}, got {tuple(arr.shape)}"
                )

        _check_len("keyframe_paths", len(self.keyframe_paths), N)
        _check_len("mask_paths", len(self.mask_paths), N)
        _check_len("normal_paths", len(self.normal_paths), N)

        _check_shape("bbox_xyxy", self.bbox_xyxy, (N, 4))
        _check_shape("beta", self.beta, (10,))
        _check_shape("betas_per_frame", self.betas_per_frame, (N, 10))
        _check_shape("theta_per_frame", self.theta_per_frame, (N, 24, 3))
        _check_shape("pred_cam_per_frame", self.pred_cam_per_frame, (N, 3))
        _check_shape("pred_cam_t_per_frame", self.pred_cam_t_per_frame, (N, 3))
        _check_shape("focal_length_per_frame", self.focal_length_per_frame, (N,))
        _check_shape("keypoints_2d", self.keypoints_2d, (N, 17, 2))
        _check_shape("keypoint_scores", self.keypoint_scores, (N, 17))


def save_stage1_result(result: Stage1Result, path: Path) -> None:
    """Write Stage1Result to a single .npz at `path`.

    Strings stored as 0-d unicode arrays; paths stored as their str() form;
    list[Path] stored as 1-d unicode arrays. load_stage1_result inverts this.
    """
    assert path.suffix == ".npz", (
        f"save_stage1_result: path must end in .npz to match np.savez's "
        f"own behavior (it silently appends .npz otherwise, desyncing the "
        f"file path the caller thinks they wrote). Got: {path}"
    )
    np.savez(
        str(path),
        run_id=np.array(result.run_id),
        run_dir=np.array(str(result.run_dir)),
        video_path=np.array(str(result.video_path)),
        height_cm=np.array(result.height_cm, dtype=np.float32),
        weight_kg=np.array(result.weight_kg, dtype=np.float32),
        gender=np.array(result.gender),
        keyframe_paths=np.array([str(p) for p in result.keyframe_paths]),
        n_frames=np.array(result.n_frames, dtype=np.int64),
        image_size_wh=np.array(result.image_size_wh, dtype=np.int64),
        bbox_xyxy=result.bbox_xyxy.astype(np.float32),
        beta=result.beta.astype(np.float32),
        betas_per_frame=result.betas_per_frame.astype(np.float32),
        theta_per_frame=result.theta_per_frame.astype(np.float32),
        pred_cam_per_frame=result.pred_cam_per_frame.astype(np.float32),
        pred_cam_t_per_frame=result.pred_cam_t_per_frame.astype(np.float32),
        focal_length_per_frame=result.focal_length_per_frame.astype(np.float32),
        keypoints_2d=result.keypoints_2d.astype(np.float32),
        keypoint_scores=result.keypoint_scores.astype(np.float32),
        mask_paths=np.array([str(p) for p in result.mask_paths]),
        normal_paths=np.array([str(p) for p in result.normal_paths]),
    )


def load_stage1_result(path: Path) -> Stage1Result:
    """Inverse of save_stage1_result. Casts strings/paths/scalars back to Python types.

    allow_pickle=False is safe here: every field is stored as a native numpy
    dtype (no object arrays). Strings are unicode arrays (<U...) and round-trip
    without pickle.
    """
    data = np.load(str(path), allow_pickle=False)
    return Stage1Result(
        run_id=str(data["run_id"].item()),
        run_dir=Path(str(data["run_dir"].item())),
        video_path=Path(str(data["video_path"].item())),
        height_cm=float(data["height_cm"].item()),
        weight_kg=float(data["weight_kg"].item()),
        gender=str(data["gender"].item()),
        keyframe_paths=[Path(str(p)) for p in data["keyframe_paths"]],
        n_frames=int(data["n_frames"].item()),
        image_size_wh=tuple(int(x) for x in data["image_size_wh"]),
        bbox_xyxy=data["bbox_xyxy"],
        beta=data["beta"],
        betas_per_frame=data["betas_per_frame"],
        theta_per_frame=data["theta_per_frame"],
        pred_cam_per_frame=data["pred_cam_per_frame"],
        pred_cam_t_per_frame=data["pred_cam_t_per_frame"],
        focal_length_per_frame=data["focal_length_per_frame"],
        keypoints_2d=data["keypoints_2d"],
        keypoint_scores=data["keypoint_scores"],
        mask_paths=[Path(str(p)) for p in data["mask_paths"]],
        normal_paths=[Path(str(p)) for p in data["normal_paths"]],
    )


def _make_box_prompt(image: np.ndarray) -> np.ndarray:
    """Returns [x1, y1, x2, y2] heuristic center crop: 80% width, 90% height."""
    h, w = image.shape[:2]
    return np.array(
        [int(w * 0.10), int(h * 0.05), int(w * 0.90), int(h * 0.95)],
        dtype=np.float32,
    )


def segment_keyframes(
    keyframe_paths: list[Path],
    bboxes_xyxy: np.ndarray,
    keypoints_2d: np.ndarray,
    keypoint_scores: np.ndarray,
    run_dir: Path,
) -> list[Path]:
    """SAM 2 person mask per keyframe, prompted with YOLOX bbox + positive-point
    torso anchors from RTMPose.

    Under M6 (option c), the bbox is the YOLOX detection from
    extract_keypoints, shared with extract_smpl_params. The bbox alone is
    ambiguous on frames where the person occupies a narrow vertical strip
    inside a near-image-width bbox (e.g. spread arms): SAM's three
    multimask candidates may include the inverted background region and
    rank it highest. Adding torso keypoints as positive points
    (point_labels == 1) tells SAM which pixels are the person and breaks
    that ambiguity.

    Inputs are parallel arrays from kp_NN.npz (loaded by pipeline.run):
        bboxes_xyxy:     [N, 4]
        keypoints_2d:    [N, 17, 2]
        keypoint_scores: [N, 17]

    Raises if fewer than _SAM_MIN_TORSO_POINTS torso keypoints clear
    _SAM_KEYPOINT_SCORE_THRESHOLD on any frame.
    """
    masks_dir = run_dir / "masks"
    torso_idx = np.array(_SAM_TORSO_KEYPOINTS, dtype=np.int64)

    model = build_sam2(_SAM2_CONFIG, str(settings.sam2_checkpoint), device=settings.device)
    predictor = SAM2ImagePredictor(model)
    logger.info("SAM 2 loaded: {}", settings.sam2_checkpoint)

    saved: list[Path] = []
    for kf_path, bbox, kps, kp_scs in zip(
        keyframe_paths, bboxes_xyxy, keypoints_2d, keypoint_scores
    ):
        idx = kf_path.stem.split("_")[-1]  # "keyframe_03" -> "03"

        # Filter torso candidates by confidence; keep the COCO indices that
        # survive so the per-frame log can show which joints fired.
        torso_kps = kps[torso_idx]                # [5, 2]
        torso_scs = kp_scs[torso_idx]             # [5]
        keep = torso_scs >= _SAM_KEYPOINT_SCORE_THRESHOLD
        n_kept = int(keep.sum())
        if n_kept < _SAM_MIN_TORSO_POINTS:
            raise RuntimeError(
                f"frame {idx}: only {n_kept} torso keypoint(s) above score "
                f"{_SAM_KEYPOINT_SCORE_THRESHOLD} (need >= {_SAM_MIN_TORSO_POINTS}); "
                f"cannot trust SAM prompt. Torso scores: "
                f"{dict(zip(_SAM_TORSO_KEYPOINTS, torso_scs.tolist()))}"
            )
        positive_points = torso_kps[keep].astype(np.float32)            # [K, 2]
        positive_labels = np.ones(n_kept, dtype=np.int32)
        kept_coco = tuple(int(i) for i in torso_idx[keep].tolist())

        img_bgr = cv2.imread(str(kf_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        predictor.set_image(img_rgb)
        masks, scores, _ = predictor.predict(
            point_coords=positive_points,
            point_labels=positive_labels,
            box=bbox.astype(np.float32),
            multimask_output=True,
        )

        best = int(np.argmax(scores))
        mask = masks[best].astype(bool)

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
            "mask_{}.png: score={:.4f}, coverage={:.1f}%, positive_points={} {}",
            idx, scores[best], 100.0 * mask.sum() / mask.size,
            n_kept, kept_coco,
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
    bboxes_xyxy: np.ndarray,
    run_dir: Path,
) -> list[Path]:
    """4D Humans inference per keyframe. Returns list of .npz paths.

    bboxes_xyxy: [N, 4] XYXY, row i for keyframe_paths[i]. Under M6
    (option c) this is the YOLOX detection from extract_keypoints,
    shared with segment_keyframes. hmr2's ViTDetDataset accepts XYXY
    directly and squares the crop internally — no pre-padding needed.

    bbox_source written into smpl_NN.npz is always "detection" on this
    code path. The field is retained for backwards-compat with older
    M4/M5 runs whose .npzs may carry "mask-derived" or
    "heuristic-fallback".
    """
    out_dir = run_dir / "smpl_params"
    out_dir.mkdir(exist_ok=True)

    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.to(settings.device)
    model.eval()
    logger.info("HMR2 loaded: {}", DEFAULT_CHECKPOINT)

    saved: list[Path] = []

    for kf_path, bbox in zip(keyframe_paths, bboxes_xyxy):
        idx = kf_path.stem.split("_")[-1]  # "keyframe_03" -> "03"

        img_bgr = cv2.imread(str(kf_path))

        bbox = bbox.astype(np.float32)
        bbox_source = "detection"
        logger.info(
            "frame {}: bbox=[{:.0f},{:.0f},{:.0f},{:.0f}]",
            idx, bbox[0], bbox[1], bbox[2], bbox[3],
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
