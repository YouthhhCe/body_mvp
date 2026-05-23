import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from body_mvp import stage1, stage2, stage3
from body_mvp.config import NUM_KEYFRAMES, settings
from body_mvp.stage1 import Stage1Result
from body_mvp.stage2 import Stage2Result


def _create_run_dir(run_id: str) -> Path:
    run_dir = settings.runs_dir / run_id
    (run_dir / "keyframes").mkdir(parents=True, exist_ok=False)
    (run_dir / "masks").mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(parents=True, exist_ok=False)
    return run_dir


def extract_keyframes(video_path: Path, run_dir: Path) -> list[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Probe backwards from the reported last frame to find the actual last readable index.
        # CAP_PROP_FRAME_COUNT is codec-reported and can over-count by a variable amount.
        last_readable = reported_frames - 1
        while last_readable >= 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, last_readable)
            ok, _ = cap.read()
            if ok:
                break
            last_readable -= 1

        if last_readable < 0:
            raise RuntimeError(f"Could not read any frame from {video_path}")

        actual_frames = last_readable + 1
        if actual_frames != reported_frames:
            logger.warning(
                "Codec reported {} frames but last readable index is {} ({} actual frames)",
                reported_frames, last_readable, actual_frames,
            )

        if actual_frames < NUM_KEYFRAMES:
            raise RuntimeError(
                f"Video too short: {actual_frames} readable frames, need at least {NUM_KEYFRAMES}"
            )

        duration = actual_frames / fps if fps > 0 else 0.0
        logger.info("Video: {} frames, {:.2f} fps, {:.2f}s", actual_frames, fps, duration)

        indices = np.linspace(0, last_readable, NUM_KEYFRAMES).round().astype(int)
        saved: list[Path] = []

        for i, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
            out_path = run_dir / "keyframes" / f"keyframe_{i:02d}.jpg"
            cv2.imwrite(str(out_path), frame)
            logger.info("Wrote {}", out_path)
            saved.append(out_path)
    finally:
        cap.release()

    return saved


def _assert_consistent_keyframe_size(keyframe_paths: list[Path]) -> tuple[int, int]:
    """Return (W, H) from the first keyframe; raise if any other differs.

    Stage 1 has no per-frame resizing, so a size mismatch inside one video
    would corrupt image_size_wh in Stage1Result. Treat as fatal.
    """
    first = cv2.imread(str(keyframe_paths[0]))
    if first is None:
        raise RuntimeError(f"Could not read keyframe: {keyframe_paths[0]}")
    H, W = first.shape[:2]
    for p in keyframe_paths[1:]:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"Could not read keyframe: {p}")
        h, w = img.shape[:2]
        if (w, h) != (W, H):
            raise RuntimeError(
                f"Keyframe size mismatch: {keyframe_paths[0].name} is "
                f"{W}x{H} but {p.name} is {w}x{h}"
            )
    return (W, H)


def _load_keypoint_artifacts(
    keypoint_paths: list[Path], n_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load kp_NN.npz files, gate on bbox_source, stack into [N, ...] arrays.

    Returns (bboxes_xyxy [N,4], keypoints_2d [N,17,2], keypoint_scores [N,17]).

    Raises if any frame has bbox_source != "detection". Under M6 option (c)
    every keyframe must carry a real YOLOX detection — no silent fallback
    to a center-crop heuristic.
    """
    if len(keypoint_paths) != n_frames:
        raise RuntimeError(
            f"keypoint_paths length {len(keypoint_paths)} != n_frames {n_frames}"
        )

    bboxes = np.zeros((n_frames, 4), dtype=np.float32)
    keypoints = np.zeros((n_frames, 17, 2), dtype=np.float32)
    scores = np.zeros((n_frames, 17), dtype=np.float32)
    bad: list[tuple[int, str, str]] = []
    for i, p in enumerate(keypoint_paths):
        d = np.load(str(p))
        src = str(d["bbox_source"])
        if src != "detection":
            bad.append((i, p.name, src))
            continue
        bboxes[i] = d["bbox"].astype(np.float32)
        keypoints[i] = d["keypoints"].astype(np.float32)
        scores[i] = d["scores"].astype(np.float32)
    if bad:
        raise RuntimeError(
            f"YOLOX failed to detect a person on {len(bad)} frame(s); "
            f"M6 option (c) requires every frame to have bbox_source == 'detection'. "
            f"Offenders (frame_idx, file, bbox_source): {bad}"
        )
    return bboxes, keypoints, scores


def _load_smpl_artifacts(
    smpl_paths: list[Path], n_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load smpl_NN.npz files, stack into [N, ...] arrays.

    Returns (betas [N,10], body_pose_aa [N,23,3], global_orient_aa [N,3],
             pred_cam [N,3], pred_cam_t [N,3], focal_length [N]).

    focal_length on disk is [2]; collapsed to scalar here since hmr2 always
    produces fx == fy. The split body_pose / global_orient layout is
    preserved on disk; the 24-joint theta merge happens in run() during
    Stage1Result construction.
    """
    if len(smpl_paths) != n_frames:
        raise RuntimeError(
            f"smpl_paths length {len(smpl_paths)} != n_frames {n_frames}"
        )

    betas = np.zeros((n_frames, 10), dtype=np.float32)
    body_pose = np.zeros((n_frames, 23, 3), dtype=np.float32)
    global_orient = np.zeros((n_frames, 3), dtype=np.float32)
    pred_cam = np.zeros((n_frames, 3), dtype=np.float32)
    pred_cam_t = np.zeros((n_frames, 3), dtype=np.float32)
    focal = np.zeros(n_frames, dtype=np.float32)
    for i, p in enumerate(smpl_paths):
        d = np.load(str(p))
        betas[i] = d["betas"].astype(np.float32)
        body_pose[i] = d["body_pose"].astype(np.float32)
        global_orient[i] = d["global_orient"].astype(np.float32)
        pred_cam[i] = d["pred_cam"].astype(np.float32)
        pred_cam_t[i] = d["pred_cam_t"].astype(np.float32)
        # hmr2 sets fx == fy by construction (cfg.EXTRA.FOCAL_LENGTH broadcast over 2);
        # this assert fails loud if that assumption ever breaks upstream instead of
        # silently collapsing to fl[0] and pretending the focal length is isotropic.
        fl = d["focal_length"]
        if float(fl[0]) != float(fl[1]):
            raise RuntimeError(
                f"focal_length not isotropic on frame {i} ({p.name}): {fl}; "
                f"hmr2 should produce fx == fy"
            )
        focal[i] = float(fl[0])
    return betas, body_pose, global_orient, pred_cam, pred_cam_t, focal


def _verify_round_trip(original: Stage1Result, path: Path) -> None:
    """Load the just-written Stage1Result and fail-loud if anything drifted."""
    loaded = stage1.load_stage1_result(path)

    scalar_fields = (
        "run_id", "run_dir", "video_path",
        "height_cm", "weight_kg", "gender",
        "n_frames", "image_size_wh",
        "keyframe_paths", "mask_paths", "normal_paths",
    )
    for name in scalar_fields:
        a = getattr(original, name)
        b = getattr(loaded, name)
        if a != b:
            raise RuntimeError(
                f"round-trip mismatch on {name}: original={a!r} loaded={b!r}"
            )

    array_fields = (
        "bbox_xyxy", "beta", "betas_per_frame", "theta_per_frame",
        "pred_cam_per_frame", "pred_cam_t_per_frame", "focal_length_per_frame",
        "keypoints_2d", "keypoint_scores",
    )
    for name in array_fields:
        a = getattr(original, name)
        b = getattr(loaded, name)
        if a.shape != b.shape:
            raise RuntimeError(f"round-trip shape mismatch on {name}: {a.shape} vs {b.shape}")
        if a.dtype != b.dtype:
            raise RuntimeError(f"round-trip dtype mismatch on {name}: {a.dtype} vs {b.dtype}")
        if not np.array_equal(a, b):
            raise RuntimeError(f"round-trip value mismatch on {name}")

    logger.info("stage1_result.npz round-trip self-check passed")


def run(
    video_path: Path, height: float, weight: float, gender: str
) -> tuple[Stage1Result, Stage2Result]:
    now = datetime.now()
    run_id = now.strftime("%Y%m%d_%H%M%S")
    run_dir = _create_run_dir(run_id)

    logger.add(run_dir / "logs" / "run.log", level="DEBUG")
    logger.info(
        "Starting pipeline: run_id={}, video={}, height={}, weight={}, gender={}",
        run_id, video_path, height, weight, gender,
    )

    meta = {
        "run_id": run_id,
        "timestamp": now.isoformat(),
        "video_path": str(video_path.resolve()),
        "height": height,
        "weight": weight,
        "gender": gender,
        "num_keyframes": NUM_KEYFRAMES,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Wrote meta.json")

    # 1. Sample keyframes from the video (M2).
    keyframe_paths = extract_keyframes(video_path, run_dir)
    n_frames = len(keyframe_paths)

    # 2. Frame-size consistency assertion.
    image_size_wh = _assert_consistent_keyframe_size(keyframe_paths)
    logger.info("Keyframe size: {}x{}", image_size_wh[0], image_size_wh[1])

    # 3. Keypoints first (option c): YOLOX bbox becomes the single source
    #    of truth for "where is the person", shared by SAM and hmr2.
    logger.info("Running RTMPose (YOLOX + RTMPose) on {} keyframes", n_frames)
    keypoint_paths = stage1.extract_keypoints(keyframe_paths, run_dir)

    # 4. Gate: every keyframe must carry a real YOLOX detection.
    bboxes_xyxy, keypoints_2d, keypoint_scores = _load_keypoint_artifacts(
        keypoint_paths, n_frames
    )
    if bboxes_xyxy.shape != (n_frames, 4):
        raise RuntimeError(
            f"bboxes_xyxy.shape = {bboxes_xyxy.shape}, expected ({n_frames}, 4)"
        )

    # 5. SAM segmentation, prompted with the shared YOLOX bbox PLUS torso
    #    keypoints as positive points (resolves the wide-bbox inversion
    #    ambiguity seen on frames 06/11 of test.mp4 in the first M6 run).
    logger.info("Running SAM 2 segmentation with shared YOLOX bbox + torso points")
    mask_paths = stage1.segment_keyframes(
        keyframe_paths, bboxes_xyxy, keypoints_2d, keypoint_scores, run_dir
    )

    # 6. hmr2 SMPL inference, prompted with the same shared YOLOX bbox.
    logger.info("Running 4D Humans (hmr2) with shared YOLOX bbox")
    smpl_paths = stage1.extract_smpl_params(keyframe_paths, bboxes_xyxy, run_dir)

    # 7. Sapiens normals (background zeroed via SAM masks).
    logger.info("Running Sapiens-Normal")
    normal_paths = stage1.extract_normals(keyframe_paths, mask_paths, run_dir)

    # 8. Aggregate per-frame SMPL .npzs into [N, ...] stacks.
    (betas_per_frame, body_pose_per_frame, global_orient_per_frame,
     pred_cam_per_frame, pred_cam_t_per_frame, focal_length_per_frame
     ) = _load_smpl_artifacts(smpl_paths, n_frames)

    # 9. β aggregation: per-component mean.
    beta = betas_per_frame.mean(axis=0).astype(np.float32)

    # 10. Merge SMPL pose into the 24-joint convention:
    #     joint 0 = global_orient, joints 1-23 = body_pose.
    theta_per_frame = np.concatenate(
        [global_orient_per_frame[:, None, :], body_pose_per_frame], axis=1
    ).astype(np.float32)

    # 11. Construct Stage1Result. __post_init__ validates shapes/values.
    result = Stage1Result(
        run_id=run_id,
        run_dir=run_dir,
        video_path=video_path.resolve(),
        height_cm=height,
        weight_kg=weight,
        gender=gender,
        keyframe_paths=keyframe_paths,
        n_frames=n_frames,
        image_size_wh=image_size_wh,
        bbox_xyxy=bboxes_xyxy,
        beta=beta,
        betas_per_frame=betas_per_frame,
        theta_per_frame=theta_per_frame,
        pred_cam_per_frame=pred_cam_per_frame,
        pred_cam_t_per_frame=pred_cam_t_per_frame,
        focal_length_per_frame=focal_length_per_frame,
        keypoints_2d=keypoints_2d,
        keypoint_scores=keypoint_scores,
        mask_paths=mask_paths,
        normal_paths=normal_paths,
    )

    # 12. Persist + round-trip self-check.
    out_path = run_dir / "stage1_result.npz"
    stage1.save_stage1_result(result, out_path)
    logger.info("Wrote {}", out_path)
    # MVP-stage belt-and-suspenders: load the bundle back and verify every
    # field matches the in-memory result. Cheap (small bundle, no pixel data)
    # and catches drift in save/load semantics. Likely to gain a disable flag
    # and move out of the hot path once Stage 1 stabilizes.
    _verify_round_trip(result, out_path)

    logger.info("Stage 1 complete: run_id={}, n_frames={}", run_id, n_frames)

    # Stage 2 (MVP bypass): outputs zero ΔV without running optimization.
    # stage2.run persists Stage2Result and runs its own round-trip check.
    stage2_result = stage2.run(result)

    return result, stage2_result
