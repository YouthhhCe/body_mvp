import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from body_mvp import stage1, stage2, stage3
from body_mvp.config import NUM_KEYFRAMES, settings


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


def run(video_path: Path, height: float, weight: float, gender: str) -> None:
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

    keyframe_paths = extract_keyframes(video_path, run_dir)
    stage1.segment_keyframes(keyframe_paths, run_dir)

    stage1.run(video_path, height, weight, gender)
    stage2.run()
    stage3.run()
