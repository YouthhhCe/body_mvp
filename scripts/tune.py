"""M8 Stage 2 tuning harness.

Re-runs Stage 2 (ΔV optimization) against an EXISTING Stage 1 bundle, without
re-running Stage 1 from video. Stage 1 is height-independent, so M8 weight
tuning reuses the M7 stage1_result.npz and just overrides the user metadata
(the M7 run stored a 170/65 placeholder; the real test subject is 180/75).

Outputs land in a fresh `<ts>_m8` run dir so the M7 baseline and its pixel
artifacts (keyframes/masks/normals) are never clobbered — the loaded
Stage1Result keeps its original pixel paths pointing back at the source run.

Usage:
    python scripts/tune.py data/runs/20260522_001631 --height 180 --weight 75
"""

from datetime import datetime
from pathlib import Path

import click
import numpy as np
from loguru import logger

from body_mvp import stage1, stage2
from body_mvp.config import settings


@click.command()
@click.argument("source_run", type=click.Path(exists=True, path_type=Path))
@click.option("--height", type=float, default=None,
              help="Override height in cm (default: keep value stored in the bundle)")
@click.option("--weight", type=float, default=None,
              help="Override weight in kg (default: keep value stored in the bundle)")
def main(source_run: Path, height: float | None, weight: float | None) -> None:
    """Re-run Stage 2 on an existing Stage 1 bundle (M8 tuning loop)."""
    logger.remove()
    logger.add(lambda msg: click.echo(msg, err=True), level=settings.log_level, colorize=True)

    s1 = stage1.load_stage1_result(source_run / "stage1_result.npz")
    logger.info("Loaded Stage 1 bundle from {} (run_id={})", source_run, s1.run_id)

    if height is not None:
        logger.info("Override height_cm: {} -> {}", s1.height_cm, height)
        s1.height_cm = height
    if weight is not None:
        logger.info("Override weight_kg: {} -> {}", s1.weight_kg, weight)
        s1.weight_kg = weight

    # Fresh M8 run dir; pixel paths inside s1 still reference source_run.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_m8"
    new_dir = settings.runs_dir / run_id
    (new_dir / "logs").mkdir(parents=True, exist_ok=False)
    logger.add(new_dir / "logs" / "tune.log", level="DEBUG")
    s1.run_id = run_id
    s1.run_dir = new_dir
    logger.info("M8 tuning outputs -> {}", new_dir)

    s2 = stage2.run(s1)
    stage2.save_silhouette_debug(s1, s2)

    logger.info("Stage 2 result:")
    logger.info("  stage2_result.npz: {}", new_dir / "stage2_result.npz")
    logger.info("  iters: {}, final_loss: {:.5f}", s2.n_iterations, s2.final_loss)
    logger.info(
        "  mean IoU init -> final: {:.4f} -> {:.4f} (Δ={:+.4f})",
        float(s2.initial_iou_per_frame.mean()),
        float(s2.final_iou_per_frame.mean()),
        float(s2.final_iou_per_frame.mean() - s2.initial_iou_per_frame.mean()),
    )
    logger.info("  max|ΔV|: {:.5f}", float(np.abs(s2.delta_v).max()))
    logger.info("  overlays + loss_curve in: {}", new_dir / "stage2")


if __name__ == "__main__":
    main()
