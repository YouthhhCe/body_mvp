from pathlib import Path

import click
import numpy as np
from loguru import logger

from body_mvp import pipeline
from body_mvp.config import settings


@click.command()
@click.argument("video_path", type=click.Path(exists=True, path_type=Path))
@click.option("--height", required=True, type=float, help="Height in cm")
@click.option("--weight", required=True, type=float, help="Weight in kg")
@click.option("--gender", default="neutral", show_default=True,
              type=click.Choice(["neutral", "male", "female"]), help="Biological sex for SMPL model")
def main(video_path: Path, height: float, weight: float, gender: str) -> None:
    """Reconstruct a 3D body mesh from a spinning video."""
    logger.remove()
    logger.add(lambda msg: click.echo(msg, err=True), level=settings.log_level, colorize=True)
    s1, s2 = pipeline.run(video_path, height, weight, gender)

    beta_str = "[" + ", ".join(f"{x:.3f}" for x in s1.beta) + "]"
    logger.info("Stage 1 result:")
    logger.info("  run_dir: {}", s1.run_dir)
    logger.info("  n_frames: {}", s1.n_frames)
    logger.info("  image_size_wh: {}", s1.image_size_wh)
    logger.info("  stage1_result.npz: {}", s1.run_dir / "stage1_result.npz")
    logger.info("  aggregated β (mean across keyframes): {}", beta_str)

    logger.info("Stage 2 result:")
    logger.info("  stage2_result.npz: {}", s2.run_dir / "stage2_result.npz")
    logger.info("  iters: {}, final_loss: {:.5f}", s2.n_iterations, s2.final_loss)
    if s2.initial_iou_per_frame is not None:
        logger.info(
            "  mean IoU init -> final: {:.4f} -> {:.4f} (Δ={:+.4f})",
            float(s2.initial_iou_per_frame.mean()),
            float(s2.final_iou_per_frame.mean()),
            float(s2.final_iou_per_frame.mean() - s2.initial_iou_per_frame.mean()),
        )
    logger.info("  max|ΔV|: {:.5f}", float(np.abs(s2.delta_v).max()))


if __name__ == "__main__":
    main()
