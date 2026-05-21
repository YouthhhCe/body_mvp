from pathlib import Path

import click
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
    result = pipeline.run(video_path, height, weight, gender)

    beta_str = "[" + ", ".join(f"{x:.3f}" for x in result.beta) + "]"
    logger.info("Stage 1 result:")
    logger.info("  run_dir: {}", result.run_dir)
    logger.info("  n_frames: {}", result.n_frames)
    logger.info("  image_size_wh: {}", result.image_size_wh)
    logger.info("  stage1_result.npz: {}", result.run_dir / "stage1_result.npz")
    logger.info("  aggregated β (mean across keyframes): {}", beta_str)


if __name__ == "__main__":
    main()
