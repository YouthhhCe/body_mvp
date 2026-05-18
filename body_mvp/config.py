from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    device: str = "cuda:0"
    checkpoints_dir: Path = Path("./checkpoints")
    runs_dir: Path = Path("./data/runs")
    smpl_model_path: Path = Path("./checkpoints/smpl/SMPL_NEUTRAL.pkl")
    sam2_checkpoint: Path = Path("./checkpoints/sam2/sam2.1_hiera_small.pt")
    log_level: str = "INFO"


settings = Settings()

# Algorithm constants — not deployment config, intentionally not in .env
NUM_KEYFRAMES: int = 12
OPT_MAX_ITERS: int = 200
LEARNING_RATE: float = 5e-3
RENDER_RESOLUTION: int = 512
LOSS_WEIGHTS: dict[str, float] = {
    "silhouette": 1.0,
    "normal": 0.5,
    "keypoint": 0.1,
    "laplacian": 100.0,
    "symmetry": 0.01,
    "height": 10.0,
    "normal_consistency": 0.1,
}
