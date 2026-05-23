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
    rtmpose_det_checkpoint: Path = Path("./checkpoints/rtmpose/yolox_m.onnx")
    rtmpose_pose_checkpoint: Path = Path("./checkpoints/rtmpose/rtmpose_m_body7.onnx")
    sapiens_normal_checkpoint: Path = Path(
        "./checkpoints/sapiens/sapiens_0.3b_normal_render_people_epoch_66_torchscript.pt2"
    )
    log_level: str = "INFO"


settings = Settings()

# Algorithm constants — not deployment config, intentionally not in .env
NUM_KEYFRAMES: int = 12
OPT_MAX_ITERS: int = 200
LEARNING_RATE: float = 1e-3       # M7 initial; revisit in M8
GRAD_CLIP_NORM: float = 1.0       # M7 defense vs. pitfall #3 explosions
RENDER_RESOLUTION: int = 512      # long side; bumped 256->512 for M8 (normal/keypoint detail)
HEIGHT_TOLERANCE_M: float = 0.05  # ±5 cm dead-band for height loss; no penalty inside band
FACES_PER_PIXEL: int = 25         # SoftSilhouetteShader fragment depth
# Normal-map loss gates — D14 tuning targets.
# Diagnosed D12: face-on interior error 0.06 vs edge/grazing 0.41-0.53 (7× gap).
# Grazing surfaces and silhouette-edge pixels are excluded because SMPL vertex-normal
# interpolation is unreliable there at 288×512 resolution; 12-frame union still
# covers 360° in the face-on band. See NOTES for D12 spatial analysis.
NORMAL_GRAZING_THRESHOLD: float = 0.5   # gate: |rendered_n_z| > this; keeps front hemisphere
NORMAL_EDGE_EROSION_PX: int = 4         # erode rendered silhouette by this many pixels
LOSS_WEIGHTS: dict[str, float] = {
    "silhouette": 1.0,
    "normal": 0.5,
    "keypoint": 1e-3,   # placeholder; pixel² vs [0,1] IoU — tune in D14
    "laplacian": 100.0,
    "symmetry": 0.01,
    "height": 10.0,
    "normal_consistency": 0.1,
}
