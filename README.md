# Body MVP

Reconstruct a 3D body mesh from a spinning video.

## Quick start

```bash
# 1. Install dependencies
pip install -e .

# 2. Download model checkpoints (~5GB)
bash scripts/download_models.sh

# 3. Run pipeline on a video
python scripts/run.py data/input_videos/test.mp4 --height 170 --weight 65
```

Output: `data/runs/<run_id>/result.glb`

## Requirements

- Linux + NVIDIA GPU (≥16GB VRAM)
- Python 3.10
- CUDA 11.8 or 12.1

## See also

- `PROJECT.md` — full project spec
- `CLAUDE.md` — Claude Code working agreement
- `NOTES.md` — development log