#!/usr/bin/env bash
# Run from the repo root. Each milestone adds its models below.

set -euo pipefail

# --- M3: SAM 2.1 small (person segmentation) ---
if [ ! -f checkpoints/sam2/sam2.1_hiera_small.pt ]; then
    mkdir -p checkpoints/sam2
    wget -q --show-progress \
        -O checkpoints/sam2/sam2.1_hiera_small.pt \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
fi
