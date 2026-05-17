# Body MVP

## What this project does

Reconstruct a perceptually accurate 3D body mesh from a user's spinning video, output as a GLB file viewable in a browser. 

Goal: user can recognize themselves. Captures body shape and posture features (belly, posture, proportions). Not aiming for medical-grade accuracy.

## End-to-end flow

1. User records a spinning video (~10s, wearing tight clothes)
2. Backend processes video → 3D mesh in A-pose
3. User views mesh in browser, can rotate/zoom

## Pipeline (3 stages)

### Stage 1 — Solve
From video, get per-frame body parameters:
- Sample 8-16 keyframes from spinning video
- SAM 2 → person mask per keyframe
- 4D Humans → SMPL β (shape) + θ (pose) per keyframe, shared β across frames
- RTMPose → 2D keypoints per keyframe
- Sapiens-Normal → predicted normal map per keyframe

Output: `Stage1Result` (β, per-frame θ, masks, normals, keypoints, cameras)

### Stage 2 — Sculpt
Test-time optimization of per-vertex offset ΔV [6890, 3] in canonical T-pose space.

Losses:
- Silhouette IoU (against SAM 2 masks)
- Normal map L2 (against Sapiens predictions)
- 2D keypoint reprojection
- Laplacian smoothing (anti-spike)
- Part-aware symmetry (weak on torso/limbs, none on belly)
- Height match to user input
- Normal consistency

Output: `Stage2Result` (β unchanged, ΔV new)

### Stage 3 — Export
- Apply ΔV in canonical space
- LBS to A-pose (slight arm spread, conservative)
- Light Laplacian smoothing on LBS-affected regions (shoulders, hips)
- Export GLB
- Generate thumbnail

Output: `Stage3Result` (vertices_a_pose, vertices_canonical, glb_path, quality)

## Tech stack

- Python 3.10
- PyTorch 2.x + CUDA 11.8 or 12.1
- PyTorch3D (differentiable rendering)
- SAM 2 (Meta, Apache 2.0)
- 4D Humans (Berkeley, non-commercial license — MVP only)
- RTMPose (OpenMMLab, Apache 2.0)
- Sapiens-Normal (Meta, check version-specific license)
- smplx (Max Planck, non-commercial license — MVP only)
- trimesh (GLB export)
- Click (CLI)

No web service yet. Run via CLI:

    python scripts/run.py data/input_videos/test.mp4 --height 170 --weight 65 --gender neutral

## File layout

    body-mvp/
    ├── PROJECT.md, CLAUDE.md, NOTES.md, README.md
    ├── pyproject.toml, .env.example, .gitignore
    ├── checkpoints/
    │   └── {sam2, 4dhumans, rtmpose, sapiens, smpl}/
    ├── data/
    │   ├── input_videos/
    │   └── runs/
    ├── scripts/
    │   ├── download_models.sh
    │   └── run.py
    ├── body_mvp/
    │   ├── config.py        # All settings in one place
    │   ├── pipeline.py      # Orchestrates 3 stages
    │   ├── stage1.py        # All of stage 1 in one file
    │   ├── stage2.py        # All of stage 2 in one file
    │   ├── stage3.py        # All of stage 3 in one file
    │   ├── models.py        # Model loading (singletons)
    │   ├── losses.py        # Stage 2 loss functions
    │   ├── lbs.py           # Linear blend skinning
    │   ├── render.py        # PyTorch3D wrappers
    │   └── utils.py         # IO, visualization, timing
    ├── viewer/
    │   ├── index.html
    │   └── viewer.js        # Three.js demo (last)
    └── tests/fixtures/

## Development milestones

- [ ] M1: Project scaffolding, dependencies installed, models downloaded
- [ ] M2: Video → keyframes extracted, saved to disk
- [ ] M3: Keyframes → SAM 2 masks, visualized
- [ ] M4: Keyframes → 4D Humans β/θ, SMPL mesh visualized overlaid
- [ ] M5: Keyframes → RTMPose keypoints + Sapiens normal maps
- [ ] M6: Stage 1 end-to-end, Stage1Result saved as .npz
- [ ] M7: Stage 2 optimization loop skeleton (silhouette loss only), runs without error
- [ ] M8: Stage 2 full loss suite, parameter tuning on test video
- [ ] M9: Stage 3 A-pose + GLB export, viewable in any GLB viewer
- [ ] M10: Web viewer with Three.js
- [ ] M11: Evaluation on 5+ volunteers, iterate

## Current status
Milestone: not started

## Known issues / decisions to revisit later
- 4D Humans + smplx are non-commercial. Need to swap or license before commercial launch.
- Sapiens-Normal license depends on specific checkpoint version, verify before use.
- Loose clothing → bad results. MVP requires tight clothing.
- LBS candy-wrapping in A-pose if user's photo pose is very different — partially mitigated by conservative A-pose.

## Out of scope for MVP
- Multi-photo input mode (video only)
- Coarse-to-fine optimization
- Camera yaw joint optimization
- Pose-dependent corrective shapes
- User authentication, database, job queue
- Mobile native app
- Production-grade error handling and retries