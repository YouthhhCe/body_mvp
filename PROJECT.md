# Body MVP

> *This file is the project snapshot — what we're building, the architecture, current status. For how to maintain it (and the relationship to CLAUDE.md / NOTES.md), see CLAUDE.md.*

## Bigger picture: where this fits

This repo is **Layer 1 of a 3-layer health app**. Understanding the full app helps make the right design choices in this layer.

### The full app vision

A health/fitness app where users record a spinning video of themselves and get:
1. A personalized 3D body avatar they can rotate and zoom
2. An analysis of their body shape and posture
3. AI-generated recommendations for exercise, stretching, and nutrition

The app's core hook is "see your own body in 3D" — a visual that's immediately engaging. The deeper value is the personalized health guidance derived from that 3D model.

### The 3-layer architecture

**Layer 1 — Body Reconstruction (this repo)**
- Input: spinning video + height/weight/gender
- Output: a perceptually accurate 3D body mesh + auxiliary pose/shape data
- This is the foundation. Everything else depends on its output.

**Layer 2 — Body Understanding (future, not in this repo)**
- Input: Layer 1's output
- Extracts semantic features: pelvic tilt, shoulder asymmetry, spine curvature, belly projection, forward head, body proportions, etc.

**Layer 3 — Health Reasoning (future, not in this repo)**
- Input: Layer 2's profile
- LLM + RAG over a curated fitness/health knowledge base
- Output: exercise plans, stretching routines, dietary suggestions, risk flags

### Design implications for Layer 1

Layer 1's output is the data foundation for Layers 2 and 3:

- Posture information is preserved separately from the display mesh. Layer 2 needs the user's natural pose, not just a standardized A-pose. The A-pose mesh is for display.
- Shape information is preserved separately from pose. Layer 2 computes shape features from `delta_v` + `beta` in canonical space.
- Output is structured data + GLB, not just GLB. GLB is for the user's eyes; structured data is for downstream layers.

### Scope boundary

**This repo (Layer 1) does NOT do:**
- Feature extraction (Layer 2)
- LLM calls, health advice, nutrition plans (Layer 3)
- User accounts, payment, production app frontend

**This repo (Layer 1) DOES do:**
- Take a video + metadata, produce a mesh + structured body data
- A simple Web viewer for the mesh (for demo/validation only)
- Be callable as a Python library from future Layer 2 code

---

## What this project does

Reconstruct a perceptually accurate 3D body mesh from a user's spinning video. Output: a GLB file + structured analysis data.

Goal: user can recognize themselves. Captures body shape and posture features. Not aiming for medical-grade accuracy.

## End-to-end flow

1. User records a spinning video (~10s, wearing tight clothes)
2. Backend processes video → 3D mesh in A-pose + structured analysis data
3. User views mesh in browser, can rotate/zoom
4. Structured data is handed off to Layer 2 (future)

---

## Pipeline (3 stages)

### Stage 1 — Solve
Extract per-frame body parameters from the video by combining several models:
- Sample a set of keyframes from the spinning video (covering ~360° of orientations)
- SAM 2 → person mask per keyframe
- 4D Humans → SMPL β (shape) + θ (pose) per keyframe; β is shared/averaged across frames
- 2D keypoint detector (e.g., RTMPose) → keypoints per keyframe
- Surface normal predictor (e.g., Sapiens-Normal) → normal map per keyframe

Output: `Stage1Result`.

### Stage 2 — Sculpt
Test-time optimization of per-vertex offset ΔV [6890, 3] in canonical T-pose space, supervised against Stage 1 outputs.

Loss terms (combined with tunable weights):
- Silhouette IoU against SAM 2 masks
- Normal map agreement against predicted normals
- 2D keypoint reprojection
- Mesh Laplacian smoothing (regularizer)
- Part-aware symmetry (weak/none on belly to preserve fat distribution)
- Height match to user input
- Surface normal consistency

Hyperparameters (learning rate, iteration count, loss weights) are tuned empirically in M8.

Output: `Stage2Result` (β unchanged, ΔV new).

### Stage 3 — Export & dual output

Produces two parallel outputs:

**Display branch (for frontend viewer):**
- Apply ΔV in canonical space
- LBS to a standardized A-pose (conservative arm spread to minimize LBS artifacts)
- Light cleanup of LBS-affected regions if needed
- Export as GLB
- Generate thumbnail

**Analysis branch (for Layer 2):**
- Compute `theta_natural`: a representative natural-standing pose derived from the keyframes
- Preserve `delta_v`, `beta`, `vertices_canonical`, joint positions
- Preserve per-keyframe `theta` as auxiliary data

Output: `Stage3Result`.

---

## Data handoff to Layer 2

`Stage3Result` is the contract with future Layer 2 code.

**For display (consumed by frontend viewer):**
- `vertices_a_pose: [6890, 3]` — final mesh in standardized A-pose
- `glb_path: Path`
- `thumbnail_path: Path`

**For Layer 2 analysis:**
- `vertices_canonical: [6890, 3]` — T-pose mesh with ΔV applied
- `delta_v: [6890, 3]` — per-vertex offset in canonical space
- `beta: [10]` — SMPL shape parameters
- `theta_natural: [24, 3]` — representative natural standing pose
- `theta_per_keyframe: [N, 24, 3]` — all keyframe poses
- `joints_canonical: [24, 3]`, `joints_natural: [24, 3]`
- `scale_to_meters: float`
- `quality: QualityReport` — at minimum an overall score and a list of warnings

Layer 2 will be a separate module that imports Layer 1's `Stage3Result`.

---

## Tech stack

- Python 3.10
- PyTorch 2.4.1 + CUDA 12.1
- PyTorch3D (differentiable rendering)
- SAM 2 (Meta, Apache 2.0)
- 4D Humans (Berkeley, non-commercial license — MVP only)
- RTMPose via `rtmlib` 0.0.15 + `onnxruntime-gpu` 1.19.2 (Apache 2.0) — YOLOX-m detector + RTMPose-m body7 ONNX
- Sapiens-Normal 0.3B torchscript (Meta, Sapiens License — non-commercial, MVP only)
- smplx (Max Planck, non-commercial license — MVP only)
- trimesh, Click, loguru, pydantic

No web service yet. Run via CLI:

    python scripts/run.py data/input_videos/test.mp4 --height 170 --weight 65 --gender neutral

## Hardware

- Target: NVIDIA GPU with ≥16GB VRAM
- Developed on: RTX 3090 (24GB), Driver 570, CUDA 12.1, Ubuntu

---

## Model checkpoints

Model files live in `checkpoints/<name>/`. Versions chosen per milestone. SMPL requires manual download at smpl.is.tue.mpg.de.

Currently downloaded:
- `sam2/sam2.1_hiera_small.pt` (M3)
- `smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` (M4; 4D Humans expects v1.0.0)
- `4dhumans/` — symlinks to `~/.cache/4DHumans/` (hmr2 source hardcodes the cache path; symlinks are decorative)
- `rtmpose/yolox_m.onnx` + `rtmpose/rtmpose_m_body7.onnx` (M5; rtmlib 'balanced' mode)
- `sapiens/sapiens_0.3b_normal_render_people_epoch_66_torchscript.pt2` (M5; 0.3B chosen for VRAM headroom — see NOTES.md 2026-05-20)

---

## File layout

    body-mvp/
    ├── PROJECT.md, CLAUDE.md, NOTES.md, README.md
    ├── pyproject.toml, .env.example, .gitignore
    ├── requirements_frozen.txt
    ├── checkpoints/{sam2, 4dhumans, rtmpose, sapiens, smpl}/
    ├── data/{input_videos, runs}/
    ├── scripts/{download_models.sh, run.py}
    ├── body_mvp/
    │   ├── config.py
    │   ├── pipeline.py
    │   ├── stage1.py
    │   ├── stage2.py
    │   ├── stage3.py
    │   ├── models.py
    │   ├── losses.py
    │   ├── lbs.py
    │   ├── render.py
    │   └── utils.py
    ├── viewer/{index.html, viewer.js}
    └── tests/fixtures/

Each stage lives in a single file by design. `body_mvp/` is a flat package, not nested submodules.

---

## Development milestones

- [x] **M1 — Scaffolding**
  - Goal: runnable CLI skeleton with proper file structure
  - Acceptance: `python scripts/run.py --help` works; running on a video prints "not implemented yet"

- [x] **M2 — Video to keyframes**
  - Goal: extract a set of keyframe images from the spinning video
  - Acceptance: running the CLI on `test.mp4` produces keyframe jpgs in `data/runs/<id>/keyframes/` that visibly cover different angles of the subject

- [x] **M3 — Keyframes to masks**
  - Goal: SAM 2 produces a clean person mask for each keyframe
  - Acceptance: mask-overlay visualization shows the person cleanly segmented across all keyframes

- [x] **M4 — Keyframes to SMPL β/θ**
  - Goal: 4D Humans gives shape and pose parameters per keyframe
  - Acceptance: rendering the SMPL mesh overlaid on each original keyframe shows reasonable alignment with the body silhouette

- [x] **M5 — Keypoints + Normal maps**
  - Goal: per-frame 2D keypoints and surface normal predictions
  - Acceptance: keypoint overlay and normal map visualizations look sensible per frame

- [x] **M6 — Stage 1 end-to-end**
  - Goal: unified `Stage1Result` produced from a single CLI run, saved to disk
  - Acceptance: can load `Stage1Result` from disk in a REPL and inspect all fields

- [x] **M7 — Stage 2 minimal optimization**
  - Goal: ΔV optimization loop runs with just silhouette loss (+ basic regularizer)
  - Acceptance: loss decreases over iterations; final ΔV is non-zero; mesh silhouette visibly closer to mask than initial SMPL

- [ ] **M8 — Stage 2 full loss + tuning**
  - Goal: all loss terms enabled and weights tuned on the test video
  - Acceptance: 4-view comparison renders (sculpted mesh vs original keyframes) look like the user
  - Note: this is the longest milestone

- [ ] **M9 — Stage 3 dual output**
  - Goal: produce display GLB + complete `Stage3Result`
  - Acceptance: GLB opens in a standard GLB viewer and shows a recognizable human in A-pose; `Stage3Result` has all defined fields populated correctly

- [ ] **M10 — Web viewer**
  - Goal: in-browser viewer with rotate/zoom
  - Acceptance: user can interact with the mesh in a browser; material has some visual polish

- [ ] **M11 — Evaluation**
  - Goal: validate on multiple volunteers
  - Acceptance: for 5+ subjects, produce comparison renders and collect self-similarity scores

---

## Implementation pitfalls (known traps)

Things that have bitten others working on similar pipelines:

1. PyTorch3D fails to install when built from source; prebuilt wheels matching the PyTorch+CUDA combo work reliably. (Already handled in this environment.)
2. SMPL model files require manual download with account registration. Can't be fully automated.
3. Stage 2 optimization tends to explode on first attempts — mesh vertices flying off, NaN losses. Fixing this (lr, weights, gradient clipping) is part of M7/M8 work.
4. LBS in Stage 3 produces "candy-wrap" artifacts at joints when target pose differs significantly from photo pose. Mitigated by keeping the target A-pose conservative.
5. Loose clothing produces bad silhouettes and bad reconstruction. No algorithmic fix in MVP; relies on user wearing tight clothes.
6. Per-frame β from 4D Humans is noisy; shape parameters should be shared or averaged across keyframes.
7. PyTorch3D's silhouette renderer requires a soft/differentiable shader (e.g., `SoftSilhouetteShader`) for optimization; the default rasterizer is non-differentiable.
8. SAM 2 prompted with only a near-image-width box can return the *inverted* (background) region as its argmax candidate when the person occupies a narrow vertical strip inside the box (e.g. spread arms). All three `multimask_output=True` candidates can score below 0.30 with no good alternative to re-pick. Fix: combine the box with a small set of positive-point prompts placed on torso-core keypoints (shoulders + hips, ± nose), filtered by per-point confidence. The box alone is sufficient only for narrow person crops.

---

## Current status

M7 complete. `pipeline.run()` now runs Stage 1 **and** Stage 2, returning `(Stage1Result, Stage2Result)`. Stage 2 is a test-time optimization of the per-vertex offset ΔV `[6890, 3]` in canonical T-pose space, supervised by a single silhouette IoU loss against the SAM masks plus a Laplacian smoothing regularizer. ΔV is the only optimized parameter; β and per-frame θ are frozen. `Stage2Result` (`delta_v`, frozen β/θ, loss history, per-frame init/final IoU) is persisted as `<run_dir>/stage2_result.npz` with a bit-exact round-trip self-check; before/after silhouette overlays and a loss curve land in `<run_dir>/stage2/`.

Architectural decisions baked in by M7:
- **ΔV → posed mesh calls `smplx.lbs.lbs()` directly**, with ΔV added into the canonical mesh (`v_canonical = v_template + blend_shapes(β) + ΔV`) before per-frame LBS.
- **Camera path follows M4's, with two corrections**: hmr2's `focal_length` is in 256-crop space (rescaled by `max(Wt,Ht)/256`), and no R_180x flip is applied. M9's A-pose render reuses this path. See NOTES 2026-05-22 for the derivation.
- **Hyperparameters (lr, weights, iters, grad-clip) live in `config.py`**; M7 uses a deliberately minimal loop (no lr schedule / early stop / recovery branches). Tuning is M8.
- **M7 is silhouette-only**: silhouette IoU is semantically blind, so the final mesh does not correct hair / shoe / clothing-edge contamination or limb-pose mismatch. Those are M8's scope.

Latest end-to-end run: `data/runs/20260522_001631/` on `test.mp4` — loss decreased monotonically, final ΔV non-zero, mean per-frame IoU improved on all 12 keyframes, both round-trip self-checks pass. Next: M8.

## Out of scope for MVP

- Multi-photo input mode (video only)
- Coarse-to-fine optimization
- Camera yaw joint optimization
- Pose-dependent corrective shapes
- User authentication, database, job queue
- Mobile native app
- Production-grade error handling and retries
- Feature extraction, LLM integration, health advice (Layers 2 and 3)