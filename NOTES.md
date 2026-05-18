# Development notes

Running log of decisions, parameter experiments, and open questions. See CLAUDE.md for update rules.

---

## 2026-05-17 — Project kickoff

**Done:**
- Initial project scaffolding (folders, docs)
- Environment set up: conda env `bodymvp`, Python 3.10, PyTorch 2.4.1 + CUDA 12.1, PyTorch3D verified on RTX 3090
- PROJECT.md and CLAUDE.md drafted; iterated several times to clarify the project/collaboration boundary

**Decisions:**
- Stage 3 must produce dual output: A-pose mesh (display) + analysis data (posture, shape) for Layer 2
- Each stage stays in a single file; flat package layout, not nested
- Hyperparameters live in config.py, not locked in PROJECT.md
- Model versions chosen per-milestone, not specified upfront

**Open questions:**
- Specific algorithm for `theta_natural` — to be decided in M9
- Whether to use medoid frame or geodesic rotation averaging

**Next:**
- M1b: code scaffolding (Click CLI, empty stage stubs, config.py)

---

## 2026-05-18 — M1 scaffolding complete

**Done:**
- Created all skeleton files: `body_mvp/` package (config, pipeline, stage1/2/3, models, losses, lbs, render, utils), `scripts/run.py`, `scripts/download_models.sh`, `viewer/index.html`, `viewer/viewer.js`, `pyproject.toml`
- Installed as editable package (`pip install -e .`)
- M1 acceptance verified: `--help` shows correct CLI; running on test video produces 4 expected log lines (pipeline.run + stage1/2/3), all via loguru to stderr, no errors

**Decisions:**
- `.env` vs `config.py` split: deployment config (device, paths) in `.env` via Pydantic Settings; algorithm hyperparameters (`NUM_KEYFRAMES`, `OPT_MAX_ITERS`, `LEARNING_RATE`, `LOSS_WEIGHTS`, `RENDER_RESOLUTION`) as module-level constants in `config.py` — not env-configurable
- Stage 2/3 sub-task stubs use `*args, **kwargs` — signatures deferred to their respective milestones to avoid locking in API decisions prematurely
- torch/pytorch3d excluded from `pyproject.toml` dependencies — CUDA wheels not on PyPI, must be installed manually

**Open questions:**
- None new; prior open questions (theta_natural algorithm, rotation averaging) carry forward to M9

**Next:**
- M2: keyframe extraction from spinning video (`stage1.sample_keyframes`)

---

## 2026-05-18 — M2 video to keyframes

**Done:**
- Implemented `extract_keyframes(video_path, run_dir) -> list[Path]` in `pipeline.py`
- Added `_create_run_dir` helper, timestamp-based `run_id`, per-run loguru file sink, and `meta.json` written at run start
- 12 keyframes extracted with uniform linspace sampling; all visually cover different spin angles
- M2 acceptance criterion met

**Decisions:**
- `extract_keyframes` placed in `pipeline.py`, not `stage1.py` — video I/O is pipeline orchestration, not stage logic (model inference)
- `exist_ok=False` on run dir creation — fail-loud on same-second collision rather than silently overwriting a prior run
- `run_id` is `YYYYMMDD_HHMMSS` for free chronological ordering via `ls`
- Don't trust `CAP_PROP_FRAME_COUNT`; probe backwards for actual last readable index, then `linspace(0, last_readable, N)` — this video over-reported by 25 frames (351 vs 326 actual), which would have broken any fixed-offset workaround
- `stage1.py` stub left as `*args, **kwargs` — real signature deferred to M3–M6

**Surprises:**
- Codec over-report was 25 frames, not 1. Validates probe-over-offset approach.

**Known debt:**
- `logger.add()` inside `run()` accumulates sinks if pipeline is called multiple times in one process; harmless for single-shot CLI, needs lifecycle fix before tests or web service

**Next:**
- M3: SAM 2 person mask per keyframe

---

## 2026-05-18 — M3 keyframes to masks (partial)

**Done:**
- SAM 2.1 small installed via `pip install --no-deps git+https://github.com/facebookresearch/sam2.git`
- `segment_keyframes(keyframe_paths, run_dir)` implemented in `stage1.py`: heuristic center-crop box → `SAM2ImagePredictor` → highest-scoring mask (multimask_output=True, pick argmax score)
- Heuristic box: center 80% of frame width (10% left/right margin), 90% of frame height (5% top/bottom margin) — exact starting point when M5 revisits with keypoint-guided box
- Outputs per keyframe: `masks/mask_NN.png` (binary uint8 0/255) and `masks/overlay_NN.png` (green 40% overlay)
- `sam2_checkpoint` added to `Settings` in `config.py`; `.env.example` and `download_models.sh` updated
- `requirements_frozen.txt` regenerated with SAM-2 pinned to commit 2b90b9f5
- Full pipeline runs end-to-end on test.mp4 without errors

**Result:**
- 8/12 frames clean (front and back views): full person covered, edges acceptable for MVP
- 4/12 frames failed (03, 04, 08, 09 — side-profile views): SAM 2 selected the open wardrobe door instead of the person; heuristic center-crop box captures the door from this angle

**Decisions:**
- SAM 2 must be installed with `--no-deps`; its `torch>=2.5.1` metadata pin would upgrade our torch 2.4.1+cu121 and break PyTorch3D. Documented in `pyproject.toml` comment.
- General rule for future model installs: always use `pip install --no-deps` first, then audit the package's declared dep list manually and install only the non-conflicting ones.
- Highest-score mask strategy locked in after smoke-test inspection; score gap large for good frames (0.85–0.95) vs. failed frames (0.22–0.35)
- M3 acceptance criterion not fully met. Decision: do not add a person detector. M5 introduces a keypoint detector that provides a person bbox for free. Revisit M3 after M5: replace `_make_box_prompt` with keypoint-bbox-derived prompts. Until then, M3 masks are partial and not safe for Stage 2.

**Surprises:**
- SAM 2's `torch>=2.5.1` pip metadata caused an unintended torch 2.12.0 upgrade on first install, overwriting CUDA 12.1 libraries with CUDA 13 variants. Required: (1) uninstall 14 cu13 packages, (2) force-reinstall torch 2.4.1+cu121 from the cu121 wheel index, (3) pin numpy back to <2.
- `nvidia-cudnn-cu13` overwrote `libcudnn.so.9` from `nvidia-cudnn-cu12`; uninstalling cu13 left cuDNN missing entirely, causing `CUDNN_STATUS_NOT_INITIALIZED`. Root fix: force-reinstall torch restores all pinned CUDA deps atomically.

**Open questions:**
- After M5: which keypoints to use for the person bbox? Likely min/max of all detected keypoints with a small padding margin.

**Note:** Claude Code's file-based memory system (`~/.claude/projects/.../memory/`) was discovered to auto-write session notes without approval; all files wiped and the feature disabled via CLAUDE.md rule during M3 wrap-up.

**Next:**
- M4: 4D Humans → SMPL β/θ per keyframe