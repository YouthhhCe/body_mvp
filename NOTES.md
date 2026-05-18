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