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