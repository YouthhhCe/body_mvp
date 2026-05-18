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