# Development notes

Running log of decisions, parameter experiments, and open questions. See CLAUDE.md for update rules.

---

## 2026-05-17 — Project kickoff

**Done:** Initial scaffolding (folders, docs); conda env `bodymvp`, Python 3.10, PyTorch 2.4.1 + CUDA 12.1, PyTorch3D verified on RTX 3090.

**Decisions:**
- Stage 3 produces dual output: A-pose mesh (display) + analysis data (posture, shape) for Layer 2
- Each stage in a single file; flat package layout
- Hyperparameters in `config.py`, not locked in PROJECT.md
- Model versions chosen per-milestone, not upfront

**Open questions:** `theta_natural` algorithm — medoid frame vs geodesic rotation averaging. Defer to M9.

---

## 2026-05-18 — M1 scaffolding

**Done:** Package skeleton + editable install + CLI smoke test passed.

**Decisions:**
- `.env` for deployment config (device, paths) via Pydantic Settings; algorithm hyperparameters as module-level constants in `config.py`
- Stage 2/3 stubs use `*args, **kwargs` — signatures deferred to their milestones, avoid premature API lock-in
- torch/pytorch3d excluded from `pyproject.toml` — CUDA wheels not on PyPI, install manually

---

## 2026-05-18 — M2 video to keyframes

**Done:** `extract_keyframes` in `pipeline.py`; 12 keyframes via probe-and-linspace. Acceptance met.

**Decisions:**
- Video I/O in `pipeline.py`, not `stage1.py` — stage files are for model inference, not orchestration
- `run_id = YYYYMMDD_HHMMSS`, `exist_ok=False` on run dir — fail-loud on same-second collision
- Don't trust `CAP_PROP_FRAME_COUNT`; probe backwards for actual last readable index. Test video over-reported by 25 frames (351 vs 326).

**Known debt:** `logger.add()` inside `run()` accumulates sinks if pipeline called multiple times per process; harmless for single-shot CLI.

---

## 2026-05-18 — M3 masks (partial)

**Done:** SAM 2.1 small via `pip install --no-deps`. `segment_keyframes` with heuristic center-80% box prompt → highest-scoring mask.

**Result:** 8/12 clean (front/back); 4/12 failed (03, 04, 08, 09 — side views) — SAM selected the wardrobe door instead.

**Decisions:**
- **`--no-deps` is mandatory.** SAM 2's `torch>=2.5.1` metadata pin would auto-upgrade torch 2.4.1+cu121 and break PyTorch3D. Standard rule for all future models: `--no-deps`, audit declared deps manually, install only non-conflicting ones.
- Highest-score mask strategy locked in. Score gap on good frames (0.85–0.95) vs failed (0.22–0.35) is large.
- **Don't add a person detector to fix M3.** M5 brings RTMPose which gives a person bbox for free; revisit then. Until then M3 is partial.

**Surprises:**
- SAM 2's pip metadata silently upgraded torch to 2.12.0 on first install, overwriting CUDA 12.1 libs with CUDA 13. Recovery: uninstall 14 cu13 packages, force-reinstall torch 2.4.1+cu121, pin numpy<2. `nvidia-cudnn-cu13` overwrote `libcudnn.so.9` from `nvidia-cudnn-cu12`; force-reinstall torch restores atomically.

**Note:** Claude Code's file-based memory system was discovered to auto-write session notes without approval. Wiped and disabled via CLAUDE.md rule.

---

## 2026-05-19 — M4 third_party setup (user-driven)

**Done:** 4D Humans (`shubham-goel/4D-Humans` @ `efe18de`) cloned to `third_party/`, installed `--no-deps`, audited setup.py manually. SMPL v1.0.0 neutral placed and symlinked into hmr2's expected cache path. End-to-end forward pass verified.

**Decisions:**
- **User does this milestone's setup, not CC.** Multi-variable env stabilization (repo state, dep audit, torch compat, checkpoint paths, SMPL version) is CC's weak spot. Tagged `m3-end` before starting as a future CC capability replay.
- Skipped `[all]` extras (detectron2 ~4 GB). hmr2 called directly; person bbox from SAM mask (interim) → RTMPose (M5+).
- SMPL v1.0.0 over v1.1.0: v1.1.0's dynamic blendshapes are unused by hmr2; v1.0.0 is what the README expects.
- hmr2 outputs rotation matrices; convert to axis-angle via `pytorch3d.transforms.matrix_to_axis_angle` to match data contract `theta: [24, 3]`.

**Surprises:**
- `hmr2_data.tar.gz` from UT Austin silently truncated at 352 MB (full ~1.1 GB+); `download_models()` has no integrity check. Recovery: pulled ckpt + config from HuggingFace Space `brjathu/HMR2.0`.

**Env name correction:** `dyc_bodymvp` (HANDOFF M3 wrote `bodymvp`).

---

## 2026-05-19 — M4 implementation + cross-machine migration

**Done:** `extract_smpl_params` and `render_smpl_overlays` in `stage1.py`. 12 .npz + 12 overlay PNGs. Not integrated into `pipeline.run()` — M6's job. Acceptance reproduced on a second LAN server (same RTX 3090, Ubuntu 20.04).

**Decisions:**
- Per-frame betas saved as-is; averaging deferred to M6.
- Bbox source per frame: SAM mask bbox when coverage ≥ 8% (8 frames), else heuristic center-crop fallback (4 frames). 8% threshold from M3 data.
- `render_smpl_overlays` is a separate `.npz` reader, not called by pipeline. Decouples inference success from rendering env fragility.
- Stored 4 extra .npz fields (pred_cam, pred_cam_t, focal_length, bbox + bbox_source) — free outputs from hmr2 forward; avoids re-running inference in M6/M9.

**Surprises:**
- Migration revealed setup-doc gaps: pytorch3d wheel URL pattern, SAM 2 install ordering, chumpy `--no-build-isolation`, pyrender mandatory even for inference-only paths. Consolidated into `MIGRATION_GUIDE.md`.
- Found M1 scaffolding bug: `pyproject.toml` listed `sam2 @ git+...` but PyPI metadata declares `sam-2` (hyphenated); editable install failed. Fixed by removing the line.

**Known debt:** M3 mask debt unchanged (deferred to post-M5). `pipeline.run()` still stub — M6 territory.

**Migration food for thought (future):** Docker / `conda-pack` + rsync / a more thorough `download_models.sh`. Out of scope until shipping.

---

## 2026-05-20 — M5 keypoints + normals + M3 mask debt fix

**Done:** `extract_keypoints`, `_keypoints_to_bbox`, `refine_masks_with_keypoints`, `extract_normals` + their overlay tools in `stage1.py`. M5 acceptance passed on run `20260519_215818` (12 keypoint .npz + 12 normal .npz + overlays). M3 mask debt resolved — 4 side-view frames re-run with keypoint-derived bbox; old SAM scores 0.22–0.35 → new 0.92–0.97. M3 milestone ticked complete.

**Decisions:**
- **Mask-failure detection criterion: single `coverage < 9%`.** Run data showed a clean 4.4-pp gap (good 11.3–14.6%, failed 4.2–6.9%). A 2-of-3 voting alternative was proposed but rejected per CLAUDE.md "no future-flexibility layers".
- **RTMPose: rtmlib `Body` 'balanced' mode** (yolox-m + rtmpose-m body7, ONNX). Lightweight's yolox-tiny at 416×416 is a real risk on side views; performance's yolox-x adds ~400 MB without buying anything.
- **Sapiens-Normal: 0.3B torchscript.** Tried 1B → OOM on RTX 3090 (>24 GB peak); 0.6B fits at 18.9 GB but only 5.4 GB headroom — too tight for M7+ when PyTorch3D rendering adds VRAM pressure; 0.3B at ~14.1 GB peak leaves ~10.3 GB headroom. Activation memory dominates over parameter count at 1024×768, so going smaller doesn't help much. License: Sapiens License (non-commercial), consistent with 4D Humans / SMPL.
- **Sapiens preprocess is NOT standard ImageNet:** BGR (no swap from `cv2.imread`), pixels in 0–255 (no /255), `mean=[123.5,116.5,103.5]`, `std=[58.5,57.0,57.5]`. Input `[1,3,1024,768]`, output is half-resolution `[1,3,512,384]` — upsample before applying mask. Numerics codified in `_SAPIENS_*` module constants.
- **Sapiens output is in CAMERA frame, not world.** Aligning to SMPL/world is M8 work. Stored normals are RAW (mean magnitudes 0.95–0.97, not unit).
- **Feed FULL keyframe to Sapiens, not a person crop** — cropping degrades quality even with generous padding (per Sapiens-Pytorch-Inference project notes). Background zeroed post-inference using SAM mask.

**Surprises:**
- **VRAM measurement discrepancy on Sapiens 0.3B:** smoke test showed ~14.1 GB peak (nvidia-smi during JIT warmup); inference loop showed 2.72 GB (`torch.cuda.max_memory_allocated()`, post-warmup). The numbers measure different things. Steady-state is ~3 GB; use the 14.1 GB figure for M7/M8 budgeting since first-forward JIT warmup recurs whenever model is loaded in a fresh process.
- **Plan-doc misstatement caught mid-execution:** approved plan claimed rtmlib's default detector is "YOLOX-Nano-Person (~12 MB)". Reading rtmlib source mid-task: no Nano variant exists; the smallest is YOLOX-tiny at ~18 MB. Pattern to remember: don't write package internals from memory in approved plans; verify against source first.

**Open questions:**
- COCO-17 → SMPL 24 joint mapping for Stage 2 reprojection loss: standard answer is fixed mapping (shoulders, elbows, wrists, hips, knees, ankles); face keypoints have no SMPL analogue. Defer to M7.
- Boxer-shorts printed pattern gets segmented as background by SAM, zeroing normals there. Small area, won't materially affect Stage 2. If it bites later: dilate SAM mask before zeroing.

**Next:** M6 — `pipeline.run()` end-to-end. Aggregate keyframes + masks + SMPL + keypoints + normals into a unified `Stage1Result`, persist to disk, verify load-from-disk in REPL.

---

## 2026-05-21 — M6 Stage 1 end-to-end (option c + SAM box+points)

**Done:** `pipeline.run()` runs Stage 1 end-to-end and persists `Stage1Result` to `<run_dir>/stage1_result.npz`. `Stage1Result` is a dataclass with `__post_init__` invariants and `save/load` (native numpy dtypes, no pickle). Round-trip self-check fires on every run. Acceptance on `data/runs/20260521_132028/` — all 12 masks visually correct, β stable to 3 dp across two consecutive runs.

**Decisions:**
- **Option (c): YOLOX bbox as single source of truth, shared by SAM and hmr2.** Eliminated the `_make_box_prompt` / `_mask_to_bbox` / `refine_masks_with_keypoints` dead-code path rather than keeping brittle gates. Diagnostic origin: M4 used heuristic fallback bbox on 4 side-view frames; M5 fixed masks but never re-ran SMPL — disagreement between SAM and SMPL about person location.
- **`theta_per_frame: [N, 24, 3]` merged in Stage1Result; smpl_NN.npz keeps hmr2's split layout.** 24-joint is the data contract; hmr2's split is upstream layout. Merge only at construction site.
- **β = per-component mean across keyframes.** `betas_per_frame` preserved for future re-aggregation (medoid / IQR-trimmed) without re-inference.
- **`stage1_result.npz` is metadata-only; pixel data via path references.** Round-trip self-check cheap enough to run every invocation.
- **`bbox_source == "none"` gate after extract_keypoints**: YOLOX failure raises with offender list; no silent fallback. That heuristic-fallback IS the bug option (c) fixes.

**SAM regression (resolved):** Wide YOLOX bbox (>80% image width) + person in narrow vertical strip → SAM's three multimask candidates all scored < 0.30 with the argmax often inverted (frames 06, 11). Fix: combine box with positive torso-keypoint points (COCO 0/5/6/11/12, score gate 0.5, fail-loud if <2 survive). Post-fix all 12 frames score 0.938–0.953. Captured as PROJECT.md pitfall #8.

**Surprises:**
- **Sapiens-Normal magnitude outliers (6.42, 6.88) were downstream of the SAM inversion**, not a Sapiens artifact. `extract_normals` multiplies Sapiens output by mask; inverted mask zeroed the wrong region. After SAM fix: 1.07, 1.14. Generalization worth carrying: anomalies on the same frames often share an upstream cause.
- **SAM score is not a per-frame mask quality signal.** Pre-fix, 4 visually-clean masks scored 0.40–0.54. Score reflects prompt-information sufficiency, not mask correctness. Stage 2's silhouette loss must NOT weight per-frame by SAM score.

**M7 must read — removed from `pipeline.run()`:**
- `stage2.run()` and `stage3.run()` calls removed (M1 stubs). **M7 must wire `stage2.run()` back into `pipeline.run()`** after the optimization loop is implemented, taking Stage1Result as input.

**Production backlog (don't act on now):**
- If box+points ever fails on a future frame, try `multimask_output=False` (per SAM 2 docs for multi-prompt cases) before adding fallback chains.
- `>= 2 torso points` rule doesn't enforce geometry. If masks ever bias upper/lower body, require `>= 1 shoulder AND >= 1 hip`.
- See Surprises re: SAM score and Stage 2 silhouette weighting.

**Open questions:**
- `_MASK_COVERAGE_THRESHOLD` and `_MASK_REFINE_COVERAGE_THRESHOLD` are now unused by the pipeline. Add "no longer pipeline-called" comment on the next stage1.py drive-by.
- Boxer-shorts pattern still segmented as background by SAM (M5 carryover, unchanged by box+points).
- `hmr2/datasets/vitdet_dataset.py:62` stray `print(f'{downsampling_factor=}')` — don't patch third_party.

**Next:** M7 — Stage 2 minimal optimization. ΔV loop + silhouette loss + basic regularizer. First task: wire `stage2.run()` back into `pipeline.run()`.