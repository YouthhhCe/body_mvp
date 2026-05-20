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

---

## 2026-05-19 — M4 third_party setup (user-driven, not CC)

**Done:**
- Cloned 4D Humans (`shubham-goel/4D-Humans` @ `efe18de`) to `third_party/4D-Humans/`
- Installed hmr2 as editable package: `pip install -e . --no-deps`, then manually audited `setup.py` and installed inference-only deps (skipped detectron2, training-only hydra extras, pyrootutils)
- SMPL v1.0.0 neutral placed at `checkpoints/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`; symlinked to `~/.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl` for hmr2's expected path
- 4D Humans checkpoint lives in `~/.cache/4DHumans/`; mirrored into `checkpoints/4dhumans/` via symlinks (decorative only — hmr2 source hardcodes the cache path)
- Verified end-to-end: `from hmr2.models import load_hmr2` + forward pass on example image → output shapes match data contract (`betas (1,10)`, `body_pose (1,23,3,3)`, `global_orient (1,1,3,3)`)
- Re-ran M3 pipeline: 8/12 mask clean, 4/12 failed (03/04/08/09) — no regression

**Decisions:**
- User does this milestone's setup, not CC. Multi-variable env stabilization (repo state, dep audit, torch compat, checkpoint paths, SMPL version mismatch) is CC's weak spot. Tagged `m3-end` before starting; can replay later as a CC capability test.
- Skipped `[all]` extras (detectron2). Our stage1.py will call `load_hmr2()` directly; person bbox comes from SAM 2 mask (interim) and RTMPose (M5+). Avoids ~4GB CUDA-toolkit install.
- SMPL v1.0.0 (39MB) vs v1.1.0 (247MB): v1.1.0 adds dynamic blendshapes hmr2 doesn't use. v1.0.0 is smaller, matches what 4D Humans README expects, fully sufficient.
- hmr2 outputs rotation matrices; stage1.py will convert to axis-angle via `pytorch3d.transforms.matrix_to_axis_angle` to match data contract `theta: [24, 3]`.

**Surprises:**
- `hmr2_data.tar.gz` from UT Austin (`www.cs.utexas.edu/~pavlakos/`) silently truncated at 352MB (full ~1.1GB+). `download_models()` has no integrity check; `os.system("tar -xvf ...")` swallowed the EOF error. Result: ckpt was 367MB instead of ~2.7GB, `model_config.yaml` missing entirely.
- **Recovery**: pulled `model_config.yaml` (3KB) and full ckpt (2.7GB) from HuggingFace Space `brjathu/HMR2.0`. HF was stable from CN; UT Austin was not. Kept truncated tar.gz in cache with a `README_DO_NOT_DELETE_TAR.txt` so `download_models()` won't re-download.
- chumpy mattloper `580566ea` already patches numpy 1.24+ deprecations — `import chumpy` works on numpy 1.26.4 with no manual sed.
- pytorch-lightning 2.6.1 auto-upgrades 4D Humans' v1.8.1 ckpt format on each load (non-destructive, benign warning).

**Env delta:**
- 54 new pip packages (see `requirements_frozen.txt` diff). No version changes to torch / pytorch3d / numpy / opencv / sam2.
- Env name is `dyc_bodymvp` (HANDOFF M3 wrote `bodymvp` — update).

**Next:**
- M4 implementation by CC: write `stage1.extract_smpl_params(keyframe_paths, run_dir)`. Per-frame `.npz` with betas + body_pose (axis-angle) + global_orient + camera. Acceptance: SMPL mesh overlaid on 4-6 keyframes via hmr2's built-in `Renderer`, covering frontal / side / back angles including the 4 SAM-failed frames.

---

## 2026-05-19 — M4 implementation complete + cross-machine migration test

**Done:**
- Wrote `extract_smpl_params(keyframe_paths, mask_paths, run_dir)` and `render_smpl_overlays(keyframe_paths, npz_paths, run_dir)` in `body_mvp/stage1.py`. Not integrated into `pipeline.run()` — that's M6 (Stage1Result assembly).
- M4 acceptance passed on both machines: 12 .npz files (betas, body_pose axis-angle, global_orient axis-angle, pred_cam, pred_cam_t, focal_length, bbox, bbox_source) + 12 mesh overlay PNGs. Silhouette alignment visually OK across frontal / side / back views including the 4 SAM-failed frames.
- Cross-machine migration: copied project to a second LAN server (Ubuntu 20.04, same RTX 3090) by git clone + rsync + env rebuild. Full M3 sanity + M4 acceptance reproduced. Mask coverage matched the source machine to the third decimal place (deterministic).

**Decisions:**
- hmr2 outputs rotation matrices; we convert to axis-angle (per data contract) via `pytorch3d.transforms.matrix_to_axis_angle`. Per-frame betas saved as-is, averaging deferred to M6.
- Bbox source per frame: SAM 2 mask bbox when coverage ≥ 8% (8 frames), else fall back to M3's heuristic center-crop box (4 frames). 8% threshold derived from M3 run data; documented as module constant in stage1.py.
- `render_smpl_overlays` is a separate function from inference (a `.npz` reader). Pipeline doesn't call it; it's an acceptance tool. Rationale: decouple inference success from rendering env fragility.
- 4 npz fields beyond betas/pose were added (pred_cam, pred_cam_t, focal_length, bbox + bbox_source) — these are free outputs of the hmr2 forward; storing now avoids re-running inference in M6/M9.

**Surprises:**
- Migration testing revealed multiple gaps in our setup documentation: pytorch3d wheel URL pattern, SAM 2 install ordering, chumpy `--no-build-isolation`, hydra-core as a separate SAM 2 dep, pyrender mandatory even for inference-only paths. All consolidated into a separate `MIGRATION_GUIDE.md` rather than expanded here.
- Found and fixed a real M1 scaffolding bug: `pyproject.toml` listed `sam2 @ git+...` but SAM 2's PyPI metadata declares `sam-2` (hyphenated). `pip install -e .` fails on the name mismatch. Fix: removed the line from pyproject; SAM 2 install is now external-only (already documented in the comment).
- pyrender worked on the second server's headless GL stack without intervention. The fallback plan (keypoint reprojection) was prepared but not needed.

**Known debt / non-issues:**
- M3 mask debt unchanged (4 side-view frames still misfire; deferred to post-M5).
- `pipeline.run()` still returns "not implemented yet" — intentional, M6 integrates all Stage 1 sub-tasks into `Stage1Result` then.
- hmr2 internal `vertices_to_trimesh` has a leftover `print(...)` that emits one line per render. Cosmetic noise, leave it.

**Migration food for thought (not now):**
- One-click migration paths worth considering when MVP-level shipping demands it: Docker image (heaviest, most portable), `conda-pack` for env snapshot + rsync for data (medium effort, big speedup), or just a more thorough `scripts/download_models.sh` covering all checkpoints. Out of scope for current milestones.

**Next:**
- HANDOFF M4 → M5 decision-reviewer.
- M5: 2D keypoints (RTMPose) + surface normals (Sapiens-Normal). After RTMPose lands, re-run M3 with keypoint-derived person bbox to fix the 4 failed side-view masks (long-standing debt).

---

## 2026-05-20 — M5 keypoints + normals + M3 mask debt fix

**Done:**
- `body_mvp/stage1.py`: added `extract_keypoints`, `render_keypoint_overlays`, `_keypoints_to_bbox`, `refine_masks_with_keypoints`, `extract_normals`, `render_normal_overlays`. New module constants for both pipelines pinned to `config.py` Settings (`rtmpose_det_checkpoint`, `rtmpose_pose_checkpoint`, `sapiens_normal_checkpoint`). Not integrated into `pipeline.run()` — that's M6.
- M5 acceptance passed on run `20260519_215818`: 12 `kp_NN.npz` + 12 `keypoints/overlay_NN.png` + 12 `normal_NN.npz` + 12 `normals/vis_NN.png`. Keypoint overlays show clean COCO-17 skeletons on all 12 frames including the 4 previously mask-failed side views. Normal maps show anatomically consistent camera-frame normals (smooth surface gradients, distinct front/back/limb regions).
- M3 mask debt resolved (`refine_masks_with_keypoints`): the 4 side-view frames (03/04/08/09) were re-run with a keypoint-derived bbox (min/max of keypoints with score ≥ 0.3, padded 10% per side) as SAM's box prompt. Old SAM scores 0.22–0.35 → new 0.92–0.97. Coverages 4.2–6.9% → 9.3–12.8%. Wardrobe door correctly excluded across all 4 refined frames; M3 milestone now ticked complete.

**Decisions:**
- **Mask-failure detection criterion: single `coverage < 9%`.** Mask coverage data on the run showed a clean 4.4-pp gap (good frames 11.3–14.6%, failed 4.2–6.9%); 9% sits in the gap with margin both ways. A 2-of-3 voting alternative (coverage + aspect h/w + fill-in-bbox) was proposed but rejected — single signal works on this data, multi-signal is over-engineering for a one-off fix on one run (CLAUDE.md "no future-flexibility layers").
- **RTMPose model variant: rtmlib `Body` 'balanced' mode (yolox-m det + rtmpose-m body7 pose, both ONNX).** The 4 failed-mask frames have the subject clearly visible (just off-center); yolox-m at 640×640 is more than enough. lightweight's yolox-tiny at 416×416 would be a real risk on side views; performance's yolox-x adds ~400 MB without buying anything M5 acceptance needs.
- **Sapiens-Normal checkpoint: 0.3B torchscript.** Decision chain in M5 setup: tried 1B → OOM on RTX 3090 (>24 GB peak); 0.6B fits at ~18.9 GB but leaves only ~5.4 GB headroom — too tight for M7+ when PyTorch3D differentiable rendering adds VRAM pressure; 0.3B at ~14.1 GB peak (setup-time measurement) leaves ~10.3 GB headroom. Accepting the quality tradeoff for the headroom. License is Sapiens License (non-commercial), consistent with the existing 4D Humans / SMPL stance.
- **Sapiens preprocess locked in (NOT the same as ImageNet vision defaults):** BGR channel order (no swap from `cv2.imread`), pixels stay in 0–255 (no /255), `mean=[123.5,116.5,103.5]`, `std=[58.5,57.0,57.5]`. Input shape `[1, 3, 1024, 768]` (H=1024, W=768; `cv2.resize` takes (W,H)). Output is half-resolution `[1, 3, 512, 384]` — bilinear-upsample before applying foreground mask. Wrap forward in `torch.inference_mode()` to avoid OOM across frames. Documented in the `extract_normals` docstring and `_SAPIENS_*` module constants.
- **Sapiens output coordinate frame: CAMERA, not world.** Predictor has no way to know world orientation. Stored normals are RAW (not necessarily unit length — Sapiens output magnitudes 0.95–0.97 across frames; visualization normalizes per-pixel before mapping `(n+1)/2*255`). Aligning camera-frame normals to SMPL/world frame is M8 work for the normal-consistency loss.
- **Feed FULL keyframe to Sapiens, not a person crop.** Per Sapiens-Pytorch-Inference project notes, cropping degrades quality even with generous padding. Background gets zeroed out post-inference using the SAM mask.

**Surprises:**
- **Peak VRAM discrepancy on Sapiens 0.3B:** M5 setup smoke test measured ~14.1 GB peak (presumably from nvidia-smi during JIT warmup on the first forward, which includes allocator reserve + kernel compilation); actual inference loop on 12 frames measured 2.72 GB via `torch.cuda.max_memory_allocated()` (tensor-only, post-warmup, after `torch.inference_mode`). The two numbers measure different things — don't conflate. The real number for steady-state inference is closer to ~3 GB; the 14.1 GB figure is the safer estimate when budgeting M7/M8 alongside PyTorch3D rendering, since first-forward JIT warmup will recur whenever the model is re-loaded in a fresh process.
- **Plan-doc misstatement caught mid-execution:** the install plan I (CC) wrote claimed rtmlib's default detector is "YOLOX-Nano-Person (~12 MB)". Reading rtmlib source mid-task showed rtmlib's `Body` wrapper offers only `lightweight`/`balanced`/`performance` modes using YOLOX-tiny / -m / -x respectively — no Nano variant exists in rtmlib's defaults. The "~12 MB" was also wrong; smallest available is YOLOX-tiny at ~18 MB on the wire. Recording the correction here so future-me doesn't rediscover the discrepancy. Pattern to remember: don't write package internals from memory in approved plans; verify against source first.

**Implementation pitfalls noted (not architecture-level; not promoting to PROJECT.md):**
- `pip install rtmlib` declares `onnxruntime` (CPU) and `opencv-contrib-python` as deps. The CPU onnxruntime would shadow `onnxruntime-gpu`, and contrib opencv conflicts with plain `opencv-python`. Always install with `--no-deps` (same general rule SAM 2 introduced in M3).
- ONNX Runtime providers must be probed after install — listing `CUDAExecutionProvider` is necessary but not sufficient evidence the model runs on GPU. Our smoke test confirmed via timing + GPU utilization. The "Some nodes were not assigned to the preferred execution providers" warning from ORT is benign (shape ops kept on CPU for perf).
- rtmlib's `Body.__call__` doesn't expose intermediate bboxes. We construct `YOLOX` and `RTMPose` directly so we can store the detector bbox in the per-frame `.npz` (downstream Stage 2 keypoint reprojection loss might want it; also useful for the M3 mask refine).
- When zero persons detected (didn't happen on test.mp4, but coded for it): save zeros + `bbox_source="none"` so downstream `refine_masks_with_keypoints` can skip cleanly rather than corrupt SAM with a zero bbox.

**Open questions:**
- For Stage 2 keypoint reprojection loss (M7/M8): which subset of COCO-17 maps cleanly to SMPL's 24 joints? Standard answer is a fixed mapping (shoulders, elbows, wrists, hips, knees, ankles); face keypoints (nose, eyes, ears) don't have SMPL analogues. Defer to M7.
- Boxer-shorts pattern dotted-hole artifact in normal vis: the printed text on the underwear gets segmented as background by SAM 2, zeroing the normals there. Small in area, won't materially affect Stage 2. Leave it for now; if it bites, the fix is to dilate the SAM mask slightly before zeroing — but that's a Stage 2-era decision.

**Next:**
- M6: `pipeline.run()` end-to-end integration. Aggregate keyframes + masks + SMPL params + keypoints + normals into a unified `Stage1Result` dataclass, persist to disk under `data/runs/<id>/stage1_result.npz` (or similar), and verify load-from-disk in a REPL inspects all fields cleanly.