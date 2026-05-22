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

## 2026-05-19 — M4 SMPL β/θ via 4D Humans (user-driven setup)

**Done:** 4D Humans (`shubham-goel/4D-Humans` @ `efe18de`) cloned to `third_party/`, installed `--no-deps` with setup.py audited manually. SMPL v1.0.0 neutral placed + symlinked into hmr2's cache path. `extract_smpl_params` and `render_smpl_overlays` implemented in `stage1.py` — 12 .npz + 12 overlay PNGs. Not integrated into `pipeline.run()` (M6's job). Acceptance reproduced on a second LAN server.

**Decisions:**
- **User did this milestone's setup, not CC.** Multi-variable env stabilization (repo state, dep audit, torch compat, checkpoint paths) is CC's weak spot. Tagged `m3-end` before starting.
- Skipped `[all]` extras (detectron2 ~4 GB). hmr2 called directly; person bbox from SAM mask (interim) → RTMPose (M5+).
- SMPL **v1.0.0** over v1.1.0: v1.1.0's dynamic blendshapes are unused by hmr2; v1.0.0 is what the README expects.
- hmr2 outputs **rotation matrices**; convert to axis-angle via `pytorch3d.transforms.matrix_to_axis_angle` to match data contract `theta: [24, 3]`.
- Per-frame betas saved as-is; averaging deferred to M6.
- **Stored 4 extra .npz fields** (`pred_cam`, `pred_cam_t`, `focal_length`, `bbox` + `bbox_source`) — free outputs from hmr2 forward; avoids re-running inference in M6/M7/M9. (M7 consumes `pred_cam_t` + `focal_length` for camera setup.)
- `render_smpl_overlays` is a standalone `.npz` reader, not pipeline-called — decouples inference success from rendering env fragility.

**Carried over:** cross-machine migration surfaced setup-doc gaps (pytorch3d wheel URL, SAM 2 install ordering, chumpy `--no-build-isolation`, pyrender mandatory) — all consolidated into `MIGRATION_GUIDE.md`. Fixed M1 bug: `pyproject.toml` `sam2` line removed (PyPI metadata uses hyphenated `sam-2`, broke editable install). Env name is `dyc_bodymvp`.

---

## 2026-05-20 — M5 keypoints + normals + M3 mask debt fix

**Done:** `extract_keypoints`, `_keypoints_to_bbox`, `refine_masks_with_keypoints`, `extract_normals` + overlay tools in `stage1.py`. M5 acceptance passed. M3 mask debt resolved — 4 side-view frames re-run with keypoint-derived bbox (SAM scores 0.22–0.35 → 0.92–0.97); M3 ticked complete.

**Decisions:**
- **Mask-failure detection: single `coverage < 9%`.** Run data showed a clean 4.4-pp gap (good 11.3–14.6%, failed 4.2–6.9%). 2-of-3 voting alternative rejected per CLAUDE.md "no future-flexibility layers".
- **RTMPose: rtmlib `Body` 'balanced' mode** (yolox-m + rtmpose-m body7, ONNX). yolox-tiny risks side views; yolox-x adds ~400 MB for no gain.
- **Sapiens-Normal: 0.3B torchscript.** 1B → OOM on RTX 3090; 0.6B fits at 18.9 GB but leaves only 5.4 GB headroom (too tight for M7+ PyTorch3D rendering); 0.3B at ~14.1 GB peak leaves ~10.3 GB. Activation memory dominates over param count at 1024×768, so going smaller buys little. License: Sapiens (non-commercial).
- **Sapiens preprocess is NOT standard ImageNet** — BGR, pixels 0–255, custom mean/std, half-res output. Numerics codified in `_SAPIENS_*` constants in code.
- **Sapiens output is in CAMERA frame, RAW (non-unit).** Aligning to SMPL/world is M8 work — prerequisite for M8's normal loss.
- **Feed FULL keyframe to Sapiens, not a person crop** — cropping degrades quality. Background zeroed post-inference via SAM mask.

**Surprises:**
- **VRAM has two readings:** ~14.1 GB peak (`nvidia-smi`, includes JIT warmup) vs 2.72 GB steady-state (`max_memory_allocated`, post-warmup). Budget with the 14.1 GB figure — fresh-process warmup recurs on every load. (Codified as HANDOFF rule.)

**Open questions:**
- COCO-17 → SMPL-24 joint mapping for Stage 2 reprojection loss: standard answer is fixed mapping (shoulders/elbows/wrists/hips/knees/ankles); face keypoints have no SMPL analogue. Deferred to M8.
- Boxer-shorts printed pattern gets segmented as background by SAM, zeroing normals there. Small area. If it bites: dilate SAM mask before zeroing.

---

## 2026-05-21 — M6 Stage 1 end-to-end (option c + SAM box+points)

**Done:** `pipeline.run()` runs Stage 1 end-to-end, persists `Stage1Result` to `<run_dir>/stage1_result.npz`. `Stage1Result` is a dataclass with `__post_init__` invariants + `save/load` (native numpy dtypes, no pickle); round-trip self-check fires every run. Acceptance on `data/runs/20260521_132028/` — all 12 masks correct, β stable to 3 dp across two runs.

**Decisions:**
- **Option (c): YOLOX bbox is the single source of truth**, shared by SAM and hmr2. Eliminated the `_make_box_prompt` / `_mask_to_bbox` / `refine_masks_with_keypoints` dead-code path. Origin: M4 used heuristic fallback bbox on 4 side-view frames; M5 fixed masks but never re-ran SMPL — SAM and SMPL disagreed on person location.
- **`theta_per_frame: [N, 24, 3]`** merged in Stage1Result uses the 24-joint convention (joint 0 = global_orient, 1–23 = body_pose); per-frame `smpl_NN.npz` keeps hmr2's split layout. Merge only at construction.
- **β = per-component mean across keyframes.** `betas_per_frame` preserved for future re-aggregation without re-inference.
- **`stage1_result.npz` is metadata-only**; pixel data via path references.
- **`bbox_source == "none"` gate** after extract_keypoints: YOLOX failure raises with offender list, no silent fallback.
- **SAM prompted with box + torso-keypoint positive points** (COCO 0/5/6/11/12, score gate 0.5) — resolves the wide-bbox foreground/background inversion ambiguity. Captured as PROJECT.md pitfall #8.

**Surprises:**
- **SAM score is not a mask-quality signal.** Visually-clean masks scored as low as 0.40; score reflects prompt-information sufficiency, not correctness. **Stage 2's silhouette loss must NOT weight per-frame by SAM score.**
- Anomalies on the same frames often share an upstream cause — Sapiens magnitude outliers turned out to be downstream of the SAM inversion, not a Sapiens artifact.

**Open questions:**
- `_MASK_COVERAGE_THRESHOLD` / `_MASK_REFINE_COVERAGE_THRESHOLD` now unused by the pipeline — annotate on next stage1.py drive-by.
- Boxer-shorts pattern still segmented as background by SAM (M5 carryover).
- `hmr2/datasets/vitdet_dataset.py:62` stray `print` — don't patch third_party.

---

## 2026-05-22 — M7 Stage 2 minimal optimization

**Done:** Test-time ΔV optimization runs end-to-end. `render.py` and `losses.py` fleshed out (PyTorch3D `PerspectiveCameras` + `MeshRasterizer`/`SoftSilhouetteShader`; soft-IoU loss + uniform Laplacian smoothing). `stage2.py` is the core: `_pose_meshes`, `Stage2Result` (+save/load/round-trip), `optimize_vertex_offsets`, `save_silhouette_debug`, dev-only sanity renders. `stage2.run()` wired into `pipeline.run()` (now returns `tuple[Stage1Result, Stage2Result]`). Acceptance met on `data/runs/20260522_001631/`: loss 0.476→0.260 monotone over 200 iters, max|ΔV|=0.156, mean hard-IoU 0.780→0.884 (+0.10), all 12 frames improved. Wall ~6.6 s, peak VRAM ~0.5 GB.

**Decisions:**
- **ΔV path calls `smplx.lbs.lbs()` directly**, bypassing smplx's high-level forward. `v_canonical = v_template + blend_shapes(β) + ΔV`, then LBS per frame with detached θ. Monkey-patching `v_template` was rejected — it's a registered buffer, not a Parameter, so gradient would not flow to ΔV. β and θ_per_frame are `requires_grad=False`; ΔV is the only `nn.Parameter` (zero-init `[6890,3]`).
- **Camera transform reuses M4's path, with two corrections grounded in source** (not derived): (1) hmr2's `focal_length` is in **256-crop space** (raw 5000), so render-space focal = `fl / 256 * max(Wt, Ht)`. (2) **No R_180x flip** — hmr2's predicted `global_orient` already carries the image-frame adaptation (frame-0 ≈ 178° about X); PyTorch3D `in_ndc=False` has no OpenGL framebuffer flip to compensate for, unlike the pyrender path. `verts_world = verts_posed + pred_cam_t`, no axis flip. **M9's A-pose render reuses this exact camera path.** The "obvious" R_180x flip from hmr2's pyrender path is the trap.
- **Knobs locked, deliberately minimal** (in `config.py`): lr 1e-3, grad-clip-norm 1.0, 200 iters fixed, silhouette weight 1.0, uniform Laplacian weight 100.0. Pitfall #3 (explosion) defense is **grad clip + low lr only** — no lr schedule, early stop, patience, or recovery branches. M8 owns tuning.
- **`final_loss` persisted as float64, `loss_history` as float32.** Round-trip self-check is bit-exact, not `isclose`; float32 `final_loss` failed the exact compare. Everything else native numpy dtype, `allow_pickle=False`.
- **`save_silhouette_debug` is self-contained** — re-renders both the ΔV=0 baseline and optimized ΔV rather than copying from sanity outputs, removing an implicit ordering dependency in `pipeline.run`. `sanity_render_*` stay dev-only, NOT pipeline-called.
- matplotlib added (for loss curves; amortized for M8's multi-loss plots) — see `requirements_frozen.txt`.

**Surprises / memory aids for M8/M9:**
- **Mask-driven artifacts in final overlays (M7-expected, NOT a bug).** ΔV pulls vertices to match SAM mask boundaries that include hair / shoes / clothing edges, producing visible surface spikes. Silhouette IoU is semantically blind; uniform Laplacian can't tell "real body edge" from "hair edge". Concrete motivation for **M8's part-aware silhouette weighting + normal loss**.
- **Frame 09 (raised/forward arm) had the smallest IoU gain (+0.04).** Silhouette alone cannot fix a limb-*pose* mismatch — moving vertices in canonical space can't relocate a posed arm. **M8's keypoint reprojection loss is the intended remedy; flag frame 09 as an M8 test case.**
- **Peak VRAM ~0.5 GB**, far under the 1.2–1.5 GB pre-estimate. M8 has large headroom to raise render resolution / `faces_per_pixel` / add loss terms.
- **Duplicate SMPL load** — Stage 2 loads its own `smplx.SMPL` while hmr2 already loaded a structurally-compatible model in Stage 1. Fine for M7 (~0.3 s). Candidate for a single shared model once M8/M9 both need it.

**Open questions:**
- COCO-17 → SMPL-24 joint mapping for M8's keypoint reprojection loss still unresolved (carried from M5).
- Whether uniform Laplacian is enough regularization once M8's stronger silhouette pull lands, or whether `mesh_normal_consistency` is needed.

**Next:** M8 — Stage 2 full loss + tuning. Enable normal-map agreement, keypoint reprojection, height match, part-aware terms; tune weights on the test video. Longest milestone.
