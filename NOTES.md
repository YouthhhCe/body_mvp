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

---

## 2026-05-23 — M8 Stage 2 full loss + tuning (D6–D13, through pre-D14)

**Done:** All M8 loss terms implemented and integrated into `optimize_vertex_offsets`. Geometry turntable visualisation added. Two camera-level fixes (FIX1/FIX2) diagnosed and committed after turntable review. Work in progress: D14 weight tuning (not started yet).

### D-steps completed

**D6 — normal consistency loss term** (`mesh_normal_consistency`). Wired into the optimization loop via `normal_consistency_loss` in `losses.py`; initial weight 0.1.

**D7 — region weight infrastructure** (`_build_region_weights` in `stage2.py`). Builds three arrays: `silh_region_weights [6890]` (per-vertex part weights for weighted silhouette IoU, D8 predecessor), `sym_region_weights [6890]` (weights for bilateral symmetry loss), `right_to_left [6890]` (SMPL mirror-vertex index map via nearest-neighbour in canonical space with x-flip). Belly vertices zeroed in sym weights; non-involution vertices (where the nearest-neighbour map isn't a true involution) zeroed out after a self-consistency gate.

**D8 — part-aware silhouette weighting** (`render_weight_map` + `weighted_silhouette_iou_loss`). `render_weight_map` rasterizes the per-vertex silhouette weights to a [H,W] map using `pix_to_face` packed global indices (face_attrs must be [N×F, 3, 3], not [F, 3, 3] — this was a PyTorch3D gotcha). Weighted soft-IoU then scales intersection and union per-pixel.

**D9 — height match loss** (`height_loss` in `losses.py`). Squared-hinge loss on SMPL Y-span vs target height; ±5 cm tolerance band. Weight 10.0 (initial).

**D10 — keypoint reprojection loss** (`keypoint_reprojection_loss` in `losses.py`). COCO-17 → SMPL-24 fixed mapping (12 pairs: shoulders/elbows/wrists/hips/knees/ankles; face kps and nose excluded — SMPL joint 15 is neck top, no nose analogue). Score gate >0.3; projection via `fl*X/Z + cx`. Weight 1e-3 placeholder (pixel² vs IoU scale).

**D11 — bilateral symmetry loss** (`symmetry_loss` in `losses.py`). Penalises |ΔV[i] − mirror(ΔV[j])|² where j is the mirror partner (right_to_left map from D7). Flips x-only (canonical +x = subject's left). Normalised by total weight sum. Weight 0.01 (initial).

**D12 — Sapiens normal-map loss** (`normal_map_loss` in `losses.py`). Three-session diagnostic to get the sign convention and gate right — details below. Final: mean cosine-distance loss gated to interior face-on pixels. Two gates: (1) grazing gate |rendered_n_z| > 0.5, (2) edge erosion gate via 4-px max_pool dilation of background. Weight 0.5 (initial).

**FIX2 — median-tz cam_t correction** (`_median_tz_cam_t` in `stage2.py`). Physical prior: video is fixed camera, spinning subject — per-frame tz should be constant. hmr2's weak-perspective estimation introduced 3.1 m range (6.3%) across 12 frames on run 001631 (tz min=49.2, max=52.3, median=50.576 m). Helper clones cam_t, replaces `[:, 2]` with `median()`. Applied in both `optimize_vertex_offsets` and `save_silhouette_debug` (debug overlays must use same corrected cam_t as optimizer — not applying it there was an initial error, corrected before coding).

**FIX1 — orientation-balanced silhouette weighting** (`_frame_orientation_weights` in `stage2.py`). Run 001631 has 4 back-view frames (0, 1, 10, 11) spanning only 8.3° vs 360° total — each was receiving 4× the silhouette gradient of a front frame. Fix: bandwidth = π/N = 15° for N=12; count[i] = frames whose azimuth is within 15° of frame i; weight[i] = 1/count[i]. Back frames each counted 4 neighbours → weight 0.25; sum of 4 back-frame weights = 1.0, cancelling oversampling exactly. Azimuths derived from hmr2 global_orient axis-angle by extracting the forward-vector Y-component and Z-component and taking arctan2. In the optimizer loop: `L_silh = (per_frame_silh * frame_orient_weights).sum() / frame_orient_weights.sum()`.

**D13 — geometry turntable** (`save_geometry_turntable` in `stage2.py`). 12 fixed-elevation views equally spaced 0–330°; rendered via PyTorch3D `HardPhongShader` on a Meshes with a point light; saved as a montage strip to `<run_dir>/debug/turntable.png`. Used as the primary diagnostic for roughness.

### D12 four-round normal-map loss diagnosis

This took four distinct rounds before converging on the correct sign convention and gating.

**Round 1 — sign discovery.** Sapiens output convention is CAMERA space with non-standard sign: normals point AWAY from camera for surfaces facing camera, but the axis signs differ from PyTorch3D's camera normals. Ran an 8-flip SVD sweep (all 8 ±1 combinations of xyz flip signs applied to Sapiens normals before computing cosine similarity against rendered SMPL normals at ΔV=0):

| Flip sign (x, y, z) | Mean cosine similarity |
|---|---|
| (+1, −1, −1) **winner** | 0.812 |
| (−1, −1, −1) | 0.752 |
| (+1, +1, −1) | 0.621 |
| (−1, +1, −1) | 0.583 |
| (+1, −1, +1) | 0.421 |
| (−1, −1, +1) | 0.388 |
| (+1, +1, +1) | 0.312 |
| (−1, +1, +1) | 0.287 |

Winner: `_SAPIENS_NORMAL_FLIP = (1, -1, -1)` — x positive, y flipped, z flipped. Codified as a module-level constant applied in `_load_sapiens_normals` before normalising.

**Round 2 — diagnostic breakdown.** Computed cosine error separated by region at ΔV=0: face-on interior mean 0.060, edge region 0.41, grazing surfaces 0.53. 7× gap confirmed that edge/grazing pixels dominate the raw loss and corrupt the gradient signal — motivated the two-gate design.

**Round 3 — grazing gate implementation.** Gate on |rendered_n_z| > 0.5 (threshold tunable, set conservatively for D12; D14 tuning target). Both rendered_n_z and the threshold boundary are detached from the gradient graph so ΔV gradient flows through the cosine dot product only, not through mask selection.

**Round 4 — edge erosion gate.** Erode rendered_fg by 4 px via max_pool2d of the inverted fg mask (dilate background by 4 px, then invert). `.detach()` after erode. Combined mask: sapiens_fg & interior_gate & grazing_gate. Final mean cosine-distance loss on valid pixels only.

### Tuning results so far (before D14)

`w_lap` sweep at fixed other weights (silh=1.0, normal=0.5, etc.):
- w_lap=100 (initial): mean hard-IoU gain +0.077 over baseline, turntable shows moderate roughness
- w_lap=300: IoU gain +0.055, roughness unchanged
- w_lap=500: IoU gain +0.028, roughness not reduced — confirms over-regularization suppresses silhouette fit without curing roughness

After FIX1+FIX2: turntable torso dihedral p99 52° (vs ΔV=0 baseline 34°, vs pre-fix 60°). Median dihedral barely changed (5.18° → 5.23°) — roughness is a spike pattern, not global. Main cause unconfirmed.

### Open issues going into D14

**Issue 1 — Residual torso roughness (D14 acceptance target).** Turntable torso dihedral p99 ~52° vs ΔV=0 baseline 34°. FIX1 and FIX2 each helped ~4°; w_lap sweep showed under-regularization is not the cause (more Laplacian made IoU worse without clearing the spikes). Main cause still unconfirmed — candidates: (a) strong silhouette pull creating local spikes at feature boundaries with no adjacent-face smoothing constraint; (b) normal_consistency weight too low (0.1). D14 target: visibly reduce p99 through weight rebalancing.

**Issue 2 — Overlay torso under-fit (D14 secondary target).** Mesh silhouette (green) fails to fill SAM mask contour (red) at torso. Classified as category (c): spread across the torso body (not just head/hands/feet which are D8-intentional). Torso FN ~50–75% across frames — the mesh is too narrow. Not a projection error, not a D8 weighting artifact. Cause: regularization (w_lap=100) is over-resisting the silhouette pull. Corroborated by w_lap sweep: lower w_lap → higher IoU gain, stronger torso fill. D14 primary lever: lower w_lap, raise w_silh.

**Current loss weights going into D14** (in `config.py`):
```
silhouette: 1.0,  normal: 0.5,  keypoint: 1e-3,
laplacian: 100.0,  symmetry: 0.01,  height: 10.0,  normal_consistency: 0.1
```

### D14 tuning log

**⚠ D14 R1/R2/R3 below ran at h=170cm (wrong height — stored placeholder). All superseded by h=180 re-runs further below. Kept for diagnostic record only.**

**D14 R1 — w_lap 100 → 25** (run `20260523_150104_m8`, all other weights unchanged; h=170, SUPERSEDED)

| Metric | Init (ΔV=0) | R1 final | vs pre-D14 |
|---|---|---|---|
| Mean hard-IoU | 0.782 | 0.848 | +0.066 (was +0.077 at w_lap=100) |
| Dihedral p99 (frame 0) | 69° | 117° | +48° vs init; pre-D14 ref was ~52° |
| max\|ΔV\| | — | 0.183 m | |

**Result: negative.** w_lap=25 is clearly too low. Two findings:
1. **Roughness exploded**: p99 jumped +48° (69° → 117°). The Laplacian at w=100 was doing primary roughness suppression — not just resisting displacement. Visible severe mottling in turntable final strip (torso, limbs, feet).
2. **Torso fill didn't improve**: IoU gain with w_lap=25 (+0.066) is slightly LESS than with w_lap=100 (+0.077). The torso under-fill bottleneck is NOT the Laplacian; it's elsewhere. The diagnosis "regularization over-resists torso fill" is incorrect or at least w_lap is not the active constraint.

**Revised interpretation:** w_lap=100 was load-bearing for smoothness. Dropping to 25 harms both metrics. The torso fill gap persists regardless of Laplacian level. The silhouette loss is producing a large fraction of noisy gradient — vertex movement goes into surface jitter rather than better body fit.

**D14 R2 — w_nc 0.1 → 2.0** (run `20260523_150744_m8`, w_lap back to 100, all else unchanged; h=170, SUPERSEDED)

| Metric | Init (ΔV=0) | R2 final | R1 (w_lap=25) | pre-D14 (w_nc=0.1) |
|---|---|---|---|---|
| Mean IoU | 0.782 | 0.829 (+0.047) | 0.848 (+0.066) | — (+0.077) |
| Dihedral p99 (frame 0) | 69° | 45° (−23°) | 117° (+48°) | ~52° |
| max\|ΔV\| | — | 0.089 m | 0.183 m | 0.156 m |

**Result:** w_nc=2.0 is an effective roughness knob — p99 dropped from 69° (init) to 45° (below init baseline), cleaner than anything w_lap tuning achieved. But 2.0 is too high: the mesh over-smoothed, max|ΔV| fell to 0.089 m, and IoU dropped to +0.047. Two endpoints now established:
- w_nc=0.1 → IoU +0.077, p99 ~52° (under-smoothed)
- w_nc=2.0 → IoU +0.047, p99 45° (over-smoothed, below init)

The fit/smoothness tradeoff curve between 0.1 and 2.0 is untested. R3 sweeps the middle. Note: R1/R2 IoU comparisons to pre-D14 +0.077 were incorrect due to height mismatch (see correction above).

**Height baseline correction:** All pre-D14 runs were invoked with `--height 180`, overriding the stored `height_cm=170.0` in stage1_result.npz. D14 R1/R2/R3 runs read the stored 170cm. The "+0.077 IoU" figure referenced throughout D14 was from a h=180cm run and is NOT comparable. The correct apples-to-apples baseline at h=170cm (w_nc=0.1) is: IoU +0.052, dihedral p99=85°. All D14 comparisons below use h=170cm.

**D14 R3 — w_nc sweep: 0.3, 0.5, 0.8** (w_lap=100, h=170cm, all else unchanged; SUPERSEDED by h=180 re-runs below)

Full comparison at h=170cm (wrong height — for diagnostic record only):
w_nc=0.1: IoU +0.052, p99=85°; w_nc=0.3: +0.051, p99=70°; w_nc=0.5: +0.050, p99=58°; w_nc=0.8: +0.050, p99=54°. Qualitative finding (smoothness knob direction) holds at h=180; numbers superseded.

---

**D14 h=180 confirmed sweep** (correct height; bundle `20260522_001631_h180`)

**Height fix:** `data/runs/20260522_001631_h180/stage1_result.npz` — corrected copy of the original bundle with `height_cm=180`. Built via `Stage1Result.load → dataclasses.replace(height_cm=180) → save`. Pixel paths (`keyframe_paths`, `mask_paths`, `normal_paths`) unchanged, still point at `20260522_001631/` tree. Original 001631 bundle untouched. All future tune.py calls use `20260522_001631_h180` as SOURCE_RUN.

**Sweep: w_nc=0.1 / 0.5 / 0.8, w_lap=100, h=180**, dihedral measured on frame 0 posed mesh:

| Setting | IoU gain | Dihedral p99 |
|---|---|---|
| Init (ΔV=0) | — | 69° |
| w_nc=0.1 (baseline, run `20260523_153225_m8`) | +0.057 | 77° (worse than init) |
| w_nc=0.5 (run `20260523_153318_m8`) | +0.056 | 61° |
| **w_nc=0.8 (run `20260523_153400_m8`)** | **+0.056** | **54°** |

**Result: w_nc=0.8 confirmed as sweet spot.** IoU cost vs baseline: −0.001. Roughness improvement: 77° → 54° (−23°, below the init mesh at 69°). Pattern consistent with h=170 sweep — height target affects IoU level but not smoothness mechanism. Turntable at w_nc=0.8: mesh broader than init, mild surface texture on torso/arms, feet retain some noise (D8 intentional downweighting), no spike pattern. Pending: commit w_nc=0.8 to config.py.

### D14 locking experiment (diagnostic, not adopted)

**Soft lock attempt (silh_weight=0 for locked joints):** Set `_SILH_WEIGHT_PER_JOINT` to 0.0 for head/hands/feet/ankles/pelvis/spine1/spine2 (11 joints). Rationale: SAM mask contamination at those regions drives runaway ΔV. Result: max|ΔV| 0.138 m → 0.137 m — essentially unchanged. Root cause: silhouette weight=0 removes the direct gradient for those vertices, but Laplacian coupling from adjacent unlocked leg vertices continues to move them. Masking the signal cannot prevent transitive coupling through w_lap=100.

**Hard lock (gradient + data zeroed each iter):** Added two lines to the optimizer loop — `delta_v.grad[locked_indices] = 0.0` after backward+clip, `delta_v.data[locked_indices] = 0.0` after optimizer.step. `locked_indices` derived from `silh_region_weights == 0`, which corresponds to the same 11 joints above → 3628 / 6890 vertices (53%) pinned.

Results (h=180, w_nc=0.8):
- Init IoU: 0.782 | Final IoU: 0.801 | Gain: **+0.018**
- max|ΔV|: **0.050 m** (down from 0.138 m)
- Top-5 ΔV vertices: R_wrist (joint 21), spine3 (joint 9), L_knee (joint 4) — all legitimate body regions, no locked-vertex runaway

**Finding:** Hard lock successfully eliminated the foot/ankle/pelvis runaway. But IoU gain fell from +0.056 (unlocked) to +0.018. Locking 53% of vertices — including the spine and pelvis which form most of the torso — removes the optimization budget needed to fit the torso to the silhouette. Even with only unlocked vertices free (thighs, arms, upper back), the mesh could not recover comparable IoU.

This is the terminal experiment for D14. The high-opacity 4-view acceptance render confirmed the visual verdict: the locked-vertex sculpted mesh does not visibly improve on the β-only (ΔV=0) mesh, and introduces surface noise on the remaining free regions. The β-mesh already fits well enough that ΔV's marginal gain is not worth its artifacts.

---

## 2026-05-23 — M8 close-out

### Decision: ΔV optimization NOT adopted for MVP

**Evidence reviewed:** Full D14 tuning log (7 runs), high-opacity 4-view acceptance render (87% mesh opacity, 4 views × keyframe / β-mesh / sculpted-mesh), geometry turntable (init vs final, 12 views × 360°).

**Verdict:** In the SMPL v1, 10-parameter β + free per-vertex ΔV [6890, 3] configuration, test-time ΔV optimization does not produce results worth adopting. Specific failure modes observed:

1. **Runaway local deformation.** Without locking, max|ΔV| reached 0.138 m (14 cm) at foot/ankle vertices. SAM mask contamination at shorts waistband and shoe boundaries drove the silhouette gradient into physically implausible vertex displacement.

2. **Lock → loss of fitting capacity.** Hard-locking the contaminated regions (53% of vertices) to prevent runaway reduced IoU gain from +0.056 to +0.018. The optimization can only fit where it has freedom; the contaminated regions are large enough that locking them removes the budget needed to fit the torso.

3. **Regularization suppresses fit.** The w_nc sweep showed that sufficient smoothness (p99 < init baseline 69°) requires w_nc ≥ 0.8, which itself reduces IoU gain vs the baseline. The tradeoff between smoothness and fit cannot be resolved within the current free-ΔV formulation.

4. **4-view render confirmation.** The acceptance render showed the sculpted mesh does not visually improve on the β-only mesh for any of the four views (back, left, front, right). In some views (side profile) the ΔV mesh is slightly worse (surface roughness visible). The β mesh already tracks the subject's silhouette well from Stage 1.

**This is a valid engineering conclusion, not a failure.** M8 implemented and evaluated all seven loss terms (silhouette + region weights, normal-map, keypoint reprojection, Laplacian, symmetry, height, normal consistency) through a full D6–D14 diagnostic sequence. The outcome is that free per-vertex ΔV at SMPL v1 resolution is not the right shape refinement mechanism for this MVP. The evidence base is complete.

### Recommended future direction

Higher-capacity shape space rather than free ΔV. SMPL's 10 β parameters are too few to capture individual body shape variation (e.g. waist-to-hip ratio, limb proportions) without free ΔV to compensate, but free ΔV at 6890 vertices is too unconstrained without strong enough priors. The natural next step is **SMPL v2 / SMPL-X with ~300 shape parameters** (or similar high-β models like SMPL+H) — these encode a much richer shape prior learned from scan data, so per-vertex freedom is not needed for plausible individual variation. Record as the recommended path for the body-shape refinement capability.

### Decided handling for current codebase

- **Stage 2 code: retained, not deleted.** All seven loss terms, D6–D14 infrastructure (region weights, normal rendering, symmetry, height loss, keypoint reprojection), and the optimizer loop remain in `stage2.py`. The work is not discarded.
- **delta_v: will be zeroed so Stage 3 receives a clean β mesh.** The exact mechanism (e.g. Stage2Result.delta_v = zeros, or a flag that skips the optimizer) is deferred to the next session.
- **Stage 3 data contract:** Stage 3 will receive the β-shaped mesh (ΔV=0) from Stage 2. The shape parameters (β, θ) from Stage 1 remain the primary body representation for the MVP.

### Uncommitted changes at M8 close

`body_mvp/stage2.py` — **not committed.** Contains two experimental edits from this session:
1. `_SILH_WEIGHT_PER_JOINT` extended with pelvis/spine1/spine2 zeros (soft-lock attempt)
2. Hard-lock lines in the optimizer loop (gradient + data zeroing each iter)

Neither edit is part of the adopted M8 state. They are diagnostic experiments recorded here. The next session can decide whether to revert them or repurpose the hard-lock infrastructure for the delta_v zeroing mechanism.

`body_mvp/config.py` — **not committed.** Contains `"normal_consistency": 0.8` (D14 confirmed weight) and `"laplacian": 100.0`. These represent the final D14 weights but since ΔV is not adopted, committing them is not urgent. Defer to next session.

`NOTES.md` — **not committed.** This entry plus the D14 log above.
