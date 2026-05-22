import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import smplx
import smplx.lbs
import torch
from loguru import logger
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.structures import Meshes

from body_mvp.config import (
    FACES_PER_PIXEL,
    GRAD_CLIP_NORM,
    HEIGHT_TOLERANCE_M,
    LEARNING_RATE,
    LOSS_WEIGHTS,
    OPT_MAX_ITERS,
    RENDER_RESOLUTION,
    settings,
)
from body_mvp.losses import (
    hard_iou_per_frame,
    height_loss,
    keypoint_reprojection_loss,
    laplacian_smoothing_loss,
    normal_consistency_loss,
    symmetry_loss,
    weighted_silhouette_iou_loss,
)
from body_mvp.render import build_cameras, build_silhouette_renderer
from body_mvp.stage1 import Stage1Result

# hmr2's training crop size. focal_length_per_frame in Stage1Result is in
# this normalized space (typically 5000); render-resolution pixels need
# `fl_render = fl_raw / _HMR2_CROP_SIZE * max(Wt, Ht)`. Mirrors hmr2's own
# scaling in third_party/4D-Humans/hmr2/utils/renderer.py:render_rgba_multiple
# and our M4 render_smpl_overlays (stage1.py:512).
_HMR2_CROP_SIZE = 256

# --- D7: Region weight infrastructure ---
#
# Joints whose *anterior* (canonical z > 0) dominant-weighted vertices form
# the "belly" region.  Verified from J_regressor @ v_template:
#   0=pelvis (y=-0.22), 3=spine1 (y=-0.11), 6=spine2 (y=+0.02).
# Canonical +z is forward/anterior — confirmed empirically (face vertices are
# at positive z; back-of-head vertices at negative z).
_BELLY_CANDIDATE_JOINTS: frozenset[int] = frozenset({0, 3, 6})

# Per-dominant-joint silhouette weight. Unlisted joints default to 1.0.
# Regions susceptible to SAM mask contamination (hair, shoes, sleeves) → 0.2.
_SILH_WEIGHT_PER_JOINT: dict[int, float] = {
    12: 0.2, 15: 0.2,   # neck, head  (hair above crown)
    10: 0.2, 11: 0.2,   # L_foot, R_foot  (shoe edges)
    22: 0.2, 23: 0.2,   # L_hand, R_hand  (sleeve/glove edges)
}

# Per-dominant-joint symmetry weight. Unlisted joints default to 1.0.
# Belly-candidate anterior vertices are overridden to 0.0 (see _build_region_weights).
_SYM_WEIGHT_PER_JOINT: dict[int, float] = {
    0: 0.3, 3: 0.3, 6: 0.3,     # pelvis/spine1/spine2 posterior half — lower back
    9: 0.3, 13: 0.3, 14: 0.3,   # spine3, L_collar, R_collar — upper back
    1: 0.7, 2: 0.7,              # L_hip, R_hip
    4: 0.7, 5: 0.7,              # L_knee, R_knee
    7: 0.7, 8: 0.7,              # L_ankle, R_ankle
    10: 0.7, 11: 0.7,            # L_foot, R_foot
    16: 0.7, 17: 0.7,            # L_shoulder, R_shoulder
    18: 0.7, 19: 0.7,            # L_elbow, R_elbow
    20: 0.7, 21: 0.7,            # L_wrist, R_wrist
    22: 0.7, 23: 0.7,            # L_hand, R_hand
    12: 1.0, 15: 1.0,            # neck, head
}

# COCO-17 → SMPL-24 joint index pairs for D10 keypoint reprojection.
# Only joints with direct anatomical correspondence are included (12 pairs;
# head/nose excluded because SMPL joint 15 is the neck top, not the nose tip).
_COCO_TO_SMPL: tuple[tuple[int, int], ...] = (
    (5,  16),  # L_shoulder
    (6,  17),  # R_shoulder
    (7,  18),  # L_elbow
    (8,  19),  # R_elbow
    (9,  20),  # L_wrist
    (10, 21),  # R_wrist
    (11, 1),   # L_hip
    (12, 2),   # R_hip
    (13, 4),   # L_knee
    (14, 5),   # R_knee
    (15, 7),   # L_ankle
    (16, 8),   # R_ankle
)


@dataclass
class Stage2Result:
    """Aggregated output of Stage 2 minimal optimization (M7).

    ΔV is the per-vertex offset learned in canonical T-pose space. β and θ
    are passed through from Stage 1 unchanged — Stage 3 needs all three to
    reconstruct posed and canonical meshes. Optimization metadata
    (final_loss, loss_history, n_iterations) is kept alongside for
    debugging and M8 tuning.

    Persisted to a single .npz at the run dir root via save_stage2_result;
    reload via load_stage2_result. Native numpy dtypes only, no pickle.
    """

    run_id: str
    run_dir: Path

    # ΔV — the optimized per-vertex offset in canonical T-pose space.
    delta_v: np.ndarray              # [6890, 3] float32

    # Pass-through from Stage 1 — Stage 3 consumes both to rebuild the mesh.
    beta: np.ndarray                 # [10]      float32
    theta_per_frame: np.ndarray      # [N, 24, 3] float32

    # Optimization metadata.
    n_iterations: int
    final_loss: float
    loss_history: np.ndarray         # [T] float32 — total loss per iter

    # Quality bars per frame (silhouette IoU before/after).
    initial_iou_per_frame: np.ndarray  # [N] float32
    final_iou_per_frame: np.ndarray    # [N] float32

    # Raw (pre-weight) per-term loss history. Keys e.g. "silhouette",
    # "laplacian"; each value is [T] float32. Empty dict is valid (no
    # per-term breakdown was recorded). Drives M8 cross-run tuning
    # comparison — total loss_history alone hides which term is moving.
    per_term_history: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_iterations <= 0:
            raise ValueError(f"n_iterations: must be > 0, got {self.n_iterations}")
        if self.final_loss != self.final_loss:  # NaN
            raise ValueError("final_loss: must not be NaN")

        def _check_shape(name: str, arr: np.ndarray, expected: tuple) -> None:
            if arr.shape != expected:
                raise ValueError(
                    f"{name}: expected shape {expected}, got {tuple(arr.shape)}"
                )

        _check_shape("delta_v", self.delta_v, (6890, 3))
        _check_shape("beta", self.beta, (10,))

        N = self.theta_per_frame.shape[0]
        if N <= 0:
            raise ValueError(f"theta_per_frame: leading dim must be > 0, got {N}")
        _check_shape("theta_per_frame", self.theta_per_frame, (N, 24, 3))
        _check_shape("initial_iou_per_frame", self.initial_iou_per_frame, (N,))
        _check_shape("final_iou_per_frame", self.final_iou_per_frame, (N,))

        T = self.loss_history.shape[0]
        if T != self.n_iterations:
            raise ValueError(
                f"loss_history length {T} != n_iterations {self.n_iterations}"
            )

        for name, arr in self.per_term_history.items():
            # "__" is the npz flatten/un-flatten prefix separator; disallow
            # it in user-facing key names so save/load can't ambiguously
            # parse the field.
            if "__" in name:
                raise ValueError(
                    f"per_term_history key {name!r}: double-underscore is reserved"
                )
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f"per_term_history[{name!r}]: expected ndarray, got {type(arr)}"
                )
            if arr.shape != (self.n_iterations,):
                raise ValueError(
                    f"per_term_history[{name!r}]: expected shape "
                    f"({self.n_iterations},), got {tuple(arr.shape)}"
                )
            if arr.dtype != np.float32:
                raise ValueError(
                    f"per_term_history[{name!r}]: expected float32, got {arr.dtype}"
                )


_PER_TERM_PREFIX = "per_term__"


def save_stage2_result(result: Stage2Result, path: Path) -> None:
    """Write Stage2Result to a single .npz at `path`. Mirrors save_stage1_result:
    native dtypes, no pickle, strings as 0-d unicode arrays, paths as str.

    per_term_history is flattened into `per_term__<name>` keys so np.savez
    (no dict support) and allow_pickle=False (no pickle on load) both stay
    happy. load_stage2_result reconstructs the dict by prefix-stripping.
    """
    assert path.suffix == ".npz", (
        f"save_stage2_result: path must end in .npz to match np.savez's own "
        f"behavior (it silently appends .npz otherwise, desyncing the file "
        f"path the caller thinks they wrote). Got: {path}"
    )
    save_kwargs: dict[str, np.ndarray] = dict(
        run_id=np.array(result.run_id),
        run_dir=np.array(str(result.run_dir)),
        delta_v=result.delta_v.astype(np.float32),
        beta=result.beta.astype(np.float32),
        theta_per_frame=result.theta_per_frame.astype(np.float32),
        n_iterations=np.array(result.n_iterations, dtype=np.int64),
        # float64 to preserve full Python-float precision through the
        # round-trip self-check (single scalar, no memory cost). The
        # array-typed loss_history stays float32 per plan.
        final_loss=np.array(result.final_loss, dtype=np.float64),
        loss_history=result.loss_history.astype(np.float32),
        initial_iou_per_frame=result.initial_iou_per_frame.astype(np.float32),
        final_iou_per_frame=result.final_iou_per_frame.astype(np.float32),
    )
    for name, arr in result.per_term_history.items():
        key = f"{_PER_TERM_PREFIX}{name}"
        if key in save_kwargs:
            raise ValueError(
                f"per_term_history key {name!r} collides with reserved field {key!r}"
            )
        save_kwargs[key] = arr.astype(np.float32)
    np.savez(str(path), **save_kwargs)


def load_stage2_result(path: Path) -> Stage2Result:
    """Inverse of save_stage2_result. allow_pickle=False is safe — every
    field is stored as a native numpy dtype."""
    data = np.load(str(path), allow_pickle=False)
    per_term_history: dict[str, np.ndarray] = {
        key[len(_PER_TERM_PREFIX):]: data[key]
        for key in data.files
        if key.startswith(_PER_TERM_PREFIX)
    }
    return Stage2Result(
        run_id=str(data["run_id"].item()),
        run_dir=Path(str(data["run_dir"].item())),
        delta_v=data["delta_v"],
        beta=data["beta"],
        theta_per_frame=data["theta_per_frame"],
        n_iterations=int(data["n_iterations"].item()),
        final_loss=float(data["final_loss"].item()),
        loss_history=data["loss_history"],
        initial_iou_per_frame=data["initial_iou_per_frame"],
        final_iou_per_frame=data["final_iou_per_frame"],
        per_term_history=per_term_history,
    )


def _verify_stage2_round_trip(original: Stage2Result, path: Path) -> None:
    """Load the just-written Stage2Result and fail-loud if anything drifted.

    Same belt-and-suspenders pattern as Stage 1: cheap enough to run on
    every save (no pixel data in the bundle).
    """
    loaded = load_stage2_result(path)

    scalar_fields = ("run_id", "run_dir", "n_iterations", "final_loss")
    for name in scalar_fields:
        a = getattr(original, name)
        b = getattr(loaded, name)
        if a != b:
            raise RuntimeError(
                f"round-trip mismatch on {name}: original={a!r} loaded={b!r}"
            )

    array_fields = (
        "delta_v", "beta", "theta_per_frame", "loss_history",
        "initial_iou_per_frame", "final_iou_per_frame",
    )
    for name in array_fields:
        a = getattr(original, name)
        b = getattr(loaded, name)
        if a.shape != b.shape:
            raise RuntimeError(f"round-trip shape mismatch on {name}: {a.shape} vs {b.shape}")
        if a.dtype != b.dtype:
            raise RuntimeError(f"round-trip dtype mismatch on {name}: {a.dtype} vs {b.dtype}")
        if not np.array_equal(a, b):
            raise RuntimeError(f"round-trip value mismatch on {name}")

    orig_pt = original.per_term_history
    loaded_pt = loaded.per_term_history
    if set(orig_pt.keys()) != set(loaded_pt.keys()):
        raise RuntimeError(
            f"round-trip per_term_history key-set mismatch: "
            f"original={sorted(orig_pt)} loaded={sorted(loaded_pt)}"
        )
    for name in orig_pt:
        a, b = orig_pt[name], loaded_pt[name]
        if a.shape != b.shape:
            raise RuntimeError(
                f"round-trip shape mismatch on per_term_history[{name!r}]: "
                f"{a.shape} vs {b.shape}"
            )
        if a.dtype != b.dtype:
            raise RuntimeError(
                f"round-trip dtype mismatch on per_term_history[{name!r}]: "
                f"{a.dtype} vs {b.dtype}"
            )
        if not np.array_equal(a, b):
            raise RuntimeError(f"round-trip value mismatch on per_term_history[{name!r}]")

    logger.info("stage2_result.npz round-trip self-check passed")


def _load_smpl_model(device: str) -> smplx.SMPL:
    """Load smplx.SMPL from the symlinked checkpoint (basicModel_*.pkl).

    create_*=False suppresses smplx's default nn.Parameter allocations for
    inputs we feed manually (β, pose, transl). The model's buffers
    (v_template, shapedirs, posedirs, J_regressor, lbs_weights, parents,
    faces) are byte-identical to what hmr2 loaded in Stage 1.
    """
    model = smplx.SMPL(
        model_path=str(settings.smpl_model_path),
        gender="neutral",
        num_betas=10,
        create_betas=False,
        create_global_orient=False,
        create_body_pose=False,
        create_transl=False,
    ).to(device).eval()
    return model


def _build_region_weights(
    smpl_model: smplx.SMPL,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-vertex region weight vectors and the left-right mirror map.

    All three outputs are computed once from the SMPL template and cached as
    device tensors for use throughout the optimization loop.

    Returns
    -------
    silh_region_weights : [6890] float32
        Per-vertex weight for the silhouette loss.  Head/neck, hands, and feet
        regions are set to 0.2 (SAM contamination expected); all others 1.0.
    sym_region_weights : [6890] float32
        Per-vertex weight for the symmetry loss.  Belly (anterior
        pelvis/spine1/spine2 dominant, z > 0) = 0.0 to preserve fat
        asymmetry; posterior spine/back = 0.3; limbs/hands/feet = 0.7;
        head/neck = 1.0.
    right_to_left : [6890] int64
        Mirror index map: right_to_left[i] is vertex i's nearest neighbor in
        the x-flipped template.  Validated as a near-involution.
    """
    v_np = smpl_model.v_template.detach().cpu().numpy().astype(np.float32)  # [6890, 3]
    lbs_np = smpl_model.lbs_weights.detach().cpu().numpy()                  # [6890, 24]
    dom = lbs_np.argmax(axis=1)                                             # [6890] int
    N = v_np.shape[0]

    # --- Silhouette region weights ---
    silh_w = np.ones(N, dtype=np.float32)
    for j, w in _SILH_WEIGHT_PER_JOINT.items():
        silh_w[dom == j] = w

    # --- Symmetry region weights ---
    sym_w = np.ones(N, dtype=np.float32)
    for j, w in _SYM_WEIGHT_PER_JOINT.items():
        sym_w[dom == j] = w
    # Belly-candidate anterior vertices: override to 0.0.
    belly_mask = np.isin(dom, sorted(_BELLY_CANDIDATE_JOINTS)) & (v_np[:, 2] > 0.0)
    sym_w[belly_mask] = 0.0

    # --- Per-region vertex count stats ---
    regions: dict[str, np.ndarray] = {
        "belly":       belly_mask,
        "pelvis_back": np.isin(dom, sorted(_BELLY_CANDIDATE_JOINTS)) & ~belly_mask,
        "upper_back":  np.isin(dom, [9, 13, 14]),
        "head_neck":   np.isin(dom, [12, 15]),
        "arms":        np.isin(dom, [16, 17, 18, 19, 20, 21]),
        "hands":       np.isin(dom, [22, 23]),
        "legs":        np.isin(dom, [1, 2, 4, 5, 7, 8]),
        "feet":        np.isin(dom, [10, 11]),
    }
    total_accounted = sum(int(m.sum()) for m in regions.values())
    if total_accounted != N:
        raise RuntimeError(
            f"Region partition covers {total_accounted} vertices, expected {N}"
        )
    logger.info("D7 region vertex counts (total={}):", N)
    for name, mask in regions.items():
        logger.info("  {:12s}: {:4d}", name, int(mask.sum()))

    # --- Left-right mirror map via nearest-neighbour in x-flipped template ---
    v_mirror = v_np * np.array([-1.0, 1.0, 1.0], dtype=np.float32)
    v_t = torch.from_numpy(v_np)                   # [6890, 3]  CPU float32
    vm_t = torch.from_numpy(v_mirror)              # [6890, 3]  CPU float32
    dists = torch.cdist(v_t, vm_t)                 # [6890, 6890]
    right_to_left = dists.argmin(dim=1).to(torch.int64)  # [6890]

    # --- Involution gate: zero sym_w for non-involution vertices ---
    # SMPL's template is not vertex-level bilaterally symmetric; ~6.8% of
    # vertices have no clean mirror partner (diagnosed: half are in the outer
    # arms/hands where left/right triangulations differ slightly). These
    # vertices should not participate in the symmetry loss because the premise
    # "i and right_to_left[i] are mirror partners" doesn't hold for them.
    arange = torch.arange(N, dtype=torch.int64)
    double_mirror = right_to_left[right_to_left]
    involution_mask_t = (double_mirror == arange)           # [6890] bool tensor
    involution_mask   = involution_mask_t.numpy()           # [6890] bool numpy

    pair_dists = torch.norm(v_t - vm_t[right_to_left], dim=1)
    max_pair_dist = float(pair_dists.max().item())
    n_far_pairs   = int((pair_dists > 0.01).sum().item())

    n_non_invol       = int((~involution_mask_t).sum().item())
    n_zeroed_by_gate  = int(((sym_w > 0) & ~involution_mask).sum())
    sym_w[~involution_mask] = 0.0

    n_sym_constrained = int((sym_w > 0).sum())
    logger.info(
        "D7 mirror map: non_involution={}, zeroed_by_gate={}, "
        "sym_constrained_after_gate={}/{}, "
        "max_pair_dist={:.5f} m, pairs_over_1cm={}",
        n_non_invol, n_zeroed_by_gate, n_sym_constrained, N,
        max_pair_dist, n_far_pairs,
    )

    # Hard invariant: every vertex with sym_weight > 0 must be a valid
    # involution pair. This should always hold after the zeroing above.
    n_fail_post_gate = int(((sym_w > 0) & ~involution_mask).sum())
    if n_fail_post_gate != 0:
        raise RuntimeError(
            f"D7 invariant violated: {n_fail_post_gate} vertices have "
            f"sym_weight > 0 but are not valid involution pairs"
        )

    return (
        torch.from_numpy(silh_w).to(device),
        torch.from_numpy(sym_w).to(device),
        right_to_left.to(device),
    )


def _render_weight_map(
    meshes: Meshes,
    silh_region_weights: torch.Tensor,  # [6890] float32, on device
    renderer: torch.nn.Module,          # MeshRenderer exposing .rasterizer
) -> torch.Tensor:
    """Render per-vertex silhouette region weights to a [N, Ht, Wt] pixel map.

    Uses the nearest visible face for each pixel (K=0 slice of the rasterizer
    fragments). Background pixels (outside the mesh silhouette) receive 1.0
    so the unweighted IoU formula applies outside the body.

    Called once before the optimization loop — silh_region_weights are fixed
    vertex labels that don't change during optimization.
    """
    device = silh_region_weights.device
    N = len(meshes)

    faces = meshes.faces_padded()[0]  # [F, 3] int64, on device

    # Build face-vertex attribute tensor: [F, 3, 1] (not batched — interpolate
    # handles the N dimension via pix_to_face).
    face_attrs = silh_region_weights[faces].unsqueeze(-1)  # [F, 3, 1]

    with torch.no_grad():
        fragments = renderer.rasterizer(meshes)

    # Take only K=0 (nearest face) to avoid blending across region boundaries.
    pix_to_face_k0 = fragments.pix_to_face[..., :1]    # [N, H, W, 1]
    bary_k0        = fragments.bary_coords[..., :1, :]  # [N, H, W, 1, 3]

    # [N, H, W, 1, 1] → squeeze to [N, H, W]
    pixel_weights = interpolate_face_attributes(pix_to_face_k0, bary_k0, face_attrs)
    weight_map = pixel_weights[..., 0, 0]

    # Background pixels (pix_to_face == -1) are set to 0 by interpolate;
    # restore to 1.0 so background contributes equally on both sides of IoU.
    background = fragments.pix_to_face[..., 0] < 0   # [N, H, W]
    weight_map = weight_map.masked_fill(background, 1.0)

    return weight_map.detach()


def _pose_meshes(
    smpl_model: smplx.SMPL,
    beta: torch.Tensor,          # [10]      float, requires_grad=False
    theta: torch.Tensor,         # [N, 24, 3] float, axis-angle, requires_grad=False
    delta_v: torch.Tensor,       # [6890, 3] float, requires_grad=True (or zeros for sanity)
    pred_cam_t: torch.Tensor,    # [N, 3]    float (full-image translation from hmr2)
    device: str,
) -> tuple[Meshes, torch.Tensor]:  # (Meshes, joints_world [N,24,3])
    """β-shape + ΔV applied in canonical T-pose space, then LBS-posed per
    frame, then SMPL→OpenCV-camera flip applied.

    Gradient flow: ΔV → v_canonical → v_template (into smplx.lbs.lbs) →
    skinned verts → Meshes. β/θ/pred_cam_t are detached upstream.
    """
    N = theta.shape[0]

    # Step 1: apply shape blendshapes once (β is shared across all N frames).
    # v_canonical: [6890, 3] = template + Σ(β_k * shapedir_k) + ΔV.
    v_shape_only = smplx.lbs.blend_shapes(
        beta.unsqueeze(0), smpl_model.shapedirs
    ).squeeze(0)
    v_canonical = smpl_model.v_template + v_shape_only + delta_v  # [6890, 3]

    # Step 2: batch the canonical mesh across N frames as a *per-frame*
    # v_template input to smplx.lbs.lbs. Passing betas=zeros there makes
    # its internal `v_shaped = v_template + blend_shapes(0)` collapse to
    # v_template, so blend_shapes is applied exactly once (above) — no
    # double-counting.
    v_template_batched = v_canonical.unsqueeze(0).expand(N, -1, -1).contiguous()
    pose_batched = theta.reshape(N, 72)  # [N, 24*3] axis-angle
    betas_zero = torch.zeros(N, 10, device=device, dtype=beta.dtype)

    # Step 3: LBS. Pose blendshapes (posedirs @ (R_j - I)) are still applied
    # internally on top of v_template; they're independent of β.
    verts_posed, joints_posed = smplx.lbs.lbs(
        betas=betas_zero,
        pose=pose_batched,
        v_template=v_template_batched,
        shapedirs=smpl_model.shapedirs,
        posedirs=smpl_model.posedirs,
        J_regressor=smpl_model.J_regressor,
        parents=smpl_model.parents,
        lbs_weights=smpl_model.lbs_weights,
        pose2rot=True,
    )  # verts_posed [N, 6890, 3], joints_posed [N, 24, 3]

    # Step 4: add hmr2's full-image translation. NO additional flip is
    # needed, despite hmr2's pyrender path applying a 180° X rotation.
    #
    # Counterintuitive but verified empirically (named SMPL vertices
    # projected vs M4 overlay measurement): hmr2's predicted global_orient
    # already encodes a ~180° X rotation (|θ_0| ≈ 3.105 rad on our test
    # video, all 12 frames). The posed SMPL therefore has head at NEGATIVE
    # Y and feet at POSITIVE Y — already in OpenCV image convention
    # (Y-down). hmr2's R_180x in the pyrender renderer plus pyrender's
    # OpenGL Y-up framebuffer flip together CANCEL with the global_orient's
    # built-in flip, so applying any further flip here would invert the
    # image relative to M4.
    #
    # Verified vertex 411 (head crown): SMPL canonical y=+0.555, posed
    # y=-1.001 (global_orient flips), after +cam_t y=-0.598. PyTorch3D
    # OpenCV projection v = fy*Y/Z + cy = 37500 * (-0.598) / 51.37 + 960
    # ≈ 524, matching M4's measured head y_top=512.
    verts_world  = verts_posed  + pred_cam_t.unsqueeze(1)  # [N, 6890, 3]
    joints_world = joints_posed + pred_cam_t.unsqueeze(1)  # [N, 24, 3]

    # Step 5: wrap as a batched PyTorch3D Meshes.
    faces_np = smpl_model.faces.astype(np.int64)
    faces = torch.from_numpy(faces_np).to(device)
    faces_batched = faces.unsqueeze(0).expand(N, -1, -1)
    return Meshes(verts=verts_world, faces=faces_batched), joints_world


def _compute_render_resolution(W_orig: int, H_orig: int) -> tuple[int, int, float]:
    """Render at long-side = RENDER_RESOLUTION, preserving aspect ratio.

    Returns (Ht, Wt, scale). scale = Wt/W_orig = Ht/H_orig.
    """
    if W_orig >= H_orig:
        Wt = RENDER_RESOLUTION
        Ht = int(round(H_orig * Wt / W_orig))
    else:
        Ht = RENDER_RESOLUTION
        Wt = int(round(W_orig * Ht / H_orig))
    scale = Wt / W_orig
    return Ht, Wt, scale


def _silhouette_triptych(
    keyframe_bgr: np.ndarray,    # [H_orig, W_orig, 3] BGR
    mask_small: np.ndarray,      # [Ht, Wt] uint8 0/255
    alpha_np: np.ndarray,        # [Ht, Wt] float in [0, 1]
    Wt: int,
    Ht: int,
    label: str,
) -> np.ndarray:
    """Build the keyframe | (kf+green-silhouette+red-mask-contour) | alpha
    triptych. Returns a [Ht, 3*Wt, 3] BGR uint8 image."""
    kf_resized = cv2.resize(keyframe_bgr, (Wt, Ht), interpolation=cv2.INTER_AREA)

    panel_kf = kf_resized.copy()

    panel_overlay = kf_resized.astype(np.float32)
    silh_green = np.zeros_like(panel_overlay)
    silh_green[..., 1] = 220.0  # BGR green
    a3 = alpha_np[..., None]
    panel_overlay = panel_overlay * (1.0 - a3 * 0.55) + silh_green * (a3 * 0.55)
    contours, _ = cv2.findContours(
        (mask_small > 127).astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    panel_overlay = panel_overlay.astype(np.uint8)
    cv2.drawContours(panel_overlay, contours, -1, (0, 0, 255), 2)

    panel_alpha = np.clip(alpha_np * 255.0, 0, 255).astype(np.uint8)
    panel_alpha = cv2.cvtColor(panel_alpha, cv2.COLOR_GRAY2BGR)

    triptych = cv2.hconcat([panel_kf, panel_overlay, panel_alpha])
    cv2.putText(
        triptych, label, (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return triptych


def sanity_render_frame00(stage1_result: Stage1Result) -> Path:
    """Step-1 hard gate: render initial SMPL (ΔV=0) silhouette for frame 0,
    overlay against keyframe and SAM mask, print initial IoU. Caller (the
    user) inspects the overlay visually before any further M7 work.
    """
    device = settings.device

    kf_path = stage1_result.keyframe_paths[0]
    mask_path = stage1_result.mask_paths[0]
    keyframe_bgr = cv2.imread(str(kf_path))
    if keyframe_bgr is None:
        raise RuntimeError(f"Could not read keyframe: {kf_path}")
    H_orig, W_orig = keyframe_bgr.shape[:2]

    Ht, Wt, scale = _compute_render_resolution(W_orig, H_orig)
    logger.info(
        "Sanity render frame 0: keyframe={}x{}, render={}x{}, scale={:.4f}",
        W_orig, H_orig, Wt, Ht, scale,
    )

    smpl_model = _load_smpl_model(device)
    logger.info("SMPL loaded from {}", settings.smpl_model_path)

    beta = torch.from_numpy(stage1_result.beta).to(device).float()
    theta_0 = torch.from_numpy(stage1_result.theta_per_frame[0:1]).to(device).float()
    cam_t_0 = torch.from_numpy(stage1_result.pred_cam_t_per_frame[0:1]).to(device).float()
    fl_orig_0 = float(stage1_result.focal_length_per_frame[0])

    delta_v = torch.zeros(6890, 3, device=device)

    with torch.no_grad():
        meshes, _ = _pose_meshes(smpl_model, beta, theta_0, delta_v, cam_t_0, device)
        fl_render_val = fl_orig_0 / _HMR2_CROP_SIZE * max(Wt, Ht)
        fl_render = torch.tensor([fl_render_val], device=device)
        cameras = build_cameras(fl_render, (Ht, Wt), device)
        renderer = build_silhouette_renderer(cameras, (Ht, Wt), FACES_PER_PIXEL)
        images = renderer(meshes)            # [1, Ht, Wt, 4]
        alpha = images[..., 3]               # [1, Ht, Wt] in [0, 1]

    mask_full = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_full is None:
        raise RuntimeError(f"Could not read mask: {mask_path}")
    mask_small = cv2.resize(mask_full, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
    mask_tensor = torch.from_numpy((mask_small > 127).astype(np.float32)).unsqueeze(0).to(device)

    pred_binary = (alpha > 0.5).float()
    iou_frame0 = float(hard_iou_per_frame(pred_binary, mask_tensor).item())
    coverage_pred = float(pred_binary.mean().item())
    coverage_target = float(mask_tensor.mean().item())
    logger.info(
        "Frame 00 initial (ΔV=0): IoU={:.4f}, pred_coverage={:.4f}, "
        "target_coverage={:.4f}",
        iou_frame0, coverage_pred, coverage_target,
    )

    alpha_np = alpha[0].detach().cpu().numpy()
    label = (
        f"frame 00 | render {Wt}x{Ht} | initial IoU={iou_frame0:.3f} "
        f"| green=SMPL alpha | red=SAM mask outline"
    )
    triptych = _silhouette_triptych(keyframe_bgr, mask_small, alpha_np, Wt, Ht, label)

    out_dir = stage1_result.run_dir / "stage2"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "sanity_frame00.png"
    cv2.imwrite(str(out_path), triptych)
    logger.info("Wrote {}", out_path)
    return out_path


def sanity_render_all_frames(stage1_result: Stage1Result) -> list[Path]:
    """Step-2: batched _pose_meshes + per-frame silhouette render over all
    N keyframes. Logs per-frame initial IoU + the mean, and saves a
    triptych overlay per frame.

    Same camera math as sanity_render_frame00 but batched. Verifies the
    LBS path works for every theta (not just frame 0) and surfaces any
    frame whose silhouette is grossly misplaced (which would signal a
    camera-path issue that needs revisiting before optimization).
    """
    device = settings.device
    N = stage1_result.n_frames

    # All keyframes share size (asserted by pipeline._assert_consistent_keyframe_size).
    W_orig, H_orig = stage1_result.image_size_wh
    Ht, Wt, scale = _compute_render_resolution(W_orig, H_orig)
    logger.info(
        "Sanity render all frames: N={}, keyframe={}x{}, render={}x{}",
        N, W_orig, H_orig, Wt, Ht,
    )

    smpl_model = _load_smpl_model(device)
    logger.info("SMPL loaded from {}", settings.smpl_model_path)

    beta = torch.from_numpy(stage1_result.beta).to(device).float()
    theta = torch.from_numpy(stage1_result.theta_per_frame).to(device).float()        # [N, 24, 3]
    cam_t = torch.from_numpy(stage1_result.pred_cam_t_per_frame).to(device).float()   # [N, 3]
    fl_orig = torch.from_numpy(stage1_result.focal_length_per_frame).to(device).float()  # [N]
    delta_v = torch.zeros(6890, 3, device=device)

    with torch.no_grad():
        meshes, _ = _pose_meshes(smpl_model, beta, theta, delta_v, cam_t, device)

        fl_render = fl_orig / _HMR2_CROP_SIZE * max(Wt, Ht)                            # [N]
        cameras = build_cameras(fl_render, (Ht, Wt), device)
        renderer = build_silhouette_renderer(cameras, (Ht, Wt), FACES_PER_PIXEL)

        images = renderer(meshes)            # [N, Ht, Wt, 4]
        alpha = images[..., 3]               # [N, Ht, Wt]

    # Target masks at render resolution.
    mask_imgs: list[np.ndarray] = []         # [Ht, Wt] uint8 0/255
    for mp in stage1_result.mask_paths:
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"Could not read mask: {mp}")
        mask_imgs.append(cv2.resize(m, (Wt, Ht), interpolation=cv2.INTER_NEAREST))
    mask_stack = np.stack([(m > 127).astype(np.float32) for m in mask_imgs])           # [N, Ht, Wt]
    mask_tensor = torch.from_numpy(mask_stack).to(device)

    pred_binary = (alpha > 0.5).float()
    iou_per_frame = hard_iou_per_frame(pred_binary, mask_tensor).cpu().numpy()         # [N]
    mean_iou = float(iou_per_frame.mean())

    out_dir = stage1_result.run_dir / "stage2"
    out_dir.mkdir(exist_ok=True)
    out_paths: list[Path] = []
    alpha_np_all = alpha.detach().cpu().numpy()                                        # [N, Ht, Wt]

    logger.info("Per-frame initial IoU (ΔV=0):")
    for i in range(N):
        idx = stage1_result.keyframe_paths[i].stem.split("_")[-1]
        kf_bgr = cv2.imread(str(stage1_result.keyframe_paths[i]))
        if kf_bgr is None:
            raise RuntimeError(f"Could not read keyframe: {stage1_result.keyframe_paths[i]}")
        iou_i = float(iou_per_frame[i])
        cov_pred = float(pred_binary[i].mean().item())
        cov_tgt = float(mask_tensor[i].mean().item())
        logger.info(
            "  frame {}: IoU={:.4f}  pred_cov={:.4f}  tgt_cov={:.4f}",
            idx, iou_i, cov_pred, cov_tgt,
        )
        label = f"frame {idx} | initial IoU={iou_i:.3f} | green=SMPL alpha | red=SAM mask"
        triptych = _silhouette_triptych(
            kf_bgr, mask_imgs[i], alpha_np_all[i], Wt, Ht, label,
        )
        out_path = out_dir / f"sanity_frame_{idx}.png"
        cv2.imwrite(str(out_path), triptych)
        out_paths.append(out_path)

    logger.info(
        "Mean initial IoU across {} frames: {:.4f} (min={:.4f}, max={:.4f})",
        N, mean_iou, float(iou_per_frame.min()), float(iou_per_frame.max()),
    )
    return out_paths


def run(stage1_result: Stage1Result) -> Stage2Result:
    """M7 Stage 2 entry point. Optimizes ΔV against Stage 1's masks,
    persists Stage2Result + loss_curve.png + per-frame overlays, runs the
    round-trip self-check, returns the result. Mirrors stage1.run's
    fail-loud persistence pattern.
    """
    out_dir = stage1_result.run_dir / "stage2"
    out_dir.mkdir(exist_ok=True)
    loss_curve_path = out_dir / "loss_curve.png"

    result = optimize_vertex_offsets(
        stage1_result, loss_curve_path=loss_curve_path,
    )

    npz_path = stage1_result.run_dir / "stage2_result.npz"
    save_stage2_result(result, npz_path)
    logger.info("Wrote {}", npz_path)
    _verify_stage2_round_trip(result, npz_path)

    logger.info("Stage 2 complete: run_id={}", result.run_id)
    return result


def _load_target_masks(
    stage1_result: Stage1Result, render_hw: tuple[int, int], device: str
) -> torch.Tensor:
    """Load SAM masks from disk, resize NEAREST to render_hw, return as
    [N, Ht, Wt] float32 tensor in {0, 1} on device."""
    Ht, Wt = render_hw
    stack = np.zeros((stage1_result.n_frames, Ht, Wt), dtype=np.float32)
    for i, mp in enumerate(stage1_result.mask_paths):
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"Could not read mask: {mp}")
        m_small = cv2.resize(m, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
        stack[i] = (m_small > 127).astype(np.float32)
    return torch.from_numpy(stack).to(device)


def optimize_vertex_offsets(
    stage1_result: Stage1Result,
    loss_curve_path: Path | None = None,
) -> Stage2Result:
    """Optimize per-vertex offset ΔV in canonical T-pose space against the
    N keyframe SAM masks via differentiable silhouette rendering + uniform
    Laplacian smoothing. M8 extends this with additional loss terms.

    Knobs in config.py: LEARNING_RATE, OPT_MAX_ITERS, GRAD_CLIP_NORM,
    LOSS_WEIGHTS, FACES_PER_PIXEL, RENDER_RESOLUTION.

    No LR schedule, no early stop, no recovery branches. The grad-clip
    plus the conservative LR is the entire defense against pitfall #3
    (Stage 2 explosions).

    Per-iter raw (pre-weight) losses are collected into Stage2Result's
    per_term_history dict so cross-run tuning comparisons can see which
    term is moving. The total weighted loss is in loss_history.

    If `loss_curve_path` is provided, writes a loss curve PNG to that
    location.
    """
    device = settings.device
    N = stage1_result.n_frames

    is_cuda = device.startswith("cuda")
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()  # defaults to current device
    t0 = time.perf_counter()

    # --- Setup ---
    W_orig, H_orig = stage1_result.image_size_wh
    Ht, Wt, _scale = _compute_render_resolution(W_orig, H_orig)
    logger.info(
        "Stage 2 setup: N={}, render={}x{}, lr={}, iters={}, "
        "w_silh={}, w_lap={}, w_nc={}, w_height={}, w_kp={}, w_sym={}, grad_clip={}, "
        "target_height={:.2f}m±{:.2f}m",
        N, Wt, Ht, LEARNING_RATE, OPT_MAX_ITERS,
        LOSS_WEIGHTS["silhouette"], LOSS_WEIGHTS["laplacian"],
        LOSS_WEIGHTS["normal_consistency"], LOSS_WEIGHTS["height"],
        LOSS_WEIGHTS["keypoint"], LOSS_WEIGHTS["symmetry"],
        GRAD_CLIP_NORM,
        stage1_result.height_cm / 100.0, HEIGHT_TOLERANCE_M,
    )

    smpl_model = _load_smpl_model(device)
    silh_region_weights, sym_region_weights, right_to_left = _build_region_weights(
        smpl_model, device
    )

    # Detached pass-through tensors from Stage 1. torch.from_numpy + .to()
    # produces requires_grad=False by default, but make it explicit so a
    # future reader doesn't have to ask.
    beta = torch.from_numpy(stage1_result.beta).to(device).float().requires_grad_(False)
    theta = torch.from_numpy(stage1_result.theta_per_frame).to(device).float().requires_grad_(False)
    cam_t = torch.from_numpy(stage1_result.pred_cam_t_per_frame).to(device).float().requires_grad_(False)
    fl_orig = torch.from_numpy(stage1_result.focal_length_per_frame).to(device).float()

    # v_shape_only is beta-dependent but beta is frozen; compute once outside loop.
    # Used to re-construct v_canonical = v_template + v_shape_only + delta_v each iter.
    with torch.no_grad():
        v_shape_only = smplx.lbs.blend_shapes(
            beta.unsqueeze(0), smpl_model.shapedirs
        ).squeeze(0).detach()
    target_height_m: float = stage1_result.height_cm / 100.0

    # Keypoints scaled to render resolution (original coords × Wt/W_orig).
    kp_scale = float(Wt) / W_orig
    kp_xy = torch.from_numpy(
        stage1_result.keypoints_2d.astype(np.float32) * kp_scale
    ).to(device)  # [N, 17, 2]
    kp_scores = torch.from_numpy(
        stage1_result.keypoint_scores.astype(np.float32)
    ).to(device)  # [N, 17]

    fl_render = fl_orig / _HMR2_CROP_SIZE * max(Wt, Ht)
    cameras = build_cameras(fl_render, (Ht, Wt), device)
    renderer = build_silhouette_renderer(cameras, (Ht, Wt), FACES_PER_PIXEL)
    target_masks = _load_target_masks(stage1_result, (Ht, Wt), device)  # [N, Ht, Wt]

    # ΔV — the only optimization variable.
    delta_v = torch.nn.Parameter(torch.zeros(6890, 3, device=device, dtype=torch.float32))
    optimizer = torch.optim.Adam([delta_v], lr=LEARNING_RATE)

    # --- Initial IoU (ΔV=0) + weight map ---
    with torch.no_grad():
        meshes0, _ = _pose_meshes(smpl_model, beta, theta, delta_v, cam_t, device)
        alpha0 = renderer(meshes0)[..., 3]
        initial_iou = hard_iou_per_frame((alpha0 > 0.5).float(), target_masks).cpu().numpy()
        # Silhouette region weight map: computed once from the initial mesh.
        # silh_region_weights are fixed vertex labels — they don't change as
        # ΔV is optimized, and vertex positions shift only ~mm over 200 iters,
        # not enough to move region boundaries meaningfully.
        weight_map = _render_weight_map(meshes0, silh_region_weights, renderer)
    logger.info(
        "Initial mean IoU (ΔV=0): {:.4f} (min={:.4f}, max={:.4f})",
        float(initial_iou.mean()), float(initial_iou.min()), float(initial_iou.max()),
    )

    # --- Optimization loop ---
    # loss_history holds the TOTAL (weighted) loss; per_term_lists holds
    # the RAW (pre-weight) per-term losses, one list per active term.
    # Added terms in later M8 steps just register a new key here — the
    # save/load/round-trip path doesn't change.
    loss_history: list[float] = []
    per_term_lists: dict[str, list[float]] = {
        "silhouette": [],
        "laplacian": [],
        "normal_consistency": [],
        "height": [],
        "keypoint": [],
        "symmetry": [],
    }
    LOG_EVERY = 10

    w_silh   = float(LOSS_WEIGHTS["silhouette"])
    w_lap    = float(LOSS_WEIGHTS["laplacian"])
    w_nc     = float(LOSS_WEIGHTS["normal_consistency"])
    w_height = float(LOSS_WEIGHTS["height"])
    w_kp     = float(LOSS_WEIGHTS["keypoint"])
    w_sym    = float(LOSS_WEIGHTS["symmetry"])

    for it in range(OPT_MAX_ITERS):
        optimizer.zero_grad()

        v_canonical = smpl_model.v_template + v_shape_only + delta_v  # [6890, 3]
        meshes, joints_world = _pose_meshes(smpl_model, beta, theta, delta_v, cam_t, device)
        alpha = renderer(meshes)[..., 3]                 # [N, Ht, Wt]

        L_silh   = weighted_silhouette_iou_loss(alpha, target_masks, weight_map)
        L_lap    = laplacian_smoothing_loss(meshes)
        L_nc     = normal_consistency_loss(meshes)
        L_height = height_loss(v_canonical, target_height_m, HEIGHT_TOLERANCE_M)
        L_kp     = keypoint_reprojection_loss(
            joints_world, kp_xy, kp_scores, _COCO_TO_SMPL, fl_render, (Ht, Wt),
        )
        L_sym    = symmetry_loss(delta_v, sym_region_weights, right_to_left)
        L = (w_silh * L_silh + w_lap * L_lap + w_nc * L_nc
             + w_height * L_height + w_kp * L_kp + w_sym * L_sym)

        L.backward()
        torch.nn.utils.clip_grad_norm_([delta_v], max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        loss_history.append(float(L.item()))
        per_term_lists["silhouette"].append(float(L_silh.item()))
        per_term_lists["laplacian"].append(float(L_lap.item()))
        per_term_lists["normal_consistency"].append(float(L_nc.item()))
        per_term_lists["height"].append(float(L_height.item()))
        per_term_lists["keypoint"].append(float(L_kp.item()))
        per_term_lists["symmetry"].append(float(L_sym.item()))
        if it % LOG_EVERY == 0 or it == OPT_MAX_ITERS - 1:
            logger.info(
                "iter {:3d}: L={:.5f}  silh={:.5f}  lap={:.5f}  nc={:.5f}  "
                "height={:.5f}  kp={:.2f}  sym={:.5f}  |ΔV|_max={:.5f}",
                it, float(L.item()), float(L_silh.item()), float(L_lap.item()),
                float(L_nc.item()), float(L_height.item()), float(L_kp.item()),
                float(L_sym.item()),
                float(delta_v.detach().abs().max().item()),
            )

    per_term_history: dict[str, np.ndarray] = {
        name: np.array(lst, dtype=np.float32) for name, lst in per_term_lists.items()
    }

    # --- Final IoU + bookkeeping ---
    with torch.no_grad():
        meshes_f, _ = _pose_meshes(smpl_model, beta, theta, delta_v, cam_t, device)
        alpha_f = renderer(meshes_f)[..., 3]
        final_iou = hard_iou_per_frame((alpha_f > 0.5).float(), target_masks).cpu().numpy()

    elapsed = time.perf_counter() - t0
    peak_vram = (
        torch.cuda.max_memory_allocated() / 1024**3 if is_cuda else float("nan")
    )

    delta_v_final = delta_v.detach().cpu().numpy().astype(np.float32)
    max_abs_dv = float(np.abs(delta_v_final).max())
    logger.info(
        "Stage 2 done: final mean IoU={:.4f} (init={:.4f}, Δ={:+.4f}), "
        "max|ΔV|={:.5f}, wall={:.1f}s, peak_VRAM={:.2f} GB",
        float(final_iou.mean()), float(initial_iou.mean()),
        float(final_iou.mean() - initial_iou.mean()),
        max_abs_dv, elapsed, peak_vram,
    )

    # --- Loss curve (optional) ---
    if loss_curve_path is not None:
        iters = np.arange(OPT_MAX_ITERS)
        total_arr = np.array(loss_history, dtype=np.float32)
        silh_arr = per_term_history["silhouette"]
        lap_weighted_arr = w_lap * per_term_history["laplacian"]
        nc_weighted_arr = w_nc * per_term_history["normal_consistency"]
        height_weighted_arr = w_height * per_term_history["height"]
        kp_weighted_arr  = w_kp * per_term_history["keypoint"]
        sym_weighted_arr = w_sym * per_term_history["symmetry"]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(iters, total_arr, label="total", linewidth=2.0, color="black")
        ax.plot(iters, silh_arr, label="silhouette (w=1)", linewidth=1.2, color="tab:blue")
        ax.plot(
            iters, lap_weighted_arr,
            label=f"laplacian (w={int(w_lap)})", linewidth=1.2, color="tab:orange",
        )
        ax.plot(
            iters, nc_weighted_arr,
            label=f"normal_consistency (w={w_nc})", linewidth=1.2, color="tab:green",
        )
        ax.plot(
            iters, height_weighted_arr,
            label=f"height (w={w_height})", linewidth=1.2, color="tab:red",
        )
        ax.plot(
            iters, kp_weighted_arr,
            label=f"keypoint (w={w_kp})", linewidth=1.2, color="tab:purple",
        )
        ax.plot(
            iters, sym_weighted_arr,
            label=f"symmetry (w={w_sym})", linewidth=1.2, color="tab:brown",
        )
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.set_title(
            f"Stage 2 loss curve (lr={LEARNING_RATE}, iters={OPT_MAX_ITERS}, "
            f"grad_clip={GRAD_CLIP_NORM})"
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(loss_curve_path), dpi=120)
        plt.close(fig)
        logger.info("Wrote loss curve to {}", loss_curve_path)

    return Stage2Result(
        run_id=stage1_result.run_id,
        run_dir=stage1_result.run_dir,
        delta_v=delta_v_final,
        beta=stage1_result.beta.astype(np.float32),
        theta_per_frame=stage1_result.theta_per_frame.astype(np.float32),
        n_iterations=OPT_MAX_ITERS,
        final_loss=loss_history[-1],
        loss_history=np.array(loss_history, dtype=np.float32),
        initial_iou_per_frame=initial_iou.astype(np.float32),
        final_iou_per_frame=final_iou.astype(np.float32),
        per_term_history=per_term_history,
    )


def save_silhouette_debug(
    stage1_result: Stage1Result,
    stage2_result: Stage2Result,
    out_dir: Path | None = None,
) -> tuple[list[Path], list[Path]]:
    """Persist per-frame init/final silhouette overlay triptychs to disk.

    Self-contained: re-renders BOTH the ΔV=0 baseline and the optimized
    ΔV state with two batched forward passes. Does NOT depend on
    sanity_render_all_frames having been called first — pipeline.run
    can invoke this directly after optimize_vertex_offsets.

    The IoU values labeled on each frame come from
    stage2_result.initial_iou_per_frame / final_iou_per_frame so the
    triptych labels stay in lock-step with the stored result.

    Same 3-panel format as the sanity overlays: keyframe / keyframe +
    green SMPL silhouette + red SAM mask outline / silhouette alpha alone.
    """
    device = settings.device
    N = stage1_result.n_frames

    if out_dir is None:
        out_dir = stage1_result.run_dir / "stage2"
    out_dir.mkdir(exist_ok=True)

    W_orig, H_orig = stage1_result.image_size_wh
    Ht, Wt, _scale = _compute_render_resolution(W_orig, H_orig)

    # Single setup, two forward passes (init + final). Sub-second total
    # for N=12 at 144x256.
    smpl_model = _load_smpl_model(device)
    beta = torch.from_numpy(stage1_result.beta).to(device).float()
    theta = torch.from_numpy(stage1_result.theta_per_frame).to(device).float()
    cam_t = torch.from_numpy(stage1_result.pred_cam_t_per_frame).to(device).float()
    fl_orig = torch.from_numpy(stage1_result.focal_length_per_frame).to(device).float()
    fl_render = fl_orig / _HMR2_CROP_SIZE * max(Wt, Ht)

    cameras = build_cameras(fl_render, (Ht, Wt), device)
    renderer = build_silhouette_renderer(cameras, (Ht, Wt), FACES_PER_PIXEL)

    delta_v_init = torch.zeros(6890, 3, device=device)
    delta_v_final = torch.from_numpy(stage2_result.delta_v).to(device).float()

    with torch.no_grad():
        meshes_init, _ = _pose_meshes(smpl_model, beta, theta, delta_v_init, cam_t, device)
        alpha_init = renderer(meshes_init)[..., 3].cpu().numpy()    # [N, Ht, Wt]
        meshes_final, _ = _pose_meshes(smpl_model, beta, theta, delta_v_final, cam_t, device)
        alpha_final = renderer(meshes_final)[..., 3].cpu().numpy()

    init_paths: list[Path] = []
    final_paths: list[Path] = []
    for i, (kf_path, mp) in enumerate(
        zip(stage1_result.keyframe_paths, stage1_result.mask_paths)
    ):
        idx = kf_path.stem.split("_")[-1]
        kf_bgr = cv2.imread(str(kf_path))
        if kf_bgr is None:
            raise RuntimeError(f"Could not read keyframe: {kf_path}")
        mask_full = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask_full is None:
            raise RuntimeError(f"Could not read mask: {mp}")
        mask_small = cv2.resize(mask_full, (Wt, Ht), interpolation=cv2.INTER_NEAREST)

        init_iou = float(stage2_result.initial_iou_per_frame[i])
        final_iou = float(stage2_result.final_iou_per_frame[i])
        init_label = (
            f"frame {idx} | initial IoU={init_iou:.3f} | "
            f"green=SMPL alpha | red=SAM mask"
        )
        final_label = (
            f"frame {idx} | final IoU={final_iou:.3f} | "
            f"green=SMPL alpha | red=SAM mask"
        )

        init_triptych = _silhouette_triptych(
            kf_bgr, mask_small, alpha_init[i], Wt, Ht, init_label,
        )
        final_triptych = _silhouette_triptych(
            kf_bgr, mask_small, alpha_final[i], Wt, Ht, final_label,
        )

        init_out = out_dir / f"overlay_init_{idx}.png"
        final_out = out_dir / f"overlay_final_{idx}.png"
        cv2.imwrite(str(init_out), init_triptych)
        cv2.imwrite(str(final_out), final_triptych)
        init_paths.append(init_out)
        final_paths.append(final_out)

    logger.info(
        "Saved {} init + {} final overlays to {}",
        len(init_paths), len(final_paths), out_dir,
    )
    return init_paths, final_paths
