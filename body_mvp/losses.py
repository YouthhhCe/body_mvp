"""Stage 2 loss terms: silhouette IoU (soft + hard), weighted silhouette IoU,
height match, keypoint reprojection, bilateral symmetry, Laplacian smoothing,
normal consistency, normal-map alignment."""

import torch
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.structures import Meshes


def silhouette_iou_loss(
    pred_alpha: torch.Tensor,    # [N, H, W] in [0, 1] (SoftSilhouetteShader alpha)
    target_mask: torch.Tensor,   # [N, H, W] float, {0, 1}
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable soft-IoU loss, averaged over the batch.

    `1 - sum(min(p, t)) / sum(max(p, t))`. Standard soft-IoU per
    Kato & Harada 2018 — plays nicely with SoftSilhouetteShader's
    blended alpha because min/max preserve gradients through the
    blur band where pred_alpha ∈ (0, 1).
    """
    pmin = torch.minimum(pred_alpha, target_mask)
    pmax = torch.maximum(pred_alpha, target_mask)
    intersection = pmin.flatten(1).sum(dim=1)
    union = pmax.flatten(1).sum(dim=1) + eps
    iou = intersection / union
    return (1.0 - iou).mean()


def weighted_silhouette_iou_loss(
    pred_alpha: torch.Tensor,    # [N, H, W] in [0, 1] (SoftSilhouetteShader alpha)
    target_mask: torch.Tensor,   # [N, H, W] float, {0, 1}
    weight_map: torch.Tensor,    # [N, H, W] float, per-pixel weights ≥ 0
    eps: float = 1e-6,
    per_frame: bool = False,
) -> torch.Tensor:
    """Weighted soft-IoU loss. weight_map scales each pixel's contribution to
    both intersection and union, reducing the gradient from regions where SAM
    mask contamination is expected (hair, shoes, sleeves). Averaged over N
    frames. Same min/max formulation as silhouette_iou_loss.

    per_frame=True returns the [N] per-frame loss values (before .mean()) so
    callers can apply their own frame-level weighting (e.g. orientation-balanced
    weighting in optimize_vertex_offsets).
    """
    pmin = torch.minimum(pred_alpha, target_mask)
    pmax = torch.maximum(pred_alpha, target_mask)
    intersection = (pmin * weight_map).flatten(1).sum(dim=1)
    union = (pmax * weight_map).flatten(1).sum(dim=1) + eps
    loss = 1.0 - intersection / union   # [N]
    return loss if per_frame else loss.mean()


def hard_iou_per_frame(
    pred_binary: torch.Tensor,   # [N, H, W] bool or {0, 1} float
    target_binary: torch.Tensor, # [N, H, W] bool or {0, 1} float
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-frame binary IoU [N], for logging/reporting. Not differentiable."""
    p = pred_binary.bool()
    t = target_binary.bool()
    inter = (p & t).flatten(1).sum(dim=1).float()
    union = (p | t).flatten(1).sum(dim=1).float() + eps
    return inter / union


def height_loss(
    v_canonical: torch.Tensor,  # [6890, 3]
    target_height_m: float,
    tolerance_m: float,
) -> torch.Tensor:
    """Squared hinge loss on the mesh Y-span vs user-supplied target height.

    v_canonical is in SMPL canonical T-pose space (+Y up). Y-span = max_y -
    min_y approximates standing height in meters. Returns zero when the span
    is within [target - tolerance, target + tolerance]; otherwise (excess)².
    """
    y_span = v_canonical[:, 1].max() - v_canonical[:, 1].min()
    excess = (torch.abs(y_span - target_height_m) - tolerance_m).clamp(min=0.0)
    return excess * excess


def keypoint_reprojection_loss(
    joints_world: torch.Tensor,               # [N, 24, 3]
    keypoints_2d: torch.Tensor,               # [N, 17, 2] (x,y) in render pixels
    keypoint_scores: torch.Tensor,            # [N, 17]
    coco_to_smpl: tuple[tuple[int, int], ...],
    fl_render: torch.Tensor,                  # [N] focal length at render resolution
    image_hw: tuple[int, int],                # (Ht, Wt)
    score_threshold: float = 0.3,
) -> torch.Tensor:
    """Mean squared pixel reprojection error over COCO-SMPL joint pairs that
    pass the score gate. Returns zero if no valid pairs exist.

    Projection: u = fl*X/Z + cx, v = fl*Y/Z + cy (OpenCV, Z=depth, +Y down).
    joints_world is already in camera space (pred_cam_t applied in _pose_meshes).
    """
    Ht, Wt = image_hw
    cx = Wt * 0.5
    cy = Ht * 0.5
    coco_idxs = [pair[0] for pair in coco_to_smpl]
    smpl_idxs  = [pair[1] for pair in coco_to_smpl]

    kp_paired     = keypoints_2d[:, coco_idxs, :]     # [N, P, 2]
    joints_paired = joints_world[:, smpl_idxs, :]     # [N, P, 3]
    scores_paired = keypoint_scores[:, coco_idxs]     # [N, P]

    fl    = fl_render.unsqueeze(1)                     # [N, 1]
    Z     = joints_paired[:, :, 2].clamp(min=1e-3)    # [N, P]
    proj_u = fl * joints_paired[:, :, 0] / Z + cx     # [N, P]
    proj_v = fl * joints_paired[:, :, 1] / Z + cy     # [N, P]

    valid   = scores_paired > score_threshold          # [N, P] bool
    n_valid = valid.float().sum().clamp(min=1.0)
    err2    = (proj_u - kp_paired[:, :, 0]) ** 2 + (proj_v - kp_paired[:, :, 1]) ** 2
    return (err2 * valid.float()).sum() / n_valid


def symmetry_loss(
    delta_v: torch.Tensor,              # [6890, 3]
    sym_region_weights: torch.Tensor,   # [6890] float32 — from _build_region_weights
    right_to_left: torch.Tensor,        # [6890] int64  — mirror map from _build_region_weights
) -> torch.Tensor:
    """Weighted bilateral symmetry regularizer on ΔV in canonical T-pose space.

    For each vertex i, penalizes |ΔV[i] − mirror(ΔV[j])|² where j is its
    mirror partner. mirror flips x (canonical +x = subject's left).

    Belly vertices (sym_weight=0) and non-involution vertices (sym_weight=0
    post-gate in D7) are excluded. Normalised by total weight sum so sparse
    weights don't dilute the signal.
    """
    mirror_sign  = delta_v.new_tensor([-1., 1., 1.])    # flip x only
    dv_partner   = delta_v[right_to_left]                # [6890, 3] — partner's ΔV
    dv_reflected = dv_partner * mirror_sign              # partner's ΔV as seen from this side
    err2 = ((delta_v - dv_reflected) ** 2).sum(dim=1)   # [6890]
    w_sum = sym_region_weights.sum().clamp(min=1.0)
    return (sym_region_weights * err2).sum() / w_sum


def normal_map_loss(
    rendered_normals: torch.Tensor,   # [N, H, W, 3] unit normals, PyTorch3D camera space
    sapiens_normals: torch.Tensor,    # [N, H, W, 3] unit normals, same convention (pre-flipped)
    sapiens_fg: torch.Tensor,         # [N, H, W] bool
    rendered_fg: torch.Tensor,        # [N, H, W] bool
    grazing_threshold: float = 0.5,   # |n_z| gate; D14 tuning target
    edge_erosion_px: int = 4,         # silhouette-edge erosion; D14 tuning target
) -> torch.Tensor:
    """Mean cosine-distance loss gated to face-on interior pixels.

    Two masks, both detached so ΔV gradient flows through normal values only,
    not through the mask selection boundaries:

    grazing_gate  : |rendered_n_z| > grazing_threshold — excludes near-perpendicular
                    surfaces where SMPL vertex-normal interpolation is unreliable
                    at coarse mesh resolution. Recomputed each iteration as ΔV
                    deforms the mesh (unlike D8's weight_map which is fixed).

    interior_gate : rendered_fg eroded by edge_erosion_px pixels (max_pool dilation
                    of background) — excludes silhouette-edge pixels where
                    rasterization boundary artifacts spike the error.

    D12 diagnosis: face-on interior mean cosine error 0.060 vs edge/grazing
    0.41–0.53 (7× gap). Both thresholds are D14 tuning targets.
    """
    # Grazing gate — detach so the threshold boundary carries no gradient.
    # rendered_normals[..., 2] has grad_fn from delta_v; .detach() cuts it
    # for the mask only; the dot product below still receives the full gradient.
    grazing_gate = (rendered_normals[..., 2].abs() > grazing_threshold).detach()  # [N, H, W]

    # Edge erosion gate via dilation of background (max_pool on inverted fg).
    # Pixels within edge_erosion_px of the silhouette edge become background.
    if edge_erosion_px > 0:
        k = edge_erosion_px
        fg_4d = rendered_fg.float().unsqueeze(1)               # [N, 1, H, W]
        dilated_bg = torch.nn.functional.max_pool2d(
            1.0 - fg_4d, kernel_size=2 * k + 1, stride=1, padding=k,
        )
        interior_gate = ((1.0 - dilated_bg).squeeze(1) > 0.5).detach()  # [N, H, W] bool
    else:
        interior_gate = rendered_fg

    valid = sapiens_fg & interior_gate & grazing_gate          # [N, H, W] bool
    n_valid = valid.float().sum().clamp(min=1.0)
    dot = (rendered_normals * sapiens_normals).sum(dim=-1)    # [N, H, W]
    cosine_dist = 1.0 - dot
    return (cosine_dist * valid.float()).sum() / n_valid


def laplacian_smoothing_loss(meshes: Meshes) -> torch.Tensor:
    """Uniform Laplacian smoothing regularizer (PyTorch3D wrapper)."""
    return mesh_laplacian_smoothing(meshes, method="uniform")


def normal_consistency_loss(meshes: Meshes) -> torch.Tensor:
    """Adjacent-face normal consistency regularizer (PyTorch3D wrapper)."""
    return mesh_normal_consistency(meshes)
