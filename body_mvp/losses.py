"""Stage 2 loss terms: silhouette IoU (soft + hard), weighted silhouette IoU,
height match, keypoint reprojection, Laplacian smoothing, normal consistency."""

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
) -> torch.Tensor:
    """Weighted soft-IoU loss. weight_map scales each pixel's contribution to
    both intersection and union, reducing the gradient from regions where SAM
    mask contamination is expected (hair, shoes, sleeves). Averaged over N
    frames. Same min/max formulation as silhouette_iou_loss.
    """
    pmin = torch.minimum(pred_alpha, target_mask)
    pmax = torch.maximum(pred_alpha, target_mask)
    intersection = (pmin * weight_map).flatten(1).sum(dim=1)
    union = (pmax * weight_map).flatten(1).sum(dim=1) + eps
    iou = intersection / union
    return (1.0 - iou).mean()


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


def laplacian_smoothing_loss(meshes: Meshes) -> torch.Tensor:
    """Uniform Laplacian smoothing regularizer (PyTorch3D wrapper)."""
    return mesh_laplacian_smoothing(meshes, method="uniform")


def normal_consistency_loss(meshes: Meshes) -> torch.Tensor:
    """Adjacent-face normal consistency regularizer (PyTorch3D wrapper)."""
    return mesh_normal_consistency(meshes)
