"""Stage 2 loss terms: silhouette IoU (soft + hard), Laplacian smoothing."""

import torch
from pytorch3d.loss import mesh_laplacian_smoothing
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


def laplacian_smoothing_loss(meshes: Meshes) -> torch.Tensor:
    """Uniform Laplacian smoothing regularizer (PyTorch3D wrapper)."""
    return mesh_laplacian_smoothing(meshes, method="uniform")
