"""PyTorch3D differentiable rendering helpers for Stage 2 optimization."""

import numpy as np
import torch
from pytorch3d.renderer import (
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    RasterizationSettings,
    SoftSilhouetteShader,
)
from pytorch3d.utils import cameras_from_opencv_projection


# SoftSilhouetteShader textbook values (Kato & Harada 2018).
# blur_radius = log(1/sigma - 1) * sigma gives a soft alpha with gradient
# through ~few-pixel band in NDC space.
_SIGMA = 1e-4
_BLUR_RADIUS = float(np.log(1.0 / _SIGMA - 1.0) * _SIGMA)


def build_cameras(
    focal_length_render_px: torch.Tensor,   # [N] pixels at render resolution
    image_size_hw: tuple[int, int],         # (Ht, Wt) at render resolution
    device: str,
) -> PerspectiveCameras:
    """PyTorch3D PerspectiveCameras matching hmr2's pinhole at the origin.

    Returns a batched-N camera with identity OpenCV extrinsics. Verts passed
    to the renderer must already be in the OpenCV camera frame — apply the
    SMPL→OpenCV transform `(v_smpl + cam_t) * [1, -1, -1]` upstream (this
    mirrors hmr2's `R_180x @ (v + cam_t)` in third_party/4D-Humans/hmr2/utils/
    renderer.py:vertices_to_trimesh L245+L252-254).

    Principal point is the image center (Wt/2, Ht/2). focal_length is in
    render-resolution pixels (scale upstream from the keyframe-resolution
    value in Stage1Result.focal_length_per_frame).
    """
    N = focal_length_render_px.shape[0]
    Ht, Wt = image_size_hw

    fl = focal_length_render_px.to(device).float()
    K = torch.zeros(N, 3, 3, device=device)
    K[:, 0, 0] = fl                        # fx
    K[:, 1, 1] = fl                        # fy
    K[:, 0, 2] = Wt / 2.0                  # cx
    K[:, 1, 2] = Ht / 2.0                  # cy
    K[:, 2, 2] = 1.0

    R = torch.eye(3, device=device).unsqueeze(0).expand(N, 3, 3).contiguous()
    t = torch.zeros(N, 3, device=device)
    image_size = torch.tensor([[Ht, Wt]], device=device, dtype=torch.int64).expand(N, 2)

    return cameras_from_opencv_projection(
        R=R, tvec=t, camera_matrix=K, image_size=image_size,
    )


def build_normal_rasterizer(
    cameras: PerspectiveCameras,
    image_size_hw: tuple[int, int],
) -> MeshRasterizer:
    """Hard rasterizer for normal-map rendering — no blur, nearest face only.

    Returns a MeshRasterizer (no shader); the caller interpolates vertex
    normals from fragments via interpolate_face_attributes. blur_radius=0
    avoids the soft-blend band that would contaminate normal interpolation
    at silhouette edges.
    """
    Ht, Wt = image_size_hw
    raster_settings = RasterizationSettings(
        image_size=(Ht, Wt),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    return MeshRasterizer(cameras=cameras, raster_settings=raster_settings)


def build_silhouette_renderer(
    cameras: PerspectiveCameras,
    image_size_hw: tuple[int, int],
    faces_per_pixel: int,
) -> MeshRenderer:
    """SoftSilhouetteShader + MeshRasterizer; alpha channel of the output is
    the differentiable soft silhouette in [0, 1]."""
    Ht, Wt = image_size_hw
    raster_settings = RasterizationSettings(
        image_size=(Ht, Wt),
        blur_radius=_BLUR_RADIUS,
        faces_per_pixel=faces_per_pixel,
    )
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(),
    )
