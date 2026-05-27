from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import smplx
import smplx.lbs
import torch
import trimesh
from loguru import logger
from pytorch3d.renderer import (
    BlendParams,
    DirectionalLights,
    FoVPerspectiveCameras,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix

from body_mvp.config import settings
from body_mvp.stage1 import Stage1Result
from body_mvp.stage2 import Stage2Result


@dataclass
class QualityReport:
    """Minimal quality signal for Layer 2 consumption.

    overall_score: 1.0 → 0.0, starts at 1.0, deductions per warning.
    warnings: human-readable list of issues found.
    """

    overall_score: float
    warnings: list[str]

    def __post_init__(self) -> None:
        if not (0.0 <= self.overall_score <= 1.0):
            raise ValueError(
                f"overall_score: must be in [0.0, 1.0], got {self.overall_score}"
            )


@dataclass
class Stage3Result:
    """Output of Stage 3 — display GLB + complete analysis data for Layer 2.

    Persisted to a single .npz at the run dir root via save_stage3_result;
    reload via load_stage3_result. Native numpy dtypes only, no pickle.
    """

    run_id: str
    run_dir: Path

    # Display branch
    vertices_a_pose: np.ndarray       # [6890, 3] float32 — mesh in A-pose
    glb_path: Path
    thumbnail_path: Path

    # Analysis branch
    vertices_canonical: np.ndarray    # [6890, 3] float32 — T-pose (pure β mesh)
    delta_v: np.ndarray               # [6890, 3] float32 — zero in MVP
    beta: np.ndarray                  # [10]      float32 — SMPL shape params
    theta_natural: np.ndarray         # [24, 3]   float32 — representative standing pose
    theta_per_keyframe: np.ndarray    # [N, 24, 3] float32 — all keyframe poses
    joints_canonical: np.ndarray      # [24, 3]   float32 — T-pose joint positions
    joints_natural: np.ndarray        # [24, 3]   float32 — natural-pose joint positions
    scale_to_meters: float
    quality: QualityReport

    def __post_init__(self) -> None:
        N = self.theta_per_keyframe.shape[0]
        if N <= 0:
            raise ValueError(f"theta_per_keyframe: leading dim must be > 0, got {N}")

        def _check_shape(name: str, arr: np.ndarray, expected: tuple) -> None:
            if arr.shape != expected:
                raise ValueError(
                    f"{name}: expected shape {expected}, got {tuple(arr.shape)}"
                )

        _check_shape("vertices_a_pose", self.vertices_a_pose, (6890, 3))
        _check_shape("vertices_canonical", self.vertices_canonical, (6890, 3))
        _check_shape("delta_v", self.delta_v, (6890, 3))
        _check_shape("beta", self.beta, (10,))
        _check_shape("theta_natural", self.theta_natural, (24, 3))
        _check_shape("theta_per_keyframe", self.theta_per_keyframe, (N, 24, 3))
        _check_shape("joints_canonical", self.joints_canonical, (24, 3))
        _check_shape("joints_natural", self.joints_natural, (24, 3))

        if self.scale_to_meters <= 0:
            raise ValueError(
                f"scale_to_meters: must be > 0, got {self.scale_to_meters}"
            )


def save_stage3_result(result: Stage3Result, path: Path) -> None:
    """Write Stage3Result to a single .npz at `path`.

    Mirrors save_stage1_result / save_stage2_result: native dtypes, no pickle,
    strings as 0-d unicode arrays, paths as str. QualityReport is flattened
    into quality_score + quality_warnings arrays.
    """
    assert path.suffix == ".npz", (
        f"save_stage3_result: path must end in .npz to match np.savez's own "
        f"behavior (it silently appends .npz otherwise, desyncing the file "
        f"path the caller thinks they wrote). Got: {path}"
    )
    np.savez(
        str(path),
        run_id=np.array(result.run_id),
        run_dir=np.array(str(result.run_dir)),
        vertices_a_pose=result.vertices_a_pose.astype(np.float32),
        glb_path=np.array(str(result.glb_path)),
        thumbnail_path=np.array(str(result.thumbnail_path)),
        vertices_canonical=result.vertices_canonical.astype(np.float32),
        delta_v=result.delta_v.astype(np.float32),
        beta=result.beta.astype(np.float32),
        theta_natural=result.theta_natural.astype(np.float32),
        theta_per_keyframe=result.theta_per_keyframe.astype(np.float32),
        joints_canonical=result.joints_canonical.astype(np.float32),
        joints_natural=result.joints_natural.astype(np.float32),
        scale_to_meters=np.array(result.scale_to_meters, dtype=np.float32),
        quality_score=np.array(result.quality.overall_score, dtype=np.float64),
        quality_warnings=np.array(result.quality.warnings, dtype=np.str_),
    )


def load_stage3_result(path: Path) -> Stage3Result:
    """Inverse of save_stage3_result. allow_pickle=False is safe — every
    field is stored as a native numpy dtype."""
    data = np.load(str(path), allow_pickle=False)

    warnings = data["quality_warnings"]
    if warnings.ndim == 0:
        warnings_list: list[str] = [str(warnings.item())]
    else:
        warnings_list = [str(w) for w in warnings.tolist()]

    quality = QualityReport(
        overall_score=float(data["quality_score"].item()),
        warnings=warnings_list,
    )

    return Stage3Result(
        run_id=str(data["run_id"].item()),
        run_dir=Path(str(data["run_dir"].item())),
        vertices_a_pose=data["vertices_a_pose"],
        glb_path=Path(str(data["glb_path"].item())),
        thumbnail_path=Path(str(data["thumbnail_path"].item())),
        vertices_canonical=data["vertices_canonical"],
        delta_v=data["delta_v"],
        beta=data["beta"],
        theta_natural=data["theta_natural"],
        theta_per_keyframe=data["theta_per_keyframe"],
        joints_canonical=data["joints_canonical"],
        joints_natural=data["joints_natural"],
        scale_to_meters=float(data["scale_to_meters"].item()),
        quality=quality,
    )


def _verify_stage3_round_trip(original: Stage3Result, path: Path) -> None:
    """Load the just-written Stage3Result and fail-loud if anything drifted.

    Same belt-and-suspenders pattern as Stage 1 and Stage 2: cheap enough
    to run on every save (no pixel data in the bundle).
    """
    loaded = load_stage3_result(path)

    scalar_fields = ("run_id", "run_dir", "glb_path", "thumbnail_path", "scale_to_meters")
    for name in scalar_fields:
        a = getattr(original, name)
        b = getattr(loaded, name)
        if a != b:
            raise RuntimeError(
                f"round-trip mismatch on {name}: original={a!r} loaded={b!r}"
            )

    array_fields = (
        "vertices_a_pose", "vertices_canonical", "delta_v", "beta",
        "theta_natural", "theta_per_keyframe",
        "joints_canonical", "joints_natural",
    )
    for name in array_fields:
        a = getattr(original, name)
        b = getattr(loaded, name)
        if a.shape != b.shape:
            raise RuntimeError(
                f"round-trip shape mismatch on {name}: {a.shape} vs {b.shape}"
            )
        if a.dtype != b.dtype:
            raise RuntimeError(
                f"round-trip dtype mismatch on {name}: {a.dtype} vs {b.dtype}"
            )
        if not np.array_equal(a, b):
            raise RuntimeError(f"round-trip value mismatch on {name}")

    orig_q = original.quality
    loaded_q = loaded.quality
    if orig_q.overall_score != loaded_q.overall_score:
        raise RuntimeError(
            f"round-trip mismatch on quality.score: "
            f"{orig_q.overall_score} vs {loaded_q.overall_score}"
        )
    if orig_q.warnings != loaded_q.warnings:
        raise RuntimeError(
            f"round-trip mismatch on quality.warnings: "
            f"{orig_q.warnings} vs {loaded_q.warnings}"
        )

    logger.info("stage3_result.npz round-trip self-check passed")


# ---------------------------------------------------------------------------
# SMPL helpers
# ---------------------------------------------------------------------------

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


def _build_a_pose() -> np.ndarray:
    """Standardized conservative A-pose: arms ~35 deg below horizontal.

    SMPL rest pose has arms near-horizontal (0.5 deg below). Shoulder
    Z-rotation moves them up/down, mapping ~1:1 to arm depression angle.
    Verified experimentally (M9 planning): L_shoulder (16) -Z = arm down,
    R_shoulder (17) +Z = arm down. 0.6 rad → ~35 deg depression.

    Conservative angle chosen to minimize LBS candy-wrap (pitfall #4)
    while giving a clear A-silhouette for the viewer.

    Returns [24, 3] float32 axis-angle array. Joint 0 (global_orient) is
    identity — subject faces +Z in canonical coordinates.
    """
    pose = np.zeros((24, 3), dtype=np.float32)
    # L_shoulder (16): -Z rotates arm downward
    pose[16, 2] = -0.6
    # R_shoulder (17): +Z rotates arm downward
    pose[17, 2] = +0.6
    return pose


def _compute_canonical(
    beta: torch.Tensor,        # [10] float
    delta_v: torch.Tensor,     # [6890, 3] float
    smpl_model: smplx.SMPL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply β blendshapes + ΔV in canonical T-pose space.

    Returns (vertices_canonical [6890,3] float32,
             joints_canonical [24,3] float32,
             faces [13776,3] int64) — all on CPU.

    joints_canonical = J_regressor @ v_canonical, the rest-pose joint
    centres before any pose is applied.
    """
    v_shape_only = smplx.lbs.blend_shapes(
        beta.unsqueeze(0), smpl_model.shapedirs
    ).squeeze(0)
    v_canonical = smpl_model.v_template + v_shape_only + delta_v  # [6890, 3]

    J_dense = smpl_model.J_regressor.to_dense()  # [24, 6890]
    joints = J_dense @ v_canonical  # [24, 3]

    faces = smpl_model.faces.astype(np.int64)  # [13776, 3]

    return (
        v_canonical.detach().cpu().numpy().astype(np.float32),
        joints.detach().cpu().numpy().astype(np.float32),
        faces,
    )


def _pose_to(
    pose_aa: torch.Tensor,      # [24, 3] float (single pose, not batched)
    v_canonical: torch.Tensor,  # [6890, 3] float
    smpl_model: smplx.SMPL,
) -> tuple[np.ndarray, np.ndarray]:
    """LBS-pose the canonical mesh with `pose_aa`, no translation added.

    Reuses the exact same smplx.lbs.lbs() path that M7/Stage 2 verified
    (stage2._pose_meshes lines 726-736): pass betas=zeros with
    v_template=v_canonical (already shape-displaced), so blend_shapes
    inside lbs() is a no-op. Shape applied once, pose blendshapes applied
    on top.

    Returns (vertices [6890,3] float32, joints [24,3] float32) on CPU.
    Mesh is centred at origin — no cam_t offset.
    """
    device = v_canonical.device
    v_template_batched = v_canonical.unsqueeze(0)  # [1, 6890, 3]
    pose_batched = pose_aa.reshape(1, 72)           # [1, 72]
    betas_zero = torch.zeros(1, 10, device=device, dtype=torch.float32)

    verts, joints = smplx.lbs.lbs(
        betas=betas_zero,
        pose=pose_batched,
        v_template=v_template_batched,
        shapedirs=smpl_model.shapedirs,
        posedirs=smpl_model.posedirs,
        J_regressor=smpl_model.J_regressor,
        parents=smpl_model.parents,
        lbs_weights=smpl_model.lbs_weights,
        pose2rot=True,
    )  # verts [1, 6890, 3], joints [1, 24, 3]

    return (
        verts[0].detach().cpu().numpy().astype(np.float32),
        joints[0].detach().cpu().numpy().astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Display branch
# ---------------------------------------------------------------------------

def export_glb(vertices: np.ndarray, faces: np.ndarray, output_path: Path) -> Path:
    """Export the mesh as a binary GLB file via trimesh.

    vertices: [6890, 3] float32
    faces:   [13776, 3] int64
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(str(output_path))
    logger.info("Exported GLB: {}", output_path)
    return output_path


def render_thumbnail(
    vertices: np.ndarray,   # [6890, 3] float32 — A-pose mesh
    faces: np.ndarray,      # [13776, 3] int64
    output_path: Path,
    device: str,
) -> Path:
    """Render a 512×512 shaded PNG of the mesh from a front canonical view.

    Uses the same PyTorch3D HardPhongShader + DirectionalLights stack as the
    D13 turntable (stage2.save_geometry_turntable). Camera is placed on +Z
    looking at origin — the mesh must already be in canonical coordinates
    with the subject facing +Z (which the A-pose is by construction).

    No R_180x flip — this is a standalone render, not passing through the
    Stage 2 per-frame camera path.
    """
    Ht = Wt = 512
    vertices_t = torch.from_numpy(vertices).to(device).float()
    faces_t = torch.from_numpy(faces).to(device)

    # Front view: camera on +Z axis, looking at origin.
    R, T = look_at_view_transform(
        dist=3.0, elev=0.0, azim=[0.0],
        up=((0.0, 1.0, 0.0),),
        device=device,
    )
    cameras = FoVPerspectiveCameras(R=R, T=T, fov=40.0, device=device)

    lights = DirectionalLights(
        direction=torch.tensor([[0.0, -0.3, -1.0]], device=device),
        ambient_color=torch.tensor([[0.4, 0.4, 0.4]], device=device),
        diffuse_color=torch.tensor([[0.6, 0.6, 0.6]], device=device),
        specular_color=torch.tensor([[0.0, 0.0, 0.0]], device=device),
        device=device,
    )

    raster_settings = RasterizationSettings(
        image_size=(Ht, Wt),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights),
    )

    verts_rgb = torch.ones(1, vertices_t.shape[0], 3, device=device) * 0.75
    tex = TexturesVertex(verts_features=verts_rgb)
    mesh = Meshes(verts=[vertices_t], faces=[faces_t], textures=tex)

    blend = BlendParams(background_color=(1.0, 1.0, 1.0))  # white
    with torch.no_grad():
        rgba = renderer(mesh, blend_params=blend)  # [1, Ht, Wt, 4]
    rgb = rgba[0, ..., :3]
    img_u8 = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    # PyTorch3D renders RGB; cv2.imwrite expects BGR.
    cv2.imwrite(str(output_path), cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))
    logger.info("Thumbnail saved: {}", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Analysis branch
# ---------------------------------------------------------------------------

def compute_theta_natural(theta_per_keyframe: np.ndarray) -> np.ndarray:
    """Medoid-frame selection on body_pose to get a representative natural pose.

    theta_per_keyframe: [N, 24, 3] float32 — axis-angle, joint 0 = global_orient

    Computes pairwise per-joint Frobenius distance on rotation matrices
    (body_pose only, joints 1-23). Picks the frame that minimises the sum
    of squared distances to all other frames — the most "central" pose
    actually observed. global_orient is set to identity so the natural pose
    faces front in canonical coordinates.

    Returns [24, 3] float32 axis-angle.
    """
    N = theta_per_keyframe.shape[0]
    if N == 1:
        result = theta_per_keyframe[0].copy()
        result[0] = 0.0  # identity global_orient
        return result

    body_pose_aa = theta_per_keyframe[:, 1:, :]  # [N, 23, 3]
    device = settings.device
    body_t = torch.from_numpy(body_pose_aa.astype(np.float32)).to(device)

    # [N, 23, 3] axis-angle → [N, 23, 3, 3] rotation matrices
    R = axis_angle_to_matrix(body_t.reshape(N * 23, 3)).reshape(N, 23, 3, 3)

    # Pairwise squared Frobenius distance per frame pair, summed over joints.
    # For frames i, j: d(i,j) = Σ_k ||R_i_k - R_j_k||_F².
    # Expand for broadcasting: R_i [N, 1, 23, 3, 3], R_j [1, N, 23, 3, 3].
    R_i = R.unsqueeze(1)   # [N, 1, 23, 3, 3]
    R_j = R.unsqueeze(0)   # [1, N, 23, 3, 3]
    diff = R_i - R_j                            # [N, N, 23, 3, 3]
    sq_dist = (diff ** 2).sum(dim=(2, 3, 4))    # [N, N] — sum over joints + matrix elements

    medoid_idx = int(sq_dist.sum(dim=1).argmin().cpu().item())

    result = theta_per_keyframe[medoid_idx].copy()
    result[0] = 0.0  # identity global_orient
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def compute_quality(
    stage1_result: Stage1Result,
    stage2_result: Stage2Result,
    vertices_a_pose: np.ndarray,  # [6890, 3] float32
) -> QualityReport:
    """Five checks producing an overall score and warning list for Layer 2.

    Each check contributes one warning at most. Score starts at 1.0 and
    deducts 0.06 per warning; floor at 0.0.
    """
    warnings: list[str] = []

    _check_mask_coverage(stage1_result, warnings)
    _check_beta_plausibility(stage2_result, warnings)
    _check_tz_spread(stage1_result, warnings)
    _check_angular_coverage(stage1_result, warnings)
    _check_height_match(stage1_result, vertices_a_pose, warnings)

    score = max(0.0, 1.0 - 0.06 * len(warnings))
    return QualityReport(overall_score=score, warnings=warnings)


# --- Individual checks ---

def _check_mask_coverage(s1: Stage1Result, warnings: list[str]) -> None:
    """Check 1: mean SAM mask coverage across keyframes.

    Reads: s1.mask_paths (list of Path to mask PNGs, uint8 grayscale).

    Thresholds (0.05 mean, 0.02 per-frame) relate to the M3 failure-detection
    threshold (9% = 0.09, NOTES 2026-05-20) as follows: M3's 9% gate was an
    active re-processing trigger — frames below 9% were re-run with keypoint
    prompts (M5). After M5 refinement, all frames should score well above 9%
    (verified 11-15% on our test bundle). These Stage 3 thresholds are
    deliberately looser post-hoc quality signals: they flag catastrophically
    bad masks that somehow survived M3/M5 gates, not routine variation.
    """
    coverages: list[float] = []
    for mp in s1.mask_paths:
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            warnings.append(f"mask unreadable: {mp.name}")
            return
        fg = (m > 127).sum()
        coverages.append(float(fg) / m.size)

    min_cov = min(coverages)
    mean_cov = sum(coverages) / len(coverages)

    if mean_cov < 0.05:
        warnings.append(
            f"mask coverage mean={mean_cov:.3f} below 0.05 "
            f"(min={min_cov:.3f}, n={len(coverages)})"
        )
    elif min_cov < 0.02:
        warnings.append(
            f"mask coverage single-frame min={min_cov:.3f} below 0.02 "
            f"(mean={mean_cov:.3f})"
        )


def _check_beta_plausibility(s2: Stage2Result, warnings: list[str]) -> None:
    """Check 2: SMPL β components within typical range.

    Reads: s2.beta [10] float32.

    SMPL shape space is standardised (zero-mean, unit-variance per component
    across the training population). Values beyond ±3.5 are outliers — either
    a genuinely unusual body shape or (more likely) a corrupted inference.
    """
    beta = s2.beta
    for k in range(10):
        if abs(beta[k]) > 3.5:
            warnings.append(
                f"beta[{k}]={float(beta[k]):.3f} outside [-3.5, 3.5]"
            )
            return  # one warning covers all outlier components


def _check_tz_spread(s1: Stage1Result, warnings: list[str]) -> None:
    """Check 3: per-frame camera tz consistency.

    Reads: s1.pred_cam_t_per_frame [N, 3] float32, column 2 = tz.

    Fixed-camera physical prior: the subject spins in place, so tz (camera-
    to-subject distance) should be constant across frames. hmr2's weak-
    perspective estimation introduces jitter; excessive spread suggests
    unstable pose estimation.
    """
    tz = s1.pred_cam_t_per_frame[:, 2].astype(np.float64)
    tz_median = float(np.median(tz))
    if tz_median <= 0:
        warnings.append(f"tz median={tz_median:.2f} m — non-positive")
        return
    spread = float(tz.max() - tz.min())
    ratio = spread / tz_median
    if ratio > 0.15:
        warnings.append(
            f"tz spread={spread:.2f} m / median={tz_median:.2f} m "
            f"= {ratio:.2%} — exceeds 15%"
        )


def _check_angular_coverage(s1: Stage1Result, warnings: list[str]) -> None:
    """Check 4: keyframe angular coverage of the spin.

    Reads: s1.theta_per_frame [N, 24, 3], joint 0 = global_orient.

    Azimuths derived from global_orient via Rodrigues (same method as
    stage2._frame_orientation_weights). Maximum inter-keyframe gap > 90°
    means the keyframes missed at least a quadrant of the full spin.
    """
    N = s1.n_frames
    go = s1.theta_per_frame[:, 0, :]  # [N, 3] global_orient axis-angle
    azimuths = np.empty(N, dtype=np.float64)
    for i, g in enumerate(go):
        angle = np.linalg.norm(g)
        if angle < 1e-6:
            azimuths[i] = 0.0
            continue
        ax = g / angle
        c, s = np.cos(angle), np.sin(angle)
        fwd_x = s * ax[1] + (1.0 - c) * ax[0] * ax[2]
        fwd_z = c       + (1.0 - c) * ax[2] * ax[2]
        azimuths[i] = np.arctan2(fwd_x, fwd_z)

    sorted_az = np.sort(azimuths)
    gaps = np.diff(sorted_az)
    # Wrap-around gap: from last back to first + 2π
    wrap_gap = float(sorted_az[0] + 2.0 * np.pi - sorted_az[-1])
    max_gap_rad = max(float(gaps.max()), wrap_gap)
    max_gap_deg = float(np.degrees(max_gap_rad))

    if max_gap_deg > 90.0:
        warnings.append(
            f"keyframe angular gap max={max_gap_deg:.0f}° — exceeds 90°"
        )


def _check_height_match(
    s1: Stage1Result,
    vertices_a_pose: np.ndarray,  # [6890, 3] float32
    warnings: list[str],
) -> None:
    """Check 5: A-pose mesh height vs user-supplied height_cm.

    Reads: s1.height_cm (float), vertices_a_pose (A-pose mesh).

    SMPL vertex coordinates are in metres (scale_to_meters=1.0, verified by
    Y-span measurement on the M9 bundle: 1.7687 m for a 1.80 m subject).
    This check compares the mesh Y-span directly to height_cm/100 — it is
    NOT circular with scale_to_meters because scale_to_meters is the fixed
    SMPL unit convention, not derived from the measurement.
    """
    mesh_height_m = float(vertices_a_pose[:, 1].ptp())
    target_m = s1.height_cm / 100.0
    diff_m = abs(mesh_height_m - target_m)
    if diff_m > 0.10:
        warnings.append(
            f"mesh height={mesh_height_m:.3f} m vs target={target_m:.3f} m "
            f"(diff={diff_m:.3f} m, {diff_m / target_m * 100:.1f}%)"
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run(stage1_result: Stage1Result, stage2_result: Stage2Result) -> Stage3Result:
    """Stage 3 dual output: display GLB + complete Stage3Result for Layer 2.

    Display branch: A-pose mesh → GLB export + thumbnail render.
    Analysis branch: canonical mesh, theta_natural, quality checks.

    Persists Stage3Result to <run_dir>/stage3_result.npz and runs the
    round-trip self-check. Mirrors stage1.run / stage2.run fail-loud pattern.
    """
    device = settings.device

    smpl_model = _load_smpl_model(device)

    beta_t = torch.from_numpy(stage2_result.beta).to(device).float()
    delta_v_t = torch.from_numpy(stage2_result.delta_v).to(device).float()

    # Canonical mesh (shared by both branches)
    v_canon, joints_canon, faces = _compute_canonical(beta_t, delta_v_t, smpl_model)
    v_canon_t = torch.from_numpy(v_canon).to(device).float()

    # --- Display branch ---
    a_pose = _build_a_pose()
    a_pose_t = torch.from_numpy(a_pose).to(device).float()
    verts_a, _ = _pose_to(a_pose_t, v_canon_t, smpl_model)

    glb_path = stage1_result.run_dir / "body_mesh.glb"
    export_glb(verts_a, faces, glb_path)

    thumb_path = stage1_result.run_dir / "thumbnail.png"
    render_thumbnail(verts_a, faces, thumb_path, device)

    # --- Analysis branch ---
    theta_nat = compute_theta_natural(stage1_result.theta_per_frame)
    theta_nat_t = torch.from_numpy(theta_nat).to(device).float()
    _, joints_nat = _pose_to(theta_nat_t, v_canon_t, smpl_model)

    # --- Quality ---
    quality = compute_quality(stage1_result, stage2_result, verts_a)

    # --- Assemble ---
    result = Stage3Result(
        run_id=stage1_result.run_id,
        run_dir=stage1_result.run_dir,
        vertices_a_pose=verts_a,
        glb_path=glb_path,
        thumbnail_path=thumb_path,
        vertices_canonical=v_canon,
        delta_v=stage2_result.delta_v.astype(np.float32),
        beta=stage2_result.beta.astype(np.float32),
        theta_natural=theta_nat,
        theta_per_keyframe=stage1_result.theta_per_frame.astype(np.float32),
        joints_canonical=joints_canon,
        joints_natural=joints_nat,
        scale_to_meters=1.0,
        quality=quality,
    )

    npz_path = stage1_result.run_dir / "stage3_result.npz"
    save_stage3_result(result, npz_path)
    logger.info("Wrote {}", npz_path)
    _verify_stage3_round_trip(result, npz_path)

    logger.info(
        "Stage 3 complete: glb={}, thumbnail={}, quality_score={:.2f}, "
        "n_warnings={}, run_id={}",
        glb_path.name, thumb_path.name,
        quality.overall_score, len(quality.warnings),
        result.run_id,
    )
    return result
