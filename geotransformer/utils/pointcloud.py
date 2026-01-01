from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

try:
    from pytorch3d.ops import sample_farthest_points as fps

    _HAS_P3D = True
except Exception:
    _HAS_P3D = False

from geotransformer.utils.common import best_torch_device

# Basic Utilities


def _as_tensor(
    x: Union[np.ndarray, torch.Tensor], device: Optional[torch.device] = None, dtype=torch.float32
) -> torch.Tensor:
    """Convert numpy->torch (contiguous) or move tensor to device/dtype."""
    if torch.is_tensor(x):
        t = x.to(dtype=dtype)
    else:
        t = torch.from_numpy(np.ascontiguousarray(x)).float().to(dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def get_nearest_neighbor(
    q_points: Union[np.ndarray, torch.Tensor],
    s_points: Union[np.ndarray, torch.Tensor],
    return_index: bool = False,
    device: Optional[torch.device] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Torch-first nearest neighbor. Returns torch.Tensor(s) on `device` (inferred from inputs if not provided).
    Uses torch.cdist when possible; falls back to cKDTree CPU on OOM.
    """
    if device is None:
        if torch.is_tensor(q_points):
            device = q_points.device
        elif torch.is_tensor(s_points):
            device = s_points.device
        else:
            device = torch.device("cpu")

    q_t = _as_tensor(q_points, device=device)
    s_t = _as_tensor(s_points, device=device)

    try:
        d = torch.cdist(q_t, s_t)  # (Q,S)
        min_dists, min_idx = torch.min(d, dim=1)
        if return_index:
            return min_dists, min_idx
        return min_dists
    except RuntimeError:
        # fallback: CPU KDTree
        q_np = q_t.detach().cpu().numpy()
        s_np = s_t.detach().cpu().numpy()
        tree = cKDTree(s_np)
        distances, indices = tree.query(q_np, k=1, workers=4)
        distances_t = torch.from_numpy(distances.astype(np.float32)).float()
        indices_t = torch.from_numpy(indices.astype(np.int64)).float()
        if return_index:
            return distances_t, indices_t
        return distances_t


def regularize_normals(
    points: Union[np.ndarray, torch.Tensor],
    normals: Union[np.ndarray, torch.Tensor],
    positive: bool = True,
) -> torch.Tensor:
    """
    Regularize normals to face toward (positive=True) or away from origin (positive=False).
    Always returns a torch.Tensor on the device of `points` if `points` is a tensor, else cpu tensor.
    """
    device = points.device if torch.is_tensor(points) else torch.device("cpu")
    pts = _as_tensor(points, device=device)
    norms = _as_tensor(normals, device=device)
    dot_products = -(pts * norms).sum(dim=1, keepdim=True)  # (N,1)
    direction = dot_products > 0  # bool
    if positive:
        norms_out = norms * direction.to(dtype=norms.dtype) - norms * (~direction).to(
            dtype=norms.dtype
        )
    else:
        norms_out = norms * (~direction).to(dtype=norms.dtype) - norms * direction.to(
            dtype=norms.dtype
        )
    return norms_out


# Transformation Utilities


def apply_transform(
    points: Union[np.ndarray, torch.Tensor],
    transform: Union[np.ndarray, torch.Tensor],
    normals: Optional[Union[np.ndarray, torch.Tensor]] = None,
    device: Optional[torch.device] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Apply 4x4 rigid transform to row-vector points (N,3).
    Returns torch.Tensor(s) on `device` (inferred from `points` or given).
    """
    is_tensor = torch.is_tensor(points)
    if device is None:
        device = points.device if is_tensor else torch.device("cpu")

    pts = _as_tensor(points, device=device)
    if not torch.is_tensor(transform):
        T = _as_tensor(transform, device=device)
    else:
        T = transform.to(device=device, dtype=torch.float32)

    R = T[:3, :3]
    t = T[:3, 3]
    pts_out = pts @ R.T + t.unsqueeze(0)
    if normals is not None:
        norms = _as_tensor(normals, device=device)
        norms_out = norms @ R.T
        return pts_out, norms_out
    return pts_out


def compose_transforms(
    transforms: List[Union[np.ndarray, torch.Tensor]], device: Optional[torch.device] = None
) -> torch.Tensor:
    """Compose T = T_n @ ... @ T_0. Returns torch.Tensor on `device` (inferred if not provided)."""
    assert len(transforms) > 0
    first_tensor = next((t for t in transforms if torch.is_tensor(t)), None)
    if device is None:
        device = first_tensor.device if first_tensor is not None else torch.device("cpu")

    T = None
    for tr in transforms:
        tr_t = _as_tensor(tr, device=device)
        if T is None:
            T = tr_t.clone()
        else:
            T = tr_t @ T
    return T


def get_transform_from_rotation_translation(
    rotation: Union[np.ndarray, torch.Tensor],
    translation: Union[np.ndarray, torch.Tensor],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build 4x4 transform from rotation (3x3) and translation (3,).
    Returns torch.Tensor on `device`.
    """
    if device is None:
        if torch.is_tensor(rotation):
            device = rotation.device
        elif torch.is_tensor(translation):
            device = translation.device
        else:
            device = torch.device("cpu")

    R = _as_tensor(rotation, device=device)
    t = _as_tensor(translation, device=device)
    T = torch.eye(4, dtype=R.dtype, device=device)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def get_rotation_translation_from_transform(
    transform: Union[np.ndarray, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (rotation (3,3), translation (3,)) as torch.Tensors on same device as input."""
    if not torch.is_tensor(transform):
        T = _as_tensor(transform, device=None)
    else:
        T = transform
    return T[:3, :3], T[:3, 3]


def inverse_transform(
    transform: Union[np.ndarray, torch.Tensor], device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Inverse 4x4 rigid transform (rotation + translation only).
        WARNING: This function assumes R is orthonormal (pure rotation).
        For affine transforms with scaling/shear, use torch.linalg.inv() instead.
    Returns torch.Tensor on `device` (inferred from input).
    """
    is_t = torch.is_tensor(transform)
    if device is None:
        device = transform.device if is_t else torch.device("cpu")

    T = _as_tensor(transform, device=device)
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    return get_transform_from_rotation_translation(R_inv, t_inv, device=device)


# Rotation samplers


def random_sample_rotation(rotation_factor: float = 1.0) -> np.ndarray:
    """Non-uniform random rotation using Euler angles (numpy)."""
    euler = np.random.rand(3) * np.pi * 2 / rotation_factor
    return Rotation.from_euler("zyx", euler).as_matrix()


def random_sample_rotation_fast(
    device: Optional[torch.device] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """Uniform random rotation (Marsaglia). Returns torch.Tensor if device specified."""
    if device is not None:
        q = sample_uniform_quaternion_torch(n=1, device=device)
        R = quat_to_mat_torch(q)  # (1,3,3)
        return R[0]
    # numpy fallback
    u1, u2, u3 = np.random.rand(3)
    sqrt_1_u1 = np.sqrt(1.0 - u1)
    sqrt_u1 = np.sqrt(u1)
    two_pi_u2 = 2.0 * np.pi * u2
    two_pi_u3 = 2.0 * np.pi * u3
    w = sqrt_u1 * np.cos(two_pi_u3)
    x = sqrt_1_u1 * np.sin(two_pi_u2)
    y = sqrt_1_u1 * np.cos(two_pi_u2)
    z = sqrt_u1 * np.sin(two_pi_u3)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    return R


def random_sample_rotation_v2(max_angle: Optional[float] = None) -> np.ndarray:
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis) + 1e-12
    theta = (
        np.random.uniform(0, 2 * np.pi) if max_angle is None else np.random.uniform(0, max_angle)
    )
    return Rotation.from_rotvec(axis * theta).as_matrix()


def random_sample_rotation_v3(
    max_angle: Optional[float] = None,
    device: Optional[torch.device] = None,
    output_type: str = "auto",
) -> Union[np.ndarray, torch.Tensor]:
    if max_angle is None:
        dev = device if device is not None else torch.device("cpu")
        q = sample_uniform_quaternion_torch(n=1, device=dev)
        R = quat_to_mat_torch(q)
        if output_type == "torch" or device is not None:
            return R
        return R.cpu().numpy()
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis) + 1e-8
    angle = np.random.uniform(0, max_angle)
    half_angle = angle * 0.5
    sin_half, cos_half = np.sin(half_angle), np.cos(half_angle)
    q_np = np.array([cos_half, *(axis * sin_half)], dtype=np.float32)
    w, x, y, z = q_np
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R_np = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    if output_type == "torch" or device is not None:
        dev = device if device is not None else torch.device("cpu")
        return torch.from_numpy(R_np).float().to(dev)
    return R_np


def random_sample_transform(
    rotation_magnitude: float, translation_magnitude: float
) -> torch.Tensor:
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis) + 1e-12
    angle = np.random.uniform(0, np.deg2rad(rotation_magnitude))
    rotation = Rotation.from_rotvec(axis * angle).as_matrix()
    translation = np.random.uniform(-translation_magnitude, translation_magnitude, 3)
    return get_transform_from_rotation_translation(rotation, translation)


def random_sample_rotation_uniform(
    n: int = 1, device: Optional[torch.device] = None
) -> Union[np.ndarray, torch.Tensor]:
    if device is None:
        R = [random_sample_rotation_fast() for _ in range(n)]
        return np.stack(R) if n > 1 else R[0]
    q = sample_uniform_quaternion_torch(n, device=device)
    R = quat_to_mat_torch(q)
    return R if n > 1 else R[0]


# Helper samplers


def sample_uniform_quaternion_torch(
    n: int = 1, device: Union[str, torch.device] = torch.device("cpu")
) -> torch.Tensor:
    u = torch.rand(n, 3, device=device)
    u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
    q = torch.empty((n, 4), device=device, dtype=torch.float32)
    sqrt_1_u1, sqrt_u1 = torch.sqrt(1 - u1), torch.sqrt(u1)
    two_pi_u2, two_pi_u3 = 2 * np.pi * u2, 2 * np.pi * u3
    q[:, 0] = sqrt_1_u1 * torch.sin(two_pi_u2)
    q[:, 1] = sqrt_1_u1 * torch.cos(two_pi_u2)
    q[:, 2] = sqrt_u1 * torch.sin(two_pi_u3)
    q[:, 3] = sqrt_u1 * torch.cos(two_pi_u3)
    q = q[:, [3, 0, 1, 2]]  # [w,x,y,z]
    return q  # always return (n,4)


def quat_to_mat_torch(q: torch.Tensor) -> torch.Tensor:
    # Ensure q has shape (N,4)
    if q.ndim == 1:
        q = q.unsqueeze(0)  # (1,4)

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    Nq = w * w + x * x + y * y + z * z
    s = 2.0 / (Nq + 1e-12)
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    m00 = 1 - (yy + zz)
    m01 = xy - wz
    m02 = xz + wy
    m10 = xy + wz
    m11 = 1 - (xx + zz)
    m12 = yz - wx
    m20 = xz - wy
    m21 = yz + wx
    m22 = 1 - (xx + yy)

    mat = torch.stack(
        [
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1),
        ],
        dim=-2,
    )  # shape (N,3,3)

    if mat.shape[0] == 1:
        return mat[0]  # return (3,3) for single quaternion
    return mat  # (N,3,3) for batch


# Sampling methods
# Note: these are kept numpy signatures for compatibility, but can be easily changed to torch if desired.


def random_sample_keypoints(points: np.ndarray, feats: np.ndarray, num_keypoints: int):
    num_points = points.shape[0]
    if num_points > num_keypoints:
        indices = np.random.choice(num_points, num_keypoints, replace=False)
        return points[indices], feats[indices]
    return points, feats


def sample_keypoints_with_scores(
    points: np.ndarray, feats: np.ndarray, scores: np.ndarray, num_keypoints: int
):
    num_points = points.shape[0]
    if num_points > num_keypoints:
        indices = np.argsort(-scores)[:num_keypoints]
        return points[indices], feats[indices]
    return points, feats


def random_sample_keypoints_with_scores(
    points: np.ndarray, feats: np.ndarray, scores: np.ndarray, num_keypoints: int
):
    num_points = points.shape[0]
    if num_points > num_keypoints:
        indices = np.arange(num_points)
        probs = scores / np.sum(scores)
        indices = np.random.choice(indices, num_keypoints, replace=False, p=probs)
        return points[indices], feats[indices]
    return points, feats


def sample_keypoints_with_nms(
    points: np.ndarray, feats: np.ndarray, scores: np.ndarray, num_keypoints: int, radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    num_points = points.shape[0]
    if num_points > num_keypoints:
        radius2 = radius ** 2
        masks = np.ones(num_points, dtype=np.bool)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_points = points[sorted_indices]
        sorted_feats = feats[sorted_indices]
        indices = []
        for i in range(num_points):
            if masks[i]:
                indices.append(i)
                if len(indices) == num_keypoints:
                    break
                if i + 1 < num_points:
                    current_masks = (
                        np.sum((sorted_points[i + 1 :] - sorted_points[i]) ** 2, axis=1) < radius2
                    )
                    masks[i + 1 :] = masks[i + 1 :] & ~current_masks
        points = sorted_points[indices]
        feats = sorted_feats[indices]
    return points, feats


def random_sample_keypoints_with_nms(
    points: np.ndarray, feats: np.ndarray, scores: np.ndarray, num_keypoints: int, radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    num_points = points.shape[0]
    if num_points > num_keypoints:
        radius2 = radius ** 2
        masks = np.ones(num_points, dtype=np.bool)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_points = points[sorted_indices]
        sorted_feats = feats[sorted_indices]
        indices = []
        for i in range(num_points):
            if masks[i]:
                indices.append(i)
                if i + 1 < num_points:
                    current_masks = (
                        np.sum((sorted_points[i + 1 :] - sorted_points[i]) ** 2, axis=1) < radius2
                    )
                    masks[i + 1 :] = masks[i + 1 :] & ~current_masks
        indices = np.array(indices)
        if len(indices) > num_keypoints:
            sorted_scores = scores[sorted_indices]
            scores = sorted_scores[indices]
            probs = scores / np.sum(scores)
            indices = np.random.choice(indices, num_keypoints, replace=False, p=probs)
        points = sorted_points[indices]
        feats = sorted_feats[indices]
    return points, feats


# depth image utilities


def convert_depth_mat_to_points(
    depth_mat: np.ndarray,
    intrinsics: np.ndarray,
    scaling_factor: float = 1000.0,
    distance_limit: float = 6.0,
):
    r"""Convert depth image to point cloud.

    Args:
        depth_mat (array): (H, W)
        intrinsics (array): (3, 3)
        scaling_factor (float=1000.)

    Returns:
        points (array): (N, 3)
    """
    focal_x = intrinsics[0, 0]
    focal_y = intrinsics[1, 1]
    center_x = intrinsics[0, 2]
    center_y = intrinsics[1, 2]
    height, width = depth_mat.shape
    coords = np.arange(height * width)
    u = coords % width
    v = coords // width
    depth = depth_mat.flatten()
    z = depth / scaling_factor
    z[z > distance_limit] = 0.0
    x = (u - center_x) * z / focal_x
    y = (v - center_y) * z / focal_y
    points = np.stack([x, y, z], axis=1)
    points = points[depth > 0]
    return points


# number of points utilities

def ensure_padding_points(
    points: Union[np.ndarray, torch.Tensor],
    normals: Optional[Union[np.ndarray, torch.Tensor]],
    feats: Optional[Union[np.ndarray, torch.Tensor]],
    num_points: int,
    pad_jitter_std: float = 0.0,
    deterministic_seed: Optional[int] = None,
    force_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Ensures exactly >= `num_points` points using:
      - padding (optionally jittered) if N < num_points
    Keeps normals aligned with the chosen indices.

    Args:
        points: (N,3) float32
        normals: (N,3) float32 or None
        feats: (N,F) float32 or None
        num_points: target count
        pad_jitter_std: stddev for Gaussian jitter when padding (0.0 = no jitter)
        deterministic_seed: if not None, used for fallback random and padding picks
        force_device: optional torch.device override (else auto-select)

    Returns:
        points2: (num_points,3) float32
        normals2: (num_points,3) float32 or None
        feats2: (num_points,F) float32 or None
    """
    assert (
        (isinstance(points, np.ndarray) or torch.is_tensor(points))
        and points.ndim == 2
        and points.shape[1] == 3
    )
    dev = (
        force_device
        if force_device is not None
        else (points.device if torch.is_tensor(points) else torch.device("cpu"))
    )
    if not torch.is_tensor(points):
        pts = torch.from_numpy(np.ascontiguousarray(points)).to(device=dev, dtype=torch.float32)
    else:
        pts = points.to(device=dev, dtype=torch.float32)

    norms = None
    feats_t = None
    if normals is not None:
        norms = (
            normals.to(device=dev, dtype=torch.float32)
            if torch.is_tensor(normals)
            else torch.from_numpy(np.ascontiguousarray(normals)).to(device=dev, dtype=torch.float32)
        )
    if feats is not None:
        feats_t = (
            feats.to(device=dev, dtype=torch.float32)
            if torch.is_tensor(feats)
            else torch.from_numpy(np.ascontiguousarray(feats)).to(device=dev, dtype=torch.float32)
        )

    N = pts.shape[0]
    if N >= num_points:
        return pts, norms, feats_t

    need = num_points - N
    if N == 0:
        pts2 = torch.zeros((num_points, 3), dtype=pts.dtype, device=dev)
        norms2 = torch.zeros_like(pts2) if normals is not None else None
        feats2 = (
            torch.ones((num_points, feats_t.shape[1]), dtype=feats_t.dtype, device=dev)
            if feats_t is not None
            else None
        )
        return pts2, norms2, feats2

    if deterministic_seed is not None:
        torch.manual_seed(deterministic_seed)
    rep_idx = torch.randint(0, N, (need,), device=dev)
    pts2 = torch.cat([pts, pts[rep_idx]], dim=0)
    if pad_jitter_std > 0.0:
        jitter = torch.normal(
            mean=0.0, std=pad_jitter_std, size=(need, 3), device=dev, dtype=pts2.dtype
        )
        pts2[-need:] += jitter
    norms2 = torch.cat([norms, norms[rep_idx]], dim=0) if norms is not None else None
    feats2 = torch.cat([feats_t, feats_t[rep_idx]], dim=0) if feats_t is not None else None
    return pts2, norms2, feats2


def ensure_num_points(
    points: Union[np.ndarray, torch.Tensor],
    normals: Optional[Union[np.ndarray, torch.Tensor]],
    feats: Optional[Union[np.ndarray, torch.Tensor]],
    num_points: int,
    pad_jitter_std: float = 0.0,
    deterministic_seed: Optional[int] = None,
    force_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    assert (
        (isinstance(points, np.ndarray) or torch.is_tensor(points))
        and points.ndim == 2
        and points.shape[1] == 3
    )
    dev = (
        force_device
        if force_device is not None
        else (points.device if torch.is_tensor(points) else best_torch_device())
    )
    if not torch.is_tensor(points):
        pts = torch.from_numpy(np.ascontiguousarray(points)).to(device=dev, dtype=torch.float32)
    else:
        pts = points.to(device=dev, dtype=torch.float32)

    norms = None
    feats_t = None
    if normals is not None:
        norms = (
            normals.to(device=dev, dtype=torch.float32)
            if torch.is_tensor(normals)
            else torch.from_numpy(np.ascontiguousarray(normals)).to(device=dev, dtype=torch.float32)
        )
    if feats is not None:
        feats_t = (
            feats.to(device=dev, dtype=torch.float32)
            if torch.is_tensor(feats)
            else torch.from_numpy(np.ascontiguousarray(feats)).to(device=dev, dtype=torch.float32)
        )

    N = pts.shape[0]
    if N == num_points:
        return pts, norms, feats_t

    if N > num_points:
        if _HAS_P3D:
            with torch.no_grad():
                P = pts.unsqueeze(0)
                _, idx = fps(P, K=num_points)
                idx = idx[0]
            selected_idx = idx.to(device=dev)
        else:
            if deterministic_seed is not None:
                torch.manual_seed(deterministic_seed)
            perm = torch.randperm(N, device=dev)
            selected_idx = perm[:num_points]
        pts2 = pts[selected_idx]
        norms2 = norms[selected_idx] if norms is not None else None
        feats2 = feats_t[selected_idx] if feats_t is not None else None
        return pts2, norms2, feats2

    return ensure_padding_points(
        pts, norms, feats_t, num_points, pad_jitter_std, deterministic_seed, force_device=dev
    )
