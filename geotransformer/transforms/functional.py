import math
from typing import Optional, Union

import numpy as np
import torch


def _to_torch(
    x: Union[np.ndarray, torch.Tensor], device: Optional[torch.device] = None, dtype=torch.float32
):
    if torch.is_tensor(x):
        t = x.to(dtype=dtype)
    else:
        t = torch.from_numpy(np.ascontiguousarray(x)).float().to(dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def _maybe_numpy(x: torch.Tensor, output_numpy: bool):
    return x.cpu().numpy() if output_numpy else x


# 1) Normalize to unit sphere
def normalize_points(
    points: Union[np.ndarray, torch.Tensor],
    device: Optional[torch.device] = None,
    output_numpy: bool = False,
):
    pts = _to_torch(points, device=device)
    mean = pts.mean(dim=0, keepdim=True)
    pts = pts - mean
    max_norm = torch.linalg.norm(pts, dim=1).max()
    max_norm = max_norm if max_norm > 0 else torch.tensor(1.0, device=pts.device, dtype=pts.dtype)
    pts = pts / max_norm
    return _maybe_numpy(pts, output_numpy)


# 2) deterministic sample first K
def sample_points(
    points: Union[np.ndarray, torch.Tensor],
    num_samples: int,
    normals: Optional[Union[np.ndarray, torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    output_numpy: bool = False,
):
    pts = _to_torch(points, device=device)
    pts = pts[:num_samples]
    if normals is not None:
        norms = _to_torch(normals, device=device)[:num_samples]
        return (_maybe_numpy(pts, output_numpy), _maybe_numpy(norms, output_numpy))
    return _maybe_numpy(pts, output_numpy)


# 3) random sample (with repeat/pad) - uses torch RNG
def random_sample_points(
    points: Union[np.ndarray, torch.Tensor],
    num_samples: int,
    normals: Optional[Union[np.ndarray, torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    output_numpy: bool = False,
):
    pts = _to_torch(points, device=device)
    N = pts.shape[0]
    if seed is not None:
        torch.manual_seed(seed)
    perm = torch.randperm(N, device=pts.device)
    if N >= num_samples:
        sel = perm[:num_samples]
    else:
        reps = num_samples // N
        rem = num_samples % N
        sel = perm.repeat(reps)
        if rem > 0:
            sel = torch.cat([sel, perm[:rem]], dim=0)
    pts_out = pts[sel]
    if normals is not None:
        norms = _to_torch(normals, device=device)[sel]
        return _maybe_numpy(pts_out, output_numpy), _maybe_numpy(norms, output_numpy)
    return _maybe_numpy(pts_out, output_numpy)


# 4) scale + shift
def random_scale_shift_points(
    points: Union[np.ndarray, torch.Tensor],
    low=2.0 / 3.0,
    high=3.0 / 2.0,
    shift=0.2,
    normals: Optional[Union[np.ndarray, torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    output_numpy: bool = False,
):
    pts = _to_torch(points, device=device)
    if seed is not None:
        torch.manual_seed(seed)
    scale = torch.empty((1, 3), device=pts.device).uniform_(low, high)
    bias = torch.empty((1, 3), device=pts.device).uniform_(-shift, shift)
    pts_out = pts * scale + bias
    if normals is not None:
        norms = _to_torch(normals, device=device) * scale
        norms = torch.nn.functional.normalize(norms, dim=1)
        return _maybe_numpy(pts_out, output_numpy), _maybe_numpy(norms, output_numpy)
    return _maybe_numpy(pts_out, output_numpy)


# 5) rotate along up(z) axis
def random_rotate_points_along_up_axis(
    points: Union[np.ndarray, torch.Tensor],
    normals: Optional[Union[np.ndarray, torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    output_numpy: bool = False,
):
    pts = _to_torch(points, device=device)
    if seed is not None:
        torch.manual_seed(seed)
    theta = (2.0 * math.pi) * torch.rand((), device=pts.device).item()
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    R = torch.tensor(
        [[cos_t, sin_t, 0.0], [-sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]],
        device=pts.device,
        dtype=pts.dtype,
    )
    pts_out = pts @ R.T
    if normals is not None:
        norms = _to_torch(normals, device=device) @ R.T
        return _maybe_numpy(pts_out, output_numpy), _maybe_numpy(norms, output_numpy)
    return _maybe_numpy(pts_out, output_numpy)


# 6) random rescale (scalar)
def random_rescale_points(
    points: Union[np.ndarray, torch.Tensor],
    low=0.8,
    high=1.2,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    output_numpy: bool = False,
):
    pts = _to_torch(points, device=device)
    if seed is not None:
        torch.manual_seed(seed)
    s = torch.empty((), device=pts.device).uniform_(low, high)
    pts_out = pts * s
    return _maybe_numpy(pts_out, output_numpy)


# 7) jitter points
def random_jitter_points(
    points: Union[np.ndarray, torch.Tensor],
    scale: float,
    noise_magnitude: float = 0.05,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    output_numpy: bool = False,
):
    pts = _to_torch(points, device=device)
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn_like(pts) * scale
    noise = torch.clamp(noise, -noise_magnitude, noise_magnitude)
    pts_out = pts + noise
    return _maybe_numpy(pts_out, output_numpy)


# 8) shuffle points
def random_shuffle_points(
    points: Union[np.ndarray, torch.Tensor],
    normals: Optional[Union[np.ndarray, torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    output_numpy: bool = False,
):
    pts = _to_torch(points, device=device)
    N = pts.shape[0]
    if seed is not None:
        torch.manual_seed(seed)
    idx = torch.randperm(N, device=pts.device)
    pts_out = pts[idx]
    if normals is not None:
        norms = _to_torch(normals, device=device)[idx]
        return _maybe_numpy(pts_out, output_numpy), _maybe_numpy(norms, output_numpy)
    return _maybe_numpy(pts_out, output_numpy)


# 9) dropout points (PointNet++ style)
def random_dropout_points(
    points: Union[np.ndarray, torch.Tensor],
    max_p: float,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    output_numpy: bool = False,
):
    pts = _to_torch(points, device=device)
    N = pts.shape[0]
    if seed is not None:
        torch.manual_seed(seed)
    p = torch.rand(N, device=pts.device) * max_p
    masks = torch.rand(N, device=pts.device) < p
    if masks.any():
        pts[masks] = pts[0]
    return _maybe_numpy(pts, output_numpy)


# 10) jitter features
def random_jitter_features(
    features: Union[np.ndarray, torch.Tensor],
    mu=0.0,
    sigma=0.01,
    prob=0.95,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    output_numpy: bool = False,
):
    feats = _to_torch(features, device=device)
    do = True if torch.rand(()) < prob else False
    if seed is not None:
        torch.manual_seed(seed)
    if do:
        noise = torch.normal(mu, sigma, size=feats.shape, device=feats.device, dtype=feats.dtype)
        feats = feats + noise
    return _maybe_numpy(feats, output_numpy)


# 11) random sample plane (unit normal)
def random_sample_plane(
    device: Optional[torch.device] = None, seed: Optional[int] = None, output_numpy: bool = False
):
    if seed is not None:
        torch.manual_seed(seed)
    u = torch.rand((), device=device)  # in [0,1)
    v = torch.rand((), device=device)
    theta = torch.acos(1 - 2 * u)  # map to [0, pi]
    phi = 2 * math.pi * v
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    normal = torch.stack([x, y, z], dim=0)
    return normal.cpu().numpy() if output_numpy else normal


# 12) crop by plane
def random_crop_point_cloud_with_plane(
    points: Union[np.ndarray, torch.Tensor],
    p_normal: Optional[Union[np.ndarray, torch.Tensor]] = None,
    keep_ratio: float = 0.7,
    normals: Optional[Union[np.ndarray, torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    output_numpy: bool = False,
):
    pts = _to_torch(points, device=device)
    if p_normal is None:
        p_normal_t = random_sample_plane(device=device, seed=seed, output_numpy=False)
    else:
        p_normal_t = _to_torch(p_normal, device=device).view(3)
    dists = pts @ p_normal_t
    K = int(math.floor(pts.shape[0] * keep_ratio + 0.5))
    sel = torch.argsort(-dists)[:K]
    pts_out = pts[sel]
    if normals is not None:
        norms = _to_torch(normals, device=device)[sel]
        return _maybe_numpy(pts_out, output_numpy), _maybe_numpy(norms, output_numpy)
    return _maybe_numpy(pts_out, output_numpy)


# 13) random sample viewpoint (keep numpy-like return by default)
def random_sample_viewpoint(
    limit: float = 500.0,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    output_numpy: bool = False,
):
    if seed is not None:
        torch.manual_seed(seed)
    v = torch.rand(3, device=device) + torch.tensor([limit, limit, limit], device=device) * (
        torch.randint(0, 2, (3,), device=device) * 2 - 1
    )
    return v.cpu().numpy() if output_numpy else v


# 14) crop by viewpoint
def random_crop_point_cloud_with_point(
    points: Union[np.ndarray, torch.Tensor],
    viewpoint: Optional[Union[np.ndarray, torch.Tensor]] = None,
    keep_ratio: float = 0.7,
    normals: Optional[Union[np.ndarray, torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    output_numpy: bool = False,
):
    pts = _to_torch(points, device=device)
    if viewpoint is None:
        vp = random_sample_viewpoint(device=device, seed=seed, output_numpy=False)
    else:
        vp = _to_torch(viewpoint, device=device).view(3)
    dists = torch.linalg.norm(vp - pts, dim=1)
    K = int(math.floor(pts.shape[0] * keep_ratio + 0.5))
    sel = torch.argsort(dists)[:K]
    pts_out = pts[sel]
    if normals is not None:
        norms = _to_torch(normals, device=device)[sel]
        return _maybe_numpy(pts_out, output_numpy), _maybe_numpy(norms, output_numpy)
    return _maybe_numpy(pts_out, output_numpy)
