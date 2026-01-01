from typing import Tuple

import numpy as np
import torch
from geotransformer.modules.ops import (
    apply_transform,
    get_rotation_translation_from_transform,
    pairwise_distance,
)


def modified_chamfer_distance(
    raw_points: torch.Tensor,
    ref_points: torch.Tensor,
    src_points: torch.Tensor,
    gt_transform: torch.Tensor,
    transform: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    All inputs expected to be torch.Tensor: shapes
      raw_points: (B, N_raw, 3)
      ref_points: (B, N_ref, 3)
      src_points: (B, N_src, 3)
      gt_transform, transform: (B, 4, 4)
    Returns single scalar or (B,) depending on reduction.
    """
    assert reduction in ["mean", "sum", "none"]

    # P_t -> Q_raw
    aligned_src_points = apply_transform(src_points, transform)  # (B, N_src, 3)
    sq_dist_mat_p_q = pairwise_distance(aligned_src_points, raw_points)  # (B, N_src, N_raw)
    nn_sq_distances_p_q = sq_dist_mat_p_q.min(dim=-1)[0]  # (B, N_src)
    chamfer_distance_p_q = torch.sqrt(nn_sq_distances_p_q).mean(dim=-1)  # (B,)

    # Q -> P_raw
    # Note: we want composed_transform = transform @ inv(gt_transform)
    gt_inv = torch.inverse(gt_transform)
    composed_transform = torch.matmul(transform, gt_inv)  # (B, 4, 4)
    aligned_raw_points = apply_transform(raw_points, composed_transform)  # (B, N_raw, 3)
    sq_dist_mat_q_p = pairwise_distance(ref_points, aligned_raw_points)  # (B, N_ref, N_raw)
    nn_sq_distances_q_p = sq_dist_mat_q_p.min(dim=-1)[0]  # (B, N_ref)
    chamfer_distance_q_p = torch.sqrt(nn_sq_distances_q_p).mean(dim=-1)  # (B,)

    chamfer_distance = chamfer_distance_p_q + chamfer_distance_q_p  # (B,)

    if reduction == "mean":
        return chamfer_distance.mean()
    elif reduction == "sum":
        return chamfer_distance.sum()
    else:  # "none"
        return chamfer_distance


def relative_rotation_error(gt_rotations: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    r"""Isotropic Relative Rotation Error.

    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    Args:
        gt_rotations (Tensor): ground truth rotation matrix (*, 3, 3)
        rotations (Tensor): estimated rotation matrix (*, 3, 3)

    Returns:
        rre (Tensor): relative rotation errors (*)
    """
    # compute R^T * R_gt (same as provided implementation but device-correct)
    mat = torch.matmul(rotations.transpose(-1, -2), gt_rotations)
    trace = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
    x = 0.5 * (trace - 1.0)
    x = x.clamp(min=-1.0, max=1.0)
    # use torch.acos for differentiability and device-awareness
    x = torch.acos(x)
    rre = 180.0 * x / torch.tensor(np.pi, device=x.device, dtype=x.dtype)
    return rre


def relative_translation_error(
    gt_translations: torch.Tensor, translations: torch.Tensor
) -> torch.Tensor:
    r"""Isotropic Relative Rotation Error.

    RTE = \lVert t - \bar{t} \rVert_2

    Args:
        gt_translations (Tensor): ground truth translation vector (*, 3)
        translations (Tensor): estimated translation vector (*, 3)

    Returns:
        rre (Tensor): relative rotation errors (*)
    """
    return torch.linalg.norm(gt_translations - translations, dim=-1)


def isotropic_transform_error(
    gt_transforms: torch.Tensor, transforms: torch.Tensor, reduction: str = "mean"
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the isotropic Relative Rotation Error and Relative Translation Error.

    Args:
        gt_transforms (Tensor): ground truth transformation matrix (*, 4, 4)
        transforms (Tensor): estimated transformation matrix (*, 4, 4)
        reduction (str='mean'): reduction method, 'mean', 'sum' or 'none'

    Returns:
        rre (Tensor): relative rotation error.
        rte (Tensor): relative translation error.
    """
    assert reduction in ["mean", "sum", "none"]
    gt_rotations, gt_translations = get_rotation_translation_from_transform(gt_transforms)
    rotations, translations = get_rotation_translation_from_transform(transforms)

    rre = relative_rotation_error(gt_rotations, rotations)  # (*)
    rte = relative_translation_error(gt_translations, translations)  # (*)

    if reduction == "mean":
        return rre.mean(), rte.mean()
    elif reduction == "sum":
        return rre.sum(), rte.sum()
    else:
        return rre, rte


def _rotation_angle_from_relative(R_rel: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation angle (radians) from relative rotation matrix R_rel.
    R_rel: (..., 3, 3)
    Returns: angles (...,) in radians, non-negative, in [0, pi].
    Numerically stable via trace formula.
    """
    # trace: (...,)
    tr = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    x = 0.5 * (tr - 1.0)
    # clamp for numerical stability
    x = torch.clamp(x, min=-1.0, max=1.0)
    angles = torch.acos(x)
    return angles


def _rotation_error_angle_deg(gt_rot: torch.Tensor, est_rot: torch.Tensor) -> torch.Tensor:
    """
    Compute per-sample rotation error (in degrees) from gt_rot and est_rot.
    gt_rot, est_rot: (B, 3, 3)
    Returns: (B,) angles in degrees
    """
    # relative rotation = est^T * gt  OR  R_rel = R_est.transpose(-1,-2) @ R_gt
    R_rel = torch.matmul(est_rot.transpose(-1, -2), gt_rot)
    angles_rad = _rotation_angle_from_relative(R_rel)  # (...,)
    angles_deg = angles_rad * (
        180.0 / torch.tensor(np.pi, device=angles_rad.device, dtype=angles_rad.dtype)
    )
    return angles_deg


def anisotropic_transform_error(
    gt_transforms: torch.Tensor, transforms: torch.Tensor, reduction: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fully-torch anisotropic transform error (MSE, MAE) by default.

    Args:
        gt_transforms: (B,4,4) tensor
        transforms: (B,4,4) tensor
        reduction: "mean"|"sum"|"none"

    Returns:
        (r_mse, r_mae, t_mse, t_mae) either scalars (if reduction != "none") or (B,) tensors.
        All returned tensors are on the same device as `transforms`.
    """
    assert reduction in ["mean", "sum", "none"]

    device = transforms.device
    # Extract rotations and translations
    gt_rot = gt_transforms[..., :3, :3].to(device=device)
    gt_t = gt_transforms[..., :3, 3].to(device=device)
    est_rot = transforms[..., :3, :3].to(device=device)
    est_t = transforms[..., :3, 3].to(device=device)

    # Rotation error (degrees) per-sample
    angles_deg = _rotation_error_angle_deg(gt_rot, est_rot)  # (B,)

    # Rotation MSE / MAE (we treat the angle as the scalar per-sample error)
    r_mse_t = angles_deg ** 2
    r_mae_t = torch.abs(angles_deg)

    # Translation errors: compute per-sample mean-squared / mean-abs on components
    # Option A: use L2 per-sample -> mse = mean( (t_gt - t_est)^2 ) across components
    #           mae = mean( |t_gt - t_est| ) across components
    t_diff = gt_t - est_t  # (B,3)
    t_mse_t = torch.mean(t_diff ** 2, dim=-1)  # (B,)
    t_mae_t = torch.mean(torch.abs(t_diff), dim=-1)  # (B,)

    # Reduction
    if reduction == "mean":
        return r_mse_t.mean(), r_mae_t.mean(), t_mse_t.mean(), t_mae_t.mean()
    elif reduction == "sum":
        return r_mse_t.sum(), r_mae_t.sum(), t_mse_t.sum(), t_mae_t.sum()
    else:
        # "none": return per-sample tensors
        return r_mse_t, r_mae_t, t_mse_t, t_mae_t
