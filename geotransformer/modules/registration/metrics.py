import numpy as np
import torch

from geotransformer.modules.ops import apply_transform, pairwise_distance, get_rotation_translation_from_transform
from geotransformer.utils.registration import compute_transform_mse_and_mae


def modified_chamfer_distance(raw_points, ref_points, src_points, gt_transform, transform, reduction='mean'):
    r"""Compute the modified chamfer distance.

    Args:
        raw_points (Tensor): (B, N_raw, 3)
        ref_points (Tensor): (B, N_ref, 3)
        src_points (Tensor): (B, N_src, 3)
        gt_transform (Tensor): (B, 4, 4)
        transform (Tensor): (B, 4, 4)
        reduction (str='mean'): reduction method, 'mean', 'sum' or 'none'

    Returns:
        chamfer_distance
    """
    assert reduction in ['mean', 'sum', 'none']

    # P_t -> Q_raw
    aligned_src_points = apply_transform(src_points, transform)  # (B, N_src, 3)
    sq_dist_mat_p_q = pairwise_distance(aligned_src_points, raw_points)  # (B, N_src, N_raw)
    nn_sq_distances_p_q = sq_dist_mat_p_q.min(dim=-1)[0]  # (B, N_src)
    chamfer_distance_p_q = torch.sqrt(nn_sq_distances_p_q).mean(dim=-1)  # (B)

    # Q -> P_raw
    composed_transform = torch.matmul(transform, torch.inverse(gt_transform))  # (B, 4, 4)
    aligned_raw_points = apply_transform(raw_points, composed_transform)  # (B, N_raw, 3)
    sq_dist_mat_q_p = pairwise_distance(ref_points, aligned_raw_points)  # (B, N_ref, N_raw)
    nn_sq_distances_q_p = sq_dist_mat_q_p.min(dim=-1)[0]  # (B, N_ref)
    chamfer_distance_q_p = torch.sqrt(nn_sq_distances_q_p).mean(dim=-1)  # (B)

    # sum up
    chamfer_distance = chamfer_distance_p_q + chamfer_distance_q_p  # (B)

    if reduction == 'mean':
        chamfer_distance = chamfer_distance.mean()
    elif reduction == 'sum':
        chamfer_distance = chamfer_distance.sum()
    return chamfer_distance


def relative_rotation_error(gt_rotations, rotations):
    r"""Isotropic Relative Rotation Error.

    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    Args:
        gt_rotations (Tensor): ground truth rotation matrix (*, 3, 3)
        rotations (Tensor): estimated rotation matrix (*, 3, 3)

    Returns:
        rre (Tensor): relative rotation errors (*)
    """
    mat = torch.matmul(rotations.transpose(-1, -2), gt_rotations)
    trace = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
    x = 0.5 * (trace - 1.0)
    x = x.clamp(min=-1.0, max=1.0)
    x = torch.arccos(x)
    rre = 180.0 * x / np.pi
    return rre


def relative_translation_error(gt_translations, translations):
    r"""Isotropic Relative Rotation Error.

    RTE = \lVert t - \bar{t} \rVert_2

    Args:
        gt_translations (Tensor): ground truth translation vector (*, 3)
        translations (Tensor): estimated translation vector (*, 3)

    Returns:
        rre (Tensor): relative rotation errors (*)
    """
    rte = torch.linalg.norm(gt_translations - translations, dim=-1)
    return rte


def isotropic_transform_error(gt_transforms, transforms, reduction='mean'):
    r"""Compute the isotropic Relative Rotation Error and Relative Translation Error.

    Args:
        gt_transforms (Tensor): ground truth transformation matrix (*, 4, 4)
        transforms (Tensor): estimated transformation matrix (*, 4, 4)
        reduction (str='mean'): reduction method, 'mean', 'sum' or 'none'

    Returns:
        rre (Tensor): relative rotation error.
        rte (Tensor): relative translation error.
    """
    assert reduction in ['mean', 'sum', 'none']

    gt_rotations, gt_translations = get_rotation_translation_from_transform(gt_transforms)
    rotations, translations = get_rotation_translation_from_transform(transforms)

    rre = relative_rotation_error(gt_rotations, rotations)  # (*)
    rte = relative_translation_error(gt_translations, translations)  # (*)

    if reduction == 'mean':
        rre = rre.mean()
        rte = rte.mean()
    elif reduction == 'sum':
        rre = rre.sum()
        rte = rte.sum()

    return rre, rte


def anisotropic_transform_error(gt_transforms, transforms, reduction='mean'):
    r"""Compute the anisotropic Relative Rotation Error and Relative Translation Error.

    This function calls numpy-based implementation to achieve batch-wise computation and thus is non-differentiable.

    Args:
        gt_transforms (Tensor): ground truth transformation matrix (B, 4, 4)
        transforms (Tensor): estimated transformation matrix (B, 4, 4)
        reduction (str='mean'): reduction method, 'mean', 'sum' or 'none'

    Returns:
        r_mse (Tensor): rotation mse.
        r_mae (Tensor): rotation mae.
        t_mse (Tensor): translation mse.
        t_mae (Tensor): translation mae.
    """
    assert reduction in ['mean', 'sum', 'none']

    batch_size = gt_transforms.shape[0]
    gt_transforms_array = gt_transforms.detach().cpu().numpy()
    transforms_array = transforms.detach().cpu().numpy()

    all_r_mse = []
    all_r_mae = []
    all_t_mse = []
    all_t_mae = []
    for i in range(batch_size):
        r_mse, r_mae, t_mse, t_mae = compute_transform_mse_and_mae(gt_transforms_array[i], transforms_array[i])
        all_r_mse.append(r_mse)
        all_r_mae.append(r_mae)
        all_t_mse.append(t_mse)
        all_t_mae.append(t_mae)
    r_mse = torch.as_tensor(all_r_mse).to(gt_transforms)
    r_mae = torch.as_tensor(all_r_mae).to(gt_transforms)
    t_mse = torch.as_tensor(all_t_mse).to(gt_transforms)
    t_mae = torch.as_tensor(all_t_mae).to(gt_transforms)

    if reduction == 'mean':
        r_mse = r_mse.mean()
        r_mae = r_mae.mean()
        t_mse = t_mse.mean()
        t_mae = t_mae.mean()
    elif reduction == 'sum':
        r_mse = r_mse.sum()
        r_mae = r_mae.sum()
        t_mse = t_mse.sum()
        t_mae = t_mae.sum()

    return r_mse, r_mae, t_mse, t_mae
