from typing import Optional, Tuple, Union

import numpy as np
import torch
from geotransformer.utils.pointcloud import (
    apply_transform,
    get_nearest_neighbor,
    get_rotation_translation_from_transform,
)
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# Metrics


def compute_relative_rotation_error(
    gt_rotation: torch.Tensor, est_rotation: torch.Tensor
) -> torch.Tensor:
    r"""Compute the isotropic Relative Rotation Error.

    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    Args:
        gt_rotation (array): ground truth rotation matrix (3, 3)
        est_rotation (array): estimated rotation matrix (3, 3)

    Returns:
        rre (float): relative rotation error.
    """
    # Ensure tensors
    if not torch.is_tensor(gt_rotation):
        gt_rotation = torch.from_numpy(np.asarray(gt_rotation)).float()
    if not torch.is_tensor(est_rotation):
        est_rotation = torch.from_numpy(np.asarray(est_rotation)).float()

    # Put on same device
    device = gt_rotation.device
    est_rotation = est_rotation.to(device=device, dtype=torch.float32)
    gt_rotation = gt_rotation.to(device=device, dtype=torch.float32)

    x = 0.5 * (torch.trace(est_rotation.T @ gt_rotation) - 1.0)
    x = torch.clamp(x, -1.0, 1.0)
    x = torch.acos(x)
    rre = 180.0 * x / torch.tensor(np.pi, device=device, dtype=torch.float32)
    return rre


def compute_relative_translation_error(
    gt_translation: torch.Tensor, est_translation: torch.Tensor
) -> torch.Tensor:
    """
    RTE = ||t_gt - t_est||_2

    Args:
        gt_translation (array): ground truth translation vector (3,)
        est_translation (array): estimated translation vector (3,)

    Returns torch scalar.
    """
    if not torch.is_tensor(gt_translation):
        gt_translation = torch.from_numpy(np.asarray(gt_translation)).float()
    if not torch.is_tensor(est_translation):
        est_translation = torch.from_numpy(np.asarray(est_translation)).float()

    device = gt_translation.device
    gt_translation = gt_translation.to(device=device, dtype=torch.float32)
    est_translation = est_translation.to(device=device, dtype=torch.float32)

    return torch.linalg.norm(gt_translation - est_translation)


def compute_registration_error(
    gt_transform: torch.Tensor, est_transform: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the isotropic Relative Rotation Error and Relative Translation Error.

    Args:
        gt_transform (array): ground truth transformation matrix (4, 4)
        est_transform (array): estimated transformation matrix (4, 4)

    Returns:
        rre (float): relative rotation error.
        rte (float): relative translation error.
    """
    gt_rotation, gt_translation = get_rotation_translation_from_transform(gt_transform)
    est_rotation, est_translation = get_rotation_translation_from_transform(est_transform)
    rre = compute_relative_rotation_error(gt_rotation, est_rotation)
    rte = compute_relative_translation_error(gt_translation, est_translation)
    return rre, rte


def compute_rotation_mse_and_mae(
    gt_rotation: torch.Tensor, est_rotation: torch.Tensor
) -> Tuple[float, float]:
    """
    Compute anisotropic rotation error (MSE and MAE) based on Euler angles (xyz, degrees).
    Returns numpy floats (same as original). If inputs are torch, we'll convert to CPU numpy,
    call SciPy, then return Python floats (to match original semantics).
    """
    # convert to numpy matrices on CPU for exact match with original SciPy behaviour
    if torch.is_tensor(gt_rotation):
        gt_np = gt_rotation.detach().cpu().numpy()
    else:
        gt_np = np.asarray(gt_rotation)
    if torch.is_tensor(est_rotation):
        est_np = est_rotation.detach().cpu().numpy()
    else:
        est_np = np.asarray(est_rotation)

    # SciPy's from_matrix / as_euler gives the same semantics as from_dcm in older versions
    gt_euler_angles = Rotation.from_matrix(gt_np).as_euler("xyz", degrees=True)
    est_euler_angles = Rotation.from_matrix(est_np).as_euler("xyz", degrees=True)
    mse = np.mean((gt_euler_angles - est_euler_angles) ** 2)
    mae = np.mean(np.abs(gt_euler_angles - est_euler_angles))
    return float(mse), float(mae)


def compute_translation_mse_and_mae(
    gt_translation: torch.Tensor, est_translation: torch.Tensor
) -> Tuple[float, float]:
    """
    Translation MSE / MAE (returns floats)
    """
    if torch.is_tensor(gt_translation):
        gt = gt_translation.detach().cpu().numpy()
    else:
        gt = np.asarray(gt_translation)
    if torch.is_tensor(est_translation):
        est = est_translation.detach().cpu().numpy()
    else:
        est = np.asarray(est_translation)

    mse = np.mean((gt - est) ** 2)
    mae = np.mean(np.abs(gt - est))
    return float(mse), float(mae)


def compute_transform_mse_and_mae(
    gt_transform: torch.Tensor, est_transform: torch.Tensor
) -> Tuple[float, float, float, float]:
    """
    Rotation MSE/MAE (Euler degrees) and translation MSE/MAE (floats).
    This preserves original SciPy-based semantics.
    """
    gt_rotation, gt_translation = get_rotation_translation_from_transform(gt_transform)
    est_rotation, est_translation = get_rotation_translation_from_transform(est_transform)
    r_mse, r_mae = compute_rotation_mse_and_mae(gt_rotation, est_rotation)
    t_mse, t_mae = compute_translation_mse_and_mae(gt_translation, est_translation)
    return r_mse, r_mae, t_mse, t_mae


def compute_registration_rmse(
    src_points: torch.Tensor, gt_transform: torch.Tensor, est_transform: torch.Tensor
) -> torch.Tensor:
    r"""Compute re-alignment error (approximated RMSE in 3DMatch).

    Used in Rotated 3DMatch.

    Args:
        src_points (array): source point cloud. (N, 3)
        gt_transform (array): ground-truth transformation. (4, 4)
        est_transform (array): estimated transformation. (4, 4)

    Returns:
        error (float): root mean square error.
    """
    # Convert to tensors if needed
    if not torch.is_tensor(src_points):
        src_points = torch.from_numpy(np.asarray(src_points)).float()
    device = src_points.device
    src_points = src_points.to(device=device, dtype=torch.float32)
    gt_points = apply_transform(src_points, gt_transform)
    est_points = apply_transform(src_points, est_transform)
    error = torch.linalg.norm(gt_points - est_points, dim=1).mean()
    return error


def compute_modified_chamfer_distance(
    raw_points: torch.Tensor,
    ref_points: torch.Tensor,
    src_points: torch.Tensor,
    gt_transform: torch.Tensor,
    est_transform: torch.Tensor,
) -> torch.Tensor:
    r"""Compute the modified chamfer distance (RPMNet)."""
    # ensure tensors
    if not torch.is_tensor(raw_points):
        raw_points = torch.from_numpy(np.asarray(raw_points)).float()
    if not torch.is_tensor(ref_points):
        ref_points = torch.from_numpy(np.asarray(ref_points)).float()
    if not torch.is_tensor(src_points):
        src_points = torch.from_numpy(np.asarray(src_points)).float()
    device = raw_points.device
    raw_points = raw_points.to(device=device, dtype=torch.float32)
    ref_points = ref_points.to(device=device, dtype=torch.float32)
    src_points = src_points.to(device=device, dtype=torch.float32)
    # P_t -> Q_raw
    aligned_src_points = apply_transform(src_points, est_transform)
    chamfer_distance_p_q = get_nearest_neighbor(aligned_src_points, raw_points).mean()
    # Q -> P_raw
    composed_transform = est_transform @ torch.inverse(gt_transform)
    aligned_raw_points = apply_transform(raw_points, composed_transform)
    chamfer_distance_q_p = get_nearest_neighbor(ref_points, aligned_raw_points).mean()
    # sum up
    chamfer_distance = chamfer_distance_p_q + chamfer_distance_q_p
    return chamfer_distance


def compute_correspondence_residual(
    ref_corr_points: torch.Tensor, src_corr_points: torch.Tensor, transform: torch.Tensor
) -> torch.Tensor:
    r"""Computing the mean distance between a set of correspondences."""
    if not torch.is_tensor(ref_corr_points):
        ref_corr_points = torch.from_numpy(np.asarray(ref_corr_points)).float()
    if not torch.is_tensor(src_corr_points):
        src_corr_points = torch.from_numpy(np.asarray(src_corr_points)).float()

    device = ref_corr_points.device
    ref_corr_points = ref_corr_points.to(device=device, dtype=torch.float32)
    src_corr_points = src_corr_points.to(device=device, dtype=torch.float32)

    src_trans = apply_transform(src_corr_points, transform)
    residuals = torch.sqrt(torch.sum((ref_corr_points - src_trans) ** 2, dim=1))
    mean_residual = residuals.mean()
    return mean_residual


def compute_inlier_ratio(
    ref_corr_points: torch.Tensor,
    src_corr_points: torch.Tensor,
    transform: torch.Tensor,
    positive_radius: float = 0.1,
) -> torch.Tensor:
    r"""Computing the inlier ratio between a set of correspondences."""
    if not torch.is_tensor(ref_corr_points):
        ref_corr_points = torch.from_numpy(np.asarray(ref_corr_points)).float()
    if not torch.is_tensor(src_corr_points):
        src_corr_points = torch.from_numpy(np.asarray(src_corr_points)).float()

    device = ref_corr_points.device
    ref_corr_points = ref_corr_points.to(device=device, dtype=torch.float32)
    src_corr_points = src_corr_points.to(device=device, dtype=torch.float32)

    src_trans = apply_transform(src_corr_points, transform)
    residuals = torch.sqrt(torch.sum((ref_corr_points - src_trans) ** 2, dim=1))
    inlier_ratio = torch.mean((residuals < positive_radius).to(dtype=torch.float32))
    return inlier_ratio


def compute_overlap(
    ref_points: torch.Tensor,
    src_points: torch.Tensor,
    transform: Optional[torch.Tensor] = None,
    positive_radius: float = 0.1,
) -> torch.Tensor:
    """
    Fraction of ref points whose nearest neighbor in (transformed) src is within positive_radius.
    Returns torch scalar (torch.float32).
    """
    if not torch.is_tensor(ref_points):
        ref_points = torch.from_numpy(np.asarray(ref_points)).float()
    if not torch.is_tensor(src_points):
        src_points = torch.from_numpy(np.asarray(src_points)).float()

    device = ref_points.device
    ref_points = ref_points.to(device=device, dtype=torch.float32)
    src_points = src_points.to(device=device, dtype=torch.float32)

    if transform is not None:
        src_points = apply_transform(src_points, transform)
    nn_distances = get_nearest_neighbor(ref_points, src_points)  # tensor of distances
    overlap = torch.mean((nn_distances < positive_radius).to(dtype=torch.float32))
    return overlap


# Ground Truth Utilities


def get_correspondences(
    ref_points: Union[torch.Tensor, np.ndarray],
    src_points: Union[torch.Tensor, np.ndarray],
    transform: Union[torch.Tensor, np.ndarray],
    matching_radius: float,
    output_type: str = "torch",  # "torch" or "numpy"
    device: Optional[torch.device] = None,
):
    """
    Find ground-truth correspondences within matching_radius.
    Torch-first: if inputs are tensors -> use torch.cdist path; else use cKDTree as before.
    Returns Nx2 indices array/tensor of correspondences (ref_idx, src_idx).
    """
    # If inputs are tensors -> work in torch
    if torch.is_tensor(ref_points) and torch.is_tensor(src_points):
        dev = device if device is not None else ref_points.device
        ref_t = ref_points.to(device=dev, dtype=torch.float32)
        src_t = src_points.to(device=dev, dtype=torch.float32)
        if torch.is_tensor(transform):
            transform_t = transform.to(device=dev, dtype=torch.float32)
            transform_np = None
        else:
            transform_t = torch.from_numpy(np.asarray(transform).astype(np.float32)).to(device=dev)
        # Transform src into ref frame
        src_trans = apply_transform(src_t, transform_t)
        # compute pairwise dists and threshold (note: memory O(N*M))
        dists = torch.cdist(ref_t, src_trans)  # (N_ref, N_src)
        mask = dists <= float(matching_radius)
        # build index pairs
        ref_idx, src_idx = torch.nonzero(mask, as_tuple=True)
        corr_indices = torch.stack(
            [ref_idx.to(dtype=torch.long), src_idx.to(dtype=torch.long)], dim=1
        )
        if output_type == "numpy":
            return corr_indices.cpu().numpy()
        return corr_indices
    else:
        # convert to numpy and use KD-tree (preserves previous behaviour)
        if torch.is_tensor(ref_points):
            ref_np = ref_points.detach().cpu().numpy()
        else:
            ref_np = np.asarray(ref_points)
        if torch.is_tensor(src_points):
            src_np = src_points.detach().cpu().numpy()
        else:
            src_np = np.asarray(src_points)
        if torch.is_tensor(transform):
            transform_np = transform.detach().cpu().numpy()
        else:
            transform_np = np.asarray(transform)
        src_trans_np = apply_transform(src_np, transform_np)
        tree = cKDTree(src_trans_np)
        indices_list = tree.query_ball_point(ref_np, matching_radius)
        corr_indices = np.array(
            [(i, j) for i, indices in enumerate(indices_list) for j in indices], dtype=np.int64
        )
        if output_type == "torch":
            dev = device if device is not None else torch.device("cpu")
            return torch.from_numpy(corr_indices).float().to(dev)
        return corr_indices


# Matching Utilities


def extract_corr_indices_from_feats(
    ref_feats: Union[torch.Tensor, np.ndarray],
    src_feats: Union[torch.Tensor, np.ndarray],
    mutual: bool = False,
    bilateral: bool = False,
    output_type: str = "torch",  # "torch" or "numpy"
    device: Optional[torch.device] = None,
):
    """
    Return (ref_indices, src_indices) as tensor or numpy array.
    Torch-first: uses get_nearest_neighbor (which itself is tensor-first).
    """
    # Convert to tensors for NN computations
    if not torch.is_tensor(ref_feats):
        ref_feats_t = torch.from_numpy(np.asarray(ref_feats)).float()
    else:
        ref_feats_t = ref_feats
    if not torch.is_tensor(src_feats):
        src_feats_t = torch.from_numpy(np.asarray(src_feats)).float()
    else:
        src_feats_t = src_feats

    dev = (
        device
        if device is not None
        else (ref_feats_t.device if torch.is_tensor(ref_feats_t) else torch.device("cpu"))
    )
    ref_feats_t = ref_feats_t.to(device=dev, dtype=torch.float32)
    src_feats_t = src_feats_t.to(device=dev, dtype=torch.float32)

    # get nearest neighbor (distances, indices)
    _, ref_nn_indices = get_nearest_neighbor(ref_feats_t, src_feats_t, return_index=True)
    # ensure indices are torch long on device
    if not torch.is_tensor(ref_nn_indices):
        ref_nn_indices = torch.from_numpy(np.asarray(ref_nn_indices)).to(
            device=dev, dtype=torch.long
        )
    else:
        ref_nn_indices = ref_nn_indices.to(device=dev, dtype=torch.long)

    if mutual or bilateral:
        _, src_nn_indices = get_nearest_neighbor(src_feats_t, ref_feats_t, return_index=True)
        if not torch.is_tensor(src_nn_indices):
            src_nn_indices = torch.from_numpy(np.asarray(src_nn_indices)).to(
                device=dev, dtype=torch.long
            )
        else:
            src_nn_indices = src_nn_indices.to(device=dev, dtype=torch.long)

        ref_indices = torch.arange(ref_feats_t.shape[0], device=dev, dtype=torch.long)

        if mutual:
            # mutual matching: src_nn[ref_nn[i]] == i
            # src_nn_indices is (S,), ref_nn_indices is (R,)
            # build mask of mutual matches
            # note: may index out of bound if shapes differ; keep shapes consistent
            mapped = src_nn_indices[ref_nn_indices]  # (R,)
            ref_masks = mapped == ref_indices
            ref_corr_indices = ref_indices[ref_masks]
            src_corr_indices = ref_nn_indices[ref_masks]
        else:
            # bilateral: union of ref->src and src->ref
            src_indices = torch.arange(src_feats_t.shape[0], device=dev, dtype=torch.long)
            ref_corr_indices = torch.cat([ref_indices, src_nn_indices], dim=0)
            src_corr_indices = torch.cat([ref_nn_indices, src_indices], dim=0)
    else:
        ref_corr_indices = torch.arange(ref_feats_t.shape[0], device=dev, dtype=torch.long)
        src_corr_indices = ref_nn_indices

    if output_type == "numpy":
        return ref_corr_indices.cpu().numpy(), src_corr_indices.cpu().numpy()
    return ref_corr_indices, src_corr_indices


def extract_correspondences_from_feats(
    ref_points: Union[torch.Tensor, np.ndarray],
    src_points: Union[torch.Tensor, np.ndarray],
    ref_feats: Union[torch.Tensor, np.ndarray],
    src_feats: Union[torch.Tensor, np.ndarray],
    mutual: bool = False,
    return_feat_dist: bool = False,
    output_type: str = "torch",
    device: Optional[torch.device] = None,
):
    """
    Returns [ref_corr_points, src_corr_points, (optional) feat_dists]
    Torch-first: outputs are tensors on `device` (or cpu).
    """
    ref_corr_indices, src_corr_indices = extract_corr_indices_from_feats(
        ref_feats, src_feats, mutual=mutual, output_type="torch", device=device
    )
    # convert inputs to tensors if needed
    if not torch.is_tensor(ref_points):
        ref_points_t = torch.from_numpy(np.asarray(ref_points)).float()
    else:
        ref_points_t = ref_points
    if not torch.is_tensor(src_points):
        src_points_t = torch.from_numpy(np.asarray(src_points)).float()
    else:
        src_points_t = src_points
    dev = (
        device
        if device is not None
        else (ref_points_t.device if torch.is_tensor(ref_points_t) else torch.device("cpu"))
    )
    ref_points_t = ref_points_t.to(device=dev, dtype=torch.float32)
    src_points_t = src_points_t.to(device=dev, dtype=torch.float32)

    def index(arr, idx):
        return arr[idx]

    ref_corr_points = index(ref_points_t, ref_corr_indices)
    src_corr_points = index(src_points_t, src_corr_indices)

    outputs = [ref_corr_points, src_corr_points]

    if return_feat_dist:
        # get corr feats as tensors then compute L2
        if not torch.is_tensor(ref_feats):
            ref_feats_t = torch.from_numpy(np.asarray(ref_feats)).float().to(device=dev)
        else:
            ref_feats_t = ref_feats.to(device=dev, dtype=torch.float32)
        if not torch.is_tensor(src_feats):
            src_feats_t = torch.from_numpy(np.asarray(src_feats)).float().to(device=dev)
        else:
            src_feats_t = src_feats.to(device=dev, dtype=torch.float32)
        ref_corr_feats = ref_feats_t[ref_corr_indices]
        src_corr_feats = src_feats_t[src_corr_indices]
        feat_dists = torch.norm(ref_corr_feats - src_corr_feats, dim=1)
        outputs.append(feat_dists)

    if output_type == "numpy":
        return [o.cpu().numpy() if torch.is_tensor(o) else o for o in outputs]
    return outputs


# Evaluation Utilities


def evaluate_correspondences(ref_points, src_points, transform, positive_radius: float = 0.1):
    """
    Wrapper to compute overlap, inlier_ratio, residual.
    Returns dict with torch scalars (unless inputs were numpy, in which case converted to torch).
    """
    overlap = compute_overlap(ref_points, src_points, transform, positive_radius=positive_radius)
    inlier_ratio = compute_inlier_ratio(
        ref_points, src_points, transform, positive_radius=positive_radius
    )
    residual = compute_correspondence_residual(ref_points, src_points, transform)
    num_corr = (
        ref_points.shape[0]
        if not torch.is_tensor(ref_points)
        else torch.tensor(ref_points.shape[0], dtype=torch.long, device=ref_points.device)
    )
    return {
        "overlap": overlap,
        "inlier_ratio": inlier_ratio,
        "residual": residual,
        "num_corr": num_corr,
    }


def evaluate_sparse_correspondences(
    ref_points, src_points, ref_corr_indices, src_corr_indices, gt_corr_indices
):
    """
    All inputs are assumed numpy-like (indices). This function builds confusion matrices with numpy.
    For large clouds this can be memory heavy, kept as original.
    Returns python floats.
    """
    # Convert to numpy for matrix ops (keeps original behaviour)
    ref_points_np = (
        ref_points.detach().cpu().numpy() if torch.is_tensor(ref_points) else np.asarray(ref_points)
    )
    src_points_np = (
        src_points.detach().cpu().numpy() if torch.is_tensor(src_points) else np.asarray(src_points)
    )
    ref_corr_indices_np = (
        ref_corr_indices.detach().cpu().numpy()
        if torch.is_tensor(ref_corr_indices)
        else np.asarray(ref_corr_indices)
    )
    src_corr_indices_np = (
        src_corr_indices.detach().cpu().numpy()
        if torch.is_tensor(src_corr_indices)
        else np.asarray(src_corr_indices)
    )
    gt_corr_indices_np = (
        gt_corr_indices.detach().cpu().numpy()
        if torch.is_tensor(gt_corr_indices)
        else np.asarray(gt_corr_indices)
    )

    ref_gt_corr_indices = gt_corr_indices_np[:, 0]
    src_gt_corr_indices = gt_corr_indices_np[:, 1]

    gt_corr_mat = np.zeros((ref_points_np.shape[0], src_points_np.shape[0]), dtype=np.float32)
    gt_corr_mat[ref_gt_corr_indices, src_gt_corr_indices] = 1.0
    num_gt_correspondences = gt_corr_mat.sum()

    pred_corr_mat = np.zeros_like(gt_corr_mat)
    pred_corr_mat[ref_corr_indices_np, src_corr_indices_np] = 1.0
    num_pred_correspondences = pred_corr_mat.sum()

    pos_corr_mat = gt_corr_mat * pred_corr_mat
    num_pos_correspondences = pos_corr_mat.sum()

    precision = float(num_pos_correspondences / (num_pred_correspondences + 1e-12))
    recall = float(num_pos_correspondences / (num_gt_correspondences + 1e-12))

    pos_corr_mat = pos_corr_mat > 0
    gt_corr_mat = gt_corr_mat > 0
    ref_hit_ratio = np.any(pos_corr_mat, axis=1).sum() / (np.any(gt_corr_mat, axis=1).sum() + 1e-12)
    src_hit_ratio = np.any(pos_corr_mat, axis=0).sum() / (np.any(gt_corr_mat, axis=0).sum() + 1e-12)
    hit_ratio = 0.5 * (ref_hit_ratio + src_hit_ratio)

    return {
        "precision": precision,
        "recall": recall,
        "hit_ratio": hit_ratio,
    }
