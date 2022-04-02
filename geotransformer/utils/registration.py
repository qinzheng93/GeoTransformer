import warnings

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from geotransformer.utils.pointcloud import (
    apply_transform,
    get_nearest_neighbor,
    get_rotation_translation_from_transform,
)


# Metrics


def compute_relative_rotation_error(gt_rotation: np.ndarray, est_rotation: np.ndarray):
    r"""Compute the isotropic Relative Rotation Error.

    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    Args:
        gt_rotation (array): ground truth rotation matrix (3, 3)
        est_rotation (array): estimated rotation matrix (3, 3)

    Returns:
        rre (float): relative rotation error.
    """
    x = 0.5 * (np.trace(np.matmul(est_rotation.T, gt_rotation)) - 1.0)
    x = np.clip(x, -1.0, 1.0)
    x = np.arccos(x)
    rre = 180.0 * x / np.pi
    return rre


def compute_relative_translation_error(gt_translation: np.ndarray, est_translation: np.ndarray):
    r"""Compute the isotropic Relative Translation Error.

    RTE = \lVert t - \bar{t} \rVert_2

    Args:
        gt_translation (array): ground truth translation vector (3,)
        est_translation (array): estimated translation vector (3,)

    Returns:
        rte (float): relative translation error.
    """
    return np.linalg.norm(gt_translation - est_translation)


def compute_registration_error(gt_transform: np.ndarray, est_transform: np.ndarray):
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


def compute_rotation_mse_and_mae(gt_rotation: np.ndarray, est_rotation: np.ndarray):
    r"""Compute anisotropic rotation error (MSE and MAE)."""
    gt_euler_angles = Rotation.from_dcm(gt_rotation).as_euler('xyz', degrees=True)  # (3,)
    est_euler_angles = Rotation.from_dcm(est_rotation).as_euler('xyz', degrees=True)  # (3,)
    mse = np.mean((gt_euler_angles - est_euler_angles) ** 2)
    mae = np.mean(np.abs(gt_euler_angles - est_euler_angles))
    return mse, mae


def compute_translation_mse_and_mae(gt_translation: np.ndarray, est_translation: np.ndarray):
    r"""Compute anisotropic translation error (MSE and MAE)."""
    mse = np.mean((gt_translation - est_translation) ** 2)
    mae = np.mean(np.abs(gt_translation - est_translation))
    return mse, mae


def compute_transform_mse_and_mae(gt_transform: np.ndarray, est_transform: np.ndarray):
    r"""Compute anisotropic rotation and translation error (MSE and MAE)."""
    gt_rotation, gt_translation = get_rotation_translation_from_transform(gt_transform)
    est_rotation, est_translation = get_rotation_translation_from_transform(est_transform)
    r_mse, r_mae = compute_rotation_mse_and_mae(gt_rotation, est_rotation)
    t_mse, t_mae = compute_translation_mse_and_mae(gt_translation, est_translation)
    return r_mse, r_mae, t_mse, t_mae


def compute_registration_rmse(src_points: np.ndarray, gt_transform: np.ndarray, est_transform: np.ndarray):
    r"""Compute re-alignment error (approximated RMSE in 3DMatch).

    Used in Rotated 3DMatch.

    Args:
        src_points (array): source point cloud. (N, 3)
        gt_transform (array): ground-truth transformation. (4, 4)
        est_transform (array): estimated transformation. (4, 4)

    Returns:
        error (float): root mean square error.
    """
    gt_points = apply_transform(src_points, gt_transform)
    est_points = apply_transform(src_points, est_transform)
    error = np.linalg.norm(gt_points - est_points, axis=1).mean()
    return error


def compute_modified_chamfer_distance(
    raw_points: np.ndarray,
    ref_points: np.ndarray,
    src_points: np.ndarray,
    gt_transform: np.ndarray,
    est_transform: np.ndarray,
):
    r"""Compute the modified chamfer distance (RPMNet)."""
    # P_t -> Q_raw
    aligned_src_points = apply_transform(src_points, est_transform)
    chamfer_distance_p_q = get_nearest_neighbor(aligned_src_points, raw_points).mean()
    # Q -> P_raw
    composed_transform = np.matmul(est_transform, np.linalg.inv(gt_transform))
    aligned_raw_points = apply_transform(raw_points, composed_transform)
    chamfer_distance_q_p = get_nearest_neighbor(ref_points, aligned_raw_points).mean()
    # sum up
    chamfer_distance = chamfer_distance_p_q + chamfer_distance_q_p
    return chamfer_distance


def compute_correspondence_residual(ref_corr_points, src_corr_points, transform):
    r"""Computing the mean distance between a set of correspondences."""
    src_corr_points = apply_transform(src_corr_points, transform)
    residuals = np.sqrt(((ref_corr_points - src_corr_points) ** 2).sum(1))
    mean_residual = np.mean(residuals)
    return mean_residual


def compute_inlier_ratio(ref_corr_points, src_corr_points, transform, positive_radius=0.1):
    r"""Computing the inlier ratio between a set of correspondences."""
    src_corr_points = apply_transform(src_corr_points, transform)
    residuals = np.sqrt(((ref_corr_points - src_corr_points) ** 2).sum(1))
    inlier_ratio = np.mean(residuals < positive_radius)
    return inlier_ratio


def compute_overlap(ref_points, src_points, transform=None, positive_radius=0.1):
    r"""Compute the overlap of two point clouds."""
    if transform is not None:
        src_points = apply_transform(src_points, transform)
    nn_distances = get_nearest_neighbor(ref_points, src_points)
    overlap = np.mean(nn_distances < positive_radius)
    return overlap


# Ground Truth Utilities


def get_correspondences(ref_points, src_points, transform, matching_radius):
    r"""Find the ground truth correspondences within the matching radius between two point clouds.

    Return correspondence indices [indices in ref_points, indices in src_points]
    """
    src_points = apply_transform(src_points, transform)
    src_tree = cKDTree(src_points)
    indices_list = src_tree.query_ball_point(ref_points, matching_radius)
    corr_indices = np.array(
        [(i, j) for i, indices in enumerate(indices_list) for j in indices],
        dtype=np.long,
    )
    return corr_indices


# Matching Utilities


def extract_corr_indices_from_feats(
    ref_feats: np.ndarray,
    src_feats: np.ndarray,
    mutual: bool = False,
    bilateral: bool = False,
):
    r"""Extract correspondence indices from features.

    Args:
        ref_feats (array): (N, C)
        src_feats (array): (M, C)
        mutual (bool = False): whether use mutual matching
        bilateral (bool = False): whether use bilateral non-mutual matching, ignored if `mutual` is True.

    Returns:
        ref_corr_indices: (M,)
        src_corr_indices: (M,)
    """
    ref_nn_indices = get_nearest_neighbor(ref_feats, src_feats, return_index=True)[1]
    if mutual or bilateral:
        src_nn_indices = get_nearest_neighbor(src_feats, ref_feats, return_index=True)[1]
        ref_indices = np.arange(ref_feats.shape[0])
        if mutual:
            ref_masks = np.equal(src_nn_indices[ref_nn_indices], ref_indices)
            ref_corr_indices = ref_indices[ref_masks]
            src_corr_indices = ref_nn_indices[ref_corr_indices]
        else:
            src_indices = np.arange(src_feats.shape[0])
            ref_corr_indices = np.concatenate([ref_indices, src_nn_indices], axis=0)
            src_corr_indices = np.concatenate([ref_nn_indices, src_indices], axis=0)
    else:
        ref_corr_indices = np.arange(ref_feats.shape[0])
        src_corr_indices = ref_nn_indices
    return ref_corr_indices, src_corr_indices


def extract_correspondences_from_feats(
    ref_points: np.ndarray,
    src_points: np.ndarray,
    ref_feats: np.ndarray,
    src_feats: np.ndarray,
    mutual: bool = False,
    return_feat_dist: bool = False,
):
    r"""Extract correspondences from features."""
    ref_corr_indices, src_corr_indices = extract_corr_indices_from_feats(ref_feats, src_feats, mutual=mutual)

    ref_corr_points = ref_points[ref_corr_indices]
    src_corr_points = src_points[src_corr_indices]
    outputs = [ref_corr_points, src_corr_points]
    if return_feat_dist:
        ref_corr_feats = ref_feats[ref_corr_indices]
        src_corr_feats = src_feats[src_corr_indices]
        feat_dists = np.linalg.norm(ref_corr_feats - src_corr_feats, axis=1)
        outputs.append(feat_dists)
    return outputs


# Evaluation Utilities


def evaluate_correspondences(ref_points, src_points, transform, positive_radius=0.1):
    overlap = compute_overlap(ref_points, src_points, transform, positive_radius=positive_radius)
    inlier_ratio = compute_inlier_ratio(ref_points, src_points, transform, positive_radius=positive_radius)
    residual = compute_correspondence_residual(ref_points, src_points, transform)

    return {
        'overlap': overlap,
        'inlier_ratio': inlier_ratio,
        'residual': residual,
        'num_corr': ref_points.shape[0],
    }


def evaluate_sparse_correspondences(ref_points, src_points, ref_corr_indices, src_corr_indices, gt_corr_indices):
    ref_gt_corr_indices = gt_corr_indices[:, 0]
    src_gt_corr_indices = gt_corr_indices[:, 1]

    gt_corr_mat = np.zeros((ref_points.shape[0], src_points.shape[0]))
    gt_corr_mat[ref_gt_corr_indices, src_gt_corr_indices] = 1.0
    num_gt_correspondences = gt_corr_mat.sum()

    pred_corr_mat = np.zeros_like(gt_corr_mat)
    pred_corr_mat[ref_corr_indices, src_corr_indices] = 1.0
    num_pred_correspondences = pred_corr_mat.sum()

    pos_corr_mat = gt_corr_mat * pred_corr_mat
    num_pos_correspondences = pos_corr_mat.sum()

    precision = num_pos_correspondences / (num_pred_correspondences + 1e-12)
    recall = num_pos_correspondences / (num_gt_correspondences + 1e-12)

    pos_corr_mat = pos_corr_mat > 0
    gt_corr_mat = gt_corr_mat > 0
    ref_hit_ratio = np.any(pos_corr_mat, axis=1).sum() / (np.any(gt_corr_mat, axis=1).sum() + 1e-12)
    src_hit_ratio = np.any(pos_corr_mat, axis=0).sum() / (np.any(gt_corr_mat, axis=0).sum() + 1e-12)
    hit_ratio = 0.5 * (ref_hit_ratio + src_hit_ratio)

    return {
        'precision': precision,
        'recall': recall,
        'hit_ratio': hit_ratio,
    }
