import torch
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .point_cloud_utils import (
    get_rotation_translation_from_transform, apply_transform, get_nearest_neighbor, pairwise_distance
)


# Metrics

def compute_relative_rotation_error(gt_rotation, est_rotation):
    r"""
    [PyTorch/Numpy] Compute the isotropic Relative Rotation Error.
    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    :param gt_rotation: torch.Tensor (3, 3) or numpy.ndarray (3, 3)
    :param est_rotation: torch.Tensor (3, 3) or numpy.ndarray (3, 3)
    :return rre: torch.Tensor () or float
    """
    if isinstance(gt_rotation, torch.Tensor):
        x = 0.5 * (torch.trace(torch.matmul(est_rotation.T, gt_rotation)) - 1.)
        x = torch.clip(x, -1., 1.)
        x = torch.arccos(x)
    else:
        x = 0.5 * (np.trace(np.matmul(est_rotation.T, gt_rotation)) - 1.)
        x = np.clip(x, -1., 1.)
        x = np.arccos(x)
    rre = 180. * x / np.pi
    return rre


def compute_relative_translation_error(gt_translation, est_translation):
    r"""
    [Pytorch/Numpy] Compute the isotropic Relative Translation Error.
    RTE = \lVert t - \bar{t} \rVert_2

    :param gt_translation: torch.Tensor (3,) or numpy.ndarray (3,)
    :param est_translation: torch.Tensor (3,) or numpy.ndarray (3,)
    :return rte: torch.Tensor () or float
    """
    if isinstance(gt_translation, torch.Tensor):
        rte = torch.linalg.norm(gt_translation - est_translation)
    else:
        rte = np.linalg.norm(gt_translation - est_translation)
    return rte


def compute_registration_error(gt_transform, est_transform):
    r"""
    [PyTorch/Numpy] Compute the isotropic Relative Rotation Error and Relative Translation Error

    :param gt_transform: torch.Tensor (4, 4) or numpy.ndarray (4, 4)
    :param est_transform: numpy.ndarray (4, 4)
    :return rre: float
    :return rte: float
    """
    gt_rotation, gt_translation = get_rotation_translation_from_transform(gt_transform)
    est_rotation, est_translation = get_rotation_translation_from_transform(est_transform)
    rre = compute_relative_rotation_error(gt_rotation, est_rotation)
    rte = compute_relative_translation_error(gt_translation, est_translation)
    return rre, rte


def compute_rotation_mse_and_mae(gt_rotation, est_rotation):
    r"""
    [Numpy] Compute anisotropic rotation error (MSE and MAE).
    """
    gt_euler_angles = Rotation.from_dcm(gt_rotation).as_euler('xyz', degrees=True)  # (3,)
    est_euler_angles = Rotation.from_dcm(est_rotation).as_euler('xyz', degrees=True)  # (3,)
    mse = np.mean((gt_euler_angles - est_euler_angles) ** 2)
    mae = np.mean(np.abs(gt_euler_angles - est_euler_angles))
    return mse, mae


def compute_translation_mse_and_mae(gt_translation, est_translation):
    r"""
    [Numpy] Compute anisotropic translation error (MSE and MAE).
    """
    mse = np.mean((gt_translation - est_translation) ** 2)
    mae = np.mean(np.abs(gt_translation - est_translation))
    return mse, mae


def compute_transform_mse_and_mae(gt_transform, est_transform):
    r"""
    [Numpy] Compute anisotropic rotation and translation error (MSE and MAE).
    """
    gt_rotation, gt_translation = get_rotation_translation_from_transform(gt_transform)
    est_rotation, est_translation = get_rotation_translation_from_transform(est_transform)
    r_mse, r_mae = compute_rotation_mse_and_mae(gt_rotation, est_rotation)
    t_mse, t_mae = compute_translation_mse_and_mae(gt_translation, est_translation)
    return r_mse, r_mae, t_mse, t_mae


def compute_chamfer_distance(points, gt_transform, est_transform):
    r"""Compute Chamfer Distance.
    """
    if isinstance(points, torch.Tensor):
        transform = torch.matmul(torch.inverse(gt_transform), est_transform)
        new_points = apply_transform(points, transform)
        chamfer_distance = torch.linalg.norm(points - new_points, p=2, dim=1).mean()
    else:
        transform = np.matmul(np.linalg.inv(gt_transform), est_transform)
        new_points = apply_transform(points, transform)
        chamfer_distance = np.linalg.norm(points - new_points, ord=2, axis=1).mean()
    return chamfer_distance


def compute_modified_chamfer_distance(raw_points, ref_points, src_points, gt_transform, est_transform):
    r"""
    [Numpy] Compute the modified chamfer distance.
    """
    # P_t -> Q_raw
    aligned_src_points = apply_transform(src_points, est_transform)
    chamfer_distance_p_q = pairwise_distance(aligned_src_points, raw_points).min(1).mean()
    # Q -> P_raw
    composed_transform = np.matmul(est_transform, np.linalg.inv(gt_transform))
    aligned_raw_points = apply_transform(raw_points, composed_transform)
    chamfer_distance_q_p = pairwise_distance(ref_points, aligned_raw_points).min(1).mean()
    # sum up
    chamfer_distance = chamfer_distance_p_q + chamfer_distance_q_p
    return chamfer_distance


def compute_overlap(ref_points, src_points, transform, positive_radius=0.1):
    r"""
    [Numpy] Compute the overlap of two point clouds.
    """
    src_points = apply_transform(src_points, transform)
    dist = get_nearest_neighbor(ref_points, src_points)
    overlap = np.mean(dist < positive_radius)
    return overlap


def compute_inlier_ratio(ref_points, src_points, transform, positive_radius=0.1):
    r"""
    [Numpy] Computing the inlier ratio between a set of correspondences.
    """
    src_points = apply_transform(src_points, transform)
    distances = np.sqrt(((ref_points - src_points) ** 2).sum(1))
    inlier_ratio = np.mean(distances < positive_radius)
    return inlier_ratio


def compute_mean_distance(ref_points, src_points, transform):
    r"""
    [Numpy] Computing the mean distance between a set of correspondences.
    """
    src_points = apply_transform(src_points, transform)
    distances = np.sqrt(((ref_points - src_points) ** 2).sum(1))
    mean_distance = np.mean(distances)
    return mean_distance


# Ground Truth Utilities

def get_corr_indices(ref_points, src_points, transform, matching_radius):
    r"""
    [Numpy] Find the ground truth correspondences within the matching radius between two point clouds.

    Return correspondence indices [indices in ref_points, indices in src_points]
    """
    src_points = apply_transform(src_points, transform)
    src_tree = cKDTree(src_points)
    indices_list = src_tree.query_ball_point(ref_points, matching_radius)
    correspondences = np.array([(i, j) for i, indices in enumerate(indices_list) for j in indices], dtype=np.long)
    return correspondences


@torch.no_grad()
def get_node_corr_indices_and_overlaps(
        ref_nodes,
        src_nodes,
        ref_knn_points,
        src_knn_points,
        transform,
        pos_radius,
        ref_masks=None,
        src_masks=None,
        ref_knn_masks=None,
        src_knn_masks=None
):
    r"""
    Generate ground truth node correspondences.
    Each node is composed of its k nearest points. A pair of points match if the distance between them is below
    `self.pos_radius`.

    :param ref_nodes: torch.Tensor (M, 3)
    :param src_nodes: torch.Tensor (N, 3)
    :param ref_knn_points: torch.Tensor (M, K, 3)
    :param src_knn_points: torch.Tensor (N, K, 3)
    :param transform: torch.Tensor (4, 4)
    :param pos_radius: float
    :param ref_masks: torch.BoolTensor (M,) (default: None)
    :param src_masks: torch.BoolTensor (N,) (default: None)
    :param ref_knn_masks: torch.BoolTensor (M, K) (default: None)
    :param src_knn_masks: torch.BoolTensor (N, K) (default: None)

    :return corr_indices: torch.LongTensor (num_corr, 2)
    :return corr_overlaps: torch.Tensor (num_corr,)
    """
    src_nodes = apply_transform(src_nodes, transform)
    src_knn_points = apply_transform(src_knn_points, transform)

    if ref_masks is not None and src_masks is not None:
        node_masks = torch.logical_and(ref_masks.unsqueeze(1), src_masks.unsqueeze(0))
    else:
        node_masks = None

    # filter out non-overlapping patches using enclosing sphere
    ref_knn_distances = torch.sqrt(((ref_knn_points - ref_nodes.unsqueeze(1)) ** 2).sum(-1))  # (M, K)
    if ref_knn_masks is not None:
        ref_knn_distances[~ref_knn_masks] = 0.
    ref_radius = ref_knn_distances.max(1)[0]  # (M,)
    src_knn_distances = torch.sqrt(((src_knn_points - src_nodes.unsqueeze(1)) ** 2).sum(-1))  # (N, K)
    if src_knn_masks is not None:
        src_knn_distances[~src_knn_masks] = 0.
    src_radius = src_knn_distances.max(1)[0]  # (N,)
    dist_map = torch.sqrt(pairwise_distance(ref_nodes, src_nodes))  # (M, N)
    masks = torch.gt(ref_radius.unsqueeze(1) + src_radius.unsqueeze(0) + pos_radius - dist_map, 0)  # (M, N)
    if node_masks is not None:
        masks = torch.logical_and(masks, node_masks)
    ref_indices, src_indices = torch.nonzero(masks, as_tuple=True)

    if ref_knn_masks is not None and src_knn_masks is not None:
        ref_knn_masks = ref_knn_masks[ref_indices]  # (B, K)
        src_knn_masks = src_knn_masks[src_indices]  # (B, K)
        node_knn_masks = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))
    else:
        node_knn_masks = None

    # compute overlaps
    ref_knn_points = ref_knn_points[ref_indices]  # (B, K, 3)
    src_knn_points = src_knn_points[src_indices]  # (B, K, 3)
    dist_map = pairwise_distance(ref_knn_points, src_knn_points)  # (B, K, K)
    if node_knn_masks is not None:
        dist_map[~node_knn_masks] = 1e12
    point_corr_map = torch.lt(dist_map, pos_radius ** 2)
    ref_overlap_counts = torch.count_nonzero(point_corr_map.sum(-1), dim=-1).float()
    src_overlap_counts = torch.count_nonzero(point_corr_map.sum(-2), dim=-1).float()
    if node_knn_masks is not None:
        ref_overlaps = ref_overlap_counts / ref_knn_masks.sum(-1).float()
        src_overlaps = src_overlap_counts / src_knn_masks.sum(-1).float()
    else:
        ref_overlaps = ref_overlap_counts / ref_knn_points.shape[1]  # (B,)
        src_overlaps = src_overlap_counts / src_knn_points.shape[1]  # (B,)
    overlaps = (ref_overlaps + src_overlaps) / 2  # (B,)

    masks = torch.gt(overlaps, 0)
    ref_corr_indices = ref_indices[masks]
    src_corr_indices = src_indices[masks]
    corr_indices = torch.stack([ref_corr_indices, src_corr_indices], dim=1)
    corr_overlaps = overlaps[masks]

    return corr_indices, corr_overlaps


# Evaluation Utilities

def evaluate_correspondences(ref_points, src_points, transform, positive_radius=0.1):
    overlap = compute_overlap(ref_points, src_points, transform, positive_radius=positive_radius)
    inlier_ratio = compute_inlier_ratio(ref_points, src_points, transform, positive_radius=positive_radius)
    mean_distance = compute_mean_distance(ref_points, src_points, transform)
    result_dict = {
        'overlap': overlap,
        'inlier_ratio': inlier_ratio,
        'mean_dist': mean_distance,
        'num_corr': ref_points.shape[0]
    }
    return result_dict
