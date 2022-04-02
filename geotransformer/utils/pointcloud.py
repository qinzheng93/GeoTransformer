from typing import Tuple, List, Optional, Union, Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


# Basic Utilities


def get_nearest_neighbor(
    q_points: np.ndarray,
    s_points: np.ndarray,
    return_index: bool = False,
):
    r"""Compute the nearest neighbor for the query points in support points."""
    s_tree = cKDTree(s_points)
    distances, indices = s_tree.query(q_points, k=1, n_jobs=-1)
    if return_index:
        return distances, indices
    else:
        return distances


def regularize_normals(points, normals, positive=True):
    r"""Regularize the normals towards the positive/negative direction to the origin point.

    positive: the origin point is on positive direction of the normals.
    negative: the origin point is on negative direction of the normals.
    """
    dot_products = -(points * normals).sum(axis=1, keepdims=True)
    direction = dot_products > 0
    if positive:
        normals = normals * direction - normals * (1 - direction)
    else:
        normals = normals * (1 - direction) - normals * direction
    return normals


# Transformation Utilities


def apply_transform(points: np.ndarray, transform: np.ndarray, normals: Optional[np.ndarray] = None):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    points = np.matmul(points, rotation.T) + translation
    if normals is not None:
        normals = np.matmul(normals, rotation.T)
        return points, normals
    else:
        return points


def compose_transforms(transforms: List[np.ndarray]) -> np.ndarray:
    r"""
    Compose transforms from the first one to the last one.
    T = T_{n_1} \circ T_{n_2} \circ ... \circ T_1 \circ T_0
    """
    final_transform = transforms[0]
    for transform in transforms[1:]:
        final_transform = np.matmul(transform, final_transform)
    return final_transform


def get_transform_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    r"""Get rigid transform matrix from rotation matrix and translation vector.

    Args:
        rotation (array): (3, 3)
        translation (array): (3,)

    Returns:
        transform: (4, 4)
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def get_rotation_translation_from_transform(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""Get rotation matrix and translation vector from rigid transform matrix.

    Args:
        transform (array): (4, 4)

    Returns:
        rotation (array): (3, 3)
        translation (array): (3,)
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    r"""Inverse rigid transform.

    Args:
        transform (array): (4, 4)

    Return:
        inv_transform (array): (4, 4)
    """
    rotation, translation = get_rotation_translation_from_transform(transform)  # (3, 3), (3,)
    inv_rotation = rotation.T  # (3, 3)
    inv_translation = -np.matmul(inv_rotation, translation)  # (3,)
    inv_transform = get_transform_from_rotation_translation(inv_rotation, inv_translation)  # (4, 4)
    return inv_transform


def random_sample_rotation(rotation_factor: float = 1.0) -> np.ndarray:
    # angle_z, angle_y, angle_x
    euler = np.random.rand(3) * np.pi * 2 / rotation_factor  # (0, 2 * pi / rotation_range)
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    return rotation


def random_sample_rotation_v2() -> np.ndarray:
    axis = np.random.rand(3) - 0.5
    axis = axis / np.linalg.norm(axis) + 1e-8
    theta = np.pi * np.random.rand()
    euler = axis * theta
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    return rotation


def random_sample_transform(rotation_magnitude: float, translation_magnitude: float) -> np.ndarray:
    euler = np.random.rand(3) * np.pi * rotation_magnitude / 180.0  # (0, rot_mag)
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    translation = np.random.uniform(-translation_magnitude, translation_magnitude, 3)
    transform = get_transform_from_rotation_translation(rotation, translation)
    return transform


# Sampling methods


def random_sample_keypoints(
    points: np.ndarray,
    feats: np.ndarray,
    num_keypoints: int,
) -> Tuple[np.ndarray, np.ndarray]:
    num_points = points.shape[0]
    if num_points > num_keypoints:
        indices = np.random.choice(num_points, num_keypoints, replace=False)
        points = points[indices]
        feats = feats[indices]
    return points, feats


def sample_keypoints_with_scores(
    points: np.ndarray,
    feats: np.ndarray,
    scores: np.ndarray,
    num_keypoints: int,
) -> Tuple[np.ndarray, np.ndarray]:
    num_points = points.shape[0]
    if num_points > num_keypoints:
        indices = np.argsort(-scores)[:num_keypoints]
        points = points[indices]
        feats = feats[indices]
    return points, feats


def random_sample_keypoints_with_scores(
    points: np.ndarray,
    feats: np.ndarray,
    scores: np.ndarray,
    num_keypoints: int,
) -> Tuple[np.ndarray, np.ndarray]:
    num_points = points.shape[0]
    if num_points > num_keypoints:
        indices = np.arange(num_points)
        probs = scores / np.sum(scores)
        indices = np.random.choice(indices, num_keypoints, replace=False, p=probs)
        points = points[indices]
        feats = feats[indices]
    return points, feats


def sample_keypoints_with_nms(
    points: np.ndarray,
    feats: np.ndarray,
    scores: np.ndarray,
    num_keypoints: int,
    radius: float,
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
                    current_masks = np.sum((sorted_points[i + 1 :] - sorted_points[i]) ** 2, axis=1) < radius2
                    masks[i + 1 :] = masks[i + 1 :] & ~current_masks
        points = sorted_points[indices]
        feats = sorted_feats[indices]
    return points, feats


def random_sample_keypoints_with_nms(
    points: np.ndarray,
    feats: np.ndarray,
    scores: np.ndarray,
    num_keypoints: int,
    radius: float,
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
                    current_masks = np.sum((sorted_points[i + 1 :] - sorted_points[i]) ** 2, axis=1) < radius2
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
    depth_mat: np.ndarray, intrinsics: np.ndarray, scaling_factor: float = 1000.0, distance_limit: float = 6.0
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
    v = coords / width
    depth = depth_mat.flatten()
    z = depth / scaling_factor
    z[z > distance_limit] = 0.0
    x = (u - center_x) * z / focal_x
    y = (v - center_y) * z / focal_y
    points = np.stack([x, y, z], axis=1)
    points = points[depth > 0]
    return points
