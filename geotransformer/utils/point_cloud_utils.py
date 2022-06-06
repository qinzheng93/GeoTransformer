import torch
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.linalg import expm, norm


# Basic Utilities

def pairwise_distance(points0, points1, normalized=False, clamp=False):
    r"""
    [PyTorch/Numpy] Pairwise distance of two point clouds.

    :param points0: torch.Tensor (d0, ..., dn, num_point0, num_feature)
    :param points1: torch.Tensor (d0, ..., dn, num_point1, num_feature)
    :param normalized: bool (default: False)
        If True, the points are normalized, so a2 and b2 both 1. This enables us to use 2 instead of a2 + b2 for
        simplicity.
    :param clamp: bool (default: False)
        If True, all value will be assured to be non-negative.
    :return: dist: torch.Tensor (d0, ..., dn, num_point0, num_point1)
    """
    if isinstance(points0, torch.Tensor):
        ab = torch.matmul(points0, points1.transpose(-1, -2))
        if normalized:
            dist2 = 2 - 2 * ab
        else:
            a2 = torch.sum(points0 ** 2, dim=-1).unsqueeze(-1)
            b2 = torch.sum(points1 ** 2, dim=-1).unsqueeze(-2)
            dist2 = a2 - 2 * ab + b2
        if clamp:
            dist2 = torch.maximum(dist2, torch.zeros_like(dist2))
    else:
        ab = np.matmul(points0, points1.transpose(-1, -2))
        if normalized:
            dist2 = 2 - 2 * ab
        else:
            a2 = np.expand_dims(np.sum(points0 ** 2, axis=-1), axis=-1)
            b2 = np.expand_dims(np.sum(points1 ** 2, axis=-1), axis=-2)
            dist2 = a2 - 2 * ab + b2
        if clamp:
            dist2 = np.maximum(dist2, np.zeros_like(dist2))
    return dist2


def get_nearest_neighbor(ref_points, src_points, return_index=False):
    r"""
    [PyTorch/Numpy] For each item in ref_points, find its nearest neighbor in src_points.

    The PyTorch implementation is based on pairwise distances, thus it cannot be used for large point clouds.
    """
    if isinstance(ref_points, torch.Tensor):
        distances = torch.sqrt(pairwise_distance(ref_points, src_points))
        nn_distances, nn_indices = distances.min(dim=1)
        if return_index:
            return nn_distances, nn_indices
        else:
            return nn_distances
    else:
        kd_tree1 = cKDTree(src_points)
        distances, indices = kd_tree1.query(ref_points, k=1, n_jobs=-1)
        if return_index:
            return distances, indices
        else:
            return distances


def get_point_to_node(points, nodes, return_counts=False):
    r"""
    [PyTorch/Numpy] Distribute points to the nearest node. Each point is distributed to only one node.

    :param points: torch.Tensor (num_point, num_channel)
    :param nodes: torch.Tensor (num_node, num_channel)
    :param return_counts: bool (default: False)
        If True, return the number of points in each node.
    :return: indices: torch.Tensor (num_point)
        The indices of the nodes to which the points are distributed.
    """
    if isinstance(points, torch.Tensor):
        distances = pairwise_distance(points, nodes)
        indices = distances.min(dim=1)[1]    # 取每个点最近的node的索引
        if return_counts:
            unique_indices, unique_counts = torch.unique(indices, return_counts=True)   # 去除重复数据 返回node的索引以及node的个数
            if torch.cuda.is_available():
                node_sizes = torch.zeros(nodes.shape[0], dtype=torch.long).cuda()
            else:
                node_sizes = torch.zeros(nodes.shape[0], dtype=torch.long)
            node_sizes[unique_indices] = unique_counts
            return indices, node_sizes
        else:
            return indices
    else:
        _, indices = get_nearest_neighbor(points, nodes, return_index=True)
        if return_counts:
            unique_indices, unique_counts = np.unique(indices, return_counts=True)
            node_sizes = np.zeros(nodes.shape[0], dtype=np.int64)
            node_sizes[unique_indices] = unique_counts
            return indices, node_sizes
        else:
            return indices


def get_knn_indices(points, nodes, k, return_distance=False):
    r"""
    [PyTorch] Find the k nearest points for each node.

    :param points: torch.Tensor (num_point, num_channel)
    :param nodes: torch.Tensor (num_node, num_channel)
    :param k: int
    :param return_distance: bool
    :return knn_indices: torch.Tensor (num_node, k)
    """
    k = min(k, points.shape[0])
    dists = pairwise_distance(nodes, points)
    knn_distances, knn_indices = dists.topk(dim=1, k=k, largest=False)
    if return_distance:
        return torch.sqrt(knn_distances), knn_indices
    else:
        return knn_indices


@torch.no_grad()
def get_point_to_node_indices_and_masks(points, nodes, num_sample, return_counts=False):
    r"""
    [PyTorch] Perform point-to-node partition to the point cloud.

    :param points: torch.Tensor (num_point, 3)
    :param nodes: torch.Tensor (num_node, 3)
    :param num_sample: int
    :param return_counts: bool, whether to return `node_sizes`

    :return point_node_indices: torch.LongTensor (num_point,)
    :return node_sizes [Optional]: torch.LongTensor (num_node,)
    :return node_masks: torch.BoolTensor (num_node,)
    :return node_knn_indices: torch.LongTensor (num_node, max_point)
    :return node_knn_masks: torch.BoolTensor (num_node, max_point)
    """
    point_to_node, node_sizes = get_point_to_node(points, nodes, return_counts=True)
    node_masks = torch.gt(node_sizes, 0)      # torch.gt(input,other)比较input>other的数据，返回bool

    node_knn_indices = get_knn_indices(points, nodes, num_sample)  # (num_node, max_point)
    if torch.cuda.is_available():
        node_indices = torch.arange(nodes.shape[0]).cuda().unsqueeze(1).expand(-1, num_sample)
    else:
        node_indices = torch.arange(nodes.shape[0]).unsqueeze(1).expand(-1, num_sample)
    node_knn_masks = torch.eq(point_to_node[node_knn_indices], node_indices)
    sentinel_indices = torch.full_like(node_knn_indices, points.shape[0])  # 返回一个与node_knn_indices一样size，值为points.shape[0]
    node_knn_indices = torch.where(node_knn_masks, node_knn_indices, sentinel_indices)

    if return_counts:
        return point_to_node, node_sizes, node_masks, node_knn_indices, node_knn_masks
    else:
        return point_to_node, node_masks, node_knn_indices, node_knn_masks


# Transformation Utilities

def apply_transform(points, transform):
    r"""
    [PyTorch/Numpy] Apply a rigid transform to points.

    Given a point cloud P(3, N) and a transform matrix T in the form of
      | R t |
      | 0 1 |,
    the output point cloud Q = RP + t.

    In the implementation, P is (N, 3), so R should be transposed.

    There are two cases supported:
    1. points is (d0, .., dn, 3), transform is (4, 4), the output points are (d0, ..., dn, 3).
       In this case, the transform is applied to all points.
    2. points is (B, N, 3), transform is (B, 4, 4), the output points are (B, N, 3).
       In this case, the transform is applied batch-wise. The points can be broadcast if B=1.

    :param points: torch.Tensor (d0, ..., dn, 3) or (B, N, 3)
    :param transform: torch.Tensor (4, 4) or (B, 4, 4)
    :return: points, torch.Tensor (d0, ..., dn, 3) or (B, N, 3)
    """
    if transform.ndim == 2:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        points_shape = points.shape
        points = points.reshape(-1, 3)
        points = points @ rotation.transpose(-1, -2) + translation
        points = points.reshape(*points_shape)
    elif transform.ndim == 3 and points.ndim == 3:
        rotation = transform[:, :3, :3]  # (B, 3, 3)
        translation = transform[:, None, :3, 3]  # (B, 1, 3)
        points = points @ rotation.transpose(-1, -2) + translation
    else:
        raise ValueError('Incompatible shapes between points {} and transform {}.'.format(
            tuple(points.shape), tuple(transform.shape)
        ))
    return points


def compose_transforms(transforms):
    r"""
    Compose transforms from the first one to the last one.
    T = T_{n_1} \circ T_{n_2} \circ ... \circ T_1 \circ T_0
    :param transforms: list of torch.Tensor [(4, 4)]
    :return transform: torch.Tensor (4, 4)
    """
    final_transform = transforms[0]
    for transform in transforms[1:]:
        final_transform = torch.matmul(transform, final_transform)
    return final_transform


def get_transform_from_rotation_translation(rotation, translation):
    r"""
    [PyTorch/Numpy] Get rigid transform matrix from rotation matrix and translation vector.

    :param rotation: torch.Tensor (3, 3) or numpy.ndarray (3, 3)
    :param translation: torch.Tensor (3,) or numpy.ndarray (3,)
    :return transform: torch.Tensor (4, 4) or numpy.ndarray (4, 4)
    """
    if isinstance(rotation, torch.Tensor):
        transform = torch.eye(4).to(rotation.device)
    else:
        transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def get_rotation_translation_from_transform(transform):
    r"""
    [PyTorch/Numpy] Get rotation matrix and translation vector from rigid transform matrix.

    :param transform: torch.Tensor (4, 4) or numpy.ndarray (4, 4)
    :return rotation: torch.Tensor (3, 3) or numpy.ndarray (3, 3)
    :return translation: torch.Tensor (3,) or numpy.ndarray (3,)
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation


def random_sample_rotation(rotation_factor=1.):
    # angle_z, angle_y, angle_x
    euler = np.random.rand(3) * np.pi * 2 / rotation_factor  # (0, 2 * pi / rotation_range)
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    return rotation

