import warnings

import torch
from geotransformer.modules.ops.index_select import index_select
from geotransformer.modules.ops.pairwise_distance import pairwise_distance


def get_point_to_node_indices(
    points: torch.Tensor, nodes: torch.Tensor, return_counts: bool = False
):
    r"""Compute Point-to-Node partition indices of the point cloud.

    Distribute points to the nearest node. Each point is distributed to only one node.

    Args:
        points (Tensor): point cloud (N, C)
        nodes (Tensor): node set (M, C)
        return_counts (bool=False): whether return the number of points in each node.

    Returns:
        indices (LongTensor): index of the node that each point belongs to (N,)
        node_sizes (longTensor): the number of points in each node.
    """
    sq_dist_mat = pairwise_distance(points, nodes)
    indices = sq_dist_mat.min(dim=1)[1]
    if return_counts:
        unique_indices, unique_counts = torch.unique(indices, return_counts=True)
        node_sizes = torch.zeros(nodes.shape[0], dtype=torch.long).cuda()
        node_sizes[unique_indices] = unique_counts
        return indices, node_sizes
    else:
        return indices


@torch.no_grad()
def knn_partition(points: torch.Tensor, nodes: torch.Tensor, k: int, return_distance: bool = False):
    r"""k-NN partition of the point cloud.

    Find the k nearest points for each node.

    Args:
        points: torch.Tensor (num_point, num_channel)
        nodes: torch.Tensor (num_node, num_channel)
        k: int
        return_distance: bool

    Returns:
        knn_indices: torch.Tensor (num_node, k)
        knn_indices: torch.Tensor (num_node, k)
    """
    k = min(k, points.shape[0])
    sq_dist_mat = pairwise_distance(nodes, points)
    knn_sq_distances, knn_indices = sq_dist_mat.topk(dim=1, k=k, largest=False)
    if return_distance:
        knn_distances = torch.sqrt(knn_sq_distances)
        return knn_distances, knn_indices
    else:
        return knn_indices


@torch.no_grad()
def point_to_node_partition(
    points: torch.Tensor, nodes: torch.Tensor, point_limit: int, return_count: bool = False,
):
    r"""Point-to-Node partition to the point cloud.

    Safely handles sparse point clouds where nodes may have fewer points than point_limit.
    Always returns tensors with shape (M, point_limit) for architectural compatibility.

    Args:
        points (Tensor): (N, 3)
        nodes (Tensor): (M, 3)
        point_limit (int): max number of points to each node
        return_count (bool=False): whether to return `node_sizes`

    Returns:
        point_to_node (Tensor): (N,)
        node_sizes (LongTensor): (M,)          # only if return_count
        node_masks (BoolTensor): (M,)
        node_knn_indices (LongTensor): (M, K)  # K = point_limit
        node_knn_masks (BoolTensor): (M, K)
    """
    device = points.device
    M = nodes.shape[0]
    N = points.shape[0]

    # (M, N)
    sq_dist_mat = pairwise_distance(nodes, points)

    # Assign each point to nearest node
    point_to_node = sq_dist_mat.min(dim=0)[1]  # (N,)
    node_masks = torch.zeros(M, dtype=torch.bool, device=device)  # (M,)
    node_masks.index_fill_(0, point_to_node, True)

    # Build matching mask so each point only counts for its own node
    matching_masks = torch.zeros_like(sq_dist_mat, dtype=torch.bool)  # (M, N)
    point_indices = torch.arange(N, device=device)  # (N,)
    matching_masks[point_to_node, point_indices] = True  # (M, N)

    # Mask out points that do not belong to the node
    sq_dist_mat.masked_fill_(~matching_masks, 1e12)  # (M, N)

    # Global K for topk: cannot exceed N and cannot exceed point_limit
    max_k = min(point_limit, N)

    if max_k == 0:
        # Degenerate case: no points at all
        sentinel = N
        node_knn_indices = torch.full((M, point_limit), sentinel, dtype=torch.long, device=device)
        node_knn_masks = torch.zeros((M, point_limit), dtype=torch.bool, device=device)
    else:
        # Top-k with safe k (only get what's available)
        small_knn_indices = sq_dist_mat.topk(k=max_k, dim=1, largest=False)[1]  # (M, max_k)

        # CRITICAL: index_select BEFORE padding with sentinel
        # Map to node indices of those points
        node_knn_node_indices = index_select(point_to_node, small_knn_indices, dim=0)  # (M, max_k)
        node_indices = torch.arange(M, device=device).unsqueeze(1).expand(-1, max_k)  # (M, max_k)

        # Valid neighbors: points whose assigned node == this node
        small_knn_masks = torch.eq(node_knn_node_indices, node_indices)  # (M, max_k)

        # Fill invalid neighbors with sentinel index
        sentinel = N
        small_knn_indices = small_knn_indices.masked_fill(~small_knn_masks, sentinel)

        # Now allocate full (M, point_limit) and copy valid entries
        node_knn_indices = torch.full((M, point_limit), sentinel, dtype=torch.long, device=device,)
        node_knn_masks = torch.zeros((M, point_limit), dtype=torch.bool, device=device,)
        node_knn_indices[:, :max_k] = small_knn_indices
        node_knn_masks[:, :max_k] = small_knn_masks

    if return_count:
        unique_indices, unique_counts = torch.unique(point_to_node, return_counts=True)
        node_sizes = torch.zeros(M, dtype=torch.long, device=device)  # (M,)
        node_sizes.index_put_([unique_indices], unique_counts)
        return point_to_node, node_sizes, node_masks, node_knn_indices, node_knn_masks
    else:
        return point_to_node, node_masks, node_knn_indices, node_knn_masks


@torch.no_grad()
def point_to_node_partition_bug(
    points: torch.Tensor, nodes: torch.Tensor, point_limit: int, return_count: bool = False,
):
    r"""Point-to-Node partition to the point cloud.

    BUG: this implementation ignores point_to_node indices when building patches. However, the points that do not
    belong to a superpoint should be masked out.


    Args:
        points (Tensor): (N, 3)
        nodes (Tensor): (M, 3)
        point_limit (int): max number of points to each node
        return_count (bool=False): whether to return `node_sizes`

    Returns:
        point_to_node (Tensor): (N,)
        node_sizes (LongTensor): (M,)
        node_masks (BoolTensor): (M,)
        node_knn_indices (LongTensor): (M, K)
        node_knn_masks (BoolTensor) (M, K)
    """
    warnings.warn("There is a bug in this implementation. Use `point_to_node_partition` instead.")
    sq_dist_mat = pairwise_distance(nodes, points)  # (M, N)

    point_to_node = sq_dist_mat.min(dim=0)[1]  # (N,)
    node_masks = torch.zeros(nodes.shape[0], dtype=torch.bool).cuda()  # (M,)
    node_masks.index_fill_(0, point_to_node, True)

    node_knn_indices = sq_dist_mat.topk(k=point_limit, dim=1, largest=False)[1]  # (M, K)
    node_knn_node_indices = index_select(point_to_node, node_knn_indices, dim=0)  # (M, K)
    node_indices = (
        torch.arange(nodes.shape[0]).cuda().unsqueeze(1).expand(-1, point_limit)
    )  # (M, K)
    node_knn_masks = torch.eq(node_knn_node_indices, node_indices)  # (M, K)
    node_knn_indices.masked_fill_(~node_knn_masks, points.shape[0])

    if return_count:
        unique_indices, unique_counts = torch.unique(point_to_node, return_counts=True)
        node_sizes = torch.zeros(nodes.shape[0], dtype=torch.long).cuda()  # (M,)
        node_sizes.index_put_([unique_indices], unique_counts)
        return point_to_node, node_sizes, node_masks, node_knn_indices, node_knn_masks
    else:
        return point_to_node, node_masks, node_knn_indices, node_knn_masks


@torch.no_grad()
def ball_query_partition(
    points: torch.Tensor,
    nodes: torch.Tensor,
    radius: float,
    point_limit: int,
    return_count: bool = False,
):
    node_knn_distances, node_knn_indices = knn_partition(
        points, nodes, point_limit, return_distance=True
    )
    node_knn_masks = torch.lt(node_knn_distances, radius)  # (N, k)
    sentinel_indices = torch.full_like(node_knn_indices, points.shape[0])  # (N, k)
    node_knn_indices = torch.where(node_knn_masks, node_knn_indices, sentinel_indices)  # (N, k)

    if return_count:
        node_sizes = node_knn_masks.sum(1)  # (N,)
        return node_knn_indices, node_knn_masks, node_sizes
    else:
        return node_knn_indices, node_knn_masks
