from typing import Optional

import torch

from geotransformer.modules.ops import index_select, apply_transform, pairwise_distance, get_point_to_node_indices


# Extract correspondences


@torch.no_grad()
def extract_correspondences_from_scores(
    score_mat: torch.Tensor,
    mutual: bool = False,
    bilateral: bool = False,
    has_dustbin: bool = False,
    threshold: float = 0.0,
    return_score: bool = False,
):
    r"""Extract the indices of correspondences from matching scores matrix (max selection).

    Args:
        score_mat (Tensor): the logarithmic matching probabilities (N, M) or (N + 1, M + 1) according to `has_dustbin`
        mutual (bool = False): whether to get mutual correspondences.
        bilateral (bool = False), whether bilateral non-mutual matching, ignored if `mutual` is set.
        has_dustbin (bool = False): whether to use slack variables.
        threshold (float = 0): confidence threshold.
        return_score (bool = False): return correspondence scores.

    Returns:
        ref_corr_indices (LongTensor): (C,)
        src_corr_indices (LongTensor): (C,)
        corr_scores (Tensor): (C,)
    """
    score_mat = torch.exp(score_mat)
    ref_length, src_length = score_mat.shape

    ref_max_scores, ref_max_indices = torch.max(score_mat, dim=1)
    ref_indices = torch.arange(ref_length).cuda()
    ref_corr_scores_mat = torch.zeros_like(score_mat)
    ref_corr_scores_mat[ref_indices, ref_max_indices] = ref_max_scores
    ref_corr_masks_mat = torch.gt(ref_corr_scores_mat, threshold)

    if mutual or bilateral:
        src_max_scores, src_max_indices = torch.max(score_mat, dim=0)
        src_indices = torch.arange(src_length).cuda()
        src_corr_scores_mat = torch.zeros_like(score_mat)
        src_corr_scores_mat[src_max_indices, src_indices] = src_max_scores
        src_corr_masks_mat = torch.gt(src_corr_scores_mat, threshold)

        if mutual:
            corr_masks_mat = torch.logical_and(ref_corr_masks_mat, src_corr_masks_mat)
        else:
            corr_masks_mat = torch.logical_or(ref_corr_masks_mat, src_corr_masks_mat)
    else:
        corr_masks_mat = ref_corr_masks_mat

    if has_dustbin:
        corr_masks_mat = corr_masks_mat[:-1, :-1]

    ref_corr_indices, src_corr_indices = torch.nonzero(corr_masks_mat, as_tuple=True)

    if return_score:
        corr_scores = score_mat[ref_corr_indices, src_corr_indices]
        return ref_corr_indices, src_corr_indices, corr_scores
    else:
        return ref_corr_indices, src_corr_indices


@torch.no_grad()
def extract_correspondences_from_scores_threshold(
    scores_mat: torch.Tensor, threshold: float, has_dustbin: bool = False, return_score: bool = False
):
    r"""Extract the indices of correspondences from matching scores matrix (thresholding selection).

    Args:
        score_mat (Tensor): the logarithmic matching probabilities (N, M) or (N + 1, M + 1) according to `has_dustbin`
        threshold (float = 0): confidence threshold
        has_dustbin (bool = False): whether to use slack variables
        return_score (bool = False): return correspondence scores

    Returns:
        ref_corr_indices (LongTensor): (C,)
        src_corr_indices (LongTensor): (C,)
        corr_scores (Tensor): (C,)
    """
    scores_mat = torch.exp(scores_mat)
    if has_dustbin:
        scores_mat = scores_mat[:-1, :-1]
    masks = torch.gt(scores_mat, threshold)
    ref_corr_indices, src_corr_indices = torch.nonzero(masks, as_tuple=True)

    if return_score:
        corr_scores = scores_mat[ref_corr_indices, src_corr_indices]
        return ref_corr_indices, src_corr_indices, corr_scores
    else:
        return ref_corr_indices, src_corr_indices


@torch.no_grad()
def extract_correspondences_from_scores_topk(
    scores_mat: torch.Tensor, k: int, has_dustbin: bool = False, largest: bool = True, return_score: bool = False
):
    r"""Extract the indices of correspondences from matching scores matrix (global top-k selection).

    Args:
        score_mat (Tensor): the scores (N, M) or (N + 1, M + 1) according to `has_dustbin`.
        k (int): top-k.
        has_dustbin (bool = False): whether to use slack variables.
        largest (bool = True): whether to choose the largest ones.
        return_score (bool = False): return correspondence scores.

    Returns:
        ref_corr_indices (LongTensor): (C,)
        src_corr_indices (LongTensor): (C,)
        corr_scores (Tensor): (C,)
    """
    corr_indices = scores_mat.view(-1).topk(k=k, largest=largest)[1]
    ref_corr_indices = corr_indices // scores_mat.shape[1]
    src_corr_indices = corr_indices % scores_mat.shape[1]
    if has_dustbin:
        ref_masks = torch.ne(ref_corr_indices, scores_mat.shape[0] - 1)
        src_masks = torch.ne(src_corr_indices, scores_mat.shape[1] - 1)
        masks = torch.logical_and(ref_masks, src_masks)
        ref_corr_indices = ref_corr_indices[masks]
        src_corr_indices = src_corr_indices[masks]

    if return_score:
        corr_scores = scores_mat[ref_corr_indices, src_corr_indices]
        return ref_corr_indices, src_corr_indices, corr_scores
    else:
        return ref_corr_indices, src_corr_indices


@torch.no_grad()
def extract_correspondences_from_feats(
    ref_feats: torch.Tensor,
    src_feats: torch.Tensor,
    mutual: bool = False,
    bilateral: bool = False,
    return_feat_dist: bool = False,
):
    r"""Extract the indices of correspondences from feature distances (nn selection).

    Args:
        ref_feats (Tensor): features of reference point cloud (N, C).
        src_feats (Tensor): features of source point cloud (M, C).
        mutual (bool = False): whether to get mutual correspondences.
        bilateral (bool = False), whether bilateral non-mutual matching, ignored if `mutual` is set.
        return_feat_dist (bool = False): return feature distances.

    Returns:
        ref_corr_indices (LongTensor): (C,)
        src_corr_indices (LongTensor): (C,)
        corr_feat_dists (Tensor): (C,)
    """
    feat_dists_mat = pairwise_distance(ref_feats, src_feats)

    ref_corr_indices, src_corr_indices = extract_correspondences_from_scores(
        -feat_dists_mat,
        mutual=mutual,
        has_dustbin=False,
        bilateral=bilateral,
    )

    if return_feat_dist:
        corr_feat_dists = feat_dists_mat[ref_corr_indices, src_corr_indices]
        return ref_corr_indices, src_corr_indices, corr_feat_dists
    else:
        return ref_corr_indices, src_corr_indices


# Patch correspondences


@torch.no_grad()
def dense_correspondences_to_node_correspondences(
    ref_points: torch.Tensor,
    src_points: torch.Tensor,
    ref_nodes: torch.Tensor,
    src_nodes: torch.Tensor,
    corr_indices: torch.Tensor,
    return_score: bool = False,
):
    r"""Generate patch correspondences from point correspondences and the number of point correspondences within each
    patch correspondences.

    For each point correspondence, convert it to patch correspondence by replacing the point indices to the
    corresponding patch indices.

    We also define the proxy score for each patch correspondence as a estimation of the overlap ratio:
    s = (#point_corr / #point_in_ref_patch + #point_corr / #point_in_src_patch) / 2

    Args:
        ref_points: reference point cloud
        src_points: source point cloud
        ref_nodes: reference patch points
        src_nodes: source patch points
        corr_indices: point correspondences
        return_score: whether return the proxy score for each patch correspondences

    Returns:
        node_corr_indices (LongTensor): (C, 2)
        node_corr_counts (LongTensor): (C,)
        node_corr_scores (Tensor): (C,)
    """
    ref_point_to_node, ref_node_sizes = get_point_to_node_indices(ref_points, ref_nodes, return_counts=True)
    src_point_to_node, src_node_sizes = get_point_to_node_indices(src_points, src_nodes, return_counts=True)

    ref_corr_indices = corr_indices[:, 0]
    src_corr_indices = corr_indices[:, 1]
    ref_node_corr_indices = ref_point_to_node[ref_corr_indices]
    src_node_corr_indices = src_point_to_node[src_corr_indices]

    node_corr_indices = ref_node_corr_indices * src_nodes.shape[0] + src_node_corr_indices
    node_corr_indices, node_corr_counts = torch.unique(node_corr_indices, return_counts=True)
    ref_node_corr_indices = node_corr_indices // src_nodes.shape[0]
    src_node_corr_indices = node_corr_indices % src_nodes.shape[0]
    node_corr_indices = torch.stack([ref_node_corr_indices, src_node_corr_indices], dim=1)

    if return_score:
        ref_node_corr_scores = node_corr_counts / ref_node_sizes[ref_node_corr_indices]
        src_node_corr_scores = node_corr_counts / src_node_sizes[src_node_corr_indices]
        node_corr_scores = (ref_node_corr_scores + src_node_corr_scores) / 2
        return node_corr_indices, node_corr_counts, node_corr_scores
    else:
        return node_corr_indices, node_corr_counts


@torch.no_grad()
def get_node_correspondences(
    ref_nodes: torch.Tensor,
    src_nodes: torch.Tensor,
    ref_knn_points: torch.Tensor,
    src_knn_points: torch.Tensor,
    transform: torch.Tensor,
    pos_radius: float,
    ref_masks: Optional[torch.Tensor] = None,
    src_masks: Optional[torch.Tensor] = None,
    ref_knn_masks: Optional[torch.Tensor] = None,
    src_knn_masks: Optional[torch.Tensor] = None,
):
    r"""Generate ground-truth superpoint/patch correspondences.

    Each patch is composed of at most k nearest points of the corresponding superpoint.
    A pair of points match if the distance between them is smaller than `self.pos_radius`.

    Args:
        ref_nodes: torch.Tensor (M, 3)
        src_nodes: torch.Tensor (N, 3)
        ref_knn_points: torch.Tensor (M, K, 3)
        src_knn_points: torch.Tensor (N, K, 3)
        transform: torch.Tensor (4, 4)
        pos_radius: float
        ref_masks (optional): torch.BoolTensor (M,) (default: None)
        src_masks (optional): torch.BoolTensor (N,) (default: None)
        ref_knn_masks (optional): torch.BoolTensor (M, K) (default: None)
        src_knn_masks (optional): torch.BoolTensor (N, K) (default: None)

    Returns:
        corr_indices: torch.LongTensor (C, 2)
        corr_overlaps: torch.Tensor (C,)
    """
    src_nodes = apply_transform(src_nodes, transform)
    src_knn_points = apply_transform(src_knn_points, transform)

    # generate masks
    if ref_masks is None:
        ref_masks = torch.ones(size=(ref_nodes.shape[0],), dtype=torch.bool).cuda()
    if src_masks is None:
        src_masks = torch.ones(size=(src_nodes.shape[0],), dtype=torch.bool).cuda()
    if ref_knn_masks is None:
        ref_knn_masks = torch.ones(size=(ref_knn_points.shape[0], ref_knn_points.shape[1]), dtype=torch.bool).cuda()
    if src_knn_masks is None:
        src_knn_masks = torch.ones(size=(src_knn_points.shape[0], src_knn_points.shape[1]), dtype=torch.bool).cuda()

    node_mask_mat = torch.logical_and(ref_masks.unsqueeze(1), src_masks.unsqueeze(0))  # (M, N)

    # filter out non-overlapping patches using enclosing sphere
    ref_knn_dists = torch.linalg.norm(ref_knn_points - ref_nodes.unsqueeze(1), dim=-1)  # (M, K)
    ref_knn_dists.masked_fill_(~ref_knn_masks, 0.0)
    ref_max_dists = ref_knn_dists.max(1)[0]  # (M,)
    src_knn_dists = torch.linalg.norm(src_knn_points - src_nodes.unsqueeze(1), dim=-1)  # (N, K)
    src_knn_dists.masked_fill_(~src_knn_masks, 0.0)
    src_max_dists = src_knn_dists.max(1)[0]  # (N,)
    dist_mat = torch.sqrt(pairwise_distance(ref_nodes, src_nodes))  # (M, N)
    intersect_mat = torch.gt(ref_max_dists.unsqueeze(1) + src_max_dists.unsqueeze(0) + pos_radius - dist_mat, 0)
    intersect_mat = torch.logical_and(intersect_mat, node_mask_mat)
    sel_ref_indices, sel_src_indices = torch.nonzero(intersect_mat, as_tuple=True)

    # select potential patch pairs
    ref_knn_masks = ref_knn_masks[sel_ref_indices]  # (B, K)
    src_knn_masks = src_knn_masks[sel_src_indices]  # (B, K)
    ref_knn_points = ref_knn_points[sel_ref_indices]  # (B, K, 3)
    src_knn_points = src_knn_points[sel_src_indices]  # (B, K, 3)

    point_mask_mat = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))  # (B, K, K)

    # compute overlaps
    dist_mat = pairwise_distance(ref_knn_points, src_knn_points)  # (B, K, K)
    dist_mat.masked_fill_(~point_mask_mat, 1e12)
    point_overlap_mat = torch.lt(dist_mat, pos_radius ** 2)  # (B, K, K)
    ref_overlap_counts = torch.count_nonzero(point_overlap_mat.sum(-1), dim=-1).float()  # (B,)
    src_overlap_counts = torch.count_nonzero(point_overlap_mat.sum(-2), dim=-1).float()  # (B,)
    ref_overlaps = ref_overlap_counts / ref_knn_masks.sum(-1).float()  # (B,)
    src_overlaps = src_overlap_counts / src_knn_masks.sum(-1).float()  # (B,)
    overlaps = (ref_overlaps + src_overlaps) / 2  # (B,)

    overlap_masks = torch.gt(overlaps, 0)
    ref_corr_indices = sel_ref_indices[overlap_masks]
    src_corr_indices = sel_src_indices[overlap_masks]
    corr_indices = torch.stack([ref_corr_indices, src_corr_indices], dim=1)
    corr_overlaps = overlaps[overlap_masks]

    return corr_indices, corr_overlaps


@torch.no_grad()
def node_correspondences_to_dense_correspondences(
    ref_knn_points,
    src_knn_points,
    ref_knn_indices,
    src_knn_indices,
    node_corr_indices,
    transform,
    matching_radius,
    ref_knn_masks=None,
    src_knn_masks=None,
    return_distance=False,
):
    if ref_knn_masks is None:
        ref_knn_masks = torch.ones_like(ref_knn_indices)
    if src_knn_masks is None:
        src_knn_masks = torch.ones_like(src_knn_indices)

    src_knn_points = apply_transform(src_knn_points, transform)
    ref_node_corr_indices = node_corr_indices[:, 0]  # (P,)
    src_node_corr_indices = node_corr_indices[:, 1]  # (P,)
    ref_node_corr_knn_indices = ref_knn_indices[ref_node_corr_indices]  # (P, K)
    src_node_corr_knn_indices = src_knn_indices[src_node_corr_indices]  # (P, K)
    ref_node_corr_knn_points = ref_knn_points[ref_node_corr_indices]  # (P, K, 3)
    src_node_corr_knn_points = src_knn_points[src_node_corr_indices]  # (P, K, 3)
    ref_node_corr_knn_masks = ref_knn_masks[ref_node_corr_indices]  # (P, K)
    src_node_corr_knn_masks = src_knn_masks[src_node_corr_indices]  # (P, K)
    dist_mat = torch.sqrt(pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points))  # (P, K, K)
    corr_mat = torch.lt(dist_mat, matching_radius)
    mask_mat = torch.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))
    corr_mat = torch.logical_and(corr_mat, mask_mat)  # (P, K, K)
    batch_indices, row_indices, col_indices = torch.nonzero(corr_mat, as_tuple=True)  # (C,) (C,) (C,)
    ref_corr_indices = ref_node_corr_knn_indices[batch_indices, row_indices]
    src_corr_indices = src_node_corr_knn_indices[batch_indices, col_indices]
    corr_indices = torch.stack([ref_corr_indices, src_corr_indices], dim=1)
    if return_distance:
        corr_distances = dist_mat[batch_indices, row_indices, col_indices]
        return corr_indices, corr_distances
    else:
        return corr_indices


@torch.no_grad()
def get_node_overlap_ratios(
    ref_points,
    src_points,
    ref_knn_points,
    src_knn_points,
    ref_knn_indices,
    src_knn_indices,
    node_corr_indices,
    transform,
    matching_radius,
    ref_knn_masks,
    src_knn_masks,
    eps=1e-5,
):
    corr_indices = node_correspondences_to_dense_correspondences(
        ref_knn_points,
        src_knn_points,
        ref_knn_indices,
        src_knn_indices,
        node_corr_indices,
        transform,
        matching_radius,
        ref_knn_masks=ref_knn_masks,
        src_knn_masks=ref_knn_masks,
    )
    unique_ref_corr_indices = torch.unique(corr_indices[:, 0])
    unique_src_corr_indices = torch.unique(corr_indices[:, 1])
    ref_overlap_masks = torch.zeros(ref_points.shape[0] + 1).cuda()  # pad for following indexing
    src_overlap_masks = torch.zeros(src_points.shape[0] + 1).cuda()  # pad for following indexing
    ref_overlap_masks.index_fill_(0, unique_ref_corr_indices, 1.0)
    src_overlap_masks.index_fill_(0, unique_src_corr_indices, 1.0)
    ref_knn_overlap_masks = index_select(ref_overlap_masks, ref_knn_indices, dim=0)  # (N', K)
    src_knn_overlap_masks = index_select(src_overlap_masks, src_knn_indices, dim=0)  # (M', K)
    ref_knn_overlap_ratios = (ref_knn_overlap_masks * ref_knn_masks).sum(1) / (ref_knn_masks.sum(1) + eps)
    src_knn_overlap_ratios = (src_knn_overlap_masks * src_knn_masks).sum(1) / (src_knn_masks.sum(1) + eps)
    return ref_knn_overlap_ratios, src_knn_overlap_ratios


@torch.no_grad()
def get_node_occlusion_ratios(
    ref_points,
    src_points,
    ref_knn_points,
    src_knn_points,
    ref_knn_indices,
    src_knn_indices,
    node_corr_indices,
    transform,
    matching_radius,
    ref_knn_masks,
    src_knn_masks,
    eps=1e-5,
):
    ref_knn_overlap_ratios, src_knn_overlap_ratios = get_node_overlap_ratios(
        ref_points,
        src_points,
        ref_knn_points,
        src_knn_points,
        ref_knn_indices,
        src_knn_indices,
        node_corr_indices,
        transform,
        matching_radius,
        ref_knn_masks,
        src_knn_masks,
        eps=eps,
    )
    ref_knn_occlusion_ratios = 1.0 - ref_knn_overlap_ratios
    src_knn_occlusion_ratios = 1.0 - src_knn_overlap_ratios
    return ref_knn_occlusion_ratios, src_knn_occlusion_ratios
