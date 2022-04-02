import torch

from geotransformer.modules.ops import index_select


def nearest_upsample(x, upsample_indices):
    """Pools features from the closest neighbors.

    WARNING: this function assumes the neighbors are ordered.

    Args:
        x: [n1, d] features matrix
        upsample_indices: [n2, max_num] Only the first column is used for pooling

    Returns:
        x: [n2, d] pooled features matrix
    """
    # Add a last row with minimum features for shadow pools
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)
    # Get features for each pooling location [n2, d]
    x = index_select(x, upsample_indices[:, 0], dim=0)
    return x


def knn_interpolate(s_feats, q_points, s_points, neighbor_indices, k, eps=1e-8):
    r"""K-NN interpolate.

    WARNING: this function assumes the neighbors are ordered.

    Args:
        s_feats (Tensor): (M, C)
        q_points (Tensor): (N, 3)
        s_points (Tensor): (M, 3)
        neighbor_indices (LongTensor): (N, X)
        k (int)
        eps (float)

    Returns:
        q_feats (Tensor): (N, C)
    """
    s_points = torch.cat((s_points, torch.zeros_like(s_points[:1, :])), 0)  # (M + 1, 3)
    s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)  # (M + 1, C)
    knn_indices = neighbor_indices[:, :k].contiguous()
    knn_points = index_select(s_points, knn_indices, dim=0)  # (N, k, 3)
    knn_feats = index_select(s_feats, knn_indices, dim=0)  # (N, k, C)
    knn_sq_distances = (q_points.unsqueeze(1) - knn_points).pow(2).sum(dim=-1)  # (N, k)
    knn_masks = torch.ne(knn_indices, s_points.shape[0] - 1).float()  # (N, k)
    knn_weights = knn_masks / (knn_sq_distances + eps)  # (N, k)
    knn_weights = knn_weights / (knn_weights.sum(dim=1, keepdim=True) + eps)  # (N, k)
    q_feats = (knn_feats * knn_weights.unsqueeze(-1)).sum(dim=1)  # (N, C)
    return q_feats


def maxpool(x, neighbor_indices):
    """Max pooling from neighbors.

    Args:
        x: [n1, d] features matrix
        neighbor_indices: [n2, max_num] pooling indices

    Returns:
        pooled_feats: [n2, d] pooled features matrix
    """
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)
    neighbor_feats = index_select(x, neighbor_indices, dim=0)
    pooled_feats = neighbor_feats.max(1)[0]
    return pooled_feats


def global_avgpool(x, batch_lengths):
    """Global average pooling over batch.

    Args:
        x: [N, D] input features
        batch_lengths: [B] list of batch lengths

    Returns:
        x: [B, D] averaged features
    """
    # Loop over the clouds of the batch
    averaged_features = []
    i0 = 0
    for b_i, length in enumerate(batch_lengths):
        # Average features for each batch cloud
        averaged_features.append(torch.mean(x[i0 : i0 + length], dim=0))
        # Increment for next cloud
        i0 += length
    # Average features in each batch
    x = torch.stack(averaged_features)
    return x
