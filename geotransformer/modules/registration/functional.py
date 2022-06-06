import torch


def weighted_procrustes(src_points, tgt_points, weights=None, weight_thresh=0., eps=1e-5, return_transform=False):
    r"""
    Compute rigid transformation from `src_points` to `tgt_points` using weighted SVD.

    Modified from [PointDSC](https://github.com/XuyangBai/PointDSC/blob/master/models/common.py).

    :param src_points: torch.Tensor (batch_size, num_corr, 3) or (num_corr, 3)
    :param tgt_points: torch.Tensor (batch_size, num_corr, 3) or (num_corr, 3)
    :param weights: torch.Tensor (batch_size, num_corr) or (num_corr,) (default: None)
    :param weight_thresh: float (default: 0.)
    :param eps: float (default: 1e-5)
    :param return_transform: bool (default: False)

    :return R: torch.Tensor (batch_size, 3, 3) or (3, 3)
    :return t: torch.Tensor (batch_size, 3) or (3,)
    :return transform: torch.Tensor (batch_size, 4, 4) or (4, 4)
    """
    if src_points.ndim == 2:
        src_points = src_points.unsqueeze(0)
        tgt_points = tgt_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_points.shape[0]
    if weights is None:
        weights = torch.ones_like(src_points[:, :, 0])
    weights = torch.where(torch.lt(weights, weight_thresh), torch.zeros_like(weights), weights)
    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)
    weights = weights.unsqueeze(2)  # (B, N, 1)

    src_centroid = torch.sum(src_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    tgt_centroid = torch.sum(tgt_points * weights, dim=1, keepdim=True)  # (B, 1, 3)
    src_points_centered = src_points - src_centroid  # (B, N, 3)
    tgt_points_centered = tgt_points - tgt_centroid  # (B, N, 3)

    H = src_points_centered.permute(0, 2, 1) @ (weights * tgt_points_centered)
    U, _, V = torch.svd(H.cpu())  # H = USV^T
    if torch.cuda.is_available():
        Ut, V = U.transpose(1, 2).cuda(), V.cuda()
        eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    else:
        Ut, V = U.transpose(1, 2), V
        eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
    eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    R = V @ eye @ Ut

    t = tgt_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
    t = t.squeeze(2)

    if return_transform:
        if torch.cuda.is_available():
            transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        else:
            transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t
        if squeeze_first:
            transform = transform.squeeze(0)
        return transform
    else:
        if squeeze_first:
            R = R.squeeze(0)
            t = t.squeeze(0)
        return R, t
