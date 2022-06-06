import math
import numpy as np
import torch


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def fmat(arr):
    return np.around(arr,3)


def to_tensor(x, use_cuda):
    if use_cuda:
        return torch.tensor(x).cuda()
    else:
        return torch.tensor(x)


def gather_points(points, inds):
    '''

    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]


def square_dists(points1, points2):
    '''
    Calculate square dists between two group points
    :param points1: shape=(B, N, C)
    :param points2: shape=(B, M, C)
    :return:
    '''
    B, N, C = points1.shape
    _, M, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, N, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, M)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.clamp(dists, min=1e-8)
    return dists.float()


def ball_query(xyz, new_xyz, radius, K, rt_density=False):
    '''
    :param xyz: shape=(B, N, 3)
    :param new_xyz: shape=(B, M, 3)
    :param radius: int
    :param K: int, an upper limit samples
    :return: shape=(B, M, K)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    M = new_xyz.shape[1]
    grouped_inds = torch.arange(0, N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, M, 1)
    dists = square_dists(new_xyz, xyz)    # 点与点之间的距离
    grouped_inds[dists > radius ** 2] = N   # 距离大于r**2的设为N
    if rt_density:
        density = torch.sum(grouped_inds < N, dim=-1)
        density = density / N
    grouped_inds = torch.sort(grouped_inds, dim=-1)[0][:, :, :K]      # sort[0]为取值，对索引排序
    grouped_min_inds = grouped_inds[:, :, 0:1].repeat(1, 1, min(K, grouped_inds.size(2)))    # 取最小索引，矩阵size和inds相同
    grouped_inds[grouped_inds == N] = grouped_min_inds[grouped_inds == N]    # tensor1[tensor2]true/false矩阵保留为true的值，在grouped_min_inds中保留[grouped_inds == N]为N的值，即将grouped_inds中为N的值设为min
    if rt_density:
        return grouped_inds, density
    return grouped_inds


def sample_and_group(xyz, points, M, radius, K, use_xyz=True, rt_density=False):
    '''
    :param xyz: shape=(B, N, 3)
    :param points: shape=(B, N, C)
    :param M: int
    :param radius:float
    :param K: int
    :param use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    :return: new_xyz, shape=(B, M, 3); new_points, shape=(B, M, K, C+3);
             group_inds, shape=(B, M, K); grouped_xyz, shape=(B, M, K, 3)
    '''
    assert M < 0
    new_xyz = xyz
    if rt_density:
        grouped_inds, density = ball_query(xyz, new_xyz, radius, K,
                                           rt_density=True)
    else:
        grouped_inds = ball_query(xyz, new_xyz, radius, K, rt_density=False)   # (B, M ,K)
    grouped_xyz = gather_points(xyz, grouped_inds)   # (B, M, K, 3)
    grouped_xyz -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, min(K, grouped_inds.size(2)), 1) # 球采样后点的坐标-中心点坐标
    if points is not None:
        grouped_points = gather_points(points, grouped_inds)   # points为特征,(B, M, K, C)
        if use_xyz:
            new_points = torch.cat((grouped_xyz.float(), grouped_points.float()), dim=-1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    if rt_density:
        return new_xyz, new_points, grouped_inds, grouped_xyz, density
    return new_xyz, new_points, grouped_inds, grouped_xyz


def angle(v1: torch.Tensor, v2: torch.Tensor):
    """Compute angle between 2 vectors
    For robustness, we use the same formulation as in PPFNet, i.e.
        angle(v1, v2) = atan2(cross(v1, v2), dot(v1, v2)).
    This handles the case where one of the vectors is 0.0, since torch.atan2(0.0, 0.0)=0.0
    Args:
        v1: (B, *, 3)
        v2: (B, *, 3)
    Returns:
    """

    cross_prod = torch.stack([v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1],
                              v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2],
                              v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]], dim=-1)
    cross_prod_norm = torch.norm(cross_prod, dim=-1)
    dot_prod = torch.sum(v1 * v2, dim=-1)

    return torch.atan2(cross_prod_norm, dot_prod)
