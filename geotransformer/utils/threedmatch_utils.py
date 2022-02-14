import numpy as np
import open3d as o3d

from .point_cloud_utils import apply_transform, get_nearest_neighbor
from .registration_utils import compute_overlap
from .open3d_utils import make_open3d_point_cloud


def read_3dmatch_pose(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    lines = lines[1:]
    pose = []
    for line in lines:
        pose_row = [float(x) for x in line.strip().split()]
        pose.append(pose_row)
    pose = np.stack(pose, axis=0)
    return pose


def compute_3dmatch_overlap_and_info(ref_pcd, src_pcd, transform, voxel_size=0.006):
    ref_pcd = ref_pcd.voxel_down_sample(0.01)
    src_pcd = src_pcd.voxel_down_sample(0.01)
    ref_points = np.asarray(ref_pcd.points)
    src_points = np.asarray(src_pcd.points)

    # compute overlap
    overlap = compute_overlap(ref_points, src_points, transform, positive_radius=voxel_size * 5)

    # compute info
    src_points = apply_transform(src_points, transform)
    nn_distances, nn_indices = get_nearest_neighbor(ref_points, src_points, return_index=True)
    nn_indices = nn_indices[nn_distances < voxel_size]
    if nn_indices.shape[0] > 5000:
        nn_indices = np.random.choice(nn_indices, 5000, replace=False)
    src_corr_points = src_points[nn_indices]
    if src_corr_points.shape[0] > 0:
        g = np.zeros([src_corr_points.shape[0], 3, 6])
        g[:, :3, :3] = np.eye(3)
        g[:, 0, 4] = src_corr_points[:, 2]
        g[:, 0, 5] = -src_corr_points[:, 1]
        g[:, 1, 3] = -src_corr_points[:, 2]
        g[:, 1, 5] = src_corr_points[:, 0]
        g[:, 2, 3] = src_corr_points[:, 1]
        g[:, 2, 4] = -src_corr_points[:, 0]
        gt = g.transpose([0, 2, 1])
        gtg = np.matmul(gt, g)
        cov_matrix = gtg.sum(0)
    else:
        cov_matrix = np.zeros((6, 6))

    return overlap, cov_matrix
