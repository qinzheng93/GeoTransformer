import numpy as np
import open3d as o3d


def make_scaling_along_axis(points, axis=2, alpha=0):
    if isinstance(axis, int):
        new_scaling_axis = np.zeros(3)
        new_scaling_axis[axis] = 1
        axis = new_scaling_axis
    if not isinstance(axis, np.ndarray):
        axis = np.asarray(axis)
    axis /= np.linalg.norm(axis)
    projections = np.matmul(points, axis)
    upper = np.amax(projections)
    lower = np.amin(projections)
    scales = 1 - ((projections - lower) / (upper - lower) * (1 - alpha) + alpha)
    return scales


def make_open3d_colors(points, base_color, scaling_axis=2, scaling_alpha=0):
    if not isinstance(base_color, np.ndarray):
        base_color = np.asarray(base_color)
    colors = np.ones_like(points) * base_color
    scales = make_scaling_along_axis(points, axis=scaling_axis, alpha=scaling_alpha)
    colors = colors * scales.reshape(-1, 1)
    return colors


def make_open3d_point_cloud(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def make_open3d_registration_feature(data):
    r"""
    Make open3d registration features

    :param data: numpy.ndarray (N, C)
    :return feats: o3d.pipelines.registration.Feature
    """
    feats = o3d.pipelines.registration.Feature()
    feats.data = data.T
    return feats


def make_open3d_axes(origin=None, scale=1.):
    if origin is None:
        origin = np.zeros((1, 3))
    axis_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float) * scale
    axis_points = origin + axis_vectors
    points = np.concatenate([origin, axis_points], axis=0)
    lines = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.long)
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float)
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector(points)
    axes.lines = o3d.utility.Vector2iVector(lines)
    axes.colors = o3d.utility.Vector3dVector(colors)
    return axes


def make_open3d_corr_lines(ref_corr_points, src_corr_points, label):
    num_corr = ref_corr_points.shape[0]
    corr_points = np.concatenate([ref_corr_points, src_corr_points], axis=0)
    corr_indices = [(i, i + num_corr) for i in range(num_corr)]
    corr_lines = o3d.geometry.LineSet()
    corr_lines.points = o3d.utility.Vector3dVector(corr_points)
    corr_lines.lines = o3d.utility.Vector2iVector(corr_indices)
    if label == 'pos':
        corr_lines.paint_uniform_color(np.asarray([0., 1., 0.]))
    elif label == 'neg':
        corr_lines.paint_uniform_color(np.asarray([1., 0., 0.]))
    else:
        raise ValueError('Unsupported `label` {} for correspondences'.format(label))
    return corr_lines


def registration_with_ransac_from_feats(
        src_points,
        ref_points,
        src_feats,
        ref_feats,
        distance_threshold=0.05,
        ransac_n=3,
        num_iteration=50000,
        val_iteration=1000
):
    r"""
    Compute the transformation matrix from src_points to ref_points
    """
    src_pcd = make_open3d_point_cloud(src_points)
    ref_pcd = make_open3d_point_cloud(ref_points)
    src_feats = make_open3d_registration_feature(src_feats)
    ref_feats = make_open3d_registration_feature(ref_feats)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_pcd, ref_pcd, src_feats, ref_feats, distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                  o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(num_iteration, val_iteration)
    )

    return result.transformation


def registration_with_ransac_from_correspondences(
        src_points,
        ref_points,
        correspondences=None,
        distance_threshold=0.05,
        ransac_n=3,
        num_iteration=10000
):
    r"""
    Compute the transformation matrix from src_points to ref_points
    """
    src_pcd = make_open3d_point_cloud(src_points)
    ref_pcd = make_open3d_point_cloud(ref_points)

    if correspondences is None:
        indices = np.arange(src_points.shape[0])
        correspondences = np.stack([indices, indices], axis=1)
    correspondences = o3d.utility.Vector2iVector(correspondences)

    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        src_pcd, ref_pcd, correspondences, distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(num_iteration, num_iteration)
    )

    return result.transformation
