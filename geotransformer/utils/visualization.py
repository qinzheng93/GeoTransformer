from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from geotransformer.utils.open3d import (
    make_open3d_axes,
    make_open3d_corr_lines,
    make_open3d_point_cloud,
)
from sklearn.manifold import TSNE
from tqdm import tqdm


def draw_point_to_node(
    points: np.ndarray,
    nodes: np.ndarray,
    point_to_node: np.ndarray,
    node_colors: Optional[np.ndarray] = None,
    window_name: str = "Point to Node Visualization",
):
    points = _to_numpy(points)
    nodes = _to_numpy(nodes)
    point_to_node = _to_numpy(point_to_node)
    node_colors = _to_numpy(node_colors) if node_colors is not None else None
    if node_colors is None:
        node_colors = np.random.rand(*nodes.shape)
    # point_colors = node_colors[point_to_node] * make_scaling_along_axis(points, alpha=0.3).reshape(-1, 1)
    point_colors = node_colors[point_to_node]
    node_colors = np.ones_like(nodes) * np.array([[1, 0, 0]])

    ncd = make_open3d_point_cloud(nodes, colors=node_colors)
    pcd = make_open3d_point_cloud(points, colors=point_colors)
    axes = make_open3d_axes()

    o3d.visualization.draw([pcd, ncd, axes], window_name=window_name)


def draw_node_correspondences(
    ref_points: np.ndarray,
    ref_nodes: np.ndarray,
    ref_point_to_node: np.ndarray,
    src_points: np.ndarray,
    src_nodes: np.ndarray,
    src_point_to_node: np.ndarray,
    node_correspondences: np.ndarray,
    ref_node_colors: Optional[np.ndarray] = None,
    src_node_colors: Optional[np.ndarray] = None,
    offsets: tuple = (0, 2, 0),
    window_name: str = "Node Correspondences Visualization",
):
    ref_points = _to_numpy(ref_points)
    ref_nodes = _to_numpy(ref_nodes)
    ref_point_to_node = _to_numpy(ref_point_to_node)
    src_points = _to_numpy(src_points)
    src_nodes = _to_numpy(src_nodes)
    src_point_to_node = _to_numpy(src_point_to_node)
    node_correspondences = _to_numpy(node_correspondences)
    ref_node_colors = _to_numpy(ref_node_colors) if ref_node_colors is not None else None
    src_node_colors = _to_numpy(src_node_colors) if src_node_colors is not None else None
    offsets = np.array(offsets).reshape(1, 3)

    src_nodes = src_nodes + offsets
    src_points = src_points + offsets

    if ref_node_colors is None:
        ref_node_colors = np.random.rand(*ref_nodes.shape)
    # src_point_colors = src_node_colors[src_point_to_node] * make_scaling_along_axis(src_points).reshape(-1, 1)
    ref_point_colors = ref_node_colors[ref_point_to_node]
    ref_node_colors = np.ones_like(ref_nodes) * np.array([[1, 0, 0]])

    if src_node_colors is None:
        src_node_colors = np.random.rand(*src_nodes.shape)
    # tgt_point_colors = tgt_node_colors[tgt_point_to_node] * make_scaling_along_axis(tgt_points).reshape(-1, 1)
    src_point_colors = src_node_colors[src_point_to_node]
    src_node_colors = np.ones_like(src_nodes) * np.array([[1, 0, 0]])

    ref_ncd = make_open3d_point_cloud(ref_nodes, colors=ref_node_colors)
    ref_pcd = make_open3d_point_cloud(ref_points, colors=ref_point_colors)
    src_ncd = make_open3d_point_cloud(src_nodes, colors=src_node_colors)
    src_pcd = make_open3d_point_cloud(src_points, colors=src_point_colors)
    corr_lines = make_open3d_corr_lines(ref_nodes, src_nodes, "pos")
    axes = make_open3d_axes(scale=0.1)

    o3d.visualization.draw(
        [ref_pcd, ref_ncd, src_pcd, src_ncd, corr_lines, axes], window_name=window_name,
    )


def get_colors_with_tsne(data: np.ndarray) -> np.ndarray:
    r"""
    Use t-SNE to project high-dimension feats to rgbd
    :param data: (N, C)
    :return colors: (N, 3)
    """
    data = _to_numpy(data)
    # Robust: check for degenerate input (constant or zero variance)
    if np.allclose(np.var(data, axis=0), 0) or np.allclose(np.ptp(data, axis=0), 0):
        return np.ones((data.shape[0], 3)) * 0.5
    tsne = TSNE(n_components=1, perplexity=40, n_iter=300, random_state=0)
    tsne_results = tsne.fit_transform(data).reshape(-1)
    tsne_min = np.min(tsne_results)
    tsne_max = np.max(tsne_results)
    # Avoid division by zero if all results are identical
    if np.isclose(tsne_max, tsne_min):
        normalized_tsne_results = np.zeros_like(tsne_results)
    else:
        normalized_tsne_results = (tsne_results - tsne_min) / (tsne_max - tsne_min)
    colors = plt.cm.Spectral(normalized_tsne_results)[:, :3]
    return colors


def write_points_to_obj(
    file_name: str,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    radius: float = 0.02,
    resolution: int = 6,
):
    points = _to_numpy(points)
    colors = _to_numpy(colors) if colors is not None else None

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    vertices = np.asarray(sphere.vertices)
    triangles = np.asarray(sphere.triangles) + 1

    v_lines = []
    f_lines = []

    num_point = points.shape[0]
    for i in tqdm(range(num_point)):
        n = i * vertices.shape[0]

        for j in range(vertices.shape[0]):
            new_vertex = points[i] + vertices[j]
            line = "v {:.6f} {:.6f} {:.6f}".format(new_vertex[0], new_vertex[1], new_vertex[2])
            if colors is not None:
                line += " {:.6f} {:.6f} {:.6f}".format(colors[i, 0], colors[i, 1], colors[i, 2])
            v_lines.append(line + "\n")

        for j in range(triangles.shape[0]):
            new_triangle = triangles[j] + n
            line = "f {} {} {}\n".format(new_triangle[0], new_triangle[1], new_triangle[2])
            f_lines.append(line)

    with open(file_name, "w") as f:
        f.writelines(v_lines)
        f.writelines(f_lines)


def convert_points_to_mesh(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    radius: float = 0.02,
    resolution: int = 6,
) -> o3d.geometry.TriangleMesh:
    points = _to_numpy(points)
    colors = _to_numpy(colors) if colors is not None else None

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    vertices = np.asarray(sphere.vertices)
    triangles = np.asarray(sphere.triangles)

    new_vertices = points[:, None, :] + vertices[None, :, :]
    if colors is not None:
        new_vertex_colors = np.broadcast_to(colors[:, None, :], new_vertices.shape)
    new_vertices = new_vertices.reshape(-1, 3)
    new_vertex_colors = new_vertex_colors.reshape(-1, 3)
    bases = np.arange(points.shape[0]) * vertices.shape[0]
    new_triangles = bases[:, None, None] + triangles[None, :, :]
    new_triangles = new_triangles.reshape(-1, 3)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    mesh.vertex_colors = o3d.utility.Vector3dVector(new_vertex_colors)
    mesh.triangles = o3d.utility.Vector3iVector(new_triangles)

    return mesh


def write_points_to_ply(
    file_name: str,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    radius: float = 0.02,
    resolution: int = 6,
):
    points = _to_numpy(points)
    colors = _to_numpy(colors) if colors is not None else None
    mesh = convert_points_to_mesh(points, colors=colors, radius=radius, resolution=resolution)
    o3d.io.write_triangle_mesh(file_name, mesh, write_vertex_normals=False)


def write_correspondences_to_obj(
    file_name: str, src_corr_points: np.ndarray, tgt_corr_points: np.ndarray
):
    src_corr_points = _to_numpy(src_corr_points)
    tgt_corr_points = _to_numpy(tgt_corr_points)
    v_lines = []
    l_lines = []

    num_corr = src_corr_points.shape[0]
    for i in tqdm(range(num_corr)):
        n = i * 2

        src_point = src_corr_points[i]
        tgt_point = tgt_corr_points[i]

        line = "v {:.6f} {:.6f} {:.6f}\n".format(src_point[0], src_point[1], src_point[2])
        v_lines.append(line)

        line = "v {:.6f} {:.6f} {:.6f}\n".format(tgt_point[0], tgt_point[1], tgt_point[2])
        v_lines.append(line)

        line = "l {} {}\n".format(n + 1, n + 2)
        l_lines.append(line)

    with open(file_name, "w") as f:
        f.writelines(v_lines)
        f.writelines(l_lines)


# Utility helpers for converting KNN patch structures (node → points)


def _to_numpy(x: Optional[Union[torch.Tensor, np.ndarray]]) -> Optional[np.ndarray]:
    """Convert torch or numpy to numpy."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


# Prepare "point-to-node" visualization (flatten KNN patches)


def build_point_to_node_visualization(
    src_points: np.ndarray, knn_indices: np.ndarray, knn_masks: Optional[np.ndarray] = None
):
    """
    Convert KNN patch representation (M nodes, K neighbors) into:

        subset_points: (C, 3)
        point_to_node: (C,)
        node_centers:  (M, 3)

    Args:
        src_points : (N, 3)
        knn_indices: (M, K)
        knn_masks  : (M, K) Optional; boolean mask

    Returns:
        subset_points (C, 3)
        point_to_node (C,)  -- each entry ∈ [0, M-1]
        node_centers (M, 3) -- node positions, derived as mean of KNN patch or valid neighbors
    """
    src_points = _to_numpy(src_points)
    knn_indices = _to_numpy(knn_indices)

    M, K = knn_indices.shape

    if knn_masks is None:
        knn_masks = np.ones((M, K), dtype=bool)
    else:
        knn_masks = _to_numpy(knn_masks).astype(bool)

    # Flatten
    flat_idx = knn_indices.reshape(-1)
    flat_mask = knn_masks.reshape(-1)
    node_ids = np.repeat(np.arange(M), K)

    # Filter
    valid_idx = flat_idx[flat_mask]
    valid_nodes = node_ids[flat_mask]

    # Build subset point cloud & mapping
    subset_points = src_points[valid_idx]  # (C, 3)
    point_to_node = valid_nodes  # (C,)

    # Node centers for visualization (use mean of valid neighbors)
    node_centers = np.zeros((M, 3), dtype=np.float32)
    for i in range(M):
        mask = knn_masks[i]
        pts = src_points[knn_indices[i][mask]]
        if len(pts) == 0:
            node_centers[i] = 0.0
        else:
            node_centers[i] = pts.mean(axis=0)

    return subset_points, point_to_node, node_centers


# Prepare node correspondence visualization (for ref/src pair)


def build_node_correspondence_visualization(
    ref_knn_points: np.ndarray,
    ref_knn_masks: np.ndarray,
    src_knn_points: np.ndarray,
    src_knn_masks: np.ndarray,
):
    """
    Prepare flattened ref/src KNN patches and node centers for visualization.

    Args:
        ref_knn_points : (P, K, 3)
        ref_knn_masks  : (P, K) or None
        src_knn_points : (P, K, 3)
        src_knn_masks  : (P, K) or None

    Returns:
        ref_points         (C_ref, 3)
        ref_point_to_node  (C_ref,)
        ref_nodes_centers  (P, 3)
        src_points         (C_src, 3)
        src_point_to_node  (C_src,)
        src_nodes_centers  (P, 3)
        node_correspondences (P, 2)  - identity mapping [0..P-1 ↔ 0..P-1]
    """

    ref_knn_points = _to_numpy(ref_knn_points)
    src_knn_points = _to_numpy(src_knn_points)

    if ref_knn_masks is None:
        ref_knn_masks = np.ones(ref_knn_points.shape[:2], dtype=bool)
    else:
        ref_knn_masks = _to_numpy(ref_knn_masks).astype(bool)

    if src_knn_masks is None:
        src_knn_masks = np.ones(src_knn_points.shape[:2], dtype=bool)
    else:
        src_knn_masks = _to_numpy(src_knn_masks).astype(bool)

    P, K, _ = ref_knn_points.shape

    # --- Flatten ref ---
    ref_flat = ref_knn_points.reshape(P * K, 3)
    ref_mask_flat = ref_knn_masks.reshape(P * K)
    ref_points = ref_flat[ref_mask_flat]  # (C_ref, 3)
    ref_point_to_node = np.repeat(np.arange(P), K)[ref_mask_flat]

    # Node centers = 0th point of each patch, or average
    ref_nodes_centers = ref_knn_points[:, 0, :]  # (P, 3)

    # --- Flatten src ---
    src_flat = src_knn_points.reshape(P * K, 3)
    src_mask_flat = src_knn_masks.reshape(P * K)
    src_points = src_flat[src_mask_flat]  # (C_src, 3)
    src_point_to_node = np.repeat(np.arange(P), K)[src_mask_flat]
    src_nodes_centers = src_knn_points[:, 0, :]

    # Node correspondences: identity P×2
    node_correspondences = np.stack(
        [np.arange(P, dtype=np.int32), np.arange(P, dtype=np.int32)], axis=1
    )

    return (
        ref_points,
        ref_point_to_node,
        ref_nodes_centers,
        src_points,
        src_point_to_node,
        src_nodes_centers,
        node_correspondences,
    )
