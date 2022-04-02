import os.path as osp
from typing import Dict, Optional

import numpy as np
import torch.utils.data
import open3d as o3d
from IPython import embed

from geotransformer.utils.common import load_pickle
from geotransformer.utils.pointcloud import random_sample_transform, apply_transform, inverse_transform, regularize_normals
from geotransformer.utils.registration import compute_overlap
from geotransformer.utils.open3d import estimate_normals, voxel_downsample
from geotransformer.transforms.functional import (
    normalize_points,
    random_jitter_points,
    random_shuffle_points,
    random_sample_points,
    random_crop_point_cloud_with_plane,
    random_sample_viewpoint,
    random_crop_point_cloud_with_point,
)


class ModelNetPairDataset(torch.utils.data.Dataset):
    # fmt: off
    ALL_CATEGORIES = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain',
        'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel',
        'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool',
        'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
    ]
    ASYMMETRIC_CATEGORIES = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'car', 'chair', 'curtain', 'desk', 'door', 'dresser',
        'glass_box', 'guitar', 'keyboard', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant',
        'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'toilet', 'tv_stand', 'wardrobe', 'xbox'
    ]
    ASYMMETRIC_INDICES = [
        0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36,
        38, 39
    ]
    # fmt: on

    def __init__(
        self,
        dataset_root: str,
        subset: str,
        num_points: int = 1024,
        voxel_size: Optional[float] = None,
        rotation_magnitude: float = 45.0,
        translation_magnitude: float = 0.5,
        noise_magnitude: Optional[float] = None,
        keep_ratio: float = 0.7,
        crop_method: str = 'plane',
        asymmetric: bool = True,
        class_indices: str = 'all',
        deterministic: bool = False,
        twice_sample: bool = False,
        twice_transform: bool = False,
        return_normals: bool = True,
        return_occupancy: bool = False,
        min_overlap: Optional[float] = None,
        max_overlap: Optional[float] = None,
        estimate_normal: bool = False,
        overfitting_index: Optional[int] = None,
    ):
        super(ModelNetPairDataset, self).__init__()

        assert subset in ['train', 'val', 'test']
        assert crop_method in ['plane', 'point']

        self.dataset_root = dataset_root
        self.subset = subset

        self.num_points = num_points
        self.voxel_size = voxel_size
        self.rotation_magnitude = rotation_magnitude
        self.translation_magnitude = translation_magnitude
        self.noise_magnitude = noise_magnitude
        self.keep_ratio = keep_ratio
        self.crop_method = crop_method
        self.asymmetric = asymmetric
        self.class_indices = self.get_class_indices(class_indices, asymmetric)
        self.deterministic = deterministic
        self.twice_sample = twice_sample
        self.twice_transform = twice_transform
        self.return_normals = return_normals
        self.return_occupancy = return_occupancy
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.check_overlap = self.min_overlap is not None or self.max_overlap is not None
        self.estimate_normal = estimate_normal
        self.overfitting_index = overfitting_index

        data_list = load_pickle(osp.join(dataset_root, f'{subset}.pkl'))
        data_list = [x for x in data_list if x['label'] in self.class_indices]
        if overfitting_index is not None and deterministic:
            data_list = [data_list[overfitting_index]]
        self.data_list = data_list

    def get_class_indices(self, class_indices, asymmetric):
        r"""Generate class indices.
        'all' -> all 40 classes.
        'seen' -> first 20 classes.
        'unseen' -> last 20 classes.
        list|tuple -> unchanged.
        asymmetric -> remove symmetric classes.
        """
        if isinstance(class_indices, str):
            assert class_indices in ['all', 'seen', 'unseen']
            if class_indices == 'all':
                class_indices = list(range(40))
            elif class_indices == 'seen':
                class_indices = list(range(20))
            else:
                class_indices = list(range(20, 40))
        if asymmetric:
            class_indices = [x for x in class_indices if x in self.ASYMMETRIC_INDICES]
        return class_indices

    def __getitem__(self, index):
        if self.overfitting_index is not None:
            index = self.overfitting_index

        data_dict: Dict = self.data_list[index]
        raw_points = data_dict['points'].copy()
        raw_normals = data_dict['normals'].copy()
        label = data_dict['label']

        # set deterministic
        if self.deterministic:
            np.random.seed(index)

        # normalize raw point cloud
        raw_points = normalize_points(raw_points)

        # once sample on raw point cloud
        if not self.twice_sample:
            raw_points, raw_normals = random_sample_points(raw_points, self.num_points, normals=raw_normals)

        # split reference and source point cloud
        ref_points = raw_points.copy()
        ref_normals = raw_normals.copy()

        # twice transform
        if self.twice_transform:
            transform = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
            ref_points, ref_normals = apply_transform(ref_points, transform, normals=ref_normals)

        src_points = ref_points.copy()
        src_normals = ref_normals.copy()

        # random transform to source point cloud
        transform = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
        inv_transform = inverse_transform(transform)
        src_points, src_normals = apply_transform(src_points, inv_transform, normals=src_normals)

        raw_ref_points = ref_points
        raw_ref_normals = ref_normals
        raw_src_points = src_points
        raw_src_normals = src_normals

        while True:
            ref_points = raw_ref_points
            ref_normals = raw_ref_normals
            src_points = raw_src_points
            src_normals = raw_src_normals
            # crop
            if self.keep_ratio is not None:
                if self.crop_method == 'plane':
                    ref_points, ref_normals = random_crop_point_cloud_with_plane(
                        ref_points, keep_ratio=self.keep_ratio, normals=ref_normals
                    )
                    src_points, src_normals = random_crop_point_cloud_with_plane(
                        src_points, keep_ratio=self.keep_ratio, normals=src_normals
                    )
                else:
                    viewpoint = random_sample_viewpoint()
                    ref_points, ref_normals = random_crop_point_cloud_with_point(
                        ref_points, viewpoint=viewpoint, keep_ratio=self.keep_ratio, normals=ref_normals
                    )
                    src_points, src_normals = random_crop_point_cloud_with_point(
                        src_points, viewpoint=viewpoint, keep_ratio=self.keep_ratio, normals=src_normals
                    )

            # data check
            is_available = True
            # check overlap
            if self.check_overlap:
                overlap = compute_overlap(ref_points, src_points, transform, positive_radius=0.05)
                if self.min_overlap is not None:
                    is_available = is_available and overlap >= self.min_overlap
                if self.max_overlap is not None:
                    is_available = is_available and overlap <= self.max_overlap
            if is_available:
                break

        if self.twice_sample:
            # twice sample on both point clouds
            ref_points, ref_normals = random_sample_points(ref_points, self.num_points, normals=ref_normals)
            src_points, src_normals = random_sample_points(src_points, self.num_points, normals=src_normals)

        # random jitter
        if self.noise_magnitude is not None:
            ref_points = random_jitter_points(ref_points, scale=0.01, noise_magnitude=self.noise_magnitude)
            src_points = random_jitter_points(src_points, scale=0.01, noise_magnitude=self.noise_magnitude)

        # random shuffle
        ref_points, ref_normals = random_shuffle_points(ref_points, normals=ref_normals)
        src_points, src_normals = random_shuffle_points(src_points, normals=src_normals)

        if self.voxel_size is not None:
            # voxel downsample reference point cloud
            ref_points, ref_normals = voxel_downsample(ref_points, self.voxel_size, normals=ref_normals)
            src_points, src_normals = voxel_downsample(src_points, self.voxel_size, normals=src_normals)

        new_data_dict = {
            'raw_points': raw_points.astype(np.float32),
            'ref_points': ref_points.astype(np.float32),
            'src_points': src_points.astype(np.float32),
            'transform': transform.astype(np.float32),
            'label': int(label),
            'index': int(index),
        }

        if self.estimate_normal:
            ref_normals = estimate_normals(ref_points)
            ref_normals = regularize_normals(ref_points, ref_normals)
            src_normals = estimate_normals(src_points)
            src_normals = regularize_normals(src_points, src_normals)

        if self.return_normals:
            new_data_dict['raw_normals'] = raw_normals.astype(np.float32)
            new_data_dict['ref_normals'] = ref_normals.astype(np.float32)
            new_data_dict['src_normals'] = src_normals.astype(np.float32)

        if self.return_occupancy:
            new_data_dict['ref_feats'] = np.ones_like(ref_points[:, :1]).astype(np.float32)
            new_data_dict['src_feats'] = np.ones_like(src_points[:, :1]).astype(np.float32)

        return new_data_dict

    def __len__(self):
        return len(self.data_list)
