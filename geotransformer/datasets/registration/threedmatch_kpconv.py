import os.path as osp
import pickle
import random
from functools import partial
import open3d as o3d

import torch
import torch.utils.data
import numpy as np

from ...utils.point_cloud_utils import random_sample_rotation, get_transform_from_rotation_translation
from ...modules.kpconv.helpers import generate_input_data, calibrate_neighbors


class ThreeDMatchPairKPConvDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_root,
            subset,
            max_point=30000,
            use_augmentation=True,
            noise_magnitude=0.005,
            rotation_factor=1
    ):
        super(ThreeDMatchPairKPConvDataset, self).__init__()

        self.dataset_root = dataset_root
        # self.metadata_root = osp.join(self.dataset_root, 'metadata')
        self.metadata_root = osp.join('/root/aiyang/GeoTransformer/data/3DMatch', 'metadata')
        self.data_root = osp.join(self.dataset_root, 'data', 'indoor')
        self.subset = subset
        self.use_augmentation = use_augmentation
        self.noise_magnitude = noise_magnitude
        self.rotation_factor = rotation_factor
        self.max_point = max_point

        with open(osp.join(self.metadata_root, f'{subset}.pkl'), 'rb') as f:
            self.metadata_list = pickle.load(f)

    def __len__(self):
        return len(self.metadata_list)

    def _load_point_cloud(self, file_name):
        points = torch.load(osp.join(self.data_root, file_name)).astype(np.float32)
        if self.max_point is not None and points.shape[0] > self.max_point:
            indices = np.random.permutation(points.shape[0])[:self.max_point]
            points = points[indices]
        return points

    def _augment_point_cloud(self, points0, points1, rotation, translation):
        aug_rotation = random_sample_rotation(self.rotation_factor)
        if random.random() > 0.5:
            points0 = np.matmul(points0, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
        else:
            points1 = np.matmul(points1, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)
        points0 += (np.random.rand(points0.shape[0], 3) - 0.5) * self.noise_magnitude
        points1 += (np.random.rand(points1.shape[0], 3) - 0.5) * self.noise_magnitude
        return points0, points1, rotation, translation

    def __getitem__(self, index):
        data_dict = {}
        # metadata
        metadata = self.metadata_list[index]
        data_dict['scene_name'] = metadata['scene_name']
        data_dict['frag_id0'] = metadata['frag_id0']
        data_dict['frag_id1'] = metadata['frag_id1']
        data_dict['overlap'] = metadata['overlap']

        # get transformation
        rotation = metadata['rotation']
        translation = metadata['translation']

        # get pointcloud
        points0 = self._load_point_cloud(metadata['pcd0'])
        points1 = self._load_point_cloud(metadata['pcd1'])

        if self.use_augmentation:
            points0, points1, rotation, translation = self._augment_point_cloud(points0, points1, rotation, translation)

        # get correspondence at fine level
        transform = get_transform_from_rotation_translation(rotation, translation)

        # normal
        # src_pcd, tgt_pcd = normal(npy2pcd(src_points)), normal(npy2pcd(tgt_points))
        src_pcd = o3d.geometry.PointCloud()
        tgt_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(points0)
        tgt_pcd.points = o3d.utility.Vector3dVector(points1)
        src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30),
                                 fast_normal_computation=False)
        tgt_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30),
                                 fast_normal_computation=False)
        src_pcd.orient_normals_towards_camera_location([0, 0, 0])
        tgt_pcd.orient_normals_towards_camera_location([0, 0, 0])


        data_dict['points0'] = points0.astype(np.float32)
        data_dict['points1'] = points1.astype(np.float32)
        data_dict['feats0'] = np.ones((points0.shape[0], 1), dtype=np.float32)
        data_dict['feats1'] = np.ones((points1.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)
        data_dict['normals0'] = np.array(src_pcd.normals).astype(np.float32)
        data_dict['normals1'] = np.array(tgt_pcd.normals).astype(np.float32)

        return data_dict


def threedmatch_kpconv_collate_fn(data_dicts, config, neighborhood_limits):
    new_data_dicts = []

    for data_dict in data_dicts:
        new_data_dict = {}

        new_data_dict['scene_name'] = data_dict['scene_name']
        new_data_dict['frag_id0'] = data_dict['frag_id0']
        new_data_dict['frag_id1'] = data_dict['frag_id1']
        new_data_dict['overlap'] = data_dict['overlap']
        new_data_dict['transform'] = torch.from_numpy(data_dict['transform'])

        feats = np.concatenate([data_dict['feats0'], data_dict['feats1']], axis=0)
        new_data_dict['features'] = torch.from_numpy(feats)

        points0, points1 = data_dict['points0'], data_dict['points1']
        normals0, normals1 = data_dict['normals0'], data_dict['normals1']
        points = np.concatenate([points0, points1], axis=0)
        normals = np.concatenate([normals0, normals1], axis=0)
        lengths = np.array([points0.shape[0], points1.shape[0]])
        stacked_points = torch.from_numpy(points)
        stacked_normals = torch.from_numpy(normals)
        stacked_lengths = torch.from_numpy(lengths)

        input_points, input_normals, input_neighbors, input_pools, input_upsamples, input_lengths = generate_input_data(
            stacked_points, stacked_normals, stacked_lengths, config, neighborhood_limits
        )

        new_data_dict['points'] = input_points
        new_data_dict['normals'] = input_normals
        new_data_dict['neighbors'] = input_neighbors
        new_data_dict['pools'] = input_pools
        new_data_dict['upsamples'] = input_upsamples
        new_data_dict['stack_lengths'] = input_lengths

        new_data_dicts.append(new_data_dict)

    if len(new_data_dicts) == 1:
        return new_data_dicts[0]
    else:
        return new_data_dicts


def get_dataloader(
        dataset,
        config,
        batch_size,
        num_workers,
        shuffle=False,
        neighborhood_limits=None,
        drop_last=True
):
    if neighborhood_limits is None:
        neighborhood_limits = calibrate_neighbors(dataset, config, collate_fn=threedmatch_kpconv_collate_fn)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(threedmatch_kpconv_collate_fn, config=config, neighborhood_limits=neighborhood_limits),
        drop_last=drop_last
    )
    return dataloader, neighborhood_limits
