from geotransformer.datasets.registration.modelnet.dataset import ModelNetPairDataset
from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)


def train_valid_data_loader(cfg, distributed):
    train_dataset = ModelNetPairDataset(
        cfg.data.dataset_root,
        "train",
        num_points=cfg.data.num_points,
        voxel_size=cfg.data.voxel_size,
        rotation_magnitude=cfg.data.rotation_magnitude,
        translation_magnitude=cfg.data.translation_magnitude,
        noise_magnitude=cfg.train.noise_magnitude,
        keep_ratio=cfg.data.keep_ratio,
        crop_method=cfg.data.crop_method,
        asymmetric=cfg.data.asymmetric,
        class_indices=cfg.train.class_indices,
        deterministic=False,
        twice_sample=cfg.data.twice_sample,
        twice_transform=cfg.data.twice_transform,
        return_normals=False,
        return_occupancy=True,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
    )

    valid_dataset = ModelNetPairDataset(
        cfg.data.dataset_root,
        "val",
        num_points=cfg.data.num_points,
        voxel_size=cfg.data.voxel_size,
        rotation_magnitude=cfg.data.rotation_magnitude,
        translation_magnitude=cfg.data.translation_magnitude,
        noise_magnitude=cfg.test.noise_magnitude,
        keep_ratio=cfg.data.keep_ratio,
        crop_method=cfg.data.crop_method,
        asymmetric=cfg.data.asymmetric,
        class_indices=cfg.test.class_indices,
        deterministic=True,
        twice_sample=cfg.data.twice_sample,
        twice_transform=cfg.data.twice_transform,
        return_normals=False,
        return_occupancy=True,
    )
    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
    )

    return train_loader, valid_loader, neighbor_limits


def test_data_loader(cfg):
    train_dataset = ModelNetPairDataset(
        cfg.data.dataset_root,
        "train",
        num_points=cfg.data.num_points,
        voxel_size=cfg.data.voxel_size,
        rotation_magnitude=cfg.data.rotation_magnitude,
        translation_magnitude=cfg.data.translation_magnitude,
        noise_magnitude=cfg.train.noise_magnitude,
        keep_ratio=cfg.data.keep_ratio,
        crop_method=cfg.data.crop_method,
        asymmetric=cfg.data.asymmetric,
        class_indices=cfg.train.class_indices,
        deterministic=False,
        twice_sample=cfg.data.twice_sample,
        twice_transform=cfg.data.twice_transform,
        return_normals=False,
        return_occupancy=True,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )

    test_dataset = ModelNetPairDataset(
        cfg.data.dataset_root,
        "test",
        num_points=cfg.data.num_points,
        voxel_size=cfg.data.voxel_size,
        rotation_magnitude=cfg.data.rotation_magnitude,
        translation_magnitude=cfg.data.translation_magnitude,
        noise_magnitude=cfg.test.noise_magnitude,
        keep_ratio=cfg.data.keep_ratio,
        crop_method=cfg.data.crop_method,
        asymmetric=cfg.data.asymmetric,
        class_indices=cfg.test.class_indices,
        deterministic=True,
        twice_sample=cfg.data.twice_sample,
        twice_transform=cfg.data.twice_transform,
        return_normals=False,
        return_occupancy=True,
    )
    test_loader = build_dataloader_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
    )

    return test_loader, neighbor_limits


def run_test():
    from easydict import EasyDict as edict
    import numpy as np
    from tqdm import tqdm
    import torch

    from geotransformer.utils.torch import to_cuda
    from geotransformer.utils.open3d import make_open3d_point_cloud, draw_geometries
    from geotransformer.modules.ops import get_point_to_node_indices, pairwise_distance, apply_transform
    from config import make_cfg

    def visualize(points_f, points_c):
        pcd = make_open3d_point_cloud(points_f.detach().cpu().numpy())
        pcd.paint_uniform_color([0, 0, 1])
        ncd = make_open3d_point_cloud(points_c.detach().cpu().numpy())
        ncd.paint_uniform_color([1, 0, 0])
        draw_geometries(pcd, ncd)

    cfg = make_cfg()
    train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg, False)
    print(neighbor_limits)

    all_node_counts = []
    all_lengths_c = []
    all_lengths_f = []
    all_node_sizes = []
    all_matching_counts = []
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for i, data_dict in pbar:
        data_dict = to_cuda(data_dict)
        ref_length_c = data_dict["lengths"][-1][0].item()
        src_length_c = data_dict["lengths"][-1][1].item()
        ref_length_f = data_dict["lengths"][0][0].item()
        src_length_f = data_dict["lengths"][0][1].item()
        transform = data_dict["transform"]

        points_c = data_dict["points"][-1].detach()
        points_f = data_dict["points"][0].detach()
        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        src_points_f = apply_transform(src_points_f, transform)
        src_points_c = apply_transform(src_points_c, transform)

        all_lengths_c.append(ref_length_c)
        all_lengths_c.append(src_length_c)
        all_lengths_f.append(ref_length_f)
        all_lengths_f.append(src_length_f)

        # visualize(ref_points_f, ref_points_c)
        # visualize(src_points_f, src_points_c)

        _, ref_node_sizes = get_point_to_node_indices(ref_points_f, ref_points_c, return_counts=True)
        _, src_node_sizes = get_point_to_node_indices(src_points_f, src_points_c, return_counts=True)

        sq_dist_mat = torch.sqrt(pairwise_distance(ref_points_f, src_points_f))
        matching_mat = torch.lt(sq_dist_mat, cfg.model.ground_truth_matching_radius)
        ref_matching_counts = matching_mat.sum(dim=1)
        src_matching_counts = matching_mat.sum(dim=0)
        ref_matching_counts = ref_matching_counts[ref_matching_counts > 0]
        src_matching_counts = src_matching_counts[src_matching_counts > 0]
        all_matching_counts += ref_matching_counts.detach().cpu().numpy().tolist()
        all_matching_counts += src_matching_counts.detach().cpu().numpy().tolist()

        ref_node_sizes = ref_node_sizes[ref_node_sizes > 0]
        src_node_sizes = src_node_sizes[src_node_sizes > 0]
        all_node_sizes += ref_node_sizes.detach().cpu().numpy().tolist()
        all_node_sizes += src_node_sizes.detach().cpu().numpy().tolist()
        all_node_counts.append(ref_node_sizes.shape[0])
        all_node_counts.append(src_node_sizes.shape[0])

    print(
        "matching_counts, mean: {:.3f}, min: {}, max: {}".format(
            np.mean(all_matching_counts), np.min(all_matching_counts), np.max(all_matching_counts)
        )
    )
    print(
        "lengths_c, mean: {:.3f}, min: {}, max: {}".format(
            np.mean(all_lengths_c), np.min(all_lengths_c), np.max(all_lengths_c)
        )
    )
    print(
        "lengths_f, mean: {:.3f}, min: {}, max: {}".format(
            np.mean(all_lengths_f), np.min(all_lengths_f), np.max(all_lengths_f)
        )
    )
    print(
        "node_counts, mean: {:.3f}, min: {}, max: {}".format(
            np.mean(all_node_counts), np.min(all_node_counts), np.max(all_node_counts)
        )
    )
    print(
        "node_sizes, mean: {:.3f}, min: {}, max: {}".format(
            np.mean(all_node_sizes), np.min(all_node_sizes), np.max(all_node_sizes)
        )
    )
    print(np.percentile(all_node_sizes, 80))
    print(np.percentile(all_node_sizes, 85))
    print(np.percentile(all_node_sizes, 90))
    print(np.percentile(all_node_sizes, 95))
    print(np.percentile(all_node_sizes, 99))


if __name__ == "__main__":
    run_test()
