from geotransformer.datasets.registration.threedmatch_kpconv import ThreeDMatchPairKPConvDataset, get_dataloader


def train_valid_data_loader(config):
    train_dataset = ThreeDMatchPairKPConvDataset(
        config.dataset_root, 'train',
        max_point=config.train_max_num_point,
        use_augmentation=config.train_use_augmentation,
        noise_magnitude=config.train_augmentation_noise,
        rotation_factor=config.train_rotation_factor
    )
    train_data_loader, neighborhood_limits = get_dataloader(
        train_dataset, config,
        config.train_batch_size,
        config.train_num_worker,
        shuffle=False,
        neighborhood_limits=None,
        drop_last=True
    )

    valid_dataset = ThreeDMatchPairKPConvDataset(
        config.dataset_root, 'val',
        max_point=config.test_max_num_point,
        use_augmentation=False
    )
    valid_data_loader, _ = get_dataloader(
        valid_dataset, config, config.test_batch_size, config.test_num_worker,
        shuffle=False,
        neighborhood_limits=neighborhood_limits,
        drop_last=False
    )

    return train_data_loader, valid_data_loader, neighborhood_limits


def test_data_loader(config, benchmark):
    train_dataset = ThreeDMatchPairKPConvDataset(
        config.dataset_root, 'train',
        max_point=config.train_max_num_point,
        use_augmentation=config.train_use_augmentation,
        noise_magnitude=config.train_augmentation_noise,
        rotation_factor=config.train_rotation_factor
    )
    _, neighborhood_limits = get_dataloader(
        train_dataset, config,
        config.train_batch_size,
        config.train_num_worker,
        shuffle=True,
        neighborhood_limits=None,
        drop_last=True
    )

    test_dataset = ThreeDMatchPairKPConvDataset(
        config.dataset_root, benchmark,
        max_point=config.test_max_num_point,
        use_augmentation=False
    )
    test_data_loader, _ = get_dataloader(
        test_dataset, config, config.test_batch_size, config.test_num_worker,
        shuffle=False,
        neighborhood_limits=neighborhood_limits,
        drop_last=False
    )

    return test_data_loader
