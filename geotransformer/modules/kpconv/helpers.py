from functools import partial

import numpy as np
import torch
import torch.utils.data

from ...cpp_wrappers.cpp_subsampling import grid_subsampling as cpp_subsampling
from ...cpp_wrappers.cpp_neighbors import radius_neighbors as cpp_neighbors


def batch_grid_subsampling_kpconv(
        points,
        batches_len,
        features=None,
        labels=None,
        sampleDl=0.1,
        max_p=0,
        verbose=0
):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    """
    if features is None and labels is None:
        s_points, s_len = cpp_subsampling.subsample_batch(
            points, batches_len, sampleDl=sampleDl, max_p=max_p, verbose=verbose
        )
        return torch.from_numpy(s_points), torch.from_numpy(s_len)
    elif labels is None:
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(
            points, batches_len, features=features, sampleDl=sampleDl, max_p=max_p, verbose=verbose
        )
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features)
    elif features is None:
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(
            points, batches_len, classes=labels, sampleDl=sampleDl, max_p=max_p, verbose=verbose
        )
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_labels)
    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(
            points, batches_len, features=features, classes=labels, sampleDl=sampleDl, max_p=max_p, verbose=verbose
        )
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features), torch.from_numpy(s_labels)


def batch_neighbors_kpconv(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Computes neighbors for a batch of queries and supports, apply radius search
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B) the list of lengths of batch elements in supports
    :param radius: float32
    :param max_neighbors: int
    :return: neighbors indices
    """
    neighbors = cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)
    if max_neighbors > 0:
        neighbors = neighbors[:, :max_neighbors]
    return torch.from_numpy(neighbors)


def generate_input_data(stacked_points, stacked_normals, stacked_lengths, config, neighborhood_limits):
    # Starting radius of convolutions
    radius_normal = config.first_subsampling_dl * config.conv_radius

    # Starting layer
    layer_blocks = []
    layer = 0

    # Lists of inputs
    input_points = []
    input_normals = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_batches_len = []

    for block_i, block in enumerate(config.architecture):
        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Get all blocks of the layer
        if not ('pool' in block or 'strided' in block):
            layer_blocks += [block]
            if block_i < len(config.architecture) - 1 and not ('upsample' in config.architecture[block_i + 1]):
                continue

        # Convolution neighbors indices
        # *****************************
        if layer_blocks:
            # Convolutions are done in this layer, compute the neighbors with the good radius
            if np.any(['deformable' in block for block in layer_blocks[:-1]]):
                radius = radius_normal * config.deform_radius / config.conv_radius
            else:
                radius = radius_normal
            conv_i = batch_neighbors_kpconv(
                stacked_points, stacked_points, stacked_lengths, stacked_lengths, radius, neighborhood_limits[layer]
            )
        else:
            # This layer only perform pooling, no neighbors required
            conv_i = torch.zeros((0, 1), dtype=torch.int64)

        # Pooling neighbors indices
        # *************************

        # If end of layer is a pooling operation
        if 'pool' in block or 'strided' in block:
            # New subsampling length
            dl = 2 * radius_normal / config.conv_radius

            # Subsampled points
            pool_p, pool_b, new_normals = batch_grid_subsampling_kpconv(stacked_points, stacked_lengths, stacked_normals, sampleDl=dl)

            # Radius of pooled neighbors
            if 'deformable' in block:
                radius = radius_normal * config.deform_radius / config.conv_radius
            else:
                radius = radius_normal

            # Subsample indices
            pool_i = batch_neighbors_kpconv(
                pool_p, stacked_points, pool_b, stacked_lengths, radius, neighborhood_limits[layer]
            )

            # Upsample indices (with the radius of the next layer to keep wanted density)
            up_i = batch_neighbors_kpconv(
                stacked_points, pool_p, stacked_lengths, pool_b, 2 * radius, neighborhood_limits[layer]
            )
        else:
            # No pooling in the end of this layer, no pooling indices required
            pool_i = torch.zeros((0, 1), dtype=torch.int64)
            pool_p = torch.zeros((0, 3), dtype=torch.float32)
            pool_b = torch.zeros((0,), dtype=torch.int64)
            up_i = torch.zeros((0, 1), dtype=torch.int64)

        # Updating input lists
        input_points += [stacked_points.float()]
        input_normals += [stacked_normals.float()]
        input_neighbors += [conv_i.long()]
        input_pools += [pool_i.long()]
        input_upsamples += [up_i.long()]
        input_batches_len += [stacked_lengths]

        # New points for next layer
        stacked_points = pool_p
        stacked_lengths = pool_b
        stacked_normals = new_normals

        # Update radius and reset blocks
        radius_normal *= 2
        layer += 1
        layer_blocks = []

    return input_points,  input_normals, input_neighbors, input_pools, input_upsamples, input_batches_len


def generate_input_data_gpu(stacked_points, stacked_lengths, config, neighborhood_limits):
    # Starting radius of convolutions
    radius_normal = config.first_subsampling_dl * config.conv_radius

    # Starting layer
    layer_blocks = []
    layer = 0

    # Lists of inputs
    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_batches_len = []

    for block_i, block in enumerate(config.architecture):
        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Get all blocks of the layer
        if not ('pool' in block or 'strided' in block):
            layer_blocks += [block]
            if block_i < len(config.architecture) - 1 and not ('upsample' in config.architecture[block_i + 1]):
                continue

        # Convolution neighbors indices
        # *****************************
        if layer_blocks:
            # Convolutions are done in this layer, compute the neighbors with the good radius
            if np.any(['deformable' in block for block in layer_blocks[:-1]]):
                radius = radius_normal * config.deform_radius / config.conv_radius
            else:
                radius = radius_normal
            conv_i = batch_neighbors_kpconv_gpu(
                stacked_points, stacked_points, stacked_lengths, stacked_lengths, radius, neighborhood_limits[layer]
            )
        else:
            # This layer only perform pooling, no neighbors required
            conv_i = torch.zeros((0, 1), dtype=torch.int64).cuda()

        # Pooling neighbors indices
        # *************************

        # If end of layer is a pooling operation
        if 'pool' in block or 'strided' in block:
            # New subsampling length
            dl = 2 * radius_normal / config.conv_radius

            # Subsampled points
            pool_p, pool_b = batch_grid_subsampling_kpconv_gpu(stacked_points, stacked_lengths, dl)

            # Radius of pooled neighbors
            if 'deformable' in block:
                radius = radius_normal * config.deform_radius / config.conv_radius
            else:
                radius = radius_normal

            # Subsample indices
            pool_i = batch_neighbors_kpconv_gpu(
                pool_p, stacked_points, pool_b, stacked_lengths, radius, neighborhood_limits[layer]
            )

            # Upsample indices (with the radius of the next layer to keep wanted density)
            up_i = batch_neighbors_kpconv_gpu(
                stacked_points, pool_p, stacked_lengths, pool_b, 2 * radius, neighborhood_limits[layer]
            )
        else:
            # No pooling in the end of this layer, no pooling indices required
            pool_i = torch.zeros((0, 1), dtype=torch.int64).cuda()
            pool_p = torch.zeros((0, 3), dtype=torch.float32).cuda()
            pool_b = torch.zeros((0,), dtype=torch.int64).cuda()
            up_i = torch.zeros((0, 1), dtype=torch.int64).cuda()

        # Updating input lists
        input_points += [stacked_points.float()]
        input_neighbors += [conv_i.long()]
        input_pools += [pool_i.long()]
        input_upsamples += [up_i.long()]
        input_batches_len += [stacked_lengths]

        # New points for next layer
        stacked_points = pool_p
        stacked_lengths = pool_b

        # Update radius and reset blocks
        radius_normal *= 2
        layer += 1
        layer_blocks = []

    return input_points, input_neighbors, input_pools, input_upsamples, input_batches_len


def calibrate_neighbors(dataset, config, collate_fn, keep_ratio=0.8, samples_threshold=2000):
    # From config parameter, compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (config.deform_radius + 1) ** 3))
    neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        data_dict = collate_fn([dataset[i]], config, neighborhood_limits=[hist_n] * config.num_layers)

        # update histogram
        counts = [torch.sum(neighbors < neighbors.shape[0], dim=1).numpy() for neighbors in data_dict['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighb_hists += np.vstack(hists)

        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold:
            break

    cumsum = np.cumsum(neighb_hists.T, axis=0)
    percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

    neighborhood_limits = percentiles

    return neighborhood_limits


def make_kpconv_dataloader(
        dataset,
        config,
        batch_size,
        num_workers,
        collate_fn,
        shuffle=False,
        neighborhood_limits=None,
        drop_last=True,
        sampler=None
):
    if neighborhood_limits is None:
        neighborhood_limits = calibrate_neighbors(dataset, config, collate_fn=collate_fn)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=partial(collate_fn, config=config, neighborhood_limits=neighborhood_limits),
        drop_last=drop_last
    )
    return dataloader, neighborhood_limits
