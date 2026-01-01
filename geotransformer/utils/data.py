from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from geotransformer.modules.ops import grid_subsample, radius_search
from geotransformer.utils.torch import build_dataloader
from torch import Tensor

# Stack mode utilities (torch-first)


def _to_tensor(x, device: Optional[torch.device]):
    """Helper: convert numpy->torch and move to device if provided."""
    if torch.is_tensor(x):
        t = x
    else:
        t = torch.from_numpy(np.ascontiguousarray(x)).float()
    if device is not None:
        t = t.to(device)
    return t


def precompute_data_stack_mode(
    points: Union[np.ndarray, Tensor],
    lengths: Union[np.ndarray, Tensor],
    num_stages: int,
    voxel_size: float,
    radius: float,
    neighbor_limits: List[int],
    device: Optional[torch.device] = None,
) -> Dict[str, List[Tensor]]:
    """
    Precompute multi-stage data for stack mode, returns everything as torch.Tensor on `device`.
    - points: stacked points (N,3)
    - lengths: (B,) lengths per sample in the stack
    """
    assert num_stages == len(neighbor_limits)

    # Convert to tensor and move to device (default to cpu if None)
    if device is None:
        device = points.device if torch.is_tensor(points) else torch.device("cpu")

    points = _to_tensor(points, device)
    lengths = _to_tensor(lengths, device)

    points_list: List[Tensor] = []
    lengths_list: List[Tensor] = []
    neighbors_list: List[Tensor] = []
    subsampling_list: List[Tensor] = []
    upsampling_list: List[Tensor] = []

    # Stage 0 -> downsample iteratively
    cur_points = points
    cur_lengths = lengths
    cur_voxel = float(voxel_size)
    for i in range(num_stages):
        points_list.append(cur_points)
        lengths_list.append(cur_lengths)
        # prepare for next stage
        if i < num_stages - 1:
            # grid_subsample is a CPU C++ op; it returns CPU tensors — move them to device
            s_points_cpu, s_lengths_cpu = grid_subsample(
                cur_points.cpu(), cur_lengths.cpu(), cur_voxel
            )
            cur_points = s_points_cpu.to(device)
            cur_lengths = s_lengths_cpu.to(device)
            del s_points_cpu, s_lengths_cpu  # Delete CPU intermediates to free RAM
            cur_voxel *= 2.0

    # Neighbors / subsampling / upsampling using radius_search (C++ CPU op)
    cur_radius = float(radius)
    for i in range(num_stages):
        P = points_list[i]
        L = lengths_list[i]
        # radius_search returns CPU tensor — keep it on CPU or move to device
        neigh_cpu = radius_search(
            P.cpu(), P.cpu(), L.cpu(), L.cpu(), cur_radius, neighbor_limits[i]
        )
        neighbors_list.append(neigh_cpu.to(device))
        del neigh_cpu  # Delete CPU intermediate

        if i < num_stages - 1:
            subP = points_list[i + 1]
            subL = lengths_list[i + 1]

            subsamp_cpu = radius_search(
                subP.cpu(), P.cpu(), subL.cpu(), L.cpu(), cur_radius, neighbor_limits[i]
            )
            subsampling_list.append(subsamp_cpu.to(device))
            del subsamp_cpu  # Delete CPU intermediate

            up_cpu = radius_search(
                P.cpu(), subP.cpu(), L.cpu(), subL.cpu(), cur_radius * 2.0, neighbor_limits[i + 1]
            )
            upsampling_list.append(up_cpu.to(device))
            del up_cpu  # Delete CPU intermediate

        cur_radius *= 2.0

    return {
        "points": points_list,
        "lengths": lengths_list,
        "neighbors": neighbors_list,
        "subsampling": subsampling_list,
        "upsampling": upsampling_list,
    }


def single_collate_fn_stack_mode(
    data_dicts: List[Dict[str, Any]],
    num_stages: int,
    voxel_size: float,
    search_radius: float,
    neighbor_limits: List[int],
    precompute_data: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    r"""Collate for single-cloud batch (stack mode). Returns tensors on `device`."""
    batch_size = len(data_dicts)
    collated_dict: Dict[str, Any] = {}

    # Gather and convert to tensors
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(np.ascontiguousarray(value)).float()
            if device is not None and isinstance(value, Tensor):
                value = value.to(device)
            collated_dict.setdefault(key, []).append(value)

    # Concatenate features and points
    normals = None
    if "normals" in collated_dict:
        normals = torch.cat(collated_dict.pop("normals"), dim=0)
        if device is not None:
            normals = normals.to(device)

    feats = torch.cat(collated_dict.pop("feats"), dim=0)
    if device is not None:
        feats = feats.to(device)

    points_list = collated_dict.pop("points")
    # compute lengths and stacked points; ensure tensors on device
    lengths = torch.LongTensor([p.shape[0] for p in points_list])
    points = torch.cat(
        [
            p.to(device)
            if isinstance(p, Tensor)
            else torch.from_numpy(np.ascontiguousarray(p)).to(device).float()
            for p in points_list
        ],
        dim=0,
    )
    lengths = lengths.to(device) if device is not None else lengths

    # If the dataset had single-sample items (batch_size == 1), preserve original keys as scalars
    if batch_size == 1:
        for key, value_list in collated_dict.items():
            collated_dict[key] = value_list[0]

    if normals is not None:
        collated_dict["normals"] = normals
    collated_dict["features"] = feats

    if precompute_data:
        input_dict = precompute_data_stack_mode(
            points, lengths, num_stages, voxel_size, search_radius, neighbor_limits, device=device
        )
        collated_dict.update(input_dict)
    else:
        collated_dict["points"] = points
        collated_dict["lengths"] = lengths

    collated_dict["batch_size"] = batch_size
    return collated_dict


def registration_collate_fn_stack_mode(
    data_dicts: List[Dict[str, Any]],
    num_stages: int,
    voxel_size: float,
    search_radius: float,
    neighbor_limits: List[int],
    precompute_data: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    r"""Collate for registration pairs (ref then src). Returns tensors on `device`."""
    batch_size = len(data_dicts)
    collated_dict: Dict[str, Any] = {}
    if device is None:
        ref_points = data_dicts[0]["ref_points"]
        device = ref_points.device if torch.is_tensor(ref_points) else torch.device("cpu")

    # Gather & to-tensor
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(np.ascontiguousarray(value)).float()
            if device is not None and isinstance(value, Tensor):
                value = value.to(device)
            collated_dict.setdefault(key, []).append(value)

    # Concatenate features: ref_feats + src_feats
    ref_feats_list = collated_dict.pop("ref_feats")
    src_feats_list = collated_dict.pop("src_feats")
    feats = torch.cat(ref_feats_list + src_feats_list, dim=0)
    if device is not None:
        feats = feats.to(device)
    del ref_feats_list, src_feats_list  # Free list memory

    # Concatenate normals: ref_normals + src_normals (only if present)
    normals = None
    if "ref_normals" in collated_dict and "src_normals" in collated_dict:
        ref_normals_list = collated_dict.pop("ref_normals")
        src_normals_list = collated_dict.pop("src_normals")
        normals = torch.cat(ref_normals_list + src_normals_list, dim=0)
        if device is not None:
            normals = normals.to(device)
        del ref_normals_list, src_normals_list  # Free list memory

    # Concatenate points: ref_points + src_points (stacked)
    ref_points_list = collated_dict.pop("ref_points")
    src_points_list = collated_dict.pop("src_points")
    points_list = ref_points_list + src_points_list
    del ref_points_list, src_points_list  # Free list memory
    lengths = torch.LongTensor([p.shape[0] for p in points_list])
    points = torch.cat(
        [
            p.to(device)
            if isinstance(p, Tensor)
            else torch.from_numpy(np.ascontiguousarray(p)).float().to(device)
            for p in points_list
        ],
        dim=0,
    )
    lengths = lengths.to(device) if device is not None else lengths

    if batch_size == 1:
        for key, value_list in collated_dict.items():
            collated_dict[key] = value_list[0]

    collated_dict["features"] = feats
    if normals is not None:
        collated_dict["normals"] = normals
    if precompute_data:
        input_dict = precompute_data_stack_mode(
            points, lengths, num_stages, voxel_size, search_radius, neighbor_limits, device=device
        )
        collated_dict.update(input_dict)
    else:
        collated_dict["points"] = points
        collated_dict["lengths"] = lengths

    collated_dict["batch_size"] = batch_size
    return collated_dict


def calibrate_neighbors_stack_mode(
    dataset: Any,
    collate_fn: Any,
    num_stages: int,
    voxel_size: float,
    search_radius: float,
    keep_ratio: float = 0.8,
    sample_threshold: int = 2000,
) -> torch.Tensor:
    """
    Compute neighbor limits per stage based on sampled dataset.
    Returns a numpy array neighbor_limits (num_stages,)
    """
    # upper bound histogram bins
    hist_n = int(np.ceil(4 / 3 * np.pi * (search_radius / voxel_size + 1) ** 3))
    neighbor_hists = np.zeros((num_stages, hist_n), dtype=np.int64)
    max_neighbor_limits = [hist_n] * num_stages
    device = (
        next(dataset.parameters()).device if hasattr(dataset, "parameters") else torch.device("cpu")
    )
    # iterate dataset samples (use collate_fn to produce precomputed neighbors)
    for i in range(len(dataset)):
        data_dict = collate_fn(
            [dataset[i]],
            num_stages,
            voxel_size,
            search_radius,
            max_neighbor_limits,
            precompute_data=True,
            device=device,
        )
        # data_dict["neighbors"] is a list of tensors (on device), convert to cpu arrays for histogram
        counts_list = []
        for neighbors in data_dict["neighbors"]:
            # neighbors shape: (N_total, k) where k is neighbor_limit; neighbors filled with sentinel (e.g., M) for missing
            # count valid neighbors per query: neighbors < neighbors.shape[1]
            # Use CPU to get numpy bincount
            counts = (
                (neighbors.to(torch.device("cpu")) < neighbors.shape[1])
                .sum(dim=1)
                .numpy()
                .astype(np.int64)
            )
            counts_list.append(counts)
        # compute hist for each stage and accumulate
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts_list]
        neighbor_hists += np.vstack(hists)

        if np.min(np.sum(neighbor_hists, axis=1)) > sample_threshold:
            break

    # cumulative sum across bins (transpose to mimic original logic)
    cum_sum = np.cumsum(neighbor_hists.T, axis=0)
    neighbor_limits = (
        torch.from_numpy(np.sum(cum_sum < (keep_ratio * cum_sum[hist_n - 1, :]), axis=0))
        .long()
        .to(device)
    )
    return neighbor_limits


def build_dataloader_stack_mode(
    dataset: Any,
    collate_fn: Any,
    num_stages: int,
    voxel_size: float,
    search_radius: float,
    neighbor_limits: List[int],
    batch_size: int = 1,
    num_workers: int = 1,
    shuffle: bool = False,
    drop_last: bool = False,
    distributed: bool = False,
    precompute_data: bool = True,
    device: Optional[torch.device] = None,
) -> Any:
    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=partial(
            collate_fn,
            num_stages=num_stages,
            voxel_size=voxel_size,
            search_radius=search_radius,
            neighbor_limits=neighbor_limits,
            precompute_data=precompute_data,
            device=device,
        ),
        drop_last=drop_last,
        distributed=distributed,
    )
    return dataloader


def validate_dataset(dataset: Any, max_workers: int = 4,) -> Tuple[Any, List[int]]:
    """Pre-validate dataset and filter out bad samples (same as before)."""
    from concurrent.futures import ThreadPoolExecutor

    valid_indices: List[int] = []
    failed_indices: List[int] = []

    def check_sample(idx: int) -> Tuple[int, bool]:
        try:
            _ = dataset[idx]
            return idx, True
        except Exception as e:
            print(f"Sample {idx} failed: {e}")
            return idx, False

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(check_sample, range(len(dataset)))
        for idx, is_valid in results:
            if is_valid:
                valid_indices.append(idx)
            else:
                failed_indices.append(idx)

    from torch.utils.data import Subset

    return Subset(dataset, valid_indices), failed_indices
