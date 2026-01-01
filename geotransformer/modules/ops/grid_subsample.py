import importlib
from typing import Optional, Tuple, Union

import numpy as np
import torch

ext_module = importlib.import_module("geotransformer.ext")


def grid_subsample(
    points: Union[torch.Tensor, np.ndarray],
    lengths: Union[torch.Tensor, np.ndarray],
    voxel_size: float,
    *,
    out_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper around the C++ ext grid_subsampling implementation.

    - Accepts torch.Tensor or numpy-like inputs.
    - Moves inputs to CPU (float32 / int64, contiguous) before calling the C++ op.
    - The C++ op runs on CPU and returns CPU tensors (s_points, s_lengths).
    - Optionally moves outputs to `out_device` before returning.
    """
    if not torch.is_tensor(points):
        points = torch.as_tensor(points, dtype=torch.float32)
    if not torch.is_tensor(lengths):
        lengths = torch.as_tensor(lengths, dtype=torch.long)

    points_cpu = points.to(torch.device("cpu"), dtype=torch.float32).contiguous()
    lengths_cpu = lengths.to(torch.device("cpu"), dtype=torch.long).contiguous()

    s_points_cpu, s_lengths_cpu = ext_module.grid_subsampling(
        points_cpu, lengths_cpu, float(voxel_size)
    )

    if out_device is not None:
        return s_points_cpu.to(out_device), s_lengths_cpu.to(out_device)
    return s_points_cpu, s_lengths_cpu
