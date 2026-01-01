import importlib
from typing import Optional, Union

import numpy as np
import torch

ext_module = importlib.import_module("geotransformer.ext")


def radius_search(
    q_points: Union[torch.Tensor, np.ndarray],
    s_points: Union[torch.Tensor, np.ndarray],
    q_lengths: Union[torch.Tensor, np.ndarray],
    s_lengths: Union[torch.Tensor, np.ndarray],
    radius: float,
    neighbor_limit: int,
    *,
    out_device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Wrapper around the C++ ext module radius_neighbors.

    - Accepts torch.Tensor or numpy-like inputs.
    - Moves inputs to CPU (float32 / int64, contiguous) before calling the C++ op.
    - The C++ op runs on CPU and returns a CPU tensor.
    - Optionally moves the result to `out_device` (e.g., GPU) before returning.

    Returns:
        neighbor_indices: LongTensor of shape (N, k') where k' == neighbor_limit (if >0)
                          or the number of neighbors returned by the ext module (if neighbor_limit<=0).
    """
    # --- Input validation / conversion to torch ---
    if not torch.is_tensor(q_points):
        q_points = torch.as_tensor(q_points, dtype=torch.float32)
    if not torch.is_tensor(s_points):
        s_points = torch.as_tensor(s_points, dtype=torch.float32)
    if not torch.is_tensor(q_lengths):
        q_lengths = torch.as_tensor(q_lengths, dtype=torch.long)
    if not torch.is_tensor(s_lengths):
        s_lengths = torch.as_tensor(s_lengths, dtype=torch.long)

    # Ensure CPU, dtype, and contiguous (C++ extension usually expects this)
    q_points_cpu = q_points.to(torch.device("cpu"), dtype=torch.float32).contiguous()
    s_points_cpu = s_points.to(torch.device("cpu"), dtype=torch.float32).contiguous()
    q_lengths_cpu = q_lengths.to(torch.device("cpu"), dtype=torch.long).contiguous()
    s_lengths_cpu = s_lengths.to(torch.device("cpu"), dtype=torch.long).contiguous()

    # --- Call into C++ extension (CPU) ---
    # ext_module.radius_neighbors is expected to return a CPU torch Tensor (LongTensor).
    neighbor_indices_cpu = ext_module.radius_neighbors(
        q_points_cpu, s_points_cpu, q_lengths_cpu, s_lengths_cpu, float(radius)
    )

    # neighbor_limit slicing (still on CPU)
    if neighbor_limit > 0:
        neighbor_indices_cpu = neighbor_indices_cpu[:, :neighbor_limit].contiguous()

    # Move to out_device if requested (e.g., GPU) — otherwise keep on CPU
    if out_device is not None:
        return neighbor_indices_cpu.to(out_device)
    return neighbor_indices_cpu
