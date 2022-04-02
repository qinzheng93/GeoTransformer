import math
import random
from typing import Callable
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
import torch.backends.cudnn as cudnn


# Distributed Data Parallel Utilities


def all_reduce_tensor(tensor, world_size=1):
    r"""Average reduce a tensor across all workers."""
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor)
    reduced_tensor /= world_size
    return reduced_tensor


def all_reduce_tensors(x, world_size=1):
    r"""Average reduce all tensors across all workers."""
    if isinstance(x, list):
        x = [all_reduce_tensors(item, world_size=world_size) for item in x]
    elif isinstance(x, tuple):
        x = (all_reduce_tensors(item, world_size=world_size) for item in x)
    elif isinstance(x, dict):
        x = {key: all_reduce_tensors(value, world_size=world_size) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        x = all_reduce_tensor(x, world_size=world_size)
    return x


# Dataloader Utilities


def reset_seed_worker_init_fn(worker_id):
    r"""Reset seed for data loader worker."""
    seed = torch.initial_seed() % (2 ** 32)
    # print(worker_id, seed)
    np.random.seed(seed)
    random.seed(seed)


def build_dataloader(
    dataset,
    batch_size=1,
    num_workers=1,
    shuffle=None,
    collate_fn=None,
    pin_memory=False,
    drop_last=False,
    distributed=False,
):
    if distributed:
        sampler = torch.utils.data.DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = shuffle

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        worker_init_fn=reset_seed_worker_init_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return data_loader


# Common Utilities


def initialize(seed=None, cudnn_deterministic=True, autograd_anomaly_detection=False):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    if cudnn_deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False
    torch.autograd.set_detect_anomaly(autograd_anomaly_detection)


def release_cuda(x):
    r"""Release all tensors to item or numpy array."""
    if isinstance(x, list):
        x = [release_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = (release_cuda(item) for item in x)
    elif isinstance(x, dict):
        x = {key: release_cuda(value) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        if x.numel() == 1:
            x = x.item()
        else:
            x = x.detach().cpu().numpy()
    return x


def to_cuda(x):
    r"""Move all tensors to cuda."""
    if isinstance(x, list):
        x = [to_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = (to_cuda(item) for item in x)
    elif isinstance(x, dict):
        x = {key: to_cuda(value) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        x = x.cuda()
    return x


def load_weights(model, snapshot):
    r"""Load weights and check keys."""
    state_dict = torch.load(snapshot)
    model_dict = state_dict['model']
    model.load_state_dict(model_dict, strict=False)

    snapshot_keys = set(model_dict.keys())
    model_keys = set(model.model_dict().keys())
    missing_keys = model_keys - snapshot_keys
    unexpected_keys = snapshot_keys - model_keys

    return missing_keys, unexpected_keys


# Learning Rate Scheduler


class CosineAnnealingFunction(Callable):
    def __init__(self, max_epoch, eta_min=0.0):
        self.max_epoch = max_epoch
        self.eta_min = eta_min

    def __call__(self, last_epoch):
        next_epoch = last_epoch + 1
        return self.eta_min + 0.5 * (1.0 - self.eta_min) * (1.0 + math.cos(math.pi * next_epoch / self.max_epoch))


class WarmUpCosineAnnealingFunction(Callable):
    def __init__(self, total_steps, warmup_steps, eta_init=0.1, eta_min=0.1):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.normal_steps = total_steps - warmup_steps
        self.eta_init = eta_init
        self.eta_min = eta_min

    def __call__(self, last_step):
        # last_step starts from -1, which means last_steps=0 indicates the first call of lr annealing.
        next_step = last_step + 1
        if next_step < self.warmup_steps:
            return self.eta_init + (1.0 - self.eta_init) / self.warmup_steps * next_step
        else:
            if next_step > self.total_steps:
                return self.eta_min
            next_step -= self.warmup_steps
            return self.eta_min + 0.5 * (1.0 - self.eta_min) * (1 + np.cos(np.pi * next_step / self.normal_steps))


def build_warmup_cosine_lr_scheduler(optimizer, total_steps, warmup_steps, eta_init=0.1, eta_min=0.1, grad_acc_steps=1):
    total_steps //= grad_acc_steps
    warmup_steps //= grad_acc_steps
    cosine_func = WarmUpCosineAnnealingFunction(total_steps, warmup_steps, eta_init=eta_init, eta_min=eta_min)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_func)
    return scheduler
