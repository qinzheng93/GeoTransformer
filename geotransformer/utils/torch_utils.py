import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.data


# Distributed Data Parallel Utilities

def all_reduce_tensor(tensor, world_size=1):
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor)
    reduced_tensor /= world_size
    return reduced_tensor


def all_reduce_dict(tensor_dict, world_size=1):
    reduced_tensor_dict = {}
    for key, value in tensor_dict.items():
        reduced_tensor_dict[key] = all_reduce_tensor(value, world_size=world_size)
    return reduced_tensor_dict


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples


def get_sampler(dataset, distributed, training, batch_size=1):
    if distributed:
        if training:
            return torch.utils.data.DistributedSampler(dataset)
        else:
            return SequentialDistributedSampler(dataset, batch_size)
    else:
        return None


# Common Utilities

def reset_numpy_random_seed(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    # print(worker_id, seed)
    np.random.seed(seed)


def _to_cuda(x):
    if isinstance(x, torch.Tensor):
        return x.cuda()
    else:
        return x


def to_cuda(data_dict):
    for key, value in data_dict.items():
        if isinstance(value, list):
            data_dict[key] = [_to_cuda(x) for x in value]
        else:
            data_dict[key] = _to_cuda(value)
    return data_dict


def torch_safe_divide(a, b):
    if b.item() != 0:
        return a / b
    else:
        return torch.zeros_like(a)


def random_sample_from_scores(scores, size, replace=False):
    r"""
    Random sample with `scores` as probability.

    :param scores: torch.Tensor (N,)
    :param size: int
    :param replace: bool
    :return sel_indices: torch.LongTensor (size,)
    """
    probs = scores / scores.sum()
    probs = probs.detach().cpu().numpy()
    indices = np.arange(scores.shape[0])
    sel_indices = np.random.choice(indices, size=size, p=probs, replace=replace)
    sel_indices = torch.from_numpy(sel_indices).cuda()
    return sel_indices


def index_select(data, index, dim):
    r"""
    Returns a tensor `output` which indexes the `data` tensor along dimension `dim` using the entries in `index`
    which is a `LongTensor`.

    Different from `torch.index_select`, `index` does not has to be 1-D.
    The `dim`-th dimension of `data` will be expanded to the number of dimensions in `index`.

    For example, suppose the shape `data` is $(a_0, a_1, ..., a_{n-1})$, the shape of `index` is
    $(b_0, b_1, ..., b_{m-1})$, and `dim` is $i$, then `output` is $(n+m-1)$-d tensor, whose shape is
    $(a_0, ..., a_{i-1}, b_0, b_1, ..., b_{m-1}, a_{i+1}, ..., a_{n-1})$.

    :param data: torch.Tensor, (a_0, a_1, ..., a_{n-1})
    :param index: torch.LongTensor, (b_0, b_1, ..., b_{m-1})
    :param dim: int
    :return output: torch.Tensor, (a_0, ..., a_{dim-1}, b_0, ..., b_{m-1}, a_{dim+1}, ..., a_{n-1})
    """
    output = data.index_select(dim, index.view(-1))

    if index.ndim > 1:
        output_shape = data.shape[:dim] + index.shape + data.shape[dim:][1:]
        output = output.view(*output_shape)

    return output


# Learning Rate Scheduler

class CosineAnnealingFunction:
    def __init__(self, max_epoch, eta_min=0):
        self.max_epoch = max_epoch
        self.eta_min = eta_min

    def __call__(self, last_epoch):
        return self.eta_min + (1 - self.eta_min) * (1 + math.cos(math.pi * last_epoch / self.max_epoch)) / 2


# Modules

class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, eps=0.1):
        super(SmoothCrossEntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, preds, labels):
        device = preds.device
        one_hot = torch.zeros_like(preds).to(device).duplicate_removal(1, labels.unsqueeze(1), 1)
        labels = one_hot * (1 - self.eps) + self.eps / preds.shape[1]
        log_probs = F.log_softmax(preds, dim=1)
        loss = -(labels * log_probs).sum(dim=1).mean()
        return loss


class MonteCarloDropout(nn.Module):
    def __init__(self, p=0.1):
        super(MonteCarloDropout, self).__init__()
        self.p = p

    def forward(self, x):
        out = nn.functional.dropout(x, p=self.p, training=True)
        return out


def get_activation(activation, **kwargs):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'lrelu':
        if 'negative_slope' in kwargs:
            negative_slope = kwargs['negative_slope']
        else:
            negative_slope = 0.01
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
    elif activation == 'elu':
        return nn.ELU(inplace=True)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'gelu':
        return nn.GELU()
    else:
        raise RuntimeError('Activation function {} is not supported.'.format(activation))


def get_dropout(p, monte_carlo_dropout=False):
    if p is not None and p > 0:
        if monte_carlo_dropout:
            return MonteCarloDropout(p)
        else:
            return nn.Dropout(p)
    else:
        return None


class ConvBlock1d(nn.Sequential):
    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=None,
                 batch_norm=True,
                 batch_norm_after_activation=False,
                 activation='relu',
                 dropout=None,
                 monte_carlo_dropout=False,
                 **kwargs):
        super(ConvBlock1d, self).__init__()
        if bias is None:
            bias = not batch_norm
        layers = []
        layers.append(('conv', nn.Conv1d(input_dim,
                                         output_dim,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=bias)))
        if batch_norm:
            layers.append(('bn', nn.BatchNorm1d(output_dim)))
        if activation is not None:
            layers.append((activation, get_activation(activation, **kwargs)))
        if batch_norm and activation is not None and batch_norm_after_activation:
            layers[-2], layers[-1] = layers[-1], layers[-2]
        if dropout is not None:
            layers.append(('dp', get_dropout(dropout, monte_carlo_dropout=monte_carlo_dropout)))
        for name, module in layers:
            self.add_module(name, module)
    
    def forward(self, inputs):
        return super(ConvBlock1d, self).forward(inputs)


class ConvBlock2d(nn.Sequential):
    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=None,
                 batch_norm=True,
                 batch_norm_after_activation=False,
                 activation='relu',
                 dropout=None,
                 monte_carlo_dropout=False,
                 **kwargs):
        super(ConvBlock2d, self).__init__()
        if bias is None:
            bias = not batch_norm
        layers = []
        layers.append(('conv', nn.Conv2d(input_dim,
                                         output_dim,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=bias)))
        if batch_norm:
            layers.append(('bn', nn.BatchNorm2d(output_dim)))
        if activation is not None:
            layers.append((activation, get_activation(activation, **kwargs)))
        if batch_norm and activation is not None and batch_norm_after_activation:
            layers[-2], layers[-1] = layers[-1], layers[-2]
        if dropout is not None:
            layers.append(('dp', get_dropout(dropout, monte_carlo_dropout=monte_carlo_dropout)))
        for name, module in layers:
            self.add_module(name, module)

    def forward(self, inputs):
        return super(ConvBlock2d, self).forward(inputs)


class LinearBlock(nn.Sequential):
    def __init__(self,
                 input_dim,
                 output_dim,
                 bias=None,
                 batch_norm=True,
                 batch_norm_after_activation=False,
                 activation='relu',
                 dropout=None,
                 monte_carlo_dropout=False,
                 **kwargs):
        super(LinearBlock, self).__init__()
        if bias is None:
            bias = not batch_norm
        layers = []
        layers.append(('fc', nn.Linear(input_dim, output_dim, bias=bias)))
        if batch_norm:
            layers.append(('bn', nn.BatchNorm1d(output_dim)))
        if activation is not None:
            layers.append((activation, get_activation(activation, **kwargs)))
        if batch_norm and activation is not None and batch_norm_after_activation:
            layers[-2], layers[-1] = layers[-1], layers[-2]
        if dropout is not None:
            layers.append(('dp', get_dropout(dropout, monte_carlo_dropout=monte_carlo_dropout)))
        for name, module in layers:
            self.add_module(name, module)

    def forward(self, inputs):
        return super(LinearBlock, self).forward(inputs)


def create_conv1d_blocks(input_dim,
                         output_dims,
                         kernel_size,
                         stride=1,
                         padding=0,
                         dilation=1,
                         groups=1,
                         bias=None,
                         batch_norm=True,
                         batch_norm_after_activation=False,
                         activation='relu',
                         dropout=None,
                         monte_carlo_dropout=False,
                         start_index=1,
                         **kwargs):
    r"""
    Create a list of ConvBlock1d. The name of the i-th ConvBlock1d is `conv{i+start_index}`.

    :param input_dim: int
        The number of the input channels.
    :param output_dims: list of int or int
        If `output_dims` is a list of int, it represents the numbers of the output channels in each ConvBlock1d.
        If `output_dims` is a int, it means there is only one ConvBlock1d.
    :param kernel_size: int
        The kernel size in convolution.
    :param stride: int
        The stride in convolution.
    :param padding: int
        The padding in convolution.
    :param dilation: int
        The dilation in convolution.
    :param groups: int
        The groups in convolution.
    :param bias: bool
        Whether bias is used or not. If None, bias is set according to batch_norm.
    :param batch_norm: bool
        Whether batch normalization is used or not.
    :param batch_norm_after_activation: bool
        If True, every ConvBlock1d is in the order of [Conv1d, Activation_Fn, BatchNorm1d].
        If False, every ConvBlock1d is in the order of [Conv1d, BatchNorm1d, Activation_Fn].
    :param activation: str
        The name of the activation function in each ConvBlock1d.
    :param dropout: None or float
        If None, no dropout is used.
        If a float, the dropout probability in each ConvBlock1d. The Dropout is used at the end of each ConvBlock1d.
    :param monte_carlo_dropout: bool
        If Monte Carlo dropout is used.
    :param start_index: int
        The index used in the name of the first ConvBlock1d.
    """
    if isinstance(output_dims, int):
        output_dims = [output_dims]
    layers = []
    for i, output_dim in enumerate(output_dims):
        layers.append(('conv{}'.format(start_index + i),
                       ConvBlock1d(input_dim,
                                   output_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=groups,
                                   bias=bias,
                                   batch_norm=batch_norm,
                                   batch_norm_after_activation=batch_norm_after_activation,
                                   activation=activation,
                                   dropout=dropout,
                                   monte_carlo_dropout=monte_carlo_dropout,
                                   **kwargs)))
        input_dim = output_dim
    return layers


def create_conv2d_blocks(input_dim,
                         output_dims,
                         kernel_size,
                         stride=1,
                         padding=0,
                         dilation=1,
                         groups=1,
                         bias=None,
                         batch_norm=True,
                         batch_norm_after_activation=False,
                         activation='relu',
                         dropout=None,
                         monte_carlo_dropout=False,
                         start_index=1,
                         **kwargs):
    r"""
    Create a list of ConvBlock2d. The name of the i-th ConvBlock2d is `conv{i+start_index}`.

    :param input_dim: int
        The number of the input channels.
    :param output_dims: list of int or int
        If `output_dims` is a list of int, it represents the numbers of the output channels in each ConvBlock2d.
        If `output_dims` is a int, it means there is only one ConvBlock2d.
    :param kernel_size: int
        The kernel size in convolution.
    :param stride: int
        The stride in convolution.
    :param padding: int
        The padding in convolution.
    :param dilation: int
        The dilation in convolution.
    :param groups: int
        The groups in convolution.
    :param bias: bool
        Whether bias is used or not. If None, bias is set according to batch_norm.
    :param batch_norm: bool
        Whether batch normalization is used or not.
    :param batch_norm_after_activation: bool
        If True, every ConvBlock2d is in the order of [Conv2d, Activation_Fn, BatchNorm2d].
        If False, every ConvBlock2d is in the order of [Conv2d, BatchNorm2d, Activation_Fn].
    :param activation: str
        The name of the activation function in each ConvBlock2d.
    :param dropout: None or float
        If None, no dropout is used.
        If a float, the dropout probability in each ConvBlock2d. The Dropout is used at the end of each ConvBlock2d.
    :param monte_carlo_dropout: bool
        If Monte Carlo dropout is used.
    :param start_index: int
        The index used in the name of the first ConvBlock2d.
    """
    if isinstance(output_dims, int):
        output_dims = [output_dims]
    layers = []
    for i, output_dim in enumerate(output_dims):
        layers.append(('conv{}'.format(start_index + i),
                       ConvBlock2d(input_dim,
                                   output_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=groups,
                                   bias=bias,
                                   batch_norm=batch_norm,
                                   batch_norm_after_activation=batch_norm_after_activation,
                                   activation=activation,
                                   dropout=dropout,
                                   monte_carlo_dropout=monte_carlo_dropout,
                                   **kwargs)))
        input_dim = output_dim
    return layers


def create_linear_blocks(input_dim,
                         output_dims,
                         bias=None,
                         batch_norm=True,
                         batch_norm_after_activation=False,
                         activation='relu',
                         dropout=None,
                         monte_carlo_dropout=False,
                         start_index=1,
                         **kwargs):
    r"""
    Create a list of LinearBlock. The name of the i-th LinearBlock is `conv{i+start_index}`.

    :param input_dim: int
        The number of the input channels.
    :param output_dims: list of int or int
        If `output_dims` is a list of int, it represents the numbers of the output channels in each LinearBlock.
        If `output_dims` is a int, it means there is only one LinearBlock.
    :param bias: bool
        Whether bias is used or not. If None, bias is set according to batch_norm.
    :param batch_norm: bool
        Whether batch normalization is used or not.
    :param batch_norm_after_activation: bool
        If True, every LinearBlock is in the order of [Conv2d, Activation_Fn, BatchNorm2d].
        If False, every LinearBlock is in the order of [Conv2d, BatchNorm2d, Activation_Fn].
    :param activation: str
        The name of the activation function in each LinearBlock.
    :param dropout: None or float
        If None, no dropout is used.
        If a float, the dropout probability in each LinearBlock. The Dropout is used at the end of each LinearBlock.
    :param monte_carlo_dropout: bool
        If Monte Carlo dropout is used.
    :param start_index: int
        The index used in the name of the first LinearBlock.
    """
    if isinstance(output_dims, int):
        output_dims = [output_dims]
    layers = []
    for i, output_dim in enumerate(output_dims):
        layers.append(('fc{}'.format(start_index + i),
                       LinearBlock(input_dim,
                                   output_dim,
                                   bias=bias,
                                   batch_norm=batch_norm,
                                   batch_norm_after_activation=batch_norm_after_activation,
                                   activation=activation,
                                   dropout=dropout,
                                   monte_carlo_dropout=monte_carlo_dropout,
                                   **kwargs)))
        input_dim = output_dim
    return layers


class DepthwiseConv1d(nn.Sequential):
    def __init__(self,
                 input_dim,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 depth_multiplier=1,
                 bias=None,
                 batch_norm=True,
                 batch_norm_after_activation=False,
                 activation='relu',
                 dropout=None,
                 **kwargs):
        super(DepthwiseConv1d, self).__init__()

        if not isinstance(depth_multiplier, int) or depth_multiplier <= 0:
            raise ValueError('`depth_multiplier` ({}) must be a positive integer.'.format(depth_multiplier))

        if bias is None:
            bias = not batch_norm
        output_dim = input_dim * depth_multiplier

        layers = []
        layers.append(('conv', nn.Conv1d(input_dim,
                                         output_dim,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=input_dim,
                                         bias=bias)))
        if batch_norm:
            layers.append(('bn', nn.BatchNorm1d(output_dim)))
        if activation is not None:
            layers.append((activation, get_activation(activation, **kwargs)))
        if batch_norm and activation is not None and batch_norm_after_activation:
            layers[-2], layers[-1] = layers[-1], layers[-2]
        if dropout is not None:
            layers.append(('dp', nn.Dropout(dropout)))
        for name, module in layers:
            self.add_module(name, module)

    def forward(self, inputs):
        return super(DepthwiseConv1d, self).forward(inputs)


class DepthwiseConv2d(nn.Sequential):
    def __init__(self,
                 input_dim,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 depth_multiplier=1,
                 bias=None,
                 batch_norm=True,
                 batch_norm_after_activation=False,
                 activation='relu',
                 dropout=None,
                 **kwargs):
        super(DepthwiseConv2d, self).__init__()

        if not isinstance(depth_multiplier, int) or depth_multiplier <= 0:
            raise ValueError('`depth_multiplier` ({}) must be a positive integer.'.format(depth_multiplier))

        if bias is None:
            bias = not batch_norm
        output_dim = input_dim * depth_multiplier

        layers = []
        layers.append(('conv', nn.Conv2d(input_dim,
                                         output_dim,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=input_dim,
                                         bias=bias)))
        if batch_norm:
            layers.append(('bn', nn.BatchNorm2d(output_dim)))
        if activation is not None:
            layers.append((activation, get_activation(activation, **kwargs)))
        if batch_norm and activation is not None and batch_norm_after_activation:
            layers[-2], layers[-1] = layers[-1], layers[-2]
        if dropout is not None:
            layers.append(('dp', nn.Dropout(dropout)))
        for name, module in layers:
            self.add_module(name, module)

    def forward(self, inputs):
        return super(DepthwiseConv2d, self).forward(inputs)


class SeparableConv1d(nn.Sequential):
    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 depth_multiplier=1,
                 bias=None,
                 batch_norm=True,
                 batch_norm_after_activation=False,
                 activation='relu',
                 dropout=None,
                 **kwargs):
        super(SeparableConv1d, self).__init__()

        if not isinstance(depth_multiplier, int) or depth_multiplier <= 0:
            raise ValueError('`depth_multiplier` ({}) must be a positive integer.'.format(depth_multiplier))

        if bias is None:
            bias = not batch_norm
        hidden_dim = input_dim * depth_multiplier

        layers = []
        layers.append(('dwconv', nn.Conv1d(input_dim,
                                           hidden_dim,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           dilation=dilation,
                                           groups=input_dim)))
        layers.append(('pwconv', nn.Conv1d(hidden_dim, output_dim, kernel_size=1, bias=bias)))
        if batch_norm:
            layers.append(('bn', nn.BatchNorm1d(output_dim)))
        if activation is not None:
            layers.append((activation, get_activation(activation, **kwargs)))
        if batch_norm and activation is not None and batch_norm_after_activation:
            layers[-2], layers[-1] = layers[-1], layers[-2]
        if dropout is not None:
            layers.append(('dp', nn.Dropout(dropout)))
        for name, module in layers:
            self.add_module(name, module)

    def forward(self, inputs):
        return super(SeparableConv1d, self).forward(inputs)


class SeparableConv2d(nn.Sequential):
    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 depth_multiplier=1,
                 bias=None,
                 batch_norm=True,
                 batch_norm_after_activation=False,
                 activation='relu',
                 dropout=None,
                 **kwargs):
        super(SeparableConv2d, self).__init__()

        if not isinstance(depth_multiplier, int) or depth_multiplier <= 0:
            raise ValueError('`depth_multiplier` ({}) must be a positive integer.'.format(depth_multiplier))

        if bias is None:
            bias = not batch_norm
        hidden_dim = input_dim * depth_multiplier

        layers = []
        layers.append(('dwconv', nn.Conv2d(input_dim,
                                           hidden_dim,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           dilation=dilation,
                                           groups=input_dim)))
        layers.append(('pwconv', nn.Conv2d(hidden_dim, output_dim, kernel_size=1, bias=bias)))
        if batch_norm:
            layers.append(('bn', nn.BatchNorm2d(output_dim)))
        if activation is not None:
            layers.append((activation, get_activation(activation, **kwargs)))
        if batch_norm and activation is not None and batch_norm_after_activation:
            layers[-2], layers[-1] = layers[-1], layers[-2]
        if dropout is not None:
            layers.append(('dp', nn.Dropout(dropout)))
        for name, module in layers:
            self.add_module(name, module)

    def forward(self, inputs):
        return super(SeparableConv2d, self).forward(inputs)
