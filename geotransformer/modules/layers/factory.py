from typing import Union, Dict, Optional, Tuple

import torch.nn as nn


NORM_LAYERS = {
    'BatchNorm1d': nn.BatchNorm1d,
    'BatchNorm2d': nn.BatchNorm2d,
    'BatchNorm3d': nn.BatchNorm3d,
    'InstanceNorm1d': nn.InstanceNorm1d,
    'InstanceNorm2d': nn.InstanceNorm2d,
    'InstanceNorm3d': nn.InstanceNorm3d,
    'GroupNorm': nn.GroupNorm,
    'LayerNorm': nn.LayerNorm,
}


ACT_LAYERS = {
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'ELU': nn.ELU,
    'GELU': nn.GELU,
    'Sigmoid': nn.Sigmoid,
    'Softplus': nn.Softplus,
    'Tanh': nn.Tanh,
    'Identity': nn.Identity,
}


CONV_LAYERS = {
    'Linear': nn.Linear,
    'Conv1d': nn.Conv1d,
    'Conv2d': nn.Conv2d,
    'Conv3d': nn.Conv3d,
}


def parse_cfg(cfg: Union[str, Dict]) -> Tuple[str, Dict]:
    assert isinstance(cfg, (str, Dict)), 'Illegal cfg type: {}.'.format(type(cfg))
    if isinstance(cfg, str):
        cfg = {'type': cfg}
    else:
        cfg = cfg.copy()
    layer = cfg.pop('type')
    return layer, cfg


def build_dropout_layer(p: Optional[float], **kwargs) -> nn.Module:
    r"""Factory function for dropout layer."""
    if p is None or p == 0:
        return nn.Identity()
    else:
        return nn.Dropout(p=p, **kwargs)


def build_norm_layer(num_features, norm_cfg: Optional[Union[str, Dict]]) -> nn.Module:
    r"""Factory function for normalization layers."""
    if norm_cfg is None:
        return nn.Identity()
    layer, kwargs = parse_cfg(norm_cfg)
    assert layer in NORM_LAYERS, f'Illegal normalization: {layer}.'
    if layer == 'GroupNorm':
        kwargs['num_channels'] = num_features
    elif layer == 'LayerNorm':
        kwargs['normalized_shape'] = num_features
    else:
        kwargs['num_features'] = num_features
    return NORM_LAYERS[layer](**kwargs)


def build_act_layer(act_cfg: Optional[Union[str, Dict]]) -> nn.Module:
    r"""Factory function for activation functions."""
    if act_cfg is None:
        return nn.Identity()
    layer, kwargs = parse_cfg(act_cfg)
    assert layer in ACT_LAYERS, f'Illegal activation: {layer}.'
    if layer == 'LeakyReLU':
        if 'negative_slope' not in kwargs:
            kwargs['negative_slope'] = 0.2
    return ACT_LAYERS[layer](**kwargs)


def build_conv_layer(conv_cfg: Union[str, Dict]) -> nn.Module:
    r"""Factory function for convolution or linear layers."""
    layer, kwargs = parse_cfg(conv_cfg)
    assert layer in CONV_LAYERS, f'Illegal layer: {layer}.'
    return CONV_LAYERS[layer](**kwargs)
