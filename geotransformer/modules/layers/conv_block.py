import warnings

import torch
import torch.nn as nn

from geotransformer.modules.layers.factory import build_conv_layer, build_norm_layer, build_act_layer


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        padding_mode='zeros',
        depth_multiplier=None,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        act_before_norm=False,
    ):
        r"""Conv-Norm-Act Block.

        Args:
            act_before_norm (bool=False): If True, conv-act-norm. If False, conv-norm-act.
        """
        super().__init__()

        assert conv_cfg is not None

        if isinstance(norm_cfg, str):
            norm_cfg = {'type': norm_cfg}
        if isinstance(act_cfg, str):
            act_cfg = {'type': act_cfg}

        norm_type = norm_cfg['type']
        if norm_type in ['BatchNorm', 'InstanceNorm']:
            norm_cfg['type'] = norm_type + conv_cfg[-2:]

        self.act_before_norm = act_before_norm

        bias = True
        if not self.act_before_norm:
            # conv-norm-act
            norm_type = norm_cfg['type']
            if norm_type.startswith('BatchNorm') or norm_type.startswith('InstanceNorm'):
                bias = False
        if conv_cfg == 'Linear':
            layer_cfg = {
                'type': conv_cfg,
                'in_features': in_channels,
                'out_features': out_channels,
                'bias': bias,
            }
        elif conv_cfg.startswith('SeparableConv'):
            if groups != 1:
                warnings.warn(f'`groups={groups}` is ignored when building {conv_cfg} layer.')
            layer_cfg = {
                'type': conv_cfg,
                'in_channels': in_channels,
                'out_channels': out_channels,
                'kernel_size': kernel_size,
                'stride': stride,
                'padding': padding,
                'dilation': dilation,
                'depth_multiplier': depth_multiplier,
                'bias': bias,
                'padding_mode': padding_mode,
            }
        else:
            if depth_multiplier is not None:
                warnings.warn(f'`depth_multiplier={depth_multiplier}` is ignored when building {conv_cfg} layer.')
            layer_cfg = {
                'type': conv_cfg,
                'in_channels': in_channels,
                'out_channels': out_channels,
                'kernel_size': kernel_size,
                'stride': stride,
                'padding': padding,
                'dilation': dilation,
                'groups': groups,
                'bias': bias,
                'padding_mode': padding_mode,
            }

        self.conv = build_conv_layer(layer_cfg)

        norm_layer = build_norm_layer(out_channels, norm_cfg)
        act_layer = build_act_layer(act_cfg)
        if self.act_before_norm:
            self.act = act_layer
            self.norm = norm_layer
        else:
            self.norm = norm_layer
            self.act = act_layer

    def forward(self, x):
        x = self.conv(x)
        if self.act_before_norm:
            x = self.norm(self.act(x))
        else:
            x = self.act(self.norm(x))
        return x
