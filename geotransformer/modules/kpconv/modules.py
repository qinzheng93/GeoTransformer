import torch
import torch.nn as nn

from geotransformer.modules.kpconv.functional import maxpool, nearest_upsample, global_avgpool, knn_interpolate
from geotransformer.modules.kpconv.kpconv import KPConv


class KNNInterpolate(nn.Module):
    def __init__(self, k, eps=1e-8):
        super(KNNInterpolate, self).__init__()
        self.k = k
        self.eps = eps

    def forward(self, s_feats, q_points, s_points, neighbor_indices):
        if self.k == 1:
            return nearest_upsample(s_feats, neighbor_indices)
        else:
            return knn_interpolate(s_feats, q_points, s_points, neighbor_indices, self.k, eps=self.eps)


class MaxPool(nn.Module):
    @staticmethod
    def forward(s_feats, neighbor_indices):
        return maxpool(s_feats, neighbor_indices)


class GlobalAvgPool(nn.Module):
    @staticmethod
    def forward(feats, lengths):
        return global_avgpool(feats, lengths)


class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels):
        r"""Initialize a group normalization block.

        Args:
            num_groups: number of groups
            num_channels: feature dimension
        """
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.norm = nn.GroupNorm(self.num_groups, self.num_channels)

    def forward(self, x):
        x = x.transpose(0, 1).unsqueeze(0)  # (N, C) -> (B, C, N)
        x = self.norm(x)
        x = x.squeeze(0).transpose(0, 1)  # (B, C, N) -> (N, C)
        return x.squeeze()


class UnaryBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group_norm, has_relu=True, bias=True, layer_norm=False):
        r"""Initialize a standard unary block with GroupNorm and LeakyReLU.

        Args:
            in_channels: dimension input features
            out_channels: dimension input features
            group_norm: number of groups in group normalization
            bias: If True, use bias
            layer_norm: If True, use LayerNorm instead of GroupNorm
        """
        super(UnaryBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_norm = group_norm
        self.mlp = nn.Linear(in_channels, out_channels, bias=bias)
        if layer_norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = GroupNorm(group_norm, out_channels)
        if has_relu:
            self.leaky_relu = nn.LeakyReLU(0.1)
        else:
            self.leaky_relu = None

    def forward(self, x):
        x = self.mlp(x)
        x = self.norm(x)
        if self.leaky_relu is not None:
            x = self.leaky_relu(x)
        return x


class LastUnaryBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        r"""Initialize a standard last_unary block without GN, ReLU.

        Args:
            in_channels: dimension input features
            out_channels: dimension input features
        """
        super(LastUnaryBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        x = self.mlp(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        radius,
        sigma,
        group_norm,
        negative_slope=0.1,
        bias=True,
        layer_norm=False,
    ):
        r"""Initialize a KPConv block with ReLU and BatchNorm.

        Args:
            in_channels: dimension input features
            out_channels: dimension input features
            kernel_size: number of kernel points
            radius: convolution radius
            sigma: influence radius of each kernel point
            group_norm: group number for GroupNorm
            negative_slope: leaky relu negative slope
            bias: If True, use bias in KPConv
            layer_norm: If True, use LayerNorm instead of GroupNorm
        """
        super(ConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.KPConv = KPConv(in_channels, out_channels, kernel_size, radius, sigma, bias=bias)
        if layer_norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = GroupNorm(group_norm, out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, s_feats, q_points, s_points, neighbor_indices):
        x = self.KPConv(s_feats, q_points, s_points, neighbor_indices)
        x = self.norm(x)
        x = self.leaky_relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        radius,
        sigma,
        group_norm,
        strided=False,
        bias=True,
        layer_norm=False,
    ):
        r"""Initialize a ResNet bottleneck block.

        Args:
            in_channels: dimension input features
            out_channels: dimension input features
            kernel_size: number of kernel points
            radius: convolution radius
            sigma: influence radius of each kernel point
            group_norm: group number for GroupNorm
            strided: strided or not
            bias: If True, use bias in KPConv
            layer_norm: If True, use LayerNorm instead of GroupNorm
        """
        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strided = strided

        mid_channels = out_channels // 4

        if in_channels != mid_channels:
            self.unary1 = UnaryBlock(in_channels, mid_channels, group_norm, bias=bias, layer_norm=layer_norm)
        else:
            self.unary1 = nn.Identity()

        self.KPConv = KPConv(mid_channels, mid_channels, kernel_size, radius, sigma, bias=bias)
        if layer_norm:
            self.norm_conv = nn.LayerNorm(mid_channels)
        else:
            self.norm_conv = GroupNorm(group_norm, mid_channels)

        self.unary2 = UnaryBlock(
            mid_channels, out_channels, group_norm, has_relu=False, bias=bias, layer_norm=layer_norm
        )

        if in_channels != out_channels:
            self.unary_shortcut = UnaryBlock(
                in_channels, out_channels, group_norm, has_relu=False, bias=bias, layer_norm=layer_norm
            )
        else:
            self.unary_shortcut = nn.Identity()

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, s_feats, q_points, s_points, neighbor_indices):
        x = self.unary1(s_feats)

        x = self.KPConv(x, q_points, s_points, neighbor_indices)
        x = self.norm_conv(x)
        x = self.leaky_relu(x)

        x = self.unary2(x)

        if self.strided:
            shortcut = maxpool(s_feats, neighbor_indices)
        else:
            shortcut = s_feats
        shortcut = self.unary_shortcut(shortcut)

        x = x + shortcut
        x = self.leaky_relu(x)

        return x
