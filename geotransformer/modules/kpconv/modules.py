#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Define network blocks
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional import gather, radius_gaussian, closest_pool, max_pool, global_average
from .kernel_points import load_kernels


class KPConv(nn.Module):
    def __init__(self, kernel_size, p_dim, in_channels, out_channels, KP_extent, radius,
                 fixed_kernel_points='center', KP_influence='linear', aggregation_mode='sum',
                 deformable=False, modulated=False):
        """
        Initialize parameters for KPConvDeformable.
        :param kernel_size: Number of kernel points.
        :param p_dim: dimension of the point space.
        :param in_channels: dimension of input features.
        :param out_channels: dimension of output features.
        :param KP_extent: influence radius of each kernel point.
        :param radius: radius used for kernel point init. Even for deformable, use the config.conv_radius
        :param fixed_kernel_points: fix position of certain kernel points ('none', 'center' or 'verticals').
        :param KP_influence: influence function of the kernel points ('constant', 'linear', 'gaussian').
        :param aggregation_mode: choose to sum influences, or only keep the closest ('closest', 'sum').
        :param deformable: choose deformable or not
        :param modulated: choose if kernel weights are modulated in addition to deformed
        """
        super(KPConv, self).__init__()

        # Save parameters
        self.K = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated

        # Running variable containing deformed KP distance to input points. (used in regularization loss)
        self.min_d2 = None
        self.deformed_KP = None
        self.offset_features = None

        # Initialize weights
        self.weights = nn.Parameter(torch.zeros(self.K, in_channels, out_channels))

        # Initiate weights for offsets
        if deformable:
            if modulated:
                self.offset_dim = (self.p_dim + 1) * self.K
            else:
                self.offset_dim = self.p_dim * self.K
            self.offset_conv = KPConv(
                self.K, self.p_dim, self.in_channels, self.offset_dim, KP_extent, radius,
                fixed_kernel_points=fixed_kernel_points, KP_influence=KP_influence, aggregation_mode=aggregation_mode
            )
            self.offset_bias = nn.Parameter(torch.zeros(self.offset_dim))
        else:
            self.offset_dim = None
            self.offset_conv = None
            self.offset_bias = None

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        self.register_buffer('kernel_points', self.initialize_kernel_points())

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.deformable:
            nn.init.zeros_(self.offset_bias)

    def initialize_kernel_points(self):
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """
        # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
        kernel_points = load_kernels(self.radius, self.K, dimension=self.p_dim, fixed=self.fixed_kernel_points)
        return torch.from_numpy(kernel_points).float()

    def forward(self, q_pts, s_pts, neighb_inds, x):
        ###################
        # Offset generation
        ###################
        if self.deformable:
            # Get offsets with a KPConv that only takes part of the features
            self.offset_features = self.offset_conv(q_pts, s_pts, neighb_inds, x) + self.offset_bias
            if self.modulated:
                # Get offset (in normalized scale) from features
                unscaled_offsets = self.offset_features[:, :self.p_dim * self.K]
                unscaled_offsets = unscaled_offsets.view(-1, self.K, self.p_dim)
                # Get modulations
                modulations = 2 * torch.sigmoid(self.offset_features[:, self.p_dim * self.K:])
            else:
                # Get offset (in normalized scale) from features
                unscaled_offsets = self.offset_features.view(-1, self.K, self.p_dim)
                # No modulations
                modulations = None
            # Rescale offset for this layer
            offsets = unscaled_offsets * self.KP_extent
        else:
            offsets = None
            modulations = None

        ######################
        # Deformed convolution
        ######################
        # Add a fake point in the last row for shadow neighbors
        s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + 1e6), 0)
        # Get neighbor points [n_points, n_neighbors, dim]
        neighbors = s_pts[neighb_inds, :]
        # Center every neighborhood
        neighbors = neighbors - q_pts.unsqueeze(1)
        # Apply offsets to kernel points [n_points, n_kpoints, dim]
        if self.deformable:
            self.deformed_KP = offsets + self.kernel_points
            deformed_K_points = self.deformed_KP.unsqueeze(1)
        else:
            deformed_K_points = self.kernel_points
        # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
        neighbors.unsqueeze_(2)
        differences = neighbors - deformed_K_points
        # Get the square distances [n_points, n_neighbors, n_kpoints]
        sq_distances = torch.sum(differences ** 2, dim=3)
        # Optimization by ignoring points outside a deformed KP range
        if self.deformable:
            # Save distances for loss
            self.min_d2, _ = torch.min(sq_distances, dim=1)
            # Boolean of the neighbors in range of a kernel point [n_points, n_neighbors]
            in_range = torch.any(torch.lt(sq_distances, self.KP_extent ** 2), dim=2).int()
            # New value of max neighbors
            new_max_neighb = torch.max(torch.sum(in_range, dim=1))
            # For each row of neighbors, indices of the ones that are in range [n_points, new_max_neighb]
            neighb_row_bool, neighb_row_inds = torch.topk(in_range, new_max_neighb.item(), dim=1)
            # Gather new neighbor indices [n_points, new_max_neighb]
            new_neighb_inds = neighb_inds.gather(1, neighb_row_inds)
            # Gather new distances to KP [n_points, new_max_neighb, n_kpoints]
            neighb_row_inds.unsqueeze_(2)
            neighb_row_inds = neighb_row_inds.expand(-1, -1, self.K)
            sq_distances = sq_distances.gather(1, neighb_row_inds)
            # New shadow neighbors have to point to the last shadow point
            new_neighb_inds *= neighb_row_bool
            new_neighb_inds -= (neighb_row_bool.long() - 1) * int(s_pts.shape[0] - 1)
        else:
            new_neighb_inds = neighb_inds

        # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
        if self.KP_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = torch.ones_like(sq_distances)
            all_weights = torch.transpose(all_weights, 1, 2)
        elif self.KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
            all_weights = torch.transpose(all_weights, 1, 2)
        elif self.KP_influence == 'gaussian':
            # Influence in gaussian of the distance.
            sigma = self.KP_extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = torch.transpose(all_weights, 1, 2)
        else:
            raise ValueError('Unknown influence function type (config.KP_influence)')

        # In case of closest mode, only the closest KP can influence each point
        if self.aggregation_mode == 'closest':
            neighbors_1nn = torch.argmin(sq_distances, dim=2)
            all_weights *= torch.transpose(F.one_hot(neighbors_1nn, self.K), 1, 2)
        elif self.aggregation_mode != 'sum':
            raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

        # Add a zero feature for shadow neighbors
        x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

        # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
        neighb_x = gather(x, new_neighb_inds)

        # Apply distance weights [n_points, n_kpoints, in_fdim]
        weighted_features = torch.matmul(all_weights, neighb_x)

        # Apply modulations
        if self.deformable and self.modulated:
            weighted_features *= modulations.unsqueeze(2)

        # Apply network weights [n_kpoints, n_points, out_fdim]
        weighted_features = weighted_features.permute((1, 0, 2))
        kernel_outputs = torch.matmul(weighted_features, self.weights)

        # Convolution sum [n_points, out_fdim]
        # return torch.sum(kernel_outputs, dim=0)
        output_features = torch.sum(kernel_outputs, dim=0, keepdim=False)

        # normalization term.
        neighbor_features_sum = torch.sum(neighb_x, dim=-1)
        neighbor_num = torch.sum(torch.gt(neighbor_features_sum, 0.0), dim=-1)
        neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))
        output_features = output_features / neighbor_num.unsqueeze(1)

        return output_features

    def __repr__(self):
        return 'KPConv(radius: {:.2f}, extent: {:.2f}, in_feat: {:d}, out_feat: {:d})'.format(
            self.radius, self.KP_extent, self.in_channels, self.out_channels
        )


def block_decider(block_name, radius, in_dim, out_dim, layer_ind, config):
    if block_name == 'unary':
        return UnaryBlock(in_dim, out_dim, config.normalization, config.normalization_momentum)
    elif block_name == 'last_unary':
        return LastUnaryBlock(in_dim, out_dim)
    # elif block_name in ['simple',
    #                     'simple_deformable',
    #                     'simple_invariant',
    #                     'simple_equivariant',
    #                     'simple_strided',
    #                     'simple_deformable_strided',
    #                     'simple_invariant_strided',
    #                     'simple_equivariant_strided']:
    elif 'simple' in block_name:
        return SimpleBlock(block_name, in_dim, out_dim, radius, layer_ind, config)
    # elif block_name in ['resnetb',
    #                     'resnetb_invariant',
    #                     'resnetb_equivariant',
    #                     'resnetb_deformable',
    #                     'resnetb_strided',
    #                     'resnetb_deformable_strided',
    #                     'resnetb_equivariant_strided',
    #                     'resnetb_invariant_strided']:
    elif 'resnetb' in block_name:
        return ResnetBottleneckBlock(block_name, in_dim, out_dim, radius, layer_ind, config)
    elif 'conv' in block_name:
        return ConvolutionBlock(block_name, in_dim, out_dim, radius, layer_ind, config)
    elif block_name == 'max_pool' or block_name == 'max_pool_wide':
        return MaxPoolBlock(layer_ind)
    elif block_name == 'global_average':
        return GlobalAverageBlock()
    elif block_name == 'nearest_upsample':
        return NearestUpsampleBlock(layer_ind)
    else:
        raise ValueError('Unknown block name in the architecture definition : ' + block_name)


class NormalizationBlock(nn.Module):
    def __init__(self, in_dim, method, momentum):
        """
        Initialize a normalization block. If network does not use normalization, replace with biases.
        :param in_dim: dimension input features
        :param method: normalization method to use ('batch_norm', 'instance_norm' or None)
        :param momentum: normalization momentum
        """
        super(NormalizationBlock, self).__init__()
        self.momentum = momentum
        self.method = method
        self.only_bias = self.method is None
        self.in_dim = in_dim
        if self.method == 'batch_norm':
            self.norm = nn.BatchNorm1d(in_dim, momentum=self.momentum)
        elif self.method == 'instance_norm':
            self.norm = nn.InstanceNorm1d(in_dim, momentum=self.momentum)
        elif self.method == 'group_norm':
            self.norm = nn.GroupNorm(32, in_dim)
        elif self.method is None:
            self.bias = nn.Parameter(torch.zeros(in_dim, dtype=torch.float32))
            nn.init.zeros_(self.bias)
        else:
            raise ValueError('Normalization method "{}" is not supported yet.'.format(self.method))

    def forward(self, x):
        if self.method is not None:
            x = x.unsqueeze(2)
            x = x.transpose(0, 2)
            x = self.norm(x)
            x = x.transpose(0, 2)
            return x.squeeze()
        else:
            return x + self.bias

    def __repr__(self):
        return 'NormalizationBlock(in_feat: {:d}, method: {}, momentum: {:.3f}, only_bias: {})'.format(
            self.in_dim, self.method, self.momentum, self.only_bias
        )


class UnaryBlock(nn.Module):
    def __init__(self, in_dim, out_dim, normalization, normalization_momentum, no_relu=False):
        """
        Initialize a standard unary block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param normalization: boolean indicating if we use Batch Norm
        :param normalization_momentum: Batch norm momentum
        """
        super(UnaryBlock, self).__init__()
        self.normalization_momentum = normalization_momentum
        self.normalization = normalization
        self.no_relu = no_relu
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = NormalizationBlock(out_dim, self.normalization, self.normalization_momentum)
        if not no_relu:
            self.leaky_relu = nn.LeakyReLU(0.1)
        return

    def forward(self, x, batch=None):
        x = self.mlp(x)
        x = self.norm(x)
        if not self.no_relu:
            x = self.leaky_relu(x)
        return x

    def __repr__(self):
        return 'UnaryBlock(in_feat: {:d}, out_feat: {:d}, normalization: {:s}, ReLU: {:s})'.format(
            self.in_dim, self.out_dim, self.normalization, str(not self.no_relu)
        )


class LastUnaryBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        """
        Initialize a standard last_unary block without BN, ReLU.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        """
        super(LastUnaryBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, batch=None):
        x = self.mlp(x)
        return x

    def __repr__(self):
        return 'LastUnaryBlock(in_feat: {:d}, out_feat: {:d})'.format(self.in_dim, self.out_dim)


class SimpleBlock(nn.Module):
    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config):
        """
        Initialize a simple convolution block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param config: parameters
        """
        super(SimpleBlock, self).__init__()

        # get KP_extent from current radius
        current_extent = radius * config.KP_extent / config.conv_radius

        # Get other parameters
        self.momentum = config.normalization_momentum
        self.normalization = config.normalization
        self.layer_ind = layer_ind
        self.block_name = block_name
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Define the KPConv class
        self.KPConv = KPConv(
            config.num_kernel_points, config.in_points_dim, in_dim, out_dim // 2, current_extent, radius,
            fixed_kernel_points=config.fixed_kernel_points, KP_influence=config.KP_influence,
            aggregation_mode=config.aggregation_mode, deformable='deform' in block_name, modulated=config.modulated
        )

        # Other opperations
        self.norm = NormalizationBlock(out_dim // 2, self.normalization, self.momentum)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x, batch):
        if 'strided' in self.block_name:
            q_pts = batch['points'][self.layer_ind + 1]
            s_pts = batch['points'][self.layer_ind]
            neighb_inds = batch['pools'][self.layer_ind]
        else:
            q_pts = batch['points'][self.layer_ind]
            s_pts = batch['points'][self.layer_ind]
            neighb_inds = batch['neighbors'][self.layer_ind]

        x = self.KPConv(q_pts, s_pts, neighb_inds, x)
        return self.leaky_relu(self.norm(x))


class ConvolutionBlock(nn.Module):
    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config):
        """
        Initialize a KPConv block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param config: parameters
        """
        super(ConvolutionBlock, self).__init__()

        # get KP_extent from current radius
        current_extent = radius * config.KP_extent / config.conv_radius

        # Get other parameters
        self.momentum = config.normalization_momentum
        self.normalization = config.normalization
        self.layer_ind = layer_ind
        self.block_name = block_name
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Define the KPConv class
        self.KPConv = KPConv(
            config.num_kernel_points, config.in_points_dim, in_dim, out_dim, current_extent, radius,
            fixed_kernel_points=config.fixed_kernel_points, KP_influence=config.KP_influence,
            aggregation_mode=config.aggregation_mode, deformable='deform' in block_name, modulated=config.modulated
        )

        # Other opperations
        self.norm = NormalizationBlock(out_dim, self.normalization, self.momentum)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x, batch):
        if 'strided' in self.block_name:
            q_pts = batch['points'][self.layer_ind + 1]
            s_pts = batch['points'][self.layer_ind]
            neighb_inds = batch['pools'][self.layer_ind]
        else:
            q_pts = batch['points'][self.layer_ind]
            s_pts = batch['points'][self.layer_ind]
            neighb_inds = batch['neighbors'][self.layer_ind]

        x = self.KPConv(q_pts, s_pts, neighb_inds, x)
        return self.leaky_relu(self.norm(x))


class ResnetBottleneckBlock(nn.Module):
    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config):
        """
        Initialize a resnet bottleneck block.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param config: parameters
        """
        super(ResnetBottleneckBlock, self).__init__()

        # get KP_extent from current radius
        current_extent = radius * config.KP_extent / config.conv_radius

        # Get other parameters
        self.momentum = config.normalization_momentum
        self.normalization = config.normalization
        self.block_name = block_name
        self.layer_ind = layer_ind
        self.in_dim = in_dim
        self.out_dim = out_dim

        # First downscaling mlp
        if in_dim != out_dim // 4:
            self.unary1 = UnaryBlock(in_dim, out_dim // 4, self.normalization, self.momentum)
        else:
            self.unary1 = nn.Identity()

        # KPConv block
        self.KPConv = KPConv(
            config.num_kernel_points, config.in_points_dim, out_dim // 4, out_dim // 4, current_extent, radius,
            fixed_kernel_points=config.fixed_kernel_points, KP_influence=config.KP_influence,
            aggregation_mode=config.aggregation_mode, deformable='deform' in block_name, modulated=config.modulated
        )
        self.norm_conv = NormalizationBlock(out_dim // 4, self.normalization, self.momentum)

        # Second upscaling mlp
        self.unary2 = UnaryBlock(out_dim // 4, out_dim, self.normalization, self.momentum, no_relu=True)

        # Shortcut optional mpl
        if in_dim != out_dim:
            self.unary_shortcut = UnaryBlock(in_dim, out_dim, self.normalization, self.momentum, no_relu=True)
        else:
            self.unary_shortcut = nn.Identity()

        # Other operations
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, features, batch):
        if 'strided' in self.block_name:
            q_pts = batch['points'][self.layer_ind + 1]
            s_pts = batch['points'][self.layer_ind]
            neighb_inds = batch['pools'][self.layer_ind]
        else:
            q_pts = batch['points'][self.layer_ind]
            s_pts = batch['points'][self.layer_ind]
            neighb_inds = batch['neighbors'][self.layer_ind]

        # First downscaling mlp
        x = self.unary1(features)

        # Convolution
        x = self.KPConv(q_pts, s_pts, neighb_inds, x)
        x = self.leaky_relu(self.norm_conv(x))

        # Second upscaling mlp
        x = self.unary2(x)

        # Shortcut
        if 'strided' in self.block_name:
            shortcut = max_pool(features, neighb_inds)
        else:
            shortcut = features
        shortcut = self.unary_shortcut(shortcut)

        return self.leaky_relu(x + shortcut)


class GlobalAverageBlock(nn.Module):
    def __init__(self):
        """
        Initialize a global average block.
        """
        super(GlobalAverageBlock, self).__init__()

    def forward(self, x, batch):
        return global_average(x, batch['stack_lengths'][-1])


class NearestUpsampleBlock(nn.Module):
    def __init__(self, layer_ind):
        """
        Initialize a nearest upsampling block.
        """
        super(NearestUpsampleBlock, self).__init__()
        self.layer_ind = layer_ind

    def forward(self, x, batch):
        return closest_pool(x, batch['upsamples'][self.layer_ind - 1])

    def __repr__(self):
        return 'NearestUpsampleBlock(layer: {:d} -> {:d})'.format(self.layer_ind, self.layer_ind - 1)


class MaxPoolBlock(nn.Module):
    def __init__(self, layer_ind):
        """
        Initialize a max pooling block.
        """
        super(MaxPoolBlock, self).__init__()
        self.layer_ind = layer_ind

    def forward(self, x, batch):
        return max_pool(x, batch['pools'][self.layer_ind + 1])
