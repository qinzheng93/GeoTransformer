import torch
import torch.nn as nn
import numpy as np

from .modules import block_decider


def make_kpfcnn_encoder(config, in_dim):
    encoder_blocks = []
    encoder_skip_dims = []
    encoder_skips = []

    layer = 0
    radius = config.first_subsampling_dl * config.conv_radius
    out_dim = config.first_feats_dim

    for block_i, block in enumerate(config.architecture):
        if ('equivariant' in block) and (not out_dim % 3 == 0):
            raise ValueError('Equivariant block but features dimension is not a factor of 3')
        if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
            encoder_skips.append(block_i)
            encoder_skip_dims.append(in_dim)
        if 'upsample' in block:
            break
        encoder_blocks.append(block_decider(block, radius, in_dim, out_dim, layer, config))
        if 'simple' in block:
            in_dim = out_dim // 2
        else:
            in_dim = out_dim
        if 'pool' in block or 'strided' in block:
            layer += 1
            radius *= 2
            out_dim *= 2

    encoder_dict = {
        'encoder_blocks': encoder_blocks,
        'encoder_skip_dims': encoder_skip_dims,
        'encoder_skips': encoder_skips,
        'layer': layer,
        'radius': radius,
        'out_dim': in_dim
    }
    return encoder_dict


def make_kpfcnn_decoder(config, encoder_dict, in_dim, final_dim):
    decoder_blocks = []
    decoder_concats = []
    decoder_output_dims = []
    decoder_outputs = []  # we want the input tensors for all layers in this list

    encoder_skip_dims = encoder_dict['encoder_skip_dims']
    layer = encoder_dict['layer']
    radius = encoder_dict['radius']
    out_dim = in_dim

    start_i = 0
    for block_i, block in enumerate(config.architecture):
        if 'upsample' in block:
            start_i = block_i
            break

    for block_i, block in enumerate(config.architecture[start_i:]):
        if 'upsample' in block:
            decoder_output_dims.append(in_dim)
            decoder_outputs.append(block_i)
        if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
            in_dim += encoder_skip_dims[layer]
            decoder_concats.append(block_i)
        if block == 'last_unary':
            out_dim = final_dim
        decoder_blocks.append(block_decider(block, radius, in_dim, out_dim, layer, config))
        in_dim = out_dim
        if 'upsample' in block:
            layer -= 1
            radius *= 0.5
            out_dim = out_dim // 2

    if 'upsample' not in config.architecture[-1]:
        decoder_output_dims.append(out_dim)
        # we don't add the block_i of the last layer to decoder_outputs because we want the output tensor of this layer
        # add block_i to decoder_outputs will output the input tensor of the layer

    decoder_output_dims.reverse()  # reverse (from fine to coarse)

    encoder_dict = {
        'decoder_blocks': decoder_blocks,
        'decoder_concats': decoder_concats,
        'decoder_output_dims': decoder_output_dims,
        'decoder_outputs': decoder_outputs,
        'layer': layer,
        'radius': radius,
        'out_dim': in_dim,
    }
    return encoder_dict


class KPEncoder(nn.Module):
    def __init__(self, encoder_dict):
        super(KPEncoder, self).__init__()
        self.encoder_blocks = nn.ModuleList(encoder_dict['encoder_blocks'])
        self.encoder_skips = encoder_dict['encoder_skips']

    def forward(self, feats, data_dict):
        skip_feats = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_feats.append(feats)
            feats = block_op(feats, data_dict)
        return feats, skip_feats


class KPDecoder(nn.Module):
    def __init__(self, decoder_dict, feature_pyramid=False):
        super(KPDecoder, self).__init__()
        self.decoder_blocks = nn.ModuleList(decoder_dict['decoder_blocks'])
        self.decoder_concats = decoder_dict['decoder_concats']
        self.decoder_outputs = decoder_dict['decoder_outputs']
        self.feature_pyramid = feature_pyramid

    def forward(self, feats, skip_feats, data_dict):
        feature_pyramid = []
        skip_layer_id = len(skip_feats) - 1
        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_outputs:
                feature_pyramid.append(feats)
            if block_i in self.decoder_concats:
                feats = torch.cat([feats, skip_feats[skip_layer_id]], dim=1)
                skip_layer_id -= 1
            feats = block_op(feats, data_dict)
        feature_pyramid.append(feats)
        if self.feature_pyramid:
            feature_pyramid.reverse()  # reverse (from fine to coarse)
            return feature_pyramid
        else:
            return feats


class KPFCNN(nn.Module):
    def __init__(self, config):
        super(KPFCNN, self).__init__()

        # Encoder part
        encoder_dict = make_kpfcnn_encoder(config, config.in_features_dim)
        self.encoder_blocks = nn.ModuleList(encoder_dict['encoder_blocks'])
        self.encoder_skips = encoder_dict['encoder_skips']

        # Decoder part
        decoder_dict = make_kpfcnn_decoder(config, encoder_dict, encoder_dict['out_dim'], config.final_feats_dim)
        self.decoder_blocks = nn.ModuleList(decoder_dict['decoder_blocks'])
        self.decoder_concats = decoder_dict['decoder_concats']

    def forward(self, batch):
        x = batch['features'].clone().detach()

        # encoder part
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)

        # decoder part
        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)

        return x
