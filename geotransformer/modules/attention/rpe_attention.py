# Attention layers for tensor shape (B, N, C).
# Implemented with `nn.Linear` and `nn.LayerNorm` (with affine).

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.torch_utils import get_dropout
from .vanilla_attention import AttentionOutput, TransformerLayer


class GeoMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head, dropout=0.1):
        super(GeoMultiHeadAttention, self).__init__()
        if d_model % num_head != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_head` ({}).'.format(d_model, num_head))

        self.d_model = d_model
        self.num_head = num_head
        self.d_model_per_head = d_model // num_head

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        self.proj_p = nn.Linear(self.d_model, self.d_model)

        self.dropout = get_dropout(dropout)

    def _transpose_for_scores(self, x):
        x = x.view(x.shape[0], x.shape[1], self.num_head, self.d_model_per_head)
        x = x.permute(0, 2, 1, 3)
        return x

    def _transpose_rpe_for_scores(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], self.num_head, self.d_model_per_head)
        x = x.permute(0, 3, 1, 2, 4)
        return x

    def forward(self, input_q, input_k, input_v, input_p, key_masks=None):
        r"""
        :param input_q: torch.Tensor (B, N, C)
        :param input_k: torch.Tensor (B, M, C)
        :param input_v: torch.Tensor (B, M, C)
        :param input_p: torch.Tensor (B, N, M, C), relative positional embedding
        :param key_masks: torch.Tensor (B, M), True if ignored, False if preserved
        :return hidden_states: torch.Tensor (B, C, N)
        """
        endpoints = {}

        q = self.proj_q(input_q)
        k = self.proj_k(input_k)
        v = self.proj_v(input_v)
        p = self.proj_p(input_p)

        q = self._transpose_for_scores(q)
        k = self._transpose_for_scores(k)
        v = self._transpose_for_scores(v)
        p = self._transpose_rpe_for_scores(p)

        attention_scores_p = torch.matmul(q.unsqueeze(3), p.transpose(-1, -2)).squeeze(3)
        attention_scores_e = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores_e + attention_scores_p
        attention_scores = attention_scores / self.d_model_per_head ** 0.5
        if key_masks is not None:
            attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1), float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)

        endpoints['attention_scores'] = attention_scores

        if self.dropout is not None:
            attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)

        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], self.d_model)

        return hidden_states, endpoints


class GeoAttentionLayer(nn.Module):
    def __init__(self, d_model, num_head, dropout=0.1):
        super(GeoAttentionLayer, self).__init__()
        self.attention = GeoMultiHeadAttention(d_model, num_head, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = get_dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_states, memory_states, position_states, memory_masks=None):
        hidden_states, endpoints = self.attention(
            input_states, memory_states, memory_states, position_states, key_masks=memory_masks
        )
        hidden_states = self.linear(hidden_states)
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, endpoints


class GeoTransformerLayer(nn.Module):
    def __init__(self, d_model, num_head, dropout=0.1, activation_fn='gelu', **kwargs):
        super(GeoTransformerLayer, self).__init__()
        self.attention = GeoAttentionLayer(d_model, num_head, dropout=dropout)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn, **kwargs)

    def forward(self, input_states, memory_states, position_states, memory_masks=None):
        hidden_states, endpoints = self.attention(
            input_states, memory_states, position_states, memory_masks=memory_masks
        )
        output_states = self.output(hidden_states)
        return output_states, endpoints


class GeometricTransformer(nn.Module):
    def __init__(self, blocks, d_model, num_head, dropout=0.1, activation_fn='gelu'):
        super(GeometricTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            if block == 'self':
                layers.append(GeoTransformerLayer(d_model, num_head, dropout=dropout, activation_fn=activation_fn))
            elif block == 'cross':
                layers.append(TransformerLayer(d_model, num_head, dropout=dropout, activation_fn=activation_fn))
            else:
                raise ValueError('Unsupported block type "{}" in `RPEConditionalTransformer`.'.format(block))
        self.layers = nn.ModuleList(layers)

    def forward(self, feats0, feats1, embeddings0, embeddings1, masks0=None, masks1=None):
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, _ = self.layers[i](feats0, feats0, embeddings0, memory_masks=masks0)
                feats1, _ = self.layers[i](feats1, feats1, embeddings1, memory_masks=masks1)
            else:
                feats0, _ = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, _ = self.layers[i](feats1, feats0, memory_masks=masks0)
        return feats0, feats1
