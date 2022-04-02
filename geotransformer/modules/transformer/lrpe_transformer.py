r"""Transformer with Learnable Relative Positional Embeddings.

Relative positional embedding is injected in each multi-head attention layer.

The shape of input tensor should be (B, N, C).
Implemented with `nn.Linear` and `nn.LayerNorm` (with affine).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from geotransformer.modules.layers import build_dropout_layer
from geotransformer.modules.transformer.output_layer import AttentionOutput
from geotransformer.modules.transformer.positional_embedding import LearnablePositionalEmbedding


class LRPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_embeddings, dropout=None):
        super(LRPEMultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError(f'"d_model" ({d_model}) is not divisible by "num_heads" ({num_heads}).')

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads
        self.num_embeddings = num_embeddings

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)

        self.embedding = LearnablePositionalEmbedding(num_embeddings, d_model, dropout=dropout)

        self.dropout = build_dropout_layer(dropout)

    def transpose_for_scores(self, x):
        x = x.view(x.shape[0], x.shape[1], self.num_heads, self.d_model_per_head)
        x = x.permute(0, 2, 1, 3)
        return x

    def get_embeddings(self, q, emb_indices):
        emb_all_indices = torch.arange(self.num_embeddings).cuda()  # (P,)
        emb_bank = rearrange(self.embedding(emb_all_indices), 'p (h c) -> h p c', h=self.num_heads)
        attention_scores = torch.einsum('bhnc,hpc->bhnp', q, emb_bank)
        emb_indices = emb_indices.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (B, N, M) -> (B, H, N, M)
        attention_scores = torch.gather(attention_scores, dim=-1, index=emb_indices)  # (B, H, N, P) -> (B, H, N, M)
        return attention_scores

    def forward(
        self,
        input_q,
        input_k,
        input_v,
        emb_indices_qk,
        key_masks=None,
        attention_factors=None,
    ):
        r"""Scaled Dot-Product Attention with Learnable Relative Positional Embedding (forward)

        Args:
            input_q: torch.Tensor (B, N, C)
            input_k: torch.Tensor (B, M, C)
            input_v: torch.Tensor (B, M, C)
            emb_indices_qk: torch.Tensor (B, N, M), relative position indices
            key_masks: torch.Tensor (B, M), True if ignored, False if preserved
            attention_factors: torch.Tensor (B, N, M)

        Returns
            hidden_states: torch.Tensor (B, N, C)
            attention_scores: torch.Tensor (B, H, N, M)
        """
        q = rearrange(self.proj_q(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(self.proj_k(input_k), 'b m (h c) -> b h m c', h=self.num_heads)
        v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)

        attention_scores_p = self.get_embedding_attention(q, emb_indices_qk)

        attention_scores_e = torch.einsum('bhnc,bhmc->bhnm', q, k)
        attention_scores = (attention_scores_e + attention_scores_p) / self.d_model_per_head ** 0.5
        if attention_factors is not None:
            attention_scores = attention_factors.unsqueeze(1) * attention_scores
        if key_masks is not None:
            attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)

        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')

        return hidden_states, attention_scores


class LRPEAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, rpe_size, dropout=None):
        super(LRPEAttentionLayer, self).__init__()
        self.attention = LRPEMultiHeadAttention(d_model, num_heads, rpe_size, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_states,
        memory_states,
        position_states,
        memory_masks=None,
        attention_factors=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            memory_states,
            position_states,
            key_masks=memory_masks,
            attention_factors=attention_factors,
        )
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores


class LRPETransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, rpe_size, dropout=None, activation_fn='ReLU'):
        super(LRPETransformerLayer, self).__init__()
        self.attention = LRPEAttentionLayer(d_model, num_heads, rpe_size, dropout=dropout)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(
        self,
        input_states,
        memory_states,
        position_states,
        memory_masks=None,
        attention_factors=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            position_states,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
        )
        output_states = self.output(hidden_states)
        return output_states, attention_scores
