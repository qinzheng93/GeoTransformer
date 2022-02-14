import math

import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError('Cannot use sinusoidal positional encoding with odd d_model ({:d})'.format(d_model))
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, emb_indices):
        r"""
        Sinusoidal Positional Embedding

        :param emb_indices: torch.Tensor (*)
        :return embeddings: torch.Tensor (*, D)
        """
        input_shape = emb_indices.shape
        omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)  # (BxMxN, d_model/2, 1)
        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (BxMxN, d_model/2, 2)
        embeddings = embeddings.view(*input_shape, self.d_model)  # (B, M, N, d_model)
        embeddings = embeddings.detach()
        return embeddings


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, num_embedding, embedding_dim):
        super(LearnablePositionalEmbedding, self).__init__()
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_embedding, embedding_dim)  # (L, D)

    def forward(self, emb_indices):
        r"""
        Learnable Positional Embedding
        `emb_indices` are truncated to fit the finite embedding space.

        :param emb_indices: torch.LongTensor (*)
        :return embeddings: torch.Tensor (*, D)
        """
        input_shape = emb_indices.shape
        emb_indices = emb_indices.view(-1)
        max_emd_indices = torch.ones_like(emb_indices) * (self.num_embedding - 1)
        emb_indices = torch.minimum(emb_indices, max_emd_indices)
        embeddings = self.embeddings(emb_indices)  # (*, D)
        embeddings = embeddings.view(*input_shape, self.embedding_dim)
        return embeddings
