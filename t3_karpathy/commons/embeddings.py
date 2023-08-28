import torch
from torch import nn as nn

from t3_karpathy.commons.commons import BaseTransformerConfig
from t3_karpathy.commons.feed_forwards import GeluFeedForward


def distance_triangle(n, my_device):
    arange_matrix = torch.arange(n, device=my_device).view(-1, 1) - torch.arange(n, device=my_device).view(1, -1)
    lower_triangular = torch.tril(arange_matrix)
    return lower_triangular

class PositionalEmbedding(nn.Module):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__()
        self.config = config
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        self.position_embedding_ff = GeluFeedForward(config.n_embed, config.n_embed, config.n_embed, config.dropout)
        self.position_embedding_ff_ln = nn.LayerNorm(config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, b, t):
        pos_embedding_arrange = torch.arange(t, device=self.config.my_device)
        pos_emb = self.position_embedding_table(pos_embedding_arrange).repeat(b, 1, 1)  # (B,T,C)
        pos_emb = self.position_embedding_ff.forward(pos_emb)
        # pos_emb = self.position_embedding_ff_ln(pos_emb)
        pos_emb = self.dropout(pos_emb)

        # pos_emb = pos_emb.unsqueeze(1).repeat(1, t, 1, 1)  # (B,T,C) -> (B,T,T,C)
        # k = pos_emb
        # q = pos_emb.transpose(1, 2)
        # pos_emb = torch.cat([k, q], dim=-1)  # (B,T,T,C)

        # return k + q
        return pos_emb


class DistancePositionalEmbedding(nn.Module):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__()
        self.config = config
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        self.position_embedding_ff = GeluFeedForward(config.n_embed, config.n_embed * 2, config.n_embed * 2, config.dropout)
        self.position_embedding_ff_ln = nn.LayerNorm(config.n_embed * 2)

    def forward(self, b):
        pos_embedding_arrange = distance_triangle(self.config.block_size, self.config.my_device)
        pos_emb = self.position_embedding_table(pos_embedding_arrange)
        pos_emb = pos_emb.repeat(b, 1, 1, 1)  # (B, T, T, C)
        pos_emb = self.position_embedding_ff.forward(pos_emb)
        # pos_emb = self.position_embedding_ff_ln(pos_emb)
        return pos_emb
