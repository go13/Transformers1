from collections import OrderedDict

import torch
import time
import torch.nn as nn
from torch.nn import functional as F

from t3_karpathy.commons import AbstractRunner, BaseTransformerConfig
from t3_karpathy.transformer_config import TransformerConfig


class FeedForward(nn.Module):
    def __init__(self, inp_n_embd, hidden_n_embd, out_n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_n_embd, hidden_n_embd),
            nn.ReLU(),
            nn.Linear(hidden_n_embd, out_n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AttentionHead(nn.Module):
    """ one head of self-attention """

    def __init__(self, block_size: int, n_embd: int, head_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # self.key = FeedForward(n_embd, n_embd * 8, head_size, dropout)
        # self.query = FeedForward(n_embd, n_embd * 8, head_size, dropout)
        # self.value = FeedForward(n_embd, n_embd * 8, head_size, dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, t, c = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * c ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        # do we not need to normalize longer rows in the triangle diagonal?
        return out


class NNAttentionHead(nn.Module):
    """ one head of self-attention """

    def __init__(self, block_size: int, n_embd: int, head_size: int, dropout: float):
        super().__init__()
        self.att = FeedForward(n_embd * 4, n_embd, 1, 0)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, t, c = x.shape
        pos_embedding_arrange = torch.arange(t, device='cuda')
        pos_emb = self.position_embedding_table(pos_embedding_arrange).repeat(b, 1, 1)  # (B,T,C)

        x1 = torch.cat([pos_emb, x], dim=-1) # (B,T,C * 2)

        k = x1.unsqueeze(1).repeat(1, t, 1, 1)  # (B,T,C) -> (B,T,T,C)
        q = x1.unsqueeze(1).repeat(1, t, 1, 1).transpose(1, 2)  # (B,T,C) -> (B,T,T,C)

        a2 = torch.cat([k, q], dim=-1) # (B,T,T,C)

        a2 = self.att(a2) # (B,T,T,C * 2) -> (B,T,T,1)

        wei = a2.squeeze(dim=-1) * c ** -0.5
        # compute attention scores ("affinities")
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, inp_size: int, n_embd: int, head_size: int, n_head: int, dropout: float, my_device='cuda'):
        super().__init__()
        self.my_device = my_device
        self.position_embedding_table = nn.Embedding(inp_size, n_embd)
        # self.register_buffer('pos_embedding_arrange', torch.arange(inp_size, device=my_device))

        self.heads = nn.ModuleList([NNAttentionHead(inp_size, n_embd, head_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class Block(nn.Module):
    def __init__(self, dropout: float, block_size: int, hidden_emb: int, inp_embd: int, out_emb: int, n_head: int, head_size: int):
        super().__init__()

        self.ln1 = nn.LayerNorm(inp_embd)
        self.sa = MultiHeadAttention(block_size, inp_embd, head_size, n_head, dropout)

        self.ln2 = nn.LayerNorm(inp_embd)
        self.ffwd = FeedForward(inp_embd, hidden_emb, out_emb, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

    @staticmethod
    def create(config: TransformerConfig):
        block_size = config.block_size
        out_size = config.n_embed
        hidden_size = config.hidden_size
        dropout = config.dropout
        n_embd = config.n_embed
        head_size = config.head_size
        n_head = config.n_head
        return Block(dropout, block_size, hidden_size, n_embd, out_size, n_head, head_size)


class KarpathyTransformerModel(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)
        # self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        self.blocks = nn.Sequential(*[Block.create(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embed)  # final layer norm
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)

    def forward_vs_target(self, idx, targets):
        logits = self.forward(idx)

        b, t, c = logits.shape
        logits_view = logits.view(b * t, c)
        targets = targets.view(b * t)
        loss = F.cross_entropy(logits_view, targets)

        return logits_view, loss

    def forward(self, idx):
        b, t = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)

        x = tok_emb # + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        return logits

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.config.block_size:]
            # get the predictions
            logits = self.forward(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class KarpathyRunner(AbstractRunner):
    def __init__(self, config: TransformerConfig):
        super().__init__(config, KarpathyTransformerModel(config))
        pass

    def generate(self, context, max_new_tokens):
        return self.model.generate(context, max_new_tokens)
