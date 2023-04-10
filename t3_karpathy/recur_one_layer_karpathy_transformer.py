from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F

from t3_karpathy.transformer_config import TransformerConfig
from commons.commons import AbstractRunner
# todo create step embedding and leave only one trans layer and iterate it. while extract weights using attention in another transformer
# todo extract weights into separate transformer and learn layers to read write weights based on memory

# stats
# TransformerConfig(n_embed=64, n_head=4, n_layer=16, batch_size=32) - step 20000: train loss 1.5061, val loss 1.6938
class PlainFeedForward(nn.Module):
    def __init__(self, inp_n_embd, hidden_n_embd, out_n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_n_embd, out_n_embd, bias=False),
        )

    def forward(self, x):
        return self.net(x)


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


class SuperFeedForward(nn.Module):
    def __init__(self, inp_n_embd, hidden_n_embd, out_n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_n_embd, hidden_n_embd),
            nn.ReLU(),
            nn.Linear(hidden_n_embd, hidden_n_embd),
            nn.Dropout(dropout),
            nn.Linear(hidden_n_embd, hidden_n_embd),
            nn.ReLU(),
            nn.Linear(hidden_n_embd, out_n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SlimNNAttentionHead(nn.Module):
    """ one head of self-attention """

    def __init__(self, block_size: int, n_embd: int, head_size: int, dropout: float, att_nn: nn.Module):
        super().__init__()
        # self.pos_em_ff = FeedForward(n_embd, n_embd, n_embd, dropout)
        # self.pos_em_ff = nn.Linear(n_embd, n_embd)
        # self.st_em_ff = FeedForward(n_embd * 2, 2 * n_embd, n_embd, dropout)
        self.st_pos_em_ff = FeedForward(n_embd, n_embd, n_embd, dropout)
        self.att = att_nn #FeedForward(n_embd * 4, 4 * n_embd, 1, dropout)
        # self.att = nn.Linear(n_embd * 6, 1, bias=False)
        self.att2 = nn.Linear(n_embd, 1)

        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, st_pos_emb):
        # st_emb (B, T, C)
        b, t, c = x.shape

        st_pos_emb = self.st_pos_em_ff(st_pos_emb)
        # pos_emb = self.pos_em_ff(pos_emb)
        # x1 = pos_emb + x
        x1 = torch.cat([st_pos_emb, x], dim=-1) # (B,T,C * 2)

        x1 = x1.unsqueeze(1).repeat(1, t, 1, 1)

        k = x1  # (B,T,C) -> (B,T,T,C)
        q = x1.transpose(1, 2)  # (B,T,C) -> (B,T,T,C)

        a2 = torch.cat([k, q], dim=-1) # (B,T,T,C)
        # a2 = k + q

        a2 = self.att(a2) # (B,T,T,C * 2) -> (B,T,T,1)
        a2 = self.att2(a2) # (B,T,T,C * 2) -> (B,T,T,1)

        wei = a2.squeeze(dim=-1) * c ** -0.5
        # compute attention scores ("affinities")
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class NNAttentionHead(nn.Module):
    """ one head of self-attention """

    def __init__(self, block_size: int, n_embd: int, head_size: int, dropout: float):
        super().__init__()
        # self.pos_em_ff = nn.Linear(n_embd, n_embd, bias=False)
        self.att2 = FeedForward(n_embd * 2, n_embd, 1, dropout)
        # self.att2 = nn.Linear(n_embd * 4, 1, bias=False)

        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb):
        b, t, c = x.shape

        # pos_emb = self.pos_em_ff(pos_emb)
        x1 = pos_emb + x
        # x1 = torch.cat([pos_emb, x], dim=-1) # (B,T,C * 2)

        k = x1.unsqueeze(1).repeat(1, t, 1, 1)  # (B,T,C) -> (B,T,T,C)
        q = x1.unsqueeze(1).repeat(1, t, 1, 1).transpose(1, 2)  # (B,T,C) -> (B,T,T,C)

        a2 = torch.cat([k, q], dim=-1) # (B,T,T,C)

        a2 = self.att2(a2) # (B,T,T,C * 2) -> (B,T,T,1)

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

    def __init__(self, inp_size: int, n_embd: int, head_size: int, n_head: int, dropout: float, att_nn, my_device='cuda'):
        super().__init__()
        self.my_device = my_device

        self.heads = nn.ModuleList([SlimNNAttentionHead(inp_size, n_embd, head_size, dropout, att_nn) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, st_pos_emb):
        out = torch.cat([h(x, st_pos_emb) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class Block(nn.Module):
    def __init__(self, dropout: float, block_size: int, hidden_emb: int, inp_embd: int, out_emb: int, n_head: int, head_size: int, att_nn):
        super().__init__()

        self.st_pos_em_ff = FeedForward(inp_embd, hidden_emb, inp_embd, dropout)

        self.ln1 = nn.LayerNorm(inp_embd)
        self.sa = MultiHeadAttention(block_size, inp_embd, head_size, n_head, dropout, att_nn)

        self.ln2 = nn.LayerNorm(inp_embd)
        self.ffwd = FeedForward(inp_embd, hidden_emb, out_emb, dropout)

    def forward(self, x, st_pos_emb):
        st_pos_emb = st_pos_emb + self.st_pos_em_ff(st_pos_emb)
        x = x + self.sa(self.ln1(x), st_pos_emb)
        x = x + self.ffwd(self.ln2(x))
        return x, st_pos_emb

    @staticmethod
    def create(config: TransformerConfig, att_nn):
        block_size = config.block_size
        out_size = config.n_embed
        hidden_size = config.hidden_size
        dropout = config.dropout
        n_embd = config.n_embed
        head_size = config.head_size
        n_head = config.n_head
        return Block(dropout, block_size, hidden_size, n_embd, out_size, n_head, head_size, att_nn)


class PositionalEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        self.position_embedding_ff = FeedForward(config.n_embed, config.n_embed, config.n_embed, config.dropout)
        # self.position_embedding_ff_ln = nn.LayerNorm(config.n_embed)

    def forward(self, b, t):
        pos_embedding_arrange = torch.arange(t, device=self.config.my_device)
        pos_emb = self.position_embedding_table(pos_embedding_arrange).repeat(b, 1, 1)  # (B,T,C)
        pos_emb = self.position_embedding_ff(pos_emb)
        # pos_emb = self.position_embedding_ff_ln(pos_emb)
        return pos_emb


class BlockSequence(nn.Module):
    def __init__(self, config: TransformerConfig, att_nn):
        super().__init__()
        self.blocks = nn.Sequential(*[Block.create(config, att_nn) for _ in range(1)])

    def forward(self, x, st_pos_emb):
        for block in self.blocks:
            x, st_pos_emb = block(x, st_pos_emb)
        return x, st_pos_emb


class KarpathyTransformerModel(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)

        self.pe1 = PositionalEmbedding(config)
        self.se1 = PositionalEmbedding(config)
        self.st_em_ff = FeedForward(config.n_embed * 2, config.n_embed, config.n_embed, config.dropout)
        self.att_nn = SuperFeedForward(config.n_embed * 4, 4 * config.n_embed, config.n_embed, config.dropout)

        self.blocks = BlockSequence(config, self.att_nn)
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
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)

        x = tok_emb # + pos_emb  # (B,T,C)
        b, t, c = x.shape

        pos_emb = self.pe1(b, t)
        step_emb = self.se1(b, self.config.n_layer)

        for i in range(self.config.n_layer):
            st_emb = step_emb[:, i, :].unsqueeze(1).repeat(1, t, 1)   # (B,T,C)

            st_pos_emb = torch.cat([pos_emb, st_emb], dim=-1)  # (B,T,C * 2)
            st_pos_emb = self.st_em_ff(st_pos_emb)

            x, st_pos_emb = self.blocks(x, st_pos_emb)  # (B,T,C)

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
        super().__init__(config, KarpathyTransformerModel(config), None)
        pass

    def generate(self, context, max_new_tokens):
        return self.model.generate(context, max_new_tokens)
