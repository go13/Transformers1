import torch
import torch.nn as nn
from torch.nn import functional as F

from t3_karpathy.commons.commons import AbstractRunner, BaseTransformerConfig, AbstractDataLoader
from t3_karpathy.commons.embeddings import PositionalEmbedding, DistancePositionalEmbedding
from t3_karpathy.commons.feed_forwards import GeluFeedForward, LinearFeedForward
from t4.generic_dataloader import GenericDataloader


class NNAttentionHead(nn.Module):
    """ one head of self-attention """

    def __init__(self, block_size: int, n_embd: int, head_size: int, dropout: float):
        super().__init__()
        self.att = GeluFeedForward(n_embd * 2, n_embd, 1, dropout, bias=True)
        # self.att = LinearFeedForward(n_embd * 3, n_embd, 1, dropout)
        # self.att2 = nn.Linear(n_embd * 4, 1, bias=False)

        self.value = nn.Linear(n_embd, head_size, bias=True)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb, pos_dist_emb):
        b, t, c = x.shape

        # pos_emb = self.pos_em_ff(pos_emb)
        # x1 = pos_emb + x
        x1 = x  # + pos_emb
        # x1 = torch.cat([pos_emb, x], dim=-1) # (B,T,C * 2)

        x_tmp = x1.unsqueeze(1).repeat(1, t, 1, 1)  # (B,T,C) -> (B,T,T,C)

        k = x_tmp
        q = x_tmp.transpose(1, 2)

        # a2 = torch.cat([k, q, pos_emb], dim=-1) # (B,T,T,C)
        # a2 = torch.cat([k, q], dim=-1)  # (B,T,T,C)
        a2 = torch.cat([k, q], dim=-1) + pos_dist_emb  # (B,T,T,C)
        # a2 = torch.cat([k, q, pos_emb], dim=-1)   # (B,T,T,C)

        a2 = self.att.forward(a2)  # (B,T,T,C * 2) -> (B,T,T,1)

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

    def __init__(self, config: BaseTransformerConfig):
        super().__init__()

        dropout = config.dropout
        block_size = config.block_size
        n_embed = config.n_embed
        head_size = config.head_size
        n_head = config.n_head

        self.heads = nn.ModuleList([NNAttentionHead(block_size, n_embed, head_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, st_pos_emb, pos_dist_emb):
        out = torch.cat([h(x, st_pos_emb, pos_dist_emb) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class Block(nn.Module):
    def __init__(self, config: BaseTransformerConfig, causal=True):
        super().__init__()

        self.sa = MultiHeadAttention(config)

        dropout = config.dropout
        hidden_emb = config.hidden_size
        n_embed = config.n_embed
        out_emb = config.n_embed

        self.ln2 = nn.LayerNorm(n_embed)
        self.ffwd = GeluFeedForward(n_embed, hidden_emb, out_emb, dropout, bias=False)

    def forward(self, x, st_pos_emb, pos_dist_emb):
        x = x + self.sa.forward(x, st_pos_emb, pos_dist_emb)
        x = x + self.ffwd.forward(self.ln2(x))
        return x, st_pos_emb

    @staticmethod
    def create(config: BaseTransformerConfig, causal=True):
        return Block(config, causal=causal)


class BlockSequence(nn.Module):
    def __init__(self, config: BaseTransformerConfig, causal=True):
        super().__init__()
        self.blocks = nn.Sequential(*[Block.create(config, causal) for _ in range(config.n_layer)])

    def forward(self, x, st_pos_emb, pos_dist_emb):
        for block in self.blocks:
            x, st_pos_emb = block(x, st_pos_emb, pos_dist_emb)
        return x, st_pos_emb


class TransformerConfig(BaseTransformerConfig):
    def __init__(self, input_embed, my_device='cuda', precision=torch.bfloat16, batch_size=128, block_size=256,
                 n_embed=16, n_head=2, n_layer=4, learning_rate=1e-3):
        super().__init__(my_device, precision, batch_size, block_size, n_embed, n_head, n_layer, learning_rate)
        self.input_embed = input_embed


class TransformerModel(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.pos_emb = PositionalEmbedding(config)
        self.pos_dist_emb = DistancePositionalEmbedding(config)

        self.ffwd1 = LinearFeedForward(
            config.input_embed, config.hidden_size, config.n_embed, config.dropout, bias=False)

        self.encoder = BlockSequence(config, causal=False)

        self.ln_mid = nn.LayerNorm(config.n_embed)

        self.decoder = BlockSequence(config, causal=False)

        self.ffwd2 = LinearFeedForward(config.n_embed, config.hidden_size, config.input_embed, config.dropout, bias=True)

        # print(self)

    def forward_vs_target(self, inp, targets):
        output = self.forward(inp)

        mse_loss = torch.nn.MSELoss(reduction='mean')
        loss = mse_loss(output, targets)

        return output, loss

    def forward(self, inp):
        b, t, c = inp.shape

        pos_emb = self.pos_emb.forward(b, t)
        step_emb = self.pos_dist_emb.forward(b)#, self.config.n_layer)

        x = inp

        x = self.ffwd1.forward(x)

        x = self.encoder.forward(x)

        x = self.ln_mid.forward(x)

        x = self.decoder.forward(x)

        x = self.ffwd2.forward(x)
        return x

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


class TransformerRunner(AbstractRunner):
    def __init__(self, config: TransformerConfig, data):
        super().__init__(config, TransformerModel(config), GenericDataloader(config, data))
        pass

    def generate(self, context, max_new_tokens):
        return self.model.generate(context, max_new_tokens)
