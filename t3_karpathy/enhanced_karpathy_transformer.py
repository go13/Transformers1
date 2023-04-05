import torch
import torch.nn as nn
from torch.nn import functional as F
from flash_attn.modules.mha import MHA

from t3_karpathy.commons import AbstractRunner, BaseTransformerConfig, AbstractDataLoader
from t3_karpathy.transformer_config import TransformerConfig
# todo create step embedding and leave only one trans layer and iterate it. while extract weights using attention in another transformer
# todo extract weights into separate transformer and learn layers to read write weights based on memory


class FeedForward(nn.Module):
    def __init__(self, inp_n_embd, hidden_n_embd, out_n_embd, dropout, bias=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_n_embd, hidden_n_embd, bias=bias),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(hidden_n_embd, out_n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class NNAttentionHead(nn.Module):
    """ one head of self-attention """

    def __init__(self, block_size: int, n_embd: int, head_size: int, dropout: float):
        super().__init__()
        # self.pos_em_ff = nn.Linear(n_embd, n_embd, bias=False)
        self.att = FeedForward(n_embd * 2, n_embd, 1, dropout, bias=True)
        # self.att = LinearFeedForward(n_embd * 3, n_embd, 1, dropout)
        # self.att2 = nn.Linear(n_embd * 4, 1, bias=False)

        self.value = nn.Linear(n_embd, head_size, bias=True)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb, pos_dist_emb):
        b, t, c = x.shape

        # pos_emb = self.pos_em_ff(pos_emb)
        # x1 = pos_emb + x
        x1 = x #+ pos_emb
        # x1 = torch.cat([pos_emb, x], dim=-1) # (B,T,C * 2)

        x_tmp = x1.unsqueeze(1).repeat(1, t, 1, 1) # (B,T,C) -> (B,T,T,C)

        k = x_tmp
        q = x_tmp.transpose(1, 2)

        # a2 = torch.cat([k, q, pos_emb], dim=-1) # (B,T,T,C)
        # a2 = torch.cat([k, q], dim=-1)  # (B,T,T,C)
        a2 = torch.cat([k, q], dim=-1) + pos_dist_emb # (B,T,T,C)
        # a2 = torch.cat([k, q, pos_emb], dim=-1)   # (B,T,T,C)

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


class FlashMultiHeadAttention(nn.Module):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__()
        assert torch.bfloat16 == config.precision, 'only bfloat16 is supported'
        #
        # self.flash_mha = FlashMHA(
        #     embed_dim=config.n_embed,  # total channels (= num_heads * head_dim)
        #     num_heads=config.n_head,
        #     device=config.my_device,
        #     dtype=config.precision,
        #     attention_dropout=config.dropout,
        #     causal=True # auto-regressive or not
        # )

        # MHA + rotary requires flash-attention\csrc\rotary>pip install .
        self.flash_mha = MHA(
            embed_dim=config.n_embed,  # total channels (= num_heads * head_dim)
            num_heads=config.n_head,
            device=config.my_device,
            dtype=config.precision,
            dropout=config.dropout,
            use_flash_attn=True,
            return_residual=True,
            dwconv=True,
            rotary_emb_dim=config.head_size,
            causal=True # auto-regressive or not
        )

    def forward(self, x, st_pos_emb, pos_dist_emb):
        # inp = torch.cat(x, st_pos_emb, dim=-1)
        inp = x
        # inp = x + st_pos_emb
        out = self.flash_mha(inp)[0]
        return out


class Block(nn.Module):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__()

        # self.sa = MultiHeadAttention(config)
        self.sa = FlashMultiHeadAttention(config)

        dropout = config.dropout
        hidden_emb = config.hidden_size
        n_embed = config.n_embed
        out_emb = config.n_embed

        self.ln2 = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward(n_embed, hidden_emb, out_emb, dropout, bias=False)

    def forward(self, x, st_pos_emb, pos_dist_emb):
        x = x + self.sa(x, st_pos_emb, pos_dist_emb)
        x = x + self.ffwd(self.ln2(x))
        return x, st_pos_emb

    @staticmethod
    def create(config: BaseTransformerConfig):
        return Block(config)


def distance_triangle(n, my_device):
    arange_matrix = torch.arange(n, device=my_device).view(-1, 1) - torch.arange(n, device=my_device).view(1, -1)
    lower_triangular = torch.tril(arange_matrix)
    return lower_triangular


class PositionalEmbedding(nn.Module):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__()
        self.config = config
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        self.position_embedding_ff = FeedForward(config.n_embed, config.n_embed, config.n_embed, config.dropout)
        self.position_embedding_ff_ln = nn.LayerNorm(config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, b, t):
        pos_embedding_arrange = torch.arange(t, device=self.config.my_device)
        pos_emb = self.position_embedding_table(pos_embedding_arrange).repeat(b, 1, 1)  # (B,T,C)
        pos_emb = self.position_embedding_ff(pos_emb)
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
        self.position_embedding_ff = FeedForward(config.n_embed, config.n_embed * 2, config.n_embed * 2, config.dropout)
        self.position_embedding_ff_ln = nn.LayerNorm(config.n_embed * 2)

    def forward(self, b):
        pos_embedding_arrange = distance_triangle(self.config.block_size, self.config.my_device)
        pos_emb = self.position_embedding_table(pos_embedding_arrange)
        pos_emb = pos_emb.repeat(b, 1, 1, 1)  # (B, T, T, C)
        pos_emb = self.position_embedding_ff(pos_emb)
        # pos_emb = self.position_embedding_ff_ln(pos_emb)
        return pos_emb


class BlockSequence(nn.Module):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__()
        self.blocks = nn.Sequential(*[Block.create(config) for _ in range(config.n_layer)])

    def forward(self, x, st_pos_emb, pos_dist_emb):
        for block in self.blocks:
            x, st_pos_emb = block(x, st_pos_emb, pos_dist_emb)
        return x, st_pos_emb


class KarpathyTransformerModel(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)

        # self.pos_emb1 = PositionalEmbedding(config)
        # self.pos_dist_emb = DistancePositionalEmbedding(config)
        self.pos_ffwd = FeedForward(config.n_embed * 3, config.n_embed, config.n_embed * 2, config.dropout)
        self.pos_ln = nn.LayerNorm(config.n_embed * 2)

        self.blocks = BlockSequence(config)
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

        pos_dist_emb = None #self.pos_dist_emb(b)

        pos_emb = None #self.pos_emb1(b, t)

        # pos_emb = torch.cat([pos_emb_dist, pos_emb], dim=-1)
        #
        # pos_emb = self.pos_ffwd(pos_emb)
        #
        # pos_emb = self.pos_ln(pos_emb)
        # pos_emb = pos_emb_dist + pos_emb

        # x, st_pos_emb = self.blocks(x, pos_emb)  # (B,T,C)
        x, st_pos_emb = self.blocks(x, pos_emb, pos_dist_emb)  # (B,T,C)

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


class EnhancedKarpathyRunner(AbstractRunner):
    def __init__(self, config: TransformerConfig, data_loader: AbstractDataLoader):
        super().__init__(config, KarpathyTransformerModel(config), data_loader)
        pass

    def generate(self, context, max_new_tokens):
        return self.model.generate(context, max_new_tokens)
