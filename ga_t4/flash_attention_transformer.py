import torch
import torch.nn as nn
from torch.nn import functional as F
from flash_attn.modules.mha import MHA

from t3_karpathy.commons.commons import AbstractRunner, BaseTransformerConfig, AbstractDataLoader
from t3_karpathy.commons.feed_forwards import GeluFeedForward
from t3_karpathy.transformer_config import TransformerConfig


# todo create step embedding and leave only one trans layer and iterate it. while extract weights using attention in another transformer
# todo extract weights into separate transformer and learn layers to read write weights based on memory


class FlashMultiHeadAttention(nn.Module):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__()
        assert torch.bfloat16 == config.precision, 'only bfloat16 is supported'

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
            causal=True  # auto-regressive or not
        )

    def forward(self, x):
        inp = x
        out = self.flash_mha(inp)[0]
        return out


class Block(nn.Module):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__()

        self.sa = FlashMultiHeadAttention(config)

        dropout = config.dropout
        hidden_emb = config.hidden_size
        n_embed = config.n_embed
        out_emb = config.n_embed

        self.ln2 = nn.LayerNorm(n_embed)
        self.ffwd = GeluFeedForward(n_embed, hidden_emb, out_emb, dropout, bias=False)

    def forward(self, x):
        x = self.sa(x) # should be x + self.sa(x) normally
        x = self.ffwd(self.ln2(x)) # should be x + self.ffwd(self.ln2(x)) normally
        return x

    @staticmethod
    def create(config: BaseTransformerConfig):
        return Block(config)


class BlockSequence(nn.Module):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__()
        self.blocks = nn.Sequential(*[Block.create(config) for _ in range(config.n_layer)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class FlashAttentionTransformerModel(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)

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

        x = tok_emb  # + pos_emb  # (B,T,C)
        b, t, c = x.shape

        x = self.blocks(x)  # (B,T,C)

        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        return logits


class FlashAttentionRunner(AbstractRunner):
    def __init__(self, config: TransformerConfig, data_loader: AbstractDataLoader):
        super().__init__(config, FlashAttentionTransformerModel(config), data_loader)
        pass

    def generate(self, context, max_new_tokens):
        return self.model.generate(context, max_new_tokens)
