import torch
import torch.nn as nn
from torch.nn import functional as F
from flash_attn.modules.mha import MHA

from t3_karpathy.commons.commons import AbstractRunner, BaseTransformerConfig, AbstractDataLoader
from t3_karpathy.commons.feed_forwards import GeluFeedForward
from t3_karpathy.transformer_config import TransformerConfig
from t4.generic_dataloader import GenericDataloader


class FlashMultiHeadAttention(nn.Module):
    def __init__(self, config: BaseTransformerConfig, causal=True):
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
            causal=causal  # auto-regressive or not
        )

    def forward(self, x):
        out = self.flash_mha(x)[0]
        return out


class Block(nn.Module):
    def __init__(self, config: BaseTransformerConfig, causal=True):
        super().__init__()

        self.sa = FlashMultiHeadAttention(config, causal=causal)

        dropout = config.dropout
        hidden_emb = config.hidden_size
        n_embed = config.n_embed
        out_emb = config.n_embed

        self.ln2 = nn.LayerNorm(n_embed)
        self.ffwd = GeluFeedForward(n_embed, hidden_emb, out_emb, dropout, bias=False)

    def forward(self, x):
        x = x + self.sa.forward(x)
        x = x + self.ffwd.forward(self.ln2(x))
        return x


class BlockSequence(nn.Module):
    def __init__(self, config: BaseTransformerConfig, causal=True):
        super().__init__()
        self.blocks = nn.Sequential(*[Block(config, causal) for _ in range(config.n_layer)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class TransformerConfig(BaseTransformerConfig):
    def __init__(self, vocab_size, my_device='cuda', precision=torch.bfloat16, batch_size=128, block_size=256,
                 n_embed=16, n_head=2, n_layer=4, learning_rate=1e-3):
        super().__init__(my_device, precision, batch_size, block_size, n_embed, n_head, n_layer, learning_rate)
        self.vocab_size = vocab_size


class SymbolicTransformerModel(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embed)

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

    def forward(self, inp):
        x = self.token_embeddings(inp)  # (B,T,C)

        x = self.blocks.forward(x)

        x = self.ln_f(x)
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


class TransformerModel(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.blocks = BlockSequence(config)

    def forward_vs_target(self, idx, targets):
        logits = self.forward(idx)

        b, t, c = logits.shape
        logits_view = logits.view(b * t, c)
        targets = targets.view(b * t)
        loss = F.cross_entropy(logits_view, targets)

        return logits_view, loss

    def forward(self, inp):
        return self.blocks.forward(inp)

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


class SymbolicTransformerRunner(AbstractRunner):
    def __init__(self, config: TransformerConfig, train_data, val_data):
        super().__init__(config, SymbolicTransformerModel(config), GenericDataloader(config, train_data, val_data))
        pass

    def generate(self, context, max_new_tokens):
        return self.model.generate(context, max_new_tokens)


class TransformerRunner(AbstractRunner):
    def __init__(self, config: TransformerConfig, train_data, val_data):
        super().__init__(config, TransformerModel(config), GenericDataloader(config, train_data, val_data))
        pass

    def generate(self, context, max_new_tokens):
        return self.model.generate(context, max_new_tokens)
