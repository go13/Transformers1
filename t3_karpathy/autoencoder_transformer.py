import torch
from torch import nn as nn
from torch.nn import functional as F

from t3_karpathy.karpathy_transformer import Block
from t3_karpathy.transformer_config import TransformerConfig
from t3_karpathy.karpathy_transformer import AbstractRunner


class AutoencoderTransformerModel(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks1 = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        self.ln_mid = nn.LayerNorm(config.n_embd)
        self.mid = nn.Linear(config.n_embd, config.vocab_size)

        self.blocks2 = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        self.ln_out = nn.LayerNorm(config.n_embd)
        self.out = nn.Linear(config.n_embd, config.vocab_size)

        self.out.weight = self.token_embedding_table.weight

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
        pos_emb = self.position_embedding_table(torch.arange(t, device=self.config.my_device))  # (T,C)
        x = tok_emb + pos_emb

        x = self.blocks1(x)

        x = self.ln_mid(x)
        x = self.mid(x)

        x = self.blocks2(x)

        x = self.ln_out(x)
        x = self.out(x)

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


class AutoencoderRunner(AbstractRunner):
    def __init__(self, config: TransformerConfig):
        super().__init__(config, AutoencoderTransformerModel(config))
        pass

    def generate(self, context, max_new_tokens):
        return self.model.generate(context, max_new_tokens)