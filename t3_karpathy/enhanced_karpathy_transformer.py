from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F

from t3_karpathy.transformer_config import TransformerConfig


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
            nn.Sigmoid(),
            # nn.ReLU(),
            nn.Linear(hidden_n_embd, out_n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class NNAttentionHead(nn.Module):
    """ one head of self-attention """

    def __init__(self, block_size: int, n_embd: int, head_size: int, dropout: float):
        super().__init__()
        self.att2 = FeedForward(n_embd * 4, 4 * n_embd, 1, 0)

        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb):
        b, t, c = x.shape

        x1 = torch.cat([pos_emb, x], dim=-1) # (B,T,C * 2)

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

    def __init__(self, inp_size: int, n_embd: int, head_size: int, n_head: int, dropout: float, my_device='cuda'):
        super().__init__()
        self.my_device = my_device

        self.heads = nn.ModuleList([NNAttentionHead(inp_size, n_embd, head_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb):
        out = torch.cat([h(x, pos_emb) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class BlockSequence(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.blocks = nn.Sequential(*[Block.create(config) for _ in range(config.n_layer)])

    def forward(self, x, pos_emb):
        for block in self.blocks:
            x, pos_emb = block(x, pos_emb)
        return x, pos_emb


class Block(nn.Module):
    def __init__(self, dropout: float, block_size: int, hidden_emb: int, inp_embd: int, out_emb: int, n_head: int, head_size: int):
        super().__init__()

        self.ln1 = nn.LayerNorm(inp_embd)
        self.sa = MultiHeadAttention(block_size, inp_embd, head_size, n_head, dropout)

        self.ln2 = nn.LayerNorm(inp_embd)
        self.ffwd = FeedForward(inp_embd, hidden_emb, out_emb, dropout)
        self.ln3 = nn.LayerNorm(inp_embd)

    def forward(self, x, pos_emb):
        x = x + self.sa(self.ln1(x), pos_emb)
        x = x + self.ffwd(self.ln2(x))
        x = self.ln3(x)
        return x, pos_emb

    @staticmethod
    def create(config: TransformerConfig):
        block_size = config.block_size
        out_size = config.n_embd
        hidden_size = config.hidden_size
        dropout = config.dropout
        n_embd = config.n_embd
        head_size = config.head_size
        n_head = config.n_head
        return Block(dropout, block_size, hidden_size, n_embd, out_size, n_head, head_size)


class PositionalEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.position_embedding_ff = FeedForward(config.n_embd, config.n_embd, config.n_embd, config.dropout)
        self.position_embedding_ff_ln = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        b, t, c = x.shape
        pos_embedding_arrange = torch.arange(t, device=self.config.my_device)
        pos_emb = self.position_embedding_table(pos_embedding_arrange).repeat(b, 1, 1)  # (B,T,C)
        pos_emb = self.position_embedding_ff(pos_emb)
        pos_emb = self.position_embedding_ff_ln(pos_emb)
        return pos_emb


class KarpathyTransformerModel(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)

        self.pe1 = PositionalEmbedding(config)
        self.pe2 = PositionalEmbedding(config)
        self.pe3 = PositionalEmbedding(config)
        self.pe4 = PositionalEmbedding(config)

        self.blocks = BlockSequence(config)
        # self.ln_f = nn.LayerNorm(config.n_embd)  # final layer norm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

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

        pos_emb1 = self.pe1(x)
        pos_emb2 = self.pe2(x)
        pos_emb3 = self.pe3(x)
        pos_emb4 = self.pe4(x)

        x, pos_emb = self.blocks(x, pos_emb1)  # (B,T,C)
        x, pos_emb = self.blocks(x, pos_emb2)  # (B,T,C)
        x, pos_emb = self.blocks(x, pos_emb3)  # (B,T,C)
        x, pos_emb = self.blocks(x, pos_emb4)  # (B,T,C)

        # x = self.ln_f(x)  # (B,T,C)
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


class AbstractRunner(object):
    def __init__(self, config: TransformerConfig, model):
        self.model = model.to(config.my_device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.config = config
        self.current_iteration = 0

        print(sum(p.numel() for p in self.model.parameters()) / 1e6, 'M parameters')

    def forward(self, x):
        return self.model(x)

    def learn(self, x, y):
        self.model.train()
        out, loss = self.model.forward_vs_target(x, y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return out, loss

    @torch.no_grad()
    def evaluate(self, get_batch, eval_iters):
        self.model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch()
            logits, loss = self.model.forward_vs_target(x, y)
            losses[k] = loss.item()
        return losses.mean()

    def train_iterate(self, n_iter, get_train_batch, get_val_batch):
        for _ in range(n_iter):
            if self.current_iteration % self.config.eval_interval == 0:
                train_losses = self.evaluate(get_train_batch, self.config.eval_iters)
                val_losses = self.evaluate(get_val_batch, self.config.eval_iters)
                print(f"step {self.current_iteration}: train loss {train_losses:.4f}, val loss {val_losses:.4f}")

            x, y = get_train_batch()

            logits, loss = self.learn(x, y)

            self.current_iteration += 1

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, new_state_dict):
        self.model.load_state_dict(OrderedDict(new_state_dict))

    def generate(self, *args):
        raise NotImplementedError()


class KarpathyRunner(AbstractRunner):
    def __init__(self, config: TransformerConfig):
        super().__init__(config, KarpathyTransformerModel(config))
        pass

    def generate(self, context, max_new_tokens):
        return self.model.generate(context, max_new_tokens)
