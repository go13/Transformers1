import torch
from torch import nn as nn
from torch.nn import functional as F

from ga_t3.accumulative_trainer import StringAccumulativeTrainer
from t3_karpathy.karpathy_transformer import FeedForward, Block
from t3_karpathy.transformer_config import TransformerConfig
from t3_karpathy.karpathy_transformer import AbstractRunner


class CompressingAttentionHead(nn.Module):
    def __init__(self, inp_size: int, out_size: int, inp_embd: int, out_embd: int, head_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(out_embd, head_size, bias=False)
        self.query = nn.Linear(inp_embd, head_size, bias=False)
        self.value = nn.Linear(inp_embd, head_size, bias=False)
        # self.key = FeedForward(out_embd, out_embd * 4, head_size, dropout)
        # self.query = FeedForward(inp_embd, inp_embd * 4, head_size, dropout)
        # self.value = FeedForward(inp_embd, inp_embd * 4, head_size, dropout)
        #
        self.register_buffer('tril', torch.tril(torch.ones(inp_size, out_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, reduced_embeddings):
        b, t, c = x.shape
        _, tr, _ = reduced_embeddings.shape
        k = self.key(reduced_embeddings)  # (B,Tr,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * c ** -0.5  # (B, T, C) @ (B, C, Tr) -> (B, T, Tr)
        wei = wei.masked_fill(self.tril[:t, :tr] == 0, float('-inf'))  # (B, T, Tr)
        wei = F.softmax(wei, dim=-1)  # (B, T, Tr)
        wei = self.dropout(wei)
        wei = wei.transpose(-2, -1)  # (B, T, Tr) -> (B, Tr, T)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, Tr, T) @ (B, T, C) -> (B, Tr, C)
        return out


class CompressingMultiHeadAttention(nn.Module):
    def __init__(self, inp_size: int, out_size: int, inp_embd: int, out_embd: int, head_size: int, n_head: int, dropout: float):
        super().__init__()

        self.heads = nn.ModuleList([CompressingAttentionHead(inp_size, out_size, inp_embd, out_embd, head_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(out_embd, out_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, reduced_embeddings):
        out = torch.cat([h(x, reduced_embeddings) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class CompressingBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config: TransformerConfig, inp_size: int, inp_embd: int, out_size: int, out_embd: int):
        super().__init__()
        self.config = config

        dropout = config.dropout
        n_head = config.n_head
        head_size = out_embd // n_head

        self.register_buffer('reduced_pos_emb_ids', torch.arange(out_size, device=self.config.my_device))

        print(f"in_size={inp_size}, in_embd={inp_embd}, out_size={out_size}, out_embd={out_embd}")

        self.block = Block(dropout, inp_size, inp_embd * 4, inp_embd, out_embd, n_head, inp_embd // n_head)

        self.reduced_position_embedding_table = nn.Embedding(out_size, inp_embd)

        self.ln1 = nn.LayerNorm(out_embd)
        self.sa1 = CompressingMultiHeadAttention(inp_size, out_size, inp_embd, out_embd, head_size, n_head, dropout)

        self.ln2 = nn.LayerNorm(out_embd)
        self.ffwd = FeedForward(out_embd, out_embd * 4, out_embd, dropout)

    def forward(self, x):
        b, t, c = x.shape

        x = x + self.block(x)

        reduced_pos_emb = self.reduced_position_embedding_table(self.reduced_pos_emb_ids.repeat(b, 1))  # (B,T,C)

        x = self.sa1(self.ln1(x), reduced_pos_emb)

        x = x + self.ffwd(self.ln2(x))

        return x


class CompressingTransformerModel(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        block_size = config.block_size
        n_embd = config.n_embed

        def get_block_size(i):
            res = block_size // (2 ** i)
            if res < 1:
                res = 1
            return res

        def get_emb_size(i):
            return n_embd * 2 ** i

        block_sizes = [get_block_size(i) for i in range(config.n_layer + 1)]
        #emb_sizes = [get_emb_size(i) for i in range(config.n_layer + 1)]
        emb_sizes = [n_embd for i in range(config.n_layer + 1)]

        self.token_embedding_table = nn.Embedding(config.vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, n_embd)
        self.blocks1 = nn.Sequential(*[CompressingBlock(
            config,
            block_sizes[i],
            emb_sizes[i],
            block_sizes[i + 1],
            emb_sizes[i + 1],
        ) for i in range(config.n_layer)])

        mid_embd = emb_sizes[config.n_layer]
        self.ln_mid = nn.LayerNorm(mid_embd)
        self.mid = nn.Linear(mid_embd, mid_embd)

        self.blocks2 = nn.Sequential(*[CompressingBlock(
            config,
            block_sizes[config.n_layer - i],
            emb_sizes[config.n_layer - i],
            block_sizes[config.n_layer - i - 1],
            emb_sizes[config.n_layer - i - 1]
        ) for i in range(config.n_layer)])

        out_embd = emb_sizes[0]

        self.ln_out = nn.LayerNorm(out_embd)
        self.out = nn.Linear(out_embd, config.vocab_size)

        self.out.weight = self.token_embedding_table.weight

    def forward_vs_target(self, idx, targets):
        logits = self.forward(idx)

        b, t, c = logits.shape
        logits_view = logits.view(b * t, c)
        targets = targets.view(b * t)
        loss = F.cross_entropy(logits_view, targets)

        return logits_view, loss

    def half_fwd_in(self, idx):
        b, t = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(t, device=self.config.my_device))  # (T,C)
        x = tok_emb + pos_emb
        x = self.blocks1(x)
        x = self.ln_mid(x)
        x = self.mid(x)
        return x

    def half_fwd_out(self, x):
        x = self.blocks2(x)
        x = self.ln_out(x)
        x = self.out(x)
        return x

    def forward(self, idx):
        x = self.half_fwd_in(idx)

        x = self.half_fwd_out(x)
        return x

    def generate(self, idx1):
        logits = self.forward(idx1)

        b, t, c = logits.shape
        probs = F.softmax(logits, dim=-1)
        probs = probs.reshape(b * t, c)
        idx = torch.multinomial(probs, num_samples=1)
        idx = idx.reshape(b, t)

        return idx


class CompressingRunner(AbstractRunner):
    def __init__(self, config: TransformerConfig):
        super().__init__(config, CompressingTransformerModel(config))
        pass

    def generate(self, context):
        return self.model.generate(context)


class CompressingAccumulativeTrainer(StringAccumulativeTrainer):

    def __init__(self, config: TransformerConfig):
        super().__init__(config, CompressingRunner(config))
        self.data_x = []
        self.data_y = []
        self.data_dict = dict()

    def get_batch(self):
        ix = torch.randint(len(self.data_x), (self.config.batch_size,))

        x = torch.stack([torch.tensor(self.config.token_codec.encode(self.data_x[i])) for i in ix])
        y = torch.stack([torch.tensor(self.config.token_codec.encode(self.data_y[i])) for i in ix])

        x, y = x.to(self.config.my_device), y.to(self.config.my_device)

        return x, y

    def add_sample(self, x, y):
        key = x
        if key in self.data_dict:
            self.data_dict[key] += 1
            return self.data_dict[key]

        self.data_x += [x]
        self.data_y += [y]

        self.data_dict[key] = 1

        return 1

    def predict_list(self, lst1):
        lst1 = [self.config.token_codec.encode(x) for x in lst1]

        x1 = torch.tensor(lst1).to(self.config.my_device)

        out = self.runner.generate(x1)
        out = out.tolist()

        return [self.config.token_codec.decode(o) for o in out]

    def predict(self, x1):
        return self.predict_list([x1])[0]

    def train_iterate(self, n_iter, get_train_batch, get_val_batch):
        self.runner.train_eval(n_iter, get_train_batch, get_val_batch)

    def train(self, n=1):
        losses = 0
        for i in range(n):
            x, y = self.get_batch()
            o, loss = self.runner.learn(x, y)
            l = loss.item()
            losses += l
        av_loss = losses / n

        return av_loss, len(self.data_x)