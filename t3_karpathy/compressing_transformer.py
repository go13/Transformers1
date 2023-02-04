import torch
from torch import nn as nn
from torch.nn import functional as F

from t3_karpathy.karpathy_transformer import Block
from t3_karpathy.transformer_config import TransformerConfig
from t3_karpathy.karpathy_transformer import AbstractRunner


class CompressingTransformerModel(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        block_size = config.block_size
        hidden_size = config.hidden_size
        out_size = config.n_embd
        dropout = config.dropout
        n_embd = config.n_embd
        head_size = config.head_size
        n_head = config.n_head

        block_sizes = [ block_size for i in range(config.n_layer) ]
        # block_sizes = [ block_size // 2 ** i for i in range(config.n_layer) ]

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks1 = nn.Sequential(*[Block(dropout, block_sizes[i], hidden_size, out_size, n_embd, n_head, head_size) for i in range(config.n_layer)])

        self.ln_mid = nn.LayerNorm(config.n_embd)
        self.mid = nn.Linear(config.n_embd, config.n_embd)

        self.blocks2 = nn.Sequential(*[Block(dropout, block_sizes[config.n_layer - i - 1], hidden_size, out_size, n_embd, n_head, head_size) for i in range(config.n_layer)])

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
        x = self.half_fwd_in(idx1)

        logits = self.half_fwd_out(x)

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


class CompressingAccumulativeTrainer(AbstractRunner):

    def __init__(self, config: TransformerConfig):
        super().__init__(config, CompressingTransformerModel(config))
        self.runner: CompressingRunner = CompressingRunner(config)
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

    def train(self, n=1):
        losses = 0
        for i in range(n):
            x, y = self.get_batch()
            o, loss = self.runner.learn(x, y)
            l = loss.item()
            losses += l
        av_loss = losses / n

        return av_loss, len(self.data_x)