import torch
from torch import nn as nn
from torch.nn import functional as F

from ga_t3.accumulative_trainer import StringAccumulativeTrainer
from t3_karpathy.karpathy_transformer import MultiHeadAttention, FeedForward
from t3_karpathy.transformer_config import TransformerConfig
from t3_karpathy.karpathy_transformer import AbstractRunner


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, dropout: float, block_size: int, hidden_size: int, out_size: int, n_embd: int, n_head: int, head_size: int):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(block_size, n_embd, head_size, n_head, dropout)

        self.ln2 = nn.LayerNorm(block_size)
        self.ffwd = FeedForward(block_size, block_size * 2, out_size, dropout)

    def forward(self, x):
        b, t, c = x.shape
        x = x + self.sa(self.ln1(x))

        # transpose x (b, t, c) to (b, c, t)
        x = x.transpose(1, 2)

        x = self.ffwd(self.ln2(x))

        x = x.transpose(1, 2)

        return x

    @staticmethod
    def create_with_block_size(config: TransformerConfig, input_size: int, output_size: int):
        out_size = output_size
        hidden_size = config.hidden_size
        dropout = config.dropout
        n_embd = config.n_embd
        head_size = config.head_size
        n_head = config.n_head
        return Block(dropout, input_size, hidden_size, out_size, n_embd, n_head, head_size)


class CompressingTransformerModel(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        block_size = config.block_size

        def get_block_size(i):
            res = block_size // (2 ** i)
            if res < 1:
                res = 1
            return res

        block_sizes = [get_block_size(i) for i in range(config.n_layer + 1)]

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks1 = nn.Sequential(*[Block.create_with_block_size(config, block_sizes[i], block_sizes[i + 1]) for i in range(config.n_layer)])

        self.ln_mid = nn.LayerNorm(config.n_embd)
        self.mid = nn.Linear(config.n_embd, config.n_embd)

        self.blocks2 = nn.Sequential(*[Block.create_with_block_size(config, block_sizes[config.n_layer - i], block_sizes[config.n_layer - i - 1]) for i in range(config.n_layer)])

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
        self.runner.train_iterate(n_iter, get_train_batch, get_val_batch)

    def train(self, n=1):
        losses = 0
        for i in range(n):
            x, y = self.get_batch()
            o, loss = self.runner.learn(x, y)
            l = loss.item()
            losses += l
        av_loss = losses / n

        return av_loss, len(self.data_x)