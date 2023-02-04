import torch
from torch import nn as nn
from torch.nn import functional as F

from ga_t3.accumulative_trainer import AbstractAccumulativeTrainer
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
        self.blocks1 = nn.Sequential(*[Block.create(config) for _ in range(config.n_layer)])

        self.ln_mid = nn.LayerNorm(config.n_embd)
        self.mid = nn.Linear(config.n_embd, config.n_embd)

        self.blocks2 = nn.Sequential(*[Block.create(config) for _ in range(config.n_layer)])

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

    def generate(self, idx1, idx2):
        # x1 = self.config.token_codec.encode(idx1)
        # x2 = self.config.token_codec.encode(idx2)

        x1 = self.half_fwd_in(idx1)
        x2 = self.half_fwd_in(idx2)
        # create random mask of 0 and 1 for x1 and inverse mask for x2 of x1.shape
        mask = torch.randint(2, x1.shape, device=self.config.my_device)
        mask_inv = 1 - mask
        # combine x1 and x2 with mask
        x = x1 * mask + x2 * mask_inv

        logits = self.half_fwd_out(x)

        b, t, c = logits.shape
        probs = F.softmax(logits, dim=-1)
        probs = probs.reshape(b * t, c)
        idx = torch.multinomial(probs, num_samples=1)
        idx = idx.reshape(b, t)

        return idx


class AutoencoderRunner(AbstractRunner):
    def __init__(self, config: TransformerConfig):
        super().__init__(config, AutoencoderTransformerModel(config))
        pass

    def generate(self, context, max_new_tokens):
        return self.model.generate(context, max_new_tokens)


class AutoencoderAccumulativeTrainer(AbstractAccumulativeTrainer):

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.runner: AutoencoderRunner = AutoencoderRunner(config)
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

    def predict_list(self, lst1, lst2):
        lst1 = [self.config.token_codec.encode(x) for x in lst1]
        lst2 = [self.config.token_codec.encode(x) for x in lst2]

        x1 = torch.tensor(lst1).to(self.config.my_device)
        x2 = torch.tensor(lst2).to(self.config.my_device)

        out = self.runner.generate(x1, x2)
        out = out.tolist()

        return [self.config.token_codec.decode(o) for o in out]

    def predict(self, x1, x2):
        return self.predict_list([x1], [x2])[0]

    def train(self, n=1):
        losses = 0
        for i in range(n):
            x, y = self.get_batch()
            o, loss = self.runner.learn(x, y)
            l = loss.item()
            losses += l
        av_loss = losses / n

        return av_loss, len(self.data_x)