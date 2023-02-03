from collections import OrderedDict

import torch
from torch import nn as nn
from torch.nn import functional as F

from ga_t3.accumulative_trainer import AbstractAccumulativeTrainer

from t3_karpathy.karpathy_transformer import Block, FeedForward
from t3_karpathy.transformer_config import TransformerConfig


class CrossoverTransformerModel(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks1 = nn.Sequential(*[Block.create_block(config) for _ in range(config.n_layer)])
        self.blocks2 = nn.Sequential(*[Block.create_block(config) for _ in range(config.n_layer)])

        mid_size = config.n_embd * config.block_size

        self.mid = FeedForward(mid_size * 2, mid_size * 1, mid_size, config.dropout)

        self.blocks3 = nn.Sequential(*[Block.create_block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.out = nn.Linear(config.n_embd, config.vocab_size)

    def forward_vs_target(self, idx1, idx2, targets):
        logits = self.forward(idx1, idx2)

        b, t, c = logits.shape
        logits_view = logits.view(b * t, c)
        targets = targets.view(b * t)
        loss = F.cross_entropy(logits_view, targets)

        return logits_view, loss

    def forward(self, idx1, idx2):
        b1, t1 = idx1.shape
        b2, t2 = idx2.shape

        tok_emb1 = self.token_embedding_table(idx1)
        pos_emb1 = self.position_embedding_table(torch.arange(t1, device=self.config.my_device))

        tok_emb2 = self.token_embedding_table(idx2)
        pos_emb2 = self.position_embedding_table(torch.arange(t2, device=self.config.my_device))

        x1 = tok_emb1 + pos_emb1
        x1 = self.blocks1(x1)

        x2 = tok_emb2 + pos_emb2
        x2 = self.blocks2(x2)

        x = torch.cat((x1, x2), dim=1)
        x = x.reshape(b1, -1)

        x = self.mid(x)

        x = x.reshape(b1, t1, -1)

        x = self.blocks3(x)

        x = self.ln_f(x)
        x = self.out(x)

        return x

    def generate(self, x1, x2):
        logits = self.forward(x1, x2)
        b, t, c = logits.shape
        probs = F.softmax(logits, dim=-1)
        probs = probs.reshape(b * t, c)
        idx = torch.multinomial(probs, num_samples=1)
        idx = idx.reshape(b, t)
        return idx


class CrossoverRunner(object):
    def __init__(self, config: TransformerConfig):
        self.model = CrossoverTransformerModel(config).to(config.my_device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.config = config
        self.current_iteration = 0

        print(sum(p.numel() for p in self.model.parameters()) / 1e6, 'M parameters')

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def generate(self, x1, x2):
        return self.model.generate(x1, x2)

    def learn(self, x1, x2, y, f):
        self.model.train()
        out, loss = self.model.forward_vs_target(x1, x2, y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward(gradient=f.sum())
        self.optimizer.step()
        return out, loss

    @torch.no_grad()
    def evaluate(self, get_batch, eval_iters):
        self.model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x1, x2, y = get_batch()
            logits, loss = self.model.forward_vs_target(x1, x2, y)
            losses[k] = loss.item()
        return losses.mean()

    def train_iterate(self, n_iter, get_train_batch, get_val_batch):
        for _ in range(n_iter):
            if self.current_iteration % self.config.eval_interval == 0:
                train_losses = self.evaluate(get_train_batch, self.config.eval_iters)
                val_losses = self.evaluate(get_val_batch, self.config.eval_iters)
                print(f"step {self.current_iteration}: train loss {train_losses:.4f}, val loss {val_losses:.4f}")

            x1, x2, y, f = get_train_batch()

            logits, loss = self.learn(x1, x2, y, f)

            self.current_iteration += 1

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, new_state_dict):
        self.model.load_state_dict(OrderedDict(new_state_dict))


class CrossoverAccumulativeTrainer(AbstractAccumulativeTrainer):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.runner: CrossoverRunner = CrossoverRunner(config)
        self.data_x1 = []
        self.data_x2 = []
        self.data_y = []
        self.data_child_f = []
        self.data_dict = dict()
    #
    # def get_fitness_histogram(self):
    #     f_hist = dict()
    #     for x1, x2, x3 in zip(self.data_x1, self.data_x2, self.data_x3):
    #         num = self.data_dict[x1]
    #         f = int(y)
    #         if f in f_hist:
    #             f_hist[f] += num
    #         else:
    #             f_hist[f] = num
    #     return f_hist

    # def get_xy_histogram(self):
    #     xy_hist = dict()
    #     i = 0
    #     for x, y in self.data_dict.items():
    #         xy_hist[i] = y
    #         i += 1
    #     return xy_hist

    def get_batch(self):
        ix = torch.randint(len(self.data_x1), (self.config.batch_size,))
        x1 = torch.stack([torch.tensor(self.config.token_codec.encode(self.data_x1[i])) for i in ix])
        x2 = torch.stack([torch.tensor(self.config.token_codec.encode(self.data_x2[i])) for i in ix])
        y = torch.stack([torch.tensor(self.config.token_codec.encode(self.data_y[i])) for i in ix])
        f = torch.stack([torch.tensor(self.data_child_f[i]) for i in ix])
        x1, x2, y, f = x1.to(self.config.my_device), x2.to(self.config.my_device), y.to(self.config.my_device), f.to(self.config.my_device)
        return x1, x2, y, f

    def add_sample(self, x1, x2, y, child_f):
        key = (x1, x2)
        if key in self.data_dict:
            self.data_dict[key] += 1
            return self.data_dict[key]

        self.data_x1 += [x1]
        self.data_x2 += [x2]
        self.data_y += [y]
        self.data_child_f += [child_f]

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
            x1, x2, y, f = self.get_batch()
            o, loss = self.runner.learn(x1, x2, y, f)
            l = loss.item()
            self.loss_hist += [l]
            losses += l
        av_loss = losses / n

        return av_loss, len(self.data_x1)