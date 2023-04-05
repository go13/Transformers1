import torch
from torch import nn as nn

from t3_karpathy.commons import AbstractAccumulativeTrainer, AbstractRunner
from t3_karpathy.enhanced_karpathy_transformer import BlockSequence
from t3_karpathy.commons.embeddings import PositionalEmbedding

from t3_karpathy.transformer_config import TransformerConfig


class SentimentalFeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        inp_size = config.n_embed * config.block_size
        hidden_size = inp_size // 2  #config.hidden_size # * config.block_size
        dropout = config.dropout
        out_size = 1

        self.net = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.Dropout(dropout),
            nn.Sigmoid(),
            nn.Linear(hidden_size, out_size),
            # FeedForward(inp_size, hidden_size, out_size, dropout),
        )

    def forward(self, x):
        return self.net(x)


class SentimentalTransformerModel(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)
        # self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        #
        self.pos_emb1 = PositionalEmbedding(config)

        self.blocks = BlockSequence(config)
        # self.blocks = nn.Sequential(*[Block.create(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.out = SentimentalFeedForward(config)

    def forward_vs_target(self, idx, targets):
        output = self.forward(idx)

        mse_loss = torch.nn.MSELoss(reduction='mean')
        loss = mse_loss(output, targets)

        return output, loss

    def forward(self, idx):
        # idx and targets are both (B,T) tensor of integers
        x = self.token_embedding_table(idx)  # (B,T,C)
        b, t, c = x.shape

        pos_emb = self.pos_emb1(b, t)

        x, pos_emb = self.blocks(x, pos_emb, pos_emb)  # (B,T,C)

        x = self.ln_f(x)  # (B,T,C)
        x = x.reshape(b, -1)
        x = self.out(x)
        x = x.reshape(b)
        return x


    # def forward(self, idx):
    #     # idx and targets are both (B,T) tensor of integers
    #     x = self.token_embedding_table(idx)  # (B,T,C)
    #     b, t, c = x.shape
    #
    #     pos_emb = self.pos_emb1(b, t)
    #
    #     x, pos_emb = self.blocks(x, pos_emb)  # (B,T,C)
    #
    #     x = self.ln_f(x)  # (B,T,C)
    #     x = x.reshape(b, -1)
    #     x = self.out(x)
    #     x = x.reshape(b)
    #     return x

class SentimentalRunner(AbstractRunner):
    def __init__(self, config: TransformerConfig):
        super().__init__(config, SentimentalTransformerModel(config), None)
        pass


class SentimentalAccumulativeTrainer(AbstractAccumulativeTrainer):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.runner: SentimentalRunner = SentimentalRunner(config)
        self.data_x = []
        self.data_y = []
        self.data_dict = dict()

        self.last_data_x = []
        self.last_data_y = []

    def get_fitness_histogram(self):
        f_hist = dict()
        for x, y in zip(self.data_x, self.data_y):
            num = self.data_dict[x]
            f = int(y)
            if f in f_hist:
                f_hist[f] += num
            else:
                f_hist[f] = num
        return f_hist

    def get_xy_histogram(self):
        xy_hist = dict()
        i = 0
        for x, y in self.data_dict.items():
            xy_hist[i] = y
            i += 1
        return xy_hist

    def get_batch(self, data_x, data_y):
        ix = torch.randint(len(data_x), (self.config.batch_size,))
        x = torch.stack([torch.tensor(self.config.token_codec.encode(data_x[i])) for i in ix])
        y = torch.stack([torch.tensor(data_y[i]) for i in ix])
        x, y = x.to(self.config.my_device), y.to(self.config.my_device, dtype=self.config.precision)
        return x, y

    def add_sample(self, x, y):
        self.last_data_x += [x]
        self.last_data_y += [y]

        if x in self.data_dict:
            self.data_dict[x] += 1
            return self.data_dict[x]

        self.data_x += [x]
        self.data_y += [y]

        self.data_dict[x] = 1

        return 1

    def predict_list(self, lst):
        lst = [self.config.token_codec.encode(x) for x in lst]
        x = torch.tensor(lst).to(self.config.my_device)
        out = self.runner.forward(x)
        return out.tolist()

    def predict(self, x):
        encoded_x = self.config.token_codec.encode(x)
        x = torch.tensor(encoded_x).to(self.config.my_device)
        x = x.reshape(1, x.shape[0])
        out = self.runner.forward(x)
        return out.item()

    def train(self, n=1):
        losses = 0
        for i in range(n):
            x, y = self.get_batch(self.data_x, self.data_y)
            o, loss = self.runner.learn(x, y)
            l = loss.item()
            self.loss_hist += [l]
            losses += l
        av_loss = losses / n

        return av_loss, len(self.data_x)

    def train_recent(self, n=1):
        for i in range(n):
            x, y = self.get_batch(self.last_data_x, self.last_data_y)
            o, loss = self.runner.learn(x, y)

        self.last_data_x = []
        self.last_data_y = []