import time

import torch
import pandas as pd
from torch import nn as nn

from ga_t3.accumulative_trainer import AbstractAccumulativeTrainer
from src.performance_utils import timeit
from t3_karpathy.commons import AbstractCodec
from t3_karpathy.enhanced_karpathy_transformer import BlockSequence, PositionalEmbedding, DistancePositionalEmbedding, FeedForward

from t3_karpathy.karpathy_transformer import AbstractRunner
from t3_karpathy.transformer_config import BaseTransformerConfig


class TimeseriesFeedForward(nn.Module):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__()

        inp_size = config.n_embd * config.block_size
        hidden_size = config.hidden_size
        dropout = config.dropout
        out_size = 1

        self.net = nn.Sequential(
            nn.Linear(inp_size, hidden_size, bias=False),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, x):
        return self.net(x)


class TimeseriesTransformerModel(nn.Module):

    def __init__(self, config: BaseTransformerConfig):
        super().__init__()
        self.config = config

        self.pos_emb1 = PositionalEmbedding(config)
        self.pos_emb_dist = DistancePositionalEmbedding(config)

        kernel_size = 8
        right_pad = kernel_size - 1
        n_kernels = config.n_embd * 4
        self.conv1d1 = nn.Conv1d(
            in_channels=1,
            out_channels=n_kernels,
            kernel_size=kernel_size,
            bias=True,
        )
        self.padding1 = nn.ConstantPad1d((0, right_pad), 0)

        self.input_ffwd = FeedForward(n_kernels, n_kernels, config.n_embd, config.dropout)

        self.blocks = BlockSequence(config)

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.out = TimeseriesFeedForward(config)

    def forward_vs_target(self, idx, targets):
        output = self.forward(idx)

        mse_loss = torch.nn.MSELoss(reduction='mean')
        loss = mse_loss(output, targets)

        return output, loss

    def forward(self, inp):
        # idx and targets are both (B,T) tensor of integers
        b, t = inp.shape

        pos_emb = self.pos_emb_dist(b)

        x = inp.unsqueeze(1)

        x = self.conv1d1(x)

        x = self.padding1(x)

        x = x.transpose(-1, -2)

        x = self.input_ffwd(x)

        x, pos_emb = self.blocks(x, pos_emb)  # (B,T,C)

        x = self.ln_f(x)  # (B,T,C)
        x = x.reshape(b, -1)
        x = self.out(x)

        # x = x.squeeze(-1)
        return x


class TimeseriesRunner(AbstractRunner):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__(config, TimeseriesTransformerModel(config))
        pass


class TimeseriesCodec(AbstractCodec):

    def __init__(self):
        pass

    def encode(self, x):
        return x

    def decode(self, y):
        return y


class TimeseriesDataloader(object):

    def __init__(self, config: BaseTransformerConfig):
        self.config = config
        self.codec = TimeseriesCodec()

        df = pd.read_csv('F:\\workspace\\ai\\Transformers1\\timeseries\\US-Stock-Dataset\\Data\\Stocks\\TSLA.csv')
        df_close = df['Close'].values

        self.data = torch.tensor(df_close, dtype=torch.float) # todo
        n = int(0.9 * len(self.data))  # first 90% will be train, rest val
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    # data loading
    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.config.block_size, (self.config.batch_size,))
        x = torch.stack([data[i:i + self.config.block_size] for i in ix])
        y = torch.stack([data[i + self.config.block_size:i + self.config.block_size + 1] for i in ix])
        x, y = x.to(self.config.my_device), y.to(self.config.my_device)
        return x, y

    def get_train_batch(self):
        return self.get_batch('train')

    def get_val_batch(self):
        return self.get_batch('test')


class TimeseriesPandasTrainer(AbstractAccumulativeTrainer):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__(config)
        self.runner: TimeseriesRunner = TimeseriesRunner(config)
        self.dataloader = TimeseriesDataloader(config)

    def get_batch(self, data_x, data_y):
        ix = torch.randint(len(data_x), (self.config.batch_size,))
        x = torch.stack([torch.tensor(self.dataloader.codec.encode(data_x[i])) for i in ix])
        y = torch.stack([torch.tensor(data_y[i]) for i in ix])
        x, y = x.to(self.config.my_device), y.to(self.config.my_device)
        return x, y

    def predict_list(self, lst):
        lst = [self.dataloader.codec.encode(x) for x in lst]
        x = torch.tensor(lst).to(self.config.my_device)
        out = self.runner.forward(x)
        return out.tolist()

    def predict(self, x):
        encoded_x = self.dataloader.codec.encode(x)
        x = torch.tensor(encoded_x).to(self.config.my_device)
        x = x.reshape(1, x.shape[0])
        out = self.runner.forward(x)
        return out.item()

    def train(self, n=1):
        losses = 0
        for i in range(n):
            x, y = self.dataloader.get_train_batch()
            o, loss = self.runner.learn(x, y)
            l = loss.item()
            self.loss_hist += [l]
            losses += l
        av_loss = losses / n

        return av_loss


config = BaseTransformerConfig(batch_size=64, block_size=128, n_embed=32, n_head=4, n_layer=4)
trainer1 = TimeseriesPandasTrainer(config)


def step():
    start_time = time.time()
    av_loss = trainer1.train(1)
    end_time = time.time()
    time_taken = end_time - start_time

    print(f"st={st}, av_loss={av_loss}, time_taken={time_taken}")


for st in range(10000):
    step()