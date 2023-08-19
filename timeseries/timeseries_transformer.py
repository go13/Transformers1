import sys
import os

module_path = os.path.join(os.path.dirname(__file__), 't3_karpathy')
sys.path.append(module_path)

import torch
from torch import nn as nn
import glob

from t3_karpathy.commons.commons import AbstractCodec, AbstractAccumulativeTrainer, AbstractRunner, \
    BaseTransformerConfig, TimeseriesFeedForward
from t3_karpathy.commons.feed_forwards import GeluFeedForward
from t3_karpathy.enhanced_karpathy_transformer import BlockSequence
from t3_karpathy.commons.embeddings import PositionalEmbedding, DistancePositionalEmbedding

from timeseries.csv_reader import read_and_merge_csv_files


class TimeseriesTransformerConfig(BaseTransformerConfig):

    def __init__(self, my_device='cuda', precision=torch.bfloat16, batch_size=64, block_size=256, n_embed=32, n_head=4,
                 n_layer=4, kernel_size=4, channels=12, learning_rate=1e-3):
        super().__init__(my_device, precision, batch_size, block_size, n_embed, n_head, n_layer, learning_rate)
        self.channels = channels
        self.kernel_size = kernel_size


class TimeseriesTransformerModel(nn.Module):

    def __init__(self, config: TimeseriesTransformerConfig):
        super().__init__()
        self.config = config
        self.channels = config.channels
        self.pos_emb1 = PositionalEmbedding(config)
        self.pos_emb_dist = DistancePositionalEmbedding(config)

        kernel_size = config.kernel_size
        right_pad = kernel_size - 1
        n_kernels = config.n_embed * 16
        self.conv1d1 = nn.Conv1d(
            in_channels=self.channels,
            out_channels=n_kernels,
            kernel_size=kernel_size,
            bias=True,
        )
        self.padding1 = nn.ConstantPad1d((0, right_pad), 0)

        self.input_ffwd = GeluFeedForward(n_kernels, config.n_embed, config.n_embed, config.dropout)

        self.blocks = BlockSequence(config)

        self.ln_f = nn.LayerNorm(config.n_embed)

        inp_size = config.n_embed
        hidden_size = config.hidden_size
        dropout = config.dropout
        out_size = config.channels // 2
        self.out = TimeseriesFeedForward(inp_size, hidden_size, out_size, dropout)

    def forward_vs_target(self, idx, targets):
        output = self.forward(idx)
        mid_indx = targets.shape[-1] // 2
        # targets = targets[:, :, :mid_indx]  # first half - absolute
        targets = targets[:, :, mid_indx:]  # 2nd - delta
        mse_loss = torch.nn.MSELoss(reduction='mean')
        loss = mse_loss(output, targets)

        return output, loss

    def forward(self, inp):
        # idx and targets are both (B,T) tensor of integers
        b, t, c = inp.shape

        pos_emb = self.pos_emb_dist(b)

        x = inp.transpose(-1, -2)

        x = self.conv1d1(x)

        x = self.padding1(x)

        x = x.transpose(-1, -2)

        x = self.input_ffwd(x)

        x, pos_emb = self.blocks(x, pos_emb, pos_emb)  # (B,T,C)

        x = self.ln_f(x)  # (B,T,C)
        # x = x.reshape(b, -1)

        x = self.out(x)

        # x = x.squeeze(-1)
        return x


class TimeseriesRunner(AbstractRunner):
    def __init__(self, config: TimeseriesTransformerConfig, model: TimeseriesTransformerModel):
        super().__init__(config, model, None)
        pass


class TimeseriesCodec(AbstractCodec):

    def __init__(self):
        pass

    def encode(self, x):
        return x

    def decode(self, y):
        return y


class TimeseriesDataloader(object):

    def __init__(self, directory_path, stocks_to_load, my_device='cuda', add_diff=True):
        self.codec = TimeseriesCodec()

        df, found_files = read_and_merge_csv_files(directory_path, stocks_to_load, start_date='2000-01-01',
                                                   end_date='2020-12-31')

        df.drop(columns=['Date'], axis=1, inplace=True)

        prices = torch.tensor(df.values, dtype=torch.float, device=my_device)
        prices_diff = torch.diff(prices, dim=0)

        if add_diff:
            self.data = torch.concat([prices[1:], prices_diff], dim=1)
        else:
            self.data = prices

        n = int(0.9 * len(self.data))  # first 90% will be trained, rest - eval
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

        self.found_files = found_files

        print("Found files: ", found_files)

    def get_number_of_channels(self):
        return self.found_files * 2

    def get_train_data(self):
        return self.train_data

    def get_val_data(self):
        return self.val_data

    def get_data(self):
        return self.data


class TimeseriesPandasTrainer(AbstractAccumulativeTrainer):
    def __init__(self, config: TimeseriesTransformerConfig, dataloader: TimeseriesDataloader,
                 model: TimeseriesTransformerModel):
        super().__init__(config)
        self.runner: TimeseriesRunner = TimeseriesRunner(config, model)
        self.dataloader = dataloader

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
            x, y = self.get_train_batch()
            o, loss = self.runner.learn(x, y)
            l = loss.item()
            self.loss_hist += [l]
            losses += l
        av_loss = losses / n

        return av_loss

    def train_eval(self, n_iter):
        self.runner.train_iterate(n_iter, self.get_train_batch, self.get_val_batch)

    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        data = self.dataloader.get_train_data() if split == 'train' else self.dataloader.get_val_data()
        ix = torch.randint(len(data) - self.config.block_size - 1, (self.config.batch_size,))
        x = torch.stack([data[i:i + self.config.block_size] for i in ix])
        y = torch.stack([data[i + 1: i + self.config.block_size + 1] for i in ix])
        x, y = x.to(self.config.my_device, dtype=self.config.precision), y.to(self.config.my_device,
                                                                              dtype=self.config.precision)
        return x, y

    def get_train_batch(self):
        return self.get_batch('train')

    def get_val_batch(self):
        return self.get_batch('test')


stocks_to_load = [
    "AAPL", "TSLA",
    "A", "GOOG", "AMZN", "PYPL", "NVDA", "AMD",
    "NFLX", "MSFT", "INTC", "CSCO", "ADBE", "CRM", "QCOM", "TXN", "AVGO",
    "INTU", "ORCL", "COST", "SBUX", "AMGN", "CHTR", "GILD", "CMCSA", "BKNG",
    "MDLZ", "FISV", "BIIB", "MU", "MCD", "AMAT", "ADP", "ILMN", "ATVI", "ISRG",
    "ADSK", "LRCX", "BIDU", "JD", "REGN", "WBA", "VRTX", "KHC", "WMT", "ZM", "MELI",
    "TMUS", "CTSH", "XLNX", "PCAR", "ALGN", "WDAY", "SIRI", "CTXS", "ADI", "EXC", "LULU",
    "MAR", "KLAC", "PAYX", "EA", "ILMN", "ALXN", "MNST", "BMRN", "EBAY", "CTAS", "VRSK",
    "IDXX", "CDNS", "NXPI", "ASML", "INCY", "KLAC", "MCHP", "SNPS", "SWKS", "VRSN",
    "WDC", "WYNN", "XLNX", "ZBRA", "ZTS", "AEP", "AIG", "ALL", "AXP", "BA", "BAC",
    "BK", "BLK", "C", "CAT", "CL", "COF", "COP", "COST", "CSCO", "CVS", "CVX",
    "DD", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FDX", "GD", "GE", "GILD",
    "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KHC", "KMI",
    "KO", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "MMM",
    'BLND', 'BLNG', 'BLNKW', 'BLNGU', 'BLNGW', 'BLNK', 'BLPH', 'BLRX',
    'BLTE', 'BLU', 'BLUA', 'BLUE', 'BLW', 'BLX', 'BLZE', 'BMA', 'BMAC', 'BMAQ', 'BMAQR', 'BMAQU'
]

directory_path = '../us-stock-dataset/Data/Stocks'

# stocks_to_load = [s.split("\\")[1].replace(".csv", "") for s in glob.glob(directory_path + "//*.csv")]
#
# stock_number_to_load = 200
#
# stocks_to_load = stocks_to_load[1000:1000+stock_number_to_load]

# dataloader = TimeseriesDataloader(directory_path, stocks_to_load)
# config = TimeseriesTransformerConfig(
#     batch_size=32,
#     block_size=512,
#     n_embed=32,
#     n_head=4,
#     n_layer=8,
#     kernel_size=1,
#     learning_rate=1e-3,
#     channels=dataloader.get_number_of_channels()
# )
# model = TimeseriesTransformerModel(config)
# trainer1 = TimeseriesPandasTrainer(config, dataloader=dataloader, model=model)
#
# trainer1.train_eval(20000)
