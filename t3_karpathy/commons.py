from torch import nn as nn


class BaseTransformerConfig:

    def __init__(self, my_device='cuda', batch_size=64, block_size=32, n_embed=64, n_head=4, n_layer=4, learning_rate=1e-2):
        self.my_device = my_device

        # karpathy parameters
        self.batch_size = batch_size
        self.block_size = block_size
        self.n_embd = n_embed
        self.hidden_size = self.n_embd * 4

        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = 0.1
        self.head_size = self.n_embd // self.n_head

        self.max_iters = 15000
        self.eval_interval = 100
        self.learning_rate = learning_rate
        self.eval_iters = 200

        self.norm_eps: float = 1e-5   # llma
        self.max_seq_len: int = 2048  # llma
        self.multiple_of: int = 256   # llma


class SentimentalFeedForward(nn.Module):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__()

        inp_size = config.n_embd * config.block_size
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


class AbstractCodec(object):
    def __init__(self):
        pass

    def encode(self, s: str) -> list:
        raise NotImplementedError()

    def decode(self, l: list) -> str:
        raise NotImplementedError()